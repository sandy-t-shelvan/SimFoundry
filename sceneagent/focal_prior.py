# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A video's focal length, self-calibrated from its own verified matches.

An MP4 carries no EXIF, so COLMAP starts every video camera at
``1.2 * max(width, height)`` (2304 px at 1080p). The repo videos' real
focals are 874 px (Mailbox), about 1550 px (PutCupInBowl) and 1663 px
(PutMarkerInCup). Mapping with the intrinsics held at the guess registers
every frame but bends the geometry, and one bundle adjustment does not walk
back from 2304 to 1663 (it settled at 2355-2464 px), so placement then
fails. Letting the mapper refine them from the start is what diverged in
the first place (KAN-52; ``colmap_stage.VIDEO_MAPPER_LADDER``).

The matcher has already estimated a fundamental matrix F for every
verified pair. With square pixels and the principal point at the image
centre, ``E = K^T F K`` is an essential matrix only at the true focal,
where its two non-zero singular values are equal (Mendonca and Cipolla,
CVPR 1999). The estimate is the focal that minimises the inlier-weighted
mean of ``(s1 - s2) / (s1 + s2)`` over every uncalibrated pair. On the
repo videos' matches it gave 893 / 874 px (Mailbox), 1554 / 1549 and
1536 / 1604 (PutCupInBowl), 1656 / 1663 (PutMarkerInCup) and 1401 / 1401
(the bin's stills), each against the focal of the model that placed.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

# two_view_geometries.config UNCALIBRATED: F estimated, not planar.
UNCALIBRATED = 3
MIN_INLIERS = 50
MIN_PAIRS = 20
# The search spans COLMAP's own plausible focal range, a tenth of it on
# each side trimmed (``pose_diagnosis.MIN_FOCAL_RATIO`` is 0.1 of the size).
LOW_RATIO, HIGH_RATIO = 0.3, 3.0
COARSE_STEPS = 400
_PAIR_ID_BASE = 2147483647


def essential_cost(
    fundamentals: np.ndarray, weights: np.ndarray, focal: float, cx: float, cy: float
) -> float:
    """Inlier-weighted mean of ``(s1 - s2) / (s1 + s2)`` of ``K^T F K``."""
    k = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]])
    singular = np.linalg.svd(k.T @ fundamentals @ k, compute_uv=False)
    ratio = (singular[:, 0] - singular[:, 1]) / (singular[:, 0] + singular[:, 1])
    return float(np.sum(weights * ratio) / np.sum(weights))


def estimate_focal(
    fundamentals: np.ndarray,
    weights: np.ndarray,
    width: int,
    height: int,
) -> dict[str, Any]:
    """The focal (px) whose ``K^T F K`` is most essential, or why there is none."""
    size = float(max(width, height))
    record: dict[str, Any] = {"ok": False, "pairs": len(fundamentals)}
    if len(fundamentals) < MIN_PAIRS:
        record["why"] = f"{len(fundamentals)} uncalibrated pairs, needs {MIN_PAIRS}"
        return record
    cx, cy = width / 2.0, height / 2.0
    grid = size * np.exp(
        np.linspace(np.log(LOW_RATIO), np.log(HIGH_RATIO), COARSE_STEPS)
    )
    costs = [essential_cost(fundamentals, weights, f, cx, cy) for f in grid]
    best = int(np.argmin(costs))
    if best in (0, len(grid) - 1):
        record["why"] = (
            f"cost is lowest at the edge of the search ({grid[best]:.0f} px)"
        )
        return record
    # Golden-section search between the coarse neighbours.
    low, high = float(grid[best - 1]), float(grid[best + 1])
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    for _ in range(40):
        a = high - ratio * (high - low)
        b = low + ratio * (high - low)
        if essential_cost(fundamentals, weights, a, cx, cy) < essential_cost(
            fundamentals, weights, b, cx, cy
        ):
            high = b
        else:
            low = a
    focal = (low + high) / 2.0
    record.update(
        ok=True,
        focal_px=round(float(focal), 2),
        ratio_to_size=round(float(focal / size), 4),
        cost=round(essential_cost(fundamentals, weights, focal, cx, cy), 5),
        colmap_guess_px=1.2 * size,
    )
    return record


def database_fundamentals(database: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Every uncalibrated pair's F (row-major blob) and inlier count, and the camera size.

    COLMAP stores F so that ``x2^T F x1 = 0`` in pixels, row by row: on a
    fresh PutMarkerInCup database the median epipolar distance of the
    inliers was 0.76 px read row-major and 23 px read transposed.
    """
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as con:
        cameras = list(con.execute("SELECT width, height FROM cameras"))
        rows = list(
            con.execute(
                "SELECT rows, F FROM two_view_geometries WHERE config = ? AND rows >= ?",
                (UNCALIBRATED, MIN_INLIERS),
            )
        )
    if len(cameras) != 1:
        raise ValueError(f"expected one shared camera, found {len(cameras)}")
    width, height = int(cameras[0][0]), int(cameras[0][1])
    blobs = [(n, f) for n, f in rows if f is not None and len(f) == 72]
    fundamentals = np.array(
        [np.frombuffer(f, dtype=np.float64).reshape(3, 3) for _, f in blobs]
    )
    weights = np.array([float(n) for n, _ in blobs])
    return fundamentals.reshape(-1, 3, 3), weights, width, height


def apply_focal_prior(database: Path, focal: float) -> None:
    """Start the shared camera at ``focal``, distortion 0, and mark it a prior.

    COLMAP's mapper initialises from the camera row in the database; the
    principal point stays at the image centre.
    """
    with sqlite3.connect(database) as con:
        (camera_id, model, width, height, params) = con.execute(
            "SELECT camera_id, model, width, height, params FROM cameras"
        ).fetchone()
        values = np.frombuffer(params, dtype=np.float64).copy()
        # SIMPLE_PINHOLE, SIMPLE_RADIAL, RADIAL: f, cx, cy, then distortion.
        if model not in (0, 2, 3):
            raise ValueError(f"camera model {model} has more than one focal")
        values[0] = focal
        values[1], values[2] = width / 2.0, height / 2.0
        values[3:] = 0.0
        con.execute(
            "UPDATE cameras SET params = ?, prior_focal_length = 1 WHERE camera_id = ?",
            (values.tobytes(), camera_id),
        )


def self_calibrate(database: Path) -> dict[str, Any]:
    """Estimate the focal from ``database`` and, when that works, write it there."""
    try:
        fundamentals, weights, width, height = database_fundamentals(Path(database))
        record = estimate_focal(fundamentals, weights, width, height)
        if record["ok"]:
            apply_focal_prior(Path(database), float(record["focal_px"]))
            record["applied"] = True
        return record
    except (sqlite3.Error, ValueError) as exc:
        return {"ok": False, "why": f"{type(exc).__name__}: {exc}"}
