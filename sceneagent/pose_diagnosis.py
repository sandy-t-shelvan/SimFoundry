# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Why a video could not be posed: numbers read from COLMAP's own outputs.

When no mapper rung passes the pose gate, the ledger reason must say what
was wrong with the video, not only that the gate failed (KAN-52). This
reads the match database the run just wrote and the camera of each
attempt's largest model, and names the first likely cause that the
numbers support. The cut-offs below label a failure for a person; they do
not decide a pass. The pose gate (``reconstruction_gate``) is unchanged.
"""

from __future__ import annotations

import sqlite3
import statistics
import struct
from pathlib import Path
from typing import Any

# COLMAP camera model id -> (name, number of params, number of focal params).
# From colmap/sensor/models.h; the order of params is focal(s), principal
# point, then the extra (distortion) params.
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3, 1),
    1: ("PINHOLE", 4, 2),
    2: ("SIMPLE_RADIAL", 4, 1),
    3: ("RADIAL", 5, 1),
    4: ("OPENCV", 8, 2),
    5: ("OPENCV_FISHEYE", 8, 2),
    6: ("FULL_OPENCV", 12, 2),
    7: ("FOV", 5, 2),
    8: ("SIMPLE_RADIAL_FISHEYE", 4, 1),
    9: ("RADIAL_FISHEYE", 5, 1),
    10: ("THIN_PRISM_FISHEYE", 12, 2),
}
# COLMAP's own defaults for discarding a camera as bogus
# (Mapper.min_focal_length_ratio, max_focal_length_ratio, max_extra_param).
MIN_FOCAL_RATIO = 0.1
MAX_FOCAL_RATIO = 10.0
MAX_EXTRA_PARAM = 1.0
# two_view_geometries.config values (colmap/estimators/two_view_geometry.h).
PLANAR_OR_PANORAMIC = frozenset({4, 5, 6})
UNVERIFIED = frozenset({0, 1})  # UNDEFINED, DEGENERATE
_PAIR_ID_BASE = 2147483647

# Labels for a person (ours, not the gate's).
FEW_KEYPOINTS = 500
FEW_ADJACENT_INLIERS = 50
MOSTLY_PLANAR = 0.6
CONNECTED_SHARE = 0.8


def read_cameras_bin(blob: bytes) -> list[dict[str, Any]]:
    """Parse COLMAP's ``cameras.bin``."""
    try:
        (count,) = struct.unpack_from("<Q", blob, 0)
        offset = 8
        cameras = []
        for _ in range(count):
            camera_id, model_id, width, height = struct.unpack_from(
                "<iiQQ", blob, offset
            )
            offset += 24
            if model_id not in CAMERA_MODELS:
                raise ValueError(f"unknown camera model {model_id}")
            name, n_params, n_focal = CAMERA_MODELS[model_id]
            params = list(struct.unpack_from(f"<{n_params}d", blob, offset))
            offset += 8 * n_params
            cameras.append(
                {
                    "camera_id": camera_id,
                    "model": name,
                    "width": width,
                    "height": height,
                    "focal_px": params[:n_focal],
                    "extra": params[n_focal + 2 :],
                }
            )
    except struct.error as exc:
        raise ValueError(f"truncated cameras.bin: {exc}") from exc
    if offset != len(blob):
        raise ValueError("cameras.bin has trailing bytes")
    return cameras


def camera_is_plausible(camera: dict[str, Any]) -> bool:
    """COLMAP's bogus-camera rule with its default limits, as a predicate.

    Every focal over ``max(width, height)`` inside [0.1, 10] and every
    extra parameter at most 1 in magnitude. A camera that fails it is the
    one COLMAP's image filter deregisters images for.
    """
    size = max(int(camera["width"]), int(camera["height"]))
    if size <= 0:
        return False
    focal_ok = all(
        MIN_FOCAL_RATIO <= f / size <= MAX_FOCAL_RATIO for f in camera["focal_px"]
    )
    return focal_ok and all(abs(k) <= MAX_EXTRA_PARAM for k in camera["extra"])


def match_stats(database: Path) -> dict[str, Any]:
    """Keypoints and verified matches, in frame (name) order, from the database."""
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as con:
        names = dict(con.execute("SELECT image_id, name FROM images"))
        keypoints = dict(con.execute("SELECT image_id, rows FROM keypoints"))
        pairs = list(
            con.execute("SELECT pair_id, rows, config FROM two_view_geometries")
        )
    order = sorted(names, key=lambda image_id: names[image_id])
    position = {image_id: i for i, image_id in enumerate(order)}
    verified: dict[tuple[int, int], tuple[int, int]] = {}
    for pair_id, rows, config in pairs:
        second = pair_id % _PAIR_ID_BASE
        first = (pair_id - second) // _PAIR_ID_BASE
        if first in position and second in position:
            a, b = sorted((position[first], position[second]))
            verified[(a, b)] = (int(rows or 0), int(config or 0))
    adjacent = [verified.get((i, i + 1), (0, 0)) for i in range(len(order) - 1)]
    good = [rows for rows, config in adjacent if rows and config not in UNVERIFIED]
    planar = [
        config
        for rows, config in adjacent
        if rows and config not in UNVERIFIED and config in PLANAR_OR_PANORAMIC
    ]
    counts = [int(keypoints.get(image_id) or 0) for image_id in order]
    return {
        "images": len(order),
        "median_keypoints": statistics.median(counts) if counts else 0,
        "adjacent_pairs": len(adjacent),
        "adjacent_verified": len(good),
        "median_adjacent_inliers": statistics.median([rows for rows, _ in adjacent])
        if adjacent
        else 0,
        "planar_share": round(len(planar) / len(good), 3) if good else None,
        "connected_share": _largest_component(len(order), verified) / len(order)
        if order
        else 0.0,
    }


def _largest_component(n: int, verified: dict[tuple[int, int], tuple[int, int]]) -> int:
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for (a, b), (rows, config) in verified.items():
        if rows and config not in UNVERIFIED:
            parent[find(a)] = find(b)
    sizes: dict[int, int] = {}
    for i in range(n):
        root = find(i)
        sizes[root] = sizes.get(root, 0) + 1
    return max(sizes.values(), default=0)


def diagnose(database: Path, attempts: list[dict[str, Any]]) -> str:
    """The first likely cause the numbers support, with the numbers.

    Never raises: a database it cannot read is itself the answer.
    """
    try:
        stats = match_stats(Path(database))
    except sqlite3.Error as exc:
        return f"match database unreadable ({exc})"
    numbers = (
        f"median {stats['median_keypoints']:g} keypoints per frame, "
        f"median {stats['median_adjacent_inliers']:g} verified matches between "
        f"neighbouring frames, {stats['connected_share']:.0%} of frames in one "
        f"match graph, planar-or-panoramic share {stats['planar_share']}"
    )
    bogus = [
        a
        for a in attempts
        if isinstance(a.get("camera"), dict) and not camera_is_plausible(a["camera"])
    ]
    if stats["images"] == 0:
        cause = "no frames reached the database"
    elif stats["median_keypoints"] < FEW_KEYPOINTS:
        cause = "too few features: frames are blurry, dark or textureless"
    elif stats["connected_share"] < CONNECTED_SHARE:
        cause = (
            "the video falls apart into separately matched segments: a cut, "
            "fast motion or an occlusion breaks frame-to-frame matching"
        )
    elif stats["median_adjacent_inliers"] < FEW_ADJACENT_INLIERS:
        cause = "neighbouring frames barely match: motion blur or fast camera motion"
    elif (stats["planar_share"] or 0) >= MOSTLY_PLANAR:
        cause = (
            "not enough parallax: the camera mostly turns in place, or the "
            "frames see one plane"
        )
    elif bogus:
        camera = bogus[-1]["camera"]
        cause = (
            "camera intrinsics diverged during mapping (focal "
            f"{camera['focal_px'][0]:.0f} px for a {camera['width']} px frame, "
            f"distortion {camera['extra']}): too little parallax to fix them"
        )
    else:
        cause = "the mapper could not grow one model over the frames"
    return f"{cause} ({numbers})"
