# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SF-43: background completion and orientation on a converted scene (paper §3.3).

The paper removes the foreground Gaussians, completes the exposed background
from detected planes (colours from nearby plane surface, LaMa optional), and
its Orientation agent (Figure 7, agent 07) runs Isaac Lab sweeps that orient
the 3DGS background to the scene. This module is the CPU part, in Gaussian
space rather than on one image:

1. ``foreground_mask``: every Gaussian in any object's set is foreground.
2. ``fit_support_plane``: up to ``MAX_PLANES`` planes are peeled off the
   background with ``infill.fit_plane``; the support plane is the largest one
   with the objects on its side. Its normal is turned to point at them.
3. ``fill_support_holes``: the plane is gridded; a cell is a hole when an
   object stood over it and no background Gaussian landed in it. Each hole
   cell gets one new Gaussian on the plane, coloured from the nearest known
   cell (``infill.fill_hole``) or, when LaMa is installed and asked for, from
   a LaMa inpaint of the plane texture. Either way the method is recorded.
4. ``gravity_alignment``: the rigid transform that puts the support plane at
   z = 0 with its normal on +Z, so gravity is -Z. The residual tilt is
   measured by refitting the plane after the transform.
5. ``refine_with_sweep``: the hook for the Isaac Lab sweep. It needs the GPU
   box, so on CPU it records ``not_run``; with a ``settle`` callable it scores
   small tilts and keeps the best.

Yaw stays arbitrary: gravity fixes two of three rotation angles, and the
editor's ``background_io`` already treats a registered scan's yaw as
arbitrary (AGENTS.md). Positions stay in the splat's units; metric scale is
the bin pipeline's job (``bin_scale``).

    python -m sceneagent.background_scene --out /workspace/sf4-smoke/bin
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sceneagent import infill
from sceneagent.bin_association import PLY_TYPES

# Up to this many planes are peeled from the background before the support
# plane is chosen (a bin has a floor and walls). Ours, not the paper's.
MAX_PLANES = 3
# A peeled plane must hold this share of the remaining background points to
# count. Ours.
MIN_PLANE_SHARE = 0.05
# Foreground Gaussians this far above the support plane, in units of the
# plane's inlier band, still "stand over" a cell. Objects resting on the plane
# are within a few bands; a floater high above does not open a hole. Ours.
STANDING_HEIGHT_BANDS = 60.0
# Grid cell = this many times the mean inlier spacing on the plane, so a cell
# usually holds a few Gaussians. Ours.
CELL_SPACINGS = 2.0
# Small tilts the sweep hook tries, in degrees, about the plane's two axes.
SWEEP_TILTS_DEG = (0.0, 1.0, -1.0)
# 3DGS DC colour: rgb = 0.5 + SH_C0 * f_dc.
SH_C0 = 0.28209479177387814

Settle = Callable[[np.ndarray], float]


def foreground_mask(count: int, sets: Sequence[np.ndarray]) -> np.ndarray:
    """True for every Gaussian index named in any object's set."""
    mask = np.zeros(int(count), dtype=bool)
    for indices in sets:
        chosen = np.asarray(indices, dtype=np.int64).ravel()
        if chosen.size and (chosen.min() < 0 or chosen.max() >= count):
            raise ValueError("an object set names a Gaussian outside the splat")
        mask[chosen] = True
    return mask


def fit_support_plane(
    background: np.ndarray,
    foreground: np.ndarray,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """The plane the objects rest on, with its normal pointing at them.

    Planes are peeled in order of size. The support plane is the largest one
    whose up side (the side the median foreground Gaussian lies on) holds the
    objects, which rules out a wall the objects merely stand beside.
    """
    remaining = np.arange(len(background))
    planes: list[dict[str, Any]] = []
    fg_centre = np.median(foreground, axis=0) if len(foreground) else None
    for _ in range(MAX_PLANES):
        if len(remaining) < 3:
            break
        try:
            plane = infill.fit_plane(background[remaining], seed=seed)
        except infill.PlaneFitError:
            break
        inliers = remaining[np.asarray(plane["inliers"], dtype=np.int64)]
        if len(inliers) < MIN_PLANE_SHARE * len(remaining):
            break
        normal = np.asarray(plane["normal"], dtype=float)
        origin = np.asarray(plane["origin"], dtype=float)
        if fg_centre is not None and float(np.dot(fg_centre - origin, normal)) < 0:
            normal = -normal
        # Under: the share of foreground whose foot on this plane falls inside
        # the plane's own footprint. A wall scores low, the floor high.
        planes.append(
            {
                "origin": origin,
                "normal": normal,
                "inliers": inliers,
                "threshold": float(plane["threshold"]),
                "residual_rms": float(plane["residual_rms"]),
                "under": _share_over(background[inliers], foreground, origin, normal),
            }
        )
        remaining = np.setdiff1d(remaining, inliers, assume_unique=True)
    if not planes:
        raise infill.PlaneFitError("no plane in the background")
    # Among planes the objects stand over, the one they rest on is the one
    # they are closest to: a bin wall also has objects in front of it, but
    # most of them sit far from it.
    top = max(p["under"] for p in planes)
    candidates = [p for p in planes if p["under"] >= 0.5 * top] or planes
    if fg_centre is not None:
        for p in candidates:
            p["median_height"] = float(
                np.median(np.abs((foreground - p["origin"]) @ p["normal"]))
            )
        best = min(candidates, key=lambda p: (p["median_height"], -len(p["inliers"])))
    else:
        best = max(candidates, key=lambda p: len(p["inliers"]))
    best["n_planes"] = len(planes)
    return best


def _share_over(
    support: np.ndarray, foreground: np.ndarray, origin: np.ndarray, normal: np.ndarray
) -> float:
    if not len(foreground):
        return 0.0
    u_axis, v_axis = infill.plane_axes(normal)
    s = np.c_[(support - origin) @ u_axis, (support - origin) @ v_axis]
    f = np.c_[(foreground - origin) @ u_axis, (foreground - origin) @ v_axis]
    lo, hi = s.min(axis=0), s.max(axis=0)
    above = (foreground - origin) @ normal > 0
    inside = np.all((f >= lo) & (f <= hi), axis=1)
    return float(np.mean(above & inside))


def _cell_size(points_uv: np.ndarray) -> float:
    lo, hi = points_uv.min(axis=0), points_uv.max(axis=0)
    area = max(float(np.prod(hi - lo)), 1e-12)
    spacing = np.sqrt(area / max(len(points_uv), 1))
    return CELL_SPACINGS * float(spacing)


def fill_support_holes(
    positions: np.ndarray,
    colors: np.ndarray,
    plane: dict[str, Any],
    foreground: np.ndarray,
    *,
    use_lama: bool = False,
    work_dir: Path | None = None,
    cell: float | None = None,
) -> dict[str, Any]:
    """New Gaussians on the support plane where objects hid it.

    ``positions``/``colors`` are the background Gaussians (colours 0..1 RGB),
    ``plane`` comes from ``fit_support_plane``. Returns the new positions and
    colours, the grid and its masks, and how the colours were made.
    """
    origin = np.asarray(plane["origin"], dtype=float)
    normal = np.asarray(plane["normal"], dtype=float)
    u_axis, v_axis = infill.plane_axes(normal)
    support = positions[plane["inliers"]]
    support_rgb = colors[plane["inliers"]]
    uv = np.c_[(support - origin) @ u_axis, (support - origin) @ v_axis]
    size = float(cell) if cell else _cell_size(uv)
    lo = uv.min(axis=0)
    shape = np.floor((uv.max(axis=0) - lo) / size).astype(int) + 1
    width, height = int(shape[0]), int(shape[1])
    known = np.zeros((height, width), dtype=bool)
    image = np.zeros((height, width, 3), dtype=np.float64)
    counts = np.zeros((height, width), dtype=np.int64)
    cols, rows = _cells(uv, lo, size, width, height)
    np.add.at(image, (rows, cols), support_rgb)
    np.add.at(counts, (rows, cols), 1)
    known = counts > 0
    image[known] /= counts[known][:, None]
    # Where an object stood: its Gaussians near the plane, projected down.
    height_above = (foreground - origin) @ normal
    band = max(float(plane["threshold"]), 1e-9)
    standing = foreground[
        (height_above > -band) & (height_above < STANDING_HEIGHT_BANDS * band)
    ]
    covered = np.zeros_like(known)
    if len(standing):
        f_uv = np.c_[(standing - origin) @ u_axis, (standing - origin) @ v_axis]
        inside = np.all((f_uv >= lo) & (f_uv <= lo + shape * size), axis=1)
        f_cols, f_rows = _cells(f_uv[inside], lo, size, width, height)
        covered[f_rows, f_cols] = True
        # An object's Gaussians can be sparser than the grid; a cell between
        # two covered cells is still under the object. Closing (dilate, then
        # erode) by one cell fills those gaps without growing the outline.
        from scipy.ndimage import binary_closing

        covered = binary_closing(covered, iterations=1, border_value=0) | covered
    hole = covered & ~known
    rgb8 = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    method, lama_status = "nearest_plane_sample", "not_requested"
    filled = rgb8
    if hole.any():
        filled = infill.fill_hole(rgb8, known, hole)
        if use_lama:
            painted, lama_status = _lama_fill(rgb8, hole, work_dir)
            if painted is not None:
                filled, method = painted, "lama"
    h_rows, h_cols = np.nonzero(hole)
    centres_uv = lo + (np.c_[h_cols, h_rows] + 0.5) * size
    new_positions = origin + centres_uv[:, :1] * u_axis + centres_uv[:, 1:] * v_axis
    new_colors = filled[h_rows, h_cols].astype(np.float64) / 255.0
    return {
        "positions": new_positions,
        "colors": new_colors,
        "n_filled": int(hole.sum()),
        "n_known_cells": int(known.sum()),
        "n_covered_cells": int(covered.sum()),
        "cell": size,
        "grid": [height, width],
        "color_method": method,
        "lama": lama_status,
        "texture_before": np.where(known[..., None], rgb8, 0).astype(np.uint8),
        "texture_after": np.where((known | hole)[..., None], filled, 0).astype(
            np.uint8
        ),
    }


def _cells(uv: np.ndarray, lo: np.ndarray, size: float, width: int, height: int):
    cols = np.clip(np.floor((uv[:, 0] - lo[0]) / size).astype(int), 0, width - 1)
    rows = np.clip(np.floor((uv[:, 1] - lo[1]) / size).astype(int), 0, height - 1)
    return cols, rows


def _lama_fill(
    rgb8: np.ndarray, hole: np.ndarray, work_dir: Path | None
) -> tuple[np.ndarray | None, str]:
    """LaMa on the plane texture; ``None`` when it did not write one."""
    if work_dir is None:
        return None, "no_work_dir"
    work_dir.mkdir(parents=True, exist_ok=True)
    texture = infill.write_rgb_png(
        work_dir / "plane_texture.png", rgb8.reshape(len(rgb8), -1).tolist()
    )
    mask = work_dir / "plane_hole_mask.png"
    infill._write_lama_mask(mask, hole)
    painted = infill.lama_inpaint(
        texture, work_dir / "plane_texture_lama.png", mask_path=mask
    )
    status = str(painted.get("lama"))
    if not (painted.get("emitted") and painted.get("artifact")):
        return None, status
    from PIL import Image

    with Image.open(painted["artifact"]) as handle:
        out = np.asarray(handle.convert("RGB").resize((rgb8.shape[1], rgb8.shape[0])))
    return np.where(hole[..., None], out, rgb8).astype(np.uint8), status


def rotation_onto_z(normal: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit ``normal`` onto +Z (Rodrigues)."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(n, z)
    s, c = float(np.linalg.norm(axis)), float(np.dot(n, z))
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = axis / s
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * kx + (1 - c) * (kx @ kx)


def gravity_alignment(
    plane: dict[str, Any], support: np.ndarray, foreground: np.ndarray
) -> dict[str, Any]:
    """4x4 transform putting the support plane at z = 0, normal on +Z.

    The residual is measured, not assumed: the support points are moved and
    refit, and the angle between that normal and +Z is reported. ``tilt_deg``
    is how far the splat frame's +Z was from the plane normal before.
    """
    origin = np.asarray(plane["origin"], dtype=float)
    rotation = rotation_onto_z(plane["normal"])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ origin
    moved = apply_transform(transform, support)
    refit = infill.fit_plane(moved)
    refit_normal = np.asarray(refit["normal"], dtype=float)
    residual = float(np.degrees(np.arccos(min(1.0, abs(float(refit_normal[2]))))))
    z_before = float(np.clip(np.asarray(plane["normal"], dtype=float)[2], -1.0, 1.0))
    fg_z = (
        apply_transform(transform, foreground)[:, 2] if len(foreground) else np.zeros(0)
    )
    return {
        "transform": transform,
        "tilt_deg": float(np.degrees(np.arccos(z_before))),
        "residual_tilt_deg": residual,
        "plane_height_after": float(np.median(moved[:, 2])),
        "foreground_above_share": float(np.mean(fg_z > 0)) if len(fg_z) else None,
        "yaw": "arbitrary (gravity fixes two of three angles)",
    }


def apply_transform(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to N x 3 points."""
    return np.asarray(points, dtype=float) @ transform[:3, :3].T + transform[:3, 3]


def refine_with_sweep(
    transform: np.ndarray, settle: Settle | None = None
) -> dict[str, Any]:
    """The Orientation agent's Isaac Lab sweep (paper Fig. 7, agent 07).

    ``settle(transform) -> float`` runs the scene under ``transform`` in Isaac
    Lab with the objects resting and returns how far they drift (lower is
    better). That needs the GPU box, so without it the plane-fit transform is
    kept and the sweep is recorded as not run. With it, small tilts about X
    and Y are tried and the lowest drift wins; a tie keeps the plane fit.
    """
    if settle is None:
        return {
            "transform": transform,
            "sweep": "not_run",
            "reason": "the Isaac Lab sweep runs on the GPU box (SF-43 dc2)",
        }
    scores = []
    best, best_score = transform, float(settle(transform))
    scores.append({"tilt_x_deg": 0.0, "tilt_y_deg": 0.0, "drift": best_score})
    for tx in SWEEP_TILTS_DEG:
        for ty in SWEEP_TILTS_DEG:
            if tx == 0.0 and ty == 0.0:
                continue
            candidate = _tilted(transform, tx, ty)
            score = float(settle(candidate))
            scores.append({"tilt_x_deg": tx, "tilt_y_deg": ty, "drift": score})
            if score < best_score:
                best, best_score = candidate, score
    return {
        "transform": best,
        "sweep": "run",
        "scores": scores,
        "best_drift": best_score,
    }


def _tilted(transform: np.ndarray, tx_deg: float, ty_deg: float) -> np.ndarray:
    ax, ay = np.radians(tx_deg), np.radians(ty_deg)
    rx = np.array(
        [[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]]
    )
    ry = np.array(
        [[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]]
    )
    tilt = np.eye(4)
    tilt[:3, :3] = ry @ rx
    return tilt @ transform


def complete_scene_background(
    positions: np.ndarray,
    colors: np.ndarray,
    sets: Sequence[np.ndarray],
    out_dir: Path,
    *,
    use_lama: bool = False,
    settle: Settle | None = None,
) -> dict[str, Any]:
    """Run steps 1-5 and write ``out_dir/background_completion.json``.

    Also writes ``background_filled.npz`` (the kept background mask, the new
    Gaussians, the transform) and top-down before/after plane renders.
    """
    positions = np.asarray(positions, dtype=float)
    colors = np.asarray(colors, dtype=float)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fg = foreground_mask(len(positions), sets)
    background, foreground = positions[~fg], positions[fg]
    plane = fit_support_plane(background, foreground)
    filled = fill_support_holes(
        background, colors[~fg], plane, foreground, use_lama=use_lama, work_dir=out_dir
    )
    aligned = gravity_alignment(plane, background[plane["inliers"]], foreground)
    swept = refine_with_sweep(aligned["transform"], settle)
    infill.write_rgb_png(
        out_dir / "background_before.png",
        filled["texture_before"].reshape(filled["grid"][0], -1).tolist(),
    )
    infill.write_rgb_png(
        out_dir / "background_after.png",
        filled["texture_after"].reshape(filled["grid"][0], -1).tolist(),
    )
    np.savez(
        out_dir / "background_filled.npz",
        background_mask=~fg,
        new_positions=filled["positions"],
        new_colors=filled["colors"],
        transform=swept["transform"],
    )
    record = {
        "stage": "background_completion",
        "paper": "§3.3 plane completion, Fig. 7 agent 07 orientation",
        "n_gaussians": len(positions),
        "n_foreground_removed": int(fg.sum()),
        "plane": {
            "origin": [float(v) for v in plane["origin"]],
            "normal": [float(v) for v in plane["normal"]],
            "n_inliers": len(plane["inliers"]),
            "residual_rms": plane["residual_rms"],
            "n_planes_peeled": int(plane["n_planes"]),
            "foreground_over_share": plane["under"],
        },
        "fill": {
            k: filled[k]
            for k in (
                "n_filled",
                "n_known_cells",
                "n_covered_cells",
                "cell",
                "grid",
                "color_method",
                "lama",
            )
        },
        "orientation": {
            "transform": swept["transform"].tolist(),
            "tilt_deg": aligned["tilt_deg"],
            "residual_tilt_deg": aligned["residual_tilt_deg"],
            "plane_height_after": aligned["plane_height_after"],
            "foreground_above_share": aligned["foreground_above_share"],
            "yaw": aligned["yaw"],
            "sweep": swept["sweep"],
            **({"sweep_scores": swept["scores"]} if "scores" in swept else {}),
            **({"sweep_reason": swept["reason"]} if "reason" in swept else {}),
        },
        "units": "splat (metric scale comes from bin_scale)",
        "rules": "ours: MAX_PLANES, MIN_PLANE_SHARE, STANDING_HEIGHT_BANDS, CELL_SPACINGS",
        "artifacts": [
            "background_before.png",
            "background_after.png",
            "background_filled.npz",
        ],
    }
    (out_dir / "background_completion.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record


def read_ply_xyz_rgb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Positions and DC colours (0..1) of a binary 3DGS PLY."""
    with Path(path).open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY")
        fields: list[tuple[str, str]] = []
        count = None
        in_vertex = False
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"{path} has no end_header")
            parts = line.decode("ascii").split()
            if parts == ["end_header"]:
                break
            if parts[:1] == ["element"]:
                in_vertex = parts[1] == "vertex"
                count = int(parts[2]) if in_vertex else count
            elif parts[:1] == ["property"] and in_vertex:
                fields.append((parts[2], PLY_TYPES[parts[1]]))
        data = np.fromfile(handle, dtype=np.dtype(fields), count=count or 0)
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    names = {name for name, _ in fields}
    if {"f_dc_0", "f_dc_1", "f_dc_2"} <= names:
        dc = np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], axis=1)
        rgb = np.clip(0.5 + SH_C0 * dc.astype(np.float64), 0.0, 1.0)
    elif {"red", "green", "blue"} <= names:
        rgb = np.stack([data["red"], data["green"], data["blue"]], axis=1) / 255.0
    else:
        rgb = np.full_like(xyz, 0.5)
    return xyz, rgb


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="the bin run dir")
    parser.add_argument(
        "--lama", action="store_true", help="LaMa for the plane texture, if installed"
    )
    args = parser.parse_args(argv)
    splat = json.loads((args.out / "splat_evidence.json").read_text(encoding="utf-8"))
    positions, colors = read_ply_xyz_rgb(Path(splat["ply"]))
    with np.load(args.out / "association" / "objects.npz") as sets:
        object_sets = [sets[name] for name in sets.files]
    record = complete_scene_background(
        positions, colors, object_sets, args.out / "background", use_lama=args.lama
    )
    print(
        json.dumps(
            {
                "fill": record["fill"],
                "orientation": {
                    k: record["orientation"][k]
                    for k in ("tilt_deg", "residual_tilt_deg", "sweep")
                },
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
