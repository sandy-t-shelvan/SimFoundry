"""Optional plane infill and LaMa for SceneAgent section 3.3.

The paper completes exposed background with detected planes and names LaMa
as optional texture inpainting. ``plane_infill`` still refuses to write an
image when the caller has neither a plane render nor association output.

``complete_background`` fits a plane to background points and fills the hole
from nearby plane-surface colors. That render is produced here. It is not a
copy of an input image. LaMa is optional and off by default: the import runs
only when ``use_lama`` is true, and a missing checkpoint does not produce a
forged inpaint. A side simple-lama would feed at 8 pixels is padded to 16
and cropped back, because the checkpoint's reflection pad rejects the 1x1
map that downsample produces. That call is still the real model. Counts
without background points still take the placeholder path, which is not a
completed background. Gaussian ``centers`` are not
treated as that plane. This fit does not read a camera pose.
"""

from __future__ import annotations

import itertools
import json
import struct
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

# Inlier band as a fraction of the cloud's median radius. This repository's
# own fitting parameter, not a value published in the paper and not a
# SceneAgent accept/reject threshold. Outliers farther than this, relative to
# the inlier cloud, drop out of the refit. A published residual was not given.
PLANE_INLIER_SCALE = 0.05
# Hole pixels copy the nearest splatted plane-surface sample. This
# repository's part-1 color method. It does not sample a posed input frame.
COLOR_METHOD = "nearest_plane_sample"
# Temporary distance block is at most this many hole pixels by this many
# sources. The bound is this repository's, so a photo-sized hole does not
# allocate an n_hole by n_source array. Not a published threshold.
FILL_HOLE_CHUNK = 64
_RANK_EPS = 1e-8
# big-lama downsamples three times by 2, then an FFC block reflection-pads
# by 1. That pad rejects a 1x1 map. simple-lama rounds each side up to a
# multiple of 8 and no further, so a side of 8 is fed as 8 and becomes
# [1, 128, 1, 1]. 16 is the smallest multiple of 8 that still has 2 pixels
# after those downs. Not a published input size.
_LAMA_DOWNSAMPLE = 8
_LAMA_MIN_SIDE = 16


def write_rgb_png(path, rows):
    """Write an 8-bit RGB PNG.

    Each row is a flat sequence of channel bytes, three per pixel. The IHDR
    width is that pixel count. Treating the channel count as the width yields
    a color-type-2 file Pillow cannot decode.
    """
    height = len(rows)
    channels = len(rows[0]) if height else 0
    if channels % 3 != 0:
        raise ValueError("each PNG row needs 3 channel bytes per pixel")
    width = channels // 3
    raw = b"".join(b"\x00" + bytes(row) for row in rows)

    def chunk(tag, data):
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(png)
    return destination


def plane_infill(plane_render_path, out_path=None, use_lama=False, association=None):
    """Refuse when no plane render and no association output exist.

    A present render, with ``association`` omitted, still does not run LaMa.
    That is the historical hook. Association output goes through
    ``complete_background`` so the missing piece can be plane completion.
    """
    source = Path(plane_render_path) if plane_render_path else None
    exists = bool(source and source.is_file() and source.stat().st_size > 0)
    destination = Path(out_path) if out_path else None
    if not exists and association:
        return complete_background(association, out_path=out_path, use_lama=use_lama)
    if (
        destination is not None
        and destination.exists()
        and (source is None or destination.resolve() != source.resolve())
    ):
        destination.unlink()
    if not exists:
        return {
            "ok": False,
            "emitted": False,
            "artifact": None,
            "lama": "not_started",
            "use_lama": bool(use_lama),
            "status": "open",
            "reason": "no plane render exists",
        }
    return {
        "ok": False,
        "emitted": False,
        "artifact": None,
        "lama": "not_started",
        "use_lama": bool(use_lama),
        "status": "open",
        "plane_render": str(source),
        "reason": "A plane render exists. LaMa and plane infill were not run, so no image was written.",
    }


def _kept_total(association):
    kept = association.get("n_passed_equation_1", association.get("kept"))
    total = association.get("n_gaussians", association.get("total"))
    if kept is None or total is None:
        return None, None
    return float(kept), float(total)


def detect_plane(association):
    """Axis of smallest spread across association centers, when centers exist.

    This is a placeholder detector for the plane-fill image. It is not the
    paper's unpublished plane fitter.
    """
    centers = association.get("centers") or association.get("points") or []
    if len(centers) < 3:
        return None
    sample = list(centers)[:5000]
    axes = list(zip(*sample))
    best: dict[str, Any] | None = None
    for index, axis in enumerate("xyz"):
        values = [float(value) for value in axes[index]]
        spread = max(values) - min(values)
        offset = sorted(values)[len(values) // 2]
        if best is None or spread < best["spread"]:
            best = {
                "normal_axis": axis,
                "offset": offset,
                "spread": spread,
                "samples": len(sample),
                "method": "ours_smallest_spread_axis",
            }
    return best


def plane_fill_from_association(association, out_path):
    """Write a plane-fill placeholder when association output can support one.

    The file is a flat field with a hole. It is not a LaMa inpaint.
    """
    if not association:
        return {
            "ok": False,
            "kind": None,
            "artifact": None,
            "lama": False,
            "plane": None,
            "reason": "plane completion is the missing piece",
        }
    kept, total = _kept_total(association)
    centers = association.get("centers") or association.get("points")
    if kept is None and not centers:
        return {
            "ok": False,
            "kind": None,
            "artifact": None,
            "lama": False,
            "plane": None,
            "reason": "plane completion is the missing piece",
        }
    plane = detect_plane(association)
    removed = 0.25
    if kept is not None and total:
        removed = min(0.9, max(0.05, 1.0 - (kept / total)))
    side = 16
    hole = max(1, round(side * removed))
    rows = []
    for y in range(side):
        row: list[int] = []
        for x in range(side):
            if x < hole and y < hole:
                row.extend((0, 0, 0))
            else:
                row.extend((180, 170, 150))
        rows.append(row)
    path = write_rgb_png(out_path, rows)
    return {
        "ok": True,
        "kind": "plane_fill_placeholder",
        "artifact": str(path),
        "lama": False,
        "forged_inpaint": False,
        "plane": plane,
        "removed_fraction": removed,
        "reason": "Plane-fill placeholder from association output. Not a LaMa result.",
    }


def _lama_backend():
    try:
        import simple_lama_inpainting  # noqa: F401
    except ImportError:
        return None, "LaMa weights are not installed"
    return "simple_lama_inpainting", ""


def lama_inpaint(image_path, out_path=None, mask_path=None):
    """Inpaint only when an image exists and LaMa is installed.

    No image in means no image out. Weights absent means the destination is
    not written. The function does not copy the input and call it an inpaint.
    """
    source = Path(image_path) if image_path else None
    destination = Path(out_path) if out_path else None
    exists = bool(source and source.is_file() and source.stat().st_size > 0)
    if (
        destination is not None
        and destination.exists()
        and (source is None or destination.resolve() != source.resolve())
    ):
        destination.unlink()
    if not exists:
        return {
            "ok": False,
            "emitted": False,
            "artifact": None,
            "forged": False,
            "status": "no_image",
            "weights_installed": None,
            "lama": "not_called",
            "reason": "no image in, no image out",
        }
    # `exists` already requires a real file. Restate it so the type checker
    # can see that the path used below is present.
    assert source is not None
    backend, why = _lama_backend()
    if backend is None:
        return {
            "ok": False,
            "emitted": False,
            "artifact": None,
            "forged": False,
            "status": "weights_absent",
            "weights_installed": False,
            "lama": "weights_absent",
            "image_in": str(source),
            "reason": why,
        }
    from PIL import Image

    image = Image.open(source).convert("RGB")
    if mask_path and Path(mask_path).is_file():
        mask = Image.open(mask_path).convert("L")
    else:
        mask = Image.new("L", image.size, 0)
        hole = max(1, image.size[0] // 4)
        mask.paste(255, (0, 0, hole, hole))
    result = _run_lama(image, mask)
    if destination is None:
        destination = source.with_name(source.stem + "_lama.png")
    result.save(destination)
    return {
        "ok": True,
        "emitted": True,
        "artifact": str(destination),
        "forged": False,
        "status": "inpainted",
        "weights_installed": True,
        "lama": "simple_lama_inpainting",
        "image_in": str(source),
        "reason": "LaMa wrote an inpaint.",
    }


def _lama_fed_side(side: int) -> int:
    """Side length simple-lama feeds the checkpoint, before our own pad.

    ``prepare_img_and_mask`` pads up to a multiple of 8. A multiple of 8 is
    unchanged. The model output keeps that padded size.
    """
    remainder = side % _LAMA_DOWNSAMPLE
    if remainder == 0:
        return side
    return side + (_LAMA_DOWNSAMPLE - remainder)


def _pad_to_lama_minimum(image: Any, mask: Any) -> tuple[Any, Any, bool]:
    """Pad only sides the checkpoint cannot run, and do not invent a hole.

    Edge replication is the image pad because reflect and symmetric both
    reject a pad wider than the side, and a 2 pixel image needs 14. The mask
    pad is 0 so the extra canvas is context, not a hole. Sides the library
    already feeds at 16 or more are returned unchanged.
    """
    from PIL import Image

    width, height = image.size
    out_width = _LAMA_MIN_SIDE if _lama_fed_side(width) < _LAMA_MIN_SIDE else width
    out_height = _LAMA_MIN_SIDE if _lama_fed_side(height) < _LAMA_MIN_SIDE else height
    if out_width == width and out_height == height:
        return image, mask, False
    rgb = np.asarray(image.convert("RGB"))
    padded_rgb = np.pad(
        rgb,
        ((0, out_height - height), (0, out_width - width), (0, 0)),
        mode="edge",
    )
    levels = np.asarray(mask.convert("L"))
    padded_mask = np.pad(
        levels,
        ((0, out_height - height), (0, out_width - width)),
        mode="constant",
        constant_values=0,
    )
    return (
        Image.fromarray(padded_rgb, mode="RGB"),
        Image.fromarray(padded_mask, mode="L"),
        True,
    )


def _run_lama(image, mask):
    """Call LaMa. Tests replace this with a fake that records the mask.

    A side simple-lama would feed at 8 pixels becomes a 1x1 map, and the
    checkpoint's reflection pad of 1 raises. Pad only those sides to 16,
    run SimpleLama, and crop back to the size the caller passed. A side
    already fed at 16 or more is not resized, so an image the library
    already accepts keeps the library's output size.
    """
    from simple_lama_inpainting import SimpleLama

    width, height = image.size
    image, mask, padded = _pad_to_lama_minimum(image, mask)
    result = SimpleLama()(image, mask)
    if padded:
        result = result.crop((0, 0, width, height))
    return result


def _write_lama_mask(path, hole: np.ndarray):
    """Write the hole as an RGB PNG. White is the hole, black is not.

    ``lama_inpaint`` converts it to ``L``. 255 and 0 survive that conversion,
    so the mask passed to LaMa is the hole and nothing outside it.
    """
    rows = []
    for y in range(hole.shape[0]):
        row: list[int] = []
        for x in range(hole.shape[1]):
            value = 255 if bool(hole[y, x]) else 0
            row.extend((value, value, value))
        rows.append(row)
    return write_rgb_png(path, rows)


class PlaneFitError(ValueError):
    """Raised when background points do not determine a plane, or the hole cannot be filled."""


def _as_points(points: Any) -> np.ndarray:
    """Return Nx3 float64 points, or raise ``PlaneFitError``."""
    try:
        array = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PlaneFitError("points must be an Nx3 array") from exc
    if array.ndim == 1 and array.size == 0:
        raise PlaneFitError("too few points to fit a plane")
    if array.ndim != 2 or array.shape[1] != 3:
        raise PlaneFitError("points must be an Nx3 array")
    if len(array) < 3:
        raise PlaneFitError("too few points to fit a plane")
    if not np.isfinite(array).all():
        raise PlaneFitError("points must be finite")
    return array


def _canonical_normal(normal: np.ndarray) -> np.ndarray:
    """Unit normal whose largest component is positive.

    The singular vector is unsigned. Flipping it here makes a z-up plane
    report +Z, so two fits of the same cloud agree.
    """
    unit = np.asarray(normal, dtype=np.float64)
    unit = unit / np.linalg.norm(unit)
    axis = int(np.argmax(np.abs(unit)))
    if unit[axis] < 0.0:
        unit = -unit
    return unit


def plane_axes(normal: Sequence[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Right-handed axes spanning the plane.

    The first world axis that is not parallel to ``normal`` is projected into
    the plane and used as U. V is ``normal × U``. For a +Z normal this is
    world X and world Y, so a test can predict pixels without a camera.
    """
    unit = _canonical_normal(np.asarray(normal, dtype=np.float64))
    tangent = None
    for axis in (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ):
        candidate = axis - unit * float(np.dot(axis, unit))
        if float(np.linalg.norm(candidate)) > 1e-6:
            tangent = candidate
            break
    if tangent is None:
        raise PlaneFitError("plane normal is not finite")
    u_axis = tangent / np.linalg.norm(tangent)
    v_axis = np.cross(unit, u_axis)
    return u_axis, v_axis


def _cloud_scale(points: np.ndarray) -> float:
    centered = points - points.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    if len(singular) == 0:
        return 0.0
    return float(singular[0])


def _is_collinear(points: np.ndarray) -> bool:
    """True when the second singular value is numerical dust.

    A line, or a pile of coincident points, has no second direction. A thin
    triangle whose second singular value clears ``_RANK_EPS`` is still a plane.
    """
    centered = points - points.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    if len(singular) < 2:
        return True
    scale = max(float(singular[0]), 1e-12)
    return float(singular[1]) <= _RANK_EPS * scale


def _default_threshold(points: np.ndarray) -> float:
    center = np.median(points, axis=0)
    radii = np.linalg.norm(points - center, axis=1)
    scale = float(np.median(radii))
    if scale <= 1e-12:
        scale = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    return max(1e-6, PLANE_INLIER_SCALE * scale)


def _sample_triplets(count: int, trials: int, seed: int) -> list[tuple[int, int, int]]:
    if count <= 24:
        return list(itertools.combinations(range(count), 3))
    rng = np.random.default_rng(seed)
    chosen: set[tuple[int, int, int]] = set()
    limit = min(trials, count * (count - 1) * (count - 2) // 6)
    attempts = 0
    while len(chosen) < limit and attempts < limit * 20:
        first, second, third = (
            int(item) for item in rng.choice(count, size=3, replace=False)
        )
        ordered = tuple(sorted((first, second, third)))
        chosen.add((ordered[0], ordered[1], ordered[2]))
        attempts += 1
    return list(chosen)


def _svd_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    origin = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - origin, full_matrices=False)
    normal = _canonical_normal(vh[-1])
    return origin, normal


def fit_plane(
    points: Any,
    *,
    residual: float | None = None,
    seed: int = 0,
    trials: int = 128,
) -> dict[str, Any]:
    """Robust plane through background points.

    A RANSAC sample of three points proposes a plane. The sample with the
    most inliers is refit by SVD on those inliers. The inlier band is
    ``residual`` when the caller passes one, otherwise
    ``PLANE_INLIER_SCALE`` times the median radius. That band is how outliers
    stay out of the normal. ``PLANE_INLIER_SCALE`` is this repository's own
    value, not a number published in the paper.

    Raises ``PlaneFitError`` when fewer than three points are given, or when
    the points are collinear (including a coincident pile).
    """
    cloud = _as_points(points)
    if _is_collinear(cloud):
        raise PlaneFitError("points are collinear")
    threshold = float(residual) if residual is not None else _default_threshold(cloud)
    if threshold <= 0.0:
        raise PlaneFitError("residual threshold must be positive")
    scale = max(_cloud_scale(cloud), 1e-12)
    best_inliers: np.ndarray | None = None
    best_count = 0
    best_cost = float("inf")
    for i0, i1, i2 in _sample_triplets(len(cloud), trials, seed):
        p0 = cloud[i0]
        edge_a = cloud[i1] - p0
        edge_b = cloud[i2] - p0
        cross = np.cross(edge_a, edge_b)
        area = float(np.linalg.norm(cross))
        if area <= _RANK_EPS * scale:
            continue
        normal = cross / area
        distances = np.abs((cloud - p0) @ normal)
        inliers = np.flatnonzero(distances <= threshold)
        count = int(inliers.size)
        if count < 3:
            continue
        cost = float(np.dot(distances[inliers], distances[inliers]))
        if count > best_count or (count == best_count and cost < best_cost):
            best_inliers = inliers
            best_count = count
            best_cost = cost
    if best_inliers is None or best_count < 3:
        raise PlaneFitError("points are collinear")
    origin, normal = _svd_plane(cloud[best_inliers])
    distances = np.abs((cloud - origin) @ normal)
    inliers = np.flatnonzero(distances <= threshold)
    if inliers.size < 3 or _is_collinear(cloud[inliers]):
        raise PlaneFitError("points are collinear")
    origin, normal = _svd_plane(cloud[inliers])
    distances = np.abs((cloud - origin) @ normal)
    kept = distances[inliers]
    return {
        "origin": origin,
        "normal": normal,
        "inliers": inliers.astype(int),
        "n_inliers": int(inliers.size),
        "residual_rms": float(np.sqrt(np.mean(kept * kept))),
        "threshold": threshold,
        "fit": "ransac_svd",
    }


def _as_colors(colors: Any, count: int) -> np.ndarray:
    try:
        array = np.asarray(colors)
    except (TypeError, ValueError) as exc:
        raise PlaneFitError("colors must be Nx3, one RGB triple per point") from exc
    if array.shape != (count, 3):
        raise PlaneFitError("colors must be Nx3, one RGB triple per point")
    values = array.astype(np.float64)
    if not np.isfinite(values).all():
        raise PlaneFitError("colors must be finite")
    # A 0..1 float cloud is a fraction of full scale. An 8-bit cloud is already
    # in 0..255. Mixing the two would paint the hole with the wrong values.
    if np.issubdtype(array.dtype, np.floating) and float(np.max(values)) <= 1.0:
        values = values * 255.0
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def fill_hole(image: Any, known: Any, hole: Any) -> np.ndarray:
    """Fill hole pixels from the nearest known plane-surface color.

    ``image`` is HxWx3 uint8. ``known`` marks pixels that carry a splatted
    sample. ``hole`` marks pixels to replace. A source is a known pixel that
    is not itself a hole: a sample that landed in the removed region is the
    object, not the background. Distance is squared Euclidean on pixel
    coordinates. Ties take the smaller row, then the smaller column.

    Non-hole pixels are copied unchanged, including zeros where no sample
    landed. The hole is the only region this function paints.
    """
    canvas = np.array(np.asarray(image), dtype=np.uint8, copy=True)
    if canvas.ndim != 3 or canvas.shape[2] != 3:
        raise PlaneFitError("image must have shape HxWx3")
    known_mask = np.asarray(known, dtype=bool)
    hole_mask = np.asarray(hole, dtype=bool)
    height, width = canvas.shape[:2]
    if known_mask.shape != (height, width) or hole_mask.shape != (height, width):
        raise PlaneFitError("known and hole masks must match the image")
    if not hole_mask.any():
        return canvas
    source = known_mask & ~hole_mask
    if not source.any():
        raise PlaneFitError("no plane-surface color to fill the hole")
    # argwhere is row-major, so the first minimum is the smaller row, then column.
    source_yx = np.argwhere(source)
    hole_yx = np.argwhere(hole_mask)
    nearest = _nearest_source(hole_yx, source_yx)
    canvas[hole_yx[:, 0], hole_yx[:, 1]] = canvas[
        source_yx[nearest, 0], source_yx[nearest, 1]
    ]
    return canvas


def _chunk_distances(hole_block: np.ndarray, source_block: np.ndarray) -> np.ndarray:
    """Squared distances for one chunk. Both blocks are at most ``FILL_HOLE_CHUNK``."""
    diff = hole_block[:, None, :].astype(np.int64) - source_block[None, :, :].astype(
        np.int64
    )
    return np.sum(diff * diff, axis=2)


def _nearest_source(hole_yx: np.ndarray, source_yx: np.ndarray) -> np.ndarray:
    """Nearest source index for each hole pixel.

    Blocks are ``FILL_HOLE_CHUNK`` on each side, so the temporary array is
    not ``n_hole`` by ``n_source``. Sources stay in row-major order and a
    tie keeps the earlier source, matching one full ``argmin``.
    """
    best_dist = np.full(len(hole_yx), np.iinfo(np.int64).max, dtype=np.int64)
    best_index = np.zeros(len(hole_yx), dtype=np.int64)
    for source_start in range(0, len(source_yx), FILL_HOLE_CHUNK):
        source_block = source_yx[source_start : source_start + FILL_HOLE_CHUNK]
        for hole_start in range(0, len(hole_yx), FILL_HOLE_CHUNK):
            hole_block = hole_yx[hole_start : hole_start + FILL_HOLE_CHUNK]
            dist2 = _chunk_distances(hole_block, source_block)
            local = np.argmin(dist2, axis=1)
            local_dist = dist2[np.arange(len(hole_block)), local]
            span = slice(hole_start, hole_start + len(hole_block))
            better = local_dist < best_dist[span]
            best_dist[span] = np.where(better, local_dist, best_dist[span])
            best_index[span] = np.where(better, local + source_start, best_index[span])
    return best_index


def _to_pixels(coords: np.ndarray, size: int) -> np.ndarray:
    if size <= 1:
        return np.zeros(len(coords), dtype=int)
    low = float(coords.min())
    high = float(coords.max())
    span = high - low
    if span <= 1e-12:
        return np.zeros(len(coords), dtype=int)
    scaled = (coords - low) / span * float(size - 1)
    return np.clip(np.rint(scaled), 0, size - 1).astype(int)


def _rows_from_image(image: np.ndarray) -> list[list[int]]:
    rows: list[list[int]] = []
    for y in range(image.shape[0]):
        row: list[int] = []
        for x in range(image.shape[1]):
            pixel = image[y, x]
            row.extend((int(pixel[0]), int(pixel[1]), int(pixel[2])))
        rows.append(row)
    return rows


def render_plane_background(
    points: Any,
    colors: Any,
    hole: Any,
    out_path: Any = None,
) -> dict[str, Any]:
    """Orthographic render of the fitted plane, with the hole filled.

    Only the fitted inliers are projected. An outlier must not stretch the
    UV box or become a color source. Points are projected on the plane axes
    from ``plane_axes``. Their bounding box maps onto the hole mask's pixel
    grid. Samples that project into the hole are not color sources. Hole
    pixels take the nearest source color. The PNG, when ``out_path`` is set,
    is that array and nothing else.
    """
    cloud = _as_points(points)
    rgb = _as_colors(colors, len(cloud))
    mask = np.asarray(hole, dtype=bool)
    if mask.ndim != 2:
        raise PlaneFitError("hole mask must be 2-dimensional")
    height, width = int(mask.shape[0]), int(mask.shape[1])
    if height < 1 or width < 1:
        raise PlaneFitError("hole mask must be 2-dimensional")
    plane = fit_plane(cloud)
    inliers = np.asarray(plane["inliers"], dtype=int)
    inlier_cloud = cloud[inliers]
    inlier_rgb = rgb[inliers]
    u_axis, v_axis = plane_axes(plane["normal"])
    relative = inlier_cloud - plane["origin"]
    xs = _to_pixels(relative @ u_axis, width)
    ys = _to_pixels(relative @ v_axis, height)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    known = np.zeros((height, width), dtype=bool)
    for index in range(len(inlier_cloud)):
        y = int(ys[index])
        x = int(xs[index])
        if mask[y, x] or known[y, x]:
            continue
        image[y, x] = inlier_rgb[index]
        known[y, x] = True
    filled = fill_hole(image, known, mask)
    artifact = None
    if out_path is not None:
        artifact = str(write_rgb_png(out_path, _rows_from_image(filled)))
    return {"image": filled, "plane": plane, "artifact": artifact, "known": known}


def background_ledger_record(
    method: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    decision: str = "accept",
    plane_inlier_scale: float = PLANE_INLIER_SCALE,
    color_method: str = COLOR_METHOD,
) -> dict[str, Any]:
    """One background run: method ``plane`` or ``lama``, plus inputs and outputs.

    ``plane_inlier_scale`` is this repository's fitting parameter, not a
    published residual. ``color_method`` names how hole pixels were colored.
    """
    if method not in ("plane", "lama"):
        raise ValueError("background ledger method must be plane or lama")
    return {
        "item": "background",
        "section": "3.3",
        "method": method,
        "inputs": inputs,
        "outputs": outputs,
        "decision": decision,
        "plane_inlier_scale": float(plane_inlier_scale),
        "color_method": color_method,
    }


def append_background_ledger(
    ledger_path: Any, record: dict[str, Any]
) -> dict[str, Any]:
    """Append ``record`` to ``ledger.json`` without rewriting earlier entries."""
    path = Path(ledger_path)
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("ledger.json must be an object")
    else:
        payload = {"section": "3.3", "entries": []}
    entries = payload.setdefault("entries", [])
    if not isinstance(entries, list):
        raise ValueError("ledger entries must be a list")
    entries.append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return record


def _lookup_points(points: Any, association: Any) -> tuple[Any, str | None]:
    """Background points only. ``centers`` stay on the placeholder path.

    Association centers are Gaussian samples. They are not the exposed
    background, and fitting them would couple this step to whatever produced
    the centers.
    """
    if points is not None:
        return points, "points"
    if isinstance(association, dict):
        if "background_points" in association:
            return association["background_points"], "association.background_points"
        if "points" in association:
            return association["points"], "association.points"
    return None, None


def _lookup_colors(colors: Any, association: Any) -> Any:
    if colors is not None:
        return colors
    if isinstance(association, dict) and "colors" in association:
        return association["colors"]
    return None


def _lookup_hole(hole: Any, association: Any) -> Any:
    if hole is not None:
        return hole
    if isinstance(association, dict) and "hole" in association:
        return association["hole"]
    return None


def _json_vec(vector: Any) -> list[float]:
    return [float(item) for item in vector]


def _ledger_destination(out_path: Any, ledger_path: Any) -> Path | None:
    if ledger_path is not None:
        return Path(ledger_path)
    if out_path is not None:
        return Path(out_path).parent / "ledger.json"
    return None


def _complete_from_fitted_plane(
    points: Any,
    point_source: str,
    association: Any,
    hole_image: Any,
    out_path: Any,
    use_lama: bool,
    colors: Any,
    hole: Any,
    ledger_path: Any,
) -> dict[str, Any]:
    """Fit, render, optionally inpaint, and record the run."""
    if out_path is None:
        raise PlaneFitError("out_path is required to write the rendered background")
    rgb = _lookup_colors(colors, association)
    if rgb is None:
        raise PlaneFitError("colors are required to fill the hole from plane surfaces")
    mask = _lookup_hole(hole, association)
    if mask is None:
        raise PlaneFitError("a hole mask is required to complete the background")
    destination = Path(out_path)
    # LaMa, when requested, writes a second file. The plane render keeps its
    # own name so a missing checkpoint cannot erase the only image.
    plane_path = (
        destination.with_name("plane_background.png") if use_lama else destination
    )
    rendered = render_plane_background(points, rgb, mask, plane_path)
    image = rendered["image"]
    plane = rendered["plane"]
    height, width = int(image.shape[0]), int(image.shape[1])
    hole_image_path = str(hole_image) if hole_image else None
    inputs = {
        "n_points": len(_as_points(points)),
        "point_source": point_source,
        "n_hole_pixels": int(np.asarray(mask, dtype=bool).sum()),
        "image_size": [height, width],
        "use_lama": bool(use_lama),
        "hole_image": hole_image_path,
    }
    outputs: dict[str, Any] = {
        "artifact": str(plane_path),
        "width": width,
        "height": height,
        "plane_normal": _json_vec(plane["normal"]),
        "plane_origin": _json_vec(plane["origin"]),
        "n_inliers": int(plane["n_inliers"]),
        "forged": False,
    }
    method = "plane"
    lama_status = "not_requested"
    artifact = str(plane_path)
    emitted = True
    reason = "Fitted a plane to the background points and filled the hole from nearby plane-surface colors."
    if use_lama:
        mask_file = destination.with_name("hole_mask.png")
        _write_lama_mask(mask_file, np.asarray(mask, dtype=bool))
        painted = lama_inpaint(plane_path, destination, mask_path=mask_file)
        lama_status = str(painted.get("lama"))
        if painted.get("emitted") and painted.get("artifact"):
            method = "lama"
            artifact = str(painted["artifact"])
            outputs["plane_artifact"] = str(plane_path)
            outputs["artifact"] = artifact
            reason = "LaMa inpainted the fitted-plane render."
        else:
            outputs["lama"] = lama_status
            reason = (
                "Fitted a plane and filled the hole. LaMa did not write an inpaint."
            )
    record = background_ledger_record(method, inputs, outputs)
    ledger_file = _ledger_destination(destination, ledger_path)
    if ledger_file is not None:
        append_background_ledger(ledger_file, record)
    return {
        "ok": True,
        "emitted": emitted,
        "artifact": artifact,
        "forged": False,
        "plane_completion": "fitted_plane",
        "lama": lama_status,
        "use_lama": bool(use_lama),
        "method": method,
        "plane": {
            "normal": outputs["plane_normal"],
            "origin": outputs["plane_origin"],
            "n_inliers": outputs["n_inliers"],
            "residual_rms": float(plane["residual_rms"]),
            "fit": plane["fit"],
        },
        "shape": [height, width, 3],
        "ledger": record,
        "reason": reason,
    }


def complete_background(
    association=None,
    hole_image=None,
    out_path=None,
    use_lama=False,
    points=None,
    colors=None,
    hole=None,
    ledger_path=None,
):
    """Fit a plane to background points and fill the hole.

    ``use_lama`` defaults to false. LaMa is imported only on that path, and
    only after the plane render exists. Counts without background points keep
    the placeholder path.
    """
    supplied, source_name = _lookup_points(points, association)
    if supplied is not None:
        return _complete_from_fitted_plane(
            supplied,
            source_name or "points",
            association,
            hole_image,
            out_path,
            bool(use_lama),
            colors,
            hole,
            ledger_path,
        )
    return _complete_without_points(association, hole_image, out_path, use_lama)


def _complete_without_points(
    association=None, hole_image=None, out_path=None, use_lama=False
):
    """Placeholder path. Not a fitted plane and not a completed background."""
    image = Path(hole_image) if hole_image else None
    placeholder = None
    if image is None or not image.is_file() or image.stat().st_size == 0:
        image = None
        if association is not None:
            placeholder_path = out_path
            if use_lama or placeholder_path is None:
                placeholder_path = (
                    Path(out_path).with_name("plane_fill_placeholder.png")
                    if out_path
                    else Path("plane_fill_placeholder.png")
                )
            placeholder = plane_fill_from_association(association, placeholder_path)
            if placeholder.get("artifact"):
                image = Path(placeholder["artifact"])
    if image is None:
        return {
            "ok": False,
            "emitted": False,
            "artifact": None,
            "forged": False,
            "plane_completion": "missing",
            "lama": "not_called",
            "placeholder": placeholder,
            "reason": "plane completion is the missing piece",
        }
    if not use_lama:
        return {
            "ok": bool(placeholder and placeholder.get("ok")) or image.is_file(),
            "emitted": False,
            "artifact": str(image),
            "forged": False,
            "plane_completion": "placeholder" if placeholder else "image",
            "lama": "not_requested",
            "placeholder": placeholder,
            "reason": "An image exists. LaMa was not requested.",
        }
    painted = lama_inpaint(image, out_path)
    painted["plane_completion"] = "placeholder" if placeholder else "image"
    painted["placeholder"] = placeholder
    return painted
