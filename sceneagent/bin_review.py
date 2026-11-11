# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SF-19: the per-object reviewer on the bin. It draws the views, chooses a retry, and logs it.

Section 3.1's reviewer inspects, for each object, a registration overlay,
eight isolated views (six axis-aligned, two oblique) and two views of the
object in the assembled scene. :mod:`sceneagent.reviewer_views` named those
eleven files; this module draws them (CPU only, numpy and Pillow):

- ``registration_overlay``: the object's photo, its SAM mask outline in
  green and the placed mesh's projected silhouette in red;
- ``axis_*`` and ``oblique_*``: the placed mesh, shaded, with the object's
  Gaussians drawn over it in blue, seen orthographically from each side;
- ``assembled_1`` and ``assembled_2``: every placed object projected onto the
  two photos where this object's mask is largest, this one in red.

Each view is measured: the overlay by its mask IoU, the isolated views by the
IoU of the mesh's and the Gaussians' footprints. The retry is chosen from
those numbers (ours; the paper's reviewer is a VLM looking at the images,
which is SF-18 and has no key on the box):

- gate accepted: ``keep``;
- isolated views agree poorly (mean < ``ISOLATED_OK``): the mesh sits wrongly
  on its own Gaussians, so ``icp_flip`` restarts ICP from the fit turned half
  a turn about each axis;
- isolated views agree but the overlay does not (< ``OVERLAY_OK``): the pose
  fits the Gaussians and the shape misses the outline, so
  ``sam3d_next_view`` regenerates the mesh from the object's next-largest mask;
- otherwise ``keep``.

A retry is kept only when it improves the number that triggered it (the
two-way fit for ``icp_flip``, the silhouette for ``sam3d_next_view``). Every
decision, the numbers behind it and the outcome go into ``bin_review.json``
and the ledger, and the bin is scored again into a second score file.

    python -m sceneagent.bin_review --out /workspace/sf4-smoke/bin
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from sceneagent import bin_place
from sceneagent.bin_association import project, splat_to_colmap
from sceneagent.bin_run import DATA_DIR, SCENE, SOURCE
from sceneagent.reviewer_views import REQUIRED_VIEWS, write_view_manifest
from sceneagent.similarity_icp import apply_similarity, silhouette_iou

VIEWS_ITEM = "reviewer views"
RETRY_ITEM = "reviewer retries"
SCORE_ITEM = "bin score (reviewed)"
REQUIRED = (VIEWS_ITEM, RETRY_ITEM, SCORE_ITEM)
# With --verdicts the VLM chooses the retries. Its items and files sit beside
# the rule's, so the two reviews of one placement can be compared.
VLM_RETRY_ITEM = "reviewer retries (VLM)"
VLM_SCORE_ITEM = "bin score (VLM reviewed)"
VLM_REQUIRED = (VIEWS_ITEM, VLM_RETRY_ITEM, VLM_SCORE_ITEM)
VLM_DECISIONS = ("keep", "icp_flip", "sam3d_next_view")
# Fit distances are rounded to 5 decimals and placement is deterministic
# (runs 42934419274 and 42939458339 matched exactly), so any real difference
# means a different scene.
PLACEMENT_TOL = 1e-6
# Ours. The silhouette cut reuses the ours_iteration_1 gate's minimum.
OVERLAY_OK = 0.5
ISOLATED_OK = 0.5
TILE_PX = 256
FOOTPRINT_DIVISOR = 4
RENDER_POINTS = 60000
OBLIQUE = {
    "oblique_1": np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0),
    "oblique_2": np.array([-1.0, -1.0, 1.0]) / np.sqrt(3.0),
}
AXES = {
    "axis_pos_x": np.array([1.0, 0, 0]),
    "axis_neg_x": np.array([-1.0, 0, 0]),
    "axis_pos_y": np.array([0, 1.0, 0]),
    "axis_neg_y": np.array([0, -1.0, 0]),
    "axis_pos_z": np.array([0, 0, 1.0]),
    "axis_neg_z": np.array([0, 0, -1.0]),
}
HALF_TURNS = {
    "x": np.diag([1.0, -1.0, -1.0]),
    "y": np.diag([-1.0, 1.0, -1.0]),
    "z": np.diag([-1.0, -1.0, 1.0]),
}

Runner = Callable[..., Any]


def _dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    from scipy.ndimage import binary_dilation

    return binary_dilation(mask, iterations=iterations) if iterations else mask


def view_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Right and up axes for looking along ``direction``."""
    d = direction / np.linalg.norm(direction)
    reference = (
        np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    )
    right = np.cross(d, reference)
    right /= np.linalg.norm(right)
    return right, np.cross(right, d)


def render_isolated(
    points: np.ndarray,
    normals: np.ndarray,
    gaussians: np.ndarray,
    direction: np.ndarray,
    size: int = TILE_PX,
) -> tuple[np.ndarray, float | None]:
    """Orthographic shaded mesh with the Gaussians over it; and their footprint IoU."""
    d = direction / np.linalg.norm(direction)
    right, up = view_basis(d)
    centre = points.mean(axis=0)
    both = np.vstack([points, gaussians])
    extent = (
        1.1
        * 2
        * max(
            np.abs((both - centre) @ right).max(),
            np.abs((both - centre) @ up).max(),
            1e-9,
        )
    )

    def pixels(cloud: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = (cloud - centre) @ right / extent + 0.5
        y = 0.5 - (cloud - centre) @ up / extent
        col = np.clip((x * (size - 1)).astype(int), 0, size - 1)
        row = np.clip((y * (size - 1)).astype(int), 0, size - 1)
        return row, col

    row, col = pixels(points)
    g_row, g_col = pixels(gaussians)
    # Footprints on a coarser grid, so gaps between samples do not count as
    # disagreement. Measured in the frame that holds every Gaussian.
    coarse = -(-size // FOOTPRINT_DIVISOR)
    mesh_mask = np.zeros((coarse, coarse), dtype=bool)
    mesh_mask[row // FOOTPRINT_DIVISOR, col // FOOTPRINT_DIVISOR] = True
    g_coarse = np.zeros((coarse, coarse), dtype=bool)
    g_coarse[g_row // FOOTPRINT_DIVISOR, g_col // FOOTPRINT_DIVISOR] = True
    iou = silhouette_iou(_dilate(mesh_mask, 1), _dilate(g_coarse, 1))
    # The picture is framed on the mesh instead: a few floaters far out would
    # otherwise shrink the object to a dot (run 42934419274, object 7). A
    # Gaussian outside that frame is not drawn. The IoU above is unchanged.
    view = (
        1.3
        * 2
        * max(
            np.abs((points - centre) @ right).max(),
            np.abs((points - centre) @ up).max(),
            1e-9,
        )
    )
    image = np.full((size, size, 3), 255, dtype=np.uint8)
    x, y = (points - centre) @ right / view + 0.5, 0.5 - (points - centre) @ up / view
    row = np.clip((y * (size - 1)).astype(int), 0, size - 1)
    col = np.clip((x * (size - 1)).astype(int), 0, size - 1)
    depth = (points - centre) @ d
    shade = (60 + 180 * np.abs(normals @ d)).astype(np.uint8)
    order = np.argsort(-depth)  # far first, so the nearest point is written last
    image[row[order], col[order]] = shade[order, None]
    gx = (gaussians - centre) @ right / view + 0.5
    gy = 0.5 - (gaussians - centre) @ up / view
    inside = (gx >= 0) & (gx <= 1) & (gy >= 0) & (gy <= 1)
    g_mask = np.zeros((size, size), dtype=bool)
    g_mask[
        (gy[inside] * (size - 1)).astype(int), (gx[inside] * (size - 1)).astype(int)
    ] = True
    image[g_mask] = (0.5 * image[g_mask] + 0.5 * np.array([30, 90, 255])).astype(
        np.uint8
    )
    return image, iou


def draw_overlay(
    photo: np.ndarray, mask: np.ndarray, rendered: np.ndarray
) -> np.ndarray:
    """Photo dimmed, mask outline green, rendered silhouette tinted red."""
    out = (photo.astype(float) * 0.6).astype(np.uint8)
    out[rendered] = (0.5 * out[rendered] + 0.5 * np.array([255, 40, 40])).astype(
        np.uint8
    )
    outline = mask & ~_erode(mask)
    out[_dilate(outline, 2)] = np.array([40, 255, 40], dtype=np.uint8)
    return out


def crop_to_object(
    image: np.ndarray, region: np.ndarray, margin: float = 0.5, min_px: int = 96
) -> np.ndarray:
    """``image`` cut to ``region``'s box plus a margin, enlarged to at least 512 px.

    The whole photo shrunk into one tile leaves a small object a few pixels
    wide (run 42934419274, object 7), too small to judge red against green.
    Display only: nothing is measured on the crop.
    """
    from PIL import Image

    rows, cols = np.nonzero(region)
    if rows.size == 0:
        return image
    height, width = region.shape
    half = max(int(np.ptp(rows)) + 1, int(np.ptp(cols)) + 1, min_px) * (0.5 + margin)
    cy, cx = (rows.min() + rows.max()) / 2, (cols.min() + cols.max()) / 2
    top, bottom = int(max(cy - half, 0)), int(min(cy + half, height))
    left, right = int(max(cx - half, 0)), int(min(cx + half, width))
    crop = Image.fromarray(image[top:bottom, left:right])
    scale = max(1.0, 512 / max(crop.size))
    if scale > 1.0:
        crop = crop.resize(
            (round(crop.width * scale), round(crop.height * scale)),
            Image.Resampling.LANCZOS,
        )
    return np.array(crop)


def _erode(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_erosion

    return binary_erosion(mask, iterations=1)


def draw_assembled(
    photo: np.ndarray,
    clouds: dict[str, np.ndarray],
    focus: str,
    image: dict[str, Any],
    camera: dict[str, Any],
    transform: np.ndarray,
    dp_scale: float,
) -> np.ndarray:
    """Every placed object projected onto the photo; ``focus`` in red, the rest grey."""
    out = (photo.astype(float) * 0.6).astype(np.uint8)
    height, width = out.shape[:2]
    for name, cloud in sorted(clouds.items(), key=lambda kv: kv[0] == focus):
        u, v, z = project(splat_to_colmap(cloud, transform, dp_scale), image, camera)
        with np.errstate(invalid="ignore"):
            ok = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        mask = np.zeros((height, width), dtype=bool)
        mask[np.floor(v[ok]).astype(int), np.floor(u[ok]).astype(int)] = True
        colour = np.array([255, 40, 40]) if name == focus else np.array([200, 200, 200])
        out[_dilate(mask, 1)] = colour.astype(np.uint8)
    return out


def _thumb(array: np.ndarray, size: int = TILE_PX):
    from PIL import Image

    img = Image.fromarray(array)
    img.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def save_views(views: dict[str, np.ndarray], folder: Path, name: str) -> Path:
    """Write the eleven PNGs and one labelled contact sheet; return the sheet."""
    from PIL import Image, ImageDraw

    view_dir = folder / "reviewer_views"
    view_dir.mkdir(parents=True, exist_ok=True)
    tiles = []
    for view_name, _group in REQUIRED_VIEWS:
        Image.fromarray(views[view_name]).save(view_dir / f"{name}-{view_name}.png")
        tile = _thumb(views[view_name])
        ImageDraw.Draw(tile).text((4, 4), view_name, fill=(0, 0, 0))
        tiles.append(tile)
    columns = 4
    rows = -(-len(tiles) // columns)
    sheet = Image.new("RGB", (columns * TILE_PX, rows * TILE_PX), "white")
    for k, tile in enumerate(tiles):
        sheet.paste(tile, ((k % columns) * TILE_PX, (k // columns) * TILE_PX))
    path = folder / f"{name}.png"
    sheet.save(path)
    return path


def choose_retry(
    gate: str,
    overlay_iou: float | None,
    isolated_mean: float | None,
    has_next_view: bool,
) -> str:
    """The retry the drawn views call for. See the module docstring."""
    if gate == "accept":
        return "keep"
    if isolated_mean is None or isolated_mean < ISOLATED_OK:
        return "icp_flip"
    if (overlay_iou is None or overlay_iou < OVERLAY_OK) and has_next_view:
        return "sam3d_next_view"
    return "keep"


def load_verdicts(path: Path) -> tuple[dict[str, Any], dict[int, dict[str, str]]]:
    """The VLM verdict file from :mod:`sceneagent.vlm_review`, checked.

    It is model output, so every field is validated: an object is an integer
    and a decision one of the three retries. Anything else raises. Returns the
    file's metadata (model, prompt hash, call count) and the verdict per object.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("objects"), list):
        raise ValueError("verdict file has no objects list")
    verdicts: dict[int, dict[str, str]] = {}
    for item in payload["objects"]:
        number = item.get("object") if isinstance(item, dict) else None
        decision = item.get("decision") if isinstance(item, dict) else None
        if not isinstance(number, int) or isinstance(number, bool):
            raise ValueError(f"verdict object {number!r} is not an integer")
        if decision not in VLM_DECISIONS:
            raise ValueError(f"verdict decision {decision!r} is not a retry")
        reason = item.get("reason")
        verdicts[number] = {
            "decision": decision,
            "reason": reason[:400] if isinstance(reason, str) else "",
        }
    meta = {
        key: payload.get(key)
        for key in (
            "source_run",
            "model",
            "prompt_sha256",
            "prompt_source",
            "vlm_calls",
        )
    }
    meta["failed_objects"] = [
        f.get("object") for f in payload.get("failed") or [] if isinstance(f, dict)
    ]
    judged = payload.get("judged_placement")
    if judged is None:
        meta["judged_placement"] = None
    else:
        if not isinstance(judged, list):
            raise ValueError("judged_placement is not a list")
        placement: dict[int, float] = {}
        for item in judged:
            number = item.get("object") if isinstance(item, dict) else None
            fit = item.get("fit_total") if isinstance(item, dict) else None
            if not isinstance(number, int) or isinstance(number, bool):
                raise ValueError(f"judged object {number!r} is not an integer")
            if not isinstance(fit, (int, float)) or isinstance(fit, bool):
                raise ValueError(f"judged fit_total {fit!r} is not a number")
            placement[number] = float(fit)
        meta["judged_placement"] = placement
    return meta, verdicts


def check_placement(
    judged: dict[int, float] | None, records: dict[int, dict[str, Any]]
) -> None:
    """Refuse verdicts judged on a different scene than the one placed now.

    The verdict file carries the fit distance of every object in the run
    whose sheets the VLM saw. Object numbers are only meaningful for that
    placement: after a new grouping, "object 8" can be another object, and
    the review would still finish with a plausible score.
    """
    if judged is None:
        raise ValueError(
            "the verdict file records no judged placement, so it cannot be "
            "checked against this scene; judge this run's sheets again"
        )
    if set(judged) != set(records):
        raise ValueError(
            f"verdicts were judged on objects {sorted(judged)}, "
            f"this placement has {sorted(records)}"
        )
    moved = [
        n
        for n, record in records.items()
        if abs(float(record["fit_total"]) - judged[n]) > PLACEMENT_TOL
    ]
    if moved:
        raise ValueError(
            f"objects {sorted(moved)} are placed differently from the judged run"
        )


def vlm_decision(
    verdict: dict[str, str] | None, has_next_view: bool
) -> tuple[str, str]:
    """The retry to run for one object under the VLM, and why.

    No verdict (the call failed) means no retry; the rule is not substituted.
    ``sam3d_next_view`` with no other view to use cannot run, so it keeps.
    """
    if verdict is None:
        return "keep", "no VLM verdict"
    decision = verdict["decision"]
    if decision == "sam3d_next_view" and not has_next_view:
        return "keep", "VLM chose sam3d_next_view; no other view"
    return decision, "VLM"


def flip_starts(
    similarity: dict[str, Any], source: np.ndarray
) -> dict[str, tuple[float, np.ndarray, np.ndarray]]:
    """The fit turned half a turn about each mesh axis, about the mesh's centre."""
    scale = float(similarity["scale"])
    rotation = np.asarray(similarity["rotation"], dtype=float)
    translation = np.asarray(similarity["translation"], dtype=float)
    centre = source.mean(axis=0)
    anchor = scale * rotation @ centre + translation
    starts = {}
    for axis, turn in HALF_TURNS.items():
        r = rotation @ turn
        starts[f"flip/{axis}"] = (scale, r, anchor - scale * r @ centre)
    return starts


def _flip_icp(
    inputs: dict[str, Any], record: dict[str, Any]
) -> tuple[str, dict[str, Any], dict[str, Any], Path, np.ndarray]:
    """ICP from the fit turned half a turn about each axis; the best of the three."""
    number = record["object"]
    glb = Path(record["glb"])
    target = inputs["positions"][inputs["sets"][f"object_{number}"]]
    source = bin_place.sample_mesh(glb, bin_place.ICP_SOURCE_POINTS)
    name, result, tried = bin_place.best_icp(
        source, target, flip_starts(record["similarity"], source), record["pose_scale"]
    )
    return name, result, tried, glb, target


def _similarity_json(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "scale": float(result["scale"]),
        "rotation": np.asarray(result["rotation"]).tolist(),
        "translation": np.asarray(result["translation"]).tolist(),
    }


def icp_flip_candidate(
    out: Path, inputs: dict[str, Any], record: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """The flip retry's result, returned whether or not the fit test would keep it.

    ``test_accepts`` is what the fit-distance test would decide; the caller
    decides what to keep.
    """
    name, result, tried, glb, target = _flip_icp(inputs, record)
    dense = bin_place.sample_mesh(glb, bin_place.POINTS_PER_OBJECT, seed=1)
    cloud = apply_similarity(
        dense, result["scale"], result["rotation"], result["translation"]
    )
    outcome: dict[str, Any] = {
        "kind": "icp_flip",
        "before_fit": record["fit_total"],
        "after_fit": tried["best_total"],
        "start": name,
        "test": "fit distance falls",
        "test_accepts": bool(tried["best_total"] < record["fit_total"]),
        "unchanged": bool(tried["best_total"] == record["fit_total"]),
        "glb": str(glb),
        "view": record["view"],
        "similarity": _similarity_json(result),
        **bin_place.measure_fit(out, inputs, record["object"], glb, target, result),
    }
    return cloud, outcome


def retry_icp_flip(
    out: Path, inputs: dict[str, Any], record: dict[str, Any]
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """ICP from the three half turns; return a new cloud only when the fit improves."""
    number = record["object"]
    name, result, tried, glb, target = _flip_icp(inputs, record)
    outcome: dict[str, Any] = {
        "before_fit": record["fit_total"],
        "after_fit": tried["best_total"],
        "start": name,
    }
    if tried["best_total"] >= record["fit_total"]:
        outcome["kept"] = False
        return None, outcome
    dense = bin_place.sample_mesh(glb, bin_place.POINTS_PER_OBJECT, seed=1)
    cloud = apply_similarity(
        dense, result["scale"], result["rotation"], result["translation"]
    )
    outcome.update(
        kept=True, **bin_place.measure_fit(out, inputs, number, glb, target, result)
    )
    outcome["similarity"] = _similarity_json(result)
    return cloud, outcome


def regenerate(
    out: Path,
    inputs: dict[str, Any],
    numbers: list[int],
    following: dict[int, dict[str, Any]],
    retry_dir: Path,
    runner: Runner = subprocess.run,
    sam3d_config: Callable[[], Path] | None = None,
) -> dict[int, tuple[np.ndarray | None, dict[str, Any] | None, dict[str, Any]]]:
    """SAM 3D on each object's next view, in one worker batch, then ICP.

    Returns, per object, the placed cloud and its placement record (both None
    when SAM 3D failed) and the worker's result entry. Nothing is kept here.
    """
    from sceneagent import bin_sam3d

    retry_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        {
            "object": n,
            "image": str(out / "images" / following[n]["view"]),
            "masks": str(out / "masks" / f"{Path(following[n]['view']).stem}.npz"),
            "mask_index": following[n]["mask_index"],
        }
        for n in numbers
    ]
    jobs_path = retry_dir / "jobs.json"
    jobs_path.write_text(json.dumps(jobs, indent=2) + "\n", encoding="utf-8")
    config = (sam3d_config or bin_sam3d.find_config)()
    runner(
        bin_sam3d.worker_command(jobs_path, retry_dir, config),
        env=bin_sam3d.worker_environment(),
        cwd=bin_sam3d.SAM3D_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=bin_sam3d.WORKER_TIMEOUT_S,
    )
    results_path = retry_dir / "results.json"
    results = (
        json.loads(results_path.read_text(encoding="utf-8"))["objects"]
        if results_path.is_file()
        else []
    )
    by_object = {r["object"]: r for r in results}
    placed: dict[
        int, tuple[np.ndarray | None, dict[str, Any] | None, dict[str, Any]]
    ] = {}
    for n in numbers:
        result = by_object.get(n) or {"ok": False, "error": "no result"}
        if not result.get("ok"):
            placed[n] = (None, None, result)
            continue
        glb = retry_dir / f"object_{n}.glb"
        cloud, record = bin_place.place_one(
            out, inputs, n, result["pose"], following[n]["view"], glb
        )
        placed[n] = (cloud, record, result)
    return placed


def next_views(
    association: dict[str, Any],
    detections: dict[str, list[dict[str, Any]]],
    used: dict[int, str],
) -> dict[int, dict[str, Any]]:
    """Each object's largest mask in a view other than the one SAM 3D already used."""
    area = {
        (view, d["mask_index"]): int(d.get("mask_area_px", 0))
        for view, dets in detections.items()
        for d in dets
        if d.get("kept")
    }
    chosen = {}
    for obj in association["object_list"]:
        others = [m for m in obj["masks"] if m["view"] != used.get(obj["object"])]
        if others:
            best = max(others, key=lambda m: area.get((m["view"], m["mask_index"]), 0))
            chosen[obj["object"]] = best
    return chosen


def review_object(
    out: Path,
    inputs: dict[str, Any],
    record: dict[str, Any],
    clouds: dict[str, np.ndarray],
    assembled_views: list[str],
    folder: Path | None = None,
) -> dict[str, Any]:
    """Draw the eleven views of one placed object and measure them.

    ``folder`` defaults to ``reviewer/object_<n>``; the pair builder draws a
    retry's result elsewhere so it does not overwrite the before views.
    """
    import trimesh
    from PIL import Image

    number = record["object"]
    name = f"object_{number}"
    glb = Path(record["glb"])
    sim = record["similarity"]
    rotation = np.asarray(sim["rotation"])
    mesh = trimesh.load(glb, force="mesh", process=False)
    samples, faces = trimesh.sample.sample_surface(mesh, RENDER_POINTS, seed=3)
    points = apply_similarity(
        np.asarray(samples), sim["scale"], rotation, sim["translation"]
    )
    normals = np.asarray(mesh.face_normals)[faces] @ rotation.T
    gaussians = inputs["positions"][inputs["sets"][name]]
    views: dict[str, np.ndarray] = {}
    isolated: dict[str, float | None] = {}
    for view_name, direction in {**AXES, **OBLIQUE}.items():
        views[view_name], isolated[view_name] = render_isolated(
            points, normals, gaussians, direction
        )
    images, cameras = inputs["images"], inputs["cameras"]
    view = record["view"]
    image = images[view]
    camera = cameras[image["camera_id"]]
    world = splat_to_colmap(points, inputs["transform"], inputs["scale"])
    rendered = bin_place.render_mask(
        world, image, camera, bin_place.SILHOUETTE_DILATE_PX
    )
    obj_masks = next(
        o for o in inputs["association"]["object_list"] if o["object"] == number
    )
    mask_ref = next(
        (m for m in obj_masks["masks"] if m["view"] == view), obj_masks["masks"][0]
    )
    stack = np.load(out / "masks" / f"{Path(mask_ref['view']).stem}.npz")["masks"]
    mask = stack[mask_ref["mask_index"]]
    with Image.open(out / "images" / view) as handle:
        photo = np.array(handle.convert("RGB"))
    views["registration_overlay"] = crop_to_object(
        draw_overlay(photo, mask, rendered), mask | rendered
    )
    overlay_iou = silhouette_iou(rendered, mask)
    for k, assembled in enumerate(assembled_views[:2], 1):
        a_image = images[assembled]
        with Image.open(out / "images" / assembled) as handle:
            a_photo = np.array(handle.convert("RGB"))
        views[f"assembled_{k}"] = draw_assembled(
            a_photo,
            clouds,
            name,
            a_image,
            cameras[a_image["camera_id"]],
            inputs["transform"],
            inputs["scale"],
        )
    if "assembled_2" not in views:  # an object seen in one view repeats it
        views["assembled_2"] = views["assembled_1"]
    folder = folder or out / "reviewer" / name
    sheet = save_views(views, folder, name)
    manifest = write_view_manifest(folder, object_name=name)
    scores = [v for v in isolated.values() if v is not None]
    return {
        "sheet": str(sheet),
        "manifest_status": manifest["status"],
        "overlay_iou": overlay_iou,
        "isolated_iou": isolated,
        "isolated_mean": float(np.mean(scores)) if scores else None,
    }


def run_bin_review(
    out: Path | str,
    data: Path | str = DATA_DIR,
    *,
    runner: Runner = subprocess.run,
    sam3d_config: Callable[[], Path] | None = None,
    verdicts: Path | str | None = None,
) -> int:
    """Place, draw and measure every object's views, retry, rescore.

    With ``verdicts`` (a :mod:`sceneagent.vlm_review` file) the VLM's choice
    replaces the rule's for each object, the rule's is logged beside it, and
    the evidence, ledger items and score file are the ``_vlm`` ones.
    """
    out, data = Path(out), Path(data)
    use_vlm = verdicts is not None
    required = VLM_REQUIRED if use_vlm else REQUIRED
    ledger_path = out / "ledger.json"
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        ledger = {"entries": []}
    entries = [e for e in ledger.get("entries", []) if e.get("item") not in required]
    views_entry: dict[str, Any] = {"item": VIEWS_ITEM}
    retry_entry: dict[str, Any] = {"item": VLM_RETRY_ITEM if use_vlm else RETRY_ITEM}
    score_entry: dict[str, Any] = {"item": VLM_SCORE_ITEM if use_vlm else SCORE_ITEM}
    evidence: dict[str, Any] = {
        "scene": SCENE,
        "rule": {
            "overlay_ok": OVERLAY_OK,
            "isolated_ok": ISOLATED_OK,
            "source": "ours",
        },
        "objects": [],
    }
    vlm: dict[int, dict[str, str]] = {}
    try:
        if verdicts is not None:
            evidence["vlm"], vlm = load_verdicts(Path(verdicts))
            evidence["rule"]["applied"] = "logged only; the VLM chooses"
        inputs = bin_place._load(out)
        placement = bin_place.place_objects(out, inputs)
        clouds: dict[str, np.ndarray] = dict(placement["placed"])
        # The same placement scored before any retry, so the retries' own
        # effect is the difference between the two files.
        unreviewed = bin_place.score_scene(data, inputs, clouds)
        evidence["score_before_review"] = {
            k: unreviewed[k] for k in ("f1_mean", "chamfer_mean_matched_m", "matches")
        }
        records = {r["object"]: r for r in placement["objects"] if r.get("placed")}
        if use_vlm:
            check_placement(evidence["vlm"]["judged_placement"], records)
            evidence["vlm"]["placement_check"] = "matches the judged run"
        detections = inputs["detections"]
        area = {
            (view, d["mask_index"]): int(d.get("mask_area_px", 0))
            for view, dets in detections.items()
            for d in dets
            if d.get("kept")
        }
        used = {n: r["view"] for n, r in records.items()}
        following = next_views(inputs["association"], detections, used)
        reviews: dict[int, dict[str, Any]] = {}
        for number, record in records.items():
            obj = next(
                o for o in inputs["association"]["object_list"] if o["object"] == number
            )
            by_area = sorted(
                obj["masks"], key=lambda m: -area.get((m["view"], m["mask_index"]), 0)
            )
            assembled = list(dict.fromkeys(m["view"] for m in by_area))
            review = review_object(out, inputs, record, clouds, assembled)
            # The fit the views show; a VLM run compares it with the judged run's.
            review["fit_total"] = record["fit_total"]
            rule = choose_retry(
                record["gate"],
                review["overlay_iou"],
                review["isolated_mean"],
                number in following,
            )
            if use_vlm:
                verdict = vlm.get(number)
                decision, source = vlm_decision(verdict, number in following)
                review.update(
                    decision=decision,
                    decision_source=source,
                    rule_decision=rule,
                    vlm_decision=verdict["decision"] if verdict else None,
                    vlm_reason=verdict["reason"] if verdict else None,
                )
            else:
                review["decision"] = rule
            reviews[number] = review
        drawn = sum(r["manifest_status"] == "present" for r in reviews.values())
        views_entry.update(
            decision="accept" if drawn == len(records) else "reject",
            use_gpu=False,
            objects=len(records),
            drawn=drawn,
        )
        # Retries: ICP flips on CPU, then one SAM 3D batch for the rest.
        for number, review in reviews.items():
            if review["decision"] == "icp_flip":
                cloud, outcome = retry_icp_flip(out, inputs, records[number])
                review["retry"] = outcome
                if cloud is not None:
                    clouds[f"object_{number}"] = cloud
        regenerate_numbers = [
            n for n, r in reviews.items() if r["decision"] == "sam3d_next_view"
        ]
        if regenerate_numbers:
            regenerated = regenerate(
                out,
                inputs,
                regenerate_numbers,
                following,
                out / "sam3d" / "retry",
                runner,
                sam3d_config,
            )
            for n in regenerate_numbers:
                cloud, new, result = regenerated[n]
                review = reviews[n]
                if cloud is None or new is None:
                    review["retry"] = {
                        "kept": False,
                        "error": result.get("error", "no result"),
                    }
                    continue
                before = records[n]["residuals"]["silhouette"] or 0.0
                after = new["residuals"]["silhouette"] or 0.0
                kept = after > before
                review["retry"] = {
                    "view": following[n]["view"],
                    "before_silhouette": before,
                    "after_silhouette": after,
                    "kept": kept,
                    "gate": new["gate"],
                }
                if kept:
                    clouds[f"object_{n}"] = cloud
        decisions = [r["decision"] for r in reviews.values()]
        kept = sum(bool(r.get("retry", {}).get("kept")) for r in reviews.values())
        retry_entry.update(
            decision="accept",
            use_gpu=bool(regenerate_numbers),
            keep=decisions.count("keep"),
            icp_flip=decisions.count("icp_flip"),
            sam3d_next_view=decisions.count("sam3d_next_view"),
            retries_kept=kept,
        )
        if use_vlm:
            retry_entry.update(
                vlm_model=evidence["vlm"].get("model"),
                vlm_calls=evidence["vlm"].get("vlm_calls"),
                agrees_with_rule=sum(
                    r["decision"] == r["rule_decision"] for r in reviews.values()
                ),
            )
        for number, review in reviews.items():
            evidence["objects"].append({"object": number, **review})
        score = bin_place.score_scene(
            data, inputs, clouds, tag="_vlm_reviewed" if use_vlm else "_reviewed"
        )
        evidence["score"] = score
        (out / score["file_name"]).write_text(
            json.dumps(score, indent=2) + "\n", encoding="utf-8"
        )
        score_entry.update(
            decision="accept",
            use_gpu=False,
            f1_mean=round(score["f1_mean"], 4),
            chamfer_mean_matched_m=score["chamfer_mean_matched_m"],
            f1_before_review=round(unreviewed["f1_mean"], 4),
            chamfer_before_review_m=unreviewed["chamfer_mean_matched_m"],
            delivered=len(clouds),
        )
        with zipfile.ZipFile(
            out / "review_sheets.zip", "w", zipfile.ZIP_DEFLATED
        ) as archive:
            for review in reviews.values():
                archive.write(review["sheet"], Path(review["sheet"]).name)
    except (OSError, ValueError, KeyError, RuntimeError, StopIteration) as exc:
        reason = f"{type(exc).__name__}: {exc}"[-800:]
        for entry in (views_entry, retry_entry, score_entry):
            if "decision" not in entry:
                entry.update(decision="reject", reason=reason)
    entries += [views_entry, retry_entry, score_entry]
    ledger.update(scene_name=f"{SOURCE} {SCENE}", source=SOURCE, entries=entries)
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    (out / ("bin_review_vlm.json" if use_vlm else "bin_review.json")).write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    outcomes = {e.get("item"): e.get("decision") for e in entries}
    return 0 if all(outcomes.get(item) == "accept" for item in required) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path(DATA_DIR))
    parser.add_argument(
        "--verdicts",
        type=Path,
        default=None,
        help="VLM verdict file from sceneagent.vlm_review; the VLM picks the retries",
    )
    args = parser.parse_args(argv)
    return run_bin_review(args.out, args.data, verdicts=args.verdicts)


if __name__ == "__main__":
    sys.exit(main())
