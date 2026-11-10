# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cut 2 phase 3: equation (1) association and SF-16 grouping on the bin splat.

For each of the 13 views, every Gaussian center of the bin splat is projected
through that view's COLMAP camera. Each image cell of
:func:`sceneagent.association.mask_cell_side_px` pixels keeps the minimum depth
of the centers landing in it (``z_hat``). A center belongs to a SAM mask when
it lands inside the mask and its depth passes the front-surface test of
equation (1). Masks from different views are then the same object when
:func:`sceneagent.association.masks_same_object` holds, and objects are the
connected components of that relation. An object keeps the union of its masks'
Gaussians (kappa 1, as in the paper's experiments).

Frames. ``ns-export gaussian-splat`` writes ``model.means`` in nerfstudio's
dataparser frame, and ``dataparser_transforms.json`` (beside the run's
``config.yml``) maps the COLMAP world into it: ``p = scale * (T @ [x; 1])``
(nerfstudio 1.1.5 ``save_dataparser_transform``, with ``applied_transform``
folded into ``T``). Centers are taken back to the COLMAP world to project,
and depths and distances are multiplied by ``scale``. Depths and distances
are therefore in the splat's units, which are this run's "reconstruction
units". ``T``'s rotation is orthonormal, so the scale is the only change of
length.

Units. The paper's cuts are in "reconstruction units", and it reports its
kitchen at 0.90-0.93 m per unit. A splat unit here is about 0.2 m, which
would make the 0.10-unit centroid gate 2 cm (first bin run: 94 masks, 62
objects, 288 overlapping pairs kept apart by that gate alone). So the splat
unit is measured in metres from the views (:mod:`sceneagent.bin_scale`,
Depth Anything 3 metric depth, RGB only) and every length is converted to
the paper's 0.915 m unit before the cuts. The constants are unchanged. The
dataset's metric camera centers (``scene_gt.json``) give a second
metres-per-unit figure that is reported only, as a check on the first.

``e`` in ``min(0.10, 0.6 e)`` is not defined in the paper. Here it is the
longest side of the 5th-95th percentile box of a mask's Gaussian centers,
taking the smaller of the two masks. That reading is ours and is recorded in
the evidence as such.

    python -m sceneagent.bin_association --out /workspace/sf4-smoke/bin
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
from collections.abc import Callable
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from sceneagent.association import (
    CENTROID_ABS,
    CENTROID_EXTENT_SCALE,
    CONTAINMENT_MIN,
    EPS_A,
    EPS_R,
    JACCARD_MIN,
    centroid_threshold,
    mask_cell_side_px,
    masks_same_object,
)
from sceneagent.bin_run import DATA_DIR, SCENE, SOURCE, VIEWS
from sceneagent.bin_scale import (
    PAPER_METRES_PER_UNIT,
    SCALE_RULE,
    cell_ratios,
    run_depth,
)
from sceneagent.bin_scale import summarize as summarize_scale

ITEM = "Gaussian association"
REQUIRED = (
    "benchmark scene",
    "COLMAP camera poses",
    "full-scene splat",
    "SAM masks",
    ITEM,
)
GT_OBJECTS = 20
EXTENT_PERCENTILES = (5.0, 95.0)
EXTENT_RULE = (
    "ours: the paper does not define e; longest side of the 5th-95th percentile "
    "box of the mask's Gaussian centers, the smaller of the two masks"
)

# COLMAP camera model ids and parameter counts (src/colmap/sensor/models.h).
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
}
PLY_TYPES = {
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
    "uchar": "u1",
    "uint8": "u1",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
}


def read_ply_xyz(path: Path) -> np.ndarray:
    """Vertex x, y, z of a binary little-endian PLY, as float64 N x 3."""
    with Path(path).open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY")
        fields: list[tuple[str, str]] = []
        count = None
        binary = False
        in_vertex = False
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"{path} has no end_header")
            parts = line.decode("ascii").split()
            if parts == ["end_header"]:
                break
            if parts[:1] == ["format"]:
                binary = parts[1] == "binary_little_endian"
            elif parts[:1] == ["element"]:
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    count = int(parts[2])
            elif parts[:1] == ["property"] and in_vertex:
                if parts[1] == "list" or parts[1] not in PLY_TYPES:
                    raise ValueError(f"{path}: unsupported vertex property {parts}")
                fields.append((parts[2], PLY_TYPES[parts[1]]))
        if not binary or count is None:
            raise ValueError(f"{path} is not a binary little-endian vertex PLY")
        data = np.fromfile(handle, dtype=np.dtype(fields), count=count)
    if len(data) != count:
        raise ValueError(f"{path} holds {len(data)} of {count} vertices")
    return np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)


def read_cameras_bin(path: Path) -> dict[int, dict[str, Any]]:
    blob = Path(path).read_bytes()
    (count,) = struct.unpack_from("<Q", blob, 0)
    offset = 8
    cameras = {}
    for _ in range(count):
        camera_id, model_id, width, height = struct.unpack_from("<iiQQ", blob, offset)
        offset += struct.calcsize("<iiQQ")
        if model_id not in CAMERA_MODELS:
            raise ValueError(f"COLMAP camera model id {model_id} is not supported")
        model, nparams = CAMERA_MODELS[model_id]
        params = struct.unpack_from(f"<{nparams}d", blob, offset)
        offset += 8 * nparams
        cameras[camera_id] = {
            "model": model,
            "width": int(width),
            "height": int(height),
            "params": list(params),
        }
    return cameras


def read_images_bin(path: Path) -> dict[str, dict[str, Any]]:
    """Pose and camera id of each registered image, keyed by file name."""
    blob = Path(path).read_bytes()
    (count,) = struct.unpack_from("<Q", blob, 0)
    offset = 8
    images = {}
    for _ in range(count):
        values = struct.unpack_from("<i4d3di", blob, offset)
        offset += struct.calcsize("<i4d3di")
        end = blob.index(b"\x00", offset)
        name = blob[offset:end].decode("utf-8")
        offset = end + 1
        (npoints,) = struct.unpack_from("<Q", blob, offset)
        offset += 8 + npoints * struct.calcsize("<ddq")
        images[name] = {
            "qvec": list(values[1:5]),
            "tvec": list(values[5:8]),
            "camera_id": values[8],
        }
    return images


def qvec_to_rotation(qvec: list[float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in qvec)
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def splat_to_colmap(
    points: np.ndarray, transform: np.ndarray, scale: float
) -> np.ndarray:
    """Invert ``p = scale * (T @ [x; 1])`` for N x 3 splat-frame points."""
    transform = np.asarray(transform, dtype=float)
    rotation, translation = transform[:, :3], transform[:, 3]
    return (np.asarray(points, dtype=float) / float(scale) - translation) @ rotation


def project(points: np.ndarray, image: dict[str, Any], camera: dict[str, Any]):
    """Pixel u, v and camera-frame z of COLMAP-world points (OpenCV convention)."""
    cam = points @ qvec_to_rotation(image["qvec"]).T + np.asarray(image["tvec"])
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        x, y = cam[:, 0] / z, cam[:, 1] / z
    p = camera["params"]
    model = camera["model"]
    r2 = x * x + y * y
    if model == "SIMPLE_PINHOLE":
        fx = fy = p[0]
        cx, cy = p[1], p[2]
        xd, yd = x, y
    elif model == "PINHOLE":
        fx, fy, cx, cy = p
        xd, yd = x, y
    elif model in ("SIMPLE_RADIAL", "RADIAL"):
        fx = fy = p[0]
        cx, cy = p[1], p[2]
        radial = 1 + p[3] * r2 + (p[4] * r2 * r2 if model == "RADIAL" else 0.0)
        xd, yd = x * radial, y * radial
    else:  # OPENCV
        fx, fy, cx, cy, k1, k2, p1, p2 = p
        radial = 1 + k1 * r2 + k2 * r2 * r2
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return fx * xd + cx, fy * yd + cy, z


def view_sets(
    u: np.ndarray, v: np.ndarray, depth: np.ndarray, masks: np.ndarray
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Equation (1) for one view: the Gaussian indices kept by each mask.

    ``depth`` is in reconstruction units. ``masks`` is K x H x W booleans.
    Pixel (col, row) covers [col, col+1) x [row, row+1): COLMAP puts the
    center of the top-left pixel at (0.5, 0.5).
    """
    _, height, width = masks.shape
    cell = mask_cell_side_px(min(height, width))
    with np.errstate(invalid="ignore"):
        col = np.floor(u)
        row = np.floor(v)
        visible = (depth > 0) & (col >= 0) & (col < width) & (row >= 0) & (row < height)
    index = np.flatnonzero(visible)
    col = col[index].astype(np.int64)
    row = row[index].astype(np.int64)
    z = depth[index]
    columns = -(-width // cell)
    cell_id = (row // cell) * columns + col // cell
    z_hat = np.full(columns * (-(-height // cell)), np.inf)
    np.minimum.at(z_hat, cell_id, z)
    front_hat = z_hat[cell_id]
    front = z <= front_hat + np.maximum(EPS_A, EPS_R * front_hat)
    sets = []
    for mask in masks:
        inside = mask[row, col]
        sets.append(index[inside & front])
    stats = {
        "cell_px": cell,
        "projected": int(index.size),
        "front": int(front.sum()),
    }
    return sets, stats


def set_extent(points: np.ndarray) -> float:
    low, high = np.percentile(points, EXTENT_PERCENTILES, axis=0)
    return float(np.max(high - low))


def group_masks(
    masks: list[dict[str, Any]], positions: np.ndarray
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    """Connected components of ``masks_same_object`` over masks of different views.

    Each mask is ``{"view": str, "gaussians": int array}``; empty ones never join.
    Returns the components (mask positions in ``masks``) and every pair that
    shares a Gaussian, with whether it passed each cut and merged.
    """
    centroid, extent = {}, {}
    for i, mask in enumerate(masks):
        ids = mask["gaussians"]
        if ids.size:
            centroid[i] = positions[ids].mean(axis=0)
            extent[i] = set_extent(positions[ids])
    parent = list(range(len(masks)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = []
    for a, b in combinations(sorted(centroid), 2):
        if masks[a]["view"] == masks[b]["view"]:
            continue
        set_a, set_b = masks[a]["gaussians"], masks[b]["gaussians"]
        common = np.intersect1d(set_a, set_b, assume_unique=True).size
        if not common:
            continue
        containment = common / min(set_a.size, set_b.size)
        jaccard = common / (set_a.size + set_b.size - common)
        distance = float(np.linalg.norm(centroid[a] - centroid[b]))
        e = min(extent[a], extent[b])
        same = masks_same_object(distance, e, containment, jaccard)
        if same:
            parent[root(a)] = root(b)
        pairs.append(
            {
                "masks": [a, b],
                "centroid_distance": round(distance, 5),
                "centroid_limit": round(centroid_threshold(e), 5),
                "containment": round(float(containment), 4),
                "jaccard": round(float(jaccard), 4),
                "close": distance <= centroid_threshold(e),
                "overlap": containment >= CONTAINMENT_MIN or jaccard >= JACCARD_MIN,
                "merged": same,
            }
        )
    components: dict[int, list[int]] = {}
    for i in centroid:
        components.setdefault(root(i), []).append(i)
    return sorted(components.values(), key=lambda c: (-len(c), c)), pairs


def camera_centers_colmap(images: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        name: -qvec_to_rotation(img["qvec"]).T @ np.asarray(img["tvec"])
        for name, img in images.items()
    }


def camera_centers_metric(
    scene_gt: dict[str, Any], names: list[str]
) -> dict[str, np.ndarray]:
    """Metric camera centers in the frame of one object seen in every view.

    ``scene_gt.json`` gives each object's model-to-camera pose in mm, so
    ``-R^T t`` is the camera center in that object's frame.
    """
    keys = {name: str(int(Path(name).stem)) for name in names}
    common = None
    for key in keys.values():
        ids = {entry["obj_id"] for entry in scene_gt.get(key, [])}
        common = ids if common is None else common & ids
    if not common:
        return {}
    obj = min(common)
    centers = {}
    for name, key in keys.items():
        entry = next(e for e in scene_gt[key] if e["obj_id"] == obj)
        rotation = np.asarray(entry["cam_R_m2c"], dtype=float).reshape(3, 3)
        translation = np.asarray(entry["cam_t_m2c"], dtype=float) / 1000.0
        centers[name] = -rotation.T @ translation
    return centers


def metres_per_unit(
    colmap: dict[str, np.ndarray], metric: dict[str, np.ndarray], scale: float
) -> float | None:
    """Median ratio of metric to splat-unit distances over all camera pairs."""
    names = sorted(set(colmap) & set(metric))
    ratios = []
    for a, b in combinations(names, 2):
        units = float(np.linalg.norm(colmap[a] - colmap[b])) * scale
        if units > 0:
            ratios.append(float(np.linalg.norm(metric[a] - metric[b])) / units)
    return float(np.median(ratios)) if ratios else None


def _latest_dataparser_transforms(splat_dir: Path) -> Path | None:
    found = sorted(
        (splat_dir / "outputs").glob("**/dataparser_transforms.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return found[0] if found else None


def focal_px(camera: dict[str, Any]) -> float:
    """Mean of fx and fy for the supported COLMAP models."""
    p = camera["params"]
    if camera["model"] in ("PINHOLE", "OPENCV"):
        return (p[0] + p[1]) / 2.0
    return float(p[0])


def associate(
    out: Path,
    data: Path,
    *,
    depth: Callable[[Path, Path], dict[str, Any]] = run_depth,
) -> dict[str, Any]:
    """Measure the unit, run equation (1) on every view and group the masks.

    Lengths are converted to the paper's units (``PAPER_METRES_PER_UNIT``)
    with the DA3 scale before any cut; the cuts themselves are unchanged.
    Raises on missing inputs or a failed depth worker.
    """
    bar = json.loads((out / "bin_evidence.json").read_text(encoding="utf-8"))[
        "quality_bar"
    ]
    model = out / "sparse" / str(bar["model"])
    cameras = read_cameras_bin(model / "cameras.bin")
    images = read_images_bin(model / "images.bin")
    splat_ev = json.loads((out / "splat_evidence.json").read_text(encoding="utf-8"))
    ply = Path(splat_ev["ply"])
    transforms_path = _latest_dataparser_transforms(out / "splat")
    if transforms_path is None:
        raise ValueError("no dataparser_transforms.json under splat/outputs")
    frame = json.loads(transforms_path.read_text(encoding="utf-8"))
    transform, scale = (
        np.asarray(frame["transform"], dtype=float),
        float(frame["scale"]),
    )
    positions = read_ply_xyz(ply)
    world = splat_to_colmap(positions, transform, scale)
    detections = json.loads(
        (out / "masks" / "detections.json").read_text(encoding="utf-8")
    )

    projected = {}
    for view in sorted(detections):
        if view in images:
            image = images[view]
            camera = cameras[image["camera_id"]]
            u, v, z = project(world, image, camera)
            projected[view] = (u, v, z * scale, camera)

    depth_dir = out / "depth"
    depth_meta = depth(out / "images", depth_dir)
    ratios = []
    for view, (u, v, z_units, camera) in projected.items():
        stored = np.load(depth_dir / f"{Path(view).stem}.npz")["depth"]
        size = (camera["width"], camera["height"])
        ratios.append(cell_ratios(u, v, z_units, stored, size, focal_px(camera)))
    unit = summarize_scale(ratios)
    unit.update(
        model=depth_meta.get("model"),
        package=depth_meta.get("package"),
        da3_views=depth_meta.get("views"),
        rule=SCALE_RULE,
        paper_metres_per_unit=PAPER_METRES_PER_UNIT,
    )
    to_paper = unit["to_paper_units"]

    masks: list[dict[str, Any]] = []
    views: dict[str, Any] = {}
    for view in sorted(detections):
        if view not in projected:
            views[view] = {"registered": False}
            continue
        u, v, z_units, _ = projected[view]
        stack = np.load(out / "masks" / f"{Path(view).stem}.npz")["masks"]
        sets, stats = view_sets(u, v, z_units * to_paper, stack)
        kept = [d for d in detections[view] if d.get("kept")]
        for det in kept:
            ids = sets[det["mask_index"]]
            masks.append(
                {
                    "view": view,
                    "mask_index": det["mask_index"],
                    "phrase": det.get("phrase"),
                    "gaussians": ids,
                }
            )
        stats["registered"] = True
        stats["masks"] = len(kept)
        stats["kept_per_mask"] = [int(sets[d["mask_index"]].size) for d in kept]
        views[view] = stats

    components, pairs = group_masks(masks, positions * to_paper)
    objects = []
    for number, members in enumerate(components):
        union = np.unique(np.concatenate([masks[i]["gaussians"] for i in members]))
        objects.append(
            {
                "object": number,
                "masks": [
                    {k: masks[i][k] for k in ("view", "mask_index", "phrase")}
                    for i in members
                ],
                "views": len({masks[i]["view"] for i in members}),
                "gaussians": int(union.size),
                "centroid": [round(float(c), 5) for c in positions[union].mean(axis=0)],
                "ids": union,
            }
        )

    scene_gt_path = data / "scenes" / SCENE / "scene_gt.json"
    try:
        scene_gt = json.loads(scene_gt_path.read_text(encoding="utf-8"))
        metric = camera_centers_metric(scene_gt, sorted(images))
    except (OSError, ValueError, KeyError, StopIteration):
        metric = {}
    m_per_unit = metres_per_unit(camera_centers_colmap(images), metric, scale)
    return {
        "positions": len(positions),
        "ply": str(ply),
        "colmap_model": str(model),
        "dataparser_transforms": str(transforms_path),
        "scale": scale,
        "unit": unit,
        "metres_per_unit_gt_cameras": m_per_unit,
        "views": views,
        "masks": masks,
        "pairs": pairs,
        "merged_pairs": [p for p in pairs if p["merged"]],
        "objects": objects,
    }


def pair_outcomes(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Why overlapping pairs of masks did or did not merge.

    ``too_far`` pairs pass the overlap cut and fail only the centroid one;
    their distances, against their limits, show whether the absolute
    0.10-unit gate is what keeps views of one object apart.
    """
    too_far = [p for p in pairs if p["overlap"] and not p["close"]]
    distances = sorted(p["centroid_distance"] for p in too_far)
    return {
        "total": len(pairs),
        "merged": sum(p["merged"] for p in pairs),
        "too_far": len(too_far),
        "too_little_overlap": sum(p["close"] and not p["overlap"] for p in pairs),
        "neither": sum(not p["close"] and not p["overlap"] for p in pairs),
        "too_far_distance_units": (
            {
                "min": distances[0],
                "median": distances[len(distances) // 2],
                "max": distances[-1],
            }
            if distances
            else None
        ),
        "too_far_limit_units_max": max(
            (p["centroid_limit"] for p in too_far), default=None
        ),
    }


def evidence_from(result: dict[str, Any]) -> dict[str, Any]:
    """The JSON-safe summary that the artifact carries (no index arrays)."""
    m = result["unit"]["metres_per_unit"]
    objects = [{k: v for k, v in o.items() if k != "ids"} for o in result["objects"]]
    empty = [
        {"view": x["view"], "mask_index": x["mask_index"]}
        for x in result["masks"]
        if not x["gaussians"].size
    ]
    multi = [o for o in objects if o["views"] > 1]
    return {
        "scene": SCENE,
        "rule": {
            "equation_1": {
                "eps_a": EPS_A,
                "eps_r": EPS_R,
                "cell": "min(16, round(short_side/192))",
            },
            "grouping": {
                "centroid": f"<= min({CENTROID_ABS}, {CENTROID_EXTENT_SCALE} e)",
                "containment_min": CONTAINMENT_MIN,
                "jaccard_min": JACCARD_MIN,
                "e": EXTENT_RULE,
                "pairs": "masks of different views only; objects are connected components",
            },
            "kappa": 1,
            "units": "paper units: splat lengths x unit.to_paper_units",
        },
        "inputs": {
            k: result[k]
            for k in (
                "ply",
                "colmap_model",
                "dataparser_transforms",
                "positions",
                "scale",
            )
        },
        "unit": result["unit"],
        "metres_per_unit_gt_cameras": result["metres_per_unit_gt_cameras"],
        "eps_a_m": round(EPS_A * PAPER_METRES_PER_UNIT, 4),
        "centroid_abs_m": round(CENTROID_ABS * PAPER_METRES_PER_UNIT, 4),
        "splat_unit_m": m,
        "views": result["views"],
        "masks": len(result["masks"]),
        "masks_without_gaussians": empty,
        "merged_pairs": result["merged_pairs"],
        "overlapping_pairs": pair_outcomes(result["pairs"]),
        "objects": len(objects),
        "objects_seen_in_2_or_more_views": len(multi),
        "gt_objects": GT_OBJECTS,
        "object_list": objects,
    }


def write_sets(path: Path, result: dict[str, Any]) -> None:
    """Per-object Gaussian indices for phase 4 (``object_<n>`` int32 arrays)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{
            f"object_{o['object']}": o["ids"].astype(np.int32)
            for o in result["objects"]
        },
    )


def run_bin_association(
    out: Path | str,
    data: Path | str = DATA_DIR,
    *,
    depth: Callable[[Path, Path], dict[str, Any]] = run_depth,
) -> int:
    """Measure the unit, associate and group. Return 0 when all 13 views were used and objects exist.

    The object count is reported against the 20 GT objects, not gated: the
    paper publishes no count tolerance.
    """
    out, data = Path(out), Path(data)
    ledger_path = out / "ledger.json"
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        ledger = {"entries": []}
    entries = [e for e in ledger.get("entries", []) if e.get("item") != ITEM]
    entry: dict[str, Any] = {"item": ITEM}
    evidence: dict[str, Any] = {"scene": SCENE}
    try:
        result = associate(out, data, depth=depth)
        evidence = evidence_from(result)
        write_sets(out / "association" / "objects.npz", result)
        used = sum(1 for v in result["views"].values() if v.get("registered"))
        if used == VIEWS and result["objects"]:
            entry.update(
                decision="accept",
                use_gpu=True,
                splat_unit_m=round(evidence["splat_unit_m"], 4),
                objects=evidence["objects"],
                multi_view_objects=evidence["objects_seen_in_2_or_more_views"],
                gt_objects=GT_OBJECTS,
                masks=evidence["masks"],
            )
        else:
            entry.update(
                decision="reject",
                reason=f"{used} of {VIEWS} views posed; {len(result['objects'])} objects",
            )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        subprocess.TimeoutExpired,
    ) as exc:
        entry.update(decision="reject", reason=f"{type(exc).__name__}: {exc}"[-800:])
    entries.append(entry)
    ledger.update(scene_name=f"{SOURCE} {SCENE}", source=SOURCE, entries=entries)
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    (out / "association_evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    decisions = {e.get("item"): e.get("decision") for e in entries}
    return 0 if all(decisions.get(item) == "accept" for item in REQUIRED) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path(DATA_DIR))
    args = parser.parse_args(argv)
    return run_bin_association(args.out, args.data)


if __name__ == "__main__":
    sys.exit(main())
