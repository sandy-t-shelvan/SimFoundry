# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cut 2 phase 4: SF-8 part 2 (per-object ICP) and the first bin score.

ICP. Each SAM 3D mesh (``sam3d/object_<n>.glb``) is aligned to its object's
Gaussian centers (``association/objects.npz``) by trimmed similarity ICP
(:func:`observed_icp`: Umeyama on the closest 70% of pairs, each Gaussian
paired with its nearest mesh point), starting from the SAM 3D pose. That pose maps the mesh into the chosen view's camera in
PyTorch3D's convention (+x left, +y up) and metres, so it is composed with
the flip to OpenCV's camera, the DA3 metres per splat unit
(``association_evidence.json``), the view's COLMAP pose and the nerfstudio
dataparser transform, giving one similarity from mesh to splat. The GLB's
axes relative to that pose's local frame are not documented, so three
frame fixes (none, and a quarter turn about x either way) are tried under
both camera conventions (``CAMERA_FLIPS``), each also moved onto the
Gaussians' centre (:func:`centred_starts`), plus quarter-turn seeds at
SAM 3D's size; every fit's scale stays within
``SCALE_BOUND`` of SAM 3D's own; the lowest two-way distance
(:func:`fit_score`) wins and is recorded, with each pose start's score
before and after ICP. The silhouette is the mean Jaccard index of each object mask
and the placed mesh's samples projected into that view, dilated by
``SILHOUETTE_DILATE_PX``; that rendering is ours. Residuals go through
``ours_iteration_2`` (:func:`sceneagent.alignment.accept_placement`:
silhouette and p90 in metres), whose decision is recorded and does not stop
the score; iteration 1's decision is recorded beside it. Physics export
takes only the objects this gate accepts.

Score. The reconstruction is mapped into GraspClutter6D's reference frame
(image 1's camera, metres) by one similarity for the whole scene: Umeyama
from the 13 COLMAP camera centers to the same cameras' metric centers,
which ``scene_gt.json`` gives through any object seen in every view. No
per-object refit. Each placed mesh is sampled to 10,000 surface points and
scored by :func:`sceneagent.bin_benchmark.score_bin` (Hungarian on Chamfer,
F1 at 1 cm over the 20 objects).

    python -m sceneagent.bin_place --out /workspace/sf4-smoke/bin
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from sceneagent.alignment import (
    OURS_GATE_LABEL,
    PLACEMENT_GATE_LABEL,
    accept_alignment,
    accept_placement,
)
from sceneagent.bin_association import (
    _latest_dataparser_transforms,
    camera_centers_colmap,
    project,
    qvec_to_rotation,
    read_cameras_bin,
    read_images_bin,
    read_ply_xyz,
    splat_to_colmap,
)
from sceneagent.bin_benchmark import (
    POINTS_PER_OBJECT,
    image_keys,
    reference_clouds,
    score_bin,
)
from sceneagent.bin_run import DATA_DIR, SCENE, load_scene
from sceneagent.similarity_icp import (
    TRIM_KEEP,
    apply_similarity,
    initial_similarity,
    measure_residuals,
    rotation_seeds,
    silhouette_agreement,
    umeyama,
)

ICP_ITEM = "ICP residuals"
SCORE_ITEM = "bin score"
REQUIRED = ("SAM 3D meshes", ICP_ITEM, SCORE_ITEM)
ICP_SOURCE_POINTS = 5000
SILHOUETTE_POINTS = 20000
SILHOUETTE_DILATE_PX = 3
REFERENCE_IMAGE = 1
GATE = PLACEMENT_GATE_LABEL
# PyTorch3D camera (+x left, +y up, +z forward) to OpenCV (+x right, +y down).
P3D_TO_OPENCV = np.diag([-1.0, -1.0, 1.0])
# Which camera convention the SAM 3D pose is in is not documented either: the
# pipeline's names say PyTorch3D, but the second bin place run (42761784197)
# started every object 1.4-5 splat units from its Gaussians with that flip,
# the size of a mirror image across the view centre. Both are tried.
CAMERA_FLIPS = {"p3d": P3D_TO_OPENCV, "opencv": np.eye(3)}
# A fit's scale stays within this factor of SAM 3D's own scale. Ours: the
# Gaussian sets carry stray points, and unbounded fits grew to 1.5 m there.
SCALE_BOUND = 2.0
FRAME_FIXES = {
    "none": np.eye(3),
    "x+90": np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]]),
    "x-90": np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]]),
}
SILHOUETTE_RULE = (
    "ours: object masks against the placed mesh's surface samples projected "
    f"into the same view and dilated by {SILHOUETTE_DILATE_PX} px"
)
ALIGNMENT_RULE = (
    "one similarity for the scene: Umeyama from the 13 COLMAP camera centers "
    "(splat frame) to the same cameras' metric centers in image 1's camera frame"
)


def quaternion_wxyz_to_matrix(q: list[float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def pose_to_splat(
    pose: dict[str, Any],
    image: dict[str, Any],
    transform: np.ndarray,
    dataparser_scale: float,
    metres_per_unit: float,
    frame_fix: np.ndarray,
    camera_flip: np.ndarray = P3D_TO_OPENCV,
) -> tuple[float, np.ndarray, np.ndarray]:
    """The similarity (s, R, t) from GLB coordinates to the splat frame.

    ``p_splat = S * (T_R @ X + T_t)`` with ``X = R_c^T (x_cv / k - t_c)``,
    ``x_cv = A (s R_q F p + t)`` in metres and ``k = S * metres_per_unit``
    metres per COLMAP unit.
    """
    s = float(np.mean(pose["scale"]))
    r_q = quaternion_wxyz_to_matrix(pose["rotation"])
    t_q = np.asarray(pose["translation"], dtype=float)
    r_c = qvec_to_rotation(image["qvec"])
    t_c = np.asarray(image["tvec"], dtype=float)
    t_r, t_t = transform[:, :3], transform[:, 3]
    k = dataparser_scale * metres_per_unit
    rotation = t_r @ r_c.T @ camera_flip @ r_q @ frame_fix
    scale = s / metres_per_unit
    translation = dataparser_scale * (t_r @ r_c.T @ (camera_flip @ t_q / k - t_c) + t_t)
    return scale, rotation, translation


def sample_mesh(path: Path, n: int, seed: int = 0) -> np.ndarray:
    import trimesh

    mesh = trimesh.load(path, force="mesh", process=False)
    points, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(points, dtype=float)


def _nearest_to(points: np.ndarray, cloud: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    distance, index = cKDTree(cloud).query(points, k=1)
    return np.asarray(distance, dtype=float), np.asarray(index, dtype=np.int64)


def observed_icp(
    source: np.ndarray,
    target: np.ndarray,
    start: tuple[float, np.ndarray, np.ndarray],
    iterations: int = 30,
    keep: float = TRIM_KEEP,
    scale_range: tuple[float, float] | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Trimmed similarity ICP that pairs each observed point with the mesh.

    The Gaussians cover only the sides of an object the views saw, so every
    target point should lie on the mesh but most mesh points have no target
    nearby. Pairing mesh to target (``trimmed_similarity_icp``'s direction)
    then pulls the whole mesh onto the seen side and rewards shrinking it:
    the first bin run collapsed two objects to scale 0 and blew others up.
    Here each target point takes its nearest mesh point, the closest
    ``keep`` fraction of pairs is kept, and Umeyama refits. That direction
    is ours; the paper does not state one.
    """
    scale, rotation, translation = start
    count = max(3, round(keep * len(target)))
    for _ in range(int(iterations)):
        moved = apply_similarity(source, scale, rotation, translation)
        distance, index = _nearest_to(target, moved)
        chosen = np.argsort(distance)[:count]
        paired, observed = source[index[chosen]], target[chosen]
        scale, rotation, translation = umeyama(paired, observed)
        if scale_range is not None and not scale_range[0] <= scale <= scale_range[1]:
            # Hold the scale at the bound and refit the translation for it.
            scale = float(np.clip(scale, *scale_range))
            translation = observed.mean(axis=0) - scale * (
                rotation @ paired.mean(axis=0)
            )
    return float(scale), rotation, translation


def fit_score(
    source: np.ndarray,
    target: np.ndarray,
    similarity: tuple[float, np.ndarray, np.ndarray],
) -> dict[str, float]:
    """Both directions, so neither a shrunken nor a swollen mesh scores well.

    ``observed`` is the median distance from each Gaussian to the mesh;
    ``visible`` is the median over the mesh points nearest the Gaussians
    (the closest ``TRIM_KEEP`` fraction, since the far side was never seen).
    """
    moved = apply_similarity(source, *similarity)
    to_mesh, _ = _nearest_to(target, moved)
    to_target, _ = _nearest_to(moved, target)
    visible = np.sort(to_target)[: max(3, round(TRIM_KEEP * len(to_target)))]
    observed = float(np.median(to_mesh))
    return {
        "observed": observed,
        "visible": float(np.median(visible)),
        "total": observed + float(np.median(visible)),
    }


def residual_input(
    source: np.ndarray,
    target: np.ndarray,
    similarity: tuple[float, np.ndarray, np.ndarray],
) -> dict[str, Any]:
    """The fields ``measure_residuals`` reads, as ``trimmed_similarity_icp`` returns them."""
    scale, rotation, translation = similarity
    moved = apply_similarity(source, scale, rotation, translation)
    distance, index = _nearest_to(moved, target)
    chosen = np.argsort(distance)[: max(3, round(TRIM_KEEP * len(source)))]
    return {
        "scale": scale,
        "rotation": rotation,
        "translation": translation,
        "distance": distance,
        "inlier_index": chosen,
        "target_index": index,
        "inlier_distance": distance[chosen],
    }


def centred_starts(
    pose_starts: dict[str, tuple[float, np.ndarray, np.ndarray]],
    source: np.ndarray,
    target: np.ndarray,
) -> dict[str, tuple[float, np.ndarray, np.ndarray]]:
    """Each SAM 3D start's rotation and scale, moved onto the Gaussians' centre.

    SAM 3D places the object with its own camera (MoGe depth and estimated
    intrinsics), which the third bin place run (42762718646) found a median
    3.7 splat units off; its rotation and size do not depend on that
    camera's intrinsics. So this start keeps both and takes the
    translation from the object's own Gaussians. Ours.
    """
    centre = target.mean(axis=0)
    return {
        name.replace("sam3d_pose/", "sam3d_centred/"): (
            scale,
            rotation,
            centre - scale * (rotation @ source.mean(axis=0)),
        )
        for name, (scale, rotation, _) in pose_starts.items()
    }


def seed_starts(
    source: np.ndarray, target: np.ndarray, scale: float | None = None
) -> dict[str, tuple[float, np.ndarray, np.ndarray]]:
    """Quarter-turn seeds centred on the target, at ``scale`` or the RMS size ratio."""
    if scale is None:
        scale, _, _ = initial_similarity(source, target)
    starts = {}
    for k, rotation in enumerate(rotation_seeds()):
        translation = target.mean(axis=0) - scale * (rotation @ source.mean(axis=0))
        starts[f"seed/{k}"] = (scale, rotation, translation)
    return starts


def best_icp(
    source: np.ndarray,
    target: np.ndarray,
    starts: dict[str, tuple[float, np.ndarray, np.ndarray]],
    prior_scale: float | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """ICP from each pose start and each seed; keep the lowest two-way score.

    With ``prior_scale`` (SAM 3D's own size in splat units) the seeds start
    at that size and every fit stays within ``SCALE_BOUND`` of it. Returns
    the winning start's name, the residual input for ``measure_residuals``
    and, per pose start, its score before and after ICP.
    """
    bounds = None
    if prior_scale is not None:
        bounds = (prior_scale / SCALE_BOUND, prior_scale * SCALE_BOUND)
    tried: dict[str, Any] = {}
    best: tuple[str, tuple[float, np.ndarray, np.ndarray], float] | None = None
    for name, start in {**starts, **seed_starts(source, target, prior_scale)}.items():
        try:
            fitted = observed_icp(source, target, start, scale_range=bounds)
        except ValueError:  # every pair on one mesh point: Umeyama has no spread
            continue
        score = fit_score(source, target, fitted)
        if not name.startswith("seed/"):
            tried[name] = {
                "before": round(fit_score(source, target, start)["total"], 5),
                "after": round(score["total"], 5),
            }
        if best is None or score["total"] < best[2]:
            best = (name, fitted, score["total"])
    if best is None:
        raise RuntimeError("ICP found no fit from any start")
    tried["best_total"] = round(best[2], 5)
    return best[0], residual_input(source, target, best[1]), tried


def render_mask(
    points_colmap: np.ndarray,
    image: dict[str, Any],
    camera: dict[str, Any],
    dilate: int,
) -> np.ndarray:
    """Pixels hit by the projected points, dilated by ``dilate`` px."""
    from scipy.ndimage import binary_dilation

    u, v, z = project(points_colmap, image, camera)
    width, height = camera["width"], camera["height"]
    with np.errstate(invalid="ignore"):
        ok = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    mask = np.zeros((height, width), dtype=bool)
    mask[np.floor(v[ok]).astype(int), np.floor(u[ok]).astype(int)] = True
    if dilate:
        mask = binary_dilation(mask, iterations=dilate)
    return mask


def metric_camera_centers(
    scene_gt: dict[str, Any], names: list[str]
) -> dict[str, np.ndarray]:
    """Camera centers in image 1's camera frame, metres, via one object in every view."""
    keys = {name: str(int(Path(name).stem)) for name in names}
    common = None
    for key in list(keys.values()) + [str(REFERENCE_IMAGE)]:
        ids = {entry["obj_id"] for entry in scene_gt.get(key, [])}
        common = ids if common is None else common & ids
    if not common:
        raise ValueError("no object is in every view and in the reference image")
    obj = min(common)

    def pose(key: str) -> tuple[np.ndarray, np.ndarray]:
        entry = next(e for e in scene_gt[key] if e["obj_id"] == obj)
        rotation = np.asarray(entry["cam_R_m2c"], dtype=float).reshape(3, 3)
        return rotation, np.asarray(entry["cam_t_m2c"], dtype=float) / 1000.0

    r1, t1 = pose(str(REFERENCE_IMAGE))
    centers = {}
    for name, key in keys.items():
        rotation, translation = pose(key)
        in_object = -rotation.T @ translation
        centers[name] = r1 @ in_object + t1
    return centers


def scene_similarity(
    splat_centers: dict[str, np.ndarray], metric_centers: dict[str, np.ndarray]
) -> tuple[tuple[float, np.ndarray, np.ndarray], dict[str, Any]]:
    names = sorted(set(splat_centers) & set(metric_centers))
    if len(names) < 3:
        raise ValueError(f"{len(names)} cameras in common; Umeyama needs 3")
    src = np.stack([splat_centers[n] for n in names])
    dst = np.stack([metric_centers[n] for n in names])
    scale, rotation, translation = umeyama(src, dst)
    residual = np.linalg.norm(
        apply_similarity(src, scale, rotation, translation) - dst, axis=1
    )
    return (scale, rotation, translation), {
        "rule": ALIGNMENT_RULE,
        "cameras": len(names),
        "camera_rms_m": float(np.sqrt(np.mean(residual**2))),
        "camera_max_m": float(residual.max()),
        "scale": float(scale),
    }


def _load(out: Path) -> dict[str, Any]:
    bar = json.loads((out / "bin_evidence.json").read_text(encoding="utf-8"))[
        "quality_bar"
    ]
    model = out / "sparse" / str(bar["model"])
    frame_path = _latest_dataparser_transforms(out / "splat")
    if frame_path is None:
        raise ValueError("no dataparser_transforms.json under splat/outputs")
    frame = json.loads(frame_path.read_text(encoding="utf-8"))
    splat = json.loads((out / "splat_evidence.json").read_text(encoding="utf-8"))
    association = json.loads(
        (out / "association_evidence.json").read_text(encoding="utf-8")
    )
    sam3d = json.loads((out / "sam3d_evidence.json").read_text(encoding="utf-8"))
    return {
        "cameras": read_cameras_bin(model / "cameras.bin"),
        "images": read_images_bin(model / "images.bin"),
        "transform": np.asarray(frame["transform"], dtype=float),
        "scale": float(frame["scale"]),
        "positions": read_ply_xyz(Path(splat["ply"])),
        "association": association,
        "sam3d": sam3d,
        "sets": np.load(out / "association" / "objects.npz"),
        "detections": json.loads(
            (out / "masks" / "detections.json").read_text(encoding="utf-8")
        ),
    }


def measure_fit(
    out: Path,
    inputs: dict[str, Any],
    number: int,
    glb: Path,
    target: np.ndarray,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Silhouette, residuals and gate for one placed mesh."""
    images, cameras = inputs["images"], inputs["cameras"]
    masks = {o["object"]: o["masks"] for o in inputs["association"]["object_list"]}
    silhouette_cloud = apply_similarity(
        sample_mesh(glb, SILHOUETTE_POINTS, seed=2),
        result["scale"],
        result["rotation"],
        result["translation"],
    )
    world = splat_to_colmap(silhouette_cloud, inputs["transform"], inputs["scale"])
    rendered, observed = [], []
    for m in masks.get(number, []):
        image = images[m["view"]]
        stack = np.load(out / "masks" / f"{Path(m['view']).stem}.npz")["masks"]
        rendered.append(
            render_mask(world, image, cameras[image["camera_id"]], SILHOUETTE_DILATE_PX)
        )
        observed.append(stack[m["mask_index"]])
    silhouette = silhouette_agreement(rendered, observed) if rendered else None
    residuals = measure_residuals(
        result, target, None if silhouette is None else silhouette["silhouette"]
    )
    # Splat units to metres with association's DA3 unit, so the p90 limit
    # means the same on a video as on the bin.
    m_per_unit = float(inputs["association"]["unit"]["metres_per_unit"])
    residuals["p90_m"] = residuals["p90"] * m_per_unit
    gate = accept_placement(residuals)
    return {
        "residuals": {
            k: residuals[k]
            for k in (
                "normalized_median",
                "p90",
                "p90_m",
                "median",
                "coverage",
                "silhouette",
            )
        },
        "silhouette_views": None if silhouette is None else silhouette["views"],
        "gate": gate["decision"],
        "gate_failed": gate.get("failed"),
        # What iteration 1 said, kept for comparison.
        "gate_iteration_1": accept_alignment(residuals, OURS_GATE_LABEL)["decision"],
    }


def object_depth_m(
    inputs: dict[str, Any], target: np.ndarray, view: str
) -> float | None:
    """Median camera depth of the object's Gaussians in ``view``, in DA3 metres."""
    image = inputs["images"][view]
    camera = inputs["cameras"][image["camera_id"]]
    world = splat_to_colmap(target, inputs["transform"], inputs["scale"])
    u, v, z = project(world, image, camera)
    with np.errstate(invalid="ignore"):
        ok = (
            (z > 0)
            & (u >= 0)
            & (u < camera["width"])
            & (v >= 0)
            & (v < camera["height"])
        )
    if not ok.any():
        return None
    metres_per_colmap = inputs["scale"] * float(
        inputs["association"]["unit"]["metres_per_unit"]
    )
    return float(np.median(z[ok])) * metres_per_colmap


def depth_ratio(
    inputs: dict[str, Any], target: np.ndarray, view: str, pose: dict[str, Any]
) -> float | None:
    """Our depth of the object over SAM 3D's own (MoGe) depth inside its mask.

    SAM 3D places the object in MoGe's metric frame; if MoGe's scale differs
    from ours, translation and size are both off by this factor. Ours.
    """
    theirs = pose.get("mask_depth_m")
    if not theirs:
        return None
    ours = object_depth_m(inputs, target, view)
    return None if ours is None else ours / float(theirs)


def rescale_pose(pose: dict[str, Any], ratio: float) -> dict[str, Any]:
    """SAM 3D's pose with its translation and size multiplied by ``ratio``."""
    return {
        **pose,
        "translation": [float(t) * ratio for t in pose["translation"]],
        "scale": [float(c) * ratio for c in pose["scale"]],
    }


def place_one(
    out: Path,
    inputs: dict[str, Any],
    number: int,
    pose: dict[str, Any],
    view: str,
    glb: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """ICP one SAM 3D mesh onto its Gaussians; return its 10,000-point cloud and record."""
    transform, dp_scale = inputs["transform"], inputs["scale"]
    m_per_unit = float(inputs["association"]["unit"]["metres_per_unit"])
    target = inputs["positions"][inputs["sets"][f"object_{number}"]]
    source = sample_mesh(glb, ICP_SOURCE_POINTS)
    poses = {"sam3d_pose": pose}
    ratio = depth_ratio(inputs, target, view, pose)
    if ratio is not None:
        poses["sam3d_rescaled"] = rescale_pose(pose, ratio)
    starts = {
        f"{kind}/{camera}/{fix}": pose_to_splat(
            p, inputs["images"][view], transform, dp_scale, m_per_unit, matrix, flip
        )
        for kind, p in poses.items()
        for camera, flip in CAMERA_FLIPS.items()
        for fix, matrix in FRAME_FIXES.items()
    }
    starts.update(centred_starts(starts, source, target))
    # The size bound follows SAM 3D's size in our metres when it can be measured.
    prior = next(s for name, s in starts.items() if name.startswith(list(poses)[-1]))[0]
    start_name, result, tried = best_icp(source, target, starts, prior)
    dense = sample_mesh(glb, POINTS_PER_OBJECT, seed=1)
    cloud = apply_similarity(
        dense, result["scale"], result["rotation"], result["translation"]
    )
    record: dict[str, Any] = {
        "object": number,
        "view": view,
        "glb": str(glb),
        "placed": True,
        "start": start_name,
        "starts_tried": tried,
        "pose_scale": round(prior, 5),
        "scale": result["scale"],
        "similarity": {
            "scale": float(result["scale"]),
            "rotation": np.asarray(result["rotation"]).tolist(),
            "translation": np.asarray(result["translation"]).tolist(),
        },
        "fit_total": tried["best_total"],
        "pose_depth_ratio": ratio,
    }
    record.update(measure_fit(out, inputs, number, glb, target, result))
    return cloud, record


def place_objects(out: Path, inputs: dict[str, Any]) -> dict[str, Any]:
    """ICP every meshed object; return placed clouds (splat frame) and per-object records."""
    placed: dict[str, np.ndarray] = {}
    records = []
    for obj in inputs["sam3d"].get("objects", []):
        number = obj["object"]
        if not obj.get("ok"):
            records.append(
                {
                    "object": number,
                    "view": obj["view"],
                    "placed": False,
                    "reason": "no SAM 3D mesh",
                }
            )
            continue
        glb = out / "sam3d" / f"object_{number}.glb"
        cloud, record = place_one(out, inputs, number, obj["pose"], obj["view"], glb)
        placed[f"object_{number}"] = cloud
        records.append(record)
    return {"placed": placed, "objects": records}


def score_scene(
    data: Path, inputs: dict[str, Any], placed: dict[str, np.ndarray], tag: str = ""
) -> dict[str, Any]:
    """One camera-based similarity for the scene, then ``score_bin`` against the 20 objects."""
    scene_dir = data / "scenes" / SCENE
    scene_gt = json.loads((scene_dir / "scene_gt.json").read_text(encoding="utf-8"))
    if str(REFERENCE_IMAGE) not in image_keys(scene_gt):
        raise ValueError(f"image {REFERENCE_IMAGE} is not in scene_gt.json")
    metric = metric_camera_centers(scene_gt, sorted(inputs["images"]))
    splat_centers = {
        name: inputs["scale"]
        * (inputs["transform"][:, :3] @ c + inputs["transform"][:, 3])
        for name, c in camera_centers_colmap(inputs["images"]).items()
    }
    similarity, alignment = scene_similarity(splat_centers, metric)
    _, reference = reference_clouds(scene_dir, data / "models_eval", REFERENCE_IMAGE)
    score = score_bin(placed, reference, similarity)
    score["alignment"] = alignment
    score["matcher"] = "Hungarian on symmetric Chamfer"
    score["delivered_objects"] = len(placed)
    score["reference_objects"] = len(reference)
    score["points_per_delivered"] = POINTS_PER_OBJECT
    score["file_name"] = (
        f"bin_{SCENE}_score_pts{POINTS_PER_OBJECT}_align-cameras-umeyama"
        f"_match-hungarian-chamfer{tag}.json"
    )
    return score


def run_bin_place(out: Path | str, data: Path | str = DATA_DIR) -> int:
    """ICP, place and score. Return 0 when both ledger items are written as accepted.

    ICP is accepted when every meshed object was aligned and measured; each
    object's gate decision is recorded beside it. The score is accepted when
    the score file is written; its numbers are reported, not gated.
    """
    out, data = Path(out), Path(data)
    scene = load_scene(out)
    ledger_path = out / "ledger.json"
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        ledger = {"entries": []}
    entries = [
        e
        for e in ledger.get("entries", [])
        if e.get("item") not in (ICP_ITEM, SCORE_ITEM)
    ]
    icp_entry: dict[str, Any] = {"item": ICP_ITEM}
    score_entry: dict[str, Any] = {"item": SCORE_ITEM}
    evidence: dict[str, Any] = {"scene": scene.name}
    try:
        inputs = _load(out)
        placement = place_objects(out, inputs)
        evidence["icp"] = {
            "trim_keep": TRIM_KEEP,
            "source_points": ICP_SOURCE_POINTS,
            "frame_fixes": list(FRAME_FIXES),
            "silhouette_rule": SILHOUETTE_RULE,
            "gate": GATE,
            "objects": placement["objects"],
        }
        placed = [o for o in placement["objects"] if o.get("placed")]
        if not placed:
            raise ValueError("no object was placed")
        icp_entry.update(
            decision="accept",
            use_gpu=False,
            placed=len(placed),
            gate_accepted=sum(o["gate"] == "accept" for o in placed),
        )
        if scene.ground_truth:
            score = score_scene(data, inputs, placement["placed"])
            evidence["score"] = score
            (out / score["file_name"]).write_text(
                json.dumps(score, indent=2) + "\n", encoding="utf-8"
            )
            score_entry.update(
                decision="accept",
                use_gpu=False,
                f1_mean=round(score["f1_mean"], 4),
                chamfer_mean_matched_m=score["chamfer_mean_matched_m"],
                delivered=len(placement["placed"]),
                reference=score["reference_objects"],
            )
        else:
            # No reference scans: nothing to score. Placement stands as is,
            # and no score entry is written for vast_smoke to judge.
            evidence["score"] = {"skipped": "no ground truth for this scene"}
            score_entry = {}
    except (OSError, ValueError, KeyError, RuntimeError, StopIteration) as exc:
        reason = f"{type(exc).__name__}: {exc}"[-800:]
        for entry in (icp_entry, score_entry):
            if "decision" not in entry:
                entry.update(decision="reject", reason=reason)
    entries += [e for e in (icp_entry, score_entry) if e]
    ledger.update(scene_name=scene.ledger_name, source=scene.source, entries=entries)
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    (out / "bin_score.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    decisions = {e.get("item"): e.get("decision") for e in entries}
    required = REQUIRED if scene.ground_truth else REQUIRED[:-1]
    return 0 if all(decisions.get(item) == "accept" for item in required) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path(DATA_DIR))
    args = parser.parse_args(argv)
    return run_bin_place(args.out, args.data)


if __name__ == "__main__":
    sys.exit(main())
