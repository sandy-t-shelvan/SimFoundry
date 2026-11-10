# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cut 2 phase 4: SF-8 part 2 (per-object ICP) and the first bin score.

ICP. Each SAM 3D mesh (``sam3d/object_<n>.glb``) is aligned to its object's
Gaussian centers (``association/objects.npz``) with
:func:`sceneagent.similarity_icp.trimmed_similarity_icp`, starting from the
SAM 3D pose. That pose maps the mesh into the chosen view's camera in
PyTorch3D's convention (+x left, +y up) and metres, so it is composed with
the flip to OpenCV's camera, the DA3 metres per splat unit
(``association_evidence.json``), the view's COLMAP pose and the nerfstudio
dataparser transform, giving one similarity from mesh to splat. The GLB's
axes relative to that pose's local frame are not documented, so three
frame fixes (none, and a quarter turn about x either way) are tried, plus
the plain quarter-turn seeds with no pose; the best trimmed median wins and
is recorded. The silhouette is the mean Jaccard index of each object mask
and the placed mesh's samples projected into that view, dilated by
``SILHOUETTE_DILATE_PX``; that rendering is ours. Residuals go through the
unchanged ``ours_iteration_1`` gate, whose decision is recorded and does not
stop the score.

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

from sceneagent.alignment import accept_alignment
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
from sceneagent.bin_run import DATA_DIR, SCENE, SOURCE
from sceneagent.similarity_icp import (
    TRIM_KEEP,
    apply_similarity,
    measure_residuals,
    silhouette_agreement,
    trimmed_similarity_icp,
    umeyama,
)

ICP_ITEM = "ICP residuals"
SCORE_ITEM = "bin score"
REQUIRED = ("SAM 3D meshes", ICP_ITEM, SCORE_ITEM)
ICP_SOURCE_POINTS = 5000
SILHOUETTE_POINTS = 20000
SILHOUETTE_DILATE_PX = 3
REFERENCE_IMAGE = 1
GATE = "ours_iteration_1"
# PyTorch3D camera (+x left, +y up, +z forward) to OpenCV (+x right, +y down).
P3D_TO_OPENCV = np.diag([-1.0, -1.0, 1.0])
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
    rotation = t_r @ r_c.T @ P3D_TO_OPENCV @ r_q @ frame_fix
    scale = s / metres_per_unit
    translation = dataparser_scale * (
        t_r @ r_c.T @ (P3D_TO_OPENCV @ t_q / k - t_c) + t_t
    )
    return scale, rotation, translation


def sample_mesh(path: Path, n: int, seed: int = 0) -> np.ndarray:
    import trimesh

    mesh = trimesh.load(path, force="mesh", process=False)
    points, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(points, dtype=float)


def best_icp(
    source: np.ndarray,
    target: np.ndarray,
    starts: dict[str, tuple[float, np.ndarray, np.ndarray]],
) -> tuple[str, dict[str, Any], dict[str, float]]:
    """Run ICP from each start and from the pose-free seeds; keep the lowest trimmed median."""
    tried: dict[str, float] = {}
    best: tuple[str, dict[str, Any]] | None = None
    runs = [(name, start) for name, start in starts.items()] + [("seeds", None)]
    for name, start in runs:
        if start is None:
            result = trimmed_similarity_icp(source, target)
        else:
            scale, rotation, translation = start
            result = trimmed_similarity_icp(
                source,
                target,
                initial_rotation=rotation,
                initial_scale=scale,
                initial_translation=translation,
            )
        score = float(np.median(result["inlier_distance"]))
        tried[name] = score
        if best is None or score < float(np.median(best[1]["inlier_distance"])):
            best = (name, result)
    assert best is not None
    return best[0], best[1], tried


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


def place_objects(out: Path, inputs: dict[str, Any]) -> dict[str, Any]:
    """ICP every meshed object; return placed clouds (splat frame) and per-object records."""
    images, cameras = inputs["images"], inputs["cameras"]
    transform, dp_scale = inputs["transform"], inputs["scale"]
    m_per_unit = float(inputs["association"]["unit"]["metres_per_unit"])
    masks_by_object = {
        o["object"]: o["masks"] for o in inputs["association"]["object_list"]
    }
    placed: dict[str, np.ndarray] = {}
    records = []
    for obj in inputs["sam3d"].get("objects", []):
        number = obj["object"]
        record: dict[str, Any] = {"object": number, "view": obj["view"]}
        records.append(record)
        if not obj.get("ok"):
            record.update(placed=False, reason="no SAM 3D mesh")
            continue
        glb = out / "sam3d" / f"object_{number}.glb"
        target = inputs["positions"][inputs["sets"][f"object_{number}"]]
        source = sample_mesh(glb, ICP_SOURCE_POINTS)
        starts = {
            f"sam3d_pose/{fix}": pose_to_splat(
                obj["pose"],
                images[obj["view"]],
                transform,
                dp_scale,
                m_per_unit,
                matrix,
            )
            for fix, matrix in FRAME_FIXES.items()
        }
        start_name, result, tried = best_icp(source, target, starts)
        dense = sample_mesh(glb, POINTS_PER_OBJECT, seed=1)
        cloud = apply_similarity(
            dense, result["scale"], result["rotation"], result["translation"]
        )
        silhouette_cloud = apply_similarity(
            sample_mesh(glb, SILHOUETTE_POINTS, seed=2),
            result["scale"],
            result["rotation"],
            result["translation"],
        )
        world = splat_to_colmap(silhouette_cloud, transform, dp_scale)
        rendered, observed = [], []
        for m in masks_by_object.get(number, []):
            image = images[m["view"]]
            stack = np.load(out / "masks" / f"{Path(m['view']).stem}.npz")["masks"]
            rendered.append(
                render_mask(
                    world, image, cameras[image["camera_id"]], SILHOUETTE_DILATE_PX
                )
            )
            observed.append(stack[m["mask_index"]])
        silhouette = silhouette_agreement(rendered, observed) if rendered else None
        residuals = measure_residuals(
            result, target, None if silhouette is None else silhouette["silhouette"]
        )
        gate = accept_alignment(residuals, GATE)
        placed[f"object_{number}"] = cloud
        record.update(
            placed=True,
            start=start_name,
            starts_tried={k: round(v, 5) for k, v in tried.items()},
            scale=result["scale"],
            residuals={
                k: residuals[k]
                for k in (
                    "normalized_median",
                    "p90",
                    "median",
                    "coverage",
                    "silhouette",
                )
            },
            silhouette_views=None if silhouette is None else silhouette["views"],
            gate=gate["decision"],
            gate_failed=gate.get("failed"),
        )
    return {"placed": placed, "objects": records}


def run_bin_place(out: Path | str, data: Path | str = DATA_DIR) -> int:
    """ICP, place and score. Return 0 when both ledger items are written as accepted.

    ICP is accepted when every meshed object was aligned and measured; each
    object's gate decision is recorded beside it. The score is accepted when
    the score file is written; its numbers are reported, not gated.
    """
    out, data = Path(out), Path(data)
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
    evidence: dict[str, Any] = {"scene": SCENE}
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
        _, reference = reference_clouds(
            scene_dir, data / "models_eval", REFERENCE_IMAGE
        )
        score = score_bin(placement["placed"], reference, similarity)
        score["alignment"] = alignment
        score["matcher"] = "Hungarian on symmetric Chamfer"
        score["delivered_objects"] = len(placement["placed"])
        score["points_per_delivered"] = POINTS_PER_OBJECT
        score["file_name"] = (
            f"bin_{SCENE}_score_pts{POINTS_PER_OBJECT}_align-cameras-umeyama"
            "_match-hungarian-chamfer.json"
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
            delivered=len(placement["placed"]),
            reference=len(reference),
        )
    except (OSError, ValueError, KeyError, RuntimeError, StopIteration) as exc:
        reason = f"{type(exc).__name__}: {exc}"[-800:]
        for entry in (icp_entry, score_entry):
            if "decision" not in entry:
                entry.update(decision="reject", reason=reason)
    entries += [icp_entry, score_entry]
    ledger.update(scene_name=f"{SOURCE} {SCENE}", source=SOURCE, entries=entries)
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    (out / "bin_score.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    decisions = {e.get("item"): e.get("decision") for e in entries}
    return 0 if all(decisions.get(item) == "accept" for item in REQUIRED) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path(DATA_DIR))
    args = parser.parse_args(argv)
    return run_bin_place(args.out, args.data)


if __name__ == "__main__":
    sys.exit(main())
