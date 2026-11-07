# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GraspClutter6D bin scoring for SceneAgent section 4.1.

Ground truth is read from a BOP-format scene (``scene_gt.json``, poses in mm)
and its object models (``obj_XXXXXX.ply``, mm). Each model is sampled to
10,000 surface points and placed by its pose in one reference image's camera
frame. A reconstruction is mapped onto that frame by one similarity for the
whole run, with no per-object refit, then scored by
``metrics.score_reconstruction`` (Hungarian on Chamfer cost, F1 at 1 cm over
every reference object with unmatched objects at 0, Chamfer over matched
objects only). The paper's numbers stay the paper's; this module only scores.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from sceneagent.metrics import TAU_1CM_M, score_reconstruction
from sceneagent.similarity_icp import apply_similarity

POINTS_PER_OBJECT = 10_000
MM_TO_M = 0.001


def object_name(obj_id: int) -> str:
    return f"obj_{int(obj_id):06d}"


def image_keys(gt: dict[str, Any]) -> list[str]:
    """The ``scene_gt.json`` keys that are images, lowest first.

    GraspClutter6D scene 000091 also has keys -4 to -1 with no image behind
    them and a different object set (33 ids across them, against the scene's
    20). Only positive keys are images.
    """
    return sorted((key for key in gt if int(key) > 0), key=int)


def load_gt_poses(
    scene_dir: Path | str, image_id: int | None = None
) -> tuple[int, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Return the image id and each object's (rotation, translation in m).

    ``image_id`` defaults to the lowest image key in ``scene_gt.json``.
    """
    gt = json.loads((Path(scene_dir) / "scene_gt.json").read_text(encoding="utf-8"))
    key = str(image_id) if image_id is not None else image_keys(gt)[0]
    poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for entry in gt[key]:
        name = object_name(entry["obj_id"])
        if name in poses:
            # A duplicate id would silently overwrite one instance.
            raise ValueError(f"{name} appears twice in image {key}")
        rotation = np.asarray(entry["cam_R_m2c"], dtype=float).reshape(3, 3)
        translation = np.asarray(entry["cam_t_m2c"], dtype=float) * MM_TO_M
        poses[name] = (rotation, translation)
    return int(key), poses


def sample_model_points(
    ply_path: Path | str, n: int = POINTS_PER_OBJECT, seed: int = 0
) -> np.ndarray:
    """``n`` area-weighted surface points of a mm model, in metres."""
    import trimesh

    mesh = trimesh.load(ply_path, force="mesh", process=False)
    points, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(points, dtype=float) * MM_TO_M


def reference_clouds(
    scene_dir: Path | str,
    models_dir: Path | str,
    image_id: int | None = None,
    n: int = POINTS_PER_OBJECT,
    seed: int = 0,
) -> tuple[int, dict[str, np.ndarray]]:
    """Ground-truth clouds, one per object, in the reference image's camera frame."""
    key, poses = load_gt_poses(scene_dir, image_id)
    clouds = {
        name: apply_similarity(
            sample_model_points(Path(models_dir) / f"{name}.ply", n, seed),
            1.0,
            rotation,
            translation,
        )
        for name, (rotation, translation) in poses.items()
    }
    return key, clouds


def score_bin(
    delivered: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    similarity: tuple[float, Any, Any] = (1.0, np.eye(3), np.zeros(3)),
    tau: float = TAU_1CM_M,
) -> dict[str, Any]:
    """Apply one similarity to every delivered cloud, then score."""
    scale, rotation, translation = similarity
    placed = {
        name: apply_similarity(cloud, scale, rotation, translation)
        for name, cloud in delivered.items()
    }
    result = score_reconstruction(placed, reference, tau=tau)
    result["protocol"] = {
        "benchmark": "GraspClutter6D",
        "similarity": {
            "scale": float(scale),
            "rotation": np.asarray(rotation, dtype=float).tolist(),
            "translation": np.asarray(translation, dtype=float).tolist(),
        },
        "per_object_refit": False,
        "reference_objects": len(reference),
        "points_per_reference": sorted({len(c) for c in reference.values()}),
    }
    return result
