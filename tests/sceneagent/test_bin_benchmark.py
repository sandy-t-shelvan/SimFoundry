# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GraspClutter6D bin scoring (SceneAgent section 4.1) on a synthetic BOP scene."""

from __future__ import annotations

import json
import time
import tracemalloc

import numpy as np
import pytest
import trimesh

from sceneagent import metrics
from sceneagent.bin_benchmark import (
    POINTS_PER_OBJECT,
    image_keys,
    load_gt_poses,
    reference_clouds,
    sample_model_points,
    score_bin,
)


def _rot_z(deg):
    a = np.radians(deg)
    return np.array(
        [[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]]
    )


def _bop_scene(tmp_path):
    """Two boxes in millimetres, posed in image 1, written the way BOP stores them."""
    models = tmp_path / "models"
    models.mkdir()
    trimesh.creation.box(extents=(50, 50, 50)).export(models / "obj_000001.ply")
    trimesh.creation.box(extents=(80, 40, 20)).export(models / "obj_000007.ply")
    scene = tmp_path / "scenes" / "000091"
    scene.mkdir(parents=True)
    gt = {
        "1": [
            {
                "obj_id": 1,
                "cam_R_m2c": np.eye(3).ravel().tolist(),
                "cam_t_m2c": [0.0, 0.0, 600.0],
            },
            {
                "obj_id": 7,
                "cam_R_m2c": _rot_z(30).ravel().tolist(),
                "cam_t_m2c": [150.0, -40.0, 650.0],
            },
        ],
        "2": [
            {
                "obj_id": 1,
                "cam_R_m2c": np.eye(3).ravel().tolist(),
                "cam_t_m2c": [9.0, 9.0, 9.0],
            }
        ],
        # GraspClutter6D 000091 has non-image keys -4..-1 with other objects.
        "-4": [
            {
                "obj_id": 83,
                "cam_R_m2c": np.eye(3).ravel().tolist(),
                "cam_t_m2c": [0.0, 0.0, 1.0],
            }
        ],
    }
    (scene / "scene_gt.json").write_text(json.dumps(gt), encoding="utf-8")
    return scene, models


def test_min_distances_match_brute_force():
    rng = np.random.default_rng(0)
    source, target = rng.normal(size=(230, 3)), rng.normal(size=(310, 3))
    brute = np.linalg.norm(source[:, None, :] - target[None, :, :], axis=2).min(axis=1)
    assert np.allclose(
        metrics._min_distances(source, target), brute, rtol=1e-12, atol=0
    )


def test_protocol_sized_clouds_score_without_the_full_distance_matrix():
    # 10,000 x 10,000 points at once was about 2.4 GB; the KD-tree stays small.
    rng = np.random.default_rng(1)
    x = rng.uniform(size=(POINTS_PER_OBJECT, 3))
    y = x + 0.001
    tracemalloc.start()
    started = time.perf_counter()
    chamfer = metrics.symmetric_chamfer(x, y)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert chamfer == pytest.approx(np.sqrt(3) * 0.001, rel=1e-3)
    assert peak < 200 * 2**20
    # The bin needs 400 of these; brute force took about 7 s each.
    assert elapsed < 2.0


def test_poses_come_from_the_first_image_in_metres(tmp_path):
    scene, _ = _bop_scene(tmp_path)
    image_id, poses = load_gt_poses(scene)
    assert image_id == 1
    assert set(poses) == {"obj_000001", "obj_000007"}
    rotation, translation = poses["obj_000007"]
    assert np.allclose(rotation, _rot_z(30))
    assert np.allclose(translation, [0.15, -0.04, 0.65])


def test_non_image_keys_are_never_the_default_image():
    assert image_keys({"-4": [], "-1": [], "2": [], "10": [], "1": []}) == ["1", "2", "10"]


def test_a_repeated_object_in_one_image_is_refused(tmp_path):
    scene, _ = _bop_scene(tmp_path)
    gt = json.loads((scene / "scene_gt.json").read_text())
    gt["1"].append(dict(gt["1"][0]))
    (scene / "scene_gt.json").write_text(json.dumps(gt))
    with pytest.raises(ValueError, match="obj_000001 appears twice"):
        load_gt_poses(scene)


def test_model_points_are_on_the_surface_in_metres(tmp_path):
    _, models = _bop_scene(tmp_path)
    points = sample_model_points(models / "obj_000001.ply", n=500)
    assert points.shape == (500, 3)
    assert np.allclose(np.abs(points).max(axis=0), 0.025)


def test_ground_truth_against_itself_is_perfect(tmp_path):
    scene, models = _bop_scene(tmp_path)
    _, reference = reference_clouds(scene, models, n=2000)
    _, delivered = reference_clouds(scene, models, n=2000, seed=1)
    result = score_bin(delivered, reference)
    assert result["f1_mean"] == pytest.approx(1.0)
    assert result["chamfer_mean_matched_m"] < 0.002
    assert result["protocol"]["per_object_refit"] is False


def test_one_shifted_object_loses_its_f1_only(tmp_path):
    scene, models = _bop_scene(tmp_path)
    _, reference = reference_clouds(scene, models, n=2000)
    delivered = {name: cloud.copy() for name, cloud in reference.items()}
    # 10 cm clears the 80 mm box; a smaller shift still leaves surface overlap.
    delivered["obj_000007"] = delivered["obj_000007"] + [0.10, 0.0, 0.0]
    result = score_bin(delivered, reference)
    assert result["f1_by_reference"]["obj_000001"] == pytest.approx(1.0)
    assert result["f1_by_reference"]["obj_000007"] == pytest.approx(0.0)
    assert result["f1_mean"] == pytest.approx(0.5)


def test_a_missing_object_scores_zero_and_leaves_chamfer_to_the_matched(tmp_path):
    scene, models = _bop_scene(tmp_path)
    _, reference = reference_clouds(scene, models, n=2000)
    result = score_bin({"only": reference["obj_000001"]}, reference)
    assert result["f1_by_reference"]["obj_000007"] == 0.0
    assert result["f1_mean"] == pytest.approx(0.5)
    assert result["chamfer_mean_matched_m"] == pytest.approx(0.0)


def test_one_similarity_maps_a_reconstruction_frame_onto_ground_truth(tmp_path):
    scene, models = _bop_scene(tmp_path)
    _, reference = reference_clouds(scene, models, n=2000)
    scale, rotation, translation = 2.5, _rot_z(-40), np.array([0.3, -0.2, 1.0])
    # A reconstruction lives in its own frame: undo the similarity to make one.
    inverse_rotation = rotation.T
    delivered = {
        name: (inverse_rotation @ (cloud - translation).T).T / scale
        for name, cloud in reference.items()
    }
    unaligned = score_bin(delivered, reference)
    aligned = score_bin(delivered, reference, similarity=(scale, rotation, translation))
    assert unaligned["f1_mean"] < 0.1
    assert aligned["f1_mean"] == pytest.approx(1.0)
    assert aligned["protocol"]["similarity"]["scale"] == scale
