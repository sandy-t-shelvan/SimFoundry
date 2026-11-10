# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase 4 placement math: SAM 3D pose to splat, scene similarity, ICP start, silhouette."""

from __future__ import annotations

import numpy as np
import pytest

from sceneagent import bin_place
from sceneagent.bin_association import qvec_to_rotation, splat_to_colmap
from sceneagent.similarity_icp import apply_similarity

pytest.importorskip("scipy")


def _random_rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def test_pose_to_splat_matches_the_frame_chain_step_by_step():
    rng = np.random.default_rng(3)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    pose = {
        "rotation": q.tolist(),
        "translation": [0.1, -0.05, 0.6],
        "scale": [0.2] * 3,
    }
    qvec = rng.normal(size=4)
    qvec /= np.linalg.norm(qvec)
    image = {"qvec": qvec.tolist(), "tvec": [0.3, -0.2, 1.5]}
    transform = np.hstack([_random_rotation(rng), rng.normal(size=(3, 1))])
    dp_scale, m_per_unit = 0.25, 0.2
    fix = bin_place.FRAME_FIXES["x+90"]
    p = rng.normal(size=(50, 3))

    # The chain written out: mesh -> PyTorch3D camera (m) -> OpenCV camera (m)
    # -> COLMAP units -> COLMAP world -> splat.
    p3d = (
        0.2 * (p @ (bin_place.quaternion_wxyz_to_matrix(q) @ fix).T)
        + pose["translation"]
    )
    cv = p3d * np.array([-1.0, -1.0, 1.0])
    colmap_cam = cv / (dp_scale * m_per_unit)
    r_c = qvec_to_rotation(image["qvec"])
    world = (colmap_cam - image["tvec"]) @ r_c
    expected = dp_scale * (world @ transform[:, :3].T + transform[:, 3])

    s, r, t = bin_place.pose_to_splat(pose, image, transform, dp_scale, m_per_unit, fix)
    assert np.allclose(apply_similarity(p, s, r, t), expected)
    assert np.allclose(splat_to_colmap(expected, transform, dp_scale), world)
    assert np.isclose(np.linalg.det(r), 1.0)


def test_metric_camera_centers_are_in_image_one_frame():
    rng = np.random.default_rng(5)
    world_cams = {1: (np.eye(3), np.zeros(3))}
    for key in (5, 9):
        world_cams[key] = (_random_rotation(rng), rng.normal(size=3))
    # One object at a fixed world pose; scene_gt gives model-to-camera in mm.
    r_o, t_o = _random_rotation(rng), np.array([0.1, 0.2, 0.8])
    gt = {}
    for key, (r_w2c, t_w2c) in world_cams.items():
        r_m2c = r_w2c @ r_o
        t_m2c = r_w2c @ t_o + t_w2c
        gt[str(key)] = [
            {
                "obj_id": 7,
                "cam_R_m2c": r_m2c.ravel().tolist(),
                "cam_t_m2c": (t_m2c * 1000).tolist(),
            }
        ]
    centers = bin_place.metric_camera_centers(
        gt, ["000001.png", "000005.png", "000009.png"]
    )
    assert np.allclose(centers["000001.png"], 0.0, atol=1e-9)
    r5, t5 = world_cams[5]
    assert np.allclose(centers["000005.png"], -r5.T @ t5)


def test_scene_similarity_recovers_a_known_similarity():
    rng = np.random.default_rng(7)
    splat = {f"{i:06d}.png": rng.normal(size=3) for i in range(1, 14)}
    rotation, scale, shift = _random_rotation(rng), 0.2, np.array([0.3, -0.1, 0.5])
    metric = {k: scale * rotation @ v + shift for k, v in splat.items()}
    (s, r, t), info = bin_place.scene_similarity(splat, metric)
    assert (
        s == pytest.approx(scale) and np.allclose(r, rotation) and np.allclose(t, shift)
    )
    assert info["cameras"] == 13 and info["camera_rms_m"] < 1e-9


def test_icp_from_a_near_start_beats_the_seeds_and_says_so():
    rng = np.random.default_rng(11)
    source = rng.uniform(-0.5, 0.5, size=(800, 3)) * np.array([1.0, 0.4, 0.2])
    rotation = _random_rotation(rng)
    target = apply_similarity(source, 2.0, rotation, np.array([1.0, 2.0, 3.0]))
    start = (1.9, rotation, np.array([1.02, 2.0, 3.0]))
    name, result, tried = bin_place.best_icp(source, target, {"sam3d_pose/none": start})
    assert name == "sam3d_pose/none"
    assert result["scale"] == pytest.approx(2.0, rel=1e-3)
    assert set(tried) == {"sam3d_pose/none", "seeds"}


def test_render_mask_projects_and_dilates():
    camera = {
        "model": "PINHOLE",
        "width": 20,
        "height": 10,
        "params": [10.0, 10.0, 10.0, 5.0],
    }
    image = {"qvec": [1.0, 0, 0, 0], "tvec": [0.0, 0.0, 0.0]}
    mask = bin_place.render_mask(np.array([[0.0, 0.0, 1.0]]), image, camera, dilate=1)
    assert mask.sum() == 5 and mask[5, 10]


def test_place_is_a_bin_pipeline():
    from sceneagent.vast_smoke import (
        FETCH_NAMES,
        PIPELINES,
        REMOTE_BIN_OUT,
        build_remote_command,
        remote_out,
    )

    argv = build_remote_command("/workspace/r", "/workspace/o", pipeline="place")
    assert argv[1:] == ["-m", "sceneagent.bin_place", "--out", "/workspace/o"]
    assert remote_out("place") == REMOTE_BIN_OUT
    assert PIPELINES["place"][0][-2:] == ("ICP residuals", "bin score")
    assert "bin_score.json" in FETCH_NAMES
