# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KAN-24 dc1: the orientation sweep's candidates, rotations and choice."""

from __future__ import annotations

import numpy as np
import pytest

from sceneagent import orientation_sweep as osw

# The Mailbox export's support record (box 60254439, run viewpick-mailbox-e2e1).
SUPPORT = {
    "world_from_scene": [
        [0.992241, -0.038951, 0.118074, 0.166926],
        [-0.038951, 0.804474, 0.592709, -1.044973],
        [-0.118074, -0.592709, 0.796715, 0.897252],
        [0.0, 0.0, 0.0, 1.0],
    ],
    "chosen": 1,
    "candidates": [
        {"normal": [-0.13196, -0.56264, 0.8161], "inliers": 11431, "score": 0.02993},
        {"normal": [-0.11807, -0.59271, 0.79671], "inliers": 9120, "score": 0.00165},
        {"normal": [0.81668, -0.06757, -0.57313], "inliers": 3290, "score": 1.14159},
    ],
}


@pytest.mark.parametrize("up", [[0, 0, 1], [0, 0, -1], [1, 0, 0], [0.3, -0.5, 0.8], [0.7453, -0.4259, -0.513]])
def test_the_rotation_turns_the_candidate_up_onto_z(up):
    rotation = osw.rotation_onto_z(up)
    assert np.allclose(rotation @ osw.unit(up), [0, 0, 1], atol=1e-9)
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_the_plane_fit_choice_lands_on_z_and_the_wall_does_not():
    planes = osw.plane_candidates(SUPPORT)
    chosen = next(p for p in planes if p["plane_fit_chosen"])
    assert chosen["name"] == "plane_1"
    assert np.allclose(chosen["up_export"], [0, 0, 1], atol=1e-4)
    wall = planes[2]
    assert abs(wall["up_export"][2]) < 0.6  # far from vertical: a wall, not a support
    assert wall["up_scene"] == pytest.approx(list(osw.unit(SUPPORT["candidates"][2]["normal"])), abs=1e-6)


def test_eight_tilts_of_two_degrees_surround_the_best_plane():
    rotation = np.asarray(SUPPORT["world_from_scene"])[:3, :3]
    planes = osw.plane_candidates(SUPPORT)
    tilts = osw.tilt_candidates(planes[1], rotation)
    assert len(tilts) == 8
    for tilt in tilts:
        angle = np.degrees(np.arccos(np.clip(tilt["up_export"][2], -1, 1)))
        assert 1.9 < angle < 2.9  # 2 degrees about one axis, or about both


def test_an_orientation_where_objects_fell_off_never_wins():
    scored = [
        {"name": "plane_1", "plane_fit_chosen": True, "late_drift_mean_m": 0.0, "fell_off": False},
        {"name": "plane_2", "plane_fit_chosen": False, "late_drift_mean_m": -1.0, "fell_off": True},
    ]
    assert osw.choose(scored)["name"] == "plane_1"


def test_a_tie_keeps_the_plane_fit_and_a_clear_win_does_not():
    tie = [
        {"name": "tilt", "plane_fit_chosen": False, "late_drift_mean_m": 0.0000, "fell_off": False},
        {"name": "plane_1", "plane_fit_chosen": True, "late_drift_mean_m": 0.0004, "fell_off": False},
    ]
    assert osw.choose(tie)["name"] == "plane_1"
    clear = [
        {"name": "tilt", "plane_fit_chosen": False, "late_drift_mean_m": 0.0, "fell_off": False},
        {"name": "plane_1", "plane_fit_chosen": True, "late_drift_mean_m": 0.02, "fell_off": False},
    ]
    assert osw.choose(clear)["name"] == "tilt"


@pytest.mark.parametrize("up", [[0, 0, 1], [0, 0, -1], [0.3, -0.5, 0.8], [1, 0, 0]])
def test_the_quaternion_is_the_same_rotation(up):
    rotation = osw.rotation_onto_z(up)
    w, x, y, z = osw.quat_wxyz(rotation)
    rebuilt = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    assert np.allclose(rebuilt, rotation, atol=1e-9)


def test_an_export_without_candidates_is_refused():
    with pytest.raises(ValueError, match="no support-plane"):
        osw.plane_candidates({**SUPPORT, "candidates": []})
