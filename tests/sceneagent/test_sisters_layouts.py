# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sampled scene and task sisters. No simulator is started."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sceneagent.pipeline import run_path
from sceneagent.sisters import (
    procedural_texture,
    sample_layouts,
    sample_task_sisters,
    scene_sisters,
)

ROOT_EXAMPLE = (
    Path(__file__).resolve().parents[2] / "examples" / "phase1_place_scene.json"
)


def test_sampled_poses_stay_inside_the_disk_and_name_the_anchor():
    first = sample_layouts((1.0, 2.0), radius_m=0.1, n=8, seed=0, z_m=0.4)
    again = sample_layouts((1.0, 2.0), 0.1, seed=0, z_m=0.4)
    assert again == first
    assert first["anchor_xy_m"] == [1.0, 2.0]
    assert len(first["poses"]) == 8
    off_axis = 0
    for pose in first["poses"]:
        dx = pose["object_xyz_m"][0] - 1.0
        dy = pose["object_xyz_m"][1] - 2.0
        assert pose["object_xyz_m"][2] == pytest.approx(0.4)
        assert 0.025 - 1e-9 <= pose["gap_m"] <= 0.1 + 1e-9
        assert abs(pose["yaw_rad"]) <= np.pi
        if abs(dx) > 1e-6 and abs(dy) > 1e-6:
            off_axis += 1
    assert off_axis >= 1
    other = sample_layouts((1.0, 2.0), 0.1, seed=1)
    assert other["poses"] != first["poses"]


def test_the_four_fixed_offsets_stay_available():
    layouts = scene_sisters((0.0, 0.0), 0.05)
    assert [row["relation"] for row in layouts] == [
        "left_of",
        "right_of",
        "behind",
        "in_front_of",
    ]
    assert layouts[0]["object_xy_m"] == [-0.05, 0.0]


def test_task_samples_keep_the_inside_goal_and_the_texture_fields():
    rows = sample_task_sisters("pack the apple", "apple", "bowl", n=3, seed=2)
    assert rows[0]["source"] == "capture"
    assert rows[0]["instruction"] == "pack the apple"
    assert {row["goal"] for row in rows} == {"inside"}
    assert {row["source"] for row in rows[1:]} == {"model_free_sample"}
    assert len(rows) == 4
    texture = procedural_texture(20, 0.1, 0.2, 0.3)
    assert set(texture) >= {"noise", "hue_degrees", "mottle", "speckle"}


def test_the_capture_run_writes_the_layouts_json(tmp_path):
    run_path(ROOT_EXAMPLE, tmp_path / "run")
    document = json.loads(
        (tmp_path / "run" / "layouts.json").read_text(encoding="utf-8")
    )
    assert document["anchor_xy_m"]
    assert document["poses"]
    assert all("object_xyz_m" in pose for pose in document["poses"])
    varied = json.loads(
        (tmp_path / "run" / "task_instructions.json").read_text(encoding="utf-8")
    )
    assert varied[0]["source"] == "capture"
