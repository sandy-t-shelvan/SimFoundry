# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KAN-51: an SF-35 export becomes a scene state the light editor can open.

Physics and joints are additive optional keys. A rigid object omits joints.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("pxr")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/interactive/light_editor"))

import export_import  # noqa: E402


def _record(**extra):
    record = {
        "name": "mailbox",
        "position": [1.0, 2.0, 0.0],
        "points": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        "face_counts": [3],
        "face_indices": [0, 1, 2],
        "approximation": "convexHull",
        "mass_kg": 0.4,
        "density_kg_m3": 200.0,
        "friction_static": 0.5,
        "friction_dynamic": 0.4,
        "restitution": 0.1,
        "metadata": {"body_type": "dynamic rigid", "material": "metal"},
    }
    record.update(extra)
    return record


def test_a_rigid_export_lands_in_the_scene_contract(tmp_path):
    scene_json = tmp_path / "mailbox_scene_state_exported.json"
    scene = export_import.build_document([_record()], scene_json)
    args = scene["objects_info"]["init_info"]["mailbox"]["args"]
    link = args["link_physics_materials"]["mailbox"]
    assert link["static_friction"] == 0.5
    assert link["dynamic_friction"] == 0.4
    assert link["restitution"] == 0.1
    assert "mass" not in args
    assert "joint_limits" not in args
    registry = scene["state"]["registry"]["object_registry"]["mailbox"]
    assert registry["root_link"]["pos"] == [1.0, 2.0, 0.0]
    assert registry["root_link"]["lin_vel"] == [0.0, 0.0, 0.0]
    assert registry["root_link"]["ang_vel"] == [0.0, 0.0, 0.0]
    assert "joint_pos" not in registry
    assert scene["physics_records"]["mailbox"]["body_type"] == "dynamic rigid"
    assert (tmp_path / "assets" / "mailbox.usda").is_file()
    import scene_io

    merged, preserved = scene_io.merge_settled_scene(
        scene,
        {
            "versions": {},
            "metadata": {},
            "state": scene["state"],
            "init_info": {},
            "objects_info": scene["objects_info"],
        },
    )
    assert "physics_records" in preserved
    assert merged["physics_records"]["mailbox"]["body_type"] == "dynamic rigid"


def test_joints_are_optional_and_named(tmp_path):
    record = _record(
        joints=[
            {"name": "lid", "position": 0.2, "lower": 0.0, "upper": 1.5},
        ]
    )
    scene = export_import.build_document([record], tmp_path / "lid_scene_state_exported.json")
    registry = scene["state"]["registry"]["object_registry"]["mailbox"]
    args = scene["objects_info"]["init_info"]["mailbox"]["args"]
    assert registry["joint_pos"] == [0.2]
    assert registry["joint_vel"] == [0.0]
    assert args["joint_limits"] == {"lid": {"lower": 0.0, "upper": 1.5}}
