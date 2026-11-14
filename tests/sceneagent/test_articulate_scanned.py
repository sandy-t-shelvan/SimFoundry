# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KAN-42 to KAN-45 on a scanned object: parts to links, the frame, the drive helpers."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from sceneagent import articulate_scanned as asc

ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "sceneagent" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


drive = _load("articulate_scanned_drive")
worker = _load("partsam_worker")


def _part(name, lo, hi):
    return {"name": name, "bbox_min": list(lo), "bbox_max": list(hi)}


# The Mailbox video's parts (SAM 3D frame, Y up): a lid rim, the lid's top
# surface inside it, two body shells and a floor panel.
PARTS = [
    _part("part_0", (-0.66, -0.90, -0.63), (0.65, 0.80, 0.67)),
    _part("part_1", (-0.67, -0.90, -0.65), (0.65, 0.74, 0.70)),
    _part("part_2", (-0.73, 0.61, -0.71), (0.73, 0.90, 0.71)),
    _part("part_4", (-0.60, 0.75, -0.59), (0.62, 0.88, 0.57)),
    _part("part_7", (-0.53, -0.88, -0.42), (0.50, -0.86, 0.47)),
]
JOINT = {"type": "revolute", "parent": "part_1", "child": "part_2", "axis": [-1, 0, 0], "origin": [0, 0.76, -0.62], "lower": 0.0, "upper": 1.5708}


def test_unnamed_parts_inside_the_child_move_with_it():
    body, lid, why = asc.group_parts(PARTS, JOINT)
    assert lid == ["part_2", "part_4"]
    assert body == ["part_0", "part_1", "part_7"]
    assert why["part_4"].startswith("inside the child's box (1.00)")
    assert why["part_1"] == "joint parent"


def test_a_joint_naming_a_missing_part_is_refused():
    with pytest.raises(ValueError, match="child"):
        asc.group_parts(PARTS, {**JOINT, "child": "part_9"})


def test_box_share_is_one_inside_zero_outside_and_partial_across():
    assert asc.box_inside_share((0, 0, 0), (1, 1, 1), (-1, -1, -1), (2, 2, 2)) == 1.0
    assert asc.box_inside_share((0, 0, 0), (1, 1, 1), (5, 5, 5), (6, 6, 6)) == 0.0
    assert asc.box_inside_share((0, 0, 0), (2, 1, 1), (1, 0, 0), (3, 1, 1)) == pytest.approx(0.5)


def test_the_first_movable_joint_is_used_and_a_rigid_proposal_refused():
    proposal = {"objects": [{"joints": [{"type": "fixed"}, JOINT]}]}
    assert asc.movable_joint(proposal)["child"] == "part_2"
    with pytest.raises(ValueError, match="rigid"):
        asc.movable_joint({"objects": [{"joints": [{"type": "fixed"}]}]})


def test_the_frame_is_z_up_metric_centred_and_on_the_ground():
    bounds = np.array([[-0.73, -0.90, -0.71], [0.73, 0.90, 0.71]])
    scale, transform = asc.metric_frame(bounds, [0.124, 0.196, 0.184])
    assert scale == pytest.approx(0.196 / 1.8)
    top_rear = transform[:3, :3] @ np.array([0.0, 0.90, -0.71]) + transform[:3, 3]
    floor = transform[:3, :3] @ np.array([0.0, -0.90, 0.0]) + transform[:3, 3]
    assert floor[2] == pytest.approx(0.0, abs=1e-9)  # the base sits on z = 0
    assert top_rear[2] == pytest.approx(0.196)  # SAM 3D's +Y is the scene's +Z
    assert top_rear[1] == pytest.approx(0.71 * scale)  # SAM 3D's -Z is the scene's +Y


def test_a_negative_axis_flips_the_joint_coordinate():
    assert drive.axis_token([-1.0, 0.0, 0.0]) == ("X", -1.0)
    assert drive.axis_token([0.0, 0.0, 1.0]) == ("Z", 1.0)
    with pytest.raises(ValueError):
        drive.axis_token([0.6, 0.8, 0.0])


def test_the_sweep_opens_and_closes_over_four_seconds():
    plan = drive.targets(0.0, math.pi / 2)
    assert len(plan) == 241  # 4 s at 60 Hz, both ends included
    assert plan[0] == 0.0 and plan[-1] == pytest.approx(0.0)
    assert max(plan) == pytest.approx(math.pi / 2) and plan.index(max(plan)) == 120


def test_obj_faces_are_triangulated(tmp_path):
    path = tmp_path / "q.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1/1 2/2 3/3 4/4\n")
    verts, faces = drive.read_obj(path)
    assert len(verts) == 4 and faces == [(0, 1, 2), (0, 2, 3)]


def test_partsam_labels_become_part_meshes_and_crumbs_are_listed(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    big = trimesh.creation.box(extents=(1, 1, 1))
    small = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    small.apply_translation((0, 0, 2))
    mesh = trimesh.util.concatenate([big] * 100 + [small])  # 1200 + 12 faces: the box is under 1%
    labels = np.array([0] * (len(mesh.faces) - 12) + [1] * 12)
    record = worker.split_parts(
        {"labels": labels, "vertices": np.asarray(mesh.vertices), "faces": np.asarray(mesh.faces)},
        tmp_path,
        "mesh.glb",
    )
    assert [p["name"] for p in record["parts"]] == ["part_0"]
    assert record["dropped"] == [{"label": 1, "faces": 12}]
    written = json.loads((tmp_path / "parts.json").read_text())
    assert written["parts"][0]["bbox_max"][2] == pytest.approx(0.5)
    assert (tmp_path / "parts" / "part_0.glb").is_file()
