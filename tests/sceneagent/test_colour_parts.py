# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KAN-42: the colour cut finds a cap of another colour and abstains on one colour."""

from __future__ import annotations

import json

import numpy as np
import pytest

from sceneagent import articulation_loop as loop
from sceneagent import colour_parts

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("scipy")


def _bar(path, cap_colour=(200, 20, 20, 255), body_colour=(230, 230, 230, 255)):
    """A 0.2 x 0.2 x 1.0 bar, long along Z; the top 25% is the cap's colour."""
    bar = trimesh.creation.box(extents=(0.2, 0.2, 1.0))
    bar = bar.subdivide().subdivide().subdivide()
    z = np.asarray(bar.vertices)[:, 2]
    colours = np.where((z > 0.25)[:, None], cap_colour, body_colour).astype(np.uint8)
    bar.visual = trimesh.visual.ColorVisuals(bar, vertex_colors=colours)
    bar.export(path)
    return path


def test_a_cap_of_another_colour_is_cut_off_at_its_boundary(tmp_path):
    record = colour_parts.split(_bar(tmp_path / "marker.glb"), tmp_path / "parts")
    cut = record["cut"]
    assert cut["axis"] == "Z" and abs(cut["at"] - 0.25) < 0.05
    body, cap = record["parts"]
    assert body["name"] == "part_0" and body["face_share"] > cap["face_share"]
    assert cap["bbox_min"][2] > 0.2  # the cap is the top end, whole
    assert (
        json.loads((tmp_path / "parts" / "parts.json").read_text())["method"]
        == "colour_cut"
    )


def test_a_one_colour_object_is_not_split(tmp_path):
    grey = (180, 180, 180, 255)
    assert (
        colour_parts.split(_bar(tmp_path / "bar.glb", grey, grey), tmp_path / "parts")
        is None
    )
    assert not (tmp_path / "parts" / "parts.json").exists()


class Critic:
    """Proposes a cap slide; the critic says no ``no`` times, then yes."""

    def __init__(self, no=1, joint_type="prismatic"):
        self.no, self.joint_type = no, joint_type

    def generate_images(self, prompt, images=(), search=False):
        if "Does the part" in prompt:
            verdict = "no" if self.no > 0 else "yes"
            self.no -= 1
            return {
                "text": json.dumps(
                    {"verdict": verdict, "reason": "the whole cap must move"}
                ),
                "http_status": 200,
                "model_id": "m",
            }
        joints = (
            []
            if self.joint_type == "fixed"
            else [
                {
                    "type": self.joint_type,
                    "parent": "part_0",
                    "child": "part_1",
                    "axis": [0, 0, 1],
                    "pivot": [0.0, 0.0, 0.4],
                    "lower": 0.0,
                    "upper": 0.2,
                    "rationale": "cap",
                }
            ]
        )
        return {
            "text": json.dumps({"joints": joints}),
            "http_status": 200,
            "model_id": "m",
        }


def _setup(tmp_path):
    parts_dir, run = tmp_path / "partsam", tmp_path / "run"
    (parts_dir / "parts").mkdir(parents=True)
    (run / "sam3d").mkdir(parents=True)
    source = _bar(run / "sam3d" / "object_1.glb")
    # PartSAM's parts: cut along the length, so neither is the whole cap.
    full = trimesh.load(source, force="mesh")
    x = np.asarray(full.triangles_center)[:, 0]
    parts = []
    for i, faces in enumerate((x < 0, x >= 0)):
        piece = full.submesh([np.nonzero(faces)[0]], append=True)
        path = parts_dir / "parts" / f"part_{i}.glb"
        piece.export(path)
        parts.append(
            {
                "name": f"part_{i}",
                "mesh": str(path),
                "face_share": 0.5,
                "bbox_min": piece.bounds[0].tolist(),
                "bbox_max": piece.bounds[1].tolist(),
            }
        )
    (parts_dir / "parts.json").write_text(
        json.dumps({"parent_object": "marker", "parts": parts})
    )
    (run / "bin_score.json").write_text(
        json.dumps(
            {"icp": {"objects": [{"object": 1, "mesh_extent_m": [0.02, 0.02, 0.13]}]}}
        )
    )
    return parts_dir, run


def _drive(links, out):
    out.mkdir(parents=True, exist_ok=True)
    return {"decision": "accept", "reason": "slid", "open_views": []}


def test_rejected_partsam_joints_fall_back_to_the_colour_cut(tmp_path):
    parts_dir, run = _setup(tmp_path)
    result = loop.articulate_object(
        parts_dir,
        run,
        1,
        tmp_path / "out",
        Critic(no=3),
        max_revisions=0,
        drive=_drive,
        frames=[b"a", b"b"],
    )
    assert result["partsam"]["decision"] == "rigid"
    assert result["colour_cut"]["cut"]["axis"] == "Z"
    assert result["decision"] == "articulated" and result["part_set"] == "colour_cut"
    assert (tmp_path / "out" / loop.RESULT).is_file()


def test_an_object_the_proposer_calls_rigid_never_gets_a_colour_cut(tmp_path):
    parts_dir, run = _setup(tmp_path)
    result = loop.articulate_object(
        parts_dir,
        run,
        1,
        tmp_path / "out",
        Critic(joint_type="fixed"),
        drive=_drive,
        frames=[b"a", b"b"],
    )
    assert result["decision"] == "rigid" and result["colour_cut"] is None
    assert not (tmp_path / "out" / "colour_parts").exists()
