# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PartSAM parts, USD joints, and a fracture proposal. No network and no box."""

from __future__ import annotations

import json

from sceneagent.fracture_proposal import RENDER_SKIPPED, propose_fracture
from sceneagent.hinge_proposal import propose_articulation
from sceneagent.mailbox_hinge import proposal_documents
from sceneagent.partsam_parts import parts_for_export, partsam_available


class Stub:
    def __init__(self, text):
        self.text = text
        self.images = []

    def generate_images(self, prompt, images=(), search=False):
        self.images.append(images)
        return {"text": self.text}


def test_partsam_parts_feed_the_proposal_and_completion_is_gated():
    calls = {"complete": 0}

    def worker(mesh):
        if mesh.endswith("mailbox.glb"):
            return [
                {
                    "name": "body",
                    "face_count": 12,
                    "mesh_path": "body.glb",
                    "graph_cut": False,
                },
                {
                    "name": "lid",
                    "face_count": 8,
                    "mesh_path": "lid.glb",
                    "graph_cut": True,
                },
            ]
        return [{"name": "mug", "face_count": 4, "mesh_path": "mug.glb"}]

    def completer(obj):
        calls["complete"] += 1
        return {
            "name": "tip",
            "face_count": 2,
            "mesh_path": "tip.glb",
            "source": "cubepart",
        }

    objects = parts_for_export(
        [
            {"name": "object_1", "mesh_path": "mailbox.glb"},
            {"name": "mug", "mesh_path": "mug.glb"},
        ],
        worker,
        reviews={
            "object_1": {"completion_needed": False},
            "mug": {"completion_needed": True},
        },
        completer=completer,
    )
    assert calls["complete"] == 1
    assert [part["name"] for part in objects[0]["parts"]] == ["body", "lid"]
    assert objects[0]["completion_called"] is False
    assert objects[1]["parts"][-1]["source"] == "cubepart"
    assert partsam_available(entry=__file__ + ".missing") is False
    stub = Stub(
        json.dumps(
            {
                "joints": [
                    {
                        "type": "revolute",
                        "parent": "body",
                        "child": "lid",
                        "axis": [0, 1, 0],
                        "pivot": [0, 0, 0.17],
                        "lower": 0,
                        "upper": 1.2,
                        "rationale": "The lid turns on the top edge.",
                    }
                ]
            }
        )
    )
    proposal = propose_articulation([objects[0]], [b"aa", b"bb"], stub)
    assert proposal["objects"][0]["status"] == "ok"


def test_proposed_joints_are_written_into_usd_and_the_scene_state():
    proposal = {
        "objects": [
            {
                "object": "object_1",
                "joints": [
                    {
                        "name": "body_to_lid",
                        "type": "revolute",
                        "parent": "body",
                        "child": "lid",
                        "axis": [0, 1, 0],
                        "lower": 0.0,
                        "upper": 1.5707963267948966,
                    }
                ],
            }
        ]
    }
    documents = proposal_documents(proposal)
    assert 'def PhysicsRevoluteJoint "body_to_lid"' in documents["usda"]
    assert 'physics:axis = "Y"' in documents["usda"]
    assert "physics:upperLimit = 90" in documents["usda"]
    scene = documents["scenes"][0]
    assert scene["joint_pos"] == [0.0]
    assert scene["joint_vel"] == [0.0]
    assert scene["joint_limits"]["body_to_lid"]["lower"] == 0.0


def test_fracture_is_proposed_and_not_rendered():
    stub = Stub(
        '{"breakable": true, "where": "the lid seam", "rationale": "It is thin there."}'
    )
    record = propose_fracture("object_1", [b"aa", b"bb", b"cc"], stub)
    assert record["source"] == "gemini_proposed"
    assert record["breakable"] is True
    assert record["where"] == "the lid seam"
    assert record["render"] == "skipped"
    assert record["render_reason"] == RENDER_SKIPPED
    assert len(stub.images) == 1
