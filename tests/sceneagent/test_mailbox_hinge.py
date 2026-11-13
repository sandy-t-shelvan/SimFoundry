# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mailbox lid: parts, completion gate, one revolute joint, scene state."""

from __future__ import annotations

import math

import pytest

from sceneagent.mailbox_hinge import (
    MAILBOX_MESH,
    MAILBOX_OBJECT,
    MAILBOX_RUN,
    STOP_REASON,
    apply_completion,
    hinge_from_reply,
    hinge_scene,
    hinge_usda,
    live_articulate,
    motion_failed,
    parts_manifest,
    record_verdict,
    review_parts,
    sweep_targets,
    write_urdf,
)

BODY = {"name": "body", "face_count": 12, "mesh_path": "body.glb", "graph_cut": False}
LID = {"name": "lid", "face_count": 8, "mesh_path": "lid.glb", "graph_cut": True}
REPLY = {
    "name": "lid_hinge",
    "type": "revolute",
    "parent": "body",
    "child": "lid",
    "axis": [0.0, 1.0, 0.0],
    "origin": [0.0, 0.0, 0.1],
    "lower": 0.0,
    "upper": math.pi / 2,
}


def _manifest():
    return parts_manifest(MAILBOX_OBJECT, [BODY, LID])


def test_parts_manifest_names_body_and_lid_from_the_mailbox_export():
    manifest = _manifest()
    assert manifest["object"] == MAILBOX_OBJECT
    assert manifest["source_run"] == MAILBOX_RUN
    assert MAILBOX_MESH.endswith("object_1.glb")
    assert [part["name"] for part in manifest["parts"]] == ["body", "lid"]
    assert manifest["parts"][1]["graph_cut"] is True
    assert manifest["parts"][0]["parent_object"] == MAILBOX_OBJECT


def test_a_malformed_review_is_a_failure_and_does_not_mean_no():
    failed = review_parts(["body", "lid"], "maybe the lid")
    assert failed["status"] == "failure"
    assert failed["completion_needed"] is None
    assert failed["call_cubepart"] is False
    clear = review_parts(["body", "lid"], '{"completion_needed": false}')
    assert clear["completion_needed"] is False


def test_completion_runs_only_when_needed_and_rejects_an_oversized_part():
    manifest = _manifest()
    calls = {"n": 0}

    def worker():
        calls["n"] += 1
        return {
            "name": "tip",
            "face_count": 4,
            "mesh_path": "tip.glb",
            "size": [0.01, 0.01, 0.01],
        }

    skipped = apply_completion(
        {"completion_needed": False}, manifest, worker, [0.2, 0.2, 0.2]
    )
    assert skipped["called"] is False and calls["n"] == 0
    assert skipped["skipped"] == "completion not needed"
    added = apply_completion(
        {"completion_needed": True}, manifest, worker, [0.2, 0.2, 0.2]
    )
    assert calls["n"] == 1 and added["parts"][-1]["source"] == "cubepart"

    def huge():
        return {
            "name": "tip",
            "face_count": 4,
            "mesh_path": "tip.glb",
            "size": [1.0, 0.01, 0.01],
        }

    rejected = apply_completion(
        {"completion_needed": True}, manifest, huge, [0.2, 0.2, 0.2]
    )
    assert rejected["rejected"] == "part outside 1.5x bounding box"
    assert [part["name"] for part in rejected["parts"]] == ["body", "lid"]


def test_a_canned_reply_is_one_revolute_and_the_checker_rejects_four_ways():
    manifest = _manifest()
    built = hinge_from_reply(manifest, REPLY)
    assert built["status"] == "ok"
    assert built["joint"]["type"] == "revolute"
    urdf = write_urdf(manifest, built["joint"])
    assert '<link name="body"/>' in urdf and '<link name="lid"/>' in urdf
    assert 'type="revolute"' in urdf and 'name="lid_hinge"' in urdf
    stray = {**REPLY, "child": "wheel"}
    with pytest.raises(ValueError, match="no part"):
        hinge_from_reply(manifest, stray)
    one_sided = {**REPLY, "child": "body"}
    with pytest.raises(ValueError, match="no link"):
        hinge_from_reply(manifest, one_sided)
    with pytest.raises(ValueError, match="not unit length"):
        hinge_from_reply(manifest, {**REPLY, "axis": [2.0, 0.0, 0.0]})
    with pytest.raises(ValueError, match="not below"):
        hinge_from_reply(manifest, {**REPLY, "lower": 1.0, "upper": 1.0})


def test_the_live_tool_stops_because_it_cannot_take_part_meshes():
    stopped = live_articulate()
    assert stopped == {"status": "stopped", "reason": STOP_REASON}
    again = hinge_from_reply(_manifest(), stopped)
    assert again["status"] == "stopped"
    assert "joint" not in again


def test_scene_state_and_usda_keep_the_named_hinge():
    joint = hinge_from_reply(_manifest(), REPLY)["joint"]
    scene = hinge_scene(MAILBOX_OBJECT, joint)
    registry = scene["state"]["registry"]["object_registry"][MAILBOX_OBJECT]
    assert registry["joint_pos"] == [0.0]
    assert registry["joint_vel"] == [0.0]
    limits = scene["objects_info"]["init_info"][MAILBOX_OBJECT]["args"]["joint_limits"]
    assert list(limits) == ["lid_hinge"]
    assert limits["lid_hinge"]["upper"] == pytest.approx(math.pi / 2)
    usda = hinge_usda(MAILBOX_OBJECT, joint)
    assert 'def PhysicsRevoluteJoint "lid_hinge"' in usda
    assert 'physics:axis = "Y"' in usda
    assert "physics:lowerLimit = 0" in usda
    assert "physics:upperLimit = 90" in usda
    assert "physics:body0" in usda and "physics:body1" in usda


def test_the_sweep_goes_out_and_back_and_a_reject_retries_once():
    rows = sweep_targets(0.0, math.pi / 2)
    assert rows[0] == {"t": 0.0, "joint_pos": 0.0}
    assert rows[120]["joint_pos"] == pytest.approx(math.pi / 2)
    assert rows[-1]["t"] == pytest.approx(4.0)
    assert rows[-1]["joint_pos"] == pytest.approx(0.0)
    assert len(rows) == 241
    assert motion_failed(0.02, 0.0) and motion_failed(0.0, 6.0)
    assert motion_failed(0.01, 5.0) is False
    seen = []

    def judge(limits):
        seen.append(dict(limits))
        if len(seen) == 1:
            return {"verdict": "reject", "limits": {"lower": 0.0, "upper": 1.0}}
        return {"verdict": "accept"}

    record = record_verdict(judge, {"lower": 0.0, "upper": math.pi / 2})
    assert [item["verdict"] for item in record["attempts"]] == ["reject", "accept"]
    assert record["limits"]["upper"] == 1.0
    assert len(seen) == 2
