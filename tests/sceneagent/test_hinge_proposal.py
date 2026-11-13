# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mailbox hinge proposal. The client is a stub, so CI makes no call."""

from __future__ import annotations

import json

import pytest

from sceneagent.hinge_proposal import (
    SOURCE,
    MissingGeminiKeyError,
    client_from_environ,
    part_summary,
    propose_hinge,
    shared_region,
)
from sceneagent.mailbox_hinge import MAILBOX_OBJECT, parts_manifest

BODY = {
    "name": "body",
    "face_count": 12,
    "mesh_path": "body.glb",
    "bbox_min": [0.0, 0.0, 0.0],
    "bbox_max": [0.2, 0.2, 0.2],
}
LID = {
    "name": "lid",
    "face_count": 8,
    "mesh_path": "lid.glb",
    "bbox_min": [0.0, 0.0, 0.2],
    "bbox_max": [0.2, 0.2, 0.25],
}
GOOD = json.dumps(
    {
        "axis": [0, 1, 0],
        "pivot": [0.0, 0.0, 0.2],
        "lower": 0.0,
        "upper": 1.2,
        "rationale": "The lid turns about the shared top edge.",
    }
)
BAD_AXIS = json.dumps(
    {
        "axis": [2, 0, 0],
        "pivot": [0.0, 0.0, 0.2],
        "lower": 0.0,
        "upper": 1.2,
        "rationale": "A long axis.",
    }
)
FRAMES = [b"frame-a", b"frame-b", b"frame-c"]


class Stub:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def generate(self, prompt, frames):
        self.prompts.append(prompt)
        assert frames == FRAMES or len(frames) in (2, 3, 4)
        return {"text": self.replies.pop(0)}


def _manifest():
    return parts_manifest(
        MAILBOX_OBJECT,
        [
            {key: BODY[key] for key in ("name", "face_count", "mesh_path")},
            {key: LID[key] for key in ("name", "face_count", "mesh_path")},
        ],
    )


def test_the_summary_names_the_shared_edge():
    summary = part_summary(BODY, LID)
    assert summary["contact"] == {"min": [0.0, 0.0, 0.2], "max": [0.2, 0.2, 0.2]}
    assert shared_region([0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]) is None


def test_a_valid_reply_is_one_revolute_proposal():
    stub = Stub([GOOD])
    summary = part_summary(BODY, LID)
    record = propose_hinge(_manifest(), summary, FRAMES, stub)
    assert record["status"] == "ok"
    assert record["source"] == SOURCE
    assert record["joint"]["type"] == "revolute"
    assert record["joint"]["axis"] == [0.0, 1.0, 0.0]
    assert record["joint"]["origin"] == [0.0, 0.0, 0.2]
    assert "lid_hinge" in record["urdf"]
    assert record["attempts"][0]["model_reply"] == GOOD
    assert record["attempts"][0]["error"] is None
    assert len(stub.prompts) == 1
    assert "measured" not in record["source"]


def test_a_rejected_axis_is_retried_once_then_resolved():
    stub = Stub([BAD_AXIS, GOOD])
    record = propose_hinge(_manifest(), part_summary(BODY, LID), FRAMES, stub)
    assert record["status"] == "ok"
    assert len(stub.prompts) == 2
    assert "not unit length" in stub.prompts[1]
    assert record["attempts"][0]["error"]
    assert record["attempts"][1]["model_reply"] == GOOD


def test_two_rejections_stop_unresolved():
    stub = Stub([BAD_AXIS, BAD_AXIS])
    record = propose_hinge(_manifest(), part_summary(BODY, LID), FRAMES, stub)
    assert record["status"] == "unresolved"
    assert record["source"] == SOURCE
    assert record["joint"] is None and record["urdf"] is None
    assert len(record["attempts"]) == 2
    assert len(stub.prompts) == 2


def test_frame_count_is_checked_before_a_call():
    stub = Stub([GOOD])
    with pytest.raises(ValueError, match="2 to 4"):
        propose_hinge(_manifest(), part_summary(BODY, LID), [b"only"], stub)
    assert stub.prompts == []


def test_a_missing_key_is_not_echoed(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(MissingGeminiKeyError, match="missing or empty"):
        client_from_environ()
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-value")
    assert "super-secret-value" not in repr(client_from_environ())
