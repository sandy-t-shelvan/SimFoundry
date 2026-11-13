# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Articulation proposals. The Mailbox lid is one object. The client is a stub."""

from __future__ import annotations

import json

import pytest

from sceneagent.gemini_client import MissingGeminiKeyError, client_from_environ
from sceneagent.hinge_proposal import SOURCE, propose_articulation
from sceneagent.vlm_check import advise

BODY = {"name": "body", "bbox_min": [0, 0, 0], "bbox_max": [0.2, 0.2, 0.2]}
LID = {"name": "lid", "bbox_min": [0, 0, 0.2], "bbox_max": [0.2, 0.2, 0.25]}
FRAMES = [b"frame-a", b"frame-b", b"frame-c"]
GOOD = json.dumps(
    {
        "joints": [
            {
                "type": "revolute",
                "parent": "body",
                "child": "lid",
                "axis": [0, 1, 0],
                "pivot": [0, 0, 0.2],
                "lower": 0,
                "upper": 1.2,
                "rationale": "The lid turns about the shared top edge.",
            }
        ]
    }
)
BAD = json.dumps(
    {
        "joints": [
            {
                "type": "revolute",
                "parent": "body",
                "child": "lid",
                "axis": [2, 0, 0],
                "pivot": [0, 0, 0.2],
                "lower": 0,
                "upper": 1.2,
                "rationale": "A long axis.",
            }
        ]
    }
)


class Stub:
    def __init__(self, replies):
        self.replies = list(replies)
        self.images = []

    def generate_images(self, prompt, images=(), search=False):
        self.images.append(images)
        return {"text": self.replies.pop(0)}


def _mailbox():
    return [{"name": "object_1", "parts": [BODY, LID]}]


def test_the_mailbox_lid_is_one_revolute_and_images_are_sent():
    stub = Stub([GOOD])
    record = propose_articulation(_mailbox(), FRAMES, stub)
    obj = record["objects"][0]
    assert obj["status"] == "ok" and obj["source"] == SOURCE
    assert obj["joints"][0]["type"] == "revolute"
    assert stub.images == [tuple(FRAMES)] or stub.images == [FRAMES]
    assert "measured" not in obj["source"]


def test_a_bad_axis_is_retried_once_then_a_second_reject_is_unresolved():
    stub = Stub([BAD, GOOD])
    ok = propose_articulation(_mailbox(), FRAMES, stub)
    assert ok["objects"][0]["status"] == "ok"
    assert len(stub.images) == 2
    stub = Stub([BAD, BAD])
    stopped = propose_articulation(_mailbox(), FRAMES, stub)
    assert stopped["objects"][0]["status"] == "unresolved"
    assert stopped["objects"][0]["joints"] == []
    assert len(stub.images) == 2


def test_one_part_stays_rigid_and_does_not_call():
    stub = Stub([GOOD])
    record = propose_articulation([{"name": "mug", "parts": [BODY]}], FRAMES, stub)
    assert record["objects"][0]["status"] == "rigid"
    assert stub.images == []


def test_frames_are_required():
    with pytest.raises(ValueError, match="2 to 4"):
        propose_articulation(_mailbox(), [b"only"], Stub([GOOD]))


def test_the_advisory_check_does_not_change_a_source():
    stub = Stub(["yes, the lid opens"])
    report = advise(FRAMES, "does the lid open?", stub)
    assert report["advisory"] is True and report["label"] == "vlm_check"
    assert report["verdict"] == "yes"


def test_articulate_is_a_pipeline_and_angles_span_the_limits():
    from sceneagent.articulate_drive import ZIP_NAME, angles, movable
    from sceneagent.vast_smoke import (
        FETCH_NAMES,
        PIPELINES,
        PUBLISHED_NAMES,
        build_remote_command,
    )

    assert "articulate" in PIPELINES
    argv = build_remote_command(
        "/w/repo", "/workspace/sf4-smoke/articulate", pipeline="articulate"
    )
    assert "articulate_drive.py" in argv[4]
    assert "gemini" not in " ".join(argv).lower()
    assert angles(0.0, 1.4)[0] == 0.0
    assert angles(0.0, 1.4)[-1] == pytest.approx(1.4)
    assert len(angles(0.0, 1.4)) == 8
    assert movable(
        {"objects": [{"joints": [{"type": "revolute"}, {"type": "fixed"}]}]}
    ) == [{"type": "revolute"}]
    for name in (ZIP_NAME, "articulate_proposal.json", "vlm_check.json"):
        assert name in FETCH_NAMES and name in PUBLISHED_NAMES


def test_the_shared_client_hides_the_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(MissingGeminiKeyError, match="missing or empty"):
        client_from_environ()
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-value")
    assert "super-secret-value" not in repr(client_from_environ())


def test_committed_mailbox_frames_and_objects_can_be_proposed():
    from pathlib import Path

    from sceneagent.articulate_run import _frames, _objects

    root = Path(__file__).resolve().parents[2]
    frames = _frames(root)
    objects = _objects(root)
    assert 2 <= len(frames) <= 4
    assert all(frame.startswith(b"\x89PNG\r\n\x1a\n") for frame in frames)
    assert objects[0]["name"] == "object_1"
    assert [part["name"] for part in objects[0]["parts"]] == ["body", "lid"]
