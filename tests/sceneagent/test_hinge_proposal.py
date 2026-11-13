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
        return {"text": self.replies.pop(0), "model_id": "stub-model"}


def _mailbox():
    return [{"name": "object_1", "parts": [BODY, LID]}]


def test_the_mailbox_lid_is_one_revolute_and_images_are_sent():
    stub = Stub([GOOD])
    record = propose_articulation(_mailbox(), FRAMES, stub)
    obj = record["objects"][0]
    assert obj["status"] == "ok" and obj["source"] == SOURCE
    assert obj["joints"][0]["type"] == "revolute"
    assert obj["model_id"] == "stub-model"
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
    assert report["model_id"] == "stub-model"


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


def test_active_model_id_follows_one_constant(monkeypatch):
    from sceneagent.gemini_client import DEFAULT_MODEL_ID, active_model_id

    monkeypatch.delenv("GEMINI_MODEL_ID", raising=False)
    assert active_model_id() == DEFAULT_MODEL_ID == "gemini-3.8-flash"
    monkeypatch.setenv("GEMINI_MODEL_ID", "override-id")
    assert active_model_id() == "override-id"


def test_a_null_pivot_uses_the_shared_face_and_is_written_into_usd():
    from sceneagent.mailbox_hinge import proposal_documents

    raw = json.loads(GOOD)
    raw["joints"][0]["pivot"] = None
    raw["joints"][0]["axis"] = [-1, 0, 0]
    raw["joints"][0]["upper"] = 1.571
    stub = Stub([json.dumps(raw)])
    record = propose_articulation(_mailbox(), FRAMES, stub)
    joint = record["objects"][0]["joints"][0]
    assert record["objects"][0]["status"] == "ok"
    assert joint["origin"] == pytest.approx([0.1, 0.1, 0.2])
    assert joint["pivot_source"] == "shared_face"
    assert len(stub.images) == 1
    usda = proposal_documents(record)["usda"]
    assert "point3f physics:localPos0 = (0.1, 0.1, 0.2)" in usda
    assert "point3f physics:localPos1 = (0.1, 0.1, 0.2)" in usda


def test_a_null_pivot_without_a_shared_face_is_unresolved():
    raw = json.loads(GOOD)
    raw["joints"][0]["pivot"] = None
    text = json.dumps(raw)
    stub = Stub([text, text])
    bare = [{"name": "object_1", "parts": [{"name": "body"}, {"name": "lid"}]}]
    record = propose_articulation(bare, FRAMES, stub)
    obj = record["objects"][0]
    assert obj["status"] == "unresolved"
    assert obj["joints"] == []
    assert "pivot is missing" in obj["attempts"][-1]["error"]
    assert len(stub.images) == 2


def test_a_missing_stage_names_the_prim_path():
    from sceneagent.articulate_drive import require_stage

    class Empty:
        stage = None

    with pytest.raises(
        RuntimeError, match=r"USD stage is None before spawning /World/ground"
    ):
        require_stage(Empty(), "/World/ground")


def test_the_remote_command_uses_the_shipped_proposal_path(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    from sceneagent.vast_smoke import (
        ISAACLAB_PYTHON,
        REMOTE_REPO,
        build_remote_command,
        remote_out,
    )

    out = remote_out("articulate")
    argv = build_remote_command(REMOTE_REPO, out, pipeline="articulate")
    assert argv == [
        "env",
        "-u",
        "PYTHONPATH",
        ISAACLAB_PYTHON,
        f"{REMOTE_REPO}/sceneagent/articulate_drive.py",
        "--proposal",
        f"{REMOTE_REPO}/sceneagent/articulate_proposal.json",
        "--out",
        out,
    ]
    assert out == "/workspace/sf4-smoke/articulate"
    script = Path(__file__).resolve().parents[2] / "sceneagent" / "articulate_drive.py"
    missing = tmp_path / "not-a-proposal.json"
    local_out = tmp_path / "articulate-missing-proposal"
    done = subprocess.run(
        [
            sys.executable,
            str(script),
            "--proposal",
            str(missing),
            "--out",
            str(local_out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    ledger = json.loads((local_out / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["entries"][0]["decision"] == "reject"
    assert str(missing) in ledger["entries"][0]["reason"]
    assert "isaaclab" not in done.stderr.lower()


def test_the_sweep_builds_a_simulation_context():
    from pathlib import Path

    import sceneagent.articulate_drive as drive

    text = Path(drive.__file__).read_text(encoding="utf-8")
    assert "sim_utils.SimulationContext.instance" not in text
    assert "sim_utils.SimulationContext(sim_cfg)" in text


def test_the_png_encoder_loads_by_file_path():
    from sceneagent.articulate_drive import _load_png_bytes

    assert callable(_load_png_bytes())


def test_a_failed_call_is_unresolved_and_does_not_raise():
    class Down:
        def generate_images(self, prompt, images=(), search=False):
            return {"text": "", "http_status": 503, "model_id": "stub-model"}

    record = propose_articulation(_mailbox(), FRAMES, Down())
    obj = record["objects"][0]
    assert obj["status"] == "unresolved"
    assert obj["joints"] == []
    assert obj["http_status"] == 503
    assert obj["attempts"][0]["error"] == "HTTP 503"

    class TimingOut:
        def generate_images(self, prompt, images=(), search=False):
            raise TimeoutError("timed out")

    record = propose_articulation(_mailbox(), FRAMES, TimingOut())
    assert record["objects"][0]["status"] == "unresolved"
    assert record["objects"][0]["http_status"] == 0


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
