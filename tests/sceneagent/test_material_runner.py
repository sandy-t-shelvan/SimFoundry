# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SF-20 and SF-21 on the runner: the equation (2) vote and the material audit.

No test calls Gemini. The harness is a fake that records prompts and images.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from sceneagent import material_runner as mr
from sceneagent.material_vote import MaterialVlmError
from sceneagent.physics import UNKNOWN_WEIGHT_SCALE

FAKE_KEY = "AIza" + "Q" * 35


class FakeHarness:
    model_id = "gemini-3.8-flash"

    def __init__(self, replies):
        self.replies = list(replies)
        self.call_count = 0
        self.calls = []

    def complete(self, prompt, images=(), response_schema=None):
        self.call_count += 1
        self.calls.append((prompt, list(images), response_schema))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _crops_zip(tmp_path: Path, objects=None, crops=None) -> Path:
    objects = objects or [
        {"object": 0, "name": "red apple"},
        {"object": 3, "name": "mug"},
    ]
    crops = crops or {
        "object_0_0.png": b"a0",
        "object_3_0.png": b"m0",
        "object_3_1.png": b"m1",
    }
    path = tmp_path / "crops.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("objects.json", json.dumps(objects))
        for name, blob in crops.items():
            archive.writestr(name, blob)
        archive.writestr("notes.txt", b"ignored")
    return path


def _vote(material, confidence):
    return json.dumps({"material_class": material, "confidence": confidence})


def _no_prior(tmp_path):
    return tmp_path / "absent" / "physics_prior.npz"


def test_crops_are_read_whole_and_in_order(tmp_path):
    objects = mr.read_crops(_crops_zip(tmp_path))
    assert [o["object"] for o in objects] == [0, 3]
    assert objects[1]["crops"] == [b"m0", b"m1"]
    with pytest.raises(ValueError, match="has no crop"):
        mr.read_crops(_crops_zip(tmp_path, crops={"object_0_0.png": b"a0"}))
    with pytest.raises(ValueError, match="not in objects.json"):
        mr.read_crops(
            _crops_zip(
                tmp_path,
                objects=[{"object": 0, "name": "apple"}],
                crops={"object_0_0.png": b"a", "object_9_0.png": b"x"},
            )
        )


def test_dc1_the_runner_vote_has_a_null_prior_and_records_the_sources(tmp_path):
    objects = mr.read_crops(_crops_zip(tmp_path))
    harness = FakeHarness([_vote("unknown", 0.9), _vote("glass and ceramic", 0.7)])
    result = mr.run_vote(
        objects, tmp_path / "out", harness, "123", checkpoint=_no_prior(tmp_path)
    )
    apple, mug = result["votes"]
    prior = next(s for s in apple["sources"] if s["source"] == "physics_prior")
    assert prior["null"] is True and prior["class"] is None and prior["weight"] is None
    vlm = next(s for s in apple["sources"] if s["source"] == "vlm")
    # dc2: the VLM said unknown at 0.9, which enters at 0.9 * 0.4.
    assert vlm["class"] == "unknown" and vlm["unknown_scale_applied"] is True
    assert vlm["weight"] == pytest.approx(0.9 * UNKNOWN_WEIGHT_SCALE)
    assert apple["material_class"] == "food and organic matter"  # the keyword wins
    assert mug["material_class"] == "glass and ceramic"
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["vlm_calls"] == 2 and manifest["source_run"] == "123"
    assert manifest["prompt_sha256"] == mr.sha256_text(mr.vlm_vote_prompt())
    assert harness.calls[1][1] == [(b"m0", "image/png"), (b"m1", "image/png")]
    assert harness.calls[0][2] == mr.VOTE_SCHEMA


def test_a_bad_vote_reply_fails_the_run(tmp_path):
    objects = mr.read_crops(_crops_zip(tmp_path))
    harness = FakeHarness([_vote("granite", 0.9)])
    with pytest.raises(MaterialVlmError):
        mr.run_vote(objects, tmp_path / "out", harness, checkpoint=_no_prior(tmp_path))


def _vote_record(name, material):
    return {
        "object": name,
        "material_class": material,
        "sources": [
            {"source": "physics_prior", "null": True},
            {"source": "vlm", "class": material, "weight": 0.8, "null": False},
            {"source": "keyword", "class": "unknown", "weight": 0.4, "null": False},
        ],
    }


def _audit(decision, material, reason="x"):
    return json.dumps(
        {"decision": decision, "material_class": material, "reason": reason}
    )


def test_dc2_the_audit_overrides_the_vote_and_signs_the_review(tmp_path):
    objects = mr.read_crops(_crops_zip(tmp_path))
    vote = {
        "votes": [
            _vote_record("red apple", "plastic"),
            _vote_record("mug", "glass and ceramic"),
        ]
    }
    harness = FakeHarness(
        [
            _audit("replace", "food and organic matter", "a real apple, not a toy"),
            _audit("confirm", "glass and ceramic"),
        ]
    )
    review = mr.run_audit(objects, vote, harness, "123")
    apple, mug = review["reviews"]
    assert apple["provisional_class"] == "plastic"
    assert apple["audit_class"] == apple["final_class"] == "food and organic matter"
    assert (
        apple["decision"] == "replace" and apple["reason"] == "a real apple, not a toy"
    )
    assert mug["decision"] == "confirm" and mug["final_class"] == "glass and ceramic"
    assert apple["signed_by"] == review["signed_by"] == mr.SIGNED_BY
    assert "coding agent" in mr.SIGNED_BY
    assert review["replaced"] == 1 and review["vlm_calls"] == 2
    assert "Provisional class: plastic" in harness.calls[0][0]
    assert "physics_prior: did not vote" in harness.calls[0][0]
    assert harness.calls[0][2] == mr.AUDIT_SCHEMA


def test_a_failed_audit_keeps_the_provisional_class_and_is_recorded(tmp_path):
    objects = mr.read_crops(_crops_zip(tmp_path))
    vote = {
        "votes": [_vote_record("red apple", "plastic"), _vote_record("mug", "metal")]
    }
    harness = FakeHarness([RuntimeError("HTTP 503"), _audit("replace", "metal")])
    review = mr.run_audit(objects, vote, harness)
    assert review["reviews"][0]["final_class"] == "plastic"
    assert review["reviews"][0]["decision"] == "not audited"
    assert [f["object"] for f in review["failed"]] == [0, 3]  # "replace" with itself


def test_audit_replies_must_agree_with_their_decision():
    assert mr.parse_audit(_audit("confirm", "metal"), "metal")["decision"] == "confirm"
    for bad, provisional in (
        (_audit("confirm", "wood"), "metal"),
        (_audit("replace", "metal"), "metal"),
        (_audit("keep", "metal"), "metal"),
        (_audit("replace", "granite"), "metal"),
        ("not json", "metal"),
    ):
        with pytest.raises(MaterialVlmError):
            mr.parse_audit(bad, provisional)


def test_votes_are_matched_by_position_and_checked_by_name(tmp_path):
    objects = mr.read_crops(_crops_zip(tmp_path))
    with pytest.raises(ValueError, match="1 votes for 2 objects"):
        mr.run_audit(
            objects, {"votes": [_vote_record("red apple", "plastic")]}, FakeHarness([])
        )
    swapped = {
        "votes": [_vote_record("mug", "metal"), _vote_record("red apple", "plastic")]
    }
    with pytest.raises(ValueError, match="is not 'red apple'"):
        mr.run_audit(objects, swapped, FakeHarness([]))


def test_dc3_the_cli_writes_both_files_without_the_key(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.setattr(
        "sceneagent.material_vote.configured_prior_checkpoint",
        lambda checkpoint=None: _no_prior(tmp_path),
    )
    crops = _crops_zip(tmp_path)
    out = tmp_path / "out"
    voter = FakeHarness([_vote("plastic", 0.8), _vote("glass and ceramic", 0.6)])
    assert mr.main(["vote", "--crops", str(crops), "--out-dir", str(out)], voter) == 0
    auditor = FakeHarness(
        [
            _audit("confirm", "food and organic matter"),
            _audit("replace", "metal"),
        ]
    )
    args = ["audit", "--crops", str(crops), "--vote", str(out / "material_vote.json")]
    args += ["--out", str(out / "material_review.json"), "--source-run", "42"]
    assert mr.main(args, auditor) == 0
    review = json.loads((out / "material_review.json").read_text())
    assert review["source_run"] == "42" and review["vlm_calls"] == 2
    assert review["replaced"] == 1 and review["reviews"][1]["final_class"] == "metal"
    for path in out.iterdir():
        assert FAKE_KEY not in path.read_text()
    with pytest.raises(SystemExit):
        mr.main(
            [
                "vote",
                "--crops",
                str(crops),
                "--out-dir",
                str(out),
                "--source-run",
                "x;y",
            ],
            voter,
        )
