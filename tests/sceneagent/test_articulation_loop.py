# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KAN-44/45: the articulation loop's QA, revision and fail-closed paths, with no simulator or model."""

from __future__ import annotations

import json

import numpy as np
import pytest

from sceneagent import articulation_loop as loop
from sceneagent import joint_qa

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("scipy")

# Body 2 x 2 footprint; the lid overhangs it by 0.1 on -X, +X and -Z and is
# flush on +Z, so its hinge belongs on +Z (part frame, Y up).
BODY = {"name": "part_0", "bbox_min": [-1.0, 0.0, -1.0], "bbox_max": [1.0, 2.0, 1.0]}
LID = {"name": "part_1", "bbox_min": [-1.1, 2.0, -1.1], "bbox_max": [1.1, 2.2, 1.0]}
WRONG = {"axis": [1, 0, 0], "pivot": [0.0, 2.1, -1.1]}  # front edge
RIGHT = {"axis": [1, 0, 0], "pivot": [0.0, 2.1, 1.0]}  # flush edge


def _run(tmp_path):
    parts_dir, run = tmp_path / "parts", tmp_path / "run"
    (parts_dir / "parts").mkdir(parents=True)
    run.mkdir()
    parts = []
    for part in (BODY, LID):
        lo, hi = np.array(part["bbox_min"]), np.array(part["bbox_max"])
        box = trimesh.creation.box(extents=hi - lo)
        box.apply_translation((lo + hi) / 2)
        path = parts_dir / "parts" / f"{part['name']}.glb"
        box.export(path)
        parts.append(dict(part, mesh=str(path), face_share=0.5))
    (parts_dir / "parts.json").write_text(
        json.dumps(
            {"source_mesh": "object_1.glb", "parent_object": "bin", "parts": parts}
        )
    )
    (run / "bin_score.json").write_text(
        json.dumps(
            {"icp": {"objects": [{"object": 1, "mesh_extent_m": [0.11, 0.11, 0.11]}]}}
        )
    )
    return parts_dir, run


class Client:
    """Proposes ``hinges`` in turn; the critic says ``critic``."""

    def __init__(self, hinges, critic="yes"):
        self.hinges, self.critic, self.prompts = list(hinges), critic, []

    def generate_images(self, prompt, images=(), search=False):
        self.prompts.append(prompt)
        if "Does the part open" in prompt or "Does the part slide" in prompt:
            text = json.dumps(
                {"verdict": self.critic, "reason": "the lid hinges at the back"}
            )
        else:
            hinge = self.hinges.pop(0) if len(self.hinges) > 1 else self.hinges[0]
            text = json.dumps(
                {
                    "joints": [
                        dict(
                            hinge,
                            type="revolute",
                            parent="part_0",
                            child="part_1",
                            lower=0.0,
                            upper=1.57,
                            rationale="lid",
                        )
                    ]
                }
            )
        return {"text": text, "http_status": 200, "model_id": "test-model"}


def _drive(links, out):
    out.mkdir(parents=True, exist_ok=True)
    view = out / "open_view_0.png"
    view.write_bytes(b"\x89PNG\r\n\x1a\n")
    return {"decision": "accept", "reason": "swept", "open_views": [str(view)]}


def test_qa_rejects_the_wrong_edge_and_the_revised_proposal_is_accepted(tmp_path):
    parts_dir, run = _run(tmp_path)
    client = Client([WRONG, RIGHT])
    record = loop.run_loop(
        parts_dir, run, 1, tmp_path / "out", client, drive=_drive, frames=[b"a", b"b"]
    )
    assert record["decision"] == "articulated" and record["round"] == 1
    first, second = record["rounds"]
    assert first["decision"] == "rejected by QA"
    assert "flush on +Z" in first["qa"]["problems"][0]
    assert "flush on +Z" in second["notes_in"]  # the proposer saw why
    assert "flush on +Z" in client.prompts[1]
    assert second["critic"]["verdict"] == "yes"
    assert (tmp_path / "out" / loop.LOOP).is_file()


def test_a_proposer_that_never_fixes_it_leaves_the_object_rigid(tmp_path):
    parts_dir, run = _run(tmp_path)
    record = loop.run_loop(
        parts_dir,
        run,
        1,
        tmp_path / "out",
        Client([WRONG]),
        drive=_drive,
        frames=[b"a", b"b"],
        max_revisions=2,
    )
    assert record["decision"] == "rigid" and len(record["rounds"]) == 3
    assert all(r["decision"] == "rejected by QA" for r in record["rounds"])


def test_a_critic_no_sends_its_reason_back_and_fails_closed(tmp_path):
    parts_dir, run = _run(tmp_path)
    client = Client([RIGHT], critic="no")
    record = loop.run_loop(
        parts_dir,
        run,
        1,
        tmp_path / "out",
        client,
        drive=_drive,
        frames=[b"a", b"b"],
        max_revisions=1,
    )
    assert record["decision"] == "rigid"
    assert record["rounds"][1]["notes_in"].startswith("a reviewer compared")


def test_the_flush_rule_abstains_without_clear_evidence_and_ignores_door_hinges():
    assert (
        joint_qa.flush_side({"-X": 0.005, "+X": 0.005, "-Z": 0.005, "+Z": 0.0005})
        == "+Z"
    )
    assert (
        joint_qa.flush_side({"-X": 0.0, "+X": 0.0, "-Z": 0.0, "+Z": 0.0}) is None
    )  # no lip
    assert (
        joint_qa.flush_side({"-X": 0.005, "+X": 0.0025, "-Z": 0.005, "+Z": 0.0}) is None
    )  # unclear
    lid = (np.array(LID["bbox_min"]), np.array(LID["bbox_max"]))
    assert (
        joint_qa.hinge_side([0, 1, 0], [1.1, 2.1, 0.0], lid) is None
    )  # vertical: a door
    assert joint_qa.hinge_side([1, 0, 0], [0.0, 2.1, 1.0], lid)[0] == "+Z"


def test_frames_are_the_largest_masks_spread_over_the_video(tmp_path):
    from PIL import Image

    run = tmp_path / "run"
    (run / "masks").mkdir(parents=True)
    (run / "images").mkdir()
    masks = []
    for frame, size in ((1, 10), (11, 40), (41, 30), (81, 20), (91, 35)):
        m = np.zeros((1, 64, 64), bool)
        m[0, :size, :size] = True
        np.savez(run / "masks" / f"{frame:04d}.npz", masks=m)
        Image.new("RGB", (64, 64), "red").save(run / "images" / f"{frame:04d}.jpg")
        masks.append({"view": f"{frame:04d}.jpg", "mask_index": 0, "phrase": "bin"})
    (run / "association_evidence.json").write_text(
        json.dumps({"object_list": [{"object": 1, "masks": masks}]})
    )
    crops = loop.object_frames(run, 1, count=3)
    # Largest first, 30 frames apart: 11 (40), 91 (35), 41 (30); 81 is too close to 91.
    assert len(crops) == 3 and all(c[:2] == b"\xff\xd8" for c in crops)


class Silent(Client):
    """Times out ``silent`` times before answering like ``Client``."""

    def __init__(self, hinges, silent):
        super().__init__(hinges)
        self.silent = silent

    def generate_images(self, prompt, images=(), search=False):
        if self.silent > 0:
            self.silent -= 1
            return {"text": "", "http_status": 0, "model_id": None}
        return super().generate_images(prompt, images, search)


def test_a_model_that_did_not_answer_does_not_use_up_a_revision(tmp_path):
    parts_dir, run = _run(tmp_path)
    # Two calls time out (one round each), then the wrong edge, then the right one.
    client = Silent([WRONG, RIGHT], silent=2)
    record = loop.run_loop(
        parts_dir,
        run,
        1,
        tmp_path / "out",
        client,
        drive=_drive,
        frames=[b"a", b"b"],
        max_revisions=1,
    )
    assert record["decision"] == "articulated"
    decisions = [r["decision"] for r in record["rounds"]]
    # Each timed-out round ends at once; neither uses the single revision allowed.
    assert all(d.startswith("model call failed") for d in decisions[:2])
    assert decisions[2:] == ["rejected by QA", "accepted"]


def test_a_model_that_never_answers_leaves_the_object_rigid(tmp_path):
    parts_dir, run = _run(tmp_path)
    record = loop.run_loop(
        parts_dir,
        run,
        1,
        tmp_path / "out",
        Silent([RIGHT], silent=99),
        drive=_drive,
        frames=[b"a", b"b"],
    )
    assert record["decision"] == "rigid" and "did not answer" in record["reason"]
    assert len(record["rounds"]) == loop.MAX_CALL_FAILURES + 1


def test_the_critic_is_asked_about_the_joint_type_it_sees():
    assert "edge" in loop.critic_question("revolute", "bin", 3, 3)
    question = loop.critic_question("prismatic", "marker", 3, 3)
    assert "slide" in question and "slid out" in question and "hinge" not in question
