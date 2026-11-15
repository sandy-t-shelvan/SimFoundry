# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KAN-42 dc2 and KAN-44: the parts review, the per-part URDF and the worker's manifest."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from sceneagent import articulate_scanned as asc
from sceneagent import parts_review as pr

ROOT = Path(__file__).resolve().parents[2]


class FakeClient:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def generate_images(self, prompt, images=(), search=False):
        self.calls.append((prompt, list(images)))
        return self.reply


def _parts_dir(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    (tmp_path / "parts").mkdir()
    parts = []
    for i, offset in enumerate((0.0, 1.5)):
        box = trimesh.creation.box(extents=(1, 1, 1))
        box.apply_translation((offset, 0, 0))
        path = tmp_path / "parts" / f"part_{i}.glb"
        box.export(path)
        parts.append({"name": f"part_{i}", "mesh": str(path), "face_share": 0.5})
    (tmp_path / "parts.json").write_text(json.dumps({"source_mesh": "m.glb", "parts": parts}))
    return tmp_path


def test_a_yes_reply_asks_for_cubepart_and_names_the_parts(tmp_path):
    reply = {
        "text": '{"completion_needed": true, "reason": "the tip is hidden under the cap", "parts": ["part_1"]}',
        "http_status": 200,
        "model_id": "gemini-3.8-flash",
    }
    client = FakeClient(reply)
    record = pr.review(_parts_dir(tmp_path), "marker", client)
    assert record["completion_needed"] is True and record["call_cubepart"] is True
    assert record["reason"] == "the tip is hidden under the cap"
    assert record["parts_missing_geometry"] == ["part_1"]
    assert record["model"] == "gemini-3.8-flash" and record["vlm_calls"] == 1
    prompt, images = client.calls[0]
    assert "hidden geometry" in prompt and "part_0 (50.0% of faces)" in prompt
    assert len(images) == 1 and images[0][:8] == b"\x89PNG\r\n\x1a\n"
    assert (tmp_path / "parts_review.json").is_file() and Path(record["image_path"]).is_file()


def test_a_no_reply_does_not_call_cubepart(tmp_path):
    reply = {"text": '{"completion_needed": false, "reason": "nothing hidden", "parts": []}', "http_status": 200}
    record = pr.review(_parts_dir(tmp_path), "mailbox", FakeClient(reply))
    assert record["completion_needed"] is False and record["call_cubepart"] is False


def test_a_failed_call_is_recorded_as_failed_not_as_no(tmp_path):
    record = pr.review(_parts_dir(tmp_path), "marker", FakeClient({"text": "", "http_status": 503}))
    assert record["decision"] == "failed" and record["call_failed"] == "HTTP 503"
    assert record["completion_needed"] is None and record["call_cubepart"] is False


PARTS = [{"name": n, "mesh": f"/x/parts/{n}.glb"} for n in ("part_0", "part_1", "part_2", "part_4")]
JOINT = {"type": "revolute", "parent": "part_1", "child": "part_2", "axis": [-1.0, 0.0, 0.0], "origin": [0.0, 0.76, -0.62], "lower": 0.0, "upper": 1.5708}


def test_every_part_gets_a_link_and_the_hinge_plus_fixed_joints_bind_them():
    joints = asc.link_joints(["part_0", "part_1"], ["part_2", "part_4"], JOINT)
    assert [(j["type"], j["parent"], j["child"]) for j in joints] == [
        ("revolute", "part_1", "part_2"),
        ("fixed", "part_1", "part_0"),
        ("fixed", "part_2", "part_4"),
    ]
    urdf = asc.write_urdf("mailbox", PARTS, joints, scale=0.1)
    assert urdf.count("<link name=") == 4 and urdf.count('type="fixed"') == 2
    assert '<origin xyz="0 0.076 -0.062" rpy="0 0 0"/>' in urdf  # the pivot, in metres
    assert '<limit lower="0" upper="1.5708"' in urdf and urdf.count("<limit") == 1
    assert 'filename="parts/part_4.glb" scale="0.1 0.1 0.1"' in urdf


def test_a_part_left_without_a_link_is_refused():
    with pytest.raises(ValueError, match="has no link"):
        asc.write_urdf("mailbox", PARTS, asc.link_joints(["part_1"], ["part_2"], JOINT), scale=0.1)


def test_the_worker_reads_graph_cut_from_partsam_s_config(tmp_path):
    spec = importlib.util.spec_from_file_location("pw", ROOT / "sceneagent" / "partsam_worker.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "partsam.yaml").write_text("eval_params:\n  use_graph_cut: True  # note\n")
    assert worker.graph_cut_used(tmp_path) is True
    (tmp_path / "configs" / "partsam.yaml").write_text("eval_params:\n  use_graph_cut: False\n")
    assert worker.graph_cut_used(tmp_path) is False
    assert worker.graph_cut_used(tmp_path / "missing") is None
