# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KAN-23: the inspector's views of a settled run, and the runner-side report workflow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sceneagent import inspect_run, inspector
from sceneagent.vast_smoke import FETCH_NAMES, PUBLISHED_NAMES

WORKFLOW = (
    Path(__file__).resolve().parents[2]
    / ".github/workflows/sceneagent-inspector-report.yml"
)


def _run_dir(tmp_path: Path) -> Path:
    quarter = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    world = np.eye(4)
    world[:3, :3] = quarter
    world[:3, 3] = [0.0, 0.0, 0.5]
    placed = [
        {
            "object": 1,
            "placed": True,
            "gate": "accept",
            "glb": "/box/sam3d/object_1.glb",
            "similarity": {
                "scale": 2.0,
                "rotation": np.eye(3).tolist(),
                "translation": [1.0, 0.0, 0.0],
            },
        },
        {"object": 2, "placed": True, "gate": "reject"},
    ]
    (tmp_path / "bin_score.json").write_text(json.dumps({"icp": {"objects": placed}}))
    (tmp_path / "association_evidence.json").write_text(
        json.dumps(
            {
                "splat_unit_m": 0.5,
                "object_list": [{"object": 1, "masks": [{"phrase": "mailbox"}]}],
            }
        )
    )
    (tmp_path / "export_evidence.json").write_text(
        json.dumps({"support": {"world_from_scene": world.tolist()}})
    )
    (tmp_path / "settle.json").write_text(
        json.dumps(
            {
                "decision": "accept",
                "reason": "1 objects at rest after 2 s",
                "seconds": 2.0,
                "objects": {"object_1": {"speed_m_s": 0.001, "centre_z_m": 0.09}},
            }
        )
    )
    return tmp_path


def test_the_manifest_holds_gate_passes_in_the_exports_frame(tmp_path):
    scene = inspect_run.manifest(_run_dir(tmp_path))
    assert [o["name"] for o in scene["objects"]] == ["object_1 (mailbox)"]
    item = scene["objects"][0]
    # x_world = W (s R x + t) + w: the origin maps to W t + w, in metres.
    assert np.allclose(item["translation"], [0.0, 1.0, 0.5])
    assert np.allclose(np.asarray(item["rotation"]) @ [1, 0, 0], [0, 1, 0])
    assert item["scale"] == 2.0 and item["metres_per_unit"] == 0.5


def test_the_summary_is_the_settle_that_ran_not_a_second_one(tmp_path):
    summary = inspect_run.settle_summary(_run_dir(tmp_path))
    assert summary["ran"] is True and summary["settled"] is True
    assert summary["bodies"] == [
        {"name": "object_1", "speed_m_s": 0.001, "centre_z_m": 0.09}
    ]


def test_run_inspect_publishes_the_sheet_zip_and_summary(tmp_path):
    out = _run_dir(tmp_path)
    calls = []

    def render(scene, folder):
        calls.append(json.loads(Path(scene).read_text()))
        for name in (inspector.SHEET_NAME, inspector.VIEWS_ZIP):
            (folder / name).write_bytes(b"x")
        summary = {
            "objects": ["object_1 (mailbox)"],
            "views": ["top"],
            "renderer": "cpu",
        }
        return {"settle": summary}

    assert inspect_run.run_inspect(out, render=render) == 0
    assert calls and calls[0]["objects"][0]["name"] == "object_1 (mailbox)"
    for name in (inspector.SHEET_NAME, inspector.VIEWS_ZIP, inspector.SETTLE_NAME):
        assert (out / name).is_file()
        assert name in FETCH_NAMES and name in PUBLISHED_NAMES
    summary = json.loads((out / inspector.SETTLE_NAME).read_text())
    assert summary["settled"] is True and summary["objects"] == ["object_1 (mailbox)"]
    entry = json.loads((out / "ledger.json").read_text())["entries"][-1]
    assert entry["item"] == inspect_run.ITEM and entry["decision"] == "accept"


def test_run_inspect_without_a_gate_pass_rejects(tmp_path):
    out = _run_dir(tmp_path)
    placement = json.loads((out / "bin_score.json").read_text())
    placement["icp"]["objects"][0]["gate"] = "reject"
    (out / "bin_score.json").write_text(json.dumps(placement))
    assert inspect_run.run_inspect(out, render=lambda *a: {}) == 1
    entry = json.loads((out / "ledger.json").read_text())["entries"][-1]
    assert "no exported object" in entry["reason"]


def test_the_report_workflow_is_runner_only_and_keeps_the_key_in_one_step():
    text = WORKFLOW.read_text(encoding="utf-8")
    # The workflow itself, not its header comment (which says what it lacks).
    body = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    assert body.count("secrets.GEMINI_API_KEY") == 1
    for forbidden in ("VAST", "SSH", "ssh-key", "vast_smoke", "secrets.HF_TOKEN"):
        assert forbidden not in body
    assert "python -m sceneagent.inspector report" in text
    assert "persist-credentials: false" in text
