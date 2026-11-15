# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SF-24: Gemini plans on the runner; the box runs the committed plan.

The harness is a fake; nothing opens a socket.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sceneagent import plan_runner as pr
from sceneagent import video_run as vr
from sceneagent.agent_loop import ALLOWED_STEPS, default_plan
from sceneagent.gemini_client import active_model_id
from sceneagent.orchestrator import InvalidPlanError, later_conversion

# A valid order that is not default_plan(): reconstruction before the vote.
ORDER = ["sam3d", "icp", "partsam", "articulate", "material_vote"]


def _reply(steps=ORDER, procedures=None):
    if procedures is None:
        procedures = [{"stage": s, "markdown": f"Run {s} and check it."} for s in steps]
    return json.dumps({"steps": steps, "procedures": procedures})


class FakeHarness:
    def __init__(self, text):
        self.text = text
        self.call_count = 0
        self.calls = []
        self.name = active_model_id()
        self.model_id = self.name

    def complete(self, prompt, images=(), response_schema=None):
        self.call_count += 1
        self.calls.append((prompt, response_schema))
        return self.text


def test_the_model_order_is_kept_and_the_call_is_schema_constrained():
    harness = FakeHarness(_reply())
    plan, transcript = pr.make_plan(harness, "42")
    assert [s["id"] for s in plan["steps"]] == ORDER != default_plan()[: len(ORDER)]
    assert plan["forced"] is False and plan["plan_source"] == "harness"
    assert plan["vlm_calls"] == 1 and transcript["vlm_calls"] == 1
    assert plan["prompt_sha256"] == pr.prompt_sha256()
    assert harness.calls[0] == (pr.PLAN_PROMPT, pr.PLAN_SCHEMA)
    assert plan["procedure_source"] == {s: "harness" for s in ORDER}
    assert "Codex" in plan["harness_note"]


def test_the_schema_vocabulary_is_the_allowed_stages():
    steps = pr.PLAN_SCHEMA["properties"]["steps"]["items"]["enum"]
    assert steps == list(ALLOWED_STEPS)
    for stage, deps in pr.STAGE_REQUIRES.items():
        for dep in deps:
            assert f"{stage} after" in pr.PLAN_PROMPT and dep in pr.PLAN_PROMPT


@pytest.mark.parametrize(
    "steps",
    [
        [],
        ["sam3d", "sam3d"],
        ["icp", "sam3d"],  # icp needs sam3d first
        ["sam3d", "teleport"],
    ],
)
def test_a_bad_order_is_refused_not_reordered(steps):
    with pytest.raises(InvalidPlanError):
        pr.make_plan(FakeHarness(_reply(steps, procedures=[])))


def test_stray_or_duplicate_procedures_are_refused():
    stray = [{"stage": "cubepart", "markdown": "x"}]
    with pytest.raises(InvalidPlanError, match="does not run"):
        pr.parse_plan_reply(_reply(["sam3d"], stray))
    twice = [{"stage": "sam3d", "markdown": "a"}, {"stage": "sam3d", "markdown": "b"}]
    with pytest.raises(InvalidPlanError, match="two procedures"):
        pr.parse_plan_reply(_reply(["sam3d"], twice))
    with pytest.raises(InvalidPlanError, match="not JSON"):
        pr.parse_plan_reply("first sam3d, then icp")


def test_a_missing_procedure_is_recorded_as_missing():
    plan, _ = pr.make_plan(FakeHarness(_reply(["sam3d"], procedures=[])))
    assert plan["procedure_source"] == {"sam3d": "missing"}


def test_a_figure5_harness_is_not_called():
    harness = FakeHarness(_reply())
    harness.name = "Grok CLI"
    with pytest.raises(RuntimeError, match="figure 5"):
        pr.make_plan(harness)
    assert harness.call_count == 0


def _committed(tmp_path, text=None):
    plan, _ = pr.make_plan(FakeHarness(text or _reply()))
    path = tmp_path / "harness_plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path, plan


def test_the_box_refuses_a_plan_made_with_another_prompt_or_a_bad_order(tmp_path):
    path, plan = _committed(tmp_path)
    assert [s["id"] for s in pr.load_plan(path)["steps"]] == ORDER
    for change in (
        {"prompt_sha256": "0" * 64},
        {"forced": True},
        {"plan_source": "default_fallback"},
        {"steps": [{"id": "icp"}, {"id": "sam3d"}]},
    ):
        path.write_text(json.dumps({**plan, **change}), encoding="utf-8")
        with pytest.raises(InvalidPlanError):
            pr.load_plan(path)


def test_the_box_runs_the_order_and_writes_each_procedure_after_its_stage(tmp_path):
    path, _ = _committed(tmp_path)
    seen = []

    def tool(name, ok=True):
        def run(context):
            # Each earlier stage's procedure already exists when this one runs.
            seen.append(
                (name, sorted(p.stem for p in (out / "procedures").glob("*.md")))
            )
            return {
                "ok": ok,
                "artifact": None,
                "reason": "" if ok else "bad mesh",
                "detail": {},
            }

        return run

    out = tmp_path / "run"
    tools = {s: tool(s) for s in ("sam3d", "icp", "partsam", "material_vote")}
    tools["icp"] = tool("icp", ok=False)
    tools["icp_repair"] = tool("icp_repair")
    summary = pr.run_plan(out, pr.load_plan(path), tools)
    assert summary["steps"] == ORDER
    assert seen[0] == ("sam3d", [])
    assert seen[1] == ("icp", ["sam3d"])
    ledger = json.loads((out / "ledger.json").read_text(encoding="utf-8"))["entries"]
    assert [e["item"] for e in ledger] == [
        "sam3d",
        "icp",
        "icp",
        "partsam",
        "articulate",
        "material_vote",
    ]
    articulate = next(e for e in ledger if e["item"] == "articulate")
    assert (
        articulate["reason"] == "no tool registered"
        and articulate["decision"] != "accept"
    )
    icp_md = (out / "procedures" / "icp.md").read_text(encoding="utf-8")
    assert (
        "Run icp and check it." in icp_md
        and "(repair)" in icp_md
        and "bad mesh" in icp_md
    )
    plan_json = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    assert "procedures" not in plan_json and plan_json["plan_source"] == "harness"
    later = later_conversion(out)
    assert later["order"] == ORDER and set(later["procedures"]) == set(ORDER)


def test_the_cli_writes_plan_and_transcript(tmp_path, capsys):
    harness = FakeHarness(_reply())
    assert (
        pr.main(
            ["plan", "--out-dir", str(tmp_path), "--source-run", "7"], harness=harness
        )
        == 0
    )
    plan = json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))
    transcript = json.loads((tmp_path / "transcript.json").read_text(encoding="utf-8"))
    assert plan["source_run"] == "7" and transcript["reply"] == harness.text
    assert json.loads(capsys.readouterr().out)["vlm_calls"] == 1
    with pytest.raises(SystemExit):
        pr.main(
            ["plan", "--out-dir", str(tmp_path), "--source-run", "../x"],
            harness=harness,
        )


def _box_ledger(tmp_path, **override):
    """A finished video_run ledger: the item names video_run.STAGES writes."""
    entries = [
        {"item": item, "decision": "accept"} for _, items in vr.STAGES for item in items
    ]
    for item, decision in override.items():
        entries.append({"item": item, "decision": decision, "reason": "too far"})
    path = tmp_path / "box" / "ledger.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    return path


def test_every_evidence_item_is_one_the_video_run_writes():
    written = {item for _, items in vr.STAGES for item in items}
    assert set(pr.LEDGER_EVIDENCE) <= set(ALLOWED_STEPS)
    for items in pr.LEDGER_EVIDENCE.values():
        assert items and set(items) <= written
    assert pr.LEDGER_EVIDENCE["sam3d"] == ("SAM 3D meshes",)
    assert pr.LEDGER_EVIDENCE["icp"] == ("ICP residuals",)
    assert pr.LEDGER_EVIDENCE["isaac_mesh"] == ("USD physics export",)


def test_ledger_tools_accept_only_what_the_box_accepted(tmp_path):
    # The later ICP line is the decision, as video_run.accepted reads it.
    tools = pr.ledger_tools(_box_ledger(tmp_path, **{"ICP residuals": "reject"}))
    assert set(tools) == set(ALLOWED_STEPS)
    sam3d = tools["sam3d"]({})
    assert sam3d["ok"] and "'SAM 3D meshes' accept" in sam3d["reason"]
    icp = tools["icp"]({})
    assert not icp["ok"] and "'ICP residuals' is 'reject' (too far)" in icp["reason"]
    partsam = tools["partsam"]({})
    assert not partsam["ok"] and "no box stage produces partsam" in partsam["reason"]


def test_a_missing_evidence_item_is_a_reject(tmp_path):
    path = tmp_path / "ledger.json"
    path.write_text(
        json.dumps({"entries": [{"item": "inspection views", "decision": "accept"}]}),
        encoding="utf-8",
    )
    result = pr.ledger_tools(path)["isaac_inspector"]({})
    assert not result["ok"]
    assert "'Isaac Lab settle' missing from the ledger" in result["reason"]
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="not a ledger"):
        pr.ledger_tools(path)


def test_the_cli_replays_a_ledger_and_writes_only_the_planned_procedures(
    tmp_path, capsys
):
    path, _ = _committed(tmp_path)
    ledger = _box_ledger(tmp_path)
    out = tmp_path / "run"
    assert (
        pr.main(
            ["run", "--plan", str(path), "--out", str(out), "--ledger", str(ledger)]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["steps"] == ORDER
    decisions = {
        e["item"]: e["decision"]
        for e in json.loads((out / "ledger.json").read_text(encoding="utf-8"))[
            "entries"
        ]
    }
    assert decisions == {
        "sam3d": "accept",
        "icp": "accept",
        "partsam": "reject",
        "articulate": "reject",
        "material_vote": "accept",
    }
    # Procedures for the plan's stages only, not for every allowed stage.
    assert sorted(p.stem for p in (out / "procedures").glob("*.md")) == sorted(ORDER)
    md = (out / "procedures" / "articulate.md").read_text(encoding="utf-8")
    assert "Run articulate and check it." in md and "no box stage produces" in md
    # The source ledger is read, never appended to.
    assert "partsam" not in ledger.read_text(encoding="utf-8")
    with pytest.raises(SystemExit):
        pr.main(
            [
                "run",
                "--plan",
                str(path),
                "--out",
                str(ledger.parent),
                "--ledger",
                str(ledger),
            ]
        )


def test_the_committed_plan_is_the_harness_plan_and_loads():
    plan = pr.load_plan(Path(pr.__file__).resolve().parents[1] / pr.PLAN_RELATIVE)
    assert plan["plan_source"] == "harness" and plan["forced"] is False
    assert plan["vlm_calls"] == 1 and plan["steps"]
