"""SF-24: a harness plan is ordered by the model, and an invalid plan raises.

The harness is a fake or a Gemini adapter with a fake transport. No test
opens a socket.
"""

import json
import logging
from pathlib import Path

import pytest

from sceneagent.agent_loop import ALLOWED_STEPS, default_plan, harness_status
from sceneagent.fruits_agent import plan_with_client, run_fruits
from sceneagent.harness import (
    GEMINI_MODEL_ID,
    GeminiHarness,
    HarnessCallError,
    MissingGeminiApiKeyError,
)
from sceneagent.orchestrator import (
    PLAN_SOURCE_DEFAULT_FALLBACK,
    PLAN_SOURCE_HARNESS,
    STAGE_REQUIRES,
    InvalidPlanError,
    counts_toward_dc2,
    later_conversion,
    run_harness_plan,
    stamp_plan_source,
)

SECRET = "AQ.FAKE-TEST-KEY-123"
# A valid order that is not default_plan(): reconstruction before the vote.
MODEL_ORDER = [
    "sam3d",
    "icp",
    "isaac_mesh",
    "mujoco_mesh",
    "partsam",
    "cubepart",
    "articulate",
    "isaac_inspector",
    "material_vote",
    "mujoco_load",
]


class _Harness:
    """Stand-in for the Gemini adapter. ``complete`` does not touch the network."""

    name = GEMINI_MODEL_ID
    model_id = GEMINI_MODEL_ID

    def __init__(self, text):
        self.text = text
        self.call_count = 0
        self.prompts = []

    def complete(self, prompt):
        self.call_count += 1
        self.prompts.append(prompt)
        return self.text


def _reply(steps, procedures=None):
    bodies = procedures
    if bodies is None:
        bodies = {step: f"Procedure for {step}. Later conversions read this file." for step in steps}
    return json.dumps({"steps": steps, "procedures": bodies})


def _tools(steps, calls):
    tools = {}
    for step_id in steps:

        def _tool(_ctx, name=step_id):
            calls.append(name)
            return {"ok": True, "artifact": f"{name}.txt", "reason": "", "detail": {}}

        tools[step_id] = _tool
    return tools


def _run(tmp_path, steps=None):
    chosen = list(MODEL_ORDER if steps is None else steps)
    harness = _Harness(_reply(chosen))
    calls = []
    summary = run_harness_plan(tmp_path, harness=harness, tools=_tools(chosen, calls))
    return harness, calls, summary


def _read(path):
    return path.read_text(encoding="utf-8")


def test_dependencies_cover_the_allowed_stages():
    assert set(STAGE_REQUIRES) == set(ALLOWED_STEPS)
    assert MODEL_ORDER != list(default_plan())
    assert MODEL_ORDER[0] != default_plan()[0]


def test_model_chosen_order_is_written_to_plan_json(tmp_path):
    harness, calls, summary = _run(tmp_path)
    plan = json.loads(_read(tmp_path / "plan.json"))
    written = [step["id"] for step in plan["steps"]]
    assert written == MODEL_ORDER
    assert written != list(default_plan())
    assert summary["forced_order"] == list(default_plan())
    assert plan["order_source"] == "harness"
    assert plan["plan_source"] == PLAN_SOURCE_HARNESS
    assert plan["forced"] is False
    eligible = counts_toward_dc2(plan)
    assert eligible["counts"] is True
    assert "not the live" in eligible["reason"]
    assert plan["harness"] == GEMINI_MODEL_ID
    assert calls == MODEL_ORDER
    assert harness.call_count == 1
    ledger = json.loads(_read(tmp_path / "ledger.json"))
    items = [entry["item"] for entry in ledger["entries"]]
    assert items[0] == "plan"
    assert "ingest" in items
    assert items.index("ingest") < items.index("sam3d")
    assert items.index("plan") < items.index("ingest")


def test_transcript_artifact_exists(tmp_path):
    _run(tmp_path)
    path = tmp_path / "transcript.json"
    assert path.is_file()
    body = json.loads(_read(path))
    assert body["harness"] == GEMINI_MODEL_ID
    assert body["call_count"] == 1
    assert "sam3d" in body["reply"]
    assert body["reply"].find("sam3d") < body["reply"].find("material_vote")


def test_manifest_records_the_per_run_call_count(tmp_path):
    harness, _calls, summary = _run(tmp_path)
    manifest = json.loads(_read(tmp_path / "manifest.json"))
    assert manifest["call_count"] == 1
    assert manifest["call_count"] == harness.call_count == summary["call_count"]
    assert manifest["steps"] == MODEL_ORDER
    second = _Harness(_reply(MODEL_ORDER))
    second.call_count = 4
    again = run_harness_plan(tmp_path / "next", harness=second)
    saved = json.loads(_read(tmp_path / "next" / "manifest.json"))
    assert second.call_count == 5
    assert saved["call_count"] == 1
    assert saved["call_count"] == again["call_count"]


@pytest.mark.parametrize(
    "steps",
    [
        ["partsam", "sam3d"],
        ["sam3d", "sam3d"],
        ["sam3d", "not-a-block"],
        [],
    ],
)
def test_invalid_plan_raises_and_does_not_fall_back(tmp_path, steps):
    harness = _Harness(_reply(steps) if steps else '{"steps": []}')
    with pytest.raises(InvalidPlanError):
        run_harness_plan(tmp_path, harness=harness)
    assert harness.call_count == 1
    assert not (tmp_path / "plan.json").exists()
    assert not (tmp_path / "procedures").exists()
    assert list(default_plan()) != steps


def test_dependency_violation_raises_invalid_plan(tmp_path):
    """A plan that runs PartSAM before SAM 3D is invalid. No forced order is written."""

    harness = _Harness(_reply(["partsam", "sam3d"]))
    with pytest.raises(InvalidPlanError, match="partsam requires sam3d"):
        run_harness_plan(tmp_path, harness=harness)
    assert harness.call_count == 1
    assert not (tmp_path / "plan.json").exists()


class _Flash:
    def __init__(self, text):
        self.text = text

    def generate_flash(self, prompt, image_bytes=None):
        return {
            "text": self.text,
            "model_id": "fake",
            "http_status": 200,
            "used_paper_model": False,
            "paper_model_id": "gemini-2.5-flash",
        }


def test_fallback_plan_is_recorded_and_rejected_as_dc2(tmp_path):
    empty = plan_with_client(_Flash(""))
    assert empty["used_fallback"] is True
    assert empty["fallback"] == "default_plan"
    assert empty["plan_source"] == PLAN_SOURCE_DEFAULT_FALLBACK
    assert empty["steps"] == list(default_plan())
    rejected = counts_toward_dc2(empty)
    assert rejected["counts"] is False
    assert rejected["plan_source"] == PLAN_SOURCE_DEFAULT_FALLBACK
    folder = tmp_path / "fallback"
    folder.mkdir()
    (folder / "plan.json").write_text(
        json.dumps({"section": "3.1", "steps": empty["steps"]}) + "\n",
        encoding="utf-8",
    )
    stamped = stamp_plan_source(folder, empty["plan_source"])
    assert stamped["plan_source"] == PLAN_SOURCE_DEFAULT_FALLBACK
    on_disk = json.loads(_read(folder / "plan.json"))
    assert counts_toward_dc2(on_disk)["counts"] is False
    chosen = plan_with_client(_Flash('{"steps": ["sam3d", "icp"]}'))
    assert chosen["used_fallback"] is False
    assert chosen["plan_source"] == PLAN_SOURCE_HARNESS
    assert counts_toward_dc2(chosen)["counts"] is True
    assert counts_toward_dc2({})["counts"] is False


def _fruits_paths(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"\xff\xd8")
    return {
        "frame": frame,
        "vote": tmp_path / "material_vote.json",
        "usda": tmp_path / "scene.usda",
        "contract": tmp_path / "contract.json",
        "mjcf": tmp_path / "scene.xml",
        "mujoco_report": tmp_path / "mujoco.json",
        "sam3d_dir": tmp_path / "sam3d",
    }


def _loader(path, steps=30):
    return {"ok": True, "artifact": str(path), "reason": "", "detail": {"nbody": 1}}


def _run_saved_fruits(tmp_path, reply):
    """The real ``run_fruits`` path. The client never opens a socket."""

    out = tmp_path / "run"
    run_fruits(out, _Flash(reply), _fruits_paths(tmp_path), 1.0, loader=_loader)
    return json.loads(_read(out / "plan.json"))


def test_run_fruits_saves_a_model_plan_that_is_dc2_eligible(tmp_path):
    plan = _run_saved_fruits(tmp_path, '{"steps": ["sam3d", "icp"]}')
    assert [step["id"] for step in plan["steps"]] == ["sam3d", "icp"]
    assert [step["id"] for step in plan["steps"]] != list(default_plan())
    assert plan["plan_source"] == PLAN_SOURCE_HARNESS
    verdict = counts_toward_dc2(plan)
    assert verdict["counts"] is True
    assert verdict["plan_source"] == PLAN_SOURCE_HARNESS


def test_run_fruits_saves_a_fallback_plan_that_is_not_dc2_eligible(tmp_path):
    plan = _run_saved_fruits(tmp_path, "")
    assert [step["id"] for step in plan["steps"]] == list(default_plan())
    assert plan["plan_source"] == PLAN_SOURCE_DEFAULT_FALLBACK
    verdict = counts_toward_dc2(plan)
    assert verdict["counts"] is False
    assert verdict["plan_source"] == PLAN_SOURCE_DEFAULT_FALLBACK


def test_procedures_are_readable_by_a_later_run(tmp_path):
    harness, _calls, _summary = _run(tmp_path)
    later = later_conversion(tmp_path)
    assert later["order"] == MODEL_ORDER
    assert later["order"] != list(default_plan())
    for step_id in MODEL_ORDER:
        text = later["procedures"][step_id]
        assert f"Procedure for {step_id}." in text
        assert "Later conversions read this file." in text
        assert text == _read(tmp_path / "procedures" / f"{step_id}.md")
    assert harness.call_count == 1
    reread = later_conversion(tmp_path)
    assert reread == later
    assert harness.call_count == 1


def test_figure5_harnesses_stay_uncalled(tmp_path):
    before = harness_status()
    _run(tmp_path)
    after = harness_status()
    assert before["called"] == []
    assert after["called"] == []
    assert after["status"] == "still_open"
    assert "Grok CLI" in after["harnesses"]
    assert len(after["harnesses"]) == 6


def test_missing_key_raises_before_any_call(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    called = []

    def transport(url, key, payload):
        called.append(url)
        raise AssertionError("transport was called")

    harness = GeminiHarness(transport=transport)
    with pytest.raises(MissingGeminiApiKeyError) as caught:
        run_harness_plan(tmp_path, harness=harness)
    assert called == []
    assert harness.call_count == 0
    assert "AQ." not in str(caught.value)
    assert not (tmp_path / "plan.json").exists()
    assert not (tmp_path / "transcript.json").exists()
    assert not (tmp_path / "manifest.json").exists()


def test_google_api_key_is_not_a_substitute(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", SECRET)
    called = []

    def transport(url, key, payload):
        called.append(True)
        raise AssertionError("transport was called")

    with pytest.raises(MissingGeminiApiKeyError):
        GeminiHarness(transport=transport).complete("plan")
    assert called == []


def test_fake_key_is_absent_from_logs_exceptions_and_artifacts(tmp_path, monkeypatch, caplog, capsys):
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    caplog.set_level(logging.DEBUG)

    def transport(url, key, payload):
        if key != SECRET:
            raise AssertionError("harness key did not match GEMINI_API_KEY")
        if SECRET in url:
            raise AssertionError("key was placed in the request URL")
        echoed = _reply(
            MODEL_ORDER,
            {step: f"Procedure for {step}. token {SECRET}" for step in MODEL_ORDER},
        )
        body = json.dumps({"candidates": [{"content": {"parts": [{"text": echoed}]}}]})
        return 200, body

    harness = GeminiHarness(transport=transport)
    with caplog.at_level(logging.DEBUG, logger="sceneagent.harness"):
        run_harness_plan(tmp_path, harness=harness)
    blob = "\n".join(_read(path) for path in tmp_path.rglob("*") if path.is_file())
    assert SECRET not in blob
    assert "[redacted]" in _read(tmp_path / "procedures" / "sam3d.md")
    assert SECRET not in caplog.text
    captured = capsys.readouterr()
    assert SECRET not in captured.out
    assert SECRET not in captured.err
    plan = json.loads(_read(tmp_path / "plan.json"))
    manifest = json.loads(_read(tmp_path / "manifest.json"))
    transcript = json.loads(_read(tmp_path / "transcript.json"))
    assert SECRET not in json.dumps(plan)
    assert SECRET not in json.dumps(manifest)
    assert SECRET not in json.dumps(transcript)
    assert manifest["call_count"] == 1
    assert [step["id"] for step in plan["steps"]] == MODEL_ORDER

    def fail_transport(url, key, payload):
        raise RuntimeError(f"socket failed {key}")

    with pytest.raises(HarnessCallError) as caught:
        GeminiHarness(transport=fail_transport).complete("plan")
    assert SECRET not in str(caught.value)
    assert SECRET not in repr(caught.value)
    assert caught.value.__cause__ is None
    context = caught.value.__context__
    assert context is None or SECRET not in str(context)
    assert SECRET not in caplog.text
