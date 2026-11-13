# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KAN-52: dc1, dc2, dc4, dc6 and dc8 from one run directory."""

from __future__ import annotations

import json
from pathlib import Path

from sceneagent import mvp_check
from sceneagent.physics import UNPUBLISHED_LOOKUPS
from sceneagent.video_run import STAGES

# Run 43359257095's report on Mailbox: the model replied, the scene was clean.
CLEAN_REPORT = {
    "model": "gemini-3.8-flash",
    "model_replied": True,
    "findings": [],
    "summary": "The mailbox is properly placed and has settled stably.",
    "fix_report_from_model": False,
    "vlm_calls": 1,
}


def _run(tmp_path, decision="accept", report=CLEAN_REPORT):
    (tmp_path / "settle.json").write_text(
        json.dumps({"decision": decision, "reason": "1 objects at rest after 2 s"})
    )
    if report is not None:
        (tmp_path / "isaac_lab.json").write_text(json.dumps(report))
    return tmp_path


def test_a_clean_scene_with_a_model_written_report_meets_dc4(tmp_path):
    row = mvp_check.check_dc4(_run(tmp_path))
    # fix_report_from_model is false (no fixes), and dc4 is still met.
    assert row["status"] == mvp_check.MET
    assert row["fix_report_from_model"] is False and row["findings"] == 0


def test_a_report_with_findings_also_meets_dc4(tmp_path):
    report = dict(
        CLEAN_REPORT, findings=[{"object": "x", "issue": "missing_object", "fix": "f"}]
    )
    report["fix_report_from_model"] = True
    assert mvp_check.check_dc4(_run(tmp_path, report=report))["status"] == mvp_check.MET


def test_no_reply_or_an_error_is_not_a_model_written_report(tmp_path):
    silent = dict(CLEAN_REPORT, model_replied=False)
    assert (
        mvp_check.check_dc4(_run(tmp_path, report=silent))["status"] == mvp_check.PARTLY
    )
    failed = dict(CLEAN_REPORT, error="ValueError: bad json")
    assert (
        mvp_check.check_dc4(_run(tmp_path, report=failed))["status"] == mvp_check.PARTLY
    )


def test_an_unsettled_scene_or_no_report(tmp_path):
    assert (
        mvp_check.check_dc4(_run(tmp_path, decision="reject"))["status"]
        == mvp_check.PARTLY
    )
    empty = tmp_path / "empty"
    empty.mkdir()
    assert mvp_check.check_dc4(empty)["status"] == mvp_check.NOT_MET


def test_the_run_check_lists_every_mvp_criterion(tmp_path):
    rows = mvp_check.check_run(_run(tmp_path))
    assert [r["criterion"] for r in rows] == ["dc1", "dc2", "dc4", "dc6", "dc8"]
    assert rows[2]["status"] == mvp_check.MET
    assert mvp_check.main(["--run", str(tmp_path)]) == 1


def _ledger(run):
    items = [item for _, stage_items in STAGES for item in stage_items]
    (run / "ledger.json").write_text(
        json.dumps(
            {"entries": [{"item": item, "decision": "accept"} for item in items]}
        )
    )


def _verdicts(directory, verdict="accept"):
    (directory / "step_verdicts.json").write_text(
        json.dumps(
            {
                "mode": "step",
                "stages": [{"stage": "masks", "verdict": verdict, "reason": "ok"}],
            }
        )
    )


def _physics(number, prior=False):
    record = {
        "object": f"object_{number}",
        "section": "3.4",
        "sources": {},
    }
    for key in mvp_check.SECTION_34_FIELDS:
        record[key] = None
        record["sources"][key] = "test"
    if prior:
        record["prior"] = {"sha256": "a" * 40}
    return record


def test_dc1_needs_the_ledger_and_accepted_verdicts(tmp_path):
    assert mvp_check.check_dc1(tmp_path)["status"] == mvp_check.NOT_MET
    _ledger(tmp_path)
    assert mvp_check.check_dc1(tmp_path)["status"] == mvp_check.PARTLY
    review = tmp_path / "review"
    review.mkdir()
    _verdicts(review, verdict="reject")
    assert mvp_check.check_dc1(tmp_path, review)["status"] == mvp_check.PARTLY
    _verdicts(review)
    row = mvp_check.check_dc1(tmp_path, review)
    assert row["status"] == mvp_check.MET
    assert row["files"][1].endswith("review/step_verdicts.json")


def test_dc2_is_partly_met_without_a_prior_and_met_with_one(tmp_path):
    (tmp_path / "export_evidence.json").write_text(
        json.dumps({"objects": [{"object": "object_1"}, {"object": "object_2"}]})
    )
    assert mvp_check.check_dc2(tmp_path)["status"] == mvp_check.NOT_MET
    folder = tmp_path / "physics"
    folder.mkdir()
    (folder / "object_1.json").write_text(json.dumps(_physics(1)))
    (folder / "object_2.json").write_text(json.dumps(_physics(2)))
    assert mvp_check.check_dc2(tmp_path)["status"] == mvp_check.PARTLY
    (folder / "object_1.json").write_text(json.dumps(_physics(1, prior=True)))
    (folder / "object_2.json").write_text(json.dumps(_physics(2, prior=True)))
    row = mvp_check.check_dc2(tmp_path)
    assert row["status"] == mvp_check.MET and row["with_prior"] == 2
    fallback = _physics(2, prior=True)
    fallback["plan_source"] = "fallback"
    (folder / "object_2.json").write_text(json.dumps(fallback))
    row = mvp_check.check_dc2(tmp_path)
    assert row["status"] == mvp_check.MET and row["with_prior"] == 1
    assert set(UNPUBLISHED_LOOKUPS).issubset(mvp_check.SECTION_34_FIELDS)


def test_dc6_reads_the_exported_scene_state(tmp_path):
    assert mvp_check.check_dc6(tmp_path)["status"] == mvp_check.NOT_MET
    path = tmp_path / "mailbox_scene_state_exported.json"
    path.write_text(json.dumps({"objects_info": {}}))
    assert mvp_check.check_dc6(tmp_path)["status"] == mvp_check.PARTLY
    path.write_text(
        json.dumps(
            {
                "state": {},
                "physics_records": {"cube": {"mass_kg": 0.1}},
                "objects_info": {
                    "init_info": {
                        "cube": {"args": {"link_physics_materials": {"link": {}}}}
                    }
                },
            }
        )
    )
    row = mvp_check.check_dc6(tmp_path)
    assert row["status"] == mvp_check.MET
    assert row["files"] == [str(path)]


def test_dc8_reads_the_commit_that_ran(tmp_path):
    assert mvp_check.check_dc8(tmp_path)["status"] == mvp_check.NOT_MET
    sha = "69d54530e959584e4aff8f6021afeedc1091a5ed"
    (tmp_path / "manifest.json").write_text(json.dumps({"commit_sha": sha}))
    row = mvp_check.check_dc8(tmp_path)
    assert row["status"] == mvp_check.MET and row["commit_sha"] == sha
    (tmp_path / "manifest.json").unlink()
    (tmp_path / "checked_out_commit.txt").write_text(sha + "\n")
    assert mvp_check.check_dc8(tmp_path)["commit_sha"] == sha


def test_a_complete_run_exits_zero(tmp_path):
    _run(tmp_path)
    _ledger(tmp_path)
    _verdicts(tmp_path)
    (tmp_path / "export_evidence.json").write_text(
        json.dumps({"objects": [{"object": "object_0"}]})
    )
    folder = tmp_path / "physics"
    folder.mkdir()
    (folder / "object_0.json").write_text(json.dumps(_physics(0, prior=True)))
    (tmp_path / "scene_scene_state_exported.json").write_text(
        json.dumps(
            {
                "state": {},
                "physics_records": {"o": {}},
                "objects_info": {
                    "init_info": {"o": {"args": {"link_physics_materials": {}}}}
                },
            }
        )
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"commit_sha": "a" * 40})
    )
    assert mvp_check.main(["--run", str(tmp_path)]) == 0


def test_mvp_review_workflow_keeps_the_key_on_one_step():
    text = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/sceneagent-mvp-review.yml"
    ).read_text(encoding="utf-8")
    assert "workflow_dispatch" in text and "smoke_run" in text
    assert text.count("secrets.GEMINI_API_KEY") == 1
    assert "VAST" not in text
    assert "sceneagent.step_critic judge" in text
    assert "sceneagent.plan_runner plan" in text
    assert "sceneagent.inspector report" in text
    assert "name: sceneagent-mvp-review" in text
