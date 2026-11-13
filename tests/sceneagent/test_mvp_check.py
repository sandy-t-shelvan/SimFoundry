# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KAN-52 dc4, option A: settled, and a report the model wrote, findings or not."""

from __future__ import annotations

import json

from sceneagent import mvp_check

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
    assert {r["status"] for r in rows if r["criterion"] != "dc4"} == {"not yet checked"}
    assert mvp_check.main(["--run", str(tmp_path)]) == 0
