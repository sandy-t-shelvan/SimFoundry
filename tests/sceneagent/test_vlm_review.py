# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SF-19 VLM verdict: the runner half, the box half, and the plumbing between them.

No test calls Gemini. The harness is a fake, or the real adapter with a fake
transport.
"""

from __future__ import annotations

import base64
import json
import re
import zipfile
from pathlib import Path

import pytest

from sceneagent import bin_review as br
from sceneagent import vlm_review as vr
from sceneagent.harness import GeminiHarness

ROOT = Path(__file__).resolve().parents[2]
VERDICT_WORKFLOW = ROOT / ".github" / "workflows" / "sceneagent-vlm-verdict.yml"
GPU_WORKFLOW = ROOT / ".github" / "workflows" / "sceneagent-gpu-smoke.yml"


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


def _zip(tmp_path: Path, names: dict[str, bytes]) -> Path:
    path = tmp_path / "review_sheets.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, blob in names.items():
            archive.writestr(name, blob)
    return path


def test_a_reply_must_name_one_of_the_three_retries():
    assert vr.parse_verdict('{"decision": "icp_flip", "reason": "upside down"}') == {
        "decision": "icp_flip",
        "reason": "upside down",
    }
    fenced = '```json\n{"decision": "keep", "reason": "fits"}\n```'
    assert vr.parse_verdict(fenced)["decision"] == "keep"
    long = json.dumps({"decision": "keep", "reason": "x" * 1000})
    assert len(vr.parse_verdict(long)["reason"]) == vr.REASON_CHARS
    for bad in (
        '{"decision": "flip"}',
        '["keep"]',
        "keep",
        '{"reason": "no decision"}',
    ):
        with pytest.raises(ValueError):
            vr.parse_verdict(bad)


def test_only_object_sheets_are_read_and_nothing_is_extracted(tmp_path):
    path = _zip(
        tmp_path,
        {
            "object_3.png": b"three",
            "object_12.png": b"twelve",
            "../object_4.png": b"escape",
            "notes.txt": b"no",
        },
    )
    assert vr.read_sheets(path) == {3: b"three", 12: b"twelve"}
    assert not (tmp_path.parent / "object_4.png").exists()


def test_each_sheet_is_one_call_and_a_failure_is_recorded_not_guessed():
    harness = FakeHarness(
        [
            '{"decision": "keep", "reason": "red fills green"}',
            RuntimeError("Gemini harness call failed with HTTP 503"),
            '{"decision": "teleport", "reason": "?"}',
        ]
    )
    result = vr.judge({1: b"a", 2: b"b", 5: b"c"}, harness)
    assert result["vlm_calls"] == 3
    assert result["objects"] == [
        {"object": 1, "decision": "keep", "reason": "red fills green"}
    ]
    assert [f["object"] for f in result["failed"]] == [2, 5]
    assert result["prompt_sha256"] == vr.prompt_sha256()
    prompt, images, schema = harness.calls[0]
    assert prompt == vr.PROMPT
    assert images == [(b"a", "image/png")]
    assert schema["properties"]["decision"]["enum"] == list(vr.DECISIONS)


def test_the_prompt_does_not_carry_the_rule_numbers():
    # Gemini's verdict is its own reading of the images.
    for leak in ("IoU", "0.5", "isolated_mean", "overlay_iou", "fit_total"):
        assert leak not in vr.PROMPT
    for view in ("registration_overlay", "axis_pos_x", "oblique_2", "assembled_2"):
        assert view in vr.PROMPT


def test_main_writes_verdicts_with_no_key(tmp_path, monkeypatch, capsys):
    key = "sf19-not-a-real-key"
    monkeypatch.setenv("GEMINI_API_KEY", key)
    seen = []

    def transport(url, sent_key, payload):
        seen.append((url, payload))
        assert key not in url
        reply = {"decision": "icp_flip", "reason": f"mirrored ({sent_key})"}
        body = {"candidates": [{"content": {"parts": [{"text": json.dumps(reply)}]}}]}
        return 200, json.dumps(body)

    sheets = _zip(tmp_path, {"object_0.png": b"\x89PNG0", "object_1.png": b"\x89PNG1"})
    out = tmp_path / "vlm-out" / "vlm_verdicts.json"
    code = vr.main(
        ["--sheets", str(sheets), "--out", str(out), "--source-run", "42934419274"],
        harness=GeminiHarness(transport=transport),
    )
    assert code == 0
    text = out.read_text(encoding="utf-8")
    assert key not in text and key not in capsys.readouterr().out
    verdicts = json.loads(text)
    assert verdicts["source_run"] == "42934419274"
    assert verdicts["vlm_calls"] == 2
    assert [o["decision"] for o in verdicts["objects"]] == ["icp_flip", "icp_flip"]
    assert "[redacted]" in verdicts["objects"][0]["reason"]
    parts = seen[0][1]["contents"][0]["parts"]
    assert parts[0]["text"] == vr.PROMPT
    assert parts[1]["inline_data"] == {
        "mime_type": "image/png",
        "data": base64.b64encode(b"\x89PNG0").decode("ascii"),
    }
    assert seen[0][1]["generationConfig"]["responseMimeType"] == "application/json"


def test_harness_text_calls_are_unchanged():
    def transport(url, key, payload):
        assert payload == {"contents": [{"parts": [{"text": "plan"}]}]}
        body = {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
        return 200, json.dumps(body)

    import os

    os.environ["GEMINI_API_KEY"] = "sf19-key"
    try:
        assert GeminiHarness(transport=transport).complete("plan") == "ok"
    finally:
        del os.environ["GEMINI_API_KEY"]


def test_the_box_checks_every_verdict_field(tmp_path):
    path = tmp_path / "v.json"
    good = {
        "source_run": "42934419274",
        "model": "gemini-3.8-flash",
        "prompt_sha256": "abc",
        "vlm_calls": 2,
        "objects": [
            {"object": 0, "decision": "keep", "reason": "fits"},
            {"object": 4, "decision": "sam3d_next_view", "reason": "wrong shape"},
        ],
        "failed": [{"object": 7, "error": "HTTP 503"}],
    }
    path.write_text(json.dumps(good), encoding="utf-8")
    meta, verdicts = br.load_verdicts(path)
    assert meta["vlm_calls"] == 2 and meta["failed_objects"] == [7]
    assert meta["source_run"] == "42934419274"
    assert verdicts[4]["decision"] == "sam3d_next_view"
    for bad_item in (
        {"object": "0", "decision": "keep"},
        {"object": True, "decision": "keep"},
        {"object": 1, "decision": "rm -rf"},
    ):
        path.write_text(json.dumps({"objects": [bad_item]}), encoding="utf-8")
        with pytest.raises(ValueError):
            br.load_verdicts(path)


def test_without_a_verdict_nothing_is_retried_and_the_rule_is_not_substituted():
    assert br.vlm_decision(None, True) == ("keep", "no VLM verdict")
    flip = {"decision": "icp_flip", "reason": ""}
    assert br.vlm_decision(flip, False) == ("icp_flip", "VLM")
    regen = {"decision": "sam3d_next_view", "reason": ""}
    assert br.vlm_decision(regen, True) == ("sam3d_next_view", "VLM")
    assert br.vlm_decision(regen, False)[0] == "keep"


def test_review_vlm_runs_the_reviewer_on_the_committed_verdicts():
    from sceneagent.vast_smoke import (
        BIN_PIPELINES,
        FETCH_NAMES,
        PIPELINES,
        VLM_VERDICTS_RELATIVE,
        build_remote_command,
    )

    argv = build_remote_command("/workspace/r", "/workspace/o", pipeline="review_vlm")
    assert argv[-3:] == [
        "/workspace/o",
        "--verdicts",
        "/workspace/r/" + VLM_VERDICTS_RELATIVE,
    ]
    assert "review_vlm" in BIN_PIPELINES
    # Run 42941668034: the file was committed under ci/, which is never
    # shipped, so the box found nothing. It must be in what rsync copies.
    assert VLM_VERDICTS_RELATIVE.startswith("sceneagent/")
    assert PIPELINES["review_vlm"][0][-2:] == (br.VLM_RETRY_ITEM, br.VLM_SCORE_ITEM)
    assert "bin_review_vlm.json" in FETCH_NAMES
    assert "sf4-artifacts/bin_review_vlm.json" in GPU_WORKFLOW.read_text(
        encoding="utf-8"
    )


def test_the_verdict_workflow_holds_only_the_gemini_key_and_never_reaches_the_box():
    text = VERDICT_WORKFLOW.read_text(encoding="utf-8")
    assert re.findall(r"secrets\.([A-Z_]+)", text) == ["GEMINI_API_KEY"]
    code = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    for banned in ("VAST", "SSH", "58301075", "60254439", "vast_smoke", "pull_request", "push:"):
        assert banned not in code, banned
    assert "permissions:\n  contents: read\n  actions: read\n" in text
    assert "write" not in text.split("jobs:")[0]
    # The run id is checked before it is used, and expressions stay out of
    # run scripts.
    assert text.index("grep -Eq '^[0-9]{1,20}$'") < text.index("run-id:")
    for block in re.findall(r"run: \|\n((?:          .*\n)+)", text):
        assert "${{" not in block
    # The key is in the Gemini step only.
    steps = code.split("      - ")
    holders = [s for s in steps if "GEMINI_API_KEY" in s]
    assert len(holders) == 1 and "sceneagent.vlm_review" in holders[0]


def test_the_source_run_must_be_a_run_id(tmp_path):
    sheets = _zip(tmp_path, {"object_0.png": b"x"})
    with pytest.raises(SystemExit):
        vr.main(
            [
                "--sheets",
                str(sheets),
                "--out",
                str(tmp_path / "v.json"),
                "--source-run",
                "1; rm -rf /",
            ],
            harness=FakeHarness([]),
        )


def test_the_committed_verdict_file_is_shipped_to_the_box(tmp_path):
    from sceneagent.vast_smoke import VLM_VERDICTS_RELATIVE, iter_shipped_files

    checkout = tmp_path / "checkout"
    verdicts = checkout / VLM_VERDICTS_RELATIVE
    verdicts.parent.mkdir(parents=True)
    verdicts.write_text("{}", encoding="utf-8")
    (checkout / "ci").mkdir()
    (checkout / "ci" / "vlm_verdicts.json").write_text("{}", encoding="utf-8")
    shipped = iter_shipped_files(checkout)
    assert verdicts in shipped
    assert checkout / "ci" / "vlm_verdicts.json" not in shipped


def test_every_published_file_is_fetched_and_uploaded():
    # Run 42942342408 fetched bin_review_vlm.json, then a third hand-kept list
    # dropped it before upload. One constant now feeds the copy step.
    from sceneagent.vast_smoke import FETCH_NAMES, PUBLISHED_NAMES

    workflow = GPU_WORKFLOW.read_text(encoding="utf-8")
    uploaded = set(re.findall(r"sf4-artifacts/([A-Za-z0-9_.]+)", workflow))
    assert PUBLISHED_NAMES <= set(FETCH_NAMES)
    assert PUBLISHED_NAMES <= uploaded, PUBLISHED_NAMES - uploaded
    assert "bin_review_vlm.json" in PUBLISHED_NAMES


def _records(fits):
    return {n: {"object": n, "fit_total": f} for n, f in fits.items()}


def test_verdicts_apply_only_to_the_placement_they_were_judged_on():
    # Object numbers mean nothing across placements: after a new grouping,
    # "object 8" can be another object and the score still looks plausible.
    judged = {0: 0.09243, 1: 0.03339}
    br.check_placement(judged, _records({0: 0.09243, 1: 0.03339}))
    for records, why in (
        (_records({0: 0.09243}), "objects"),
        (_records({0: 0.09243, 1: 0.03339, 2: 0.1}), "objects"),
        (_records({0: 0.09243, 1: 0.04}), "placed differently"),
    ):
        with pytest.raises(ValueError, match=why):
            br.check_placement(judged, records)
    with pytest.raises(ValueError, match="no judged placement"):
        br.check_placement(None, _records({0: 0.09243}))


def test_verdicts_without_a_judged_placement_are_refused(tmp_path):
    # Round 1's file (merged in #54) predates judged_placement. A file like it
    # must be refused; the test builds one rather than reading whatever
    # verdict file is committed, which changes every round.
    path = tmp_path / "round1.json"
    path.write_text(
        json.dumps({"objects": [{"object": 0, "decision": "keep", "reason": ""}]}),
        encoding="utf-8",
    )
    meta, _ = br.load_verdicts(path)
    assert meta["judged_placement"] is None
    with pytest.raises(ValueError, match="no judged placement"):
        br.check_placement(meta["judged_placement"], _records({0: 0.09243}))


def test_the_judged_placement_is_read_and_checked(tmp_path):
    path = tmp_path / "v.json"
    good = {
        "objects": [{"object": 0, "decision": "keep", "reason": ""}],
        "judged_placement": [{"object": 0, "fit_total": 0.09243}],
    }
    path.write_text(json.dumps(good), encoding="utf-8")
    meta, _ = br.load_verdicts(path)
    assert meta["judged_placement"] == {0: 0.09243}
    for bad in ({"object": "0", "fit_total": 0.1}, {"object": 0, "fit_total": "x"}):
        path.write_text(json.dumps({**good, "judged_placement": [bad]}), "utf-8")
        with pytest.raises(ValueError):
            br.load_verdicts(path)


def test_the_runner_copies_the_judged_runs_fit_distances(tmp_path):
    review = tmp_path / "bin_review.json"
    review.write_text(
        json.dumps(
            {
                "objects": [
                    {"object": 1, "fit_total": 0.03339},
                    {"object": 0, "fit_total": 0.09243},
                ]
            }
        ),
        encoding="utf-8",
    )
    sheets = _zip(tmp_path, {"object_0.png": b"a", "object_1.png": b"b"})
    out = tmp_path / "v.json"
    reply = '{"decision": "keep", "reason": "fits"}'
    code = vr.main(
        ["--sheets", str(sheets), "--out", str(out), "--review-json", str(review)],
        harness=FakeHarness([reply, reply]),
    )
    assert code == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["judged_placement"] == [
        {"object": 0, "fit_total": 0.09243},
        {"object": 1, "fit_total": 0.03339},
    ]
    meta, _ = br.load_verdicts(out)
    br.check_placement(meta["judged_placement"], _records({0: 0.09243, 1: 0.03339}))


def test_the_verdict_workflow_binds_verdicts_to_the_judged_run():
    text = VERDICT_WORKFLOW.read_text(encoding="utf-8")
    assert "--review-json review-input/bin_review.json" in text
