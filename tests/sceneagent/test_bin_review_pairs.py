# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SF-19 round 2: the VLM accepts retries from blind before/after pairs.

No test calls Gemini or the box. The harness is a fake; candidates are small
arrays in a temporary folder.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pytest

from sceneagent import bin_review_pairs as rp
from sceneagent import vlm_review as vr

ROOT = Path(__file__).resolve().parents[2]
VERDICT_WORKFLOW = ROOT / ".github" / "workflows" / "sceneagent-vlm-verdict.yml"


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


def test_pair_order_is_seeded_and_uses_both_orders():
    numbers = list(range(14))
    first, again = rp.pair_order(numbers, 7), rp.pair_order(numbers, 7)
    assert first == again
    assert {o["A"] for o in first.values()} == {"before", "after"}
    for order in first.values():
        assert {order["A"], order["B"]} == {"before", "after"}


def test_the_candidate_hash_moves_with_any_cloud():
    a = np.zeros((4, 3))
    entries = [{"object": 0, "kind": "icp_flip", "cloud_sha256": rp.cloud_sha256(a)}]
    moved = [{**entries[0], "cloud_sha256": rp.cloud_sha256(a + 1e-9)}]
    assert rp.candidates_hash(entries) != rp.candidates_hash(moved)
    assert rp.candidates_hash(entries) == rp.candidates_hash(list(entries))


def _saved_candidates(tmp_path: Path) -> tuple[Path, str]:
    cloud = np.arange(12, dtype=float).reshape(4, 3)
    item = {
        "object": 3,
        "decision": "icp_flip",
        "kind": "icp_flip",
        "cloud_sha256": rp.cloud_sha256(cloud),
        "test_accepts": False,
        "unchanged": False,
        "A": "after",
        "B": "before",
    }
    digest = rp.candidates_hash([item])
    folder = tmp_path / rp.PAIRS_DIR / digest[:16]
    folder.mkdir(parents=True)
    np.save(folder / "object_3.npy", cloud)
    manifest = {"candidates_hash": digest, "objects": [item], "judged_placement": []}
    (folder / "candidates.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folder, digest


def test_saved_candidates_load_only_unchanged(tmp_path):
    folder, digest = _saved_candidates(tmp_path)
    manifest, clouds = rp.load_candidates(tmp_path, digest)
    assert list(clouds) == [3] and manifest["candidates_hash"] == digest
    with pytest.raises(ValueError, match="not the set"):
        rp.load_candidates(tmp_path, digest[:16] + "0" * 48)
    np.save(folder / "object_3.npy", np.zeros((4, 3)))
    with pytest.raises(ValueError, match="has changed"):
        rp.load_candidates(tmp_path, digest)


def test_the_kept_side_maps_back_through_the_blind_order():
    item = {"A": "after", "B": "before", "unchanged": False}
    assert rp.choose(item, {"better": "A", "reason": ""}) == ("after", "VLM")
    assert rp.choose(item, {"better": "B", "reason": ""}) == ("before", "VLM")
    assert rp.choose(item, None) == ("before", "no VLM verdict")
    assert rp.choose({**item, "unchanged": True}, {"better": "A"})[0] == "before"


def test_accept_verdicts_are_checked(tmp_path):
    path = tmp_path / "a.json"
    good = {
        "mode": "accept",
        "candidates_hash": "a" * 64,
        "objects": [{"object": 3, "better": "A", "reason": "fills the outline"}],
    }
    path.write_text(json.dumps(good), encoding="utf-8")
    meta, verdicts = rp.load_accept_verdicts(path)
    assert meta["candidates_hash"] == "a" * 64 and verdicts[3]["better"] == "A"
    for bad in (
        {**good, "mode": "retry"},
        {**good, "candidates_hash": "../x"},
        {**good, "objects": [{"object": 3, "better": "after"}]},
        {**good, "objects": [{"object": "3", "better": "A"}]},
    ):
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError):
            rp.load_accept_verdicts(path)


def test_pairs_are_read_whole_and_judged_a_then_b(tmp_path):
    zpath = tmp_path / "review_pairs.zip"
    with zipfile.ZipFile(zpath, "w") as archive:
        for name, blob in {
            "object_3_A.png": b"a3",
            "object_3_B.png": b"b3",
            "object_5_A.png": b"a5",
            "object_5_B.png": b"b5",
            "notes.txt": b"x",
        }.items():
            archive.writestr(name, blob)
    pairs = vr.read_pairs(zpath)
    assert pairs == {3: (b"a3", b"b3"), 5: (b"a5", b"b5")}
    harness = FakeHarness(
        ['{"better": "B", "reason": "red fills green"}', RuntimeError("HTTP 503")]
    )
    result = vr.judge_pairs(pairs, harness, "c" * 64, "42939458339", orders=("AB",))
    assert result["mode"] == "accept" and result["candidates_hash"] == "c" * 64
    assert result["objects"] == [
        {
            "object": 3,
            "better": "B",
            "reason": "red fills green",
            "votes": [{"order": "AB", "better": "B", "reason": "red fills green"}],
        }
    ]
    assert [f["object"] for f in result["failed"]] == [5]
    assert harness.calls[0][1] == [(b"a3", "image/png"), (b"b3", "image/png")]
    assert result["prompt_sha256"] == vr.pair_prompt_sha256()
    with zipfile.ZipFile(zpath, "a") as archive:
        archive.writestr("object_9_A.png", b"lonely")
    with pytest.raises(ValueError, match="lack an A or a B"):
        vr.read_pairs(zpath)


def test_the_pair_prompt_does_not_say_which_side_is_the_retry():
    for leak in ("before", "after", "retry", "original", "new fit", "improved"):
        assert leak not in vr.PAIR_PROMPT.lower()


def test_pair_replies_must_name_a_or_b():
    assert vr.parse_pair('{"better": "A", "reason": "x"}')["better"] == "A"
    for bad in ('{"better": "after"}', '"A"', "A"):
        with pytest.raises(ValueError):
            vr.parse_pair(bad)


def test_round_two_runs_on_the_box_from_committed_verdicts():
    from sceneagent.vast_smoke import (
        ACCEPT_VERDICTS_RELATIVE,
        BIN_PIPELINES,
        FETCH_NAMES,
        PIPELINES,
        PUBLISHED_NAMES,
        VLM_VERDICTS_RELATIVE,
        build_remote_command,
    )

    build = build_remote_command("/w/r", "/w/o", pipeline="review_candidates")
    assert build[2:5] == ["sceneagent.bin_review_pairs", "build", "--out"]
    assert build[-1] == "/w/r/" + VLM_VERDICTS_RELATIVE
    apply = build_remote_command("/w/r", "/w/o", pipeline="review_accept")
    assert apply[3] == "apply" and apply[-1] == "/w/r/" + ACCEPT_VERDICTS_RELATIVE
    assert ACCEPT_VERDICTS_RELATIVE.startswith("sceneagent/")
    assert {"review_candidates", "review_accept"} <= BIN_PIPELINES
    assert PIPELINES["review_candidates"][0][-2:] == rp.BUILD_REQUIRED
    assert PIPELINES["review_accept"][0][-2:] == rp.APPLY_REQUIRED
    for name in ("candidates.json", "review_pairs.zip", "bin_review_accept.json"):
        assert name in FETCH_NAMES and name in PUBLISHED_NAMES


def test_the_verdict_workflow_mode_is_a_fixed_choice_in_one_step():
    text = VERDICT_WORKFLOW.read_text(encoding="utf-8")
    assert "options: [retry, accept]" in text
    assert re.findall(r"secrets\.([A-Z_]+)", text) == ["GEMINI_API_KEY"]
    assert "VERDICT_MODE: ${{ inputs.mode }}" in text
    assert "--mode accept --pairs review-input/review_pairs.zip" in text
    for block in re.findall(r"run: \|\n((?:          .*\n)+)", text):
        assert "${{" not in block


def test_both_pair_sheets_draw_the_overlay_on_the_same_views():
    # Round 2 drew a regeneration's after sheet on the view its mesh came
    # from, so A and B showed different photos while the prompt said they
    # differ only in the mesh. Both sheets now get the same overlay views.
    import inspect

    source = inspect.getsource(rp.draw_pairs)
    assert source.count("overlay_views=shown") == 2
    assert source.count("placement=placement") == 2
    assert "draw_pairs(" in inspect.getsource(rp.build_candidates)
    assert 'shown = list(dict.fromkeys([records[n]["view"], item["view"]]))' in source


def test_the_pair_prompt_says_the_photos_are_shared():
    assert "same photos and the same segmentation masks" in vr.PAIR_PROMPT
    assert "registration_overlay_2" in vr.PAIR_PROMPT


def _reply(side: str) -> str:
    return json.dumps({"better": side, "reason": f"picked {side}"})


def test_a_habit_of_naming_the_second_image_ties():
    # Round 3 answered B on 3 of 3. Votes are split evenly across the two
    # display orders, so a judge that always names the second image ties.
    harness = FakeHarness([_reply("B")] * 4)
    result = vr.judge_pairs({3: (b"a", b"b")}, harness, "c" * 64)
    item = result["objects"][0]
    assert item["better"] == "tie"
    assert [v["better"] for v in item["votes"]] == ["B", "A", "B", "A"]
    assert harness.calls[1][1] == [(b"b", "image/png"), (b"a", "image/png")]
    assert result["vlm_calls"] == 4 and result["vote_orders"] == list(vr.VOTE_ORDERS)


def test_a_consistent_preference_survives_both_orders():
    # Prefers the real B: says B when it is second, A when it is shown first.
    harness = FakeHarness([_reply("B"), _reply("A"), _reply("B"), _reply("A")])
    item = vr.judge_pairs({3: (b"a", b"b")}, harness, "c" * 64)["objects"][0]
    assert item["better"] == "B" and item["reason"] == "picked B"


def test_the_default_orders_are_balanced():
    assert vr.VOTE_ORDERS.count("AB") == vr.VOTE_ORDERS.count("BA") > 0


def test_one_failed_vote_fails_the_object():
    harness = FakeHarness([_reply("A"), RuntimeError("HTTP 503")])
    result = vr.judge_pairs({3: (b"a", b"b")}, harness, "c" * 64)
    assert result["objects"] == [] and [f["object"] for f in result["failed"]] == [3]


def test_a_tie_keeps_the_original(tmp_path):
    item = {"A": "after", "B": "before", "unchanged": False}
    assert rp.choose(item, {"better": "tie", "reason": ""})[0] == "before"
    path = tmp_path / "a.json"
    path.write_text(
        json.dumps(
            {
                "mode": "accept",
                "candidates_hash": "a" * 64,
                "objects": [{"object": 3, "better": "tie", "reason": "split"}],
            }
        ),
        encoding="utf-8",
    )
    assert rp.load_accept_verdicts(path)[1][3]["better"] == "tie"


def test_redraw_reuses_the_judged_candidates_and_their_order(tmp_path, monkeypatch):
    folder, digest = _saved_candidates(tmp_path)
    accept = tmp_path / "accept.json"
    accept.write_text(
        json.dumps({"mode": "accept", "candidates_hash": digest, "objects": []}),
        encoding="utf-8",
    )
    monkeypatch.setattr(rp, "_placement", lambda out: ({}, {}, {}))
    seen = {}

    def fake_draw(out, inputs, clouds, records, items, saved, where, order, placement):
        seen.update(order=order, placement=placement, where=where, saved=list(saved))
        return {3: {}}

    monkeypatch.setattr(rp, "draw_pairs", fake_draw)
    assert rp.redraw_pairs(tmp_path, accept) == 0
    assert seen["order"] == {3: {"A": "after", "B": "before"}}
    assert seen["placement"] is True and seen["saved"] == [3]
    assert seen["where"] == folder / "placement"
    copied = json.loads((tmp_path / "candidates.json").read_text(encoding="utf-8"))
    assert copied["candidates_hash"] == digest
    ledger = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["entries"][-1]["item"] == rp.REDRAW_ITEM
    assert ledger["entries"][-1]["decision"] == "accept"
    np.save(folder / "object_3.npy", np.zeros((4, 3)))
    assert rp.redraw_pairs(tmp_path, accept) == 1


def test_redraw_runs_on_the_box_from_the_committed_accept_verdicts():
    from sceneagent.vast_smoke import (
        ACCEPT_VERDICTS_RELATIVE,
        BIN_PIPELINES,
        PIPELINES,
        build_remote_command,
    )

    argv = build_remote_command("/w/r", "/w/o", pipeline="review_redraw")
    assert argv[2:4] == ["sceneagent.bin_review_pairs", "redraw"]
    assert argv[-1] == "/w/r/" + ACCEPT_VERDICTS_RELATIVE
    assert "review_redraw" in BIN_PIPELINES
    assert PIPELINES["review_redraw"][0][-1:] == rp.REDRAW_REQUIRED


def test_the_pair_prompt_describes_the_placement_tiles():
    for name in ("placement_x", "placement_y", "placement_z", "when present"):
        assert name in vr.PAIR_PROMPT
    assert "placement_" not in vr.PROMPT
