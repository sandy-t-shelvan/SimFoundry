# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KAN-48 thin slice: the scene record, the video pose layout and the resume rules."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from sceneagent import bin_place, bin_run, video_run
from sceneagent.scene_objects import lookup_prompt

ROOT = Path(__file__).resolve().parents[2]
from sceneagent.bin_run import BIN_SCENE, Scene, load_scene, mask_views, write_scene
from sceneagent.vast_smoke import (
    FETCH_NAMES,
    MAILBOX_RELATIVE,
    PIPELINES,
    PUBLISHED_NAMES,
    REMOTE_MAILBOX_OUT,
    REMOTE_PYTHON,
    build_remote_command,
    remote_out,
)

VIDEO = Scene(
    source="simfoundry_example_video",
    name="Mailbox",
    ledger_name="Mailbox",
    views=300,
    input_item="video",
    view_dir="mask_views",
    min_masked_views=24,
    ground_truth=False,
    prompt="mailbox . cube .",
)


def _ledger(out: Path, items: list[str], name: str = "Mailbox") -> None:
    entries = [{"item": item, "decision": "accept"} for item in items]
    (out / "ledger.json").write_text(
        json.dumps({"scene_name": name, "entries": entries})
    )


def test_a_directory_without_scene_json_is_the_bin(tmp_path):
    # The bin's runs predate scene.json; its behaviour must not move.
    assert load_scene(tmp_path) == BIN_SCENE
    assert BIN_SCENE.ledger_name == "GraspClutter6D 000091"
    assert (BIN_SCENE.views, BIN_SCENE.min_masked_views) == (13, 13)
    assert (BIN_SCENE.view_dir, BIN_SCENE.ground_truth) == ("images", True)


def test_scene_json_round_trips(tmp_path):
    write_scene(tmp_path, VIDEO)
    assert load_scene(tmp_path) == VIDEO
    assert not bin_run.is_bin(tmp_path)


def test_mask_views_are_every_tenth_posed_frame():
    names = [f"{i:04d}.jpg" for i in range(300, 0, -1)]
    picked = mask_views(names)
    assert len(picked) == 30
    assert picked[:3] == ["0001.jpg", "0011.jpg", "0021.jpg"]


def test_the_bin_poses_refuse_another_captures_directory(tmp_path):
    write_scene(tmp_path, VIDEO)
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "0001.jpg").write_bytes(b"frame")
    with pytest.raises(ValueError, match="Mailbox"):
        bin_run.run_bin_poses(tmp_path, tmp_path / "data")
    assert (tmp_path / "images" / "0001.jpg").is_file()


def test_video_poses_lay_out_what_the_bin_stages_read(tmp_path, monkeypatch):
    out = tmp_path / "mailbox"
    names = [f"{i:04d}.jpg" for i in range(1, 301)]

    def fake_run4(argv):
        assert argv[1:] == ["--out", str(out)]
        # Written before run4 starts, so no stage can mistake this for the bin.
        assert load_scene(out).name == "Mailbox"
        (out / "images").mkdir()
        for name in names:
            (out / "images" / name).write_bytes(name.encode())
        (out / "sparse" / "0").mkdir(parents=True)
        bar = {"pass": True, "model": "0", "inputs": 300}
        (out / "run4_evidence.json").write_text(json.dumps({"quality_bar": bar}))
        return 0

    import sceneagent.bin_association as association

    monkeypatch.setattr(
        association, "read_images_bin", lambda path: dict.fromkeys(names)
    )
    code = bin_run.run_video_poses(
        tmp_path / "Mailbox.mp4", out, "mailbox . cube .", poses=fake_run4
    )
    assert code == 0
    scene = load_scene(out)
    assert (scene.views, scene.min_masked_views, scene.ground_truth) == (300, 24, False)
    staged = sorted(p.name for p in (out / "mask_views").iterdir())
    assert len(staged) == 30 and staged[0] == "0001.jpg"
    evidence = json.loads((out / "bin_evidence.json").read_text())
    assert evidence["quality_bar"]["inputs"] == 300


def test_failed_video_poses_leave_the_pose_stage_to_run_again(tmp_path):
    def fake_run4(argv):
        bar = {"pass": False, "model": None, "inputs": 300}
        Path(argv[2], "run4_evidence.json").write_text(json.dumps({"quality_bar": bar}))
        return 1

    out = tmp_path / "mailbox"
    assert (
        bin_run.run_video_poses(tmp_path / "Mailbox.mp4", out, "x .", poses=fake_run4)
        == 1
    )
    _ledger(out, ["video", "COLMAP camera poses"])
    # run4 accepted the poses, but the mask views were never staged.
    assert not video_run.stage_done(out, "poses", ("video", "COLMAP camera poses"))


def test_masks_made_before_joined_phrases_were_dropped_run_again(tmp_path):
    out = tmp_path / "mailbox"
    out.mkdir()
    write_scene(out, VIDEO)
    _ledger(out, ["SAM masks"])
    assert not video_run.stage_done(out, "masks", ("SAM masks",))
    old = {"filter": {"phrase_must_match_prompt": True}}
    (out / "masks_evidence.json").write_text(json.dumps(old))
    assert not video_run.stage_done(out, "masks", ("SAM masks",))
    new = {"filter": {"joined_phrases": "drop"}}
    (out / "masks_evidence.json").write_text(json.dumps(new))
    assert video_run.stage_done(out, "masks", ("SAM masks",))


def test_association_made_under_earlier_video_rules_runs_again(tmp_path):
    out = tmp_path / "mailbox"
    out.mkdir()
    write_scene(out, VIDEO)
    _ledger(out, ["Gaussian association"])
    items = ("Gaussian association",)
    (out / "association_evidence.json").write_text(json.dumps({"rule": {}}))
    assert not video_run.stage_done(out, "association", items)
    # Run 43331599718's association split groups by phrase: it runs again.
    rule = {"rule": {"grouping": {"same_phrase": True}, "video_kappa_share": 0.5}}
    (out / "association_evidence.json").write_text(json.dumps(rule))
    assert not video_run.stage_done(out, "association", items)
    rule["rule"]["grouping"]["same_phrase"] = False
    (out / "association_evidence.json").write_text(json.dumps(rule))
    assert video_run.stage_done(out, "association", items)


def test_placements_judged_by_an_earlier_gate_are_placed_again(tmp_path):
    out = tmp_path / "mailbox"
    out.mkdir()
    write_scene(out, VIDEO)
    _ledger(out, ["ICP residuals"])
    old = {"icp": {"gate": "ours_iteration_1", "objects": []}}
    (out / "bin_score.json").write_text(json.dumps(old))
    assert not video_run.stage_done(out, "place", ("ICP residuals",))
    new = {"icp": {"gate": "ours_iteration_2", "objects": []}}
    (out / "bin_score.json").write_text(json.dumps(new))
    assert video_run.stage_done(out, "place", ("ICP residuals",))


def test_an_export_without_object_boxes_runs_again(tmp_path):
    out = tmp_path / "mailbox"
    out.mkdir()
    write_scene(out, VIDEO)
    _ledger(out, ["USD physics export"])
    (out / "export_evidence.json").write_text(json.dumps({"objects": []}))
    assert not video_run.stage_done(out, "export", ("USD physics export",))
    boxes = {"objects": [], "object_boxes": {}}
    (out / "export_evidence.json").write_text(json.dumps(boxes))
    assert not video_run.stage_done(out, "export", ("USD physics export",))
    boxes["settle_variants"] = list(video_run.VARIANTS)
    (out / "export_evidence.json").write_text(json.dumps(boxes))
    assert video_run.stage_done(out, "export", ("USD physics export",))


def test_the_run_resumes_and_reruns_everything_after_a_stage_that_ran(tmp_path):
    out = tmp_path / "mailbox"
    out.mkdir()
    write_scene(out, VIDEO)
    _ledger(out, ["video", "COLMAP camera poses", "full-scene splat"])
    calls = []

    def stage(name, items):
        def run():
            calls.append(name)
            ledger = json.loads((out / "ledger.json").read_text())
            ledger["entries"] += [{"item": i, "decision": "accept"} for i in items]
            (out / "ledger.json").write_text(json.dumps(ledger))
            return 0

        return run

    funcs = {name: stage(name, items) for name, items in video_run.STAGES}
    assert video_run.run_video(tmp_path / "Mailbox.mp4", out, "x .", stages=funcs) == 0
    assert calls == [
        "masks",
        "association",
        "sam3d",
        "place",
        "prior_input",
        "physics",
        "export",
        "settle",
        "inspect",
    ]
    record = json.loads((out / video_run.RECORD).read_text())
    assert record["complete"] and set(record["stages"]) == set(calls)


def test_no_stage_starts_past_the_time_budget(tmp_path):
    out = tmp_path / "mailbox"
    out.mkdir()
    write_scene(out, VIDEO)
    # start, poses' budget check, its start and end, then splat's budget check
    ticks = iter([0.0, 0.0, 0.0, 0.0, 5000.0, 5000.0])
    calls = []
    funcs = {
        name: (lambda n=name: calls.append(n) or 0) for name, _ in video_run.STAGES
    }
    code = video_run.run_video(
        tmp_path / "Mailbox.mp4",
        out,
        "x .",
        stages=funcs,
        budget_s=100,
        clock=lambda: next(ticks),
    )
    assert code == 1 and calls == ["poses"]
    record = json.loads((out / video_run.RECORD).read_text())
    assert record["stopped"] == "time budget: splat not started"


def test_a_directory_without_video_scene_json_stops_before_the_splat(tmp_path):
    out = tmp_path / "mailbox"
    funcs = {name: (lambda: 0) for name, _ in video_run.STAGES}
    assert video_run.run_video(tmp_path / "Mailbox.mp4", out, "x .", stages=funcs) == 1
    record = json.loads((out / video_run.RECORD).read_text())
    assert record["stopped"].startswith("no video scene.json")


def test_placement_without_ground_truth_writes_no_score_entry(tmp_path, monkeypatch):
    write_scene(tmp_path, VIDEO)
    _ledger(tmp_path, ["video", "COLMAP camera poses", "SAM 3D meshes"])
    monkeypatch.setattr(bin_place, "_load", lambda out: {})
    placed = {"objects": [{"placed": True, "gate": "accept"}], "placed": {}}
    monkeypatch.setattr(bin_place, "place_objects", lambda out, inputs: placed)

    def no_score(*args, **kwargs):
        raise AssertionError("a scene without ground truth must not be scored")

    monkeypatch.setattr(bin_place, "score_scene", no_score)
    assert bin_place.run_bin_place(tmp_path, tmp_path / "data") == 0
    items = [
        e["item"] for e in json.loads((tmp_path / "ledger.json").read_text())["entries"]
    ]
    assert bin_place.SCORE_ITEM not in items and bin_place.ICP_ITEM in items
    evidence = json.loads((tmp_path / "bin_score.json").read_text())
    assert evidence["score"] == {"skipped": "no ground truth for this scene"}


def test_the_mailbox_pipeline_has_its_own_dir_and_ships_the_video():
    assert remote_out("mailbox") == REMOTE_MAILBOX_OUT
    argv = build_remote_command("/w/repo", REMOTE_MAILBOX_OUT, pipeline="mailbox")
    assert argv == [
        REMOTE_PYTHON,
        "-m",
        "sceneagent.video_run",
        f"/w/repo/{MAILBOX_RELATIVE}",
        "--out",
        REMOTE_MAILBOX_OUT,
    ]
    assert PIPELINES["mailbox"][0][-6:] == (
        bin_place.ICP_ITEM,
        "fitted splat prior input",
        "§3.4 properties",
        "USD physics export",
        "Isaac Lab settle",
        "inspection views",
    )
    assert bin_place.SCORE_ITEM not in PIPELINES["mailbox"][0]
    for name in (video_run.RECORD, "scene.json"):
        assert name in FETCH_NAMES and name in PUBLISHED_NAMES


def test_the_bin_scene_is_unchanged_by_a_video_copy():
    # replace() on the frozen record must not leak into the bin default.
    replace(BIN_SCENE, name="other")
    assert BIN_SCENE.name == "000091"


def test_each_example_video_pipeline_runs_its_own_video_in_its_own_dir():
    from sceneagent.vast_smoke import VIDEO_PIPELINES

    for name, relative in VIDEO_PIPELINES.items():
        out = remote_out(name)
        assert out.endswith("/" + name)
        argv = build_remote_command("/w/repo", out, pipeline=name)
        assert argv[-3:] == [f"/w/repo/{relative}", "--out", out]
        # KAN-56: its prompt is the committed verdict for its bytes.
        assert lookup_prompt(ROOT / relative) is not None, relative
        assert PIPELINES[name][0][-1] == "inspection views"
    assert len({remote_out(name) for name in VIDEO_PIPELINES}) == len(VIDEO_PIPELINES)


def test_the_package_command_runs_a_known_video_through_video_run(tmp_path, monkeypatch):
    seen = {}

    def fake_main(argv=None):
        seen["argv"] = argv
        return 0

    monkeypatch.setattr("sceneagent.video_run.main", fake_main)
    from sceneagent.__main__ import main

    # KAN-56: a known video is known by its bytes, not its name.
    import shutil

    video = tmp_path / "renamed_copy.mp4"
    shutil.copyfile(ROOT / "docs/assets/example_videos/Mailbox.mp4", video)
    assert main([str(video), "--out", str(tmp_path / "out")]) == 0
    assert seen["argv"] == [str(video), "--out", str(tmp_path / "out")]
    with pytest.raises(SystemExit) as caught:
        main([str(video), "--out", str(tmp_path / "capped"), "--fps", "0"])
    assert caught.value.code == 2
    seen.clear()
    impostor = tmp_path / "Mailbox.mp4"
    impostor.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="not an mp4"):
        main([str(impostor), "--out", str(tmp_path / "other")])
    assert seen == {}


def test_an_explicit_prompt_is_forwarded_to_the_video_pipeline(tmp_path, monkeypatch):
    seen = {}

    def fake_main(argv=None):
        seen["argv"] = argv
        return 0

    monkeypatch.setattr("sceneagent.video_run.main", fake_main)
    from sceneagent.__main__ import main

    video = tmp_path / "Other.mp4"
    video.write_bytes(b"not a video")
    assert (
        main(
            [
                str(video),
                "--out",
                str(tmp_path / "out"),
                "--prompt",
                "cup . marker .",
            ]
        )
        == 0
    )
    assert seen["argv"][-2:] == ["--prompt", "cup . marker ."]


def test_inspection_views_drawn_before_left_out_existed_are_drawn_again(tmp_path):
    out = tmp_path / "mailbox"
    out.mkdir()
    write_scene(out, VIDEO)
    _ledger(out, ["inspection views"])
    (out / "settle_summary.json").write_text(json.dumps({"ran": True}))
    assert not video_run.stage_done(out, "inspect", ("inspection views",))
    (out / "settle_summary.json").write_text(json.dumps({"ran": True, "left_out": []}))
    assert video_run.stage_done(out, "inspect", ("inspection views",))


def _accepting_stages(out, calls):
    def stage(name, items):
        def run():
            calls.append(name)
            ledger = json.loads((out / "ledger.json").read_text())
            ledger["entries"] += [{"item": i, "decision": "accept"} for i in items]
            (out / "ledger.json").write_text(json.dumps(ledger))
            return 0

        return run

    return {name: stage(name, items) for name, items in video_run.STAGES}


def _critic(out, calls, verdicts):
    """Writes the verdict line the real critic writes; ``verdicts`` by critic stage."""

    def judge(path, critic_stage):
        calls.append(f"critic:{critic_stage}")
        verdict = verdicts.get(critic_stage, "accept")
        ledger = json.loads((path / "ledger.json").read_text())
        ledger["entries"].append(
            {"item": video_run.review_item(critic_stage), "decision": verdict}
        )
        (path / "ledger.json").write_text(json.dumps(ledger))
        outcome = "accept" if verdict == "accept" else "reject, retry skipped"
        return {"outcome": outcome, "attempts": [{"attempt": 0}]}

    return judge


def _video_out(tmp_path, done):
    out = tmp_path / "mailbox"
    out.mkdir()
    write_scene(out, VIDEO)
    _ledger(out, done)
    return out


def test_the_critic_judges_each_drawable_stage_right_after_it_runs(tmp_path):
    out = _video_out(tmp_path, ["video", "COLMAP camera poses", "full-scene splat"])
    calls = []
    code = video_run.run_video(
        tmp_path / "Mailbox.mp4",
        out,
        "x .",
        stages=_accepting_stages(out, calls),
        critic=_critic(out, calls, {}),
    )
    assert code == 0
    assert calls[:6] == [
        "masks",
        "critic:masks",
        "association",
        "critic:gaussians",
        "sam3d",
        "critic:mesh",
    ]
    record = json.loads((out / video_run.RECORD).read_text())
    assert record["complete"] and record["critic_rejects"] == []
    assert [c["outcome"] for c in record["critic"]] == ["accept"] * 3


def test_a_reject_that_stands_stops_the_run_before_anything_is_built_on_it(tmp_path):
    out = _video_out(tmp_path, ["video", "COLMAP camera poses", "full-scene splat"])
    calls = []
    code = video_run.run_video(
        tmp_path / "Mailbox.mp4",
        out,
        "x .",
        stages=_accepting_stages(out, calls),
        critic=_critic(out, calls, {"mesh": "reject"}),
    )
    assert code == 1
    assert calls[-2:] == ["sam3d", "critic:mesh"] and "place" not in calls
    record = json.loads((out / video_run.RECORD).read_text())
    assert not record["complete"] and record["critic_rejects"] == ["sam3d"]
    assert record["stopped"] == (
        "sam3d: the step critic's outcome is 'reject, retry skipped'"
    )


def _done_through_association(tmp_path):
    out = _video_out(tmp_path, ["video", "COLMAP camera poses", "full-scene splat"])
    (out / "masks_evidence.json").write_text(
        json.dumps({"filter": {video_run.JOINED_PHRASES: "drop"}})
    )
    (out / "association_evidence.json").write_text(
        json.dumps(
            {
                "rule": {
                    "grouping": {video_run.SAME_PHRASE: False},
                    "video_kappa_share": video_run.VIDEO_KAPPA_SHARE,
                }
            }
        )
    )
    ledger = json.loads((out / "ledger.json").read_text())
    ledger["entries"] += [
        {"item": "SAM masks", "decision": "accept"},
        {"item": "step review (masks)", "decision": "accept"},
        {"item": "Gaussian association", "decision": "accept"},
    ]
    (out / "ledger.json").write_text(json.dumps(ledger))
    return out


def test_a_resumed_stage_is_judged_once_and_a_standing_reject_stops_it(tmp_path):
    out = _done_through_association(tmp_path)
    calls = []
    critic = _critic(out, calls, {"gaussians": "reject"})
    stages = _accepting_stages(out, calls)
    assert video_run.run_video(
        tmp_path / "Mailbox.mp4", out, "x .", stages=stages, critic=critic
    )
    # Association was done but never judged: judged now, rejected, nothing after.
    assert calls == ["critic:gaussians"]
    calls.clear()
    assert video_run.run_video(
        tmp_path / "Mailbox.mp4", out, "x .", stages=stages, critic=critic
    )
    # The reject stands: no second judgement, no new retry budget.
    assert calls == []
    record = json.loads((out / video_run.RECORD).read_text())
    assert record["stopped"] == "association: the step critic rejected it"


def test_a_standing_masks_reject_is_recorded_and_the_objects_are_judged_next(
    tmp_path,
):
    out = _video_out(tmp_path, ["video", "COLMAP camera poses", "full-scene splat"])
    calls = []
    code = video_run.run_video(
        tmp_path / "Mailbox.mp4",
        out,
        "x .",
        stages=_accepting_stages(out, calls),
        critic=_critic(out, calls, {"masks": "reject"}),
    )
    assert code == 0
    assert calls[:4] == ["masks", "critic:masks", "association", "critic:gaussians"]
    record = json.loads((out / video_run.RECORD).read_text())
    assert record["critic"][0] == {
        "stage": "masks",
        "critic_stage": "masks",
        "outcome": "reject, retry skipped",
        "attempts": 1,
        "retried": False,
        "advisory": True,
    }
    assert record["complete"] and record["critic_rejects"] == []


def test_a_critic_reject_keeps_a_run_incomplete_even_without_the_critic(tmp_path):
    items = [i for _, stage_items in video_run.STAGES for i in stage_items]
    out = _video_out(tmp_path, items)
    calls = []
    _critic(out, calls, {"gaussians": "reject"})(out, "gaussians")
    funcs = _accepting_stages(out, calls)
    video_run.run_video(tmp_path / "Mailbox.mp4", out, "x .", stages=funcs)
    record = json.loads((out / video_run.RECORD).read_text())
    assert record["critic_rejects"] == ["association"] and not record["complete"]


def test_new_masks_make_later_critic_verdicts_stale():
    entries = [
        {"item": "Gaussian association", "decision": "accept"},
        {"item": "step review (gaussians)", "decision": "accept"},
        {"item": "step review (masks)", "decision": "accept"},
    ]
    stale = video_run.stale_after(entries, "masks", "r")
    assert [e["item"] for e in stale] == [
        "Gaussian association",
        "step review (gaussians)",
    ]
