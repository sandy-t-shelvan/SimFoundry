# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KAN-48 thin slice: the scene record, the video pose layout and the resume rules."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from sceneagent import bin_place, bin_run, video_run
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


def test_association_before_same_phrase_grouping_runs_again(tmp_path):
    out = tmp_path / "mailbox"
    out.mkdir()
    write_scene(out, VIDEO)
    _ledger(out, ["Gaussian association"])
    items = ("Gaussian association",)
    (out / "association_evidence.json").write_text(json.dumps({"rule": {}}))
    assert not video_run.stage_done(out, "association", items)
    rule = {"rule": {"grouping": {"same_phrase": True}}}
    (out / "association_evidence.json").write_text(json.dumps(rule))
    assert not video_run.stage_done(out, "association", items)
    rule["rule"]["video_kappa_share"] = 0.5
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
    assert calls == ["masks", "association", "sam3d", "place", "export", "settle"]
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
    assert PIPELINES["mailbox"][0][-3:] == (
        bin_place.ICP_ITEM,
        "USD physics export",
        "Isaac Lab settle",
    )
    assert bin_place.SCORE_ITEM not in PIPELINES["mailbox"][0]
    for name in (video_run.RECORD, "scene.json"):
        assert name in FETCH_NAMES and name in PUBLISHED_NAMES


def test_the_bin_scene_is_unchanged_by_a_video_copy():
    # replace() on the frozen record must not leak into the bin default.
    replace(BIN_SCENE, name="other")
    assert BIN_SCENE.name == "000091"
