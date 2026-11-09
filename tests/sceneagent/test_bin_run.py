# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cut 2 phase 2 bin poses: view choice, matcher, ledger, and the bin pipeline."""

from __future__ import annotations

import json

from PIL import Image

from sceneagent.bin_run import MATCHER, run_bin_poses, view_ids
from sceneagent.colmap_stage import pose_commands
from sceneagent.vast_smoke import (
    REMOTE_OUT,
    REMOTE_REPO,
    build_remote_command,
    build_remote_shell,
    collect_artifact_manifest,
)

SIZES = {1: (1920, 1080), 2: (1920, 1080), 3: (3840, 2160), 0: (1920, 1200)}


def _scene(data, wrong_id=None):
    """52 tiny-content PNGs with GraspClutter6D 000091's per-camera sizes."""
    rgb = data / "scenes" / "000091" / "rgb"
    rgb.mkdir(parents=True)
    for image_id in range(1, 53):
        size = SIZES[image_id % 4]
        if image_id == wrong_id:
            size = (640, 480)
        Image.new("RGB", size).save(rgb / f"{image_id:06d}.png")


def _fetch(data, wrong_id=None):
    def fetch(scene, out):
        assert scene == "000091"
        _scene(out, wrong_id)
        return {"scene": scene, "files": {"a": "0" * 64}, "license": "CC BY-SA 4.0"}

    return fetch


def test_views_are_one_camera_of_thirteen():
    assert view_ids() == [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49]


def test_exhaustive_matcher_replaces_sequential_on_both_colmap_majors():
    for major in (3, 4):
        sequential = pose_commands("db", "img", "sparse", True, major=major)
        exhaustive = pose_commands(
            "db", "img", "sparse", True, major=major, matcher=MATCHER
        )
        assert sequential[1][1] == "sequential_matcher"
        assert exhaustive[1][1] == "exhaustive_matcher"
        assert "--SequentialMatching.overlap" not in exhaustive[1]
        assert exhaustive[0] == sequential[0] and exhaustive[2] == sequential[2]
        gpu_flag = [a for a in sequential[1] if a.endswith("Matching.use_gpu")]
        assert gpu_flag and gpu_flag[0] in exhaustive[1]


def test_poses_pass_and_write_plan_ledger_and_evidence(tmp_path):
    seen = {}

    def colmap(image_dir, work_dir):
        seen["images"] = sorted(p.name for p in image_dir.iterdir())
        return {"ok": True, "use_gpu": True, "reason": None, "failed_step": None}

    code = run_bin_poses(
        tmp_path / "run",
        tmp_path / "data",
        fetch=_fetch(tmp_path / "data"),
        colmap=colmap,
    )
    assert code == 0
    assert seen["images"] == [f"{i:06d}.png" for i in view_ids()]
    ledger = json.loads((tmp_path / "run/ledger.json").read_text())
    assert [e["item"] for e in ledger["entries"]] == [
        "benchmark scene",
        "COLMAP camera poses",
    ]
    assert all(e["decision"] == "accept" for e in ledger["entries"])
    assert ledger["entries"][1]["use_gpu"] is True
    assert (tmp_path / "run/plan.json").is_file()
    evidence = json.loads((tmp_path / "run/bin_evidence.json").read_text())
    assert evidence["colmap"]["ok"] is True


def test_a_failed_reconstruction_is_a_reject_and_exit_1(tmp_path):
    def colmap(image_dir, work_dir):
        return {
            "ok": False,
            "reason": "registered 6 < 11",
            "failed_step": "reconstruction_gate",
        }

    code = run_bin_poses(
        tmp_path / "run",
        tmp_path / "data",
        fetch=_fetch(tmp_path / "data"),
        colmap=colmap,
    )
    assert code == 1
    pose = json.loads((tmp_path / "run/ledger.json").read_text())["entries"][1]
    assert pose["decision"] == "reject"
    assert pose["failed_step"] == "reconstruction_gate"


def test_a_view_of_the_wrong_size_is_refused_before_colmap(tmp_path):
    def colmap(image_dir, work_dir):
        raise AssertionError("COLMAP must not run")

    code = run_bin_poses(
        tmp_path / "run",
        tmp_path / "data",
        fetch=_fetch(tmp_path / "data", wrong_id=9),
        colmap=colmap,
    )
    assert code == 1
    entry = json.loads((tmp_path / "run/ledger.json").read_text())["entries"][0]
    assert entry["decision"] == "reject"
    assert "000009.png" in entry["reason"]


def test_bin_pipeline_command_and_shell():
    argv = build_remote_command(REMOTE_REPO, REMOTE_OUT, pipeline="bin")
    assert argv[1:] == ["-m", "sceneagent.bin_run", "--out", REMOTE_OUT]
    shell = build_remote_shell(REMOTE_REPO, REMOTE_OUT, pipeline="bin")
    assert "import py7zr" in shell and "py7zr==1.1.4" in shell
    assert "py7zr" not in build_remote_shell(REMOTE_REPO, REMOTE_OUT, pipeline="run4")


def test_bin_manifest_requires_its_own_ledger_items(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "plan.json").write_text("{}\n")
    entries = [
        {"item": "benchmark scene", "decision": "accept"},
        {"item": "COLMAP camera poses", "decision": "accept", "use_gpu": True},
    ]
    (run / "ledger.json").write_text(json.dumps({"entries": entries}) + "\n")
    command = build_remote_command(REMOTE_REPO, REMOTE_OUT, pipeline="bin")
    passed = collect_artifact_manifest(
        run, command=command, instance_id=58301075, pipeline="bin"
    )
    assert passed["passed"] is True
    assert passed["video"] == "GraspClutter6D 000091"
    as_video = collect_artifact_manifest(run, command=command, instance_id=58301075)
    assert as_video["passed"] is False
