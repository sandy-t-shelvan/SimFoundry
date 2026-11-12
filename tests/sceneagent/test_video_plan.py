# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""run_video writes plan.json with the same writer as run_capture."""

import json
from pathlib import Path

from sceneagent.pipeline import run_path, write_plan
from sceneagent.skills import plan_steps
from sceneagent.vast_smoke import collect_artifact_manifest, publish_artifacts
from sceneagent.video import VIDEO_SOURCE, run_video, video_plan_steps

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "phase1_place_scene.json"


def _tiny_mp4(path):
    path.write_bytes(b"\x00\x00\x00\x14ftypisom\x00\x00\x00\x00")


def test_run_capture_plan_bytes_match_the_shared_writer(tmp_path):
    run_path(EXAMPLE, tmp_path / "capture")
    write_plan(tmp_path / "direct", "phase1_place", "sceneagent_capture", plan_steps())
    assert (tmp_path / "capture" / "plan.json").read_bytes() == (
        tmp_path / "direct" / "plan.json"
    ).read_bytes()


def test_run_video_writes_plan_json_with_the_shared_writer(tmp_path, monkeypatch):
    monkeypatch.setattr("sceneagent.external_tools.shutil.which", lambda _name: None)
    calls = []
    real = write_plan

    def wrapped(out_dir, scene_name, source, steps):
        calls.append((str(out_dir), scene_name, source, steps))
        return real(out_dir, scene_name, source, steps)

    monkeypatch.setattr("sceneagent.pipeline.write_plan", wrapped)
    video = tmp_path / "clip.mp4"
    _tiny_mp4(video)
    out = tmp_path / "video"
    run_video(video, out)
    assert calls
    assert calls[0][1] == "clip"
    assert calls[0][2] == VIDEO_SOURCE
    assert calls[0][3] == video_plan_steps()
    written = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    run_path(EXAMPLE, tmp_path / "capture")
    capture = json.loads(
        (tmp_path / "capture" / "plan.json").read_text(encoding="utf-8")
    )
    assert set(written) == set(capture) == {"scene_name", "source", "steps"}
    for step in written["steps"]:
        assert {"skill", "status"} <= set(step)
        assert set(step) <= {"skill", "status", "reason", "fix_report_from_model"}
    assert "decision" not in json.dumps(written["steps"])
    original = (out / "plan.json").read_bytes()
    collect_artifact_manifest(
        out, command=["python3", "-m", "sceneagent"], instance_id=60254439
    )
    artifacts = tmp_path / "artifacts"
    publish_artifacts(out, artifacts)
    assert (out / "plan.json").read_bytes() == original
    assert (artifacts / "plan.json").read_bytes() == original
