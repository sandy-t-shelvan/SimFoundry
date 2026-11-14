# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KAN-56: detector prompts keyed by a video's bytes, named by Gemini on the runner."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from sceneagent import scene_objects as so

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "sceneagent-vlm-verdict.yml"


class FakeHarness:
    model_id = "gemini-3.8-flash"

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def complete(self, prompt, images=(), response_schema=None):
        self.calls.append((prompt, list(images), response_schema))
        return self.reply


def _video(tmp_path, name="clip.mp4", data=b"not really a video"):
    path = tmp_path / name
    path.write_bytes(data)
    return path


FRAMES = [(0.5, b"\xff\xd8a"), (1.5, b"\xff\xd8b"), (2.5, b"\xff\xd8c")]


def test_phrases_are_clean_groundingdino_text():
    assert so.phrase("  Coffee Mug.") == "coffee mug"
    assert so.phrase("T-shirt, folded!") == "t-shirt folded"
    objects = [{"name": n} for n in ("Marker", "cup", "marker", "", "Paper. Cup")]
    assert so.prompt_from_objects(objects) == "marker . cup . paper cup ."
    many = [{"name": f"thing {i}"} for i in range(10)]
    assert so.prompt_from_objects(many).count(" . ") == so.MAX_OBJECTS - 1
    with pytest.raises(ValueError, match="names no object"):
        so.prompt_from_objects([{"name": "..."}])


def test_a_reply_must_match_the_schema():
    good = json.dumps({"objects": [{"name": "cup", "movable_by_hand": True}]})
    assert so.parse_reply(good) == [{"name": "cup", "movable_by_hand": True}]
    for bad in ("not json", "[]", '{"objects": [{"name": "cup"}]}', '{"objects": [{"movable_by_hand": true}]}'):
        with pytest.raises(ValueError):
            so.parse_reply(bad)


def test_propose_sends_three_frames_and_records_the_reply(tmp_path):
    video = _video(tmp_path)
    reply = json.dumps(
        {"objects": [{"name": "Cup", "movable_by_hand": True}, {"name": "marker", "movable_by_hand": True}]}
    )
    harness = FakeHarness(reply)
    record = so.propose(video, harness, frames=FRAMES)
    prompt, images, schema = harness.calls[0]
    assert prompt == so.PROMPT and schema == so.SCHEMA
    assert [mime for _, mime in images] == ["image/jpeg"] * 3
    assert record["ok"] is True and record["prompt"] == "cup . marker ."
    assert record["video_sha256"] == so.video_sha256(video)
    assert record["frame_times_s"] == [0.5, 1.5, 2.5]
    assert record["model"] == "gemini-3.8-flash"
    assert record["prompt_sha256"] == so.prompt_sha256()


def test_a_bad_reply_is_recorded_not_raised(tmp_path):
    record = so.propose(_video(tmp_path), FakeHarness("I see a cup."), frames=FRAMES)
    assert record["ok"] is False and "not JSON" in record["reason"]
    assert record["reply"] == "I see a cup."


def test_the_prompt_follows_the_bytes_not_the_name(tmp_path):
    video = _video(tmp_path, "Mailbox.mp4")
    record = so.propose(
        video, FakeHarness('{"objects": [{"name": "mailbox", "movable_by_hand": true}]}'), frames=FRAMES
    )
    root = tmp_path / "verdicts"
    so.write_verdict(record, root)
    renamed = tmp_path / "renamed_copy.mov"
    shutil.copyfile(video, renamed)
    assert so.lookup_prompt(renamed, root) == "mailbox ."
    other = _video(tmp_path, "Mailbox2.mp4", b"different bytes, same stem prefix")
    assert so.lookup_prompt(other, root) is None


def test_resolve_prefers_the_given_prompt_and_names_the_workflow_when_missing(tmp_path):
    video = _video(tmp_path)
    root = tmp_path / "verdicts"
    assert so.resolve_prompt(video, "bowl . cup .", root) == "bowl . cup ."
    with pytest.raises(so.MissingPromptError) as err:
        so.resolve_prompt(video, None, root)
    message = str(err.value)
    assert so.video_sha256(video) in message and "mode: objects" in message and "--prompt" in message
    failed = {"video_sha256": so.video_sha256(video), "ok": False, "reason": "x"}
    so.write_verdict(failed, root)
    assert so.lookup_prompt(video, root) is None


def test_frames_come_from_inside_the_clip():
    assert so.frame_times(15.0) == [2.5, 7.5, 12.5]
    with pytest.raises(ValueError):
        so.frame_times(0.0)
    command = so.frame_command("ffmpeg", "v.mp4", 7.5)
    assert command[command.index("-ss") + 1] == "7.500"
    assert "min(iw,1024)" in command[command.index("-vf") + 1]


def test_the_objects_mode_shares_the_one_gemini_step_and_validates_the_video():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "options: [retry, accept, objects]" in text
    assert text.count("secrets.GEMINI_API_KEY") == 1
    gemini = text.split("name: Gemini verdict per object", 1)[1].split("- name:", 1)[0]
    assert "python -m sceneagent.scene_objects propose --video video-input/video" in gemini
    fetch = text.split("name: Fetch the video", 1)[1].split("- name:", 1)[0]
    assert text.index("name: Fetch the video") < text.index("name: Gemini verdict per object")
    assert "GEMINI_API_KEY" not in fetch
    assert "VIDEO: ${{ inputs.video }}" in fetch
    assert "${{" not in fetch.split("run: |", 1)[1]
    assert "*..*" in fetch
