# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Mailbox plan names stages and commands and runs none of them."""

from __future__ import annotations

import json
import subprocess

import pytest

from sceneagent.bin_run import SCENE
from sceneagent.run4_sampling import BIN_COUNT
from sceneagent.video_run import FIRST_SCENE, STAGES, video_plan, write_plan


def test_bin_scene_stays_the_graspclutter_default() -> None:
    assert SCENE == "000091"


def test_mailbox_plan_lists_the_slice_in_order(tmp_path) -> None:
    plan = video_plan(FIRST_SCENE, tmp_path)
    assert plan["scene"] == "Mailbox"
    assert plan["video"] == "docs/assets/example_videos/Mailbox.mp4"
    assert plan["frame_bins"] == BIN_COUNT == 300
    assert plan["matching"] == "all_pairs"
    assert plan["mapper"] == "global_mapper"
    assert plan["stages"] == list(STAGES)
    assert plan["executed"] is False
    assert list(plan["commands"]) == list(STAGES)
    assert plan["commands"]["poses"][-2:] == ["--out", str(tmp_path)]
    assert plan["commands"]["poses"][2] == "sceneagent.run4"
    assert plan["commands"]["place"][2] == "sceneagent.bin_place"
    assert "--out" in plan["commands"]["splat"]


def test_any_other_scene_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="Mailbox only"):
        video_plan("Fruits", tmp_path)


def test_write_plan_does_not_start_a_process(tmp_path, monkeypatch) -> None:
    def boom(*_args, **_kwargs):
        raise AssertionError("a stage was started")

    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)
    path = write_plan(tmp_path)
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["executed"] is False
    assert written["stages"][0] == "frames"
    assert written["stages"][-1] == "place"
