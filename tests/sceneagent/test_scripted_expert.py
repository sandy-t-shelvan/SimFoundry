# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Eight-phase controller. The observations are fake. No simulator is imported."""

from __future__ import annotations

import json

import pytest

from sceneagent.metrics import wilson_interval
from sceneagent.scripted_expert import (
    PHASES,
    episode_success,
    plan,
    record_trials,
    step,
)

CUP = {"xyz_m": [0.0, 0.0, 0.05], "size_m": [0.04, 0.04, 0.08]}
BOWL = {"xyz_m": [0.2, 0.0, 0.05], "size_m": [0.12, 0.12, 0.06]}


def test_the_phases_are_in_the_paper_order_and_a_wide_object_is_not_tried():
    made = plan(CUP, BOWL)
    assert made["graspable"] is True
    assert [phase["name"] for phase in made["phases"]] == list(PHASES)
    wide = plan({"xyz_m": [0, 0, 0], "size_m": [0.2, 0.2, 0.01]}, BOWL)
    assert wide["graspable"] is False
    assert wide["phases"] == []


def test_a_phase_finishes_on_the_tolerance_and_a_timeout_does_not_advance():
    made = plan(CUP, BOWL)
    state = {"phases": made["phases"], "phase_index": 0, "elapsed_s": 0.0}
    waiting = step(state, {"ee_xyz_m": [0.0, 0.0, 0.0], "gripper_closed": False})
    assert waiting["advance"] is False and waiting["failed"] is False
    close = {
        "ee_xyz_m": made["phases"][0]["ee_xyz_m"],
        "gripper_closed": False,
    }
    assert step(state, close)["advance"] is True
    grasp = {"phases": made["phases"], "phase_index": 2, "elapsed_s": 0.1}
    assert (
        step(grasp, {"ee_xyz_m": [0, 0, 0], "gripper_closed": True})["advance"] is True
    )
    late = {"phases": made["phases"], "phase_index": 0, "elapsed_s": 3.0}
    failed = step(late, close)
    assert failed["failed"] is True and failed["advance"] is False
    assert failed["reason"] == "time limit"


def test_success_needs_the_object_inside_and_at_rest():
    half = [0.05, 0.05, 0.05]
    assert episode_success([0.2, 0.0, 0.05], BOWL["xyz_m"], half, 0.0) is True
    assert episode_success([0.2, 0.0, 0.05], BOWL["xyz_m"], half, 0.02) is False
    assert episode_success([0.0, 0.0, 0.05], BOWL["xyz_m"], half, 0.0) is False


def test_seven_of_ten_matches_the_wilson_interval_and_zero_is_a_reject(tmp_path):
    low, high = wilson_interval(7, 10)
    assert low == pytest.approx(0.40, abs=0.02)
    assert high == pytest.approx(0.89, abs=0.02)
    trials = [{"trial": i, "success": False, "phase": "grasp"} for i in range(10)]
    ledger = tmp_path / "ledger.json"
    document = record_trials(trials, ledger)
    assert document["successes"] == 0
    assert document["decision"] == "reject"
    saved = json.loads(ledger.read_text(encoding="utf-8"))
    assert saved["entries"][-1]["decision"] == "reject"
