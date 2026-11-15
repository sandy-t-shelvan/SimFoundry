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


def test_a_marker_longer_than_the_cup_opening_is_planned_upright():
    import math

    from sceneagent import scripted_expert as se

    marker = {
        "xyz_m": [0.45, 0.12, 0.0115],
        "size_m": [0.133, 0.025, 0.023],
        "yaw_rad": 0.0,
    }
    cup = {"xyz_m": [0.51, -0.12, 0.05], "size_m": [0.094, 0.097, 0.1]}
    made = se.plan(marker, cup)
    assert made["upright"] is True
    assert [p["name"] for p in made["phases"]] == list(se.UPRIGHT_PHASES)
    by = {p["name"]: p for p in made["phases"]}
    # Gripped off-centre toward +x, as near the middle as the cup's depth allows:
    # the fingertips stay above the rim with the bottom 1 cm above the floor.
    grip_offset = by["grasp"]["ee_xyz_m"][0] - 0.45
    assert 0.005 <= grip_offset <= 0.133 / 2 - se.GRIP_FROM_END_M
    assert math.isclose(made["hang_m"], grip_offset + 0.133 / 2)
    assert by["lower"]["ee_xyz_m"][2] >= 0.1 + se.FINGER_CLEARANCE_M - 1e-9
    assert by["upright"]["hand_pitch_rad"] == se.UPRIGHT_PITCH_RAD
    assert by["upright"]["gripper"] == "close" and by["release"]["gripper"] == "open"
    # Carried with the hanging bottom above the rim; released 1 cm above the floor.
    rim = 0.05 + 0.05
    assert (
        by["carry"]["ee_xyz_m"][2] - made["hang_m"] >= rim + se.RIM_CLEARANCE_M - 1e-9
    )
    floor = 0.05 - 0.05 + se.FLOOR_M
    assert math.isclose(
        by["lower"]["ee_xyz_m"][2] - made["hang_m"], floor + se.ABOVE_FLOOR_M
    )


def test_an_object_that_fits_lying_down_keeps_the_eight_phases():
    from sceneagent import scripted_expert as se

    block = {"xyz_m": [0.45, 0.12, 0.02], "size_m": [0.05, 0.04, 0.04], "yaw_rad": 0.0}
    bowl = {"xyz_m": [0.51, -0.12, 0.04], "size_m": [0.2, 0.2, 0.08]}
    made = se.plan(block, bowl)
    assert "upright" not in made
    assert [p["name"] for p in made["phases"]] == list(se.PHASES)


def test_the_upright_phase_waits_for_the_hand_to_turn():
    from sceneagent import scripted_expert as se

    phase = {
        "name": "upright",
        "ee_xyz_m": [0.5, 0.1, 0.3],
        "hand_pitch_rad": se.UPRIGHT_PITCH_RAD,
        "gripper": "close",
        "time_limit_s": 2.0,
    }
    at = {"ee_xyz_m": [0.5, 0.1, 0.3]}
    assert se._finished(phase, {**at, "pitch_error_rad": 0.5}) is False
    assert se._finished(phase, at) is False  # no angle observed: not finished
    assert se._finished(phase, {**at, "pitch_error_rad": 0.02}) is True


def test_lower_is_a_guarded_move_that_stops_when_blocked():
    from sceneagent import scripted_expert as se

    phase = {
        "name": "lower",
        "ee_xyz_m": [0.5, -0.1, 0.115],
        "gripper": "close",
        "time_limit_s": 2.0,
    }
    stuck = {"ee_xyz_m": [0.5, -0.1, 0.133]}
    assert se._finished(phase, stuck) is False
    assert se._finished(phase, {**stuck, "blocked": True}) is True


def test_zero_successes_is_a_reject_at_any_trial_count():
    from sceneagent.scripted_expert import record_trials

    assert record_trials([{"success": False}] * 3)["decision"] == "reject"
    assert "decision" not in record_trials([{"success": True}, {"success": False}])
