# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The pure half of the Isaac Lab runner. No simulator is imported."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from sceneagent import scripted_expert_run as run
from sceneagent.metrics import wilson_interval

SCENE = run.scene_spec()


def _log(start_z, lift_rise, carry_rise):
    return [
        {"phase": name, "outcome": "finished", "object_xyz_m": [0.0, 0.0, start_z]}
        for name in ("approach", "descend", "grasp")
    ] + [
        {
            "phase": "lift",
            "outcome": "finished",
            "object_xyz_m": [0.0, 0.0, start_z + lift_rise],
        },
        {
            "phase": "carry",
            "outcome": "finished",
            "object_xyz_m": [0.0, 0.0, start_z + carry_rise],
        },
    ]


def test_seeded_starts_stay_within_two_centimetres_and_fifteen_degrees():
    base = SCENE["object"]
    starts = [run.start_pose(seed, base) for seed in run.TRIAL_SEEDS]
    assert [start["seed"] for start in starts] == list(range(10))
    for start in starts:
        dx = start["xyz_m"][0] - base["xyz_m"][0]
        dy = start["xyz_m"][1] - base["xyz_m"][1]
        assert math.hypot(dx, dy) <= run.START_SHIFT_M + 1e-12
        assert start["xyz_m"][2] == base["xyz_m"][2]
        assert (
            abs(math.degrees(start["yaw_rad"] - base["yaw_rad"]))
            <= run.START_YAW_DEG + 1e-9
        )
    assert run.start_pose(3, base) == starts[3]
    assert len({tuple(start["xyz_m"]) for start in starts}) == 10


def test_the_grasp_closes_across_the_short_edge_and_reaches_mid_height():
    # Marker long along x: the fingers close along y, which is the hand yaw 0.
    assert run.grasp_yaw(0.0, (0.133, 0.026, 0.026)) == pytest.approx(0.0)
    assert run.grasp_yaw(0.2, (0.133, 0.026, 0.026)) == pytest.approx(0.2)
    assert abs(run.grasp_yaw(0.0, (0.026, 0.133, 0.026))) == pytest.approx(math.pi / 2)
    assert run.control_offset((0.133, 0.026, 0.026)) == pytest.approx(
        run.FINGERTIP_OFFSET_M - 0.013
    )


def test_the_gripper_reads_closed_only_once_the_fingers_stop():
    assert run.gripper_closed(0.08, 0.0, "close") is False
    assert run.gripper_closed(0.05, -0.2, "close") is False
    assert run.gripper_closed(0.026, 0.0, "close") is True
    assert run.gripper_closed(0.0, 0.0, "close") is True
    assert run.gripper_closed(0.05, 0.2, "open") is True
    assert run.gripper_closed(0.079, 0.0, "open") is False


def test_the_success_judgement_reads_a_final_state():
    target = SCENE["target"]
    centre = list(target["xyz_m"])
    resting = run.judge(
        {"object_xyz_m": [centre[0], centre[1], 0.013], "object_speed_m_s": 0.001},
        target,
    )
    assert resting["success"] is True
    moving = run.judge(
        {"object_xyz_m": [centre[0], centre[1], 0.013], "object_speed_m_s": 0.02},
        target,
    )
    assert (
        moving["success"] is False
        and moving["reason"] == "inside the target but still moving"
    )
    outside = run.judge(
        {"object_xyz_m": [centre[0] + 0.06, centre[1], 0.013], "object_speed_m_s": 0.0},
        target,
    )
    assert (
        outside["success"] is False
        and outside["reason"] == "at rest outside the target"
    )


def test_a_failure_is_charged_to_its_phase():
    miss = {"success": False, "reason": "at rest outside the target"}
    timed = [
        {
            "phase": "approach",
            "outcome": "time limit",
            "detail": "end effector 9.0 mm from the phase target",
        }
    ]
    assert run.diagnose(timed, [0, 0, 0.013], miss) == (
        "approach",
        "time limit: end effector 9.0 mm from the phase target",
    )
    phase, reason = run.diagnose(_log(0.013, 0.0, 0.0), [0, 0, 0.013], miss)
    assert phase == "lift" and reason.startswith("object not lifted")
    phase, reason = run.diagnose(_log(0.013, 0.1, 0.0), [0, 0, 0.013], miss)
    assert phase == "carry" and reason.startswith("object dropped")
    assert run.diagnose(_log(0.013, 0.1, 0.1), [0, 0, 0.013], miss) == (
        "release",
        miss["reason"],
    )
    assert run.diagnose([], [0, 0, 0], {"success": True}) == (None, "success")


def _trials(successes):
    target = SCENE["target"]
    trials = []
    for seed in range(10):
        start = run.start_pose(seed, SCENE["object"])
        x = target["xyz_m"][0] if seed < successes else target["xyz_m"][0] + 0.2
        final = {
            "object_xyz_m": [x, target["xyz_m"][1], 0.013],
            "object_speed_m_s": 0.0,
        }
        judgement = run.judge(final, target)
        trials.append(run.trial_record(start, _log(0.013, 0.1, 0.1), final, judgement))
    return trials


def test_the_trials_record_has_the_grid_the_interval_and_an_accept(tmp_path):
    document = run.write_outputs(tmp_path, _trials(3), SCENE, {"note": "fake"})
    saved = json.loads((tmp_path / run.TRIALS_NAME).read_text(encoding="utf-8"))
    assert saved == document
    assert saved["n"] == 10 and saved["successes"] == 3
    assert saved["wilson95"] == pytest.approx(list(wilson_interval(3, 10)))
    assert saved["decision"] == "accept"
    cell = saved["place_grid"]["cell"]
    assert cell["successes"] == 3 and cell["trials"] == 10
    rows = saved["place_grid"]["trials"]
    assert [row["trial"] for row in rows] == list(range(10))
    assert set(rows[0]) == {"trial", "success", "failing_phase", "reason"}
    assert rows[5]["failing_phase"] == "release"
    assert saved["scene"]["primitive"] is True
    assert saved["trials"][0]["phases"][0]["phase"] == "approach"
    ledger = json.loads((tmp_path / run.LEDGER_NAME).read_text(encoding="utf-8"))
    assert [entry["decision"] for entry in ledger["entries"]] == ["accept"]


def test_zero_of_ten_writes_a_reject_and_no_accept(tmp_path):
    document = run.write_outputs(tmp_path, _trials(0), SCENE, {})
    assert document["decision"] == "reject"
    ledger = json.loads((tmp_path / run.LEDGER_NAME).read_text(encoding="utf-8"))
    assert [entry["decision"] for entry in ledger["entries"]] == ["reject"]


def test_the_box_script_loads_the_controller_by_path_without_isaac():
    # PYTHONPATH unset on the box: the script must find this tree's controller
    # and must not import Isaac Lab at module level.
    script = Path(run.__file__).resolve()
    code = (
        "import runpy, sys; m = runpy.run_path(sys.argv[1]); "
        "e = m['_expert'](); print(e.__file__); "
        "print('isaaclab' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(script)],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = result.stdout.split()
    assert Path(lines[0]) == script.parent / "scripted_expert.py"
    assert lines[1] == "False"
