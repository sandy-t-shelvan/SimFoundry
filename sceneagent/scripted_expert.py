# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Eight-phase scripted place controller. No simulator is imported here.

Phases, in order: approach, descend, grasp, lift, carry, lower, release,
retreat. A phase finishes when the end effector is within 5 mm of its
target, or when the gripper has done what that phase asked. A time limit
fails the episode. It does not skip to the next phase.

An episode succeeds only when the object is inside the target and slower
than 1 cm/s. The gripper opening used for the grasp check is this tree's.
"""

from __future__ import annotations

import json
from pathlib import Path

from sceneagent.metrics import wilson_interval

PHASES = (
    "approach",
    "descend",
    "grasp",
    "lift",
    "carry",
    "lower",
    "release",
    "retreat",
)
POSITION_TOLERANCE_M = 0.005
REST_SPEED_M_S = 0.01
GRIPPER_OPENING_M = 0.08
PHASE_LIMIT_S = 2.0
CLEARANCE_M = 0.1


def plan(obj, target):
    """Phase targets from the object box and the target box.

    ``obj`` and ``target`` have ``xyz_m`` and ``size_m`` (full edge lengths).
    The grasp is top-down across the shorter horizontal edge. An object wider
    than the gripper opening is not tried.
    """
    size = [float(value) for value in obj["size_m"]]
    if min(size[0], size[1]) > GRIPPER_OPENING_M:
        return {
            "graspable": False,
            "reason": "wider than the gripper opening",
            "phases": [],
        }
    origin = [float(value) for value in obj["xyz_m"]]
    goal = [float(value) for value in target["xyz_m"]]
    top = origin[2] + size[2] / 2.0
    above = top + CLEARANCE_M
    target_top = goal[2] + float(target["size_m"][2]) / 2.0 + 0.02
    poses = {
        "approach": [origin[0], origin[1], above],
        "descend": [origin[0], origin[1], top],
        "grasp": [origin[0], origin[1], top],
        "lift": [origin[0], origin[1], above],
        "carry": [goal[0], goal[1], above],
        "lower": [goal[0], goal[1], target_top],
        "release": [goal[0], goal[1], target_top],
        "retreat": [goal[0], goal[1], above],
    }
    phases = []
    for name in PHASES:
        phases.append(
            {
                "name": name,
                "ee_xyz_m": poses[name],
                "gripper": "close"
                if name in ("grasp", "lift", "carry", "lower")
                else "open",
                "time_limit_s": PHASE_LIMIT_S,
            }
        )
    return {"graspable": True, "phases": phases}


def step(state, observation):
    """One control step. A timeout returns ``failed`` and does not advance."""
    phases = state["phases"]
    index = int(state["phase_index"])
    if index >= len(phases):
        return {"done": True, "failed": False, "phase": PHASES[-1]}
    phase = phases[index]
    elapsed = float(state.get("elapsed_s", 0.0))
    if elapsed > float(phase["time_limit_s"]):
        return {
            "done": False,
            "failed": True,
            "advance": False,
            "phase": phase["name"],
            "reason": "time limit",
        }
    action = {
        "done": False,
        "failed": False,
        "advance": _finished(phase, observation),
        "phase": phase["name"],
        "ee_xyz_m": list(phase["ee_xyz_m"]),
        "gripper": phase["gripper"],
    }
    return action


def episode_success(object_xyz, target_xyz, target_half, speed):
    """Inside the target box and at rest. Either condition alone is a miss."""
    inside = all(
        abs(float(object_xyz[i]) - float(target_xyz[i])) <= float(target_half[i])
        for i in range(3)
    )
    return bool(inside and float(speed) < REST_SPEED_M_S)


def record_trials(trials, ledger_path=None):
    """Success count and a 95% Wilson interval. Zero of ten is a reject."""
    count = len(trials)
    if count < 1:
        raise ValueError("trials must not be empty")
    successes = sum(1 for trial in trials if trial.get("success"))
    low, high = wilson_interval(successes, count)
    document = {
        "n": count,
        "successes": successes,
        "wilson95": [low, high],
        "trials": list(trials),
    }
    if count == 10 and successes == 0:
        document["decision"] = "reject"
        if ledger_path is not None:
            _append_reject(Path(ledger_path))
    return document


def _finished(phase, observation):
    if phase["name"] == "grasp":
        return bool(observation.get("gripper_closed"))
    if phase["name"] == "release":
        return not bool(observation.get("gripper_closed"))
    error = _distance(observation["ee_xyz_m"], phase["ee_xyz_m"])
    return error < POSITION_TOLERANCE_M


def _distance(left, right):
    return (
        sum((float(a) - float(b)) ** 2 for a, b in zip(left, right, strict=True)) ** 0.5
    )


def _append_reject(path: Path) -> None:
    try:
        ledger = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        ledger = {"entries": []}
    entries = list(ledger.get("entries") or [])
    entries.append(
        {
            "item": "scripted expert",
            "decision": "reject",
            "reason": "0 of 10 trials put the object inside the target at rest",
        }
    )
    ledger["entries"] = entries
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
