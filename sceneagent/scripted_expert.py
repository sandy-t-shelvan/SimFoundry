# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Eight-phase scripted place controller. No simulator is imported here.

Phases, in order: approach, descend, grasp, lift, carry, lower, release,
retreat. A phase finishes when the end effector is within 5 mm of its
target, or when the gripper has done what that phase asked. A time limit
fails the episode. It does not skip to the next phase.

An object longer than the target's opening cannot go in lying down -- a
13 cm marker into a 9.5 cm cup. ``plan`` then grips it near one end and adds
one phase, ``upright``, after the lift: the hand turns 90 degrees about its
finger axis so the object hangs straight down, and is lowered until its
bottom is just above the target's floor. Every other object gets the eight
phases unchanged.

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
UPRIGHT_PHASES = PHASES[:4] + ("upright",) + PHASES[4:]
UPRIGHT_PITCH_RAD = 1.5707963267948966  # the hand turns a quarter turn
ANGLE_TOLERANCE_RAD = 0.087  # 5 degrees
GRIP_FROM_END_M = 0.02  # never grip closer than this to the object's end
FINGER_CLEARANCE_M = 0.015  # the fingertips stay this far above the rim
FLOOR_M = 0.01  # a container's floor thickness, assumed
ABOVE_FLOOR_M = 0.01  # release with the object's bottom this far above it
RIM_CLEARANCE_M = 0.03
UPRIGHT_LIMIT_S = 6.0  # the turn plus bringing the fingertips back takes longer
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
    long_side = max(size[0], size[1])
    opening = min(float(target["size_m"][0]), float(target["size_m"][1]))
    if long_side > opening:
        return _upright_plan(obj, target, size, origin, goal)
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


def _upright_plan(obj, target, size, origin, goal):
    """Grip near one end, turn the object upright, lower it in end first."""
    import math

    yaw = float(obj.get("yaw_rad", 0.0))
    # The long axis in the plane, from the object's yaw.
    if size[0] >= size[1]:
        axis = (math.cos(yaw), math.sin(yaw))
    else:
        axis = (-math.sin(yaw), math.cos(yaw))
    long_side = max(size[0], size[1])
    top = origin[2] + size[2] / 2.0
    target_size = [float(value) for value in target["size_m"]]
    floor = goal[2] - target_size[2] / 2.0 + FLOOR_M
    rim = goal[2] + target_size[2] / 2.0
    # Grip as near the middle as the target's depth allows: the further from
    # the centre of mass, the harder the object twists out of the fingers as
    # it is lifted (gripped 2 cm from its end, the marker did, every trial).
    # The hang only has to keep the fingertips above the rim with the
    # object's bottom just above the floor.
    needed_hang = rim + FINGER_CLEARANCE_M - (floor + ABOVE_FLOOR_M)
    grip_offset = min(
        max(needed_hang - long_side / 2.0, 0.005), long_side / 2.0 - GRIP_FROM_END_M
    )
    hang = grip_offset + long_side / 2.0  # from the grip down to the far end
    grip = [origin[0] + grip_offset * axis[0], origin[1] + grip_offset * axis[1]]
    high = max(top + CLEARANCE_M, rim + hang + RIM_CLEARANCE_M)
    # Released with the fingertips above the rim; if the object is too short
    # for that, its bottom ends higher than ABOVE_FLOOR_M and it drops further.
    lowered = max(floor + ABOVE_FLOOR_M + hang, rim + FINGER_CLEARANCE_M)
    poses = {
        "approach": ([grip[0], grip[1], top + CLEARANCE_M], 0.0),
        "descend": ([grip[0], grip[1], top], 0.0),
        "grasp": ([grip[0], grip[1], top], 0.0),
        "lift": ([grip[0], grip[1], high], 0.0),
        "upright": ([grip[0], grip[1], high], UPRIGHT_PITCH_RAD),
        "carry": ([goal[0], goal[1], high], UPRIGHT_PITCH_RAD),
        "lower": ([goal[0], goal[1], lowered], UPRIGHT_PITCH_RAD),
        "release": ([goal[0], goal[1], lowered], UPRIGHT_PITCH_RAD),
        "retreat": ([goal[0], goal[1], high], UPRIGHT_PITCH_RAD),
    }
    phases = []
    for name in UPRIGHT_PHASES:
        xyz, pitch = poses[name]
        phases.append(
            {
                "name": name,
                "ee_xyz_m": xyz,
                "hand_pitch_rad": pitch,
                "gripper": "close"
                if name in ("grasp", "lift", "upright", "carry", "lower")
                else "open",
                "time_limit_s": UPRIGHT_LIMIT_S if name == "upright" else PHASE_LIMIT_S,
            }
        )
    return {
        "graspable": True,
        "upright": True,
        "reason": f"longer ({long_side:.3f} m) than the target's opening ({min(target_size[:2]):.3f} m)",
        "grip_offset_m": grip_offset,
        "long_axis_xy": list(axis),
        "hang_m": hang,
        "phases": phases,
    }


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
        "hand_pitch_rad": float(phase.get("hand_pitch_rad", 0.0)),
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
    """Success count and a 95% Wilson interval. Zero successes is a reject."""
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
    if successes == 0:  # zero successes is a reject at any trial count
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
    if phase["name"] == "lower" and observation.get("blocked"):
        # A guarded move: stop where the object meets something (a tapered
        # cup's wall) and let release settle it, rather than time out pushing.
        return True
    if phase["name"] == "upright":
        # The turn itself: done when the hand is within 5 degrees of upright.
        angle = observation.get("pitch_error_rad")
        if angle is None or float(angle) > ANGLE_TOLERANCE_RAD:
            return False
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
