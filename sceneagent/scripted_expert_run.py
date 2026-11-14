# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ten Isaac Lab trials of the eight-phase scripted expert on a Franka arm.

The controller is ``scripted_expert.py``, unchanged: its ``plan`` gives each
phase a target point and a gripper command, and its ``step`` decides when a
phase finishes or times out. This runner only moves the stock Isaac Lab
Franka (``FRANKA_PANDA_HIGH_PD_CFG``) to those points through Isaac Lab's
differential IK controller, and judges the end state with the controller's
own ``episode_success`` (inside the target and slower than 1 cm/s).

The scene is a primitive one, and says so in every output. It is built from
the PutMarkerInCup run ``/workspace/runs/pmic-gemini-e2e1``, which is read,
not written:

- the object is the marker, a box of that run's exported size
  (``export_evidence.json`` ``object_boxes.object_2``) and mass
  (``export/scene_settle.usda`` ``physics:mass``);
- the target region is the paper cup's exported box, standing on the table,
  at the cup's offset from the marker in that export. No cup mesh is loaded,
  so nothing physically holds the marker inside the region;
- friction is PhysX's default, as in that export (``sceneagent:defaults``).

The cup itself is not the object because ``plan`` reports it as wider than
the gripper opening. That verdict is recorded beside the trials.

Seeds 0 to 9 each move the marker's start by up to 2 cm and turn it by up to
15 degrees. Every trial writes a phase log. Trial 0 also writes frames and,
when ffmpeg is present, ``trial0.mp4``. Zero of ten is a reject.

Run on the GPU box only, under the GPU lock::

    flock /workspace/runs/.gpu.lock env -u PYTHONPATH \\
        /root/.local/share/mamba/envs/isaaclab/bin/python3 \\
        sceneagent/scripted_expert_run.py --out /workspace/runs/kan27-<tag>

The box runs it with ``PYTHONPATH`` unset, so ``sceneagent`` is not a
package there. The controller is then loaded by file path, and every Isaac
import stays inside a function so the CPU tests never load Kit.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

ITEM = "scripted expert"
TRIALS_NAME = "expert_trials.json"
LEDGER_NAME = "ledger.json"
FRAMES_DIR = "trial0_frames"
LOGS_DIR = "phase_logs"
VIDEO_NAME = "trial0.mp4"
SOURCE_RUN = "/workspace/runs/pmic-gemini-e2e1"

TRIAL_SEEDS = tuple(range(10))
START_SHIFT_M = 0.02
START_YAW_DEG = 15.0

# Marker and paper cup from the PutMarkerInCup export. The xy offset is the
# difference of the two prims' xform translates, rounded to a millimetre.
MARKER_SIZE_M = (0.133, 0.026, 0.026)
MARKER_MASS_KG = 0.310
CUP_SIZE_M = (0.094, 0.096, 0.095)
CUP_FROM_MARKER_XY_M = (0.059, -0.243)
# Ours: where the marker starts in front of the arm's base.
MARKER_START_XY_M = (0.45, 0.12)
BASE_YAW_RAD = 0.0

# Franka hand frame to the fingertip pad centre (Isaac Lab's Franka tasks use
# the same 0.1034 m TCP offset).
FINGERTIP_OFFSET_M = 0.1034
# The gripper is "open" again once the fingers are this far apart. Ours.
OPEN_WIDTH_M = 0.075
# The fingers have stopped once the width changes slower than this. Ours.
FINGERS_STOPPED_M_S = 0.005
# Ours: how far the commanded point may move per second toward a phase target.
COMMAND_SPEED_M_S = 0.35
# Ours: the object rose this much by the end of lift, so it was held.
HELD_RISE_M = 0.03
# Ours: after retreat the arm holds still this long before the end state is read.
SETTLE_S = 1.0
PHYSICS_DT_S = 1.0 / 120.0
FRAME_EVERY_STEPS = 6


def scene_spec() -> dict[str, Any]:
    """The primitive scene: object box, target region and where they came from."""
    table_z = 0.0
    marker = {
        "name": "marker",
        "xyz_m": [
            MARKER_START_XY_M[0],
            MARKER_START_XY_M[1],
            table_z + MARKER_SIZE_M[2] / 2,
        ],
        "size_m": list(MARKER_SIZE_M),
        "yaw_rad": BASE_YAW_RAD,
        "mass_kg": MARKER_MASS_KG,
    }
    target = {
        "name": "paper cup region",
        "xyz_m": [
            MARKER_START_XY_M[0] + CUP_FROM_MARKER_XY_M[0],
            MARKER_START_XY_M[1] + CUP_FROM_MARKER_XY_M[1],
            table_z + CUP_SIZE_M[2] / 2,
        ],
        "size_m": list(CUP_SIZE_M),
    }
    return {
        "task": "PutMarkerInCup, primitive",
        "primitive": True,
        "source_run": SOURCE_RUN,
        "object": marker,
        "target": target,
        "table_z_m": table_z,
        "notes": [
            "object: a box of the marker's exported size and mass, not its mesh",
            (
                "target: the paper cup's exported box as a region on the table; "
                "no cup mesh, so nothing holds the marker in"
            ),
            "friction: PhysX default, as in the source export",
            (
                "layout: the cup's offset from the marker in the source export; "
                "the marker's place in front of the arm is ours"
            ),
        ],
    }


def start_pose(seed: int, base: dict[str, Any]) -> dict[str, Any]:
    """The seeded start: up to 2 cm in the plane and 15 degrees of yaw.

    The shift is uniform over the 2 cm disc, not over its radius.
    """
    rng = random.Random(int(seed))
    radius = START_SHIFT_M * math.sqrt(rng.random())
    heading = rng.uniform(0.0, 2.0 * math.pi)
    turn_deg = rng.uniform(-START_YAW_DEG, START_YAW_DEG)
    xyz = [float(value) for value in base["xyz_m"]]
    xyz[0] += radius * math.cos(heading)
    xyz[1] += radius * math.sin(heading)
    return {
        "seed": int(seed),
        "xyz_m": xyz,
        "yaw_rad": float(base.get("yaw_rad", 0.0)) + math.radians(turn_deg),
        "shift_m": radius,
        "turn_deg": turn_deg,
    }


def grasp_yaw(object_yaw: float, size_m: list[float] | tuple[float, ...]) -> float:
    """Hand yaw that closes the fingers across the shorter horizontal edge.

    The hand points down (a half turn about x), so its finger axis is the
    world -y axis turned by the yaw. The result is wrapped to (-pi/2, pi/2]:
    the two fingers are symmetric, and the smaller turn keeps joint 7 near home.
    """
    short = object_yaw + (math.pi / 2.0 if size_m[1] <= size_m[0] else 0.0)
    yaw = short + math.pi / 2.0
    while yaw > math.pi / 2.0:
        yaw -= math.pi
    while yaw <= -math.pi / 2.0:
        yaw += math.pi
    return yaw


def control_offset(size_m: list[float] | tuple[float, ...]) -> float:
    """Hand-frame distance to the point the controller's targets name.

    ``plan`` sends that point to the object's top for the grasp. The pads must
    then sit at the object's mid-height, so the point is half the object's
    height above the pad centre.
    """
    return FINGERTIP_OFFSET_M - float(size_m[2]) / 2.0


def gripper_closed(width_m: float, width_speed_m_s: float, command: str) -> bool:
    """What ``step`` reads as ``gripper_closed``.

    Closing: the fingers have stopped, on the object or on each other, short
    of fully open. Opening: closed until the fingers are apart again.
    """
    if command == "close":
        return width_m < OPEN_WIDTH_M and abs(width_speed_m_s) < FINGERS_STOPPED_M_S
    return width_m < OPEN_WIDTH_M


def judge(final: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    """The paper's rule on a final state, through the controller's own test."""
    expert = _expert()
    centre = [float(value) for value in target["xyz_m"]]
    half = [float(value) / 2.0 for value in target["size_m"]]
    xyz = [float(value) for value in final["object_xyz_m"]]
    speed = float(final["object_speed_m_s"])
    success = expert.episode_success(xyz, centre, half, speed)
    inside = all(abs(xyz[i] - centre[i]) <= half[i] for i in range(3))
    at_rest = speed < expert.REST_SPEED_M_S
    if success:
        reason = "inside the target and at rest"
    elif not inside and not at_rest:
        reason = "outside the target and still moving"
    elif not inside:
        reason = "at rest outside the target"
    else:
        reason = "inside the target but still moving"
    return {
        "success": bool(success),
        "inside": inside,
        "at_rest": at_rest,
        "speed_m_s": speed,
        "offset_m": [xyz[i] - centre[i] for i in range(3)],
        "reason": reason,
    }


def diagnose(
    log: list[dict[str, Any]],
    start_xyz: list[float],
    judgement: dict[str, Any],
) -> tuple[str | None, str]:
    """The phase a failed trial is charged to, and why.

    A phase that timed out is charged with its own failure. Otherwise the
    object's height after lift says whether it was held, its height after
    carry whether it was still held, and the judgement covers the rest.
    """
    if judgement.get("success"):
        return None, "success"
    for entry in log:
        if entry.get("outcome") == "time limit":
            return str(entry["phase"]), "time limit: " + str(entry.get("detail", ""))
    by_phase = {entry["phase"]: entry for entry in log}
    start_z = float(start_xyz[2])
    lift = by_phase.get("lift")
    if lift is not None:
        rise = float(lift["object_xyz_m"][2]) - start_z
        if rise < HELD_RISE_M:
            return (
                "lift",
                f"object not lifted: rose {rise * 100:.1f} cm by the end of lift",
            )
    carry = by_phase.get("carry")
    if carry is not None:
        rise = float(carry["object_xyz_m"][2]) - start_z
        if rise < HELD_RISE_M:
            return (
                "carry",
                f"object dropped during carry: {rise * 100:.1f} cm above its start",
            )
    return "release", str(judgement.get("reason", "not a success"))


def trial_record(
    start: dict[str, Any],
    log: list[dict[str, Any]],
    final: dict[str, Any],
    judgement: dict[str, Any],
) -> dict[str, Any]:
    """One trial as ``expert_trials.json`` stores it."""
    phase, reason = diagnose(log, start["xyz_m"], judgement)
    return {
        "trial": int(start["seed"]),
        "seed": int(start["seed"]),
        "start": start,
        "success": bool(judgement["success"]),
        "failing_phase": phase,
        "reason": reason,
        "final": final,
        "judgement": judgement,
        "phases": log,
    }


def place_grid(document: dict[str, Any], scene: dict[str, Any]) -> dict[str, Any]:
    """The place-grid excerpt: one row, one lighting column, ten trials."""
    n = int(document["n"])
    successes = int(document["successes"])
    return {
        "policy": "scripted_expert_isaaclab_franka",
        "row": f"{scene['object']['name']} -> {scene['target']['name']} (original object, no sisters)",
        "column": "one dome light; the website's nine lighting setups are not rendered",
        "primitive_scene": bool(scene.get("primitive")),
        "cell": {
            "successes": successes,
            "trials": n,
            "rate": successes / n,
            "wilson95": list(document["wilson95"]),
        },
        "trials": [
            {
                "trial": trial["trial"],
                "success": trial["success"],
                "failing_phase": trial["failing_phase"],
                "reason": trial["reason"],
            }
            for trial in document["trials"]
        ],
    }


def write_outputs(
    out: Path,
    trials: list[dict[str, Any]],
    scene: dict[str, Any],
    extra: dict[str, Any],
) -> dict[str, Any]:
    """``expert_trials.json`` and the ledger. Zero of ten is a reject."""
    expert = _expert()
    ledger_path = out / LEDGER_NAME
    document = expert.record_trials(trials, ledger_path)
    document["place_grid"] = place_grid(document, scene)
    document["scene"] = scene
    document.update(extra)
    if document.get("decision") != "reject":
        low, high = document["wilson95"]
        document["decision"] = "accept"
        _append_entry(
            ledger_path,
            {
                "item": ITEM,
                "decision": "accept",
                "reason": f"{document['successes']} of {document['n']} trials put the "
                f"object inside the target at rest; Wilson 95% {low:.3f} to {high:.3f}",
            },
        )
    (out / TRIALS_NAME).write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )
    return document


def _append_entry(path: Path, entry: dict[str, Any]) -> None:
    try:
        ledger = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        ledger = {"entries": []}
    ledger.setdefault("scene_name", "scripted expert")
    ledger["entries"] = list(ledger.get("entries") or []) + [entry]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")


def _expert() -> Any:
    """The merged controller: the package copy, or this tree's file on the box."""
    if __package__:
        return importlib.import_module(f"{__package__}.scripted_expert")
    return _load_by_path()


def _load_by_path() -> Any:
    """Load ``scripted_expert.py`` and its ``metrics`` import from this directory.

    With ``PYTHONPATH`` unset, ``import sceneagent`` would find another clone
    if one were installed. Loading by path pins this tree's files.
    """
    here = Path(__file__).resolve().parent
    cached = sys.modules.get("_sf_scripted_expert")
    if cached is not None:
        return cached
    if "sceneagent.metrics" not in sys.modules:
        sys.modules["sceneagent.metrics"] = _module(
            "sceneagent.metrics", here / "metrics.py"
        )
    module = _module("_sf_scripted_expert", here / "scripted_expert.py")
    sys.modules["_sf_scripted_expert"] = module
    return module


def _module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--out" not in args:
        print("usage: scripted_expert_run.py --out DIR", file=sys.stderr)
        return 2
    out = Path(args[args.index("--out") + 1])
    out.mkdir(parents=True, exist_ok=True)
    # A crashed rerun must not leave the last run's result to be read.
    for name in (LEDGER_NAME, TRIALS_NAME, VIDEO_NAME):
        (out / name).unlink(missing_ok=True)
    shutil.rmtree(out / FRAMES_DIR, ignore_errors=True)
    shutil.rmtree(out / LOGS_DIR, ignore_errors=True)
    scene = scene_spec()
    expert = _expert()
    made = expert.plan(scene["object"], scene["target"])
    if not made["graspable"]:
        _append_entry(
            out / LEDGER_NAME,
            {
                "item": ITEM,
                "decision": "reject",
                "reason": "object not graspable: " + made["reason"],
            },
        )
        return 1
    try:
        _launch(out, scene)
    except Exception:
        _append_entry(
            out / LEDGER_NAME,
            {
                "item": ITEM,
                "decision": "reject",
                "reason": "run failed: " + traceback.format_exc()[-800:],
            },
        )
        print(traceback.format_exc(), flush=True)
        return 1
    return 0


def _launch(out: Path, scene: dict[str, Any]) -> None:
    """Start Kit, run the ten trials, and exit without waiting on app close."""
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    import argparse

    from isaaclab.app import add_launcher_args, launch_simulation

    parser = argparse.ArgumentParser()
    add_launcher_args(parser)
    launcher_args = parser.parse_args([])
    launcher_args.headless = True
    launcher_args.enable_cameras = True
    import isaaclab.sim as sim_utils

    sim_cfg = sim_utils.SimulationCfg(dt=PHYSICS_DT_S)
    with launch_simulation(sim_cfg, launcher_args):
        code = 0
        try:
            _run_trials(out, scene, sim_cfg)
        except Exception:
            code = 1
            _append_entry(
                out / LEDGER_NAME,
                {
                    "item": ITEM,
                    "decision": "reject",
                    "reason": "run failed: " + traceback.format_exc()[-800:],
                },
            )
            print(traceback.format_exc(), flush=True)
        sys.stdout.flush()
        # SimulationApp.close has hung on this box; the outputs are written.
        os._exit(code)


def _run_trials(out: Path, scene: dict[str, Any], sim_cfg: Any) -> None:
    import isaaclab.sim as sim_utils
    import torch
    from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
    from isaaclab.controllers import (
        DifferentialIKController,
        DifferentialIKControllerCfg,
    )
    from isaaclab.sensors.camera import Camera, CameraCfg
    from isaaclab.utils import replace
    from isaaclab.utils.math import (
        quat_apply,
        quat_from_euler_xyz,
        subtract_frame_transforms,
    )
    from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # type: ignore[import-not-found]
    from isaaclab_physx.renderers import IsaacRtxRendererCfg

    expert = _expert()
    png_bytes = _module(
        "_sf_png", Path(__file__).resolve().parent / "isaaclab_headless_check.py"
    ).png_bytes
    sim = sim_utils.SimulationContext(sim_cfg)
    device = sim.device
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)
    light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.85, 0.85, 0.85))
    light.func("/World/Light", light)
    target = scene["target"]
    # A visual-only patch marks the target region's footprint; it has no collider.
    patch = sim_utils.CuboidCfg(
        size=(target["size_m"][0], target["size_m"][1], 0.002),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.7, 0.2)),
    )
    patch.func(
        "/World/target_patch",
        patch,
        translation=(target["xyz_m"][0], target["xyz_m"][1], 0.001),
    )
    robot = Articulation(replace(FRANKA_PANDA_HIGH_PD_CFG, prim_path="/World/Robot"))
    obj_spec = scene["object"]
    marker = RigidObject(
        RigidObjectCfg(
            prim_path="/World/Marker",
            spawn=sim_utils.CuboidCfg(
                size=tuple(obj_spec["size_m"]),
                rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
                mass_props=sim_utils.MassCfg(mass=float(obj_spec["mass_kg"])),
                collision_props=sim_utils.UsdPhysicsCollisionCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.8, 0.1, 0.1)
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(obj_spec["xyz_m"])),
        )
    )
    camera = Camera(
        CameraCfg(
            prim_path="/World/Camera",
            update_period=0,
            height=480,
            width=640,
            data_types=["rgb"],
            renderer_cfg=IsaacRtxRendererCfg(),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 100.0),
            ),
        )
    )
    sim.reset()
    camera.set_world_poses_from_view(
        torch.tensor([[1.25, -0.75, 0.65]], device=device),
        torch.tensor([[0.45, 0.0, 0.05]], device=device),
    )
    arm_ids, _ = robot.find_joints("panda_joint[1-7]")
    finger_ids, finger_names = robot.find_joints("panda_finger_joint.*")
    driven = [
        finger_ids[i] for i, name in enumerate(finger_names) if name.endswith("1")
    ]
    hand_id = robot.find_bodies("panda_hand")[0][0]
    jacobian_index = hand_id - 1 if robot.is_fixed_base else hand_id
    jacobian_joints = [j + robot.num_base_dofs for j in arm_ids]
    ik = DifferentialIKController(
        DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls"
        ),
        num_envs=1,
        device=device,
    )
    command = robot.actuators.target_command
    dt = sim.get_physics_dt()
    offset = control_offset(obj_spec["size_m"])
    trials = []
    frames = 0

    def step_sim(render: bool) -> None:
        robot.write_data_to_sim()
        marker.write_data_to_sim()
        sim.step(render=render)
        robot.update(dt)
        marker.update(dt)

    def hand_state():
        pose = robot.data.body_pose_w.torch[0, hand_id]
        point = (
            pose[:3]
            + quat_apply(
                pose[3:7].unsqueeze(0),
                torch.tensor([[0.0, 0.0, offset]], device=device),
            )[0]
        )
        return pose, point

    def width_now() -> float:
        return float(robot.data.joint_pos.torch[0, finger_ids].sum())

    for seed in TRIAL_SEEDS:
        start = start_pose(seed, obj_spec)
        quat = quat_from_euler_xyz(
            torch.tensor([0.0], device=device),
            torch.tensor([0.0], device=device),
            torch.tensor([start["yaw_rad"]], device=device),
        )[0]
        root = torch.tensor([*start["xyz_m"], *quat.tolist()], device=device).unsqueeze(
            0
        )
        marker.write_root_pose_to_sim_index(root_pose=root)
        marker.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros(1, 6, device=device)
        )
        joint_pos = robot.data.default_joint_pos.torch.clone()
        robot.write_joint_position_to_sim_index(position=joint_pos)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint_pos))
        robot.reset()
        marker.reset()
        ik.reset()
        command.set_position_index(value=joint_pos[:, arm_ids], joint_ids=arm_ids)
        command.set_position_index(
            value=torch.full((1, len(driven)), 0.04, device=device), joint_ids=driven
        )
        for _ in range(12):
            step_sim(render=False)
        obj_now = {
            **obj_spec,
            "xyz_m": marker.data.root_pos_w.torch[0].tolist(),
            "yaw_rad": start["yaw_rad"],
        }
        made = expert.plan(obj_now, target)
        yaw = grasp_yaw(start["yaw_rad"], obj_spec["size_m"])
        hand_quat = quat_from_euler_xyz(
            torch.tensor([math.pi], device=device),
            torch.tensor([0.0], device=device),
            torch.tensor([yaw], device=device),
        )[0]
        state = {"phases": made["phases"], "phase_index": 0, "elapsed_s": 0.0}
        _, point = hand_state()
        commanded = point.clone()
        last_width = width_now()
        log: list[dict[str, Any]] = []
        frame_dir = out / FRAMES_DIR
        if seed == 0:
            frame_dir.mkdir(parents=True, exist_ok=True)
        tick = 0
        while True:
            pose, point = hand_state()
            width = width_now()
            width_speed = (width - last_width) / dt
            last_width = width
            index = int(state["phase_index"])
            phase = made["phases"][min(index, len(made["phases"]) - 1)]
            observation = {
                "ee_xyz_m": point.tolist(),
                "gripper_closed": gripper_closed(width, width_speed, phase["gripper"]),
            }
            action = expert.step(state, observation)
            if action["done"]:
                break
            error = float(
                torch.linalg.norm(
                    point - torch.tensor(phase["ee_xyz_m"], device=device)
                )
            )
            if action["failed"]:
                log.append(
                    _phase_entry(
                        action["phase"], "time limit", state, error, width, marker
                    )
                )
                log[-1]["detail"] = (
                    f"gripper {'closed' if observation['gripper_closed'] else 'not closed'}, width {width * 1000:.1f} mm"
                    if phase["name"] in ("grasp", "release")
                    else f"end effector {error * 1000:.1f} mm from the phase target"
                )
                break
            if action["advance"]:
                log.append(
                    _phase_entry(
                        action["phase"], "finished", state, error, width, marker
                    )
                )
                state["phase_index"] = index + 1
                state["elapsed_s"] = 0.0
                continue
            goal = torch.tensor(action["ee_xyz_m"], device=device)
            gap = goal - commanded
            reach = COMMAND_SPEED_M_S * dt
            distance = float(torch.linalg.norm(gap))
            commanded = (
                goal.clone()
                if distance <= reach
                else commanded + gap * (reach / distance)
            )
            hand_target = (
                commanded
                - quat_apply(
                    hand_quat.unsqueeze(0),
                    torch.tensor([[0.0, 0.0, offset]], device=device),
                )[0]
            )
            root_pose = robot.data.root_pose_w.torch
            target_b, quat_b = subtract_frame_transforms(
                root_pose[:, 0:3],
                root_pose[:, 3:7],
                hand_target.unsqueeze(0),
                hand_quat.unsqueeze(0),
            )
            ik.set_command(torch.cat([target_b, quat_b], dim=1))
            hand_b, hand_quat_b = subtract_frame_transforms(
                root_pose[:, 0:3],
                root_pose[:, 3:7],
                pose[:3].unsqueeze(0),
                pose[3:7].unsqueeze(0),
            )
            jacobian = robot.data.body_link_jacobian_w.torch[
                :, jacobian_index, :, jacobian_joints
            ]
            arm_now = robot.data.joint_pos.torch[:, arm_ids]
            command.set_position_index(
                value=ik.compute(hand_b, hand_quat_b, jacobian, arm_now),
                joint_ids=arm_ids,
            )
            finger = 0.0 if action["gripper"] == "close" else 0.04
            command.set_position_index(
                value=torch.full((1, len(driven)), finger, device=device),
                joint_ids=driven,
            )
            render = seed == 0 and tick % FRAME_EVERY_STEPS == 0
            step_sim(render=render)
            if render:
                frames = _save_frame(camera, dt, frame_dir, frames, png_bytes)
            tick += 1
            state["elapsed_s"] = float(state["elapsed_s"]) + dt
        # Hold the last command with the gripper open, then read the end state.
        for settle_tick in range(round(SETTLE_S / dt)):
            render = seed == 0 and settle_tick % FRAME_EVERY_STEPS == 0
            step_sim(render=render)
            if render:
                frames = _save_frame(camera, dt, frame_dir, frames, png_bytes)
        final = {
            "object_xyz_m": marker.data.root_pos_w.torch[0].tolist(),
            "object_quat_xyzw": marker.data.root_quat_w.torch[0].tolist(),
            "object_speed_m_s": float(
                torch.linalg.norm(marker.data.root_lin_vel_w.torch[0])
            ),
            "settle_s": SETTLE_S,
        }
        judgement = judge(final, target)
        record = trial_record(start, log, final, judgement)
        trials.append(record)
        # Written as each trial ends, so a run cut short keeps its logs.
        logs = out / LOGS_DIR
        logs.mkdir(parents=True, exist_ok=True)
        (logs / f"trial_{seed}.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"trial {seed}: success={record['success']} phase={record['failing_phase']} "
            f"reason={record['reason']}",
            flush=True,
        )
    cup = {"xyz_m": target["xyz_m"], "size_m": target["size_m"]}
    extra = {
        "cup_as_object": expert.plan(cup, target),
        "frames": {
            "dir": FRAMES_DIR,
            "count": frames,
            "every_steps": FRAME_EVERY_STEPS,
        },
        "physics_dt_s": dt,
        "robot": "isaaclab_assets FRANKA_PANDA_HIGH_PD_CFG",
        "ik": "isaaclab.controllers DifferentialIKController, pose, absolute, dls",
        "control_point_offset_m": offset,
        "command_speed_m_s": COMMAND_SPEED_M_S,
    }
    document = write_outputs(out, trials, scene, extra)
    _encode_video(out)
    print(
        f"scripted expert {document['decision']}: {document['successes']} of {document['n']}, "
        f"Wilson95 {document['wilson95']}",
        flush=True,
    )


def _phase_entry(
    phase: str,
    outcome: str,
    state: dict[str, Any],
    error: float,
    width: float,
    marker: Any,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "outcome": outcome,
        "elapsed_s": round(float(state["elapsed_s"]), 4),
        "ee_error_m": error,
        "gripper_width_m": width,
        "object_xyz_m": marker.data.root_pos_w.torch[0].tolist(),
    }


def _save_frame(
    camera: Any, dt: float, folder: Path, count: int, png_bytes: Any
) -> int:
    camera.update(dt)
    frame = camera.data.output["rgb"].torch[0, ..., :3]
    if frame.dtype.is_floating_point:
        frame = (frame.clamp(0, 1) * 255).round()
    rgb = frame.to("cpu").numpy().astype("uint8")
    (folder / f"frame_{count:04d}.png").write_bytes(png_bytes(rgb))
    return count + 1


def _encode_video(out: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None or not any((out / FRAMES_DIR).glob("frame_*.png")):
        return
    rate = str(round(1.0 / (PHYSICS_DT_S * FRAME_EVERY_STEPS)))
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            rate,
            "-i",
            str(out / FRAMES_DIR / "frame_%04d.png"),
            "-pix_fmt",
            "yuv420p",
            str(out / VIDEO_NAME),
        ],
        check=False,
    )


if __name__ == "__main__":
    sys.exit(main())
