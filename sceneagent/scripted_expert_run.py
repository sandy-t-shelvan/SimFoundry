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
  (``export_evidence.json`` ``object_boxes.object_2``) and the pipeline's
  mass for it (``MARKER_MASS_SOURCE``: density x placed volume, KAN-40);
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
SOURCE_RUN = "/workspace/runs/gate-pmic-e2e1"

TRIAL_SEEDS = tuple(range(10))
START_SHIFT_M = 0.02
START_YAW_DEG = 15.0

# Marker and paper cup from the PutMarkerInCup export. The xy offset is the
# difference of the two prims' xform translates, rounded to a millimetre.
MARKER_SIZE_M = (0.133, 0.025, 0.023)  # gate-pmic-e2e1 export, object_2
# KAN-40: the physics stage's mass for that marker, density x the volume its
# placed mesh encloses. It was 0.265, the prior's log-mass slot, which does
# not see metric size; that marker slipped from the gripper 10/10 (KAN-27).
MARKER_MASS_KG = 0.0203
MARKER_MASS_SOURCE = "/workspace/runs/kan40-mass-pmic1/physics/object_2.json"
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
# The upright turn: the commanded hand orientation turns toward its target at
# this rate, so the held object swings over instead of being flung.
ANGULAR_SPEED_RAD_S = 1.5
BLOCKED_S = 0.3  # lower is blocked when the fingertips stall this long
POSTURE_GAIN = 0.05  # per step, toward the home posture, in the IK's null space only
PHYSICS_DT_S = 1.0 / 120.0
FRAME_EVERY_STEPS = 6


def scene_spec(cup_mesh: Path | str | None = None) -> dict[str, Any]:
    """The scene: object box, the target, and where they came from.

    With ``cup_mesh`` (the scanned paper cup, metres) the target is that mesh,
    a static triangle collider, so the marker has to go in through the rim;
    its box is the target the paper's success rule checks. Without it the
    target is the cup's exported box as a region on the table.
    """
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
    if cup_mesh is not None:
        import trimesh

        mesh = trimesh.load(str(cup_mesh), force="mesh", process=False)
        size = [float(v) for v in mesh.extents]
        target = {
            "name": "paper cup (scanned mesh)",
            "xyz_m": [target["xyz_m"][0], target["xyz_m"][1], table_z + size[2] / 2],
            "size_m": size,
            "mesh": str(cup_mesh),
            "faces": [len(mesh.faces)],
        }
        return {
            "task": "PutMarkerInCup, scanned cup",
            "primitive": False,
            "source_run": SOURCE_RUN,
            "object": marker,
            "target": target,
            "table_z_m": table_z,
            "notes": [
                "object: a box of the marker's exported size and mass, not its mesh",
                f"marker mass: {MARKER_MASS_SOURCE} (density x placed volume)",
                (
                    "target: the scanned paper cup mesh as a static triangle collider, "
                    "decimated for the collider; success is the marker's centre inside "
                    "the cup's box at rest"
                ),
                "friction: PhysX default, as in the source export",
                (
                    "layout: the cup's offset from the marker in the source export; "
                    "the marker's place in front of the arm is ours"
                ),
            ],
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
            f"marker mass: {MARKER_MASS_SOURCE} (density x placed volume)",
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
    # The earliest problem is charged, in phase order: a marker that was never
    # lifted is a lift failure even if a later phase then timed out.
    start_z = float(start_xyz[2])
    held_after = {
        "lift": "object not lifted: rose {:.1f} cm by the end of lift",
        "upright": "object dropped during the upright turn: {:.1f} cm above its start",
        "carry": "object dropped during carry: {:.1f} cm above its start",
    }
    for entry in log:
        if entry.get("outcome") == "time limit":
            return str(entry["phase"]), "time limit: " + str(entry.get("detail", ""))
        phase = str(entry["phase"])
        if phase in held_after:
            rise = float(entry["object_xyz_m"][2]) - start_z
            if rise < HELD_RISE_M:
                return phase, held_after[phase].format(rise * 100)
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
    cup_mesh = (
        Path(args[args.index("--cup-mesh") + 1]) if "--cup-mesh" in args else None
    )
    scene = scene_spec(cup_mesh)
    # Diagnostic overrides, recorded in the scene so a result is never read as
    # the pipeline's own physics.
    overrides: dict[str, Any] = {}
    if "--marker-mass" in args:
        overrides["marker_mass_kg"] = float(args[args.index("--marker-mass") + 1])
        scene["object"]["mass_kg"] = overrides["marker_mass_kg"]
    if "--friction" in args:
        overrides["marker_friction"] = float(args[args.index("--friction") + 1])
        scene["object"]["friction"] = overrides["marker_friction"]
    if "--trials" in args:
        overrides["trials"] = int(args[args.index("--trials") + 1])
        scene["trials"] = overrides["trials"]
    if overrides:
        scene["diagnostic_overrides"] = overrides
        scene["notes"].append(
            f"DIAGNOSTIC overrides, not the pipeline's physics: {overrides}"
        )
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
        axis_angle_from_quat,
        quat_apply,
        quat_conjugate,
        quat_error_magnitude,
        quat_from_angle_axis,
        quat_from_euler_xyz,
        quat_mul,
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
    if target.get("mesh"):
        cup_usd = write_cup_usd(Path(target["mesh"]), out / "cup.usda")
        sim_utils.UsdFileCfg(usd_path=str(cup_usd)).func(
            "/World/Cup",
            sim_utils.UsdFileCfg(usd_path=str(cup_usd)),
            translation=(target["xyz_m"][0], target["xyz_m"][1], 0.0),
        )
    else:
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
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=float(obj_spec["friction"]),
                    dynamic_friction=float(obj_spec["friction"]),
                )
                if obj_spec.get("friction") is not None
                else None,
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
    home_arm = robot.data.default_joint_pos.torch[0, arm_ids].clone()
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

    for seed in TRIAL_SEEDS[: int(scene.get("trials", len(TRIAL_SEEDS)))]:
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
        upright_quat = hand_quat  # replaced by the hand's real pose after the turn
        if made.get("upright"):
            # Grip the end nearer the base, so the body extends outward. Measured
            # on this arm: joint 6's only tilt inside its limits after the lift is
            # -90 deg, and it sends an outward-pointing body down; gripping the
            # outer end instead needed a 195 deg turn of joint 7 first, whose
            # swing took the fingertips 29 cm off and ran out the phase's time.
            base_xy = robot.data.root_pos_w.torch[0, :2]
            grasp_xy = torch.tensor(made["phases"][1]["ee_xyz_m"][:2], device=device)
            centre_xy = torch.tensor(obj_now["xyz_m"][:2], device=device)
            if float(torch.linalg.norm(grasp_xy - base_xy)) > float(
                torch.linalg.norm(centre_xy - base_xy)
            ):
                obj_now = {**obj_now, "yaw_rad": float(obj_now["yaw_rad"]) + math.pi}
                made = expert.plan(obj_now, target)
            made["grip_end"] = "nearer the base"
        state = {"phases": made["phases"], "phase_index": 0, "elapsed_s": 0.0}
        _, point = hand_state()
        commanded = point.clone()
        commanded_quat = hand_quat.clone()
        joint_turn: dict[str, Any] | None = None  # the upright turn, in joint space
        lowering: list[float] = []
        turned = False
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
            phase_quat = (
                upright_quat if float(phase.get("hand_pitch_rad", 0.0)) else hand_quat
            )
            if float(phase.get("hand_pitch_rad", 0.0)) and phase["name"] != "upright":
                # With the object hanging straight down, turning the hand about the
                # vertical changes nothing for the object; keep the fingers along
                # the line from the base at this phase's spot -- the arm's natural
                # pose there (a fixed direction contorted the arm 34 deg off during
                # the carry) -- in whichever sense, out or in, is nearer.
                spot = torch.tensor(phase["ee_xyz_m"][:2], device=device)
                radial = spot - robot.data.root_pos_w.torch[0, :2]
                approach = quat_apply(
                    upright_quat.unsqueeze(0),
                    torch.tensor([[0.0, 0.0, 1.0]], device=device),
                )[0]
                swing = math.atan2(float(radial[1]), float(radial[0])) - math.atan2(
                    float(approach[1]), float(approach[0])
                )
                swing = (swing + math.pi) % (2 * math.pi) - math.pi
                if abs(swing) > math.pi / 2:
                    swing -= math.copysign(math.pi, swing)
                about_z = quat_from_angle_axis(
                    torch.tensor([swing], device=device),
                    torch.tensor([[0.0, 0.0, 1.0]], device=device),
                )[0]
                phase_quat = quat_mul(about_z.unsqueeze(0), upright_quat.unsqueeze(0))[
                    0
                ]
            # Blocked: commanded down during lower, but the fingertips have not
            # come down 1 mm in the last BLOCKED_S.
            if phase["name"] == "lower":
                lowering.append(float(point[2]))
                window = round(BLOCKED_S / dt)
                blocked = (
                    len(lowering) > window
                    and lowering[-window - 1] - lowering[-1] < 0.001
                    and float(point[2]) > float(phase["ee_xyz_m"][2]) + 0.002
                )
            else:
                lowering.clear()
                blocked = False
            observation = {
                "blocked": blocked,
                "ee_xyz_m": point.tolist(),
                "gripper_closed": gripper_closed(width, width_speed, phase["gripper"]),
                "pitch_error_rad": float(
                    quat_error_magnitude(
                        pose[3:7].unsqueeze(0), phase_quat.unsqueeze(0)
                    )[0]
                ),
            }
            if phase["name"] == "upright" and not turned:
                observation["pitch_error_rad"] = math.pi  # not done until turned
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
                    else f"end effector {error * 1000:.1f} mm from the phase target, "
                    f"hand {math.degrees(observation['pitch_error_rad']):.1f} deg from its orientation"
                )
                try:
                    limits = robot.data.joint_pos_limits.torch[0, arm_ids].tolist()
                    joints = robot.data.joint_pos.torch[0, arm_ids].tolist()
                    log[-1]["arm_joints_vs_limits_deg"] = [
                        [
                            round(math.degrees(q), 1),
                            round(math.degrees(lo), 1),
                            round(math.degrees(hi), 1),
                        ]
                        for q, (lo, hi) in zip(joints, limits, strict=True)
                    ]
                except Exception as exc:  # diagnostics only
                    log[-1]["arm_joints_vs_limits_deg"] = f"unavailable: {exc}"
                break
            if action["advance"]:
                log.append(
                    _phase_entry(
                        action["phase"],
                        "finished",
                        state,
                        error,
                        width,
                        marker,
                        [
                            round(math.degrees(q), 1)
                            for q in robot.data.joint_pos.torch[0, arm_ids].tolist()
                        ],
                    )
                )
                state["phase_index"] = index + 1
                state["elapsed_s"] = 0.0
                continue
            if phase["name"] == "upright" and not turned:
                # A quarter turn is joint 7 then joint 6, not differential IK:
                # under IK the same turn swung joint 5 through 166 deg into its
                # limit while joints 6 and 7 had room.
                if joint_turn is None:
                    joint_turn = plan_joint_turn(
                        robot.data.body_link_jacobian_w.torch[
                            :, jacobian_index, :, jacobian_joints
                        ][0],
                        robot.data.joint_pos.torch[0, arm_ids],
                        robot.data.joint_pos_limits.torch[0, arm_ids],
                        marker.data.root_pos_w.torch[0] - point,
                        torch,
                        quat_from_angle_axis,
                        quat_apply,
                    )
                    log.append(
                        {
                            "phase": "upright",
                            "outcome": "joint turn",
                            "object_xyz_m": marker.data.root_pos_w.torch[0].tolist(),
                            **{k: v for k, v in joint_turn.items() if k != "goal"},
                        }
                    )
                    if joint_turn.get("failed"):
                        log[-1]["outcome"] = "time limit"
                        log[-1]["detail"] = joint_turn["failed"]
                        log[-1]["object_xyz_m"] = marker.data.root_pos_w.torch[
                            0
                        ].tolist()
                        break
                held = joint_turn["goal"].clone()
                now = robot.data.joint_pos.torch[0, arm_ids]
                step_rad = ANGULAR_SPEED_RAD_S * dt
                target_cmd = joint_turn.setdefault("cmd", now.clone())
                for joint in (6, 5):  # joint 7 first, then joint 6
                    gap_j = float(held[joint] - target_cmd[joint])
                    if abs(gap_j) > 1e-4:
                        target_cmd[joint] += math.copysign(
                            min(abs(gap_j), step_rad), gap_j
                        )
                        break
                command.set_position_index(
                    value=target_cmd.unsqueeze(0), joint_ids=arm_ids
                )
                command.set_position_index(
                    value=torch.zeros((1, len(driven)), device=device), joint_ids=driven
                )
                if float(torch.max(torch.abs(now - held))) < math.radians(1.0):
                    turned = True
                    upright_quat = pose[3:7].clone()
                    commanded_quat = upright_quat.clone()
                    commanded = point.clone()
                render = seed == 0 and tick % FRAME_EVERY_STEPS == 0
                step_sim(render=render)
                if render:
                    frames = _save_frame(camera, dt, frame_dir, frames, png_bytes)
                tick += 1
                state["elapsed_s"] = float(state["elapsed_s"]) + dt
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
            # Turn the commanded orientation toward the phase's, at a capped rate.
            delta = axis_angle_from_quat(
                quat_mul(
                    phase_quat.unsqueeze(0), quat_conjugate(commanded_quat.unsqueeze(0))
                )
            )[0]
            angle = float(torch.linalg.norm(delta))
            turn_step = ANGULAR_SPEED_RAD_S * dt
            if angle <= turn_step or angle < 1e-6:
                commanded_quat = phase_quat.clone()
            else:
                partial = quat_from_angle_axis(
                    torch.tensor([turn_step], device=device),
                    (delta / angle).unsqueeze(0),
                )[0]
                commanded_quat = quat_mul(
                    partial.unsqueeze(0), commanded_quat.unsqueeze(0)
                )[0]
            hand_target = (
                commanded
                - quat_apply(
                    commanded_quat.unsqueeze(0),
                    torch.tensor([[0.0, 0.0, offset]], device=device),
                )[0]
            )
            root_pose = robot.data.root_pose_w.torch
            target_b, quat_b = subtract_frame_transforms(
                root_pose[:, 0:3],
                root_pose[:, 3:7],
                hand_target.unsqueeze(0),
                commanded_quat.unsqueeze(0),
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
            arm_target = ik.compute(hand_b, hand_quat_b, jacobian, arm_now)
            # The arm has 7 joints for a 6-number pose, so one "elbow" direction
            # is free and differential IK lets it drift: under the upright turn it
            # drifted the base to -84 deg and the elbow to +-115 deg, into joint
            # limits. Pull toward the home posture in that free direction only;
            # by construction it cannot move the hand off its target.
            j = jacobian[0]
            null = torch.eye(j.shape[1], device=device) - torch.linalg.pinv(j) @ j
            arm_target = arm_target + (
                null @ (POSTURE_GAIN * (home_arm - arm_now[0]))
            ).unsqueeze(0)
            command.set_position_index(value=arm_target, joint_ids=arm_ids)
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


def plan_joint_turn(
    jacobian, joints, limits, centre_from_grip, torch, quat_from_angle_axis, quat_apply
) -> dict[str, Any]:
    """Joint 7 and joint 6 targets that turn a gripped object to hang straight down.

    Joint 6's world axis (read from the Jacobian) tilts the hand a quarter turn,
    whichever way keeps it inside its limits. Before that, joint 7 turns the
    object about its own axis so that the tilt sends the object's body (from
    the grip toward its centre) straight down; of the equivalent angles, the
    one inside joint 7's limits and smallest is used. A turn neither joint can
    make is returned as ``failed``, never forced.
    """
    margin = math.radians(5.0)
    w6 = jacobian[3:6, 5] / torch.linalg.norm(jacobian[3:6, 5])
    w7 = jacobian[3:6, 6] / torch.linalg.norm(jacobian[3:6, 6])
    q = joints.clone()
    lo, hi = limits[:, 0], limits[:, 1]
    tilts = [
        d
        for d in (-math.pi / 2, math.pi / 2)
        if float(lo[5]) + margin <= float(q[5]) + d <= float(hi[5]) - margin
    ]
    if not tilts:
        return {
            "failed": f"joint 6 at {math.degrees(float(q[5])):.0f} deg cannot tilt 90 deg"
        }
    d6 = tilts[0]
    down = torch.tensor([[0.0, 0.0, -1.0]], device=q.device)
    undo = quat_from_angle_axis(torch.tensor([-d6], device=q.device), w6.unsqueeze(0))
    need = quat_apply(undo, down)[0]  # the body direction the tilt sends down
    body = centre_from_grip / torch.linalg.norm(centre_from_grip)
    b = body - (body @ w7) * w7
    n = need - (need @ w7) * w7
    b, n = b / torch.linalg.norm(b), n / torch.linalg.norm(n)
    theta = math.atan2(float(torch.dot(w7, torch.linalg.cross(b, n))), float(b @ n))
    turns = [
        theta + k * 2 * math.pi
        for k in (-1, 0, 1)
        if float(lo[6]) + margin
        <= float(q[6]) + theta + k * 2 * math.pi
        <= float(hi[6]) - margin
    ]
    if not turns:
        return {
            "failed": f"joint 7 at {math.degrees(float(q[6])):.0f} deg cannot turn {math.degrees(theta):.0f} deg"
        }
    d7 = min(turns, key=abs)
    goal = q.clone()
    goal[6] = q[6] + d7
    goal[5] = q[5] + d6
    return {
        "goal": goal,
        "joint6_deg": round(math.degrees(d6), 1),
        "joint7_deg": round(math.degrees(d7), 1),
    }


def write_cup_usd(mesh_path: Path, usd_path: Path) -> Path:
    """The scanned cup as a static triangle collider: centred, its base at z = 0.

    No rigid body, so PhysX treats it as static and a triangle mesh (not a
    convex hull) is allowed -- the hull would fill the cup in.
    """
    import trimesh
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt  # type: ignore[import-not-found]

    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    v = mesh.vertices.copy()
    lo, hi = v.min(axis=0), v.max(axis=0)
    v[:, 0] -= (lo[0] + hi[0]) / 2
    v[:, 1] -= (lo[1] + hi[1]) / 2
    v[:, 2] -= lo[2]
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Cup")
    stage.SetDefaultPrim(root.GetPrim())
    geom = UsdGeom.Mesh.Define(stage, "/Cup/mesh")
    geom.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in v]))
    geom.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(mesh.faces)))
    geom.CreateFaceVertexIndicesAttr(
        Vt.IntArray([int(i) for f in mesh.faces for i in f])
    )
    geom.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.92, 0.92, 0.9)]))
    geom.CreateDoubleSidedAttr(True)
    UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(geom.GetPrim()).CreateApproximationAttr("none")
    stage.GetRootLayer().Save()
    return usd_path


def _phase_entry(
    phase: str,
    outcome: str,
    state: dict[str, Any],
    error: float,
    width: float,
    marker: Any,
    arm_joints: list[float] | None = None,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "outcome": outcome,
        "elapsed_s": round(float(state["elapsed_s"]), 4),
        "ee_error_m": error,
        "gripper_width_m": width,
        "object_xyz_m": marker.data.root_pos_w.torch[0].tolist(),
        "arm_joints_deg": arm_joints if arm_joints is not None else None,
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
