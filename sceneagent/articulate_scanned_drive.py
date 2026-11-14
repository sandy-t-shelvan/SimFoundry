# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KAN-45 on a scanned object: the lid on a simulated, driven revolute joint in Isaac Lab.

    env -u PYTHONPATH /root/.local/share/mamba/envs/isaaclab/bin/python3 \
        sceneagent/articulate_scanned_drive.py --links LINKS --out OUT

LINKS is what articulate_scanned.py wrote. Runs on the box under Isaac
Lab with PYTHONPATH unset (as articulate_drive.py does), so it imports
nothing from sceneagent; Isaac imports stay inside functions so CPU tests
can load the pure helpers. Unlike articulate_drive.py, which poses two
cuboids kinematically, this simulates the scanned meshes on a physics joint.

Builds ``OUT/mailbox_articulated.usda`` (fixed-base articulation: body, lid,
revolute ``hinge`` at the proposed pivot/axis/limits, convex colliders),
drives the hinge 0 -> upper -> 0 over 4 s at 60 Hz with an implicit position
actuator, and logs joint position, target, the lid's drift from the hinge and
its speed. KAN-45's "parts flying apart" rule: drift > 1 cm or speed > 5 m/s.
Masses are assumed for this test, not measured.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path

DT = 1.0 / 60.0
SWEEP_S = 4.0
MAX_DRIFT_M = 0.01
MAX_SPEED_M_S = 5.0
ASSUMED_MASS_KG = {"body": 0.30, "lid": 0.05}


def read_obj(path: Path):
    verts, faces = [], []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            verts.append(tuple(float(v) for v in line.split()[1:4]))
        elif line.startswith("f "):
            idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
            for k in range(1, len(idx) - 1):
                faces.append((idx[0], idx[k], idx[k + 1]))
    return verts, faces


def axis_token(axis):
    """USD joints take X/Y/Z; a negative axis flips the joint coordinate's sign."""
    i = max(range(3), key=lambda k: abs(axis[k]))
    if abs(abs(axis[i]) - 1.0) > 1e-6:
        raise ValueError(f"axis {axis} is not along X, Y or Z")
    return "XYZ"[i], (1.0 if axis[i] > 0 else -1.0)


def targets(lower: float, upper: float) -> list[float]:
    n = round(SWEEP_S / DT)
    half = n // 2
    up = [lower + (upper - lower) * i / half for i in range(half + 1)]
    return up + up[-2::-1]


def build_usd(path: Path, joint: dict, body, lid) -> dict:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt  # type: ignore[import-not-found]

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/mailbox")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    links = {}
    for name, (verts, faces), colour, approx in (
        ("body", body, (0.75, 0.08, 0.08), "convexDecomposition"),
        ("lid", lid, (0.85, 0.12, 0.12), "convexHull"),
    ):
        xf = UsdGeom.Xform.Define(stage, f"/mailbox/{name}")
        UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim())
        UsdPhysics.MassAPI.Apply(xf.GetPrim()).CreateMassAttr(ASSUMED_MASS_KG[name])
        mesh = UsdGeom.Mesh.Define(stage, f"/mailbox/{name}/mesh")
        mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*v) for v in verts]))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([i for f in faces for i in f]))
        mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*colour)]))
        mesh.CreateDoubleSidedAttr(True)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(
            approx
        )
        links[name] = xf
    # Fixed base: a fixed joint with only body1 attaches the body to the world.
    base = UsdPhysics.FixedJoint.Define(stage, "/mailbox/base_joint")
    base.CreateBody1Rel().SetTargets([Sdf.Path("/mailbox/body")])
    token, sign = axis_token(joint["axis"])
    lo, hi = sorted((sign * joint["lower"], sign * joint["upper"]))
    hinge = UsdPhysics.RevoluteJoint.Define(stage, "/mailbox/hinge")
    hinge.CreateBody0Rel().SetTargets([Sdf.Path("/mailbox/body")])
    hinge.CreateBody1Rel().SetTargets([Sdf.Path("/mailbox/lid")])
    hinge.CreateAxisAttr(token)
    pivot = Gf.Vec3f(*joint["pivot_m"])
    hinge.CreateLocalPos0Attr(pivot)  # body frame = world at rest
    hinge.CreateLocalPos1Attr(pivot)  # lid frame = world at rest
    hinge.CreateLowerLimitAttr(math.degrees(lo))
    hinge.CreateUpperLimitAttr(math.degrees(hi))
    drive = UsdPhysics.DriveAPI.Apply(hinge.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    stage.GetRootLayer().Save()
    return {
        "axis_token": token,
        "sign": sign,
        "limits_deg": [math.degrees(lo), math.degrees(hi)],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--links", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    joint = json.loads((a.links / "joint.json").read_text())
    body, lid = read_obj(a.links / "body.obj"), read_obj(a.links / "lid.obj")
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("HEADLESS", "1")
    from isaaclab.app import add_launcher_args, launch_simulation

    parser = argparse.ArgumentParser()
    add_launcher_args(parser)
    args = parser.parse_args([])
    args.enable_cameras = True
    import isaaclab.sim as sim_utils

    sim_cfg = sim_utils.SimulationCfg(dt=DT)
    result = {"decision": "reject"}
    with launch_simulation(sim_cfg, args):
        try:
            result = run(joint, body, lid, a.out, sim_cfg)
        except Exception:
            result["error"] = traceback.format_exc()[-2000:]
        (a.out / "physics_result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            "physics",
            result.get("decision"),
            result.get("reason", result.get("error", ""))[-300:],
            flush=True,
        )
        sys.stdout.flush()
        os._exit(0)


def run(joint, body, lid, out: Path, sim_cfg) -> dict:
    import importlib.util

    import isaaclab.sim as sim_utils
    import torch
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.sensors.camera import Camera, CameraCfg
    from isaaclab_physx.renderers import IsaacRtxRendererCfg
    from isaaclab_physx.sim.schemas import PhysxArticulationCfg

    spec = importlib.util.spec_from_file_location(
        "_png", Path(__file__).resolve().parent / "isaaclab_headless_check.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("png encoder is missing")
    png = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(png)

    usd = out / "mailbox_articulated.usda"
    meta = build_usd(usd, joint, body, lid)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.85, 0.85, 0.85))
    light.func("/World/Light", light)
    cfg = ArticulationCfg(
        prim_path="/World/Mailbox",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd),
            articulation_props=[
                PhysxArticulationCfg(
                    enabled_self_collisions=False, solver_position_iteration_count=16
                )
            ],
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), joint_pos={"hinge": 0.0}
        ),
        actuators={
            "hinge": ImplicitActuatorCfg(
                joint_names_expr=["hinge"],
                stiffness=5.0,
                damping=0.5,
                joint_effort_limit=5.0,
            )
        },
    )
    robot = Articulation(cfg=cfg)
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
                clipping_range=(0.01, 100.0),
            ),
        )
    )
    sim.reset()
    camera.set_world_poses_from_view(
        torch.tensor([[0.32, -0.38, 0.30]], device=sim.device),
        torch.tensor([[0.0, 0.0, 0.08]], device=sim.device),
    )
    names = robot.body_names
    lid_i, body_i = names.index("lid"), names.index("body")
    pivot = torch.tensor(joint["pivot_m"], device=sim.device)
    sign = meta["sign"]

    def state():
        pos = robot.data.body_link_pos_w.torch[0]
        vel = robot.data.body_lin_vel_w.torch[0]
        return pos[lid_i], pos[body_i], vel[lid_i]

    for _ in range(30):  # settle the renderer and the solver at the closed pose
        robot.write_data_to_sim()
        sim.step()
        robot.update(DT)
        camera.update(DT)
    lid0, body0, _ = state()
    r0 = float(torch.linalg.norm(lid0 - pivot))
    log, frames = [], []
    worst_drift = worst_speed = body_drift = 0.0
    plan = targets(joint["lower"], joint["upper"])
    for step, angle in enumerate(plan):
        robot.actuators.target_command.set_position_index(
            value=torch.tensor([[sign * angle]], device=sim.device)
        )
        robot.write_data_to_sim()
        sim.step()
        robot.update(DT)
        camera.update(DT)
        lid_p, body_p, lid_v = state()
        q = float(robot.data.joint_pos.torch[0, 0]) * sign
        drift = abs(float(torch.linalg.norm(lid_p - pivot)) - r0)
        speed = float(torch.linalg.norm(lid_v))
        worst_drift, worst_speed = max(worst_drift, drift), max(worst_speed, speed)
        body_drift = max(body_drift, float(torch.linalg.norm(body_p - body0)))
        log.append(
            {
                "t": round(step * DT, 4),
                "target_rad": angle,
                "joint_rad": q,
                "lid_drift_m": drift,
                "lid_speed_m_s": speed,
            }
        )
        if step % 20 == 0 or step == len(plan) - 1:
            rgb = camera.data.output["rgb"].torch[0, ..., :3]
            if rgb.dtype.is_floating_point:
                rgb = (rgb.clamp(0, 1) * 255).round()
            path = out / f"physics_{step:03d}.png"
            path.write_bytes(png.png_bytes(rgb.to("cpu").numpy().astype("uint8")))
            frames.append(str(path))
    (out / "joint_log.json").write_text(json.dumps(log) + "\n")
    peak = max(e["joint_rad"] for e in log)
    final = log[-1]["joint_rad"]
    failed = (
        worst_drift > MAX_DRIFT_M
        or worst_speed > MAX_SPEED_M_S
        or body_drift > MAX_DRIFT_M
    )
    return {
        "decision": "reject" if failed else "accept",
        "reason": (
            f"peak joint {math.degrees(peak):.1f} deg (target {math.degrees(joint['upper']):.1f}), "
            f"final {math.degrees(final):.1f} deg; lid drift from hinge max {worst_drift * 1000:.2f} mm, "
            f"lid speed max {worst_speed:.3f} m/s, body drift max {body_drift * 1000:.2f} mm"
        ),
        "limits": {"max_drift_m": MAX_DRIFT_M, "max_speed_m_s": MAX_SPEED_M_S},
        "usd": str(usd),
        "joint_usd": meta,
        "assumed_mass_kg": ASSUMED_MASS_KG,
        "frames": frames,
        "steps": len(log),
    }


if __name__ == "__main__":
    main()
