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
    verts, faces, colours = [], [], []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            values = [float(v) for v in line.split()[1:]]
            verts.append(tuple(values[:3]))
            if len(values) >= 6:  # trimesh writes "v x y z r g b" when it has colours
                colours.append(tuple(values[3:6]))
        elif line.startswith("f "):
            idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
            for k in range(1, len(idx) - 1):
                faces.append((idx[0], idx[k], idx[k + 1]))
    return verts, faces, (colours if len(colours) == len(verts) else None)


# A proposer that regresses its axis (Particulate) returns floats a fraction of
# a degree off a principal axis; a USD joint takes an X/Y/Z token. Snap within
# this angle and record it; anything further needs a rotated joint frame.
AXIS_SNAP_DEG = 5.0


# Held-open views for the reviewer: camera azimuths around the object.
OPEN_VIEW_AZIMUTHS_DEG = (-50.0, 70.0, 190.0)
OPEN_VIEW_RADIUS_M = 0.5


def axis_snap_deg(axis) -> float:
    """Angle (degrees) between ``axis`` and its nearest principal axis."""
    norm = math.sqrt(sum(float(a) ** 2 for a in axis))
    if norm == 0.0:
        raise ValueError("axis has zero length")
    largest = max(abs(float(a)) for a in axis) / norm
    return math.degrees(math.acos(min(1.0, largest)))


def axis_token(axis):
    """USD joints take X/Y/Z; a negative axis flips the joint coordinate's sign."""
    if axis_snap_deg(axis) > AXIS_SNAP_DEG:
        raise ValueError(
            f"axis {axis} is not within {AXIS_SNAP_DEG:g} deg of X, Y or Z"
        )
    i = max(range(3), key=lambda k: abs(axis[k]))
    return "XYZ"[i], (1.0 if axis[i] > 0 else -1.0)


def targets(lower: float, upper: float) -> list[float]:
    n = round(SWEEP_S / DT)
    half = n // 2
    up = [lower + (upper - lower) * i / half for i in range(half + 1)]
    return up + up[-2::-1]


def build_usd(path: Path, joint: dict, body, lid, name: str = "mailbox") -> dict:
    """The articulated object in the physics export's layout.

    ``/World/<name>`` is the object (the light editor's ``export_import`` reads
    each child of ``/World`` with a direct mesh as one object and every joint
    under it as that object's DOFs): an invisible ``proxy`` mesh of the whole
    object for the editor, the ``body`` and ``lid`` rigid links, the fixed base
    joint and the ``hinge``. The proxy has no physics and is not rendered.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt  # type: ignore[import-not-found]

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    obj = f"/World/{name}"
    root = UsdGeom.Xform.Define(stage, obj)
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    proxy = UsdGeom.Mesh.Define(stage, f"{obj}/proxy")
    body_colours = body[2] if len(body) > 2 else None
    lid_colours = lid[2] if len(lid) > 2 else None
    offset = len(body[0])
    proxy.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*v) for v in body[0] + lid[0]]))
    proxy.CreateFaceVertexCountsAttr(Vt.IntArray([3] * (len(body[1]) + len(lid[1]))))
    proxy.CreateFaceVertexIndicesAttr(
        Vt.IntArray(
            [i for f in body[1] for i in f] + [i + offset for f in lid[1] for i in f]
        )
    )
    proxy.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    links = {}
    for link, verts, faces, scanned, colour, approx in (
        (
            "body",
            body[0],
            body[1],
            body_colours,
            (0.75, 0.08, 0.08),
            "convexDecomposition",
        ),
        ("lid", lid[0], lid[1], lid_colours, (0.85, 0.12, 0.12), "convexHull"),
    ):
        xf = UsdGeom.Xform.Define(stage, f"{obj}/{link}")
        UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim())
        UsdPhysics.MassAPI.Apply(xf.GetPrim()).CreateMassAttr(ASSUMED_MASS_KG[link])
        mesh = UsdGeom.Mesh.Define(stage, f"{obj}/{link}/mesh")
        mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*v) for v in verts]))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([i for f in faces for i in f]))
        if scanned:  # the scan's own colours, so a reviewer can see the label side
            mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*c) for c in scanned]))
            UsdGeom.Primvar(mesh.GetDisplayColorAttr()).SetInterpolation(
                UsdGeom.Tokens.vertex
            )
        else:
            mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*colour)]))
        mesh.CreateDoubleSidedAttr(True)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(
            approx
        )
        links[link] = xf
    # Fixed base: a fixed joint with only body1 attaches the body to the world.
    base = UsdPhysics.FixedJoint.Define(stage, f"{obj}/base_joint")
    base.CreateBody1Rel().SetTargets([Sdf.Path(f"{obj}/body")])
    token, sign = axis_token(joint["axis"])
    lo, hi = sorted((sign * joint["lower"], sign * joint["upper"]))
    sliding = joint["type"] == "prismatic"
    usd_joint = (
        UsdPhysics.PrismaticJoint if sliding else UsdPhysics.RevoluteJoint
    ).Define(stage, f"{obj}/{joint_name(joint)}")
    usd_joint.CreateBody0Rel().SetTargets([Sdf.Path(f"{obj}/body")])
    usd_joint.CreateBody1Rel().SetTargets([Sdf.Path(f"{obj}/lid")])
    usd_joint.CreateAxisAttr(token)
    pivot = Gf.Vec3f(*joint["pivot_m"])
    usd_joint.CreateLocalPos0Attr(pivot)  # body frame = world at rest
    usd_joint.CreateLocalPos1Attr(pivot)  # lid frame = world at rest
    # USD: a revolute joint's limits are degrees, a prismatic joint's are distance (m).
    usd_joint.CreateLowerLimitAttr(lo if sliding else math.degrees(lo))
    usd_joint.CreateUpperLimitAttr(hi if sliding else math.degrees(hi))
    drive = UsdPhysics.DriveAPI.Apply(
        usd_joint.GetPrim(), "linear" if sliding else "angular"
    )
    drive.CreateTypeAttr("force")
    stage.GetRootLayer().Save()
    meta = {
        "type": joint["type"],
        "name": joint_name(joint),
        "axis_token": token,
        "axis_snap_deg": round(axis_snap_deg(joint["axis"]), 3),
        "sign": sign,
    }
    if sliding:
        meta["limits_m"] = [lo, hi]
    else:
        meta["limits_deg"] = [math.degrees(lo), math.degrees(hi)]
    return meta


def joint_name(joint: dict) -> str:
    """The USD joint's name: what the editor's joints panel shows."""
    return "slide" if joint["type"] == "prismatic" else "hinge"


# Drive gains per joint type. The hinge's are KAN-45's. A slide's spring has to
# beat gravity on a 0.05 kg part along a vertical axis (0.5 N) within a few cm.
GAINS = {"revolute": (5.0, 0.5), "prismatic": (200.0, 20.0)}


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
            pos=(0.0, 0.0, 0.0), joint_pos={joint_name(joint): 0.0}
        ),
        actuators={
            joint_name(joint): ImplicitActuatorCfg(
                joint_names_expr=[joint_name(joint)],
                stiffness=GAINS[joint["type"]][0],
                damping=GAINS[joint["type"]][1],
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
    sliding = joint["type"] == "prismatic"
    unit = torch.tensor(joint["axis"], device=sim.device, dtype=lid0.dtype)
    unit = unit / torch.linalg.norm(unit)

    def drift_of(lid_p):
        """A hinge keeps the part's distance to its pivot; a slide keeps it on its axis."""
        if sliding:
            d = lid_p - lid0
            return float(torch.linalg.norm(d - (d @ unit) * unit))
        return abs(float(torch.linalg.norm(lid_p - pivot)) - r0)

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
        drift = drift_of(lid_p)
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
    # Held open, from three sides, for the reviewer: one of them shows the
    # object's front (its label) whatever the object's orientation. Measured
    # after the sweep, so the acceptance numbers above are unaffected.
    robot.actuators.target_command.set_position_index(
        value=torch.tensor([[sign * joint["upper"]]], device=sim.device)
    )
    for _ in range(90):
        robot.write_data_to_sim()
        sim.step()
        robot.update(DT)
    open_views = []
    for k, azimuth in enumerate(OPEN_VIEW_AZIMUTHS_DEG):
        a = math.radians(azimuth)
        eye = [OPEN_VIEW_RADIUS_M * math.cos(a), OPEN_VIEW_RADIUS_M * math.sin(a), 0.30]
        camera.set_world_poses_from_view(
            torch.tensor([eye], device=sim.device),
            torch.tensor([[0.0, 0.0, 0.08]], device=sim.device),
        )
        for _ in range(12):  # let the RTX renderer settle at the new pose
            robot.write_data_to_sim()
            sim.step()
            robot.update(DT)
            camera.update(DT)
        rgb = camera.data.output["rgb"].torch[0, ..., :3]
        if rgb.dtype.is_floating_point:
            rgb = (rgb.clamp(0, 1) * 255).round()
        path = out / f"open_view_{k}.png"
        path.write_bytes(png.png_bytes(rgb.to("cpu").numpy().astype("uint8")))
        open_views.append(str(path))
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
            (
                f"peak travel {peak * 1000:.1f} mm (target {joint['upper'] * 1000:.1f}), "
                f"final {final * 1000:.1f} mm; part drift off the slide axis max {worst_drift * 1000:.2f} mm, "
                if sliding
                else f"peak joint {math.degrees(peak):.1f} deg (target {math.degrees(joint['upper']):.1f}), "
                f"final {math.degrees(final):.1f} deg; lid drift from hinge max {worst_drift * 1000:.2f} mm, "
            )
            + f"lid speed max {worst_speed:.3f} m/s, body drift max {body_drift * 1000:.2f} mm"
        ),
        "limits": {"max_drift_m": MAX_DRIFT_M, "max_speed_m_s": MAX_SPEED_M_S},
        "usd": str(usd),
        "joint_usd": meta,
        "assumed_mass_kg": ASSUMED_MASS_KG,
        "frames": frames,
        "open_views": open_views,
        "steps": len(log),
    }


if __name__ == "__main__":
    main()
