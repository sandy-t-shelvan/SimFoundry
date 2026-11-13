# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SF-35 dc2 (KAN-41): settle an exported scene in Isaac Lab and judge it.

``run_settle(out)`` runs in the pipeline's env: it starts this file under the
Isaac Lab interpreter, reads ``settle.json`` back and writes one ledger item.
Under Isaac Lab (``python3 settle_run.py --stage S --out O --prims a,b``) it
references ``export/scene_settle.usda`` under ``/World/scene`` beside Isaac
Lab's ground plane at z = 0, steps 2 s at 120 Hz with gravity -Z, renders
one frame and writes each object's final speed and centre height.

Accept (KAN-41 dc2): every object's speed below 1 cm/s at the end, and no
object's centre below the ground. The centre stands in for "nothing below
the ground"; a body that falls through ends far below and still moving.
Thresholds are the ticket's, not the paper's.

Only the standard library is imported at module level, so this file runs
under the Isaac Lab interpreter without the repository on its path.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import threading
import time
import traceback
import zlib
from pathlib import Path
from typing import Any

ITEM = "Isaac Lab settle"
SETTLE_S = 2.0
DT = 1.0 / 120.0
MAX_SPEED_M_S = 0.01
# A resting centre is above the ground; this much below it is a body that sank.
BELOW_GROUND_M = 0.0
WATCHDOG_S = 900.0
# Diagnostic only, never judged: the stage keeps stepping to RECORD_S, and a
# body's motion is summarised over WINDOW_S before 2 s and before RECORD_S.
RECORD_S = 4.0
WINDOW_S = 0.5
# Copies of the scene settled beside it, each with one physics change, to find
# what stops a body rocking in place (run 43333060395: an upright mailbox
# rocked at 0.33 rad/s, 0.6 mm net in 0.5 s). Only the scene as exported is
# judged. physics_export writes one export/scene_settle_<name>.usda for each.
VARIANTS: dict[str, dict] = {
    # The rocking body already had the default collider, a convex hull.
    "decomposition": {"approximation": "convexDecomposition"},
    "iters": {"solver_position_iterations": 32, "solver_velocity_iterations": 4},
    "decomposition_iters": {
        "approximation": "convexDecomposition",
        "solver_position_iterations": 32,
        "solver_velocity_iterations": 4,
    },
}
VARIANT_SPACING_M = 3.0
ISAACLAB_PYTHON = "/root/.local/share/mamba/envs/isaaclab/bin/python3"
RESULT = "settle.json"
FRAME = "settle_frame.png"


def judge(objects: dict[str, dict[str, Any]]) -> tuple[str, str]:
    """Accept when every object is at rest and above the ground."""
    if not objects:
        return "reject", "no object was simulated"
    moving = {
        n: o["speed_m_s"] for n, o in objects.items() if o["speed_m_s"] >= MAX_SPEED_M_S
    }
    sunk = {
        n: o["centre_z_m"]
        for n, o in objects.items()
        if o["centre_z_m"] < BELOW_GROUND_M
    }
    if moving or sunk:
        parts = []
        if moving:
            parts.append(
                "still moving after 2 s: "
                + ", ".join(f"{n} {v:.3f} m/s" for n, v in sorted(moving.items()))
            )
        if sunk:
            parts.append(
                "below the ground: "
                + ", ".join(f"{n} z={v:.3f} m" for n, v in sorted(sunk.items()))
            )
        return "reject", "; ".join(parts)
    fastest = max(o["speed_m_s"] for o in objects.values())
    return (
        "accept",
        f"{len(objects)} objects at rest after 2 s (fastest {fastest:.4f} m/s)",
    )


def _read(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _phrases(found: dict) -> str:
    """``mailbox x9, cube x2 in 11 views`` from one association object."""
    counts: dict[str, int] = {}
    for mask in found.get("masks", []):
        phrase = str(mask.get("phrase"))
        counts[phrase] = counts.get(phrase, 0) + 1
    listed = ", ".join(f"{p} x{n}" for p, n in sorted(counts.items()))
    return f"{listed or '?'} in {found.get('views', '?')} views"


def _num(value, spec: str) -> str:
    return "?" if value is None else format(value, spec)


def context(out: Path, result: dict) -> str:
    """One line of evidence for a reject: counts, then per object box, gate and motion."""
    export = _read(out / "export_evidence.json")
    boxes = export.get("object_boxes", {})
    # What each object was found as (GroundingDINO phrases, mask views) and
    # which residual failed its ICP gate.
    found = {
        f"object_{o['object']}": o
        for o in _read(out / "association_evidence.json").get("object_list", [])
    }
    icp = {
        f"object_{o['object']}": o
        for o in _read(out / "bin_score.json").get("icp", {}).get("objects", [])
    }
    left_out = export.get("excluded_gate_rejects", {})
    parts = [
        (
            f"association {export.get('association_objects')} objects "
            f"({export.get('association_dropped')} dropped under 3 views); "
            f"exported {len(boxes)}, {len(left_out)} left out by the ICP gate"
        )
    ]
    for name, motion in sorted(result.get("objects", {}).items()):
        box = boxes.get(name, {})
        size = "x".join(f"{v:.2f}" for v in box.get("size_m", []))
        parts.append(
            f"{name}: {size} m, bottom {_num(box.get('bottom_z_m'), '+.3f')} m, "
            f"gate {box.get('gate')} {icp.get(name, {}).get('gate_failed') or ''}, "
            f"found as {_phrases(found.get(name, {}))}, "
            f"v {_num(motion.get('speed_1s_m_s'), '.3f')}"
            f"->{_num(motion.get('speed_m_s'), '.3f')} m/s, "
            f"w {_num(motion.get('angular_speed_rad_s'), '.2f')} rad/s, "
            f"last 0.5 s moved {_num(motion.get('moved_last_0_5_s_m'), '.4f')} m "
            f"turned {_num(motion.get('turned_last_0_5_s_deg'), '.1f')} deg, "
            f"window max {_num((motion.get('window_2s') or {}).get('max_speed_m_s'), '.4f')} "
            f"m/s, tilt {_num(motion.get('start_tilt_deg'), '.1f')}"
            f"->{_num(motion.get('tilt_deg'), '.1f')} deg"
        )
    for variant, record in sorted(result.get("variants", {}).items()):
        for name, motion in sorted(record.get("objects", {}).items()):
            applied = record.get("applied", {}).get(name, {})
            w2, we = motion.get("window_2s", {}), motion.get("window_end", {})
            parts.append(
                f"variant {variant} {name}: max {_num(w2.get('max_speed_m_s'), '.4f')} "
                f"m/s, excursion {_num(w2.get('max_excursion_m'), '.4f')} m "
                f"(to {RECORD_S:.0f} s: {_num(we.get('max_speed_m_s'), '.4f')} m/s); "
                f"applied {applied.get('approximation')} "
                f"iters {applied.get('solver_position_iterations')}"
            )
        if record.get("skipped"):
            parts.append(f"variant {variant} skipped: {record['skipped']}")
    for name, why in sorted(export.get("excluded_evidence", {}).items()):
        parts.append(f"{name} left out: {why}")
    overlaps = export.get("overlapping_boxes", [])
    parts.append(
        "overlapping boxes: " + (", ".join("~".join(p) for p in overlaps) or "none")
    )
    return "; ".join(parts)


def run_settle(
    out: Path | str, *, runner=subprocess.run, python: str = ISAACLAB_PYTHON
) -> int:
    """Settle ``export/scene_settle.usda`` under Isaac Lab; write the ledger item."""
    out = Path(out)
    ledger_path = out / "ledger.json"
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        ledger = {"entries": []}
    entries = [e for e in ledger.get("entries", []) if e.get("item") != ITEM]
    entry: dict = {"item": ITEM, "use_gpu": True}
    try:
        export = json.loads((out / "export_evidence.json").read_text(encoding="utf-8"))
        prims = [
            o["prim"].rsplit("/", 1)[-1]
            for o in export["objects"]
            if o["route"] == "physx"
        ]
        (out / RESULT).unlink(missing_ok=True)
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        done = runner(
            [
                python,
                str(Path(__file__).resolve()),
                "--stage",
                str(out / "export" / "scene_settle.usda"),
                "--out",
                str(out),
                "--prims",
                ",".join(prims),
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=WATCHDOG_S + 120,
        )
        try:
            result = json.loads((out / RESULT).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            tail = (
                ((done.stderr or "") or (done.stdout or "")).strip().splitlines()[-4:]
            )
            raise RuntimeError(
                f"settle exited {done.returncode}: " + " | ".join(tail)
            ) from None
        reason = result["reason"]
        if result["decision"] != "accept":
            # A failed run publishes only the manifest, which carries this.
            reason = f"{reason} | {context(out, result)}"
        entry.update(decision=result["decision"], reason=reason[:2500])
        if result.get("frame"):
            entry["frame"] = result["frame"]
    except (
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        subprocess.TimeoutExpired,
    ) as exc:
        entry.update(decision="reject", reason=f"{type(exc).__name__}: {exc}"[-800:])
    entries.append(entry)
    ledger["entries"] = entries
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    return 0 if entry["decision"] == "accept" else 1


def _png(rgb) -> bytes:
    import numpy as np

    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def _write(out: Path, decision: str, reason: str, **extra) -> None:
    payload = {"decision": decision, "reason": reason, **extra}
    (out / RESULT).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _simulate(stage: Path, out: Path, prims: list[str]) -> None:
    import argparse

    from isaaclab.app import add_launcher_args, launch_simulation

    parser = argparse.ArgumentParser()
    add_launcher_args(parser)
    args = parser.parse_args([])
    # Kit starts the rendering experience only when this is set (as in the
    # render check, isaaclab_headless_check.py).
    args.enable_cameras = True
    import isaaclab.sim as sim_utils

    sim_cfg = sim_utils.SimulationCfg(dt=DT)
    # Exit from inside the context: SimulationApp.close did not return on
    # 60254439 (isaaclab_headless_check.py).
    with launch_simulation(sim_cfg, args):
        try:
            _steps(stage, out, prims, sim_cfg)
        except Exception:
            _write(out, "reject", "exception: " + traceback.format_exc()[-1500:])
        sys.stdout.flush()
        os._exit(0)


def _angular_speed(body, torch) -> float | None:
    """Diagnostic only: a missing field must not cost the settle its result."""
    try:
        return float(torch.linalg.norm(body.data.root_ang_vel_w.torch[0]))
    except (AttributeError, IndexError, TypeError):
        return None


def _pose(body) -> tuple:
    """Diagnostic: root position and quaternion (w, x, y, z) as plain floats."""
    try:
        position = [float(v) for v in body.data.root_pos_w.torch[0]]
        quat = [float(v) for v in body.data.root_quat_w.torch[0]]
    except (AttributeError, IndexError, TypeError):
        return ()
    return (position, quat)


def _last_half_second(before: tuple | None, after: tuple) -> dict:
    """How far a body moved and turned between 1.5 s and 2 s."""
    if not before or not after:
        return {}
    import math

    (p0, q0), (p1, q1) = before, after
    moved = math.dist(p0, p1)
    dot = min(1.0, abs(sum(a * b for a, b in zip(q0, q1, strict=True))))
    return {
        "moved_last_0_5_s_m": moved,
        "turned_last_0_5_s_deg": math.degrees(2 * math.acos(dot)),
    }


def window_stats(samples: list[tuple[float, list[float]]]) -> dict[str, float]:
    """Max speed and max excursion from the first position over a window of samples."""
    if not samples:
        return {}
    import math

    first = samples[0][1]
    return {
        "max_speed_m_s": max(speed for speed, _ in samples),
        "max_excursion_m": max(math.dist(first, position) for _, position in samples),
    }


def tilt_deg(quat: list[float]) -> float:
    """Angle between a body's z axis and world up, from a (w, x, y, z) quaternion."""
    import math

    _, x, y, _ = quat
    return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))


def _readback(stage, path: str) -> dict:
    """Diagnostic: what the variant's body prim actually carries on the stage."""
    try:
        prim = stage.GetPrimAtPath(path)
        geometry = stage.GetPrimAtPath(path + "/geometry")
        approximation = geometry.GetAttribute("physics:approximation")
        iterations = prim.GetAttribute("physxRigidBody:solverPositionIterationCount")
        return {
            "schemas": [str(s) for s in prim.GetAppliedSchemas()],
            "approximation": approximation.Get() if approximation else None,
            "solver_position_iterations": iterations.Get() if iterations else None,
        }
    except Exception as exc:  # diagnostic only
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}


def _steps(stage: Path, out: Path, prims: list[str], sim_cfg: object) -> None:
    import isaaclab.sim as sim_utils
    import numpy as np
    import torch
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.sensors.camera import Camera, CameraCfg
    from isaaclab_physx.renderers import IsaacRtxRendererCfg

    sim = sim_utils.SimulationContext(sim_cfg)
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)
    light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    light.func("/World/Light", light)
    scene = sim_utils.UsdFileCfg(usd_path=str(stage))
    scene.func("/World/scene", scene)
    bodies = {
        name: RigidObject(RigidObjectCfg(prim_path=f"/World/scene/{name}", spawn=None))
        for name in prims
    }
    variant_bodies: dict[str, dict] = {}
    variants: dict[str, dict] = {}
    for index, name in enumerate(VARIANTS):
        path = stage.with_name(f"scene_settle_{name}.usda")
        if not path.is_file():
            continue
        cfg = sim_utils.UsdFileCfg(usd_path=str(path))
        try:
            cfg.func(
                f"/World/v_{name}",
                cfg,
                translation=(VARIANT_SPACING_M * (index + 1), 0.0, 0.0),
            )
        except TypeError as exc:
            variants[name] = {"skipped": f"{type(exc).__name__}: {exc}"[:200]}
            continue
        variant_bodies[name] = {
            prim: RigidObject(
                RigidObjectCfg(prim_path=f"/World/v_{name}/{prim}", spawn=None)
            )
            for prim in prims
        }
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
    dt = sim.get_physics_dt()
    stage_handle = getattr(sim, "stage", None)
    for name in variant_bodies:
        variants[name] = {
            "change": VARIANTS[name],
            "applied": {
                prim: _readback(stage_handle, f"/World/v_{name}/{prim}")
                for prim in prims
            },
        }
    # The camera frames the scene as exported, not its copies.
    centres = [body.data.root_pos_w.torch[0] for body in bodies.values()]
    middle = torch.stack(centres).mean(dim=0)
    eye = middle + torch.tensor([0.8, 0.8, 0.6], device=middle.device)
    camera.set_world_poses_from_view(eye[None], middle[None])
    start_z = {n: float(b.data.root_pos_w.torch[0][2]) for n, b in bodies.items()}
    start_pose = {n: _pose(b) for n, b in bodies.items()}
    groups = {"scene": bodies, **variant_bodies}
    everything = [b for group in groups.values() for b in group.values()]
    steps = round(SETTLE_S / dt)
    total = round(RECORD_S / dt)
    window = round(WINDOW_S / dt)
    halfway: dict[str, float] = {}
    late: dict[str, tuple] = {}
    judged: dict[str, dict] = {}
    samples: dict[tuple[str, str, str], list] = {}
    for step in range(total):
        for body in everything:
            body.write_data_to_sim()
        sim.step()
        for body in everything:
            body.update(dt)
        if step < steps:
            camera.update(dt)
        span = (
            "2s"
            if steps - window <= step < steps
            else "end"
            if step >= total - window
            else None
        )
        if span:
            for group, members in groups.items():
                for name, body in members.items():
                    speed = float(torch.linalg.norm(body.data.root_lin_vel_w.torch[0]))
                    position = [float(v) for v in body.data.root_pos_w.torch[0]]
                    samples.setdefault((group, name, span), []).append(
                        (speed, position)
                    )
        if step == steps // 2 - 1:
            # Decaying between 1 s and 2 s is settling; steady is rolling,
            # rocking or being pushed.
            halfway = {
                n: float(torch.linalg.norm(b.data.root_lin_vel_w.torch[0]))
                for n, b in bodies.items()
            }
        if step == steps - steps // 4 - 1:
            # Pose at 1.5 s (run 43332392762: an upright mailbox read
            # 0.035 m/s with 0.40 rad/s of spin and barely moved).
            late = {n: _pose(b) for n, b in bodies.items()}
        if step == steps - 1:
            # The judged moment, 2 s, exactly as before; and its frame.
            for name, body in bodies.items():
                velocity = body.data.root_lin_vel_w.torch[0]
                position = body.data.root_pos_w.torch[0]
                pose = _pose(body)
                judged[name] = {
                    "speed_m_s": float(torch.linalg.norm(velocity)),
                    "centre_z_m": float(position[2]),
                    "speed_1s_m_s": halfway.get(name),
                    "angular_speed_rad_s": _angular_speed(body, torch),
                    "start_centre_z_m": start_z[name],
                    "start_tilt_deg": tilt_deg(start_pose[name][1])
                    if start_pose[name]
                    else None,
                    "tilt_deg": tilt_deg(pose[1]) if pose else None,
                    **_last_half_second(late.get(name), pose),
                }
            frame = camera.data.output["rgb"][0, ..., :3]
            if frame.dtype.is_floating_point:
                frame = (frame.clamp(0, 1) * 255).round()
            (out / FRAME).write_bytes(_png(frame.to("cpu").numpy().astype(np.uint8)))
    for name, record in judged.items():
        record["window_2s"] = window_stats(samples.get(("scene", name, "2s"), []))
        record["window_end"] = window_stats(samples.get(("scene", name, "end"), []))
    for group in variant_bodies:
        variants[group]["objects"] = {
            name: {
                "window_2s": window_stats(samples.get((group, name, "2s"), [])),
                "window_end": window_stats(samples.get((group, name, "end"), [])),
            }
            for name in prims
        }
    decision, reason = judge(judged)
    _write(
        out,
        decision,
        reason,
        objects=judged,
        variants=variants,
        frame=FRAME,
        seconds=SETTLE_S,
        recorded_s=RECORD_S,
        dt=dt,
    )


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prims", required=True)
    args = parser.parse_args(argv)
    # Kit reads sys.argv; it must not see ours.
    del sys.argv[1:]
    prims = [p for p in args.prims.split(",") if p]
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    def _timeout() -> None:
        _write(args.out, "reject", f"timed out after {WATCHDOG_S:.0f} s")
        os._exit(0)

    watchdog = threading.Timer(WATCHDOG_S, _timeout)
    watchdog.daemon = True
    watchdog.start()
    started = time.monotonic()
    try:
        _simulate(args.stage, args.out, prims)
    except Exception:
        _write(args.out, "reject", "exception: " + traceback.format_exc()[-1500:])
    if not (args.out / RESULT).is_file():
        seconds = time.monotonic() - started
        _write(
            args.out,
            "reject",
            f"simulation ended without a result after {seconds:.0f} s",
        )
    os._exit(0)


if __name__ == "__main__":
    main()
