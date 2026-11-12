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

ITEM = "Isaac Lab settle"
SETTLE_S = 2.0
DT = 1.0 / 120.0
MAX_SPEED_M_S = 0.01
# A resting centre is above the ground; this much below it is a body that sank.
BELOW_GROUND_M = 0.0
WATCHDOG_S = 900.0
ISAACLAB_PYTHON = "/root/.local/share/mamba/envs/isaaclab/bin/python3"
RESULT = "settle.json"
FRAME = "settle_frame.png"


def judge(objects: dict[str, dict[str, float]]) -> tuple[str, str]:
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
        entry.update(decision=result["decision"], reason=result["reason"])
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
    centres = [body.data.root_pos_w.torch[0] for body in bodies.values()]
    middle = torch.stack(centres).mean(dim=0)
    eye = middle + torch.tensor([0.8, 0.8, 0.6], device=middle.device)
    camera.set_world_poses_from_view(eye[None], middle[None])
    for _ in range(round(SETTLE_S / dt)):
        for body in bodies.values():
            body.write_data_to_sim()
        sim.step()
        for body in bodies.values():
            body.update(dt)
        camera.update(dt)
    objects = {}
    for name, body in bodies.items():
        velocity = body.data.root_lin_vel_w.torch[0]
        position = body.data.root_pos_w.torch[0]
        objects[name] = {
            "speed_m_s": float(torch.linalg.norm(velocity)),
            "centre_z_m": float(position[2]),
        }
    frame = camera.data.output["rgb"][0, ..., :3]
    if frame.dtype.is_floating_point:
        frame = (frame.clamp(0, 1) * 255).round()
    (out / FRAME).write_bytes(_png(frame.to("cpu").numpy().astype(np.uint8)))
    decision, reason = judge(objects)
    _write(out, decision, reason, objects=objects, frame=FRAME, seconds=SETTLE_S, dt=dt)


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
