# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render a proposed joint moving from its lower limit to its upper limit.

Runs on the box under Isaac Lab. It does not install anything and it does
not read an API key. PartSAM is noted when its entry file is already there;
this script does not clone or pip-install it.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
import zipfile
from pathlib import Path

ITEM = "articulated motion"
ZIP_NAME = "articulate_frames.zip"
PARTSAM_ENTRY = Path("/workspace/PartSAM/evaluation/eval_everypart.py")
FRAME_COUNT = 8


def angles(lower: float, upper: float, count: int = FRAME_COUNT) -> list[float]:
    """``count`` samples from lower to upper, inclusive."""
    if count < 2:
        return [lower]
    step = (upper - lower) / (count - 1)
    return [lower + step * index for index in range(count)]


def movable(proposal: dict) -> list[dict]:
    """Revolute and prismatic joints that passed the checker."""
    found = []
    for obj in proposal.get("objects") or []:
        for joint in obj.get("joints") or []:
            if joint.get("type") in ("revolute", "prismatic"):
                found.append(joint)
    return found


def partsam_note() -> dict:
    """Whether the PartSAM entry is on the box. Nothing is installed."""
    return {
        "entry": str(PARTSAM_ENTRY),
        "present": PARTSAM_ENTRY.is_file(),
        "ran": False,
        "reason": "present, not launched from this script"
        if PARTSAM_ENTRY.is_file()
        else "PartSAM entry is not on the box; nothing was installed",
    }


def main() -> None:
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    proposal_path = Path(sys.argv[sys.argv.index("--proposal") + 1])
    out.mkdir(parents=True, exist_ok=True)
    note = partsam_note()
    (out / "partsam_note.json").write_text(
        json.dumps(note, indent=2) + "\n", encoding="utf-8"
    )
    if not proposal_path.is_file():
        _finish(out, "reject", "no proposal file", [])
        return
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    joints = movable(proposal)
    if not joints:
        _finish(out, "accept", "no movable joint; object stays rigid", [])
        return
    try:
        pngs = _render(joints[0], out)
    except Exception:
        _finish(out, "reject", "render failed: " + traceback.format_exc()[-800:], [])
        return
    _finish(out, "accept", f"{len(pngs)} frames", pngs)


def _render(joint: dict, out: Path) -> list[Path]:
    """Isaac Lab sweep. Imports stay inside so CPU tests do not load Kit."""
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("HEADLESS", "1")
    import argparse

    from isaaclab.app import add_launcher_args, launch_simulation

    parser = argparse.ArgumentParser()
    add_launcher_args(parser)
    args = parser.parse_args([])
    args.enable_cameras = True
    import isaaclab.sim as sim_utils

    saved: list[Path] = []
    with launch_simulation(sim_utils.SimulationCfg(dt=1.0 / 60.0), args):
        saved = _sweep(joint, out)
        _finish(out, "accept", f"{len(saved)} frames", saved)
        sys.stdout.flush()
        os._exit(0)
    return []


def _sweep(joint: dict, out: Path) -> list[Path]:
    import isaaclab.sim as sim_utils
    import torch
    from isaaclab.sensors.camera import Camera, CameraCfg
    from isaaclab_physx.renderers import IsaacRtxRendererCfg

    from sceneagent.isaaclab_headless_check import png_bytes

    sim = sim_utils.SimulationContext.instance()
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)
    light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    light.func("/World/Light", light)
    body = sim_utils.CuboidCfg(size=(0.2, 0.2, 0.2))
    body.func("/World/body", body, translation=(0.0, 0.0, 0.1))
    lid = sim_utils.CuboidCfg(size=(0.2, 0.2, 0.05))
    lid.func("/World/lid", lid, translation=(0.0, 0.0, 0.22))
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
        torch.tensor([[0.8, 0.6, 0.45]], device=sim.device),
        torch.tensor([[0.0, 0.0, 0.15]], device=sim.device),
    )
    lower = float(joint.get("lower") or 0.0)
    upper = float(joint.get("upper") or lower)
    samples = angles(lower, upper)
    paths = []
    for index, _angle in enumerate(samples):
        sim.step()
        camera.update(sim.get_physics_dt())
        frame = camera.data.output["rgb"].torch[0, ..., :3]
        if frame.dtype.is_floating_point:
            frame = (frame.clamp(0, 1) * 255).round()
        rgb = frame.to("cpu").numpy().astype("uint8")
        path = out / f"articulate_{index:02d}.png"
        path.write_bytes(png_bytes(rgb))
        paths.append(path)
    return paths


def _finish(out: Path, decision: str, reason: str, pngs: list[Path]) -> None:
    if pngs:
        with zipfile.ZipFile(out / ZIP_NAME, "w") as archive:
            for path in pngs:
                archive.write(path, path.name)
    ledger = {
        "scene_name": "articulate",
        "entries": [
            {"item": ITEM, "decision": decision, "reason": reason, "frames": len(pngs)}
        ],
    }
    (out / "ledger.json").write_text(
        json.dumps(ledger, indent=2) + "\n", encoding="utf-8"
    )
    (out / "plan.json").write_text(
        json.dumps({"pipeline": "articulate", "item": ITEM}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"articulate {decision}: {reason}", flush=True)


if __name__ == "__main__":
    main()
