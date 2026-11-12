# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Headless Isaac Lab check for the box setup stage, and a render check.

Runs only on the GPU box, after ``s_isaaclab`` has installed Isaac Sim and
Isaac Lab. It builds a stage with a ground plane and one cube, steps 120
frames, and prints the cube's final height. The setup stage fails if that
line is missing.

``--render OUT`` also lights the scene, renders the resting cube through a
camera and writes ``isaaclab_render.png``, ``isaaclab_render.log``, a plan
and a one-item ledger under ``OUT`` (gpu-smoke ``isaaclab_render``). The
cube check only exercised PhysX; this is the first run of the RTX renderer
on the box, so every failure path still writes a reject with its reason.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
import traceback
import zlib
from pathlib import Path

RENDER_FLAG = "--render"
ITEM = "Isaac Lab render"
PNG_NAME = "isaaclab_render.png"
LOG_NAME = "isaaclab_render.log"
# The app has hung on this box before (SimulationApp.close). A hang must not
# hold the only GPU lane for the whole job.
WATCHDOG_S = 900.0
CUBE_SIZE_M = 0.2
# The cube rests with its centre at half its size. 1 cm either way, ours.
HEIGHT_TOLERANCE_M = 0.01
# Share of pixels that must read as the red cube. A lit ground plane with no
# cube would pass a "not constant" test; this one it cannot. Ours.
MIN_RED_FRACTION = 0.005
WIDTH, HEIGHT = 640, 480


def png_bytes(rgb) -> bytes:
    """Encode an (H, W, 3) uint8 array as PNG. No image library needed."""
    import numpy as np

    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError(f"expected 3 channels, got {channels}")
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


def red_fraction(rgb) -> float:
    """Share of pixels that read as the red cube."""
    import numpy as np

    rgb = np.asarray(rgb, dtype=np.int32)
    red = (
        (rgb[..., 0] > 120)
        & (rgb[..., 0] > rgb[..., 1] + 50)
        & (rgb[..., 0] > rgb[..., 2] + 50)
    )
    return float(red.mean())


def judge(height_m: float, red: float) -> tuple[str, str]:
    """Accept when the cube rests on the ground and is visible in the frame."""
    rest = CUBE_SIZE_M / 2
    if abs(height_m - rest) > HEIGHT_TOLERANCE_M:
        return "reject", f"cube centre at {height_m:.4f} m, expected {rest} m at rest"
    if red < MIN_RED_FRACTION:
        return (
            "reject",
            f"red cube covers {red:.4f} of the frame, below {MIN_RED_FRACTION}",
        )
    return (
        "accept",
        f"cube centre at {height_m:.4f} m; red cube covers {red:.4f} of the frame",
    )


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _ledger(out: Path, decision: str, reason: str, extra: dict | None = None) -> None:
    entry = {"item": ITEM, "decision": decision, "use_gpu": True, "reason": reason}
    entry.update(extra or {})
    _write_json(out / "ledger.json", {"entries": [entry]})


def _log(out: Path, text: str) -> None:
    with open(out / LOG_NAME, "a", encoding="utf-8") as handle:
        handle.write(text.rstrip("\n") + "\n")


def _kit_log_tail(lines: int = 80) -> str:
    """Error and renderer lines from the newest Kit log, for a failed run."""
    logs = glob.glob(
        os.path.expanduser("~/.nvidia-omniverse/logs/Kit/**/*.log"), recursive=True
    )
    if not logs:
        return "no Kit log found"
    newest = max(logs, key=os.path.getmtime)
    keep = ("error", "vulkan", "rtx", "gpu", "driver", "fatal", "renderer")
    with open(newest, encoding="utf-8", errors="replace") as handle:
        hits = [
            line.rstrip() for line in handle if any(k in line.lower() for k in keep)
        ]
    return f"{newest}\n" + "\n".join(hits[-lines:])


def _diagnostics(out: Path) -> None:
    """What the renderer depends on, recorded before Kit starts."""
    _log(out, f"started {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    _log(
        out,
        f"NVIDIA_DRIVER_CAPABILITIES={os.environ.get('NVIDIA_DRIVER_CAPABILITIES')}",
    )
    _log(out, f"NVIDIA_VISIBLE_DEVICES={os.environ.get('NVIDIA_VISIBLE_DEVICES')}")
    icds = sorted(
        glob.glob("/usr/share/vulkan/icd.d/*") + glob.glob("/etc/vulkan/icd.d/*")
    )
    _log(out, "vulkan icd files: " + (", ".join(icds) or "none"))
    smi = shutil.which("nvidia-smi")
    if smi:
        proc = subprocess.run([smi, "-L"], capture_output=True, text=True, check=False)
        _log(out, "nvidia-smi -L: " + (proc.stdout or proc.stderr).strip())


def _prepare(out: Path) -> None:
    """A crashed rerun must not leave the last run's accept to be fetched."""
    out.mkdir(parents=True, exist_ok=True)
    for name in ("ledger.json", "plan.json", PNG_NAME, LOG_NAME):
        (out / name).unlink(missing_ok=True)
    _write_json(
        out / "plan.json",
        {
            "pipeline": "isaaclab_render",
            "steps": [
                "ground plane, dome light, 0.2 m red cube dropped from 0.5 m",
                "120 physics frames at 1/120 s, then 30 render frames",
                "640x480 RGB from a camera 0.92 m from the cube",
            ],
            "accept": {
                "cube_rest_m": CUBE_SIZE_M / 2,
                "height_tolerance_m": HEIGHT_TOLERANCE_M,
                "min_red_fraction": MIN_RED_FRACTION,
                "thresholds": "ours",
            },
        },
    )
    _diagnostics(out)


def _fail(out: Path, reason: str) -> None:
    _log(out, reason)
    _log(out, "--- Kit log ---\n" + _kit_log_tail())
    _ledger(out, "reject", reason)


def main() -> None:
    out: Path | None = None
    if RENDER_FLAG in sys.argv:
        at = sys.argv.index(RENDER_FLAG)
        out = Path(sys.argv[at + 1])
        # Kit parses sys.argv; it must not see our flag.
        del sys.argv[at : at + 2]
        _prepare(out)

        def _timeout() -> None:
            _fail(out, f"timed out after {WATCHDOG_S:.0f} s")
            os._exit(0)

        watchdog = threading.Timer(WATCHDOG_S, _timeout)
        watchdog.daemon = True
        watchdog.start()

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("HEADLESS", "1")
    try:
        _run(out)
    except Exception:
        if out is None:
            raise
        _fail(out, "exception: " + traceback.format_exc()[-1500:])
    # Flush before exit. A redirected run block-buffers stdout, and
    # SimulationApp.close hangs after the steps (found on 60254439), so a
    # normal exit never writes it. The process is done once results are out.
    sys.stdout.flush()
    os._exit(0)


def _run(out: Path | None) -> None:
    from isaaclab.app import AppLauncher

    # Held for the whole run, as the original check held its .app.
    launcher = AppLauncher(headless=True, enable_cameras=out is not None)
    assert launcher.app is not None

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0))
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)
    looks = {}
    if out is not None:
        looks["visual_material"] = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.85, 0.1, 0.1)
        )
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
        light.func("/World/Light", light)
    cube = RigidObject(
        RigidObjectCfg(
            prim_path="/World/Cube",
            spawn=sim_utils.CuboidCfg(
                size=(CUBE_SIZE_M,) * 3,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                **looks,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
        )
    )
    camera = None
    if out is not None:
        from isaaclab.sensors import Camera, CameraCfg

        camera = Camera(
            CameraCfg(
                prim_path="/World/Camera",
                update_period=0,
                height=HEIGHT,
                width=WIDTH,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.05, 100.0),
                ),
            )
        )
    sim.reset()
    if camera is not None:
        import torch

        camera.set_world_poses_from_view(
            torch.tensor([[0.6, 0.6, 0.5]], device=sim.device),
            torch.tensor([[0.0, 0.0, 0.1]], device=sim.device),
        )
    for _ in range(120):
        sim.step()
        cube.update(sim.cfg.dt)
        if camera is not None:
            camera.update(sim.cfg.dt)
    height_m = float(cube.data.root_pos_w[0, 2])
    print(f"cube_height_m {height_m}", flush=True)
    if out is None:
        return
    # `camera` is created exactly when `out` is set. mypy does not connect those.
    assert camera is not None

    # The first RTX frames come back dark while the renderer warms up.
    for _ in range(30):
        sim.step()
        camera.update(sim.cfg.dt)
    frame = camera.data.output["rgb"][0, ..., :3]
    if frame.dtype.is_floating_point:
        frame = (frame.clamp(0, 1) * 255).round()
    rgb = frame.to("cpu").numpy().astype("uint8")
    (out / PNG_NAME).write_bytes(png_bytes(rgb))
    red = red_fraction(rgb)
    decision, reason = judge(height_m, red)
    _log(out, f"cube_height_m {height_m}")
    _log(out, f"red_fraction {red:.5f}; mean pixel {float(rgb.mean()):.1f}")
    _log(out, f"{decision}: {reason}")
    _ledger(
        out,
        decision,
        reason,
        {"cube_height_m": height_m, "red_fraction": red, "image": PNG_NAME},
    )


if __name__ == "__main__":
    main()
