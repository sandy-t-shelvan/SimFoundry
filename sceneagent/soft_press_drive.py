# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Drop one soft block, then press it, with the pinned PhysGaussian solver.

This script does not install PhysGaussian and does not choose Young's
modulus or Poisson's ratio. Those come from the trace, or the run stops.
"""

from __future__ import annotations

import json
import struct
import sys
import zipfile
import zlib
from pathlib import Path
from typing import Any

MISSING = "PhysGaussian is not installed on this box; soft_press does not install it"
UNRESOLVED = (
    "youngs_modulus_pa or poisson_ratio is unresolved; soft_press does not invent them"
)
ITEM = "soft press"
ROOT = Path("/workspace/src/PhysGaussian")
TRACE = "soft_press_trace.json"
WIDTH = 320
HEIGHT = 240
# The test rig, not a material: a 0.1 m block dropped 2 cm onto a floor in
# the solver's 1 m box, then a plate driven down through a quarter of the
# block's height, then lifted back off at the same speed. Metres and seconds. The drop is short on
# purpose: the test is the press, and an undamped jelly block dropped 0.3 m
# bounced back up and was still in the air when the plate came down.
_N_GRID = 96
_GRID_LIM = 1.0
_SIDE_M = 0.1  # 50 g over 1 L is 50 kg/m^3, a foam's density; 0.2 m gave 6 kg/m^3
_PER_SIDE = 20  # 8000 particles: about 2 per cell per axis at n_grid 96
_FLOOR_Z = 0.2
_DROP_Z = 0.22
_DROP_S = 0.45  # a 2 cm fall takes 0.06 s; the rest is landing and settling
# Global grid damping, as a rate so it does not depend on the step size.
# The solver applies grid_v_damping_scale every step; PhysGaussian's own
# configs use 0.9995 to 0.9999 per step. 5/s halves a velocity in 0.14 s.
_DAMPING_PER_S = 5.0
_PRESS_S = 0.2
_PRESS_M = 0.025  # a quarter of the block's height
_RELEASE_S = 0.3  # the plate lifts for _PRESS_S, then the block rests
_FPS = 30
_MAX_DT = 2.0e-4
_CFL = 0.3
# Accept: the block lands, the plate squeezes it, nothing escapes.
LANDED_CELLS = 2.0  # lowest particle within this many cells of the floor
SQUEEZED_SHARE = 0.9  # height under the plate below this share of the rested height
RECOVERED_SHARE = (
    0.95  # elastic: height after the plate lifts, against the rested height
)


def deps_ready(importer=None, root: Path = ROOT) -> bool:
    """True only when warp, taichi, and the pinned checkout are already there."""
    load = importer or __import__
    try:
        load("warp")
        load("taichi")
    except ImportError:
        return False
    return Path(root).is_dir()


def elastic_pair(record: dict[str, Any]) -> tuple[float, float] | None:
    """E and ν from the trace. Missing or out of range is unresolved."""
    youngs = record.get("youngs_modulus_pa")
    poisson = record.get("poisson_ratio")
    if (
        youngs is None
        or poisson is None
        or isinstance(youngs, bool)
        or isinstance(poisson, bool)
    ):
        return None
    try:
        youngs_f = float(youngs)
        poisson_f = float(poisson)
    except (TypeError, ValueError):
        return None
    if youngs_f <= 0.0 or not 0.0 < poisson_f < 0.5:
        return None
    return youngs_f, poisson_f


def main() -> None:
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    params = Path(sys.argv[sys.argv.index("--params") + 1])
    sys.exit(drive(out, params))


def drive(out: Path, params: Path, runner=None) -> int:
    """Write the ledger. ``runner`` is the simulator; the default is PhysGaussian."""
    out.mkdir(parents=True, exist_ok=True)
    if params.is_file():
        (out / "soft_press_params.json").write_bytes(params.read_bytes())
    if not deps_ready():
        _finish(out, "reject", MISSING)
        print(MISSING, flush=True)
        return 1
    if not params.is_file():
        reason = f"no params file at {params}"
        _finish(out, "reject", reason)
        print(reason, flush=True)
        return 1
    records = json.loads(params.read_text(encoding="utf-8"))
    record = records[0] if isinstance(records, list) and records else None
    if not isinstance(record, dict) or elastic_pair(record) is None:
        _finish(out, "reject", UNRESOLVED)
        print(UNRESOLVED, flush=True)
        return 1
    simulate = runner or physgaussian_drop_and_press
    try:
        frames = simulate(record, out)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        _finish(out, "reject", reason)
        print(reason, flush=True)
        return 1
    _zip_frames(out, frames)
    trace_path = out / TRACE
    if not trace_path.is_file():
        reason = (
            f"{len(frames)} frames but no {TRACE}; nothing shows the press happened"
        )
        _finish(out, "reject", reason)
        print(reason, flush=True)
        return 1
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    problems = physics_problems(trace)
    if problems:
        reason = "; ".join(problems)
        _finish(out, "reject", reason)
        print(reason, flush=True)
        return 1
    reason = (
        f"{len(frames)} frames; rested height {trace['rested_height_m']:.3f} m, "
        f"pressed {trace['pressed_height_m']:.3f} m, after release "
        f"{trace['released_height_m']:.3f} m"
    )
    _finish(out, "accept", reason)
    print(reason, flush=True)
    return 0


def physics_problems(trace: dict[str, Any]) -> list[str]:
    """Why a drop-and-press trace is not a press. Empty when it is one.

    The trace is what the simulation measured per frame: lowest particle,
    block height, whether every position is finite, and the phase.
    """
    frames = trace.get("frames") or []
    if not frames:
        return ["the trace has no frames"]
    problems = []
    if not all(f.get("finite") for f in frames):
        problems.append("a particle position went non-finite")
    cell = float(trace["cell_m"])
    floor = float(trace["floor_z_m"])
    lowest = min(float(f["min_z_m"]) for f in frames)
    if lowest < floor - cell:
        problems.append(f"particles passed through the floor (lowest {lowest:.3f} m)")
    rested = trace.get("rested_min_z_m")
    if rested is None or float(rested) > floor + LANDED_CELLS * cell:
        problems.append(f"the block did not land (lowest at press start {rested})")
    rested_h, pressed_h = trace.get("rested_height_m"), trace.get("pressed_height_m")
    if (
        not rested_h
        or pressed_h is None
        or float(pressed_h) > SQUEEZED_SHARE * float(rested_h)
    ):
        problems.append(
            f"the plate did not squeeze the block (height {rested_h} -> {pressed_h})"
        )
    released, final = trace.get("released_height_m"), trace.get("final_min_z_m")
    if rested_h and (
        released is None or float(released) < RECOVERED_SHARE * float(rested_h)
    ):
        problems.append(
            f"the block did not spring back (height {rested_h} -> {released})"
        )
    if final is None or float(final) > floor + LANDED_CELLS * cell:
        problems.append(f"the block did not end at rest on the floor (lowest {final})")
    return problems


def physgaussian_drop_and_press(record: dict[str, Any], out: Path) -> list[Path]:
    """Drop a soft block on a floor, press it with a moving plate, release.

    Jelly is the solver's elastic solid. Sand and foam need extra parameters
    the trace does not carry. Density is the object's mass over the block's
    volume, and each particle carries its share of that volume. The step is
    set by the CFL condition for the looked-up stiffness, so a stiffer
    material takes smaller steps instead of exploding. ``TRACE`` records what
    happened per frame; ``drive`` decides from it.
    """
    import math
    import sys

    import torch
    import warp as wp  # type: ignore[import-not-found]

    solver_dir = ROOT / "mpm_solver_warp"
    sys.path.insert(0, str(solver_dir))
    from mpm_solver_warp import MPM_Simulator_WARP  # type: ignore[import-not-found]

    pair = elastic_pair(record)
    if pair is None:
        raise RuntimeError(UNRESOLVED)
    youngs, poisson = pair
    mass = float(record["mass_kg"])
    count = _PER_SIDE**3
    block_volume = _SIDE_M**3
    density = mass / block_volume
    cell = _GRID_LIM / _N_GRID
    dt = rest_dt(youngs, poisson, density, cell)
    device = "cuda:0"
    wp.init()
    solver = MPM_Simulator_WARP(
        count, n_grid=_N_GRID, grid_lim=_GRID_LIM, device=device
    )
    solver.load_initial_data_from_torch(
        torch.tensor(_cube(_PER_SIDE), dtype=torch.float32, device=device),
        torch.ones(count, dtype=torch.float32, device=device) * (block_volume / count),
        n_grid=_N_GRID,
        grid_lim=_GRID_LIM,
        device=device,
    )
    frame_s = 1.0 / _FPS
    steps_per_frame = max(1, round(frame_s / dt))
    step_dt = frame_s / steps_per_frame
    solver.set_parameters_dict(
        {
            "E": youngs,
            "nu": poisson,
            "material": "jelly",
            "g": [0.0, 0.0, -9.81],
            "density": density,
            "grid_v_damping_scale": math.exp(-_DAMPING_PER_S * step_dt),
        },
        device=device,
    )
    solver.finalize_mu_lam(device=device)
    solver.add_surface_collider((0.0, 0.0, _FLOOR_Z), (0.0, 0.0, 1.0), "sticky", 0.0)
    # The plate: a slab just above the landed block, driven down at constant
    # speed for the press, then gone (released).
    speed = _PRESS_M / _PRESS_S
    top = _FLOOR_Z + _SIDE_M
    solver.set_velocity_on_cuboid(
        point=(0.5, 0.5, top + 1.5 * cell),  # bottom half a cell above the block
        size=(0.3, 0.3, cell),
        velocity=(0.0, 0.0, -speed),
        start_time=_DROP_S,
        end_time=_DROP_S + _PRESS_S,
    )
    # Lift the plate back off at the same speed. Deleting it at once instead
    # gives back the stored strain energy in one step and throws the block.
    solver.set_velocity_on_cuboid(
        point=(0.5, 0.5, top + 1.5 * cell - _PRESS_M),
        size=(0.3, 0.3, cell),
        velocity=(0.0, 0.0, speed),
        start_time=_DROP_S + _PRESS_S,
        end_time=_DROP_S + 2 * _PRESS_S,
    )
    total = _DROP_S + _PRESS_S + _RELEASE_S
    written, frames, t, step = [], [], 0.0, 0
    rested = pressed = None
    for frame in range(round(total * _FPS) + 1):
        if frame:
            for _ in range(steps_per_frame):
                solver.p2g2p(step, step_dt, device=device)
                step += 1
            t = frame * frame_s
        points = solver.mpm_state.particle_x.numpy()
        record_frame = _measure(points, t)
        frames.append(record_frame)
        if rested is None and t >= _DROP_S - 1e-9:
            rested = record_frame
        if t <= _DROP_S + _PRESS_S + 1e-9:
            pressed = record_frame
        path = out / f"soft_press_{frame:02d}.png"
        path.write_bytes(raster_png(points))
        written.append(path)
    trace = {
        "solver": "PhysGaussian mpm_solver_warp, jelly",
        "youngs_modulus_pa": youngs,
        "poisson_ratio": poisson,
        "density_kg_m3": density,
        "rig": {
            "side_m": _SIDE_M,
            "drop_from_z_m": _DROP_Z,
            "press_m": _PRESS_M,
            "press_s": _PRESS_S,
            "particles": count,
            "n_grid": _N_GRID,
        },
        "dt_s": step_dt,
        "damping_per_s": _DAMPING_PER_S,
        "cell_m": cell,
        "floor_z_m": _FLOOR_Z,
        "rested_min_z_m": rested["min_z_m"] if rested else None,
        "rested_height_m": rested["height_m"] if rested else None,
        "pressed_height_m": pressed["height_m"] if pressed else None,
        "released_height_m": frames[-1]["height_m"],
        "final_min_z_m": frames[-1]["min_z_m"],
        "frames": frames,
    }
    (out / TRACE).write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    return written


def rest_dt(youngs: float, poisson: float, density: float, cell: float) -> float:
    """A stable explicit step: CFL on the P-wave speed, capped at ``_MAX_DT``."""
    modulus = youngs * (1 - poisson) / ((1 + poisson) * (1 - 2 * poisson))
    wave = (modulus / density) ** 0.5
    return min(_MAX_DT, _CFL * cell / wave)


def _measure(points, t: float) -> dict[str, Any]:
    import math

    zs = [float(p[2]) for p in points]
    finite = all(math.isfinite(float(c)) for p in points for c in p)
    return {
        "t_s": round(t, 4),
        "finite": finite,
        "min_z_m": min(zs) if finite else None,
        "height_m": (max(zs) - min(zs)) if finite else None,
    }


# The side view's window (metres): the block, the floor and the plate, 4:3.
VIEW_X = (0.35, 0.65)
VIEW_Z = (0.17, 0.395)


def raster_png(points) -> bytes:
    """Side view of particle x and z around the block, floor line in grey."""
    image = bytearray(WIDTH * HEIGHT * 3)
    (x0, x1), (z0, z1) = VIEW_X, VIEW_Z

    def row_of(z: float) -> int:
        return round((z1 - z) / (z1 - z0) * (HEIGHT - 1))

    floor = row_of(_FLOOR_Z)
    if 0 <= floor < HEIGHT:
        image[floor * WIDTH * 3 : (floor + 1) * WIDTH * 3] = b"\x80\x80\x80" * WIDTH
    for point in points:
        column = round((float(point[0]) - x0) / (x1 - x0) * (WIDTH - 1))
        row = row_of(float(point[2]))
        if 0 <= column < WIDTH and 0 <= row < HEIGHT:
            index = (row * WIDTH + column) * 3
            image[index : index + 3] = b"\xff\xff\xff"
    return _png(bytes(image))


def _cube(per_side: int) -> list[list[float]]:
    """Particle centres filling the block, one per sub-cube, starting at ``_DROP_Z``."""
    step = _SIDE_M / per_side
    low = 0.5 - _SIDE_M / 2
    return [
        [low + (i + 0.5) * step, low + (j + 0.5) * step, _DROP_Z + (k + 0.5) * step]
        for k in range(per_side)
        for j in range(per_side)
        for i in range(per_side)
    ]


def _png(rgb: bytes) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    raw = b"".join(
        b"\x00" + rgb[row * WIDTH * 3 : (row + 1) * WIDTH * 3] for row in range(HEIGHT)
    )
    header = struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def _zip_frames(out: Path, frames: list[Path]) -> None:
    with zipfile.ZipFile(out / "soft_press_frames.zip", "w") as archive:
        for path in frames:
            archive.write(path, path.name)


def _finish(out: Path, decision: str, reason: str) -> None:
    (out / "soft_press_status.txt").write_text(reason + "\n", encoding="utf-8")
    (out / "ledger.json").write_text(
        json.dumps(
            {
                "scene_name": "soft press",
                "entries": [{"item": ITEM, "decision": decision, "reason": reason}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "plan.json").write_text(
        json.dumps({"pipeline": "soft_press", "item": ITEM}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
