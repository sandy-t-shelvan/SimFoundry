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
FRAMES = 8
WIDTH = 160
HEIGHT = 120
# Simulation setup, not a material. The domain is the solver's unit box.
_DT = 2.0e-4
_SUBSTEPS = 20
_N_GRID = 32
_GRID_LIM = 1.0
_VOLUME = 1.0e-6


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
    reason = f"{len(frames)} frames"
    _finish(out, "accept", reason)
    print(reason, flush=True)
    return 0


def physgaussian_drop_and_press(record: dict[str, Any], out: Path) -> list[Path]:
    """One block, gravity, a floor, then a ceiling that comes down.

    Jelly is the solver's elastic solid. Sand and foam need extra
    parameters the trace does not carry.
    """
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
    count = _count(record)
    mass = float(record["particle_mass"])
    density = mass / _VOLUME
    device = "cuda:0"
    wp.init()
    positions = _cube(count)
    solver = MPM_Simulator_WARP(
        count, n_grid=_N_GRID, grid_lim=_GRID_LIM, device=device
    )
    solver.load_initial_data_from_torch(
        torch.tensor(positions, dtype=torch.float32, device=device),
        torch.ones(count, dtype=torch.float32, device=device) * _VOLUME,
        n_grid=_N_GRID,
        grid_lim=_GRID_LIM,
        device=device,
    )
    solver.set_parameters_dict(
        {
            "E": youngs,
            "nu": poisson,
            "material": "jelly",
            "g": [0.0, 0.0, -9.81],
            "density": density,
        },
        device=device,
    )
    solver.finalize_mu_lam(device=device)
    solver.add_surface_collider((0.0, 0.0, 0.2), (0.0, 0.0, 1.0), "sticky", 0.0)
    press_at = FRAMES // 2 * _SUBSTEPS * _DT
    solver.add_surface_collider(
        (0.0, 0.0, 0.62),
        (0.0, 0.0, -1.0),
        "sticky",
        0.0,
        start_time=press_at,
        end_time=999.0,
    )
    written = []
    for frame in range(FRAMES):
        for step in range(frame * _SUBSTEPS, (frame + 1) * _SUBSTEPS):
            solver.p2g2p(step, _DT, device=device)
        points = solver.mpm_state.particle_x.numpy()
        path = out / f"soft_press_{frame:02d}.png"
        path.write_bytes(raster_png(points))
        written.append(path)
    return written


def raster_png(points) -> bytes:
    """Side view of particle x and z. No image library."""
    image = bytearray(WIDTH * HEIGHT * 3)
    for point in points:
        column = int(min(WIDTH - 1, max(0, float(point[0]) * (WIDTH - 1))))
        row = int(min(HEIGHT - 1, max(0, (1.0 - float(point[2])) * (HEIGHT - 1))))
        index = (row * WIDTH + column) * 3
        image[index : index + 3] = b"\xff\xff\xff"
    return _png(bytes(image))


def _cube(count: int) -> list[list[float]]:
    side = max(1, round(count ** (1.0 / 3.0)))
    points = []
    for index in range(count):
        x = index % side
        y = (index // side) % side
        z = index // (side * side)
        span = max(side - 1, 1)
        points.append(
            [0.4 + 0.2 * x / span, 0.4 + 0.2 * y / span, 0.5 + 0.2 * z / span]
        )
    return points


def _count(record: dict[str, Any]) -> int:
    for line in record.get("log") or []:
        if isinstance(line, str) and line.startswith("n_gaussians: "):
            try:
                count = int(float(line.split(":", 1)[1]))
            except ValueError:
                break
            if count >= 1:
                return min(count, 1000)
    return 1000


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
