# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KAN-24 dc1 (SF-23): the Orientation agent's Isaac Lab sweep of the scene's up-axis.

Paper Figure 7, agent 07: "Isaac Lab sweeps that orient the 3DGS background".
The export (``physics_export``) already turns the scene so the chosen support
plane lies at z = 0 with its normal up (``export_evidence.json`` ``support``:
``world_from_scene``, the plane ``candidates`` and the ``chosen`` one). This
checks that choice with physics instead of geometry alone:

* every support-plane candidate is an up-axis to try, plus small tilts of the
  best one (``TILTS_DEG``);
* for each, a copy of ``export/scene_settle.usda`` is turned so that axis
  points up and set on its own static slab; there is no ground, so a wrong up
  makes the objects slide or fall off;
* all copies settle together for ``SETTLE_S`` and each is scored by how far
  its objects still move in the second half (``LATE_FROM_S`` to the end;
  mean displacement, metres; lower is better). The first second is left
  out: an export settles in there whatever the orientation (Mailbox: the
  mailbox drops 5.8 cm and leans 22.6 degrees in the regular settle too),
  so total motion would score the placement, not the up-axis;
* ``orientation_sweep.json`` records every candidate, the chosen up-axis in
  the export frame and in the scene (splat-derived) frame, and whether the
  sweep agrees with the plane fit.

    python -m sceneagent.orientation_sweep --out RUN    # pipeline env

That starts this file under the Isaac Lab interpreter, which imports only the
standard library and numpy at module level, like ``settle_run.py``.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

ISAACLAB_PYTHON = "/root/.local/share/mamba/envs/isaaclab/bin/python3"
RESULT = "orientation_sweep.json"
SETTLE_S = 2.0
LATE_FROM_S = 1.0
DT = 1.0 / 120.0
TILTS_DEG = (-2.0, 2.0)
SPACING_M = 3.0
LIFT_M = 1.5  # copies float above nothing, so an object that slides off falls
SLAB_SIZE_M = (1.5, 1.5, 0.02)
# Late drifts within this of the best count as a tie; a tie keeps the plane fit.
TIE_M = 0.001
# An object this far below its slab fell off: that orientation is wrong.
FELL_M = 0.5
WATCHDOG_S = 900.0


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("zero vector")
    return v / n


def rotation_onto_z(up) -> np.ndarray:
    """The proper rotation that turns ``up`` onto +Z (Rodrigues)."""
    a, z = unit(up), np.array([0.0, 0.0, 1.0])
    c = float(a @ z)
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:  # upside down: half turn about X
        return np.diag([1.0, -1.0, -1.0])
    v = np.cross(a, z)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * (1.0 / (1.0 + c))


def tilted(up, tx_deg: float, ty_deg: float) -> np.ndarray:
    """``up`` tilted by ``tx`` about X then ``ty`` about Y."""
    ax, ay = math.radians(tx_deg), math.radians(ty_deg)
    rx = np.array(
        [[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]]
    )
    ry = np.array(
        [[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]]
    )
    return unit(ry @ rx @ unit(up))


def plane_candidates(support: dict[str, Any]) -> list[dict[str, Any]]:
    """Each support-plane candidate's normal as an up-axis in the export frame.

    The export applied ``world_from_scene``, so a scene-frame normal n is
    ``R @ n`` in the export frame; the chosen plane's normal lands on +Z.
    """
    rotation = np.asarray(support["world_from_scene"], dtype=float)[:3, :3]
    found = []
    for index, cand in enumerate(support.get("candidates") or []):
        up_world = unit(rotation @ unit(cand["normal"]))
        found.append(
            {
                "name": f"plane_{index}",
                "source": "support-plane candidate",
                "plane_fit_chosen": index == support.get("chosen"),
                "up_export": [round(float(v), 6) for v in up_world],
                "up_scene": [round(float(v), 6) for v in unit(cand["normal"])],
                "plane_fit_score": cand.get("score"),
                "inliers": cand.get("inliers"),
            }
        )
    if not found:
        raise ValueError("export_evidence.json lists no support-plane candidates")
    return found


def tilt_candidates(best: dict[str, Any], rotation: np.ndarray) -> list[dict[str, Any]]:
    """Small tilts of the best candidate's up-axis."""
    out = []
    for tx in (0.0, *TILTS_DEG):
        for ty in (0.0, *TILTS_DEG):
            if tx == 0.0 and ty == 0.0:
                continue
            up = tilted(best["up_export"], tx, ty)
            out.append(
                {
                    "name": f"{best['name']}_tilt_x{tx:+g}_y{ty:+g}",
                    "source": "tilt of the best plane",
                    "plane_fit_chosen": False,
                    "up_export": [round(float(v), 6) for v in up],
                    "up_scene": [round(float(v), 6) for v in unit(rotation.T @ up)],
                }
            )
    return out


def choose(scored: list[dict[str, Any]]) -> dict[str, Any]:
    """Lowest late drift among orientations where nothing fell off.

    Within ``TIE_M`` of the best, the plane fit's choice wins.
    """
    standing = [c for c in scored if not c.get("fell_off")] or scored
    best = min(standing, key=lambda c: c["late_drift_mean_m"])
    for cand in standing:
        if (
            cand.get("plane_fit_chosen")
            and cand["late_drift_mean_m"] <= best["late_drift_mean_m"] + TIE_M
        ):
            return cand
    return best


def quat_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Unit quaternion (w, x, y, z) of a proper rotation matrix."""
    r = np.asarray(rotation, dtype=float)
    w = math.sqrt(max(0.0, 1.0 + r[0, 0] + r[1, 1] + r[2, 2])) / 2.0
    if w > 1e-6:
        return (
            w,
            (r[2, 1] - r[1, 2]) / (4 * w),
            (r[0, 2] - r[2, 0]) / (4 * w),
            (r[1, 0] - r[0, 1]) / (4 * w),
        )
    i = int(np.argmax(np.diag(r)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(max(0.0, 1.0 + r[i, i] - r[j, j] - r[k, k])) * 2.0
    q = [0.0, 0.0, 0.0]
    q[i] = s / 4.0
    q[j] = (r[j, i] + r[i, j]) / s
    q[k] = (r[k, i] + r[i, k]) / s
    return ((r[k, j] - r[j, k]) / s, q[0], q[1], q[2])


def run_sweep(
    out: Path | str, *, runner=subprocess.run, python: str = ISAACLAB_PYTHON
) -> int:
    """Pipeline-env entry: start the Isaac Lab half and read ``orientation_sweep.json``."""
    out = Path(out)
    (out / RESULT).unlink(missing_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    done = runner(
        [python, str(Path(__file__).resolve()), "--isaac", "--out", str(out)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=WATCHDOG_S + 120,
    )
    try:
        result = json.loads((out / RESULT).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        tail = ((done.stderr or "") or (done.stdout or "")).strip().splitlines()[-4:]
        print(
            f"orientation sweep exited {done.returncode}: " + " | ".join(tail),
            file=sys.stderr,
        )
        return 1
    print(f"orientation sweep {result.get('decision')}: {result.get('reason')}")
    return 0 if result.get("decision") == "accept" else 1


def _isaac(out: Path) -> None:
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("HEADLESS", "1")
    import argparse

    from isaaclab.app import add_launcher_args, launch_simulation

    parser = argparse.ArgumentParser()
    add_launcher_args(parser)
    args = parser.parse_args([])
    import isaaclab.sim as sim_utils

    sim_cfg = sim_utils.SimulationCfg(dt=DT)
    with launch_simulation(sim_cfg, args):
        try:
            result = _sweep(out, sim_cfg)
        except Exception:
            result = {
                "decision": "reject",
                "reason": "sweep failed: " + traceback.format_exc()[-1500:],
            }
        (out / RESULT).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        sys.stdout.flush()
        os._exit(0)


def _settle_copies(
    out: Path, sim_cfg, candidates: list[dict[str, Any]], prims: list[str]
) -> None:
    import isaaclab.sim as sim_utils
    import torch
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from pxr import Gf, UsdGeom  # type: ignore[import-not-found]

    sim = sim_utils.SimulationContext(sim_cfg)
    stage = sim.stage
    light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    light.func("/World/Light", light)
    usd = out / "export" / "scene_settle.usda"
    bodies: list[dict[str, Any]] = []
    for index, cand in enumerate(candidates):
        root = f"/World/c{index}"
        xform = UsdGeom.Xform.Define(stage, root)
        xform.AddTranslateOp().Set(Gf.Vec3d(SPACING_M * index, 0.0, LIFT_M))
        w, x, y, z = quat_wxyz(rotation_onto_z(cand["up_export"]))
        xform.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
        slab = sim_utils.CuboidCfg(
            size=SLAB_SIZE_M, collision_props=sim_utils.CollisionPropertiesCfg()
        )
        slab.func(f"{root}/slab", slab, translation=(0.0, 0.0, -SLAB_SIZE_M[2] / 2))
        sim_utils.UsdFileCfg(usd_path=str(usd)).func(
            f"{root}/scene", sim_utils.UsdFileCfg(usd_path=str(usd))
        )
        bodies.append(
            {
                name: RigidObject(
                    RigidObjectCfg(prim_path=f"{root}/scene/{name}", spawn=None)
                )
                for name in prims
            }
        )
    sim.reset()
    dt = sim.get_physics_dt()
    start = [
        {n: b.data.root_pos_w.torch[0].clone() for n, b in group.items()}
        for group in bodies
    ]
    late_step = round(LATE_FROM_S / dt)
    late: list[dict[str, Any]] = []
    for step in range(round(SETTLE_S / dt)):
        if step == late_step:
            late = [
                {n: b.data.root_pos_w.torch[0].clone() for n, b in group.items()}
                for group in bodies
            ]
        for group in bodies:
            for body in group.values():
                body.write_data_to_sim()
        sim.step()
        for group in bodies:
            for body in group.values():
                body.update(dt)
    for cand, group, before, mid in zip(candidates, bodies, start, late, strict=True):
        now = {n: b.data.root_pos_w.torch[0] for n, b in group.items()}
        moved = {n: float(torch.linalg.norm(now[n] - before[n])) for n in group}
        late_moved = {n: float(torch.linalg.norm(now[n] - mid[n])) for n in group}
        speeds = {
            n: float(torch.linalg.norm(b.data.root_lin_vel_w.torch[0]))
            for n, b in group.items()
        }
        dropped = {n: float(before[n][2] - now[n][2]) for n in group}
        cand["drift_mean_m"] = round(sum(moved.values()) / len(moved), 5)
        cand["late_drift_m"] = {n: round(v, 6) for n, v in late_moved.items()}
        cand["late_drift_mean_m"] = round(sum(late_moved.values()) / len(late_moved), 6)
        cand["final_speed_max_m_s"] = round(max(speeds.values()), 5)
        cand["fell_off"] = max(dropped.values()) > FELL_M


def _sweep(out: Path, sim_cfg) -> dict[str, Any]:
    export = json.loads((out / "export_evidence.json").read_text(encoding="utf-8"))
    support = export["support"]
    prims = [
        o["prim"].rsplit("/", 1)[-1] for o in export["objects"] if o["route"] == "physx"
    ]
    if not prims:
        raise ValueError("no rigid object in the export")
    rotation = np.asarray(support["world_from_scene"], dtype=float)[:3, :3]
    planes = plane_candidates(support)
    tilts = tilt_candidates(next(p for p in planes if p["plane_fit_chosen"]), rotation)
    candidates = planes + tilts
    _settle_copies(out, sim_cfg, candidates, prims)
    chosen = choose(candidates)
    plane_choice = next(p for p in planes if p["plane_fit_chosen"])
    agrees = chosen["name"] == plane_choice["name"]
    return {
        "decision": "accept",
        "reason": (
            f"up-axis {chosen['name']} (scene frame {chosen['up_scene']}): late drift "
            f"{chosen['late_drift_mean_m'] * 1000:.3f} mm in {LATE_FROM_S:g}-{SETTLE_S:g} s; the plane "
            f"fit's choice {plane_choice['name']}: {plane_choice['late_drift_mean_m'] * 1000:.3f} mm; "
            f"fell off: {[c['name'] for c in candidates if c['fell_off']]}"
        ),
        "chosen": chosen["name"],
        "chosen_up_export": chosen["up_export"],
        "chosen_up_scene": chosen["up_scene"],
        "score_late_drift_mean_m": chosen["late_drift_mean_m"],
        "agrees_with_plane_fit": agrees,
        "candidates": candidates,
        "method": (
            "each candidate up-axis: a copy of export/scene_settle.usda turned so it points up, "
            f"on its own static slab with no ground, settled {SETTLE_S:g} s at {1 / DT:g} Hz; "
            f"an orientation whose objects drop more than {FELL_M:g} m fell off and loses; "
            f"score = mean object displacement from {LATE_FROM_S:g} s to the end; "
            f"ties within {TIE_M * 1000:g} mm keep the plane fit"
        ),
        "objects": prims,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--isaac", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.isaac:
        _isaac(args.out)
        return 0
    return run_sweep(args.out)


if __name__ == "__main__":
    sys.exit(main())
