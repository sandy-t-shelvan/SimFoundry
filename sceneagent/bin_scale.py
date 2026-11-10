# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Metres per splat unit for the bin, from Depth Anything 3 metric depth (RGB only).

The paper states its cuts in "reconstruction units" and reports its kitchen at
0.90-0.93 m per unit without naming the scale method. This splat's unit is
nerfstudio's normalisation, about a fifth of that, so the 0.10-unit centroid
gate would be 2 cm instead of about 9 cm. Here the splat unit is measured in
metres from the views alone: DA3METRIC-LARGE depth (``focal * output / 300``,
the model card's rule, with ``focal`` the COLMAP focal length at the processed
resolution) is compared, cell by cell, with the nearest Gaussian center's depth
in splat units, and the median ratio is the scale. That method, and the paper
scale it converts to, are ours, not the paper's.

The worker runs in the box's ``da3`` env (:mod:`sceneagent.depth_worker`).
"""

from __future__ import annotations

import json
import os
import subprocess
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

# The sa env on 58301075 (probe run 42908974589) imports depth_anything_3 from
# the checkout in /workspace/src, which is read, never written. The root Hugging
# Face cache there holds an incomplete DA3METRIC-LARGE (run 42914501132:
# LocalEntryNotFoundError offline), so the weights come into our own cache,
# downloaded once and reused, as bin_sam3d does for MoGe.
DEPTH_PYTHON = "/root/.local/share/mamba/envs/sa/bin/python3"
DA3_SRC = "/workspace/src/depth-anything-3/src"
DA3_MODEL = "depth-anything/DA3METRIC-LARGE"
DA3_FOCAL_DIVISOR = 300.0
HF_CACHE = "/workspace/sf4-smoke/hf-cache"
# Midpoint of the paper's 0.90-0.93 m per reconstruction unit (kitchen).
PAPER_METRES_PER_UNIT = 0.915
SCALE_CELL_PX = 4
SCALE_RULE = (
    "ours: median over 4 px cells (processed resolution) of DA3METRIC-LARGE depth, "
    "focal * output / 300 with the COLMAP focal at that resolution, over the nearest "
    "Gaussian center's depth in splat units; lengths are then divided by 0.915 m "
    "(the paper's 0.90-0.93 m per unit) before the paper's cuts"
)
WORKER_TIMEOUT_S = 900

Runner = Callable[..., Any]


def worker_command(images: Path, depth_dir: Path) -> list[str]:
    return [
        DEPTH_PYTHON,
        "-m",
        "sceneagent.depth_worker",
        "--images",
        str(images),
        "--out",
        str(depth_dir),
        "--model",
        DA3_MODEL,
    ]


def worker_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    """This repo first, then the DA3 source tree; weights in our own HF cache."""
    env = dict(os.environ if base is None else base)
    repo = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = os.pathsep.join([repo, DA3_SRC])
    env.pop("HF_HUB_OFFLINE", None)
    env["HF_HUB_CACHE"] = HF_CACHE
    return env


def run_depth(
    images: Path, depth_dir: Path, runner: Runner = subprocess.run
) -> dict[str, Any]:
    """Run the DA3 worker. Raises with the worker's last lines when it fails."""
    argv = worker_command(images, depth_dir)
    done = runner(
        argv,
        env=worker_environment(),
        capture_output=True,
        text=True,
        check=False,
        timeout=WORKER_TIMEOUT_S,
    )
    if done.returncode != 0:
        # Eight lines: enough to name the frame a traceback ends in.
        tail = ((done.stderr or "") or (done.stdout or "")).strip().splitlines()[-8:]
        raise RuntimeError(
            f"depth worker exited {done.returncode}: " + " | ".join(tail)
        )
    return json.loads((depth_dir / "meta.json").read_text(encoding="utf-8"))


def cell_ratios(
    u: np.ndarray,
    v: np.ndarray,
    z_units: np.ndarray,
    depth_out: np.ndarray,
    image_wh: tuple[int, int],
    focal_px: float,
) -> np.ndarray:
    """Metric DA3 depth over splat depth, one ratio per occupied cell.

    ``u``, ``v`` are full-resolution pixel coordinates of the Gaussian
    centers and ``z_units`` their depths in splat units. ``depth_out`` is the
    DA3 output at the processed resolution; it becomes metres with the focal
    length scaled to that resolution. Each ``SCALE_CELL_PX`` cell compares
    the median DA3 depth with the nearest center, the surface both see.
    """
    height_p, width_p = depth_out.shape
    width, height = image_wh
    sx, sy = width_p / width, height_p / height
    metres = depth_out * (focal_px * (sx + sy) / 2.0) / DA3_FOCAL_DIVISOR
    with np.errstate(invalid="ignore"):
        col = np.floor(u * sx)
        row = np.floor(v * sy)
        keep = (
            (z_units > 0) & (col >= 0) & (col < width_p) & (row >= 0) & (row < height_p)
        )
    col = col[keep].astype(np.int64) // SCALE_CELL_PX
    row = row[keep].astype(np.int64) // SCALE_CELL_PX
    z = z_units[keep]
    columns = -(-width_p // SCALE_CELL_PX)
    rows = -(-height_p // SCALE_CELL_PX)
    nearest = np.full(columns * rows, np.inf)
    np.minimum.at(nearest, row * columns + col, z)
    cropped = metres[: rows * SCALE_CELL_PX, : columns * SCALE_CELL_PX]
    padded = np.full((rows * SCALE_CELL_PX, columns * SCALE_CELL_PX), np.nan)
    padded[: cropped.shape[0], : cropped.shape[1]] = cropped
    blocks = padded.reshape(rows, SCALE_CELL_PX, columns, SCALE_CELL_PX)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # cells with no DA3 pixel
        da3 = np.nanmedian(
            blocks.transpose(0, 2, 1, 3).reshape(rows * columns, -1), axis=1
        )
    ok = np.isfinite(nearest) & np.isfinite(da3) & (da3 > 0)
    return da3[ok] / nearest[ok]


def summarize(ratios: list[np.ndarray]) -> dict[str, Any]:
    values = np.concatenate(ratios) if ratios else np.empty(0)
    if not values.size:
        raise ValueError("no cell had both a Gaussian center and a DA3 depth")
    p25, median, p75 = np.percentile(values, [25, 50, 75])
    return {
        "metres_per_unit": float(median),
        "p25": float(p25),
        "p75": float(p75),
        "cells": int(values.size),
        "to_paper_units": float(median) / PAPER_METRES_PER_UNIT,
    }
