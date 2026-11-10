# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cut 2 phase 2: camera poses for GraspClutter6D scene 000091 on the GPU box.

The scene is fetched once into ``--data`` (``gc6d_fetch``; a matching manifest
makes later runs a no-op). Its 52 images are four cameras at 13 viewpoints:
ids 1, 5, ..., 49 and 2, 6, ..., 50 are 1920x1080, ids 3, 7, ... are
3840x2160 and ids 4, 8, ... are 1920x1200. The paper's "13 RGB views at
1920x1080" fits either 1920x1080 camera, so the first one is used and
recorded. Its PNGs are copied unchanged and posed by COLMAP with the
exhaustive matcher (13 unordered stills, KAN-4 comment 10311 allows it). The
reconstruction gate is the repository's: at least ``max(10, ceil(0.8 * 13))``
= 11 of 13 views registered in one model.

    python -m sceneagent.bin_run --out /workspace/sf4-smoke/bin
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sceneagent.gc6d_fetch import fetch_scene
from sceneagent.pipeline import write_plan

DATA_DIR = "/workspace/data/gc6d"
SCENE = "000091"
CAMERA_FIRST_ID = 1
CAMERAS = 4
VIEWS = 13
VIEW_SIZE = (1920, 1080)
MATCHER = "exhaustive_matcher"
SOURCE = "GraspClutter6D"


def view_ids(first: int = CAMERA_FIRST_ID) -> list[int]:
    """One camera's 13 image ids: the scene interleaves four cameras."""
    return [first + CAMERAS * i for i in range(VIEWS)]


def stage_views(scene_dir: Path, image_dir: Path, ids: list[int]) -> list[str]:
    """Copy the chosen RGB PNGs unchanged. Refuse a view of the wrong size."""
    from PIL import Image

    image_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for image_id in ids:
        source = scene_dir / "rgb" / f"{image_id:06d}.png"
        with Image.open(source) as image:
            if image.size != VIEW_SIZE:
                raise ValueError(f"{source.name} is {image.size}, not {VIEW_SIZE}")
        shutil.copyfile(source, image_dir / source.name)
        names.append(source.name)
    return names


def clear_pose_state(out: Path) -> None:
    """Remove the staged images, COLMAP database and models from ``out``."""
    shutil.rmtree(out / "images", ignore_errors=True)
    shutil.rmtree(out / "sparse", ignore_errors=True)
    (out / "database.db").unlink(missing_ok=True)


def _default_colmap(image_dir: Path, work_dir: Path) -> dict[str, Any]:
    from sceneagent.gpu_headroom import require_gpu_headroom
    from sceneagent.run4 import run_colmap_on_images, run_colmap_step

    return run_colmap_on_images(
        image_dir,
        work_dir,
        gpu_check=require_gpu_headroom,
        command_runner=run_colmap_step,
        matcher=MATCHER,
    )


def run_bin_poses(
    out: Path | str,
    data: Path | str = DATA_DIR,
    *,
    fetch: Callable[[str, Path], dict[str, Any]] = fetch_scene,
    colmap: Callable[[Path, Path], dict[str, Any]] = _default_colmap,
) -> int:
    """Write plan, ledger and bin evidence. Return 0 only when the poses pass."""
    out, data = Path(out), Path(data)
    out.mkdir(parents=True, exist_ok=True)
    write_plan(
        out,
        f"{SOURCE} {SCENE}",
        SOURCE,
        [
            {"skill": "benchmark scene", "status": "implemented_phase1"},
            {
                "skill": "COLMAP camera poses",
                "status": "implemented_phase1",
                "matcher": MATCHER,
            },
        ],
    )
    evidence: dict[str, Any] = {"scene": SCENE, "camera_first_id": CAMERA_FIRST_ID}
    entries: list[dict[str, Any]] = []
    try:
        manifest = fetch(SCENE, data)
        scene_dir = data / "scenes" / SCENE
        ids = view_ids()
        # A dir another pipeline wrote can hold foreign frames and a COLMAP
        # database that already has their features; both are inputs here.
        clear_pose_state(out)
        names = stage_views(scene_dir, out / "images", ids)
        entries.append(
            {
                "item": "benchmark scene",
                "decision": "accept",
                "files": len(manifest["files"]),
                "license": manifest.get("license"),
                "views": names,
            }
        )
        result = colmap(out / "images", out)
        evidence["colmap"] = {
            key: result.get(key) for key in ("ok", "reason", "failed_step", "use_gpu")
        }
        evidence["quality_bar"] = (result.get("gate_document") or {}).get("quality_bar")
        pose: dict[str, Any] = {"item": "COLMAP camera poses", "matcher": MATCHER}
        inputs = (evidence["quality_bar"] or {}).get("inputs")
        if result.get("ok") and inputs != len(names):
            pose.update(
                decision="reject",
                reason=f"COLMAP saw {inputs} images, not the {len(names)} bin views",
            )
        elif result.get("ok"):
            pose.update(decision="accept", use_gpu=result.get("use_gpu"), frames=names)
        else:
            pose.update(
                decision="reject",
                reason=result.get("reason"),
                failed_step=result.get("failed_step"),
            )
        entries.append(pose)
    except (OSError, ValueError, KeyError) as exc:
        entries.append(
            {"item": "benchmark scene", "decision": "reject", "reason": str(exc)}
        )
    ledger = {"scene_name": f"{SOURCE} {SCENE}", "source": SOURCE, "entries": entries}
    (out / "ledger.json").write_text(
        json.dumps(ledger, indent=2) + "\n", encoding="utf-8"
    )
    (out / "bin_evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    passed = len(entries) == 2 and all(
        entry["decision"] == "accept" for entry in entries
    )
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path(DATA_DIR))
    args = parser.parse_args(argv)
    return run_bin_poses(args.out, args.data)


if __name__ == "__main__":
    sys.exit(main())
