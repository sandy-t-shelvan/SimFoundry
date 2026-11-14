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
import math
import shutil
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
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
SCENE_FILE = "scene.json"
# Paper section 3.1 runs SAM on every 10th view of a capture.
VIDEO_MASK_STRIDE = 10
VIDEO_MASK_DIR = "mask_views"
# Ours: a video view may miss every object (a frame looking past them), so a
# video keeps going when at least this share of its mask views kept a mask.
VIDEO_MIN_MASKED_SHARE = 0.8


@dataclass(frozen=True)
class Scene:
    """What the bin stages need to know about the capture in ``--out``.

    The pose stage writes it to ``scene.json``; every later stage reads it
    from there. A directory without the file is the bin's (the bin runs
    predate it), so the bin's behaviour and score do not move.
    """

    source: str
    name: str
    # The ledger's scene_name; each stage checks it is still this scene's.
    ledger_name: str
    # Images COLMAP must register into one model.
    views: int
    # The ledger item the capture is recorded under before the poses.
    input_item: str
    # Where the views that get masks and depth live, under ``--out``.
    view_dir: str = "images"
    min_masked_views: int = VIEWS
    ground_truth: bool = True
    # None keeps the bin's prompt (mask_filter.PROMPT).
    prompt: str | None = None


BIN_SCENE = Scene(
    source=SOURCE,
    name=SCENE,
    ledger_name=f"{SOURCE} {SCENE}",
    views=VIEWS,
    input_item="benchmark scene",
)


def load_scene(out: Path | str) -> Scene:
    """The scene ``out`` holds: ``scene.json`` if written, else the bin."""
    path = Path(out) / SCENE_FILE
    if not path.is_file():
        return BIN_SCENE
    return Scene(**json.loads(path.read_text(encoding="utf-8")))


def write_scene(out: Path | str, scene: Scene) -> None:
    Path(out, SCENE_FILE).write_text(
        json.dumps(asdict(scene), indent=2) + "\n", encoding="utf-8"
    )


def expected_views(out: Path | str, scene: Scene) -> int:
    """How many views the masks and association must cover.

    The bin: all 13 of its photos, however many are on disk. A video: the
    mask views its pose stage staged.
    """
    if scene == BIN_SCENE:
        return VIEWS
    folder = Path(out) / scene.view_dir
    return len([*folder.glob("*.png"), *folder.glob("*.jpg")])


def is_bin(out: Path | str) -> bool:
    """Only the bin may fall back to posing the GraspClutter6D photos."""
    return load_scene(out) == BIN_SCENE


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
    """Remove everything an earlier pose attempt left in ``out``.

    The staged images, the COLMAP database (with its WAL and SHM side
    files, so no match survives into a fresh database), the models, the
    gate's analyzer evidence and the pose records. A rerun must never judge
    or extend a previous attempt's matches.
    """
    from sceneagent.reconstruction_gate import EVIDENCE_DIRNAME

    for name in ("images", "sparse", EVIDENCE_DIRNAME):
        shutil.rmtree(out / name, ignore_errors=True)
    for name in (
        "database.db",
        "database.db-wal",
        "database.db-shm",
        "colmap_poses.json",
        "run4_evidence.json",
    ):
        (out / name).unlink(missing_ok=True)


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
    if not is_bin(out):
        # Restaging the bin photos would delete another capture's frames.
        raise ValueError(f"{out} holds {load_scene(out).name}, not the bin")
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


def mask_views(names: list[str], stride: int = VIDEO_MASK_STRIDE) -> list[str]:
    """Every ``stride``-th posed frame, from the first (paper section 3.1)."""
    return sorted(names)[::stride]


def run_video_poses(
    video: Path | str,
    out: Path | str,
    prompt: str,
    *,
    poses: Callable[[list[str]], int] | None = None,
) -> int:
    """Pose a video with run4 and lay its output out for the bin stages.

    run4 samples the 300 sharpest-per-bin frames into ``images/`` and runs
    COLMAP (sequential matcher, incremental mapper: the box has COLMAP
    3.11.1, which has no global mapper). This adds what the bin stages read:
    ``bin_evidence.json`` with run4's quality bar, the mask views (every
    10th frame the chosen model registered) and ``scene.json``.
    """
    from sceneagent.bin_association import read_images_bin
    from sceneagent.run4 import main as run4_main
    from sceneagent.run4_sampling import BIN_COUNT
    from sceneagent.video import VIDEO_SOURCE

    out, video = Path(out), Path(video)
    out.mkdir(parents=True, exist_ok=True)
    scene = Scene(
        source=VIDEO_SOURCE,
        name=video.stem,
        ledger_name=video.stem,
        views=BIN_COUNT,
        input_item="video",
        view_dir=VIDEO_MASK_DIR,
        min_masked_views=0,
        ground_truth=False,
        prompt=prompt,
    )
    # Written before anything else: a stage that finds no scene.json treats
    # the directory as the bin's.
    write_scene(out, scene)
    clear_pose_state(out)
    shutil.rmtree(out / VIDEO_MASK_DIR, ignore_errors=True)
    code = (poses or run4_main)([str(video), "--out", str(out)])
    try:
        run4 = json.loads((out / "run4_evidence.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        run4 = {}
    bar = run4.get("quality_bar") or {}
    (out / "bin_evidence.json").write_text(
        json.dumps({"scene": scene.name, "quality_bar": bar}, indent=2) + "\n",
        encoding="utf-8",
    )
    if code != 0 or not bar.get("pass") or bar.get("model") is None:
        return 1
    model = out / "sparse" / str(bar["model"])
    views = mask_views(list(read_images_bin(model / "images.bin")))
    target = out / VIDEO_MASK_DIR
    target.mkdir()
    for view in views:
        shutil.copyfile(out / "images" / view, target / view)
    write_scene(
        out,
        replace(
            scene,
            views=int(bar["inputs"]),
            min_masked_views=math.ceil(VIDEO_MIN_MASKED_SHARE * len(views)),
        ),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path(DATA_DIR))
    args = parser.parse_args(argv)
    return run_bin_poses(args.out, args.data)


if __name__ == "__main__":
    sys.exit(main())
