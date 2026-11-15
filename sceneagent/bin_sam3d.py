# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cut 2 phase 4, SF-29: one SAM 3D Objects mesh per grouped bin object.

Each object from phase 3 (``association_evidence.json``) is reconstructed
from one view: the one where its mask covers the most pixels, among masks
whose distance-normalised area is not an outlier (``MAX_AREA_TO_MEDIAN``). SAM 3D takes
that RGB view and that SAM mask and returns a mesh plus its pose in the
camera (:mod:`sceneagent.sam3d_worker`, run in the box's ``sam3d-objects``
env). Nothing else is substituted for a failed object; it stays a failure
in the evidence and the ledger.

The view choice is ours: the paper reconstructs each object with SAM 3D but
does not say which view it gives it.

    python -m sceneagent.bin_sam3d --out /workspace/sf4-smoke/bin
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from sceneagent.bin_run import load_scene

# The sam3d10 env and checkout on 58301075 (probe run 42908974589):
# sam3d_objects 0.0.1 editable from /root/sam-3d-objects, pytorch3d 0.7.8,
# kaolin 0.17.0, moge 1.0.0. The box's other env, sam3d, has no pytorch3d.
# The checkout is read, never written.
SAM3D_ENV = "/opt/conda/envs/sam3d10"
SAM3D_PYTHON = f"{SAM3D_ENV}/bin/python3"
SAM3D_ROOT = "/root/sam-3d-objects"
# Where setup.md's download leaves pipeline.yaml (checkpoints/hf), where an
# unfinished move leaves it, and the Hugging Face cache. First match wins.
CONFIG_CANDIDATES = (
    f"{SAM3D_ROOT}/checkpoints/hf/pipeline.yaml",
    f"{SAM3D_ROOT}/checkpoints/hf-download/checkpoints/pipeline.yaml",
)
HF_SNAPSHOTS = "/root/.cache/huggingface/hub/models--facebook--sam-3d-objects/snapshots"
HF_CACHE = "/workspace/sf4-smoke/hf-cache"
SEED = 42
ITEM = "SAM 3D meshes"
REQUIRED = (
    "benchmark scene",
    "COLMAP camera poses",
    "full-scene splat",
    "SAM masks",
    "Gaussian association",
    ITEM,
)
VIEW_RULE = (
    "ours: the view where the object's mask has the most pixels, among masks "
    "whose area times the object's depth squared is at most 2x the median"
)
# A mask's area times the object's depth squared is roughly the object's
# physical area as that view sees it: it stays near the median as a
# handheld camera moves in and out, and jumps when a mask takes in the
# object's surroundings. On the Mailbox video one view of the cube was
# 31107 px from 8.7 units away, 5.4x the median by that measure: the mask
# had spilled onto the tape square. The largest-mask rule chose it, SAM 3D
# made a mesh three times the cube's size and placement rejected it
# (silhouette 0.22). Excluding it chooses the view (1.02x) that placed the
# cube in another run (silhouette 0.73). PutMarkerInCup's cup and marker,
# whose largest masks are close-ups (0.94x, 1.04x), keep their views.
MAX_AREA_TO_MEDIAN = 2.0
# Below this many masks with a depth, a median says little; every mask stays eligible.
MIN_MASKS_FOR_MEDIAN = 3
WORKER_TIMEOUT_S = 5400

Runner = Callable[..., Any]


def find_config(
    candidates: tuple[str, ...] = CONFIG_CANDIDATES, snapshots: str = HF_SNAPSHOTS
) -> Path:
    """The first ``pipeline.yaml`` that exists. Raises when none does."""
    paths = [Path(c) for c in candidates]
    paths += sorted(Path(snapshots).glob("*/checkpoints/pipeline.yaml"))
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError("no SAM 3D pipeline.yaml in " + ", ".join(map(str, paths)))


def object_view_depths(
    association: dict[str, Any],
) -> dict[tuple[int, str], float]:
    """Camera-frame depth of each object's centroid in each of its mask views.

    Centroids are in the splat's dataparser frame; ``splat_to_colmap`` brings
    them into the COLMAP frame of the poses. Empty when the inputs are
    missing, so the view choice falls back to raw areas.
    """
    from sceneagent.bin_association import (
        qvec_to_rotation,
        read_images_bin,
        splat_to_colmap,
    )

    inputs = association.get("inputs") or {}
    try:
        images = read_images_bin(Path(inputs["colmap_model"]) / "images.bin")
        frame = json.loads(
            Path(inputs["dataparser_transforms"]).read_text(encoding="utf-8")
        )
    except (KeyError, OSError, ValueError):
        return {}
    transform, scale = np.asarray(frame["transform"], float), float(frame["scale"])
    depths: dict[tuple[int, str], float] = {}
    for obj in association.get("object_list") or []:
        if obj.get("centroid") is None:
            continue
        centre = splat_to_colmap(
            np.asarray([obj["centroid"]], float), transform, scale
        )[0]
        for mask in obj.get("masks") or []:
            image = images.get(mask["view"])
            if image is None:
                continue
            cam = qvec_to_rotation(image["qvec"]) @ centre + np.asarray(image["tvec"])
            if cam[2] > 0:
                depths[(int(obj["object"]), str(mask["view"]))] = float(cam[2])
    return depths


def choose_views(
    objects: list[dict[str, Any]],
    detections: dict[str, list[dict[str, Any]]],
    depths: dict[tuple[int, str], float] | None = None,
) -> list[dict[str, Any]]:
    """For each object, its largest mask that is not a distance-normalised outlier.

    A mask's area times the object's depth in that view, squared, is compared
    with the median over the object's masks; one more than
    ``MAX_AREA_TO_MEDIAN`` times it took in its surroundings and is skipped.
    Without depths for at least ``MIN_MASKS_FOR_MEDIAN`` masks, every mask is
    eligible (the old rule). The skipped views are recorded.
    """
    depths = depths or {}
    area = {
        (view, d["mask_index"]): int(d.get("mask_area_px", 0))
        for view, dets in detections.items()
        for d in dets
        if d.get("kept")
    }
    chosen = []
    for obj in objects:
        masks = list(obj["masks"])
        normalised = {
            m["view"]: area.get((m["view"], m["mask_index"]), 0)
            * depths[(int(obj["object"]), m["view"])] ** 2
            for m in masks
            if (int(obj["object"]), m["view"]) in depths
        }
        eligible, skipped, median = masks, [], None
        if len(normalised) >= MIN_MASKS_FOR_MEDIAN:
            median = float(np.median(list(normalised.values())))
            limit = MAX_AREA_TO_MEDIAN * median
            eligible = [m for m in masks if normalised.get(m["view"], 0.0) <= limit]
            skipped = sorted(m["view"] for m in masks if m not in eligible)
            if not eligible:
                eligible, skipped = masks, []
        best = max(
            eligible,
            key=lambda m: (area.get((m["view"], m["mask_index"]), 0), m["view"]),
        )
        chosen.append(
            {
                "object": obj["object"],
                "view": best["view"],
                "mask_index": best["mask_index"],
                "phrase": best.get("phrase"),
                "mask_area_px": area.get((best["view"], best["mask_index"]), 0),
                "depth_normalised": median is not None,
                "skipped_area_outliers": skipped,
            }
        )
    return chosen


def worker_command(jobs: Path, out: Path, config: Path) -> list[str]:
    return [
        SAM3D_PYTHON,
        "-m",
        "sceneagent.sam3d_worker",
        "--jobs",
        str(jobs),
        "--out",
        str(out),
        "--config",
        str(config),
        "--notebook",
        f"{SAM3D_ROOT}/notebook",
        "--seed",
        str(SEED),
    ]


def worker_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    """This repo on the path and the env named as conda would name it.

    The notebook's ``inference.py`` sets ``CUDA_HOME`` from ``CONDA_PREFIX``
    on import, which an unactivated env lacks (probe run 42759520898).
    """
    env = dict(os.environ if base is None else base)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
    env["CONDA_PREFIX"] = SAM3D_ENV
    env["PATH"] = f"{SAM3D_ENV}/bin" + os.pathsep + env.get("PATH", "")
    # SAM 3D's pipeline loads MoGe from the Hub; the box's cache lacks it
    # (run 42759919334). Downloads land in our own dir, never in /root's cache.
    env.pop("HF_HUB_OFFLINE", None)
    env["HF_HUB_CACHE"] = HF_CACHE
    return env


def run_bin_sam3d(
    out: Path | str,
    *,
    runner: Runner = subprocess.run,
    config: Callable[[], Path] = find_config,
) -> int:
    """Mesh every grouped object. Return 0 only when every object has a mesh."""
    out = Path(out)
    scene = load_scene(out)
    ledger_path = out / "ledger.json"
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        ledger = {"entries": []}
    entries = [e for e in ledger.get("entries", []) if e.get("item") != ITEM]
    entry: dict[str, Any] = {"item": ITEM}
    evidence: dict[str, Any] = {
        "scene": scene.name,
        "seed": SEED,
        "view_rule": VIEW_RULE,
    }
    try:
        association = json.loads(
            (out / "association_evidence.json").read_text(encoding="utf-8")
        )
        detections = json.loads(
            (out / "masks" / "detections.json").read_text(encoding="utf-8")
        )
        chosen = choose_views(
            association["object_list"], detections, object_view_depths(association)
        )
        sam3d_dir = out / "sam3d"
        sam3d_dir.mkdir(parents=True, exist_ok=True)
        jobs = [
            {
                "object": c["object"],
                "image": str(out / "images" / c["view"]),
                "masks": str(out / "masks" / f"{Path(c['view']).stem}.npz"),
                "mask_index": c["mask_index"],
            }
            for c in chosen
        ]
        jobs_path = sam3d_dir / "jobs.json"
        jobs_path.write_text(json.dumps(jobs, indent=2) + "\n", encoding="utf-8")
        cfg = config()
        evidence["config"] = str(cfg)
        argv = worker_command(jobs_path, sam3d_dir, cfg)
        done = runner(
            argv,
            env=worker_environment(),
            cwd=SAM3D_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=WORKER_TIMEOUT_S,
        )
        tail = ((done.stderr or "") or (done.stdout or "")).strip().splitlines()[-8:]
        evidence["worker"] = {"argv": argv, "returncode": done.returncode, "tail": tail}
        results_path = sam3d_dir / "results.json"
        if not results_path.is_file():
            raise RuntimeError(
                f"sam3d worker exited {done.returncode}: " + " | ".join(tail)
            )
        results = json.loads(results_path.read_text(encoding="utf-8"))
        by_object = {r["object"]: r for r in results["objects"]}
        evidence["load_seconds"] = results.get("load_seconds")
        evidence["objects"] = [
            {
                **c,
                **{
                    k: v
                    for k, v in by_object.get(c["object"], {}).items()
                    if k != "trace"
                },
            }
            for c in chosen
        ]
        meshed = [o for o in evidence["objects"] if o.get("ok")]
        failed = [o for o in evidence["objects"] if not o.get("ok")]
        evidence["meshed"] = len(meshed)
        evidence["failed"] = len(failed)
        if chosen and not failed:
            entry.update(decision="accept", use_gpu=True, objects=len(meshed))
        else:
            first = failed[0].get("error", "no result") if failed else "no objects"
            entry.update(
                decision="reject",
                reason=f"{len(meshed)} of {len(chosen)} objects meshed; first failure: {first}"[
                    -800:
                ],
            )
    except (
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        subprocess.TimeoutExpired,
    ) as exc:
        entry.update(decision="reject", reason=f"{type(exc).__name__}: {exc}"[-800:])
    entries.append(entry)
    ledger.update(scene_name=scene.ledger_name, source=scene.source, entries=entries)
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    (out / "sam3d_evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    decisions = {e.get("item"): e.get("decision") for e in entries}
    required = (scene.input_item, *REQUIRED[1:])
    return 0 if all(decisions.get(item) == "accept" for item in required) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    return run_bin_sam3d(args.out)


if __name__ == "__main__":
    sys.exit(main())
