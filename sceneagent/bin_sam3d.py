# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cut 2 phase 4, SF-29: one SAM 3D Objects mesh per grouped bin object.

Each object from phase 3 (``association_evidence.json``) is reconstructed
from one view: the one where its mask covers the most pixels. SAM 3D takes
that RGB view and that SAM mask and returns a mesh plus its pose in the
camera (:mod:`sceneagent.sam3d_worker`, run in the box's ``sam3d-objects``
env). Nothing else is substituted for a failed object; it stays a failure
in the evidence and the ledger.

The view choice (largest mask) is ours: the paper reconstructs each object
with SAM 3D but does not say which view it gives it.

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

from sceneagent.bin_run import SCENE, SOURCE

# The sam3d-objects env and checkout on 59601312 (probe run 42759520898):
# sam3d_objects 0.0.1 editable from /root/software/sam-3d-objects, pytorch3d
# 0.7.8, kaolin 0.17.0, moge 1.0.0. The checkout is read, never written.
SAM3D_ENV = "/root/miniforge3/envs/sam3d-objects"
SAM3D_PYTHON = f"{SAM3D_ENV}/bin/python3"
SAM3D_ROOT = "/root/software/sam-3d-objects"
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
VIEW_RULE = "ours: the view where the object's mask has the most pixels"
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


def choose_views(
    objects: list[dict[str, Any]], detections: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """For each object, its mask with the largest pixel area."""
    area = {
        (view, d["mask_index"]): int(d.get("mask_area_px", 0))
        for view, dets in detections.items()
        for d in dets
        if d.get("kept")
    }
    chosen = []
    for obj in objects:
        best = max(
            obj["masks"],
            key=lambda m: (area.get((m["view"], m["mask_index"]), 0), m["view"]),
        )
        chosen.append(
            {
                "object": obj["object"],
                "view": best["view"],
                "mask_index": best["mask_index"],
                "phrase": best.get("phrase"),
                "mask_area_px": area.get((best["view"], best["mask_index"]), 0),
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
    ledger_path = out / "ledger.json"
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        ledger = {"entries": []}
    entries = [e for e in ledger.get("entries", []) if e.get("item") != ITEM]
    entry: dict[str, Any] = {"item": ITEM}
    evidence: dict[str, Any] = {"scene": SCENE, "seed": SEED, "view_rule": VIEW_RULE}
    try:
        association = json.loads(
            (out / "association_evidence.json").read_text(encoding="utf-8")
        )
        detections = json.loads(
            (out / "masks" / "detections.json").read_text(encoding="utf-8")
        )
        chosen = choose_views(association["object_list"], detections)
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
    ledger.update(scene_name=f"{SOURCE} {SCENE}", source=SOURCE, entries=entries)
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    (out / "sam3d_evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    decisions = {e.get("item"): e.get("decision") for e in entries}
    return 0 if all(decisions.get(item) == "accept" for item in REQUIRED) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    return run_bin_sam3d(args.out)


if __name__ == "__main__":
    sys.exit(main())
