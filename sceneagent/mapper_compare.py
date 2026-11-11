# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Incremental against global COLMAP mapping, on the bin's 13 views and Fruits' 300 frames.

Every pose run so far used COLMAP's incremental ``mapper``: it registers one
image at a time. COLMAP 4's ``global_mapper`` solves every camera at once
(rotation averaging, then global positioning). This experiment runs both with
the same COLMAP 4.1.1 binary on the same features and matches, so only the
mapper differs:

- bin: exhaustive matching, then incremental and global mapping;
- fruits: sequential matching (each frame against the next 5) with both
  mappers, plus exhaustive matching with the global mapper, which connects
  every frame pair at once.

Each variant reports registered images, points, mean observations per image,
mean reprojection error and seconds. The bin has ground truth: its 13 COLMAP
camera centers are fitted to the dataset's metric camera centers with one
similarity, and the RMS of that fit (metres) is the pose accuracy. The video
has none. Inputs are copied into ``mapper_compare/`` and nothing else in the
run dir changes.

    python -m sceneagent.mapper_compare --out /workspace/sf4-smoke/bin
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from sceneagent.bin_run import DATA_DIR, SCENE, SOURCE

# COLMAP 4.1.1 with CUDA (has global_mapper) on 58301075, inside the nerf env
# (probe run 42908214934). 59601312 had it at /opt/colmap-4.1.1/bin/colmap.
COLMAP4 = "/root/.local/share/mamba/envs/nerf/bin/colmap"
FRUITS_FRAMES = "/workspace/sf4-smoke/run/images"
ITEM = "mapper comparison"
STEP_TIMEOUT_S = 2400
# (dataset, matcher, mapper)
VARIANTS = (
    ("bin", "exhaustive_matcher", "mapper"),
    ("bin", "exhaustive_matcher", "global_mapper"),
    ("fruits", "sequential_matcher", "mapper"),
    ("fruits", "sequential_matcher", "global_mapper"),
    ("fruits", "exhaustive_matcher", "global_mapper"),
)

Runner = Callable[..., Any]


def stage_images(source: Path, pattern: str, target: Path) -> int:
    """Copy the matching images into an empty ``target``; return how many."""
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    count = 0
    for path in sorted(source.glob(pattern)):
        shutil.copyfile(path, target / path.name)
        count += 1
    return count


def extract_command(database: Path, images: Path) -> list[str]:
    return [
        COLMAP4,
        "feature_extractor",
        "--database_path",
        str(database),
        "--image_path",
        str(images),
        "--ImageReader.single_camera",
        "1",
        "--FeatureExtraction.use_gpu",
        "1",
    ]


def match_command(database: Path, matcher: str) -> list[str]:
    argv = [
        COLMAP4,
        matcher,
        "--database_path",
        str(database),
        "--FeatureMatching.use_gpu",
        "1",
    ]
    if matcher == "sequential_matcher":
        argv += ["--SequentialMatching.overlap", "5"]
    return argv


def map_command(mapper: str, database: Path, images: Path, output: Path) -> list[str]:
    return [
        COLMAP4,
        mapper,
        "--database_path",
        str(database),
        "--image_path",
        str(images),
        "--output_path",
        str(output),
    ]


def run_step(argv: list[str], runner: Runner) -> dict[str, Any]:
    started = time.monotonic()
    try:
        done = runner(
            argv, capture_output=True, text=True, check=False, timeout=STEP_TIMEOUT_S
        )
        code, text = done.returncode, (done.stderr or "") + (done.stdout or "")
    except subprocess.TimeoutExpired:
        code, text = 124, f"timed out after {STEP_TIMEOUT_S} s"
    except OSError as exc:
        code, text = 127, f"{type(exc).__name__}: {exc}"
    return {
        "step": argv[1],
        "returncode": code,
        "seconds": round(time.monotonic() - started, 1),
        "tail": text.strip().splitlines()[-4:],
    }


def parse_images_txt(text: str) -> dict[str, dict[str, Any]]:
    """COLMAP ``images.txt``: a pose line per image, then its points line."""
    rows = [line for line in text.splitlines() if line and not line.startswith("#")]
    images = {}
    for line in rows[0::2]:
        parts = line.split()
        images[parts[9]] = {
            "qvec": [float(v) for v in parts[1:5]],
            "tvec": [float(v) for v in parts[5:8]],
        }
    return images


def convert_to_text(model: Path, text_dir: Path, runner: Runner) -> dict[str, Any]:
    shutil.rmtree(text_dir, ignore_errors=True)
    text_dir.mkdir(parents=True)
    argv = [
        COLMAP4,
        "model_converter",
        "--input_path",
        str(model),
        "--output_path",
        str(text_dir),
        "--output_type",
        "TXT",
    ]
    return run_step(argv, runner)


def text_model_stats(text_dir: Path) -> dict[str, Any]:
    """Registered images, points, observations per image and mean reprojection error.

    Read from the text export rather than ``model_analyzer``'s log, whose
    wording ``reconstruction_gate`` pins to COLMAP 3.11.1. The error is the
    mean of the points' own errors (``points3D.txt`` ERROR column).
    """
    images = parse_images_txt((text_dir / "images.txt").read_text(encoding="utf-8"))
    errors, observations = [], 0
    for line in (text_dir / "points3D.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        errors.append(float(parts[7]))
        observations += (len(parts) - 8) // 2
    registered = len(images)
    return {
        "registered": registered,
        "points": len(errors),
        "mean_obs_per_image": observations / registered if registered else None,
        "mean_reproj_err_px": float(np.mean(errors)) if errors else None,
        "images": images,
    }


def largest_model(sparse: Path, runner: Runner) -> tuple[Path | None, dict[str, Any]]:
    """The sub-model with the most registered images, with its stats."""
    best: tuple[Path | None, dict[str, Any]] = (None, {})
    models = (
        sorted(p for p in sparse.iterdir() if p.is_dir()) if sparse.is_dir() else []
    )
    for model in models:
        text_dir = sparse.parent / "txt" / sparse.name / model.name
        if convert_to_text(model, text_dir, runner)["returncode"] != 0:
            continue
        stats = text_model_stats(text_dir)
        if stats["registered"] > (best[1].get("registered") or 0):
            best = (model, stats)
    return best


def camera_centers(images: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    from sceneagent.bin_association import qvec_to_rotation

    return {
        name: -qvec_to_rotation(img["qvec"]).T @ np.asarray(img["tvec"])
        for name, img in images.items()
    }


def pose_accuracy(
    images: dict[str, dict[str, Any]], scene_gt: dict[str, Any]
) -> dict[str, Any]:
    """Fit COLMAP's camera centers to the metric ones; report the fit's error in metres."""
    from sceneagent.bin_place import metric_camera_centers
    from sceneagent.similarity_icp import apply_similarity, umeyama

    metric = metric_camera_centers(scene_gt, sorted(images))
    names = sorted(set(images) & set(metric))
    centers = camera_centers(images)
    src = np.stack([centers[n] for n in names])
    dst = np.stack([metric[n] for n in names])
    scale, rotation, translation = umeyama(src, dst)
    error = np.linalg.norm(
        apply_similarity(src, scale, rotation, translation) - dst, axis=1
    )
    return {
        "cameras": len(names),
        "camera_rms_m": float(np.sqrt(np.mean(error**2))),
        "camera_max_m": float(error.max()),
    }


def run_variant(
    root: Path,
    dataset: str,
    matcher: str,
    mapper: str,
    runner: Runner,
    scene_gt: dict[str, Any] | None,
) -> dict[str, Any]:
    images = root / dataset / "images"
    base = root / dataset / "features.db"
    database = root / dataset / f"{matcher}.db"
    record: dict[str, Any] = {
        "dataset": dataset,
        "matcher": matcher,
        "mapper": mapper,
        "steps": [],
    }
    if not database.is_file():
        shutil.copyfile(base, database)
        step = run_step(match_command(database, matcher), runner)
        record["steps"].append(step)
        if step["returncode"] != 0:
            database.unlink(missing_ok=True)
            record["error"] = f"{matcher} exited {step['returncode']}"
            return record
    sparse = root / dataset / f"sparse_{matcher}_{mapper}"
    shutil.rmtree(sparse, ignore_errors=True)
    sparse.mkdir(parents=True)
    step = run_step(map_command(mapper, database, images, sparse), runner)
    record["steps"].append(step)
    record["map_seconds"] = step["seconds"]
    if step["returncode"] != 0:
        record["error"] = f"{mapper} exited {step['returncode']}"
        return record
    model, stats = largest_model(sparse, runner)
    record["models"] = len([p for p in sparse.iterdir() if p.is_dir()])
    if model is None:
        record["error"] = "no model registered an image"
        return record
    posed = stats.pop("images")
    record.update(stats)
    if scene_gt is not None:
        record["pose_accuracy"] = pose_accuracy(posed, scene_gt)
    return record


def run_mapper_compare(
    out: Path | str,
    data: Path | str = DATA_DIR,
    *,
    runner: Runner = subprocess.run,
    fruits_frames: Path | str = FRUITS_FRAMES,
) -> int:
    """Run every variant; return 0 when each one produced a model."""
    out, data = Path(out), Path(data)
    root = out / "mapper_compare"
    evidence: dict[str, Any] = {"colmap": COLMAP4, "variants": []}
    entry: dict[str, Any] = {"item": ITEM}
    try:
        counts = {
            "bin": stage_images(out / "images", "*.png", root / "bin" / "images"),
            "fruits": stage_images(
                Path(fruits_frames), "*.jpg", root / "fruits" / "images"
            ),
        }
        evidence["inputs"] = counts
        scene_gt = json.loads(
            (data / "scenes" / SCENE / "scene_gt.json").read_text(encoding="utf-8")
        )
        for dataset in counts:
            base = root / dataset / "features.db"
            for stale in (root / dataset).glob("*.db"):
                stale.unlink()
            step = run_step(extract_command(base, root / dataset / "images"), runner)
            evidence[f"{dataset}_extract"] = step
            if step["returncode"] != 0:
                raise RuntimeError(
                    f"{dataset} feature_extractor exited {step['returncode']}"
                )
        for dataset, matcher, mapper in VARIANTS:
            gt = scene_gt if dataset == "bin" else None
            evidence["variants"].append(
                run_variant(root, dataset, matcher, mapper, runner, gt)
            )
        failed = [v for v in evidence["variants"] if v.get("error")]
        if failed:
            first = failed[0]
            entry.update(
                decision="reject",
                reason=f"{len(failed)} variant(s) failed; first: {first['dataset']} "
                f"{first['matcher']} {first['mapper']}: {first['error']}",
            )
        else:
            entry.update(decision="accept", use_gpu=True, variants=len(VARIANTS))
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        entry.update(decision="reject", reason=f"{type(exc).__name__}: {exc}"[-800:])
    ledger_path = out / "ledger.json"
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        ledger = {"entries": []}
    entries = [e for e in ledger.get("entries", []) if e.get("item") != ITEM] + [entry]
    ledger.update(scene_name=f"{SOURCE} {SCENE}", source=SOURCE, entries=entries)
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    (out / "mapper_compare.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if entry["decision"] == "accept" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path(DATA_DIR))
    args = parser.parse_args(argv)
    return run_mapper_compare(args.out, args.data)


if __name__ == "__main__":
    sys.exit(main())
