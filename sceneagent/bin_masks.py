# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cut 2 phase 2, SF-15: GroundingDINO boxes and SAM vit_h masks on all 13 bin views.

Text prompt -> GroundingDINO boxes -> SAM vit_h masks from those boxes, then
prompt-to-object filtering (a SimFoundry addition, KAN-4 comment 10304). Every
one of the 13 views is masked: the paper's every-10th stride is for video and
would leave two of them.

SAM is the paper's vit_h (plan decision 3). Instance 58301075 has SAM 2 but no
vit_h, so this module installs ``segment-anything`` into its own target
directory and downloads the vit_h checkpoint into ``/workspace/models``,
checking the SHA-256 prefix in the file name. GroundingDINO and torch come
from the box's ``gsam2`` env, which runs :mod:`sceneagent.mask_worker`.

    python -m sceneagent.bin_masks --out /workspace/sf4-smoke/run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sceneagent.bin_run import DATA_DIR, SCENE, SOURCE, VIEWS, run_bin_poses
from sceneagent.mask_filter import (
    BOX_THRESHOLD,
    DUPLICATE_IOU,
    MAX_BOX_AREA_FRACTION,
    PROMPT,
    TEXT_THRESHOLD,
)

# The gsam2 env on 58301075 (probe run 42693162920): GroundingDINO 0.1.0 from
# Grounded-SAM-2, torch 2.14.1+cu130.
GSAM2_PYTHON = "/root/.local/share/mamba/envs/gsam2/bin/python3"
GSAM_ROOT = "/root/SceneagentX/third_party/Grounded-SAM-2"
GDINO_CONFIG = (
    f"{GSAM_ROOT}/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
GDINO_WEIGHTS = f"{GSAM_ROOT}/gdino_checkpoints/groundingdino_swint_ogc.pth"
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
SAM_WEIGHTS = "/workspace/models/sam/sam_vit_h_4b8939.pth"
# torch.hub convention: the file name ends in the first hex digits of its SHA-256.
SAM_SHA256_PREFIX = "4b8939"
SAM_PIP = "segment-anything==1.0"
MASK_PYDEPS = "/workspace/sf4-smoke/pydeps-gsam2"

WORKER_TIMEOUT_S = 1800
ITEM = "SAM masks"

Runner = Callable[..., Any]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_sam_weights(
    path: Path = Path(SAM_WEIGHTS),
    url: str = SAM_URL,
    *,
    download: Callable[[str, Path], None] | None = None,
) -> dict[str, Any]:
    """Download vit_h once and check its hash prefix. A bad file is removed."""
    fetched = False
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        (download or _http_download)(url, partial)
        partial.replace(path)
        fetched = True
    digest = sha256_of(path)
    if not digest.startswith(SAM_SHA256_PREFIX):
        path.unlink()
        raise ValueError(
            f"{path.name} SHA-256 {digest[:12]} does not start {SAM_SHA256_PREFIX}"
        )
    return {"path": str(path), "sha256": digest, "downloaded": fetched, "url": url}


def _http_download(url: str, target: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as response, target.open("wb") as out:
        for block in iter(lambda: response.read(1 << 20), b""):
            out.write(block)


def worker_command(images: Path, masks: Path) -> list[str]:
    return [
        GSAM2_PYTHON,
        "-m",
        "sceneagent.mask_worker",
        "--images",
        str(images),
        "--out",
        str(masks),
        "--prompt",
        PROMPT,
        "--box-threshold",
        str(BOX_THRESHOLD),
        "--text-threshold",
        str(TEXT_THRESHOLD),
        "--gdino-config",
        GDINO_CONFIG,
        "--gdino-weights",
        GDINO_WEIGHTS,
        "--sam-weights",
        SAM_WEIGHTS,
    ]


def worker_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    """This repo and the segment-anything target dir; not the sa env's py7zr deps."""
    env = dict(os.environ if base is None else base)
    repo = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = os.pathsep.join([repo, MASK_PYDEPS])
    return env


def _ensure_segment_anything(runner: Runner) -> dict[str, Any]:
    check = [GSAM2_PYTHON, "-c", "import segment_anything"]
    env = worker_environment()
    done = runner(check, env=env, capture_output=True, text=True, check=False)
    if done.returncode == 0:
        return {"installed": False}
    install = [
        GSAM2_PYTHON,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--target",
        MASK_PYDEPS,
        SAM_PIP,
    ]
    done = runner(
        install, env=env, capture_output=True, text=True, check=False, timeout=600
    )
    if done.returncode != 0:
        raise RuntimeError(
            f"pip install {SAM_PIP} failed: {(done.stderr or '')[-500:]}"
        )
    return {"installed": True, "spec": SAM_PIP, "target": MASK_PYDEPS}


def summarize(detections_by_view: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    per_view = {
        view: {
            "detections": len(dets),
            "kept": sum(1 for d in dets if d.get("kept")),
        }
        for view, dets in sorted(detections_by_view.items())
    }
    return {
        "views": len(per_view),
        "views_with_masks": sum(1 for v in per_view.values() if v["kept"]),
        "kept_total": sum(v["kept"] for v in per_view.values()),
        "per_view": per_view,
    }


def run_bin_masks(
    out: Path | str,
    data: Path | str = DATA_DIR,
    *,
    poses: Callable[[Path, Path], int] = run_bin_poses,
    runner: Runner = subprocess.run,
    weights: Callable[[], dict[str, Any]] = ensure_sam_weights,
) -> int:
    """Mask every bin view. Return 0 only when all 13 views keep at least one mask."""
    out, data = Path(out), Path(data)
    images = out / "images"
    if len(list(images.glob("*.png"))) != VIEWS:
        poses(out, data)
    ledger_path = out / "ledger.json"
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        ledger = {"entries": []}
    if ledger.get("scene_name") != f"{SOURCE} {SCENE}":
        ledger = {"entries": []}
    entries = [e for e in ledger.get("entries", []) if e.get("item") != ITEM]
    evidence: dict[str, Any] = {
        "scene": SCENE,
        "prompt": PROMPT,
        "box_threshold": BOX_THRESHOLD,
        "text_threshold": TEXT_THRESHOLD,
        "filter": {
            "max_box_area_fraction": MAX_BOX_AREA_FRACTION,
            "duplicate_iou": DUPLICATE_IOU,
            "phrase_must_match_prompt": True,
        },
        "sam": "vit_h",
        "detector": "GroundingDINO SwinT-OGC",
    }
    entry: dict[str, Any] = {"item": ITEM, "sam": "vit_h", "prompt": PROMPT}
    try:
        evidence["sam_weights"] = weights()
        evidence["segment_anything"] = _ensure_segment_anything(runner)
        masks = out / "masks"
        argv = worker_command(images, masks)
        done = runner(
            argv,
            env=worker_environment(),
            capture_output=True,
            text=True,
            check=False,
            timeout=WORKER_TIMEOUT_S,
        )
        evidence["worker"] = {
            "argv": argv,
            "returncode": done.returncode,
            "output_tail": ((done.stderr or "") or (done.stdout or ""))[-2000:],
        }
        if done.returncode != 0:
            raise RuntimeError(f"mask worker exited {done.returncode}")
        detections = json.loads((masks / "detections.json").read_text(encoding="utf-8"))
        summary = summarize(detections)
        evidence["summary"] = summary
        if summary["views"] == VIEWS and summary["views_with_masks"] == VIEWS:
            entry.update(decision="accept", use_gpu=True, masks=summary["kept_total"])
        else:
            entry.update(
                decision="reject",
                reason=f"{summary['views_with_masks']} of {VIEWS} views kept a mask",
            )
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        entry.update(decision="reject", reason=f"{type(exc).__name__}: {exc}"[:500])
    entries.append(entry)
    ledger.update(scene_name=f"{SOURCE} {SCENE}", source=SOURCE, entries=entries)
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    (out / "masks_evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if entry["decision"] == "accept" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path(DATA_DIR))
    args = parser.parse_args(argv)
    return run_bin_masks(args.out, args.data)


if __name__ == "__main__":
    sys.exit(main())
