# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cut 2 phase 2, SF-15: GroundingDINO boxes and SAM vit_h masks on all 13 bin views.

Text prompt -> GroundingDINO boxes -> SAM vit_h masks from those boxes, then
prompt-to-object filtering (a SimFoundry addition, KAN-4 comment 10304). Every
one of the 13 views is masked: the paper's every-10th stride is for video and
would leave two of them.

SAM is the paper's vit_h (plan decision 3). On instance 58301075 the box's
``gsam2`` env has GroundingDINO and torch, and runs
:mod:`sceneagent.mask_worker`; ``/workspace/models/sam`` holds vit_h. The vit_h
file is checked against its published SHA-256 before use. Only when a file is
missing does this module download it, and ``segment-anything`` is installed
into its own target directory when the env lacks it.

    python -m sceneagent.bin_masks --out /workspace/sf4-smoke/bin
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

from sceneagent.bin_run import (
    DATA_DIR,
    VIEWS,
    expected_views,
    is_bin,
    load_scene,
    run_bin_poses,
)
from sceneagent.mask_filter import (
    BOX_THRESHOLD,
    DUPLICATE_IOU,
    MAX_BOX_AREA_FRACTION,
    PROMPT,
    PROMPT_SOURCE,
    TEXT_THRESHOLD,
)

# The gsam2 env on 58301075 (probe run 42908214934): GroundingDINO 0.1.0 from
# Grounded-SAM-2, torch 2.14.1+cu130.
MASK_PYTHON = "/root/.local/share/mamba/envs/gsam2/bin/python3"
# Grounded-SAM-2's GroundingDINO imports itself as ``grounding_dino.*``, so its
# checkout root goes on the worker's path (masks run of 2026-11-09 on this box:
# "No module named 'grounding_dino'").
GSAM_ROOT = "/root/SceneagentX/third_party/Grounded-SAM-2"
GDINO_ROOT = f"{GSAM_ROOT}/grounding_dino"
GDINO_CONFIG = f"{GDINO_ROOT}/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GDINO_WEIGHTS = f"{GSAM_ROOT}/gdino_checkpoints/groundingdino_swint_ogc.pth"
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
SAM_WEIGHTS = "/workspace/models/sam/sam_vit_h_4b8939.pth"
# Published SHA-256 of sam_vit_h_4b8939.pth. The "4b8939" in the name is an MD5
# prefix, not a SHA-256 one (masks run 42695882028 read a7bf3b02f3eb...).
SAM_SHA256 = "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e"
SAM_PIP = "segment-anything==1.0"
MASK_PYDEPS = "/workspace/sf4-smoke/pydeps-masks"

WORKER_TIMEOUT_S = 1800
ITEM = "SAM masks"
# Scene 000091 holds 20 YCB objects (paper §4.1). The gate only asks for a mask
# in every view; the per-view counts beside this number show how far short of
# one mask per object a view falls.
GT_OBJECTS = 20

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
    """Check vit_h's SHA-256, downloading it first only if it is missing.

    A bad file this call downloaded is removed. A bad file that was already
    there belongs to someone else's checkout and is left alone.
    """
    fetched = False
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        (download or _http_download)(url, partial)
        partial.replace(path)
        fetched = True
    digest = sha256_of(path)
    if digest != SAM_SHA256:
        if fetched:
            path.unlink()
        raise ValueError(
            f"{path.name} SHA-256 {digest[:12]}... is not the published {SAM_SHA256[:12]}..."
        )
    return {"path": str(path), "sha256": digest, "downloaded": fetched, "url": url}


def _http_download(url: str, target: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as response, target.open("wb") as out:
        for block in iter(lambda: response.read(1 << 20), b""):
            out.write(block)


def worker_command(images: Path, masks: Path, prompt: str = PROMPT) -> list[str]:
    return [
        MASK_PYTHON,
        "-m",
        "sceneagent.mask_worker",
        "--images",
        str(images),
        "--out",
        str(masks),
        "--prompt",
        prompt,
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
    """This repo, Grounded-SAM-2's root and the segment-anything target dir.

    Not the sa env's py7zr deps.
    """
    env = dict(os.environ if base is None else base)
    repo = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = os.pathsep.join([repo, GSAM_ROOT, MASK_PYDEPS])
    return env


def _ensure_segment_anything(runner: Runner) -> dict[str, Any]:
    check = [MASK_PYTHON, "-c", "import segment_anything"]
    env = worker_environment()
    done = runner(check, env=env, capture_output=True, text=True, check=False)
    if done.returncode == 0:
        return {"installed": False}
    install = [
        MASK_PYTHON,
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
    kept = [v["kept"] for v in per_view.values()] or [0]
    return {
        "views": len(per_view),
        "views_with_masks": sum(1 for v in per_view.values() if v["kept"]),
        "kept_total": sum(kept),
        "kept_per_view_min": min(kept),
        "kept_per_view_max": max(kept),
        "gt_objects": GT_OBJECTS,
        "per_view": per_view,
    }


def _mask_shortfall(summary: dict[str, Any], expected: int, needed: int) -> str:
    reason = f"{summary['views_with_masks']} of {expected} views kept a mask"
    if needed != expected:
        reason += f" (need {needed})"
    if summary["views"] != expected:
        reason += f"; the worker saw {summary['views']} views"
    return reason


def run_bin_masks(
    out: Path | str,
    data: Path | str = DATA_DIR,
    *,
    poses: Callable[[Path, Path], int] = run_bin_poses,
    runner: Runner = subprocess.run,
    weights: Callable[[], dict[str, Any]] = ensure_sam_weights,
) -> int:
    """Mask every view. Return 0 when enough views keep at least one mask.

    The bin needs all 13; a video needs ``min_masked_views`` of its mask
    views (``scene.json``).
    """
    out, data = Path(out), Path(data)
    scene = load_scene(out)
    images = out / scene.view_dir
    if is_bin(out) and len(list(images.glob("*.png"))) != VIEWS:
        poses(out, data)
    expected = expected_views(out, scene)
    prompt = scene.prompt or PROMPT
    ledger_path = out / "ledger.json"
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        ledger = {"entries": []}
    if ledger.get("scene_name") != scene.ledger_name:
        ledger = {"entries": []}
    entries = [e for e in ledger.get("entries", []) if e.get("item") != ITEM]
    evidence: dict[str, Any] = {
        "scene": scene.name,
        "prompt": prompt,
        "prompt_source": PROMPT_SOURCE if scene.prompt is None else "scene.json (ours)",
        "views_dir": scene.view_dir,
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
    entry: dict[str, Any] = {"item": ITEM, "sam": "vit_h", "prompt": prompt}
    try:
        evidence["sam_weights"] = weights()
        evidence["segment_anything"] = _ensure_segment_anything(runner)
        masks = out / "masks"
        argv = worker_command(images, masks, prompt)
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
            # Failed runs publish only the manifest, which carries this reason.
            tail = evidence["worker"]["output_tail"].strip().splitlines()[-3:]
            raise RuntimeError(
                f"mask worker exited {done.returncode}: " + " | ".join(tail)
            )
        detections = json.loads((masks / "detections.json").read_text(encoding="utf-8"))
        summary = summarize(detections)
        evidence["summary"] = summary
        if (
            expected
            and summary["views"] == expected
            and summary["views_with_masks"] >= scene.min_masked_views
        ):
            entry.update(
                decision="accept",
                use_gpu=True,
                masks=summary["kept_total"],
                views=expected,
                views_with_masks=summary["views_with_masks"],
                kept_per_view=[
                    summary["kept_per_view_min"],
                    summary["kept_per_view_max"],
                ],
            )
            if scene.ground_truth:
                entry["gt_objects"] = GT_OBJECTS
        else:
            entry.update(
                decision="reject",
                reason=_mask_shortfall(summary, expected, scene.min_masked_views),
            )
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        entry.update(decision="reject", reason=f"{type(exc).__name__}: {exc}"[-800:])
    entries.append(entry)
    ledger.update(scene_name=scene.ledger_name, source=scene.source, entries=entries)
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
