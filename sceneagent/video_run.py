# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SF-42 (KAN-48) and SF-35 (KAN-41): one video through the bin chain, then export and settle.

    python -m sceneagent.video_run docs/assets/example_videos/Mailbox.mp4 \
        --out /workspace/sf4-smoke/mailbox

The stages are the bin's own modules, in the bin's order: poses (run4's 300
sharpest-per-bin frames and COLMAP), splat, masks (every 10th posed frame),
association, SAM 3D, ICP place. The pose stage writes ``scene.json``, which
tells the later stages this is a video with no ground truth, so placement
skips the score. Nothing here is a new algorithm.

One gpu-smoke run may not hold every stage, so a run resumes: a stage whose
ledger items are already accepted is skipped, every stage after one that ran
runs again, and no new stage starts past ``START_BUDGET_S``. Seconds per
stage go to ``video_run.json``.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sceneagent.alignment import PLACEMENT_GATE_LABEL
from sceneagent.bin_association import SAME_PHRASE
from sceneagent.bin_masks import JOINED_PHRASES
from sceneagent.bin_run import is_bin, load_scene, run_video_poses

# Ours: GroundingDINO phrases for the objects Mailbox.mp4 shows (a red
# mailbox with a lid, a cyan cube). The orchestrator plan (PR #63) that
# would write this per video is after the MVP.
PROMPTS = {"Mailbox": "mailbox . cube ."}
# No new stage starts after this many seconds, so a long stage (the splat's
# train step may take up to an hour) still ends inside the job's 6,600 s.
START_BUDGET_S = 2400.0
RECORD = "video_run.json"
LOCK = ".video_run.lock"

# (stage, the ledger items it writes)
STAGES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("poses", ("video", "COLMAP camera poses")),
    ("splat", ("full-scene splat",)),
    ("masks", ("SAM masks",)),
    ("association", ("Gaussian association",)),
    ("sam3d", ("SAM 3D meshes",)),
    ("place", ("ICP residuals",)),
    # KAN-41: the placed scene as USD (gravity aligned on the support plane),
    # then 2 s in Isaac Lab.
    ("export", ("USD physics export",)),
    ("settle", ("Isaac Lab settle",)),
)


def accepted(out: Path, items: tuple[str, ...]) -> bool:
    """Whether the ledger in ``out`` holds an accept for every item."""
    try:
        ledger = json.loads((out / "ledger.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    decisions = {e.get("item"): e.get("decision") for e in ledger.get("entries", [])}
    return all(decisions.get(item) == "accept" for item in items)


def stage_done(out: Path, name: str, items: tuple[str, ...]) -> bool:
    """Accepted in the ledger; the pose stage also needs its finished scene.json.

    run4 accepts the poses before the mask views are staged, so a run that
    stopped between the two must pose again rather than skip. The mask,
    association and export stages also need evidence written by their
    current rules.
    """
    if not accepted(out, items):
        return False
    if name == "poses":
        return not is_bin(out) and load_scene(out).min_masked_views > 0
    if name == "masks":
        # Masks made before joined phrases were dropped run again, and so
        # does every stage after them.
        evidence = _evidence(out, "masks_evidence.json")
        return evidence.get("filter", {}).get(JOINED_PHRASES) == "drop"
    if name == "association":
        # Association made before differing phrases stopped merging.
        grouping = _evidence(out, "association_evidence.json").get("rule", {})
        return grouping.get("grouping", {}).get(SAME_PHRASE) is True
    if name == "place":
        # Placements judged by an earlier gate are judged again.
        icp = _evidence(out, "bin_score.json").get("icp", {})
        return icp.get("gate") == PLACEMENT_GATE_LABEL
    if name == "export":
        # An export made before it recorded each object's box runs again, so
        # a settle reject can say what the bodies were doing.
        return "object_boxes" in _evidence(out, "export_evidence.json")
    return True


def _evidence(out: Path, name: str) -> dict[str, Any]:
    try:
        evidence = json.loads((out / name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return evidence if isinstance(evidence, dict) else {}


def default_stages(video: Path, out: Path, prompt: str) -> dict[str, Callable[[], int]]:
    from sceneagent.bin_association import run_bin_association
    from sceneagent.bin_masks import run_bin_masks
    from sceneagent.bin_place import run_bin_place
    from sceneagent.bin_sam3d import run_bin_sam3d
    from sceneagent.bin_splat import run_bin_splat
    from sceneagent.physics_export import export_run
    from sceneagent.settle_run import run_settle

    return {
        "poses": lambda: run_video_poses(video, out, prompt),
        "splat": lambda: run_bin_splat(out),
        "masks": lambda: run_bin_masks(out),
        "association": lambda: run_bin_association(out),
        "sam3d": lambda: run_bin_sam3d(out),
        "place": lambda: run_bin_place(out),
        "export": lambda: export_run(out),
        "settle": lambda: run_settle(out),
    }


def run_video(
    video: Path | str,
    out: Path | str,
    prompt: str,
    *,
    stages: dict[str, Callable[[], int]] | None = None,
    budget_s: float = START_BUDGET_S,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Run or resume the slice. Return 0 when every stage is accepted."""
    video, out = Path(video), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    funcs = stages or default_stages(video, out, prompt)
    with open(out / LOCK, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print(f"another video run holds {out / LOCK}", file=sys.stderr)
            return 1
        record = _read_record(out)
        record.update(video=video.name, prompt=prompt, stopped=None)
        start = clock()
        rerun = False
        for name, items in STAGES:
            if name != "poses" and is_bin(out):
                # Without scene.json a stage would treat this as the bin's
                # directory and could restage the bin's photos over it.
                record["stopped"] = f"no video scene.json in {out} before {name}"
                break
            if not rerun and stage_done(out, name, items):
                continue
            if clock() - start > budget_s:
                record["stopped"] = f"time budget: {name} not started"
                break
            began = clock()
            code = funcs[name]()
            record["stages"][name] = {
                "code": code,
                "seconds": round(clock() - began, 1),
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            rerun = True
            _write_record(out, record)
            if code != 0:
                record["stopped"] = f"{name} failed (exit {code})"
                break
        done = all(accepted(out, items) for _, items in STAGES)
        record["complete"] = done
        _write_record(out, record)
    return 0 if done else 1


def _read_record(out: Path) -> dict[str, Any]:
    try:
        record = json.loads((out / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        record = {}
    record.setdefault("stages", {})
    return record


def _write_record(out: Path, record: dict[str, Any]) -> None:
    (out / RECORD).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompt", default=None)
    args = parser.parse_args(argv)
    prompt = args.prompt or PROMPTS.get(args.video.stem)
    if not prompt:
        print(f"no prompt for {args.video.stem}; pass --prompt", file=sys.stderr)
        return 2
    return run_video(args.video, args.out, prompt)


if __name__ == "__main__":
    sys.exit(main())
