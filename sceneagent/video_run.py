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
from sceneagent.bin_association import SAME_PHRASE, VIDEO_KAPPA_SHARE
from sceneagent.bin_masks import JOINED_PHRASES
from sceneagent.bin_run import is_bin, load_scene, run_video_poses
from sceneagent.scene_objects import MissingPromptError, resolve_prompt
from sceneagent.settle_run import VARIANTS

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
    # Fitted splat packs for the prior (KAN-39). Option B remains when the
    # packs or the pinned checkpoint are absent.
    ("prior_input", ("fitted splat prior input",)),
    # §3.4 record per placed object (KAN-40).
    ("physics", ("§3.4 properties",)),
    # KAN-41: the placed scene as USD (gravity aligned on the support plane),
    # then 2 s in Isaac Lab.
    ("export", ("USD physics export",)),
    ("settle", ("Isaac Lab settle",)),
    # KAN-23: the inspector's views of the settled scene; the fix report is
    # written on the runner (workflow sceneagent-inspector-report).
    ("inspect", ("inspection views",)),
)


# Ledger items a stage writes beside the ones its resume check reads.
ALSO_WRITES: dict[str, tuple[str, ...]] = {"place": ("bin score",)}

# KAN-52: the step critic's stage for each pipeline stage whose output it can
# draw (``step_critic.RENDERERS``). With ``--critic`` each is judged right
# after it runs, before anything is built on it.
CRITIC_STAGE: dict[str, str] = {
    "masks": "masks",
    "association": "gaussians",
    "sam3d": "mesh",
}
Critic = Callable[[Path, str], dict[str, Any]]


def review_item(critic_stage: str) -> str:
    """The ledger item the step critic writes for one of its stages."""
    return f"step review ({critic_stage})"


def last_review(out: Path, critic_stage: str) -> str | None:
    """The critic's last verdict on a stage; None when unjudged or stale."""
    try:
        ledger = json.loads((out / "ledger.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    last = None
    for entry in ledger.get("entries", []):
        if entry.get("item") == review_item(critic_stage):
            last = entry
    if last is None or "stale_after" in last:
        return None
    return last.get("decision")


def critic_rejects(out: Path) -> list[str]:
    """Pipeline stages whose last current critic verdict is a reject."""
    return [
        name
        for name, critic_stage in CRITIC_STAGE.items()
        if last_review(out, critic_stage) == "reject"
    ]


def stale_after(
    entries: list[dict[str, Any]], stage: str, reason: str
) -> list[dict[str, Any]]:
    """Reject lines for every later stage whose last ledger line is an accept.

    A stage re-run outside this loop (the step critic's mask retry) replaces
    its own output, while each later stage keeps an accept that describes the
    old one, and a resume would skip it. These lines make the resume rebuild
    from the next stage on. Earlier stages are not touched.
    """
    names = [name for name, _ in STAGES]
    if stage not in names:
        raise ValueError(f"unknown stage {stage!r}")
    later = [
        item
        for name, items in STAGES[names.index(stage) + 1 :]
        for item in (
            *items,
            *ALSO_WRITES.get(name, ()),
            *((review_item(CRITIC_STAGE[name]),) if name in CRITIC_STAGE else ()),
        )
    ]
    last = {e.get("item"): e.get("decision") for e in entries}
    return [
        {"item": item, "decision": "reject", "stale_after": stage, "reason": reason}
        for item in later
        if last.get(item) == "accept"
    ]


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
        # Association made before the current video rules runs again:
        evidence = _evidence(out, "association_evidence.json")
        grouping = evidence.get("rule", {}).get("grouping", {})
        # ... before a video's consensus needed half of an object's views,
        # and while phrases still split groups (run 43331599718).
        share = evidence.get("rule", {}).get("video_kappa_share")
        return grouping.get(SAME_PHRASE) is False and share == VIDEO_KAPPA_SHARE
    if name == "place":
        # Placements judged by an earlier gate are judged again.
        icp = _evidence(out, "bin_score.json").get("icp", {})
        return icp.get("gate") == PLACEMENT_GATE_LABEL
    if name == "inspect":
        # Views drawn before the summary named what the export left out.
        return "left_out" in _evidence(out, "settle_summary.json")
    if name == "export":
        # An export made before it recorded each object's box runs again, so
        # a settle reject can say what the bodies were doing.
        # ... and before it wrote the settle's diagnostic copies.
        evidence = _evidence(out, "export_evidence.json")
        return "object_boxes" in evidence and evidence.get("settle_variants") == list(
            VARIANTS
        )
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
    from sceneagent.fitted_prior import run_fitted_prior
    from sceneagent.inspect_run import run_inspect
    from sceneagent.physics_export import export_run
    from sceneagent.physics_props import run_physics
    from sceneagent.settle_run import run_settle

    return {
        "poses": lambda: run_video_poses(video, out, prompt),
        "splat": lambda: run_bin_splat(out),
        "masks": lambda: run_bin_masks(out),
        "association": lambda: run_bin_association(out),
        "sam3d": lambda: run_bin_sam3d(out),
        "place": lambda: run_bin_place(out),
        "prior_input": lambda: run_fitted_prior(out),
        "physics": lambda: run_physics(out),
        "export": lambda: export_run(out),
        "settle": lambda: run_settle(out),
        "inspect": lambda: run_inspect(out),
    }


def run_video(
    video: Path | str,
    out: Path | str,
    prompt: str,
    *,
    stages: dict[str, Callable[[], int]] | None = None,
    budget_s: float = START_BUDGET_S,
    clock: Callable[[], float] = time.monotonic,
    critic: Critic | None = None,
) -> int:
    """Run or resume the slice. Return 0 when every stage is accepted.

    With ``critic`` (KAN-52) each stage in ``CRITIC_STAGE`` is judged after
    it runs, and a reject is corrected where a note can change the stage's
    input (masks: the detector prompt; the retry also marks every later
    stage stale). A reject that stands stops the run there, so nothing is
    built on output the critic refused. A stage resumed as done is judged
    if it never was, and stops the run if its last verdict is a reject. A
    run is complete only with every stage accepted and no current critic
    reject, whether or not this call ran the critic.
    """
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
                if critic is not None and name in CRITIC_STAGE:
                    verdict = last_review(out, CRITIC_STAGE[name])
                    if verdict == "reject":
                        record["stopped"] = f"{name}: the step critic rejected it"
                        break
                    if verdict is None:
                        stop, retried = _judge(out, name, critic, record)
                        _write_record(out, record)
                        if stop:
                            record["stopped"] = stop
                            break
                        rerun = rerun or retried
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
            if critic is not None and name in CRITIC_STAGE:
                stop, _ = _judge(out, name, critic, record)
                _write_record(out, record)
                if stop:
                    record["stopped"] = stop
                    break
        try:
            from sceneagent.step_critic import render_review_images

            render_review_images(out)
        except (OSError, ValueError, ImportError):
            pass
        try:
            from sceneagent.packager import build_package

            build_package(out, out / "package")
        except (OSError, ValueError):
            pass
        rejected = critic_rejects(out)
        done = all(accepted(out, items) for _, items in STAGES) and not rejected
        record["critic_rejects"] = rejected
        record["complete"] = done
        _write_record(out, record)
    return 0 if done else 1


def _judge(
    out: Path, name: str, critic: Critic, record: dict[str, Any]
) -> tuple[str | None, bool]:
    """Judge one stage. Return why the run stops (None to go on) and whether it re-ran.

    Nothing drawn means there is nothing to judge; the stage's own gate has
    already decided it. Any other outcome but accept stops the run, a failed
    critic call included: the loop does not pass a stage it could not check.
    """
    result = critic(out, CRITIC_STAGE[name])
    outcome = str(result.get("outcome"))
    attempts = result.get("attempts") or []
    retried = any("exit_code" in (step.get("retry") or {}) for step in attempts)
    record.setdefault("critic", []).append(
        {
            "stage": name,
            "critic_stage": CRITIC_STAGE[name],
            "outcome": outcome,
            "attempts": len(attempts),
            "retried": retried,
        }
    )
    if outcome in ("accept", "nothing to review"):
        return None, retried
    return f"{name}: the step critic's outcome is {outcome!r}", retried


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
    parser.add_argument(
        "--prompt",
        default=None,
        help=(
            "GroundingDINO phrases ('a . b .'). Default: the committed verdict "
            "for this video's bytes (sceneagent/scene_objects.py, KAN-56)."
        ),
    )
    parser.add_argument(
        "--critic",
        action="store_true",
        help=(
            "judge masks, Gaussians and meshes as they are made, correct a "
            "rejected one where a note can, and stop on a reject that stands "
            "(KAN-52; needs the VLM key)"
        ),
    )
    args = parser.parse_args(argv)
    try:
        prompt = resolve_prompt(args.video, args.prompt)
    except MissingPromptError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    critic: Critic | None = None
    if args.critic:
        from sceneagent.harness import selected_harness
        from sceneagent.step_critic import review_and_retry

        harness = selected_harness()

        def critic(out: Path, stage: str) -> dict[str, Any]:
            return review_and_retry(out, harness, stage)

    return run_video(args.video, args.out, prompt, critic=critic)


if __name__ == "__main__":
    sys.exit(main())
