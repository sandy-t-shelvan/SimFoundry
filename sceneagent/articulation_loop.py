# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KAN-44/45: propose a joint, check it, revise it -- an actor-critic articulation loop.

    python -m sceneagent.articulation_loop --parts PARTS_DIR --run VIDEO_RUN \
        --object N --out OUT [--name NAME] [--max-revisions 2]

Paper §3.1: a reviewer "proposes a corrected pose or fix and reviews again";
§3.4: agents review the articulation in simulation. Per round:

1. **Actor** -- ``hinge_proposal`` (Gemini) proposes the joint from the parts and
   the object's own video frames; from round 1 on, with the last round's
   rejection as its note.
2. **Simulation** -- ``articulate_scanned`` builds the links (with the scan's
   colours) and ``articulate_scanned_drive`` sweeps the joint in Isaac Lab and
   renders the part held open from three sides.
3. **Deterministic QA** -- ``joint_qa``: physics, the flush-side rule, pivot on
   an edge. Specific reasons, no model.
4. **Model critic** -- only once 3 passes: ``vlm_check.advise`` compares the real
   frames with the held-open renders.

Accept when 3 and 4 pass. After ``max_revisions`` rejected revisions the
object **stays rigid** (fail closed) and the record says why. Nothing here is
specific to one object: the frames are the object's own largest masks, the
QA abstains when its evidence is unclear, and one movable joint per object is
handled (a second movable joint is ignored, and the record says so).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

LOOP = "articulation_loop.json"
MAX_REVISIONS = 2
MAX_CALL_FAILURES = 2  # model calls that never answered; these do not use up revisions
FRAMES = 3
MIN_FRAME_GAP = 30  # video frames between chosen views, so they differ
CROP_MARGIN = 0.3
ISAACLAB_PYTHON = Path(
    os.environ.get(
        "ISAACLAB_PYTHON", "/root/.local/share/mamba/envs/isaaclab/bin/python3"
    )
)
GPU_LOCK = Path("/workspace/runs/.gpu.lock")
CRITIC_INTRO = (
    "The first {n_real} images are frames of a real {name} from a video. The last "
    "{n_sim} images are a simulation of the same scanned object with its moving part "
    "{pose} about a proposed joint, seen from three sides; parts of the scan the "
    "camera never saw may be invented (for example, extra labels). "
)
CRITIC_ASK = {
    "revolute": (
        "Does the part open about the edge it would open about on the real object? "
        "Judge by the real object's shape: where the lid or door overhangs the body and "
        "where it is flush, and how the real part is built to move. Answer no if the "
        "hinge is on the wrong edge or the motion is impossible for the real object, "
        "and say which edge is right."
    ),
    "prismatic": (
        "Does the part slide out the way it would on the real object (a cap pulled off, "
        "a drawer pulled out), along the right direction? Answer no if it slides along "
        "the wrong direction, the wrong part moves, or the real part does not slide, "
        "and say what is right."
    ),
}
POSE = {"revolute": "held open", "prismatic": "slid out"}


def critic_question(joint_type: str, name: str, n_real: int, n_sim: int) -> str:
    """The critic's question for this kind of joint."""
    intro = CRITIC_INTRO.format(
        n_real=n_real, n_sim=n_sim, name=name, pose=POSE[joint_type]
    )
    return intro + CRITIC_ASK[joint_type]


def object_frames(run: Path, object_number: int, count: int = FRAMES) -> list[bytes]:
    """JPEG crops of the views where this object's mask is largest, spread over the video."""
    from PIL import Image

    evidence = json.loads(
        (run / "association_evidence.json").read_text(encoding="utf-8")
    )
    record = next(
        o for o in evidence["object_list"] if int(o["object"]) == object_number
    )
    scored = []
    for mask in record["masks"]:
        stem = Path(mask["view"]).stem
        stack = np.load(run / "masks" / f"{stem}.npz")["masks"]
        m = stack[int(mask["mask_index"])]
        scored.append((int(m.sum()), int(stem), mask["view"], m))
    chosen: list[tuple[int, int, str, np.ndarray]] = []
    for item in sorted(scored, key=lambda s: -s[0]):
        if all(abs(item[1] - c[1]) >= MIN_FRAME_GAP for c in chosen):
            chosen.append(item)
        if len(chosen) == count:
            break
    crops = []
    for _, _, view, m in sorted(chosen, key=lambda c: c[1]):
        ys, xs = np.nonzero(m)
        h, w = m.shape
        dy, dx = (
            (ys.max() - ys.min()) * CROP_MARGIN,
            (xs.max() - xs.min()) * CROP_MARGIN,
        )
        box = (
            max(0, int(xs.min() - dx)),
            max(0, int(ys.min() - dy)),
            min(w, int(xs.max() + dx)),
            min(h, int(ys.max() + dy)),
        )
        image = Image.open(run / "images" / view).convert("RGB").crop(box)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        crops.append(buffer.getvalue())
    return crops


def isaac_drive(links: Path, out: Path) -> dict[str, Any]:
    """Sweep and render in Isaac Lab, one GPU job at a time."""
    script = Path(__file__).resolve().parent / "articulate_scanned_drive.py"
    command = [
        str(ISAACLAB_PYTHON),
        str(script),
        "--links",
        str(links),
        "--out",
        str(out),
    ]
    if GPU_LOCK.parent.is_dir():
        command = ["flock", str(GPU_LOCK), *command]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    out.mkdir(parents=True, exist_ok=True)
    with (out / "drive.log").open("w") as log:
        subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, env=env, check=False
        )
    result = out / "physics_result.json"
    if not result.is_file():
        return {
            "decision": "reject",
            "reason": f"the drive wrote no result; see {out / 'drive.log'}",
        }
    return json.loads(result.read_text(encoding="utf-8"))


def run_loop(
    parts_dir: Path,
    run: Path,
    object_number: int,
    out: Path,
    client: Any,
    name: str | None = None,
    max_revisions: int = MAX_REVISIONS,
    drive: Callable[[Path, Path], dict[str, Any]] = isaac_drive,
    frames: list[bytes] | None = None,
) -> dict[str, Any]:
    from sceneagent import articulate_scanned
    from sceneagent.hinge_proposal import propose_articulation
    from sceneagent.joint_qa import review
    from sceneagent.vlm_check import advise

    manifest = json.loads((parts_dir / "parts.json").read_text(encoding="utf-8"))
    parts = manifest["parts"]
    name = name or manifest.get("parent_object") or f"object_{object_number}"
    real = frames if frames is not None else object_frames(run, object_number)
    out.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "object": name,
        "run": str(run),
        "parts": str(parts_dir),
        "real_frames": len(real),
        "rounds": [],
        "vlm_calls": 0,
    }
    notes = None
    rejections = failures = 0
    k = 0
    while True:
        folder = out / f"round_{k}"
        folder.mkdir(parents=True, exist_ok=True)
        proposal = propose_articulation(
            [{"name": name, "parts": parts}], real, client, notes=notes
        )
        attempts = proposal["objects"][0].get("attempts") or []
        record["vlm_calls"] += len(attempts)
        (folder / "articulate_proposal.json").write_text(
            json.dumps(proposal, indent=2) + "\n"
        )
        entry: dict[str, Any] = {
            "round": k,
            "notes_in": notes,
            "proposal": str(folder / "articulate_proposal.json"),
        }
        record["rounds"].append(entry)
        k += 1
        status = proposal["objects"][0]["status"]
        movable = [
            j
            for j in proposal["objects"][0]["joints"]
            if j["type"] in ("revolute", "prismatic")
        ]
        rejected = None
        if status != "ok":
            if not any(a.get("model_reply") for a in attempts):
                # The model never answered (timeout, HTTP error): not its
                # mistake, so it does not use up a revision.
                failures += 1
                entry["decision"] = (
                    f"model call failed: {attempts[-1].get('error') if attempts else status}"
                )
                if failures > MAX_CALL_FAILURES:
                    record.update(
                        decision="rigid", reason="the model did not answer; kept rigid"
                    )
                    break
                continue
            rejected = "your reply could not be used; propose the joint again"
            entry["decision"] = f"proposal {status}"
        elif not movable:
            entry["decision"] = "proposed rigid"
            record.update(
                decision="rigid", reason="the proposer found no movable joint"
            )
            break
        else:
            if len(movable) > 1:
                entry["ignored_joints"] = [j.get("name") for j in movable[1:]]
            joint = movable[0]
            links = folder / "links"
            prepared = articulate_scanned.prepare(
                parts_dir,
                folder / "articulate_proposal.json",
                run,
                object_number,
                links,
                name=name,
            )
            physics = drive(links, folder / "physics")
            by_name = {str(p["name"]): p for p in parts}
            qa = review(
                joint,
                [by_name[n] for n in prepared["body_parts"]],
                [by_name[n] for n in prepared["lid_parts"]],
                float(prepared["scale_m_per_unit"]),
                physics,
            )
            entry.update(joint=joint, physics=physics.get("reason"), qa=qa)
            if not qa["accept"]:
                entry["decision"] = "rejected by QA"
                rejected = "; ".join(qa["problems"])
            else:
                views = [Path(v).read_bytes() for v in physics.get("open_views") or []]
                verdict = advise(
                    real + views,
                    critic_question(joint["type"], name, len(real), len(views)),
                    client,
                )
                record["vlm_calls"] += 1
                entry["critic"] = {
                    k2: verdict.get(k2)
                    for k2 in ("verdict", "reason", "model_id", "call_failed")
                }
                if verdict.get("call_failed"):
                    failures += 1
                    entry["decision"] = (
                        f"critic call failed: {verdict.get('call_failed')}"
                    )
                    if failures > MAX_CALL_FAILURES:
                        record.update(
                            decision="rigid",
                            reason="the critic did not answer; kept rigid",
                        )
                        break
                    continue
                if verdict.get("verdict") == "yes":
                    entry["decision"] = "accepted"
                    record.update(
                        decision="articulated",
                        joint=joint,
                        links=str(links),
                        round=k - 1,
                    )
                    break
                entry["decision"] = "rejected by critic"
                rejected = (
                    "a reviewer compared the simulation with the real object and said: "
                    f"{verdict.get('reason')}"
                )
        rejections += 1
        notes = rejected
        if rejections > max_revisions:
            record.update(
                decision="rigid",
                reason=f"{rejections} proposals rejected (QA, critic or unusable); kept rigid",
            )
            break
    (out / LOOP).write_text(
        json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--parts", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--object", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--max-revisions", type=int, default=MAX_REVISIONS)
    args = parser.parse_args(argv)
    from sceneagent.gemini_client import client_from_environ

    record = run_loop(
        args.parts,
        args.run,
        args.object,
        args.out,
        client_from_environ(),
        name=args.name,
        max_revisions=args.max_revisions,
    )
    print(
        json.dumps(
            {
                k: record.get(k)
                for k in ("object", "decision", "round", "reason", "vlm_calls")
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
