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
CRITIC_VOTES = (
    3  # a critic "no" on a QA-passed proposal is decided by majority of this many
)
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
    history: list[str] = []
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
                question = critic_question(joint["type"], name, len(real), len(views))
                verdict = advise(real + views, question, client)
                record["vlm_calls"] += 1
                votes = [verdict]
                # The critic is noisy: the same Mailbox hinge got three yeses and
                # one "no" on separate runs. A "no" on a proposal QA has passed
                # gets two more independent looks, and the majority decides.
                if verdict.get("verdict") == "no" and not verdict.get("call_failed"):
                    majority = CRITIC_VOTES // 2 + 1
                    for _ in range(CRITIC_VOTES - 1):
                        if (
                            max(
                                sum(v.get("verdict") == w for v in votes)
                                for w in ("yes", "no")
                            )
                            >= majority
                        ):
                            break  # decided; a further look cannot change it
                        again = advise(real + views, question, client)
                        record["vlm_calls"] += 1
                        if not again.get("call_failed"):
                            votes.append(again)
                    yes = sum(v.get("verdict") == "yes" for v in votes)
                    if yes * 2 > len(votes):
                        verdict = next(v for v in votes if v.get("verdict") == "yes")
                entry["critic"] = {
                    k2: verdict.get(k2)
                    for k2 in ("verdict", "reason", "model_id", "call_failed")
                }
                entry["critic"]["votes"] = [v.get("verdict") for v in votes]
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
        # Every rejection so far, not only the last: with only the last one the
        # proposer went back to an edge QA had already rejected.
        history.append(f"round {k - 1}: {rejected}")
        notes = "Earlier proposals were rejected. " + " | ".join(history)
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


RESULT = "articulation.json"
REJECTED = ("rejected by QA", "rejected by critic")


def articulate_object(
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
    """The loop on PartSAM's parts; then, only if every proposed joint was rejected, on a colour cut.

    A run where the proposer kept proposing a joint and every one was
    rejected means it believes something moves but no joint over these parts
    can be right -- PartSAM cut the moving part in two (the marker's cap).
    ``colour_parts`` then offers a second part set: the object split where its
    colour changes most sharply. Colour differences are common (shading,
    print), so the cut is never used on its own: the same proposer, QA and
    critic must pass, and an object the proposer calls rigid never gets here.
    """
    from sceneagent.colour_parts import split

    real = frames if frames is not None else object_frames(run, object_number)
    first = run_loop(
        parts_dir,
        run,
        object_number,
        out / "partsam",
        client,
        name=name,
        max_revisions=max_revisions,
        drive=drive,
        frames=real,
    )
    result: dict[str, Any] = {
        "object": first["object"],
        "partsam": first,
        "colour_cut": None,
    }
    proposed_and_rejected = first["decision"] == "rigid" and any(
        r.get("decision") in REJECTED for r in first["rounds"]
    )
    if proposed_and_rejected:
        source = run / "sam3d" / f"object_{object_number}.glb"
        cut = split(source, out / "colour_parts") if source.is_file() else None
        if cut is not None:
            result["colour_cut"] = run_loop(
                out / "colour_parts",
                run,
                object_number,
                out / "colour_cut",
                client,
                name=first["object"],
                max_revisions=max_revisions,
                drive=drive,
                frames=real,
            )
            result["colour_cut"]["cut"] = cut["cut"]
        else:
            result["colour_cut_skipped"] = "no clear colour boundary"
    final = result["colour_cut"] or first
    result.update(
        decision=final["decision"],
        part_set="colour_cut" if result["colour_cut"] else "partsam",
        joint=final.get("joint"),
        links=final.get("links"),
        reason=final.get("reason"),
        vlm_calls=first["vlm_calls"] + (result["colour_cut"] or {}).get("vlm_calls", 0),
    )
    (out / RESULT).write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return result


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

    record = articulate_object(
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
                for k in ("object", "decision", "part_set", "reason", "vlm_calls")
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
