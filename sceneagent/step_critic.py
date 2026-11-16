# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SF-18 step-runner critic: Gemini accepts or rejects each pipeline stage's output.

Section 3.1 has an actor-critic visual review of each step, and Fig. 7's step
runner "reviews masks, fields, objects and parts". The box draws one review
image per stage; the VLM, on the GitHub runner where ``GEMINI_API_KEY`` is,
answers accept or reject with a reason and, for a reject, one short note for
the retry. The box then logs each verdict in the ledger and re-runs a
rejected stage once, carrying the note. Same split as the SF-19 reviewer
(:mod:`sceneagent.vlm_review`): the key never reaches the box.

    python -m sceneagent.step_critic render --out OUT              # box, CPU
    python -m sceneagent.step_critic judge --images review_steps.zip --out-dir DIR  # runner

``judge`` writes ``DIR/review/<stage>.json`` (image, verdict, reason, model)
and ``DIR/step_verdicts.json``. ``ledger_entries`` and ``retry_requests`` turn
committed verdicts into ledger lines and at most one retry per stage.

Where the box holds the key itself (box 60254439, decision of 2026-11-13),
``review`` closes the loop in one process, still through the harness:

    python -m sceneagent.step_critic review --out OUT --stage masks  # box, GPU on retry

It draws the stage's image, judges it, writes both ledger lines, and on a
reject re-runs the stage once with the critic's note, then judges the retry.
For masks the note can carry a revised detector prompt (``retry_prompt``);
that is the one input of the masks stage a note can change without touching
its thresholds, which SF-18 leaves out of scope. A reject whose note names no
prompt is logged as a skipped retry: GroundingDINO and SAM on the same inputs
would only draw the same masks again.

The prompts are ours (the paper does not publish its critic prompts), one per
stage, each hashed into the verdict file. ``render`` draws whichever stage
images the run directory already holds: masks from ``masks/``, Gaussians from
the splat PLY and ``association/objects.npz``, meshes from ``sam3d/object_*.glb``,
and parts from ``parts/object_*.npz``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np

STAGES = ("masks", "gaussians", "mesh", "parts")
VERDICTS = ("accept", "reject")
IMAGE_RE = re.compile(r"^(masks|gaussians|mesh|parts)\.png$")
MAX_IMAGE_BYTES = 15 * 1024 * 1024
REASON_CHARS = 400
NOTE_CHARS = 200
REVIEW_DIR = "review_steps"
REVIEW_ZIP = "review_steps.zip"
# ``review`` keeps each attempt's image and verdict here, so the retry does
# not overwrite the evidence of the reject that caused it.
CRITIC_DIR = "step_critic"
CRITIC_RECORD = "loop.json"
# KAN-52: objects the critic named as wrong, left out of the scene from SAM 3D
# on. Association writes new object numbers, so it clears this file.
LEFT_OUT = "critic_left_out.json"
OBJECT_LABEL_RE = re.compile(r"^object_([0-9]{1,4})$")
# Stages whose reject can name objects, and the stage that drops them.
OBJECT_STAGES = ("gaussians", "mesh")
# A retry prompt is GroundingDINO phrases, as ``scene_objects`` writes them.
RETRY_PHRASE_RE = re.compile(r"^[a-z0-9][a-z0-9 \-]{0,39}$")
MAX_RETRY_PHRASES = 6
# A stage re-runs at most this many times on a critic's reject (Jira SF-18:
# "a reject triggers one retry"), so a critic that never accepts cannot loop.
MAX_RETRIES = 1
TILE_PX = 512
MAX_VIEWS = 6

_COMMON = """\
You are the step-runner critic in a pipeline that rebuilds a real scene of
objects from photos or a video. Judge only what the image shows. Answer as
JSON: {"verdict": "accept" or "reject", "reason": "<one short sentence naming
what you saw>", "retry_note": "<for a reject, one short instruction for
re-running this step; empty for an accept>"}.
"""

PROMPTS: dict[str, str] = {
    "masks": _COMMON
    + """
This step segments objects in the input photos. The image tiles up to six
photos; each detected object's mask is outlined in its own colour and
labelled with the detector's phrase. Accept when every clearly visible object
has one outline that follows it, and no outline covers several objects, a
surface the objects stand on (a bin's floor, a table), or the background.
Reject when objects are missed, merged, split, or when a surface or the
background is outlined as an object.

For a reject, also give "retry_prompt": the detector's text prompt rewritten
so a re-run would fix what you saw, as lowercase noun phrases each followed
by " ." (for example "red mug . blue scissors ."), one phrase per object kind
that should be outlined, at most six. Keep every target object named below,
if a list is given. Leave it empty when a different prompt would not help.
""",
    "gaussians": _COMMON
    + """
This step assigns the 3D reconstruction's points to objects. The image shows
the scene's points with each object's points in its own colour, labelled
object_<n>. Accept when each colour forms one compact object. Reject when a
colour spreads over several objects, over the bin or the floor, or leaves a
large far-away cloud.
"""
    + "{objects}",
    "mesh": _COMMON
    + """
This step generates one mesh per object from a photo. Each row is one
object, labelled object_<n>: on the left its mesh in grey, on the right the
object's reconstructed points in blue, both seen from above. Each side is
drawn in its own frame, so their sizes and positions cannot be compared, only
their shapes; fitting the mesh to the points is checked later, by numbers.
Accept when each mesh has the shape of a plausible household object and
roughly the outline of its points. Reject when a mesh is two objects fused, a
fragment, or clearly a different kind of shape from its points.
"""
    + "{objects}",
    "parts": _COMMON
    + """
This step splits articulated objects into parts. The image shows each object
with its parts in different colours. Accept when parts follow real movable
pieces (a lid, a door, a drawer). Reject when parts cut across one rigid
piece or when an obvious movable piece is not separated.
""",
}

_OBJECTS = """
For a reject, also give "bad_objects": the labels (object_<n>) of the
objects that are wrong, so they can be left out of the scene. Leave it empty
when the problem is not in particular objects, for example a target object
that is missing altogether.
"""
PROMPTS = {
    stage: prompt.replace("{objects}", _OBJECTS if stage in OBJECT_STAGES else "")
    for stage, prompt in PROMPTS.items()
}

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "verdict": {"type": "STRING", "enum": list(VERDICTS)},
        "reason": {"type": "STRING"},
        "retry_note": {"type": "STRING"},
        # Optional, and asked for only by the masks prompt.
        "retry_prompt": {"type": "STRING"},
        # Optional, and asked for only by the gaussians and mesh prompts.
        "bad_objects": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["verdict", "reason", "retry_note"],
}


class ImageHarness(Protocol):
    """The part of :class:`sceneagent.harness.GeminiHarness` used here."""

    model_id: str
    call_count: int

    def complete(
        self,
        prompt: str,
        images: Any = (),
        response_schema: dict[str, Any] | None = None,
    ) -> str: ...


def recorded_prompt(out: Path | str) -> str:
    """The detector prompt the masks stage last used, or "" when none is recorded."""
    path = Path(out) / "masks_evidence.json"
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("prompt") or "")
    except (OSError, ValueError):
        return ""


def object_context(out: Path | str) -> str:
    """Each object's detector name and how many mask views it was seen in.

    The image alone does not say which cluster is a target: the critic left
    out the PutMarkerInCup marker, seen in 28 of 30 views, as clutter, beside
    a second "scissors" seen in 3 (kan52-loop-pmic4). Empty without an
    association.
    """
    from sceneagent.physics_props import object_names

    try:
        association = json.loads(
            (Path(out) / "association_evidence.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return ""
    names = object_names(association)
    total = association.get("views")
    total = len(total) if isinstance(total, (dict, list)) else total
    lines = []
    for obj in association.get("object_list", []):
        seen = obj.get("views")
        seen = len(seen) if isinstance(seen, (dict, list)) else seen
        lines.append(
            f"- object_{obj['object']}: {names.get(int(obj['object']), 'object')}, "
            f"seen in {seen} of {total} views"
        )
    if not lines:
        return ""
    return (
        "\nThe objects, by the detector's name for them and the number of views "
        "they were seen in:\n"
        + "\n".join(lines)
        + "\nAn object seen in most views is a real object, even when its points "
        "are noisy; one seen in a few views is more likely a duplicate or clutter.\n"
    )


def scene_context(out: Path | str) -> str:
    """The scene's target objects, from the detector prompt the masks stage used.

    The critic's prompts were written for a bin of clutter, where the bin is
    not an object; on the Mailbox video it took the mailbox -- a target -- for
    a container and told the retry to drop it. Naming the targets fixes that
    for any scene. Empty when the run recorded no prompt.
    """
    names = [p.strip() for p in recorded_prompt(out).split(".") if p.strip()]
    if not names:
        return ""
    return (
        "\nThis scene's target objects are: "
        + ", ".join(names)
        + ". Each of them is an object to outline, even when it is a container "
        "(a mailbox, a cup, a bowl). Anything else -- the surface they stand on, "
        "walls, furniture, other clutter -- is not.\n"
    )


def prompt_sha256(stage: str) -> str:
    """Hash of one stage's prompt, so a verdict names the exact wording it used."""
    return hashlib.sha256(PROMPTS[stage].encode("utf-8")).hexdigest()


def retry_prompt(text: Any) -> str:
    """``"a . b ."`` from a critic's suggested prompt, or ``""`` if it is not one.

    Each phrase must already be a GroundingDINO phrase (lower case, letters,
    digits, spaces, hyphens); nothing is rewritten into one, so a reply that
    is not a prompt is not guessed into one.
    """
    if not isinstance(text, str):
        return ""
    phrases = [p.strip() for p in text.strip().split(".")]
    phrases = [p for p in phrases if p]
    if not phrases or len(phrases) > MAX_RETRY_PHRASES:
        return ""
    if not all(RETRY_PHRASE_RE.fullmatch(p) for p in phrases):
        return ""
    return " . ".join(dict.fromkeys(" ".join(p.split()) for p in phrases)) + " ."


def object_labels(value: Any) -> str:
    """``"object_2,object_5"`` from a critic's list, keeping only real labels."""
    if not isinstance(value, list):
        return ""
    labels = [v.strip() for v in value if isinstance(v, str)]
    kept = [v for v in labels if OBJECT_LABEL_RE.fullmatch(v)]
    return ",".join(dict.fromkeys(kept))


def parse_step_verdict(text: str) -> dict[str, str]:
    """The model's reply as ``{verdict, reason, retry_note}``; raise on anything else.

    A fenced code block is unwrapped. The verdict must be accept or reject;
    nothing is mapped onto one. An accept carries no retry note. A reject
    also carries ``retry_prompt`` when the reply has a well-formed one.
    """
    body = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", body, flags=re.DOTALL)
    if fenced:
        body = fenced.group(1)
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise ValueError("verdict is not a JSON object")
    verdict = payload.get("verdict")
    if verdict not in VERDICTS:
        raise ValueError(f"verdict {verdict!r} is not one of {VERDICTS}")
    reason = payload.get("reason")
    note = payload.get("retry_note")
    reason = reason.strip() if isinstance(reason, str) else ""
    note = note.strip() if isinstance(note, str) and verdict == "reject" else ""
    parsed = {
        "verdict": verdict,
        "reason": reason[:REASON_CHARS],
        "retry_note": note[:NOTE_CHARS],
    }
    prompt = retry_prompt(payload.get("retry_prompt")) if verdict == "reject" else ""
    if prompt:
        parsed["retry_prompt"] = prompt
    bad = object_labels(payload.get("bad_objects")) if verdict == "reject" else ""
    if bad:
        parsed["bad_objects"] = bad
    return parsed


def read_stage_images(zip_path: Path) -> dict[str, bytes]:
    """``<stage>.png`` members of the zip, read in memory, in ``STAGES`` order."""
    found: dict[str, bytes] = {}
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            match = IMAGE_RE.fullmatch(info.filename)
            if not match or info.is_dir():
                continue
            if info.file_size > MAX_IMAGE_BYTES:
                raise ValueError(f"{info.filename} is larger than {MAX_IMAGE_BYTES}")
            found[match.group(1)] = archive.read(info)
    return {stage: found[stage] for stage in STAGES if stage in found}


def judge_steps(
    images: dict[str, bytes],
    harness: ImageHarness,
    source_run: str = "",
    context: str = "",
) -> dict[str, Any]:
    """One call per stage image. A failed call or bad reply is recorded, not guessed."""
    stages: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for stage, png in images.items():
        try:
            prompt = PROMPTS[stage] + context
            text = harness.complete(
                prompt,
                images=[(png, "image/png")],
                response_schema=RESPONSE_SCHEMA,
            )
            stages.append(
                {
                    "stage": stage,
                    "image": f"{REVIEW_DIR}/{stage}.png",
                    "image_sha256": hashlib.sha256(png).hexdigest(),
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "scene_objects": context.strip() or None,
                    **parse_step_verdict(text),
                }
            )
        except Exception as exc:  # recorded per stage; the key is already redacted
            failed.append(
                {"stage": stage, "error": f"{type(exc).__name__}: {exc}"[:300]}
            )
    return {
        "mode": "step",
        "source_run": source_run,
        "model": harness.model_id,
        "prompt_source": "ours",
        "vlm_calls": harness.call_count,
        "stages": stages,
        "failed": failed,
    }


def write_stage_reviews(document: dict[str, Any], out_dir: Path) -> list[Path]:
    """``review/<stage>.json`` per judged stage (Jira SF-18 dc1) and the whole file."""
    review = Path(out_dir) / "review"
    review.mkdir(parents=True, exist_ok=True)
    written = []
    for item in document["stages"]:
        path = review / f"{item['stage']}.json"
        body = {
            **item,
            "model": document["model"],
            "source_run": document["source_run"],
        }
        path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
        written.append(path)
    whole = Path(out_dir) / "step_verdicts.json"
    whole.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    written.append(whole)
    return written


def load_step_verdicts(path: Path) -> dict[str, dict[str, str]]:
    """Committed step verdicts by stage, every field checked (box side)."""
    return verdicts_from(json.loads(Path(path).read_text(encoding="utf-8")))


def verdicts_from(payload: Any) -> dict[str, dict[str, str]]:
    """Step verdicts by stage from a ``judge_steps`` document, every field checked."""
    if not isinstance(payload, dict) or payload.get("mode") != "step":
        raise ValueError("not a step verdict file")
    model = payload.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("step verdicts name no model")
    verdicts: dict[str, dict[str, str]] = {}
    for item in payload.get("stages") or []:
        stage = item.get("stage") if isinstance(item, dict) else None
        if stage not in STAGES:
            raise ValueError(f"stage {stage!r} is not one of {STAGES}")
        if item.get("verdict") not in VERDICTS:
            raise ValueError(f"verdict {item.get('verdict')!r} for {stage}")
        reason, note = item.get("reason"), item.get("retry_note")
        verdicts[stage] = {
            "verdict": item["verdict"],
            "reason": reason[:REASON_CHARS] if isinstance(reason, str) else "",
            "retry_note": note[:NOTE_CHARS] if isinstance(note, str) else "",
            "model": model,
        }
        prompt = retry_prompt(item.get("retry_prompt"))
        if prompt and item["verdict"] == "reject":
            verdicts[stage]["retry_prompt"] = prompt
        bad = item.get("bad_objects")
        bad = object_labels(bad.split(",") if isinstance(bad, str) else bad)
        if bad and item["verdict"] == "reject":
            verdicts[stage]["bad_objects"] = bad
    return verdicts


def ledger_entries(
    verdicts: dict[str, dict[str, str]], attempt: int = 0
) -> list[dict[str, Any]]:
    """One ledger line per judged stage (Jira SF-18 dc2), in ``STAGES`` order.

    ``attempt`` is 0 for the stage's first output and 1 for its retry, so the
    reject and the retry that follows it are both on the record.
    """
    entries = []
    for stage in STAGES:
        if stage not in verdicts:
            continue
        v = verdicts[stage]
        entry: dict[str, Any] = {
            "item": f"step review ({stage})",
            "decision": v["verdict"],
            "use_gpu": False,
            "attempt": attempt,
            "vlm_model": v["model"],
            "reason": v["reason"],
        }
        if v["verdict"] == "reject":
            entry["retry_note"] = v["retry_note"]
            if v.get("retry_prompt"):
                entry["retry_prompt"] = v["retry_prompt"]
            if v.get("bad_objects"):
                entry["bad_objects"] = v["bad_objects"]
        entries.append(entry)
    return entries


def retry_requests(
    verdicts: dict[str, dict[str, str]], retried: dict[str, int] | None = None
) -> list[dict[str, str]]:
    """The stages to re-run, each with the critic's note; never past ``MAX_RETRIES``."""
    retried = retried or {}
    requests = []
    for stage in STAGES:
        v = verdicts.get(stage)
        if not v or v["verdict"] != "reject" or retried.get(stage, 0) >= MAX_RETRIES:
            continue
        request = {"stage": stage, "note": v["retry_note"], "reason": v["reason"]}
        if v.get("retry_prompt"):
            request["prompt"] = v["retry_prompt"]
        if v.get("bad_objects"):
            request["objects"] = v["bad_objects"]
        requests.append(request)
    return requests


def append_ledger(out: Path, entries: list[dict[str, Any]]) -> None:
    """Add lines to ``out/ledger.json``, keeping every line already there."""
    path = Path(out) / "ledger.json"
    try:
        ledger = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        ledger = {}
    if not isinstance(ledger, dict):
        ledger = {}
    ledger["entries"] = [*(ledger.get("entries") or []), *entries]
    path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")


def rerun_masks(out: Path, request: dict[str, str]) -> int:
    """Re-run the masks stage with the critic's prompt (GPU, the box's gsam2 env)."""
    from sceneagent.bin_masks import run_bin_masks

    return run_bin_masks(out, prompt=request["prompt"], critic_note=request["note"])


def left_out_objects(out: Path | str) -> dict[int, dict[str, Any]]:
    """Objects the critic left out, by number, each with its stage and reason."""
    try:
        rows = json.loads((Path(out) / LEFT_OUT).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    objects = rows.get("objects") if isinstance(rows, dict) else None
    return {int(r["object"]): r for r in objects or [] if isinstance(r, dict)}


def clear_left_out(out: Path | str, stages: tuple[str, ...] = OBJECT_STAGES) -> None:
    """Forget what these stages left out, when the objects they judged are remade."""
    kept = [r for r in left_out_objects(out).values() if r.get("stage") not in stages]
    path = Path(out) / LEFT_OUT
    if kept:
        path.write_text(
            json.dumps({"objects": kept}, indent=2) + "\n", encoding="utf-8"
        )
    else:
        path.unlink(missing_ok=True)


def _scene_objects(out: Path) -> list[int]:
    association = json.loads(
        (out / "association_evidence.json").read_text(encoding="utf-8")
    )
    return sorted(int(o["object"]) for o in association.get("object_list", []))


def leave_out(out: Path, request: dict[str, str]) -> int:
    """The gaussians and mesh retry: leave the objects the critic named out.

    Only objects still in the scene count. Returns 2 when the critic named
    none of them and 3 when nothing would be left, so the loop stops rather
    than build an empty scene. Otherwise the objects go to ``LEFT_OUT`` and
    the retry judges the objects that remain.
    """
    out = Path(out)
    current = [n for n in _scene_objects(out) if n not in left_out_objects(out)]
    named = {
        int(m.group(1))
        for label in request.get("objects", "").split(",")
        if (m := OBJECT_LABEL_RE.fullmatch(label))
    }
    dropping = [n for n in current if n in named]
    if not dropping:
        return 2
    if len(dropping) == len(current):
        return 3
    rows = list(left_out_objects(out).values()) + [
        {
            "object": n,
            "stage": request["stage"],
            "reason": request.get("reason", ""),
            "note": request.get("note", ""),
        }
        for n in dropping
    ]
    (out / LEFT_OUT).write_text(
        json.dumps({"objects": rows}, indent=2) + "\n", encoding="utf-8"
    )
    return 0


Renderer = Callable[[Path], Path | None]
Rerunner = Callable[[Path, dict[str, str]], int]
RENDERERS: dict[str, Renderer] = {
    "masks": lambda out: render_masks(out),
    "gaussians": lambda out: render_gaussians(out),
    "mesh": lambda out: render_mesh(out),
}
RERUNNERS: dict[str, Rerunner] = {
    "masks": rerun_masks,
    "gaussians": leave_out,
    "mesh": leave_out,
}
# What a stage's re-run needs from the note before it can do anything new.
RETRY_NEEDS = {"masks": "prompt", "gaussians": "objects", "mesh": "objects"}


def review_and_retry(
    out: Path | str,
    harness: ImageHarness,
    stage: str = "masks",
    *,
    render: Renderer | None = None,
    rerun: Rerunner | None = None,
) -> dict[str, Any]:
    """Judge one stage, re-run it once on a reject with the note, judge the retry.

    Every verdict goes to the ledger with its attempt number, and each
    attempt's image and ``review/<stage>.json`` stay under
    ``out/step_critic/<stage>/attempt-<n>/``. Returns the record also written
    to ``out/step_critic/<stage>/loop.json``.

    A stage with no re-runner (Gaussians, meshes) is judged once: no input of
    it is one a note can change, so a reject is final and is logged as a
    skipped retry. A stage with nothing drawn yet is not judged.
    """
    out = Path(out)
    draw = render or RENDERERS[stage]
    rerun = rerun or RERUNNERS.get(stage)
    need = RETRY_NEEDS.get(stage)
    record: dict[str, Any] = {"stage": stage, "model": harness.model_id, "attempts": []}
    for attempt in range(MAX_RETRIES + 1):
        image = draw(out)
        if image is None:
            record["outcome"] = "nothing to review"
            break
        folder = out / CRITIC_DIR / stage / f"attempt-{attempt}"
        (folder / REVIEW_DIR).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image, folder / REVIEW_DIR / f"{stage}.png")
        context = scene_context(out)
        if stage in OBJECT_STAGES:
            context += object_context(out)
        document = judge_steps(
            {stage: Path(image).read_bytes()}, harness, context=context
        )
        write_stage_reviews(document, folder)
        step: dict[str, Any] = {"attempt": attempt, "review": str(folder)}
        record["attempts"].append(step)
        if document["failed"]:
            step["failed"] = document["failed"]
            record["outcome"] = "critic call failed"
            break
        verdicts = verdicts_from(document)
        step["verdict"] = verdicts[stage]["verdict"]
        append_ledger(out, ledger_entries(verdicts, attempt))
        requests = retry_requests(verdicts, {stage: attempt})
        if not requests:
            record["outcome"] = step["verdict"]
            break
        request = requests[0]
        reason = ""
        if rerun is None:
            reason = f"no input of the {stage} stage is one a note can change"
        elif need and not request.get(need):
            reason = f"the critic's note names no {need}; the same inputs would give the same {stage}"
        elif stage == "masks" and retry_prompt(request["prompt"]) == retry_prompt(
            recorded_prompt(out)
        ):
            # kan52-loop-pmic2: the critic sent back the prompt the masks were
            # made with, and the retry drew the same masks again.
            reason = "the critic's prompt is the one these masks were made with"
        if reason or rerun is None:
            append_ledger(
                out,
                [
                    {
                        "item": f"step retry ({stage})",
                        "decision": "skipped",
                        "attempt": attempt + 1,
                        "retry_note": request["note"],
                        "reason": reason,
                    }
                ],
            )
            step["retry"] = {"skipped": reason}
            record["outcome"] = "reject, retry skipped"
            break
        code = rerun(out, request)
        step["retry"] = {**request, "exit_code": code}
        if code != 0:
            record["outcome"] = f"retry exited {code}"
            break
    record["vlm_calls"] = harness.call_count
    path = out / CRITIC_DIR / stage / CRITIC_RECORD
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def _outline(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_dilation, binary_erosion

    edge = mask & ~binary_erosion(mask, iterations=1)
    return binary_dilation(edge, iterations=2)


def _palette(n: int) -> list[tuple[int, int, int]]:
    """Distinct saturated colours; fixed so a re-render matches byte for byte."""
    import colorsys

    return [
        tuple(int(255 * c) for c in colorsys.hsv_to_rgb((k * 0.618) % 1.0, 0.9, 1.0))  # type: ignore[misc]
        for k in range(max(n, 1))
    ]


def draw_masks_tile(
    photo: np.ndarray, masks: np.ndarray, labels: list[str], size: int = TILE_PX
) -> Any:
    """One photo, dimmed, with each mask outlined in its own colour and labelled."""
    from PIL import Image, ImageDraw

    out = (photo.astype(float) * 0.7).astype(np.uint8)
    colours = _palette(len(masks))
    for mask, colour in zip(masks, colours):
        out[_outline(mask)] = colour
    image = Image.fromarray(out)
    draw = ImageDraw.Draw(image)
    for mask, colour, label in zip(masks, colours, labels):
        rows, cols = np.nonzero(mask)
        if rows.size:
            draw.text(
                (int(cols.min()), max(int(rows.min()) - 12, 0)), label, fill=colour
            )
    image.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def render_masks(out: Path, max_views: int = MAX_VIEWS) -> Path:
    """The masks stage's review image from its own outputs; returns the PNG path.

    Reads ``masks/detections.json``, ``masks/<view>.npz`` and the views they
    were drawn on, from the scene's own view folder (``scene.json``
    ``view_dir``: ``images/`` for the bin, ``mask_views/`` for a video), the
    folder :mod:`sceneagent.bin_masks` masked. Views are taken in name order
    so the same run always draws the same image.
    """
    from PIL import Image

    from sceneagent.bin_run import load_scene

    out = Path(out)
    view_dir = out / load_scene(out).view_dir
    detections = json.loads(
        (out / "masks" / "detections.json").read_text(encoding="utf-8")
    )
    tiles = []
    for view in sorted(detections)[:max_views]:
        kept = sorted(
            (d for d in detections[view] if d.get("kept")),
            key=lambda d: d["mask_index"],
        )
        stack = np.load(out / "masks" / f"{Path(view).stem}.npz")["masks"]
        with Image.open(view_dir / view) as handle:
            photo = np.array(handle.convert("RGB"))
        labels = [str(d.get("phrase", "")) for d in kept]
        tiles.append(draw_masks_tile(photo, stack[: len(kept)], labels))
    if not tiles:
        raise ValueError("no views with detections to draw")
    columns = min(3, len(tiles))
    rows = -(-len(tiles) // columns)
    sheet = Image.new("RGB", (columns * TILE_PX, rows * TILE_PX), "white")
    for k, tile in enumerate(tiles):
        sheet.paste(tile, ((k % columns) * TILE_PX, (k // columns) * TILE_PX))
    folder = out / REVIEW_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "masks.png"
    sheet.save(path)
    return path


def _cap(points: np.ndarray, limit: int = 5000) -> np.ndarray:
    """A deterministic subset, so a re-render of a large cloud matches."""
    if len(points) <= limit:
        return points
    step = int(np.ceil(len(points) / limit))
    return points[::step]


def _bounds(cloud: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frame on the 1st to 99th percentile once there are enough points.

    A splat always has a few far Gaussians; min/max framing let them shrink
    the object to a speck (Mailbox, 2026-11-14). Points outside the frame are
    clipped to its edge, so nothing silently disappears.
    """
    if len(cloud) >= 100:
        low = np.percentile(cloud, 1, axis=0)
        high = np.percentile(cloud, 99, axis=0)
    else:
        low, high = cloud.min(axis=0), cloud.max(axis=0)
    return low, np.maximum(high - low, 1e-6)


def _frame_all(clouds: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """One XY frame that holds every cloud's 2nd-98th percentile box.

    Framing on the pooled points (``_bounds``) cut a whole object off at the
    edge when it sat apart from the rest, and the critic took the PutMarkerInCup
    marker for clutter (kan52-loop-pmic4). Each cloud's own box drops only its
    stray points.
    """
    lows, highs = [], []
    for cloud in clouds:
        if len(cloud) >= 50:
            lows.append(np.percentile(cloud[:, :2], 2, axis=0))
            highs.append(np.percentile(cloud[:, :2], 98, axis=0))
        elif len(cloud):
            lows.append(cloud[:, :2].min(axis=0))
            highs.append(cloud[:, :2].max(axis=0))
    low, high = np.min(lows, axis=0), np.max(highs, axis=0)
    margin = 0.05 * (high - low)
    return low - margin, np.maximum(high - low + 2 * margin, 1e-6)


def _tile(
    groups: list[tuple[np.ndarray, tuple[int, int, int]]],
    labels: list[str] | None = None,
    frame: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Orthographic XY scatter of the groups on one white tile, framed together.

    With ``labels`` each group's label is written at its median point, so a
    critic can name the object it means.
    """
    named = list(zip(groups, labels or [""] * len(groups), strict=True))
    kept = [(pts, colour) for pts, colour in groups if len(pts)]
    canvas = np.full((TILE_PX, TILE_PX, 3), 255, np.uint8)
    if not kept:
        return canvas
    low, span = frame or _bounds(np.concatenate([pts[:, :2] for pts, _colour in kept]))
    for pts, colour in kept:
        xy = (pts[:, :2] - low) / span
        cols = np.clip((xy[:, 0] * (TILE_PX - 5)).astype(int) + 2, 0, TILE_PX - 3)
        rows = np.clip(
            ((1.0 - xy[:, 1]) * (TILE_PX - 5)).astype(int) + 2, 0, TILE_PX - 3
        )
        for dr in (0, 1):
            for dc in (0, 1):
                canvas[rows + dr, cols + dc] = colour
    if not labels:
        return canvas
    from PIL import Image, ImageDraw

    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    for (pts, _colour), label in named:
        if not len(pts) or not label:
            continue
        xy = np.clip((np.median(pts[:, :2], axis=0) - low) / span, 0.0, 1.0)
        x = int(xy[0] * (TILE_PX - 80))
        y = int((1.0 - xy[1]) * (TILE_PX - 20))
        draw.rectangle((x - 2, y - 2, x + 62, y + 12), fill="white")
        draw.text((x, y), label, fill="black")
    return np.asarray(image)


def _write_stage(out: Path, stage: str, image: np.ndarray) -> Path:
    from PIL import Image

    folder = out / REVIEW_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stage}.png"
    Image.fromarray(image).save(path)
    return path


def _save_points(
    out: Path,
    stage: str,
    groups: list[tuple[np.ndarray, tuple[int, int, int]]],
    labels: list[str] | None = None,
    frame: tuple[np.ndarray, np.ndarray] | None = None,
) -> Path:
    """Orthographic XY scatter. Each group is points plus an RGB colour."""
    if not any(len(pts) for pts, _colour in groups):
        raise ValueError(f"no points to draw for {stage}")
    return _write_stage(out, stage, _tile(groups, labels, frame))


def _splat_ply(out: Path) -> Path | None:
    found = sorted((out / "splat" / "export").glob("*.ply"))
    return found[0] if found else None


def render_gaussians(out: Path) -> Path | None:
    """One colour per object, from the splat PLY and ``association/objects.npz``."""
    ply = _splat_ply(out)
    sets_path = out / "association" / "objects.npz"
    if ply is None or not sets_path.is_file():
        return None
    from sceneagent.bin_association import read_ply_xyz

    xyz = read_ply_xyz(ply)
    dropped = left_out_objects(out)
    with np.load(sets_path) as sets:
        keys = sorted(key for key in sets.files if key.startswith("object_"))
        colours = _palette(len(keys))
        groups = []
        labels = []
        every = []
        for key, colour in zip(keys, colours, strict=True):
            ids = np.asarray(sets[key], dtype=np.int64)
            if ids.size == 0:
                continue
            if int(ids.min()) < 0 or int(ids.max()) >= len(xyz):
                raise ValueError(f"{key} index is outside the splat")
            every.append(xyz[ids])
            if int(key.split("_")[1]) in dropped:
                continue
            groups.append((_cap(xyz[ids]), colour))
            labels.append(key)
    if not groups:
        return None
    # Framed on every object, the dropped ones too, so a re-check after a drop
    # is drawn at the same scale: framed on the Mailbox cube alone, its few
    # hundred points filled the tile and read as scattered (kan52-loop-mailbox4).
    return _save_points(out, "gaussians", groups, labels, _frame_all(every))


def render_mesh(out: Path) -> Path | None:
    """One row per object: its mesh in grey, then its Gaussians in blue.

    The two are drawn side by side, each on its own frame, not overlaid: a
    SAM 3D mesh is in its own unit frame and the Gaussians are in the splat's,
    so an overlay put the mesh in a corner and meant nothing (2026-11-14).
    """
    dropped = left_out_objects(out)
    glbs = [
        g
        for g in sorted((out / "sam3d").glob("object_*.glb"))
        if int(g.stem.split("_")[1]) not in dropped
    ][:MAX_VIEWS]
    if not glbs:
        return None
    from sceneagent.bin_place import sample_mesh

    grey = (70, 70, 70)
    blue = (30, 90, 220)
    ply = _splat_ply(out)
    sets_path = out / "association" / "objects.npz"
    xyz = None
    sets = None
    if ply is not None and sets_path.is_file():
        from sceneagent.bin_association import read_ply_xyz

        xyz = read_ply_xyz(ply)
        sets = np.load(sets_path)
    rows: list[np.ndarray] = []
    try:
        for glb in glbs:
            mesh_tile = _tile(
                [(_cap(sample_mesh(glb, 4000, seed=0), 4000), grey)], [glb.stem]
            )
            gauss: list[tuple[np.ndarray, tuple[int, int, int]]] = []
            if xyz is not None and sets is not None and glb.stem in sets.files:
                ids = np.asarray(sets[glb.stem], dtype=np.int64)
                gauss.append((_cap(xyz[ids]), blue))
            frame = _frame_all([g for g, _ in gauss]) if gauss else None
            rows.append(np.concatenate([mesh_tile, _tile(gauss, frame=frame)], axis=1))
    finally:
        if sets is not None:
            sets.close()
    return _write_stage(out, "mesh", np.concatenate(rows, axis=0))


def render_parts(out: Path) -> Path | None:
    """One colour per part. Arrays in ``parts/object_*.npz`` are Nx3 points."""
    files = sorted((out / "parts").glob("object_*.npz"))
    if not files:
        return None
    loaded: list[np.ndarray] = []
    for path in files:
        with np.load(path) as data:
            for key in sorted(data.files):
                pts = np.asarray(data[key], dtype=float)
                if pts.ndim != 2 or pts.shape[1] < 2:
                    raise ValueError(f"{path.name}:{key} is not points")
                if pts.shape[1] == 2:
                    pts = np.column_stack([pts, np.zeros(len(pts))])
                loaded.append(pts[:, :3])
    if not loaded:
        return None
    colours = _palette(len(loaded))
    groups = [(_cap(pts), colour) for pts, colour in zip(loaded, colours, strict=True)]
    return _save_points(out, "parts", groups)


def render_review_images(out: Path) -> dict[str, str]:
    """Draw every stage that has inputs, then zip the images that were written."""
    out = Path(out)
    drawn: dict[str, str] = {}
    if (out / "masks" / "detections.json").is_file():
        try:
            drawn["masks"] = str(render_masks(out))
        except (OSError, ValueError, KeyError):
            pass
    for stage, renderer in (
        ("gaussians", render_gaussians),
        ("mesh", render_mesh),
        ("parts", render_parts),
    ):
        try:
            path = renderer(out)
        except (OSError, ValueError, KeyError, ImportError):
            continue
        if path is not None:
            drawn[stage] = str(path)
    if drawn:
        pack_review_images(out)
    return drawn


def pack_review_images(out: Path) -> Path:
    """Zip every ``review_steps/<stage>.png`` for the runner to judge."""
    out = Path(out)
    archive_path = out / REVIEW_ZIP
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for stage in STAGES:
            png = out / REVIEW_DIR / f"{stage}.png"
            if png.is_file():
                archive.write(png, f"{stage}.png")
    return archive_path


def main(argv: list[str] | None = None, harness: ImageHarness | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    render = sub.add_parser("render", help="draw every stage image that has inputs")
    render.add_argument("--out", type=Path, required=True)
    masks = sub.add_parser("render-masks", help="draw the masks stage's review image")
    masks.add_argument("--out", type=Path, required=True)
    judge = sub.add_parser(
        "judge", help="ask the VLM to judge each stage image (runner)"
    )
    judge.add_argument("--images", type=Path, required=True, help="review_steps.zip")
    judge.add_argument("--out-dir", type=Path, required=True)
    judge.add_argument(
        "--source-run", default="", help="gpu-smoke run id the images came from"
    )
    review = sub.add_parser(
        "review", help="judge a stage and re-run it once on a reject (box with key)"
    )
    review.add_argument("--out", type=Path, required=True)
    review.add_argument("--stage", choices=sorted(RENDERERS), default="masks")
    args = parser.parse_args(argv)
    if args.command == "render":
        drawn = render_review_images(args.out)
        print(json.dumps({"images": drawn, "zip": str(args.out / REVIEW_ZIP)}))
        return 0 if drawn else 1
    if args.command == "render-masks":
        path = render_masks(args.out)
        print(
            json.dumps({"image": str(path), "zip": str(pack_review_images(args.out))})
        )
        return 0
    if args.command == "review":
        if harness is None:
            from sceneagent.harness import selected_harness

            harness = selected_harness()
        record = review_and_retry(args.out, harness, args.stage)
        print(
            json.dumps(
                {
                    "outcome": record.get("outcome"),
                    "model": record["model"],
                    "vlm_calls": record["vlm_calls"],
                }
            )
        )
        return 0 if record.get("outcome") == "accept" else 1
    if args.source_run and not re.fullmatch(r"[0-9]{1,20}", args.source_run):
        parser.error("--source-run must be a run id (digits only)")
    images = read_stage_images(args.images)
    if not images:
        print("no <stage>.png images in the zip", file=sys.stderr)
        return 1
    if harness is None:
        from sceneagent.harness import selected_harness

        harness = selected_harness()
    document = judge_steps(images, harness, args.source_run)
    write_stage_reviews(document, args.out_dir)
    print(
        json.dumps(
            {
                "model": document["model"],
                "vlm_calls": document["vlm_calls"],
                "verdicts": {s["stage"]: s["verdict"] for s in document["stages"]},
                "failed": len(document["failed"]),
            }
        )
    )
    return 0 if not document["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
