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

    python -m sceneagent.step_critic render-masks --out OUT        # box, CPU
    python -m sceneagent.step_critic judge --images review_steps.zip --out-dir DIR  # runner

``judge`` writes ``DIR/review/<stage>.json`` (image, verdict, reason, model)
and ``DIR/step_verdicts.json``. ``ledger_entries`` and ``retry_requests`` turn
committed verdicts into ledger lines and at most one retry per stage.

The prompts are ours (the paper does not publish its critic prompts), one per
stage, each hashed into the verdict file. The masks stage is hooked first:
``render_masks`` draws its review image from the masks stage's own outputs.
The other stages have prompts here; their review images come later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
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
# A stage re-runs at most this many times on a critic's reject (Jira SF-18:
# "a reject triggers one retry"), so a critic that never accepts cannot loop.
MAX_RETRIES = 1
TILE_PX = 512
MAX_VIEWS = 6

_COMMON = """\
You are the step-runner critic in a pipeline that rebuilds a cluttered bin of
household objects from photos. Judge only what the image shows. Answer as
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
has one outline that follows it, and no outline covers several objects, the
bin itself, or the floor. Reject when objects are missed, merged, split, or
when the container or background is outlined as an object.
""",
    "gaussians": _COMMON
    + """
This step assigns the 3D reconstruction's points to objects. The image shows
the scene's points with each object's points in its own colour. Accept when
each colour forms one compact object. Reject when a colour spreads over
several objects, over the bin or the floor, or leaves a large far-away cloud.
""",
    "mesh": _COMMON
    + """
This step generates one mesh per object from a photo and fits it to the
object's points. The image shows each mesh in grey with its points in blue.
Accept when each mesh has the shape of a plausible household object and the
points lie on its surface. Reject when a mesh is two objects fused, a
fragment, or clearly the wrong shape for its points.
""",
    "parts": _COMMON
    + """
This step splits articulated objects into parts. The image shows each object
with its parts in different colours. Accept when parts follow real movable
pieces (a lid, a door, a drawer). Reject when parts cut across one rigid
piece or when an obvious movable piece is not separated.
""",
}

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "verdict": {"type": "STRING", "enum": list(VERDICTS)},
        "reason": {"type": "STRING"},
        "retry_note": {"type": "STRING"},
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


def prompt_sha256(stage: str) -> str:
    """Hash of one stage's prompt, so a verdict names the exact wording it used."""
    return hashlib.sha256(PROMPTS[stage].encode("utf-8")).hexdigest()


def parse_step_verdict(text: str) -> dict[str, str]:
    """The model's reply as ``{verdict, reason, retry_note}``; raise on anything else.

    A fenced code block is unwrapped. The verdict must be accept or reject;
    nothing is mapped onto one. An accept carries no retry note.
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
    return {
        "verdict": verdict,
        "reason": reason[:REASON_CHARS],
        "retry_note": note[:NOTE_CHARS],
    }


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
    images: dict[str, bytes], harness: ImageHarness, source_run: str = ""
) -> dict[str, Any]:
    """One call per stage image. A failed call or bad reply is recorded, not guessed."""
    stages: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for stage, png in images.items():
        try:
            text = harness.complete(
                PROMPTS[stage],
                images=[(png, "image/png")],
                response_schema=RESPONSE_SCHEMA,
            )
            stages.append(
                {
                    "stage": stage,
                    "image": f"{REVIEW_DIR}/{stage}.png",
                    "image_sha256": hashlib.sha256(png).hexdigest(),
                    "prompt_sha256": prompt_sha256(stage),
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
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
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
        entries.append(entry)
    return entries


def retry_requests(
    verdicts: dict[str, dict[str, str]], retried: dict[str, int] | None = None
) -> list[dict[str, str]]:
    """The stages to re-run, each with the critic's note; never past ``MAX_RETRIES``."""
    retried = retried or {}
    return [
        {"stage": stage, "note": verdicts[stage]["retry_note"]}
        for stage in STAGES
        if stage in verdicts
        and verdicts[stage]["verdict"] == "reject"
        and retried.get(stage, 0) < MAX_RETRIES
    ]


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

    Reads ``masks/detections.json``, ``masks/<view>.npz`` and ``images/<view>``
    as :mod:`sceneagent.mask_worker` writes them. Views are taken in name
    order so the same run always draws the same image.
    """
    from PIL import Image

    out = Path(out)
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
        with Image.open(out / "images" / view) as handle:
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
    render = sub.add_parser(
        "render-masks", help="draw the masks stage's review image (box)"
    )
    render.add_argument("--out", type=Path, required=True)
    judge = sub.add_parser(
        "judge", help="ask the VLM to judge each stage image (runner)"
    )
    judge.add_argument("--images", type=Path, required=True, help="review_steps.zip")
    judge.add_argument("--out-dir", type=Path, required=True)
    judge.add_argument(
        "--source-run", default="", help="gpu-smoke run id the images came from"
    )
    args = parser.parse_args(argv)
    if args.command == "render-masks":
        path = render_masks(args.out)
        print(
            json.dumps({"image": str(path), "zip": str(pack_review_images(args.out))})
        )
        return 0
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
