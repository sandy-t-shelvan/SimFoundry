# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KAN-56: the detector prompt for any video, named by Gemini, keyed by content.

The mask stage needs GroundingDINO phrases for the objects in a video
(``"marker . cup ."``). They used to come from a table keyed on three file
names, so every other video stopped before poses and a renamed copy of a
known video lost its prompt. Now:

* On the GitHub runner, the only place ``GEMINI_API_KEY`` exists,
  :func:`propose` sends three frames spread across the clip to
  ``gemini-3.8-flash`` with :data:`PROMPT` (ours) and :data:`SCHEMA`, and
  :func:`write_verdict` stores the reply as ``verdicts/objects/<sha256>.json``
  under this package. The file is committed, so a person reviews it before
  the box uses it.
* On the box, :func:`resolve_prompt` takes ``--prompt`` when given, else the
  verdict whose name is the video's SHA-256. No Gemini call runs on the box.

    python -m sceneagent.scene_objects propose --video V.mp4 --out vlm-out/objects
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

VERDICT_DIR = Path(__file__).resolve().parent / "verdicts" / "objects"
WORKFLOW = "sceneagent-vlm-verdict (mode: objects)"
FRAME_COUNT = 3
# Frames go to the model at this long side: enough to name tabletop objects,
# a fraction of the tokens of 1080p.
FRAME_LONG_SIDE_PX = 1024
# GroundingDINO's text encoder takes at most 256 tokens; six short phrases
# keep it far below that and keep the mask stage to the objects that matter.
MAX_OBJECTS = 6

PROMPT = """These are frames from one handheld video that moves around a scene.

List the separate physical objects the video is about: the things standing or
lying on the main support surface (a table, desk, counter, shelf or floor)
that a robot could pick up, push or open. Name each object once, however
many frames show it.

Do not list:
- the support surface itself, walls, floors, windows, curtains or the room;
- built-in fixtures (a sink, a stove, a radiator, a socket);
- parts of an object (a lid, a handle, a cap) or its contents;
- people, hands, shadows or reflections;
- markings on a surface: tape, painted or taped lines and squares, stickers,
  labels, decals or printed patterns, even when they outline a region.

Name each with the most specific common English noun phrase of one to three
words for its shape and kind, the way a person would ("cup", "marker",
"cube", "cardboard box", "running shoe"). Never use a vague word that could
also describe a flat marking or a region, such as "block", "object", "item",
"thing", "piece" or "shape". Use no brand names, colours only when two listed
objects would otherwise share a name, and no punctuation. List the most
prominent object first, at most six. Set movable_by_hand to false for an
object one hand could not lift."""

SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "objects": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "movable_by_hand": {"type": "BOOLEAN"},
                },
                "required": ["name", "movable_by_hand"],
            },
        }
    },
    "required": ["objects"],
}

_PHRASE_DROP = re.compile(r"[^a-z0-9 \-]")


def prompt_sha256() -> str:
    """Hash of :data:`PROMPT` and :data:`SCHEMA`, recorded with every verdict."""
    text = PROMPT + json.dumps(SCHEMA, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def video_sha256(video: Path | str) -> str:
    """SHA-256 of the video's bytes: the verdict's key, whatever the file is named."""
    digest = hashlib.sha256()
    with open(video, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def phrase(name: str) -> str:
    """A GroundingDINO phrase: lower case, letters, digits, spaces and hyphens.

    The period separates phrases in a GroundingDINO prompt, so it can never
    appear inside one.
    """
    cleaned = _PHRASE_DROP.sub(" ", str(name).lower())
    return " ".join(cleaned.split())


def prompt_from_objects(objects: Sequence[dict[str, Any]]) -> str:
    """``"a . b . c ."`` from the reply, in its order, without repeats or blanks."""
    seen: list[str] = []
    for item in objects:
        text = phrase(item.get("name", ""))
        if text and text not in seen:
            seen.append(text)
    if not seen:
        raise ValueError("the reply names no object")
    return " . ".join(seen[:MAX_OBJECTS]) + " ."


def parse_reply(text: str) -> list[dict[str, Any]]:
    """The ``objects`` list from the model's JSON, checked against :data:`SCHEMA`."""
    try:
        payload = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"reply is not JSON: {exc}") from exc
    objects = payload.get("objects") if isinstance(payload, dict) else None
    if not isinstance(objects, list):
        raise ValueError("reply has no objects list")
    checked = []
    for item in objects:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError(f"object without a name: {item!r}")
        movable = item.get("movable_by_hand")
        if not isinstance(movable, bool):
            raise ValueError(f"object without movable_by_hand: {item!r}")
        checked.append({"name": item["name"], "movable_by_hand": movable})
    return checked


def frame_times(duration_s: float, count: int = FRAME_COUNT) -> list[float]:
    """Centres of ``count`` equal spans of the clip, so no frame is the first or last."""
    if not duration_s > 0:
        raise ValueError("video has no duration")
    return [round(duration_s * (i + 0.5) / count, 3) for i in range(count)]


def frame_command(ffmpeg: str, video: str, at_s: float) -> list[str]:
    """One JPEG frame at ``at_s`` to stdout, long side at most 1024 px."""
    side = FRAME_LONG_SIDE_PX
    scale = (
        f"scale=w='if(gte(iw,ih),min(iw,{side}),-2)'"
        f":h='if(gte(iw,ih),-2,min(ih,{side}))'"
    )
    return [
        ffmpeg,
        "-v",
        "error",
        "-ss",
        f"{at_s:.3f}",
        "-i",
        video,
        "-frames:v",
        "1",
        "-vf",
        scale,
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-",
    ]


def _duration(video: str, run: Callable[..., Any]) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise ValueError("ffprobe is required")
    probe = run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            video,
        ],
        check=False,
        capture_output=True,
    )
    if probe.returncode != 0:
        raise ValueError(f"ffprobe failed: {probe.stderr!r}")
    return float(probe.stdout.decode().strip())


def extract_frames(
    video: Path | str, *, run: Callable[..., Any] = subprocess.run
) -> list[tuple[float, bytes]]:
    """Three JPEG frames spread across the clip, as ``(time_s, bytes)``."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValueError("ffmpeg is required")
    frames = []
    for at_s in frame_times(_duration(str(video), run)):
        done = run(
            frame_command(ffmpeg, str(video), at_s), check=False, capture_output=True
        )
        if done.returncode != 0 or not done.stdout:
            raise ValueError(f"no frame at {at_s} s: {done.stderr!r}")
        frames.append((at_s, done.stdout))
    return frames


def propose(
    video: Path | str,
    harness: Any,
    *,
    frames: Sequence[tuple[float, bytes]] | None = None,
) -> dict[str, Any]:
    """One model call; the verdict record (``ok`` false with a reason on a bad reply)."""
    video = Path(video)
    shots = list(frames) if frames is not None else extract_frames(video)
    record: dict[str, Any] = {
        "video_sha256": video_sha256(video),
        "video_name": video.name,
        "frame_times_s": [at for at, _ in shots],
        "model": getattr(harness, "model_id", None),
        "prompt_sha256": prompt_sha256(),
        "source": "ours: prompt and schema in sceneagent/scene_objects.py",
    }
    reply = harness.complete(
        PROMPT,
        images=[(blob, "image/jpeg") for _, blob in shots],
        response_schema=SCHEMA,
    )
    try:
        objects = parse_reply(reply)
        record.update(ok=True, objects=objects, prompt=prompt_from_objects(objects))
    except ValueError as exc:
        record.update(ok=False, reason=str(exc), reply=str(reply)[:2000])
    return record


def write_verdict(record: dict[str, Any], root: Path | str = VERDICT_DIR) -> Path:
    """``<root>/<video sha256>.json``."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{record['video_sha256']}.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def lookup_prompt(video: Path | str, root: Path | str = VERDICT_DIR) -> str | None:
    """The committed prompt for this video's bytes, or None."""
    path = Path(root) / f"{video_sha256(video)}.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not record.get("ok") or not isinstance(record.get("prompt"), str):
        return None
    return record["prompt"]


class MissingPromptError(Exception):
    """No ``--prompt`` and no committed verdict for this video's bytes."""

    def __init__(self, video: Path | str, digest: str) -> None:
        super().__init__(
            f"no detector prompt for {Path(video).name} (sha256 {digest}): pass "
            f"--prompt, or run the {WORKFLOW} workflow for this video and commit "
            f"its verdicts/objects/{digest}.json"
        )


def resolve_prompt(
    video: Path | str, given: str | None = None, root: Path | str = VERDICT_DIR
) -> str:
    """``given`` when set, else the committed verdict; raise when neither exists."""
    if given:
        return given
    found = lookup_prompt(video, root)
    if found is None:
        raise MissingPromptError(video, video_sha256(video))
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser(
        "propose", help="ask Gemini for one video's objects (runner only)"
    )
    run.add_argument("--video", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    from sceneagent.harness import GeminiHarness

    record = propose(args.video, GeminiHarness())
    path = write_verdict(record, args.out)
    print(f"{path}: {record.get('prompt') or record.get('reason')}")
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
