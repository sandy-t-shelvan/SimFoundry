# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KAN-42 dc2: the model looks at an object's PartSAM parts for missing geometry.

    python -m sceneagent.parts_review --parts PARTS_DIR --object NAME

Paper §3.4 and Figure 8: after PartSAM, a VLM inspects the parts for hidden
geometry that is missing, such as a marker tip under its cap; CubePart runs
only when it says so. This renders the parts in distinct colours from three
views (``parts_sheet.png``), asks ``gemini-3.8-flash`` with
``section34.part_inspection_prompt`` plus a JSON answer shape, and parses the
reply with ``section34.inspect_parts``, so ``completion_needed`` is decided by
the existing rule. ``PARTS_DIR/parts_review.json`` records the decision, the
reason, the image path, the model and the prompt hash. A failed or unparsed
call is recorded as such, never as "no".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from sceneagent.section34 import inspect_parts, part_inspection_prompt

SHEET = "parts_sheet.png"
REVIEW = "parts_review.json"
TILE_PX = 420
SAMPLES_PER_PART = 6000
PALETTE = (
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
    (0, 128, 128), (170, 110, 40),
)  # fmt: skip
JSON_ASK = (
    'Reply with one JSON object and no other text, with keys "completion_needed" '
    '(true or false), "reason" (one sentence) and "parts" (the part names whose '
    "hidden geometry is missing; empty when none)."
)


def _rotation(axis: str, degrees: float) -> np.ndarray:
    t = np.radians(degrees)
    c, s = np.cos(t), np.sin(t)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


VIEWS = (np.eye(3), _rotation("y", 90), _rotation("x", 35) @ _rotation("y", 40))


def render_sheet(parts: list[dict[str, Any]], path: Path) -> Path:
    """Three orthographic views, one colour per part, legend beneath."""
    import trimesh
    from PIL import Image, ImageDraw

    clouds = []
    for index, part in enumerate(parts):
        mesh = trimesh.load(part["mesh"], force="mesh")
        points, _ = trimesh.sample.sample_surface(mesh, SAMPLES_PER_PART, seed=0)
        clouds.append(
            (np.asarray(points), PALETTE[index % len(PALETTE)], str(part["name"]))
        )
    sheet = Image.new("RGB", (TILE_PX * len(VIEWS), TILE_PX + 60), "white")
    for k, rotation in enumerate(VIEWS):
        turned = [(pts @ rotation.T, colour) for pts, colour, _ in clouds]
        allp = np.concatenate([p for p, _ in turned])
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        scale = (TILE_PX - 20) / max(float((hi - lo)[:2].max()), 1e-9)
        tile = Image.new("RGB", (TILE_PX, TILE_PX), "white")
        draw = ImageDraw.Draw(tile)
        order = sorted(
            ((z, x, y, colour) for pts, colour in turned for x, y, z in pts),
            key=lambda t: t[0],
        )
        for _z, x, y, colour in order:
            draw.point(
                (10 + (x - lo[0]) * scale, TILE_PX - 10 - (y - lo[1]) * scale),
                fill=colour,
            )
        sheet.paste(tile, (k * TILE_PX, 0))
    draw = ImageDraw.Draw(sheet)
    for i, (_pts, colour, name) in enumerate(clouds):
        draw.text((10 + (i % 6) * 200, TILE_PX + 8 + (i // 6) * 22), name, fill=colour)
    sheet.save(path)
    return path


def prompt_for(object_name: str, parts: list[dict[str, Any]]) -> str:
    listed = [f"{p['name']} ({p.get('face_share', 0):.1%} of faces)" for p in parts]
    return (
        f"Object: {object_name}. The image shows its parts from three views, one colour "
        "per part, with the legend below. "
        + part_inspection_prompt(listed)
        + " "
        + JSON_ASK
    )


def review(parts_dir: Path, object_name: str, client: Any) -> dict[str, Any]:
    from sceneagent.gemini_client import call_failed, safe_generate_images

    manifest = json.loads((parts_dir / "parts.json").read_text(encoding="utf-8"))
    parts = manifest["parts"]
    image = render_sheet(parts, parts_dir / SHEET)
    prompt = prompt_for(object_name, parts)
    reply = safe_generate_images(client, prompt, [image.read_bytes()])
    failure = call_failed(reply)
    text = None if failure else str(reply.get("text") or "")
    decided = inspect_parts([p["name"] for p in parts], text)
    reason, flagged = None, []
    if text:
        start, end = text.find("{"), text.rfind("}")
        try:
            payload = json.loads(text[start : end + 1]) if start >= 0 < end else {}
        except ValueError:
            payload = {}
        reason = payload.get("reason") if isinstance(payload, dict) else None
        flagged = list(payload.get("parts") or []) if isinstance(payload, dict) else []
    record = {
        "object": object_name,
        "source_mesh": manifest.get("source_mesh"),
        "image_path": str(image),
        "model": reply.get("model_id"),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "vlm_calls": 1,
        "call_failed": failure,
        "completion_needed": decided["completion_needed"],
        "call_cubepart": decided["call_cubepart"],
        "decision": decided["decision"] if not failure else "failed",
        "reason": reason,
        "parts_missing_geometry": flagged,
        "reply": text,
    }
    (parts_dir / REVIEW).write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--parts", type=Path, required=True)
    parser.add_argument("--object", required=True)
    args = parser.parse_args(argv)
    from sceneagent.gemini_client import client_from_environ

    record = review(args.parts, args.object, client_from_environ())
    print(
        json.dumps(
            {
                k: record[k]
                for k in (
                    "object",
                    "completion_needed",
                    "reason",
                    "parts_missing_geometry",
                )
            }
        )
    )
    return 0 if record["completion_needed"] is not None else 1


if __name__ == "__main__":
    sys.exit(main())
