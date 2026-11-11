# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SF-19 VLM verdict: Gemini looks at each object's contact sheet and picks the retry.

Section 3.1's reviewer is a VLM that inspects each placed object's
registration overlay, eight isolated views and two assembled-scene views.
:mod:`sceneagent.bin_review` draws those eleven views on the box into one
labelled contact sheet per object (``review_sheets.zip``) and chooses a retry
from measured numbers. This module is the VLM half. It runs on the GitHub
runner, where ``GEMINI_API_KEY`` is, and never on the box:

    python -m sceneagent.vlm_review --sheets review_sheets.zip --out vlm_verdicts.json

Gemini sees the sheet and a fixed prompt, not the reviewer's numbers, so its
verdict is its own reading of the images. It must answer with one of the
reviewer's three actions. ``bin_review --verdicts`` then applies those actions
on the box, logs the rule's choice beside each, and rescores.

The prompt is ours (the paper does not publish its reviewer prompt). The
model is ``gemini-3.8-flash`` through :class:`sceneagent.harness.GeminiHarness`,
which reads the key from the environment only and redacts it from every
reply and log line. The verdict file carries no key.
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

DECISIONS = ("keep", "icp_flip", "sam3d_next_view")
SHEET_RE = re.compile(r"^object_(\d{1,4})\.png$")
MAX_SHEET_BYTES = 15 * 1024 * 1024
REASON_CHARS = 400

PROMPT = """\
You are the per-object reviewer in a pipeline that rebuilds a cluttered bin of
household objects from photos. One object's mesh was generated from a single
photo and then fitted (position, rotation, size) to the 3D Gaussian points
that a multi-view reconstruction assigned to that object.

The image is a contact sheet of eleven labelled tiles for this one object:
- registration_overlay: the object's photo, with its segmentation mask
  outlined in green and the fitted mesh's projected silhouette in red. A good
  fit has the red silhouette filling the green outline.
- axis_pos_x, axis_neg_x, axis_pos_y, axis_neg_y, axis_pos_z, axis_neg_z,
  oblique_1, oblique_2: the fitted mesh, shaded grey, seen from that side,
  with the object's reconstructed points drawn over it in blue. A good fit has
  the blue points lying on the grey surface from every side. The points cover
  only the sides the cameras saw, so some empty grey is expected.
- assembled_1, assembled_2: every fitted object projected onto two photos,
  this object in red. A good fit has the red shape on the right object.

Choose exactly one action:
- "keep": the fit is right, or neither retry would help.
- "icp_flip": the mesh has roughly the right shape and size but sits turned
  or upside down on its points (points cover one side, the mesh extends the
  wrong way, the outline is mirrored). The fit is restarted from the mesh
  turned half a turn about each axis.
- "sam3d_next_view": the mesh shape itself is wrong for this object (wrong
  object, wrong proportions, merged with a neighbour), so no rotation fixes
  it. The mesh is generated again from a different photo.

Answer as JSON: {"decision": "<one action>", "reason": "<one short sentence
naming what you saw in the tiles>"}.
"""

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "decision": {"type": "STRING", "enum": list(DECISIONS)},
        "reason": {"type": "STRING"},
    },
    "required": ["decision", "reason"],
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


def prompt_sha256() -> str:
    """Hash of the prompt, so a verdict file names the exact wording it used."""
    return hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()


def parse_verdict(text: str) -> dict[str, str]:
    """The model's JSON reply as ``{decision, reason}``; raise on anything else.

    A fenced code block is unwrapped. The decision must be one of
    ``DECISIONS``; nothing else is mapped onto one.
    """
    body = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", body, flags=re.DOTALL)
    if fenced:
        body = fenced.group(1)
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise ValueError("verdict is not a JSON object")
    decision = payload.get("decision")
    if decision not in DECISIONS:
        raise ValueError(f"decision {decision!r} is not one of {DECISIONS}")
    reason = payload.get("reason")
    reason = reason.strip() if isinstance(reason, str) else ""
    return {"decision": decision, "reason": reason[:REASON_CHARS]}


def read_sheets(zip_path: Path) -> dict[int, bytes]:
    """``object_<n>.png`` members of the zip, read in memory (nothing extracted)."""
    sheets: dict[int, bytes] = {}
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            match = SHEET_RE.fullmatch(info.filename)
            if not match or info.is_dir():
                continue
            if info.file_size > MAX_SHEET_BYTES:
                raise ValueError(f"{info.filename} is larger than {MAX_SHEET_BYTES}")
            sheets[int(match.group(1))] = archive.read(info)
    return dict(sorted(sheets.items()))


def judge(
    sheets: dict[int, bytes], harness: ImageHarness, source_run: str = ""
) -> dict[str, Any]:
    """One Gemini call per sheet. A failed call or bad reply is recorded, not guessed."""
    objects: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for number, png in sheets.items():
        try:
            text = harness.complete(
                PROMPT, images=[(png, "image/png")], response_schema=RESPONSE_SCHEMA
            )
            objects.append({"object": number, **parse_verdict(text)})
        except Exception as exc:  # recorded per object; the key is already redacted
            failed.append(
                {"object": number, "error": f"{type(exc).__name__}: {exc}"[:300]}
            )
    return {
        "source_run": source_run,
        "model": harness.model_id,
        "prompt_sha256": prompt_sha256(),
        "prompt_source": "ours",
        "vlm_calls": harness.call_count,
        "objects": objects,
        "failed": failed,
    }


def main(argv: list[str] | None = None, harness: ImageHarness | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sheets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--source-run",
        default="",
        help="gpu-smoke run id the sheets came from, recorded in the verdicts",
    )
    args = parser.parse_args(argv)
    if args.source_run and not re.fullmatch(r"[0-9]{1,20}", args.source_run):
        parser.error("--source-run must be a run id (digits only)")
    if harness is None:
        from sceneagent.harness import selected_harness

        harness = selected_harness()
    sheets = read_sheets(args.sheets)
    if not sheets:
        print("no object_<n>.png sheets in the zip", file=sys.stderr)
        return 1
    verdicts = judge(sheets, harness, args.source_run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(verdicts, indent=2) + "\n", encoding="utf-8")
    counts = {d: 0 for d in DECISIONS}
    for item in verdicts["objects"]:
        counts[item["decision"]] += 1
    print(
        json.dumps(
            {
                "model": verdicts["model"],
                "vlm_calls": verdicts["vlm_calls"],
                "sheets": len(sheets),
                "verdicts": counts,
                "failed": len(verdicts["failed"]),
            }
        )
    )
    return 0 if not verdicts["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
