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


# Round 2: the VLM accepts a retry by comparing its before and after sheets.
# The two are shown as A and B in an order the box drew from a recorded seed,
# so "which fits better" cannot be answered by position. The prompt is ours.
PAIR_PROMPT = """\
You are reviewing two fits of the same object in a reconstructed bin of
household objects. Each image is a contact sheet with the same labelled
tiles, described below; one sheet is fit A, the other fit B. Both sheets use
the same photos and the same segmentation masks, so they differ only in the
object's mesh and its placement.

- registration_overlay (and registration_overlay_2 when present): the
  object's photo from one or two camera views, its segmentation mask
  outlined in green and the fitted mesh's projected silhouette in red. The
  views are the same in both sheets. Better: the red silhouette fills the
  green outline and stays inside it, in every view shown.
- axis_* and oblique_*: the fitted mesh, shaded grey, with the object's
  reconstructed points over it in blue. Better: the blue points lie on the
  grey surface from every side (points cover only the sides cameras saw).
- assembled_1, assembled_2: every fitted object on two photos, this object
  in red. Better: the red shape sits on the right object.

The first image is fit A and the second is fit B. Which fit matches the real
object better? Answer as JSON: {"better": "A" or "B", "reason": "<one short
sentence naming what differs in the tiles>"}.
"""

PAIR_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "better": {"type": "STRING", "enum": ["A", "B"]},
        "reason": {"type": "STRING"},
    },
    "required": ["better", "reason"],
}
PAIR_RE = re.compile(r"^object_(\d{1,4})_([AB])\.png$")
HASH_RE = re.compile(r"[0-9a-f]{64}")


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


def pair_prompt_sha256() -> str:
    """Hash of the pair prompt, recorded in every accept verdict file."""
    return hashlib.sha256(PAIR_PROMPT.encode("utf-8")).hexdigest()


def parse_pair(text: str) -> dict[str, str]:
    """The model's pair reply as ``{better, reason}``; raise on anything else."""
    body = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", body, flags=re.DOTALL)
    if fenced:
        body = fenced.group(1)
    payload = json.loads(body)
    if not isinstance(payload, dict) or payload.get("better") not in ("A", "B"):
        raise ValueError(f"pair reply {body[:80]!r} does not name A or B")
    reason = payload.get("reason")
    reason = reason.strip() if isinstance(reason, str) else ""
    return {"better": payload["better"], "reason": reason[:REASON_CHARS]}


def read_pairs(zip_path: Path) -> dict[int, tuple[bytes, bytes]]:
    """``object_<n>_A.png`` and ``_B.png`` members, read in memory; both or neither."""
    sides: dict[int, dict[str, bytes]] = {}
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            match = PAIR_RE.fullmatch(info.filename)
            if not match or info.is_dir():
                continue
            if info.file_size > MAX_SHEET_BYTES:
                raise ValueError(f"{info.filename} is larger than {MAX_SHEET_BYTES}")
            sides.setdefault(int(match.group(1)), {})[match.group(2)] = archive.read(
                info
            )
    incomplete = sorted(n for n, s in sides.items() if set(s) != {"A", "B"})
    if incomplete:
        raise ValueError(f"objects {incomplete} lack an A or a B sheet")
    return {n: (s["A"], s["B"]) for n, s in sorted(sides.items())}


def candidates_hash_from(candidates_json: Path) -> str:
    """The candidate set's hash; the only thing the runner reads from the manifest."""
    digest = json.loads(Path(candidates_json).read_text(encoding="utf-8")).get(
        "candidates_hash"
    )
    if not isinstance(digest, str) or not HASH_RE.fullmatch(digest):
        raise ValueError(f"candidates_hash {digest!r} is not a sha256")
    return digest


def judge_pairs(
    pairs: dict[int, tuple[bytes, bytes]],
    harness: ImageHarness,
    candidates_hash: str,
    source_run: str = "",
) -> dict[str, Any]:
    """One call per pair, A then B. A failed call or bad reply is recorded, not guessed."""
    objects: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for number, (a, b) in pairs.items():
        try:
            text = harness.complete(
                PAIR_PROMPT,
                images=[(a, "image/png"), (b, "image/png")],
                response_schema=PAIR_SCHEMA,
            )
            objects.append({"object": number, **parse_pair(text)})
        except Exception as exc:  # recorded per object; the key is already redacted
            failed.append(
                {"object": number, "error": f"{type(exc).__name__}: {exc}"[:300]}
            )
    return {
        "mode": "accept",
        "candidates_hash": candidates_hash,
        "source_run": source_run,
        "model": harness.model_id,
        "prompt_sha256": pair_prompt_sha256(),
        "prompt_source": "ours",
        "vlm_calls": harness.call_count,
        "objects": objects,
        "failed": failed,
    }


def read_judged_placement(review_json: Path) -> list[dict[str, Any]]:
    """Each object's fit distance in the run whose sheets are judged.

    ``bin_review --verdicts`` refuses to apply the verdicts unless the scene
    on the box is still placed this way, since object numbers mean nothing
    across placements.
    """
    payload = json.loads(Path(review_json).read_text(encoding="utf-8"))
    placement = []
    for item in payload.get("objects") or []:
        number, fit = item.get("object"), item.get("fit_total")
        if not isinstance(number, int) or not isinstance(fit, (int, float)):
            raise ValueError(f"object {number!r} has no fit_total in {review_json}")
        placement.append({"object": number, "fit_total": float(fit)})
    if not placement:
        raise ValueError(f"no objects in {review_json}")
    return sorted(placement, key=lambda item: item["object"])


def judge(
    sheets: dict[int, bytes],
    harness: ImageHarness,
    source_run: str = "",
    judged_placement: list[dict[str, Any]] | None = None,
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
        "judged_placement": judged_placement,
    }


def main(argv: list[str] | None = None, harness: ImageHarness | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--mode",
        choices=("retry", "accept"),
        default="retry",
        help="retry: pick each object's retry; accept: judge each retry's before/after pair",
    )
    parser.add_argument("--sheets", type=Path, help="review_sheets.zip (retry mode)")
    parser.add_argument("--pairs", type=Path, help="review_pairs.zip (accept mode)")
    parser.add_argument(
        "--candidates", type=Path, help="candidates.json (accept mode; hash only)"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--source-run",
        default="",
        help="gpu-smoke run id the sheets came from, recorded in the verdicts",
    )
    parser.add_argument(
        "--review-json",
        type=Path,
        default=None,
        help="bin_review.json of the same run; its fit distances bind the verdicts to that placement",
    )
    args = parser.parse_args(argv)
    if args.source_run and not re.fullmatch(r"[0-9]{1,20}", args.source_run):
        parser.error("--source-run must be a run id (digits only)")
    if args.mode == "accept":
        if args.pairs is None or args.candidates is None:
            parser.error("accept mode needs --pairs and --candidates")
        digest = candidates_hash_from(args.candidates)
        pairs = read_pairs(args.pairs)
        if not pairs:
            print("no object_<n>_A/B.png pairs in the zip", file=sys.stderr)
            return 1
        if harness is None:
            from sceneagent.harness import selected_harness

            harness = selected_harness()
        result = judge_pairs(pairs, harness, digest, args.source_run)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "mode": "accept",
                    "model": result["model"],
                    "vlm_calls": result["vlm_calls"],
                    "pairs": len(pairs),
                    "failed": len(result["failed"]),
                }
            )
        )
        return 0 if not result["failed"] else 1
    if args.sheets is None:
        parser.error("retry mode needs --sheets")
    if harness is None:
        from sceneagent.harness import selected_harness

        harness = selected_harness()
    sheets = read_sheets(args.sheets)
    if not sheets:
        print("no object_<n>.png sheets in the zip", file=sys.stderr)
        return 1
    judged = read_judged_placement(args.review_json) if args.review_json else None
    verdicts = judge(sheets, harness, args.source_run, judged)
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
