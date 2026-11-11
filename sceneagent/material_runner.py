# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SF-20 and SF-21 on the GitHub runner: the equation (2) vote and the VLM audit.

Section 3.4 decides each object's material in two passes. Equation (2) sums
three sources (the physics prior, the VLM and keywords), with a vote of
``unknown`` scaled by 0.4; then "a subsequent VLM audit and recorded material
review may replace the provisional class", and the review is "signed by a
coding agent, not a person". :mod:`sceneagent.material_vote` already holds the
vote itself. This module runs both passes where ``GEMINI_API_KEY`` lives, on
the runner and never on the box:

    python -m sceneagent.material_runner vote  --crops crops.zip --out-dir OUT
    python -m sceneagent.material_runner audit --crops crops.zip \\
        --vote OUT/material_vote.json --out OUT/material_review.json

``crops.zip`` holds ``objects.json`` (``[{"object": n, "name": "..."}]``) and
one or more crops per object named ``object_<n>_<k>.png``. The box writes it;
this module reads it in memory and never unpacks it.

Both passes call ``gemini-3.8-flash`` through
:class:`sceneagent.harness.GeminiHarness`, which reads the key from the
environment only and redacts it from replies and logs. Each output records the
model, the prompt's sha256 and ``vlm_calls``; none records the key.

The physics prior joins the vote only when Sandeep's npz is at its configured
path (``material_vote.configured_prior_checkpoint``). It is not shipped to the
runner, so a runner vote has a null prior source, as KAN-21 asks: null, not
zeros and not a default class.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from sceneagent.material_vote import (
    GEMINI_FLASH_MODEL_ID,
    MaterialVlmError,
    parse_vlm_vote,
    run_material_vote,
    vlm_vote_prompt,
)
from sceneagent.taxonomy import MATERIAL_CLASSES

CROP_RE = re.compile(r"^object_(\d{1,4})_(\d{1,3})\.png$")
MAX_CROP_BYTES = 15 * 1024 * 1024
MAX_CROPS_PER_OBJECT = 8
REASON_CHARS = 400
AUDIT_DECISIONS = ("confirm", "replace")
# Section 3.4: the review is "signed by a coding agent, not a person".
SIGNED_BY = f"coding agent: sceneagent.material_runner audit ({GEMINI_FLASH_MODEL_ID})"

VOTE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "material_class": {"type": "STRING", "enum": list(MATERIAL_CLASSES)},
        "confidence": {"type": "NUMBER"},
    },
    "required": ["material_class", "confidence"],
}

# The audit prompt is ours: the paper describes the second pass but does not
# publish its text. It shows the provisional class and every source, so the
# model audits the vote rather than voting blind a second time.
AUDIT_PROMPT = """\
Section 3.4 material audit, second pass. The images are crops of one object
from a reconstructed scene. A three-source vote (equation (2): physics prior,
VLM and keywords; a vote of "unknown" counts 0.4 of its weight) chose a
provisional material class for it. Check that class against the crops.

Object name: {name}
Provisional class: {provisional}
Sources (source: class, weight entering the vote):
{sources}

Published classes: {classes}.

If the provisional class is right, answer "confirm" with that class. If the
crops show a different material, answer "replace" with the class they show.
Answer as JSON: {{"decision": "confirm" or "replace", "material_class":
"<one published class>", "reason": "<one short sentence naming what you saw>"}}.
"""

AUDIT_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "decision": {"type": "STRING", "enum": list(AUDIT_DECISIONS)},
        "material_class": {"type": "STRING", "enum": list(MATERIAL_CLASSES)},
        "reason": {"type": "STRING"},
    },
    "required": ["decision", "material_class", "reason"],
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


def sha256_text(text: str) -> str:
    """Hash of a prompt, recorded so a verdict names the exact text it answered."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_crops(zip_path: Path) -> list[dict[str, Any]]:
    """Objects and their crops from ``crops.zip``, read in memory.

    Every object in ``objects.json`` needs at least one crop, and every crop
    needs an object, so a half-written zip fails here, before any call.
    """
    crops: dict[int, list[tuple[int, bytes]]] = {}
    with zipfile.ZipFile(zip_path) as archive:
        try:
            listed = json.loads(archive.read("objects.json").decode("utf-8"))
        except KeyError:
            raise ValueError("crops.zip has no objects.json") from None
        for info in archive.infolist():
            match = CROP_RE.fullmatch(info.filename)
            if not match or info.is_dir():
                continue
            if info.file_size > MAX_CROP_BYTES:
                raise ValueError(f"{info.filename} is larger than {MAX_CROP_BYTES}")
            crops.setdefault(int(match.group(1)), []).append(
                (int(match.group(2)), archive.read(info))
            )
    if not isinstance(listed, list) or not listed:
        raise ValueError("objects.json must be a non-empty list")
    objects: list[dict[str, Any]] = []
    seen: set[int] = set()
    for entry in listed:
        number = entry.get("object") if isinstance(entry, dict) else None
        name = entry.get("name") if isinstance(entry, dict) else None
        if not isinstance(number, int) or isinstance(number, bool) or number in seen:
            raise ValueError(f"objects.json entry {entry!r} needs a unique integer")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"object {number} needs a name")
        if number not in crops:
            raise ValueError(f"object {number} has no crop")
        seen.add(number)
        ordered = [blob for _k, blob in sorted(crops[number])]
        objects.append(
            {
                "object": number,
                "name": name.strip(),
                "crops": ordered[:MAX_CROPS_PER_OBJECT],
            }
        )
    orphans = sorted(set(crops) - seen)
    if orphans:
        raise ValueError(f"crops for objects {orphans} are not in objects.json")
    return objects


class HarnessVlm:
    """``classify_crops`` for :func:`run_material_vote`, through the harness.

    One harness call per object: the crops go inline as PNGs with the vote's
    own prompt, and the reply is parsed by ``parse_vlm_vote``, so a reply that
    names no published class or a confidence outside [0, 1] fails as before.
    """

    def __init__(self, harness: ImageHarness) -> None:
        self._harness = harness

    def __repr__(self) -> str:
        return f"HarnessVlm(model_id={self._harness.model_id!r})"

    def classify_crops(self, crops: Sequence[bytes]) -> dict[str, Any]:
        text = self._harness.complete(
            vlm_vote_prompt(),
            images=[(blob, "image/png") for blob in crops],
            response_schema=VOTE_SCHEMA,
        )
        return parse_vlm_vote(text)


def run_vote(
    objects: Sequence[Mapping[str, Any]],
    out_dir: Path,
    harness: ImageHarness,
    source_run: str = "",
    checkpoint: Path | str | None = None,
) -> dict[str, Any]:
    """Equation (2) for every object; ``material_vote.json`` and its manifest."""
    named = [{"name": o["name"], "crops": o["crops"]} for o in objects]
    result = run_material_vote(
        out_dir, named, client=HarnessVlm(harness), checkpoint=checkpoint
    )
    # Add the runner's facts to the manifest the vote already wrote.
    manifest = dict(result["manifest"])
    manifest.update(
        source_run=source_run,
        prompt_sha256=sha256_text(vlm_vote_prompt()),
        prompt_source="sceneagent.material_vote.vlm_vote_prompt (ours)",
        object_numbers=[int(o["object"]) for o in objects],
    )
    Path(result["manifest_path"]).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    result["manifest"] = manifest
    return result


def audit_prompt(name: str, provisional: str | None, sources: Sequence[Any]) -> str:
    """The second-pass prompt for one object, built from its vote record."""
    lines = []
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        if source.get("null"):
            lines.append(f"- {source.get('source')}: did not vote")
        else:
            lines.append(
                f"- {source.get('source')}: {source.get('class')}, "
                f"{source.get('weight')}"
            )
    return AUDIT_PROMPT.format(
        name=name,
        provisional=provisional if provisional is not None else "none (no vote won)",
        sources="\n".join(lines) or "- none",
        classes=", ".join(MATERIAL_CLASSES),
    )


def parse_audit(text: str, provisional: str | None) -> dict[str, str]:
    """The audit reply as ``{decision, material_class, reason}``; raise on anything else.

    ``confirm`` must repeat the provisional class and ``replace`` must change
    it, so the decision and the class cannot disagree in the record.
    """
    body = (text or "").strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", body, flags=re.DOTALL)
    if fenced:
        body = fenced.group(1)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise MaterialVlmError("audit reply was not JSON") from None
    if not isinstance(payload, dict):
        raise MaterialVlmError("audit reply was not a JSON object")
    decision = payload.get("decision")
    material = payload.get("material_class")
    if decision not in AUDIT_DECISIONS:
        raise MaterialVlmError("audit decision is not confirm or replace")
    if material not in MATERIAL_CLASSES:
        raise MaterialVlmError("audit class is not a published class")
    if decision == "confirm" and material != provisional:
        raise MaterialVlmError(
            "audit confirmed a class that is not the provisional one"
        )
    if decision == "replace" and material == provisional:
        raise MaterialVlmError("audit replaced the provisional class with itself")
    reason = payload.get("reason")
    reason = reason.strip() if isinstance(reason, str) else ""
    return {
        "decision": decision,
        "material_class": material,
        "reason": reason[:REASON_CHARS],
    }


def run_audit(
    objects: Sequence[Mapping[str, Any]],
    vote: Mapping[str, Any],
    harness: ImageHarness,
    source_run: str = "",
) -> dict[str, Any]:
    """The second pass and the recorded material review.

    One call per object with the crops and the vote record. The final class is
    the audit's; ``provisional_class`` keeps equation (2)'s. A failed call or a
    bad reply leaves that object's provisional class standing and is recorded
    under ``failed`` (nothing is guessed), so a review is never silently
    missing an object.
    """
    # The vote file lists objects in crops.zip order. Match by position and
    # check the name, since two objects can share a name (two apples).
    votes = [v for v in vote.get("votes", []) if isinstance(v, dict)]
    if len(votes) != len(objects):
        raise ValueError(
            f"material_vote.json has {len(votes)} votes for {len(objects)} objects"
        )
    reviews: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    prompts: list[str] = []
    for obj, record in zip(objects, votes, strict=True):
        name = str(obj["name"])
        if record.get("object") != name:
            raise ValueError(f"vote for {record.get('object')!r} is not {name!r}")
        provisional = record.get("material_class")
        prompt = audit_prompt(name, provisional, record.get("sources") or [])
        prompts.append(prompt)
        row: dict[str, Any] = {
            "object": int(obj["object"]),
            "name": name,
            "provisional_class": provisional,
        }
        try:
            text = harness.complete(
                prompt,
                images=[(blob, "image/png") for blob in obj["crops"]],
                response_schema=AUDIT_SCHEMA,
            )
            verdict = parse_audit(text, provisional)
        except Exception as exc:  # recorded per object; the key is already redacted
            failed.append(
                {"object": row["object"], "error": f"{type(exc).__name__}: {exc}"[:300]}
            )
            row.update(
                audit_class=None,
                decision="not audited",
                final_class=provisional,
                reason="",
            )
        else:
            row.update(
                audit_class=verdict["material_class"],
                decision=verdict["decision"],
                final_class=verdict["material_class"],
                reason=verdict["reason"],
            )
        row["signed_by"] = SIGNED_BY
        reviews.append(row)
    return {
        "section": "3.4",
        "step": "material_audit",
        "model": harness.model_id,
        "signed_by": SIGNED_BY,
        "source_run": source_run,
        "prompt_source": "ours",
        "prompt_template_sha256": sha256_text(AUDIT_PROMPT),
        "prompt_sha256": [sha256_text(p) for p in prompts],
        "vlm_calls": harness.call_count,
        "replaced": sum(1 for r in reviews if r["decision"] == "replace"),
        "reviews": reviews,
        "failed": failed,
    }


def main(argv: list[str] | None = None, harness: ImageHarness | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    vote = sub.add_parser("vote", help="equation (2) for every object in crops.zip")
    vote.add_argument("--crops", type=Path, required=True)
    vote.add_argument("--out-dir", type=Path, required=True)
    vote.add_argument("--source-run", default="")
    audit = sub.add_parser("audit", help="second VLM pass and material_review.json")
    audit.add_argument("--crops", type=Path, required=True)
    audit.add_argument("--vote", type=Path, required=True)
    audit.add_argument("--out", type=Path, required=True)
    audit.add_argument("--source-run", default="")
    args = parser.parse_args(argv)
    if args.source_run and not re.fullmatch(r"[0-9]{1,20}", args.source_run):
        parser.error("--source-run must be a run id (digits only)")
    objects = read_crops(args.crops)
    if harness is None:
        from sceneagent.harness import selected_harness

        harness = selected_harness()
    if args.stage == "vote":
        result = run_vote(objects, args.out_dir, harness, args.source_run)
        print(
            json.dumps(
                {
                    "stage": "vote",
                    "model": result["manifest"]["model_id"],
                    "vlm_calls": result["manifest"]["vlm_calls"],
                    "objects": len(objects),
                }
            )
        )
        return 0
    payload = json.loads(args.vote.read_text(encoding="utf-8"))
    review = run_audit(objects, payload, harness, args.source_run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "stage": "audit",
                "model": review["model"],
                "vlm_calls": review["vlm_calls"],
                "replaced": review["replaced"],
                "failed": len(review["failed"]),
            }
        )
    )
    return 0 if not review["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
