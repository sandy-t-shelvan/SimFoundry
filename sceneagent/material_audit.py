# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SF-21: a second VLM pass that confirms or replaces the material vote.

The vote (equation 2) is the provisional class. This pass sees the object's
crops, that class, and every source, then writes ``material_review.json``.
``sceneagent.physics_props`` already applies a review whose ``decision`` is
``confirm`` or ``replace``. A failed call keeps the provisional class and is
listed under ``failed``. The module does not read an API key; the client
comes from :func:`sceneagent.vlm_port.client_from_environ`, which calls
``generate_images``. The process exits 1 when any object call fails. A
failure keeps the provisional class and records ``http_status`` and ``error``.

    python -m sceneagent.material_audit --vote material_vote.json \\
        --crops-dir CROPS --out material_review.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sceneagent.taxonomy import MATERIAL_CLASSES, require_material
from sceneagent.vlm_port import ImageClient, client_from_environ

SIGNED_BY = "coding agent: sceneagent.material_audit"
DECISIONS = ("confirm", "replace")
REASON_CHARS = 400
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

AUDIT_INSTRUCTIONS = (
    "Section 3.4 material audit, second pass. "
    "The images, when attached, are crops of one object. "
    "Confirm the provisional class or replace it. "
    "Reply with one JSON object and no other text. "
    'Keys: "decision" (confirm or replace), "audit_class", and "reason". '
    f"audit_class is one of: {', '.join(MATERIAL_CLASSES)}. "
    "confirm only when audit_class is the provisional class. "
    "replace only when it is a different class."
)


def prompt_sha256() -> str:
    """Hash of the shared instructions, so a review names the wording it used."""

    return hashlib.sha256(AUDIT_INSTRUCTIONS.encode("utf-8")).hexdigest()


def prompt_for(
    name: str, provisional: str | None, sources: Sequence[Mapping[str, Any]]
) -> str:
    """The shared instructions plus this object's class and sources."""

    listed = json.dumps(list(sources), sort_keys=True)
    return "\n".join(
        [
            AUDIT_INSTRUCTIONS,
            f"Object: {name}.",
            f"Provisional class: {provisional}.",
            f"Sources: {listed}",
        ]
    )


def parse_audit(text: str, provisional: str | None) -> dict[str, str]:
    """``confirm`` or ``replace``. The class must agree with the decision."""

    body = text.strip()
    if body.startswith("```"):
        body = body.strip("`").removeprefix("json").strip()
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise ValueError("audit reply is not a JSON object")
    decision = payload.get("decision")
    if decision not in DECISIONS:
        raise ValueError(f"decision {decision!r} is not confirm or replace")
    audit_class = require_material(payload.get("audit_class"))
    reason = payload.get("reason")
    reason = reason.strip() if isinstance(reason, str) else ""
    if decision == "confirm" and audit_class != provisional:
        raise ValueError("confirm does not match the provisional class")
    if decision == "replace" and audit_class == provisional:
        raise ValueError("replace names the provisional class")
    return {
        "decision": decision,
        "audit_class": audit_class,
        "reason": reason[:REASON_CHARS],
    }


def load_crops(crops_dir: Path) -> dict[str, list[bytes]]:
    """``crops_dir/<object name>/*.{png,jpg,jpeg}``, in name order."""

    root = Path(crops_dir)
    found: dict[str, list[bytes]] = {}
    if not root.is_dir():
        raise ValueError(f"crops directory {root} is missing")
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        images = [
            path.read_bytes()
            for path in sorted(folder.iterdir())
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        ]
        if images:
            found[folder.name] = images
    return found


def audit_votes(
    votes: Sequence[Mapping[str, Any]],
    crops: Mapping[str, Sequence[bytes]],
    client: ImageClient,
) -> dict[str, Any]:
    """One ``generate_images`` call per vote. Failures keep the provisional class."""

    reviews: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    replaced: list[str] = []
    calls = 0
    model = None
    for vote in votes:
        name = str(vote.get("object") or vote.get("name") or "")
        provisional = vote.get("material_class")
        provisional = None if provisional is None else str(provisional)
        sources = list(vote.get("sources") or [])
        images = [blob for blob in crops.get(name, ()) if isinstance(blob, bytes)]
        reply = None
        try:
            calls += 1
            reply = client.generate_images(
                prompt_for(name, provisional, sources), images=images
            )
            if not isinstance(reply, dict):
                raise ValueError("audit reply is not an object")
            model = reply.get("model_id") or model
            status = reply.get("http_status")
            text = reply.get("text") if status in (None, 200) else ""
            if (
                status not in (None, 200)
                or not isinstance(text, str)
                or not text.strip()
            ):
                raise ValueError(f"audit call failed with HTTP {status}")
            parsed = parse_audit(text, provisional)
        except Exception as exc:
            status = reply.get("http_status") if isinstance(reply, dict) else None
            error = _error_text(reply, exc)
            failed.append({"name": name, "http_status": status, "error": error})
            reviews.append(_row(name, provisional, None, "failed", "", status, error))
            continue
        final = parsed["audit_class"]
        reviews.append(
            _row(
                name,
                provisional,
                final,
                parsed["decision"],
                parsed["reason"],
                reply.get("http_status"),
                None,
            )
        )
        if parsed["decision"] == "replace":
            replaced.append(name)
    return {
        "prompt_sha256": prompt_sha256(),
        "model": model,
        "vlm_calls": calls,
        "signed_by": SIGNED_BY,
        "replaced": replaced,
        "reviews": reviews,
        "failed": failed,
    }


def write_review(document: Mapping[str, Any], path: Path) -> Path:
    """Write ``material_review.json``."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return destination


def _error_text(reply: Any, exc: Exception) -> str:
    body = ""
    if isinstance(reply, dict):
        body = str(reply.get("error_body") or "").strip()
    if not body:
        body = f"{type(exc).__name__}: {exc}"
    return body[:300]


def _row(
    name: str,
    provisional: str | None,
    audit_class: str | None,
    decision: str,
    reason: str,
    http_status: Any = None,
    error: str | None = None,
) -> dict[str, Any]:
    final = provisional if decision == "failed" else audit_class
    return {
        "name": name,
        "provisional_class": provisional,
        "audit_class": audit_class,
        "final_class": final,
        "decision": decision,
        "reason": reason,
        "http_status": http_status,
        "error": error,
        "signed_by": SIGNED_BY,
    }


def main(argv: list[str] | None = None, client: ImageClient | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--vote", type=Path, required=True, help="material_vote.json")
    parser.add_argument("--crops-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = json.loads(args.vote.read_text(encoding="utf-8"))
    votes = payload.get("votes") if isinstance(payload, dict) else None
    if not isinstance(votes, list) or not votes:
        print("vote file has no votes", file=sys.stderr)
        return 1
    opened = client if client is not None else client_from_environ()
    document = audit_votes(votes, load_crops(args.crops_dir), opened)
    write_review(document, args.out)
    failed = document["failed"]
    print(
        json.dumps(
            {
                "ok": not failed,
                "vlm_calls": document["vlm_calls"],
                "replaced": document["replaced"],
                "failed": len(failed),
            }
        )
    )
    if failed:
        print(json.dumps({"failed": failed}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
