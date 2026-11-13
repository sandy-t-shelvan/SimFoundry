# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One proposed revolute joint for the Mailbox lid.

Articulate Anything is not called. One Gemini request sees the body and lid
bounding boxes, their shared region, and 2 to 4 frames. The reply is checked
with the existing URDF checker. A rejection is sent back once. A second
rejection is unresolved. The record is ``gemini_proposed``, never measured.

``GEMINI_API_KEY`` is read from the environment at call time. It is not
stored and it is not put in an error message.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

from sceneagent.gemini_client import API_ROOT, PAPER_MODEL_ID, redact
from sceneagent.gemini_client import _request as gemini_request
from sceneagent.mailbox_hinge import hinge_from_reply, write_urdf

SOURCE = "gemini_proposed"


class MissingGeminiKeyError(Exception):
    """``GEMINI_API_KEY`` is missing. The message does not carry a key."""

    def __init__(self) -> None:
        super().__init__("GEMINI_API_KEY is missing or empty")


def shared_region(
    body_min: Sequence[float],
    body_max: Sequence[float],
    lid_min: Sequence[float],
    lid_max: Sequence[float],
) -> dict[str, list[float]] | None:
    """The overlapping or touching box of the two parts, else none."""
    low = [max(float(a), float(b)) for a, b in zip(body_min, lid_min, strict=True)]
    high = [min(float(a), float(b)) for a, b in zip(body_max, lid_max, strict=True)]
    if any(left > right for left, right in zip(low, high, strict=True)):
        return None
    return {"min": low, "max": high}


def part_summary(body: Mapping[str, Any], lid: Mapping[str, Any]) -> dict[str, Any]:
    """Bounding boxes plus the shared region the prompt sends."""
    return {
        "body": _box(body),
        "lid": _box(lid),
        "contact": shared_region(
            body["bbox_min"], body["bbox_max"], lid["bbox_min"], lid["bbox_max"]
        ),
    }


def propose_hinge(
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    frames: Sequence[bytes],
    client: Any,
) -> dict[str, Any]:
    """One proposal, then one retry. The checker is ``hinge_from_reply``."""
    count = len(frames)
    if count < 2 or count > 4:
        raise ValueError("need 2 to 4 frames")
    first_text = str(client.generate(prompt_for(summary), frames).get("text") or "")
    joint, error = _checked(manifest, summary, first_text)
    attempts = [{"model_reply": first_text, "error": error}]
    if error is None:
        return _ok(manifest, joint, attempts)
    second_text = str(
        client.generate(prompt_for(summary, rejected=error), frames).get("text") or ""
    )
    joint, error = _checked(manifest, summary, second_text)
    attempts.append({"model_reply": second_text, "error": error})
    if error is not None:
        return {
            "status": "unresolved",
            "source": SOURCE,
            "joint": None,
            "urdf": None,
            "attempts": attempts,
        }
    return _ok(manifest, joint, attempts)


def prompt_for(summary: Mapping[str, Any], rejected: str | None = None) -> str:
    """The question, and on a retry the checker’s reason."""
    lines = [
        "Propose one revolute joint for this mailbox lid.",
        "Part summary (bounding boxes and shared region):",
        json.dumps(summary),
        "The attached frames show the mailbox.",
        "Reply with one JSON object and no other text.",
        'Keys: "axis" (3 numbers, a unit vector), "pivot" (3 numbers),',
        '"lower", "upper", and "rationale" (one line).',
        "Limits are radians. Do not describe the joint as measured.",
    ]
    if rejected:
        lines.append(f"The previous reply was rejected: {rejected}")
        lines.append("Reply with a corrected JSON object.")
    return "\n".join(lines)


class HingeGemini:
    """One generateContent call with the summary and the frames."""

    def __init__(self, api_key: str) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise MissingGeminiKeyError()
        self.api_key = api_key

    def __repr__(self) -> str:
        return "HingeGemini(api_key='[redacted]')"

    def generate(self, prompt: str, frames: Sequence[bytes]) -> dict[str, Any]:
        import base64

        parts: list[dict[str, Any]] = [{"text": prompt}]
        for frame in frames:
            parts.append(
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": base64.b64encode(frame).decode("ascii"),
                    }
                }
            )
        status, raw = gemini_request(
            f"{API_ROOT}/models/{PAPER_MODEL_ID}:generateContent",
            self.api_key,
            {"contents": [{"parts": parts}]},
        )
        raw = redact(raw, self.api_key)
        if status != 200:
            return {"text": "", "http_status": status, "model_id": PAPER_MODEL_ID}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return {"text": "", "http_status": status, "model_id": PAPER_MODEL_ID}
        text = ""
        candidates = payload.get("candidates") or []
        if candidates:
            chunks = (candidates[0].get("content") or {}).get("parts") or []
            text = "".join(
                part.get("text", "") for part in chunks if isinstance(part, dict)
            )
        return {"text": text.strip(), "http_status": status, "model_id": PAPER_MODEL_ID}


def client_from_environ() -> HingeGemini:
    """Build a client from ``GEMINI_API_KEY``. No other variable is read."""
    if "GEMINI_API_KEY" not in os.environ:
        raise MissingGeminiKeyError()
    return HingeGemini(os.environ["GEMINI_API_KEY"])


def _ok(manifest, joint, attempts) -> dict[str, Any]:
    return {
        "status": "ok",
        "source": SOURCE,
        "joint": joint,
        "urdf": write_urdf(manifest, joint),
        "attempts": attempts,
    }


def _checked(manifest, summary, text):
    parsed = _json_object(text)
    if parsed is None:
        return None, "reply is not a JSON object"
    rationale = parsed.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        return None, "rationale is missing"
    try:
        axis = [float(value) for value in parsed["axis"]]
        pivot = [float(value) for value in parsed["pivot"]]
        lower, upper = float(parsed["lower"]), float(parsed["upper"])
    except (KeyError, TypeError, ValueError):
        return None, "reply is missing axis, pivot, or limits"
    reply = {
        "name": "lid_hinge",
        "type": "revolute",
        "parent": summary["body"]["name"],
        "child": summary["lid"]["name"],
        "axis": axis,
        "origin": pivot,
        "lower": lower,
        "upper": upper,
    }
    try:
        built = hinge_from_reply(manifest, reply)
    except ValueError as exc:
        return None, str(exc)
    joint = dict(built["joint"])
    joint["rationale"] = rationale.strip()
    return joint, None


def _box(part: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": str(part["name"]),
        "bbox_min": [float(value) for value in part["bbox_min"]],
        "bbox_max": [float(value) for value in part["bbox_max"]],
    }


def _json_object(text: str) -> dict[str, Any] | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None
