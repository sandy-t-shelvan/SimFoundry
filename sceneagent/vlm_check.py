# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Advisory check of about four rendered frames. It does not change a gate."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from sceneagent.gemini_client import call_failed, safe_generate_images

LABEL = "vlm_check"
_JSON_ASK = (
    "Reply with one JSON object and no other text. "
    'Keys: "verdict" ("yes" or "no") and "reason".'
)
_LEADING = re.compile(r"(yes|no)\b")
_WRAP = re.compile(r"""^[\s*_`"'.,:;!?()\[\]-]+""")


def advise(frames: Sequence[bytes], question: str, client: Any) -> dict[str, Any]:
    """Yes or no, with a reason. ``advisory`` stays true.

    The model is asked for a JSON verdict. A prose yes or no still counts.
    A failed call is ``unparsed`` and ``unresolved``. It does not raise.
    """
    if not frames or len(frames) > 6:
        raise ValueError("need about 4 frames")
    reply = safe_generate_images(client, _prompt(question), frames)
    failure = call_failed(reply)
    text = "" if failure else str(reply.get("text") or "")
    report = {
        "label": LABEL,
        "advisory": True,
        "verdict": "unparsed" if failure else verdict_of(text),
        "reason": failure or text.strip(),
        "model_reply": text,
        "model_id": reply.get("model_id"),
        "http_status": reply.get("http_status"),
        "frames": len(frames),
    }
    if failure:
        report["status"] = "unresolved"
    return report


def _prompt(question: str) -> str:
    return question.rstrip() + "\n" + _JSON_ASK


def verdict_of(text: str) -> str:
    """JSON ``verdict`` first, then a leading yes or no with markup stripped."""
    parsed = _json_verdict(text)
    if parsed is not None:
        return parsed
    return _leading_yes_no(text) or "unparsed"


def _json_verdict(text: str) -> str | None:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get("verdict")
    if not isinstance(value, str):
        return None
    return _leading_yes_no(value)


def _leading_yes_no(text: str) -> str | None:
    cleaned = _WRAP.sub("", text.strip().lower())
    match = _LEADING.match(cleaned)
    if match is None:
        return None
    return match.group(1)
