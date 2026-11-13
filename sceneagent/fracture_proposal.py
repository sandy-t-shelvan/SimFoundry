# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Whether an object should break, and where. No break is rendered.

A fracture render needs a solver that is not installed. The record says so.
The label is ``gemini_proposed``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

SOURCE = "gemini_proposed"
RENDER_SKIPPED = "A break render needs a fracture solver that is not installed."


def propose_fracture(name: str, frames: Sequence[bytes], client: Any) -> dict[str, Any]:
    """One call with image bytes. Rendering is always skipped."""
    if not 2 <= len(frames) <= 4 or any(not frame for frame in frames):
        raise ValueError("need 2 to 4 image frames")
    reply = client.generate_images(_prompt(name), frames)
    text = str(reply.get("text") or "")
    parsed = _json_object(text)
    record: dict[str, Any] = {
        "object": name,
        "source": SOURCE,
        "breakable": None,
        "where": None,
        "rationale": None,
        "model_reply": text,
        "model_id": reply.get("model_id"),
        "render": "skipped",
        "render_reason": RENDER_SKIPPED,
    }
    if parsed is None or "breakable" not in parsed:
        record["status"] = "unresolved"
        return record
    record["status"] = "ok"
    record["breakable"] = bool(parsed["breakable"])
    record["where"] = None if parsed.get("where") is None else str(parsed["where"])
    record["rationale"] = (
        None if parsed.get("rationale") is None else str(parsed["rationale"])
    )
    return record


def _prompt(name: str) -> str:
    return (
        f"Object {name!r}. The attached images show it. "
        "Reply with one JSON object and no other text. "
        'Keys: "breakable" (true or false), "where" (a short place, or null), '
        'and "rationale" (one line). Do not describe the answer as measured.'
    )


def _json_object(text: str) -> dict[str, Any] | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None
