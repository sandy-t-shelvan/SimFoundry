# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Advisory check of about four rendered frames. It does not change a gate."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sceneagent.gemini_client import call_failed, safe_generate_images

LABEL = "vlm_check"


def advise(frames: Sequence[bytes], question: str, client: Any) -> dict[str, Any]:
    """Yes or no, with a reason. ``advisory`` stays true.

    A failed call is ``unparsed`` and ``unresolved``. It does not raise.
    """
    if not frames or len(frames) > 6:
        raise ValueError("need about 4 frames")
    reply = safe_generate_images(client, question, frames)
    failure = call_failed(reply)
    text = "" if failure else str(reply.get("text") or "")
    report = {
        "label": LABEL,
        "advisory": True,
        "verdict": "unparsed" if failure else _yes_no(text),
        "reason": failure or text.strip(),
        "model_reply": text,
        "model_id": reply.get("model_id"),
        "http_status": reply.get("http_status"),
        "frames": len(frames),
    }
    if failure:
        report["status"] = "unresolved"
    return report


def _yes_no(text: str) -> str:
    lowered = text.strip().lower()
    if lowered.startswith("yes"):
        return "yes"
    if lowered.startswith("no"):
        return "no"
    return "unparsed"
