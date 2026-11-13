# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Advisory check of about four rendered frames. It does not change a gate."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

LABEL = "vlm_check"


def advise(frames: Sequence[bytes], question: str, client: Any) -> dict[str, Any]:
    """Yes or no, with a reason. ``advisory`` stays true."""
    if not frames or len(frames) > 6:
        raise ValueError("need about 4 frames")
    reply = client.generate_images(question, frames)
    text = str(reply.get("text") or "")
    verdict = _yes_no(text)
    return {
        "label": LABEL,
        "advisory": True,
        "verdict": verdict,
        "reason": text.strip(),
        "model_reply": text,
        "model_id": reply.get("model_id"),
        "frames": len(frames),
    }


def _yes_no(text: str) -> str:
    lowered = text.strip().lower()
    if lowered.startswith("yes"):
        return "yes"
    if lowered.startswith("no"):
        return "no"
    return "unparsed"
