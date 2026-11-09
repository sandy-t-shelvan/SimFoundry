# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prompt-to-object filtering for SF-15 (a SimFoundry addition, KAN-4 comment 10304).

Standard library only: :mod:`sceneagent.mask_worker` imports it inside the
box's gsam2 env, which has no trimesh or the rest of ``sceneagent.pipeline``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

PROMPT = "object ."
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
# A box over half the view is the bin or the table, not an object in it.
MAX_BOX_AREA_FRACTION = 0.5
# Boxes overlapping this much are one object found twice; the higher score stays.
DUPLICATE_IOU = 0.8


def prompt_terms(prompt: str) -> list[str]:
    """The phrases a prompt asks for: GroundingDINO separates them with ``.``."""
    return [term.strip().lower() for term in prompt.split(".") if term.strip()]


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def filter_detections(
    detections: list[dict[str, Any]], width: int, height: int, prompt: str = PROMPT
) -> list[dict[str, Any]]:
    """Mark each detection ``kept`` or give its ``drop_reason``. Order is kept.

    A detection is ``{"phrase", "logit", "box_xyxy"}`` in pixels. It is dropped
    when its phrase names nothing in the prompt, when its box covers more than
    :data:`MAX_BOX_AREA_FRACTION` of the view, or when a higher-scoring kept
    box overlaps it by :data:`DUPLICATE_IOU` or more.
    """
    terms = prompt_terms(prompt)
    out = [dict(d, kept=False, drop_reason=None) for d in detections]
    for det in out:
        phrase = str(det["phrase"]).lower()
        x0, y0, x1, y1 = det["box_xyxy"]
        if not any(term in phrase for term in terms):
            det["drop_reason"] = "phrase not in prompt"
        elif (x1 - x0) * (y1 - y0) > MAX_BOX_AREA_FRACTION * width * height:
            det["drop_reason"] = "box covers most of the view"
    kept: list[dict[str, Any]] = []
    for det in sorted(out, key=lambda d: -float(d["logit"])):
        if det["drop_reason"]:
            continue
        if any(_iou(det["box_xyxy"], k["box_xyxy"]) >= DUPLICATE_IOU for k in kept):
            det["drop_reason"] = "duplicate of a higher-scoring box"
            continue
        det["kept"] = True
        kept.append(det)
    return out
