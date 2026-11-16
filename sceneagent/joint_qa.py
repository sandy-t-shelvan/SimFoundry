# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic checks on a proposed joint, in the proposer's own frame (KAN-44).

The articulation loop (``articulation_loop``) runs these before the model
critic. They need no model and fail with a specific reason the proposer can
act on:

* **physics** -- the Isaac Lab sweep (``articulate_scanned_drive``) rejected it;
* **flush side** -- a hinged lid's lip runs round its free edges and stops at
  its hinge. When the moving part is flush with the body on exactly one side
  (under ``FLUSH_M``) and overhangs it on every other side (at least
  ``LIP_M``), a horizontal hinge belongs on that flush side. When the evidence
  is not that clear (a plain door, a drawer, a lid with no lip) the check
  abstains rather than guessing. On the scanned Mailbox the lid overhangs by
  5.3, 6.6 and 5.3 mm and is flush (0.8 mm) on the fourth side;
* **pivot on an edge** -- a revolute pivot must lie near a boundary of the
  moving part, not inside it.

Everything is computed in PartSAM's parts frame (SAM 3D's Y-up, part units),
where the proposer writes ``axis`` and ``origin``; ``scale_m_per_unit`` turns
the overhangs into metres.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

FLUSH_M = 0.002
LIP_M = 0.003
# A pivot further than this share of the moving part's span from the hinge
# edge is not on that edge.
EDGE_SHARE = 0.25
UP = 1  # SAM 3D meshes are Y-up
HORIZONTAL = (0, 2)
NAMES = "XYZ"


def union_box(parts: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """The bounding box of several parts' boxes."""
    lo = np.min([np.asarray(p["bbox_min"], float) for p in parts], axis=0)
    hi = np.max([np.asarray(p["bbox_max"], float) for p in parts], axis=0)
    return lo, hi


def side_overhangs(
    body: tuple[np.ndarray, np.ndarray], moving: tuple[np.ndarray, np.ndarray]
) -> dict[str, float]:
    """How far the moving part sticks out past the body on each horizontal side (units)."""
    (b_lo, b_hi), (m_lo, m_hi) = body, moving
    sides = {}
    for axis in HORIZONTAL:
        sides[f"-{NAMES[axis]}"] = float(b_lo[axis] - m_lo[axis])
        sides[f"+{NAMES[axis]}"] = float(m_hi[axis] - b_hi[axis])
    return sides


def flush_side(overhang_m: dict[str, float]) -> str | None:
    """The one flush side when the others all have a lip; None when not that clear."""
    flush = [side for side, value in overhang_m.items() if value < FLUSH_M]
    lipped = [side for side, value in overhang_m.items() if value >= LIP_M]
    if len(flush) == 1 and len(lipped) == len(overhang_m) - 1:
        return flush[0]
    return None


def hinge_side(
    axis: Sequence[float],
    origin: Sequence[float],
    moving: tuple[np.ndarray, np.ndarray],
) -> tuple[str, float] | None:
    """Which side of the moving part a horizontal hinge runs along, and how far off that edge.

    Returns None for a hinge that is not horizontal (a door's vertical hinge):
    the flush-side rule is about lids.
    """
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    if abs(a[UP]) >= 0.5:
        return None
    along = HORIZONTAL[int(np.argmax([abs(a[i]) for i in HORIZONTAL]))]
    across = HORIZONTAL[0] if along == HORIZONTAL[1] else HORIZONTAL[1]
    lo, hi = moving
    o = float(np.asarray(origin, float)[across])
    centre, span = (
        (lo[across] + hi[across]) / 2,
        max(float(hi[across] - lo[across]), 1e-12),
    )
    side = f"{'+' if o >= centre else '-'}{NAMES[across]}"
    edge = hi[across] if o >= centre else lo[across]
    return side, abs(o - float(edge)) / span


def review(
    joint: dict[str, Any],
    body_parts: Sequence[dict[str, Any]],
    moving_parts: Sequence[dict[str, Any]],
    scale_m_per_unit: float,
    physics: dict[str, Any] | None,
) -> dict[str, Any]:
    """Problems with one movable joint, the facts behind them, and a note for the proposer."""
    problems: list[str] = []
    facts: dict[str, Any] = {}
    if physics is not None and physics.get("decision") != "accept":
        problems.append(f"physics: {physics.get('reason')}")
    body, moving = union_box(body_parts), union_box(moving_parts)
    overhang = {
        k: v * scale_m_per_unit for k, v in side_overhangs(body, moving).items()
    }
    facts["overhang_mm"] = {k: round(v * 1000, 1) for k, v in overhang.items()}
    flush = flush_side(overhang)
    facts["flush_side"] = flush
    if joint.get("type") == "revolute":
        hinge = hinge_side(joint["axis"], joint["origin"], moving)
        facts["hinge_side"] = hinge[0] if hinge else None
        if hinge and hinge[1] > EDGE_SHARE:
            problems.append(
                f"pivot: the pivot is {hinge[1]:.0%} of the moving part's width from its "
                f"{hinge[0]} edge, not on an edge"
            )
        if hinge and flush and hinge[0] != flush:
            axis = NAMES[
                HORIZONTAL[1] if flush[1] == NAMES[HORIZONTAL[0]] else HORIZONTAL[0]
            ]
            across = NAMES.index(flush[1])
            edge = float(moving[1][across] if flush[0] == "+" else moving[0][across])
            problems.append(
                f"flush side: the moving part overhangs the body by "
                f"{', '.join(f'{v:.1f} mm on {k}' for k, v in facts['overhang_mm'].items() if k != flush)} "
                f"but is flush on {flush} ({facts['overhang_mm'][flush]:.1f} mm). A lid's lip stops "
                f"at its hinge, so the hinge belongs on the {flush} side: axis along {axis}, pivot "
                f"on the {flush} edge ({flush[1]} = {edge:.3f} in these part coordinates), "
                f"not on {hinge[0]}"
            )
    return {"problems": problems, "facts": facts, "accept": not problems}
