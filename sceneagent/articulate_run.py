# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runner half of the articulate pipeline.

Writes the proposal next to the code so the box render can read it. The key
is used only in this process. The box command does not receive it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sceneagent.gemini_client import client_from_environ
from sceneagent.hinge_proposal import propose_articulation

PROPOSAL_NAME = "articulate_proposal.json"
ITEM = "articulated motion"


def write_proposal(
    objects: list[dict[str, Any]], frames: list[bytes], client: Any, path: Path
) -> dict[str, Any]:
    """Propose, then store the record. One-part objects stay rigid."""
    record = propose_articulation(objects, frames, client)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def prepare_for_box(checkout: Path) -> Path | None:
    """Propose from frames already extracted beside the checkout, if the key is set.

    Missing frames or a missing key leaves no proposal file. The box then
    reports that and does not invent a joint.
    """
    frames = _frames(checkout)
    objects = _objects(checkout)
    if len(frames) < 2 or not objects:
        return None
    try:
        client = client_from_environ()
    except Exception:
        return None
    path = checkout / "sceneagent" / PROPOSAL_NAME
    write_proposal(objects, frames[:4], client, path)
    return path


def _frames(checkout: Path) -> list[bytes]:
    folder = checkout / "docs" / "assets" / "articulate_frames"
    if not folder.is_dir():
        return []
    paths = sorted(folder.glob("*.png"))[:4]
    return [path.read_bytes() for path in paths if path.stat().st_size > 0]


def _objects(checkout: Path) -> list[dict[str, Any]]:
    path = checkout / "sceneagent" / "articulate_objects.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    objects = payload.get("objects") if isinstance(payload, dict) else payload
    return list(objects) if isinstance(objects, list) else []
