# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Part lists for every exported object, in the shape propose_articulation reads.

The worker is injected. This module does not clone PartSAM or install it.
CubePart runs only when that object's review says completion is needed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

PARTSAM_ENTRY = Path("/workspace/PartSAM/evaluation/eval_everypart.py")


def partsam_available(entry: Path = PARTSAM_ENTRY) -> bool:
    """True when the entry file is already on disk. Nothing is downloaded."""
    return Path(entry).is_file()


def parts_for_export(
    objects: Sequence[Mapping[str, Any]],
    worker: Callable[[str], Sequence[Mapping[str, Any]]],
    reviews: Mapping[str, Mapping[str, Any]] | None = None,
    completer: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """One parts list per exported object.

    ``worker`` is PartSAM. ``completer`` is CubePart and is called only when
    that object's review has ``completion_needed`` true.
    """
    reviews = reviews or {}
    built = []
    for obj in objects:
        name = str(obj["name"])
        parts = [dict(part) for part in worker(str(obj["mesh_path"]))]
        review = reviews.get(name) or {}
        called = False
        if review.get("completion_needed") is True:
            if completer is None:
                raise ValueError(f"{name} needs completion and no completer was given")
            parts.append(dict(completer(obj)))
            called = True
        built.append(
            {
                "name": name,
                "mesh_path": str(obj["mesh_path"]),
                "parts": parts,
                "completion_called": called,
            }
        )
    return built
