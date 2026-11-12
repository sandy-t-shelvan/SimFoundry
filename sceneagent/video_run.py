# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mailbox thin slice: stage order and commands, with nothing executed.

The bin chain stays the default for GraspClutter6D scene 000091. This
module only names the first video scene and the existing entry points a
later run would call, in order, against one ``--out`` directory. It does
not start a process, open a network connection, or touch the box.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sceneagent.run4_sampling import BIN_COUNT

# The first video on the path. The other three clips are reruns, not this slice.
FIRST_SCENE = "Mailbox"
VIDEO_RELATIVE = "docs/assets/example_videos/Mailbox.mp4"

# Measured on the Fruits clip (mapper comparison): exhaustive matching plus
# the global mapper registered the same 300 frames faster and kept more
# points. Not a paper cutoff. The bin keeps the incremental mapper.
MATCHING = "all_pairs"
MAPPER = "global_mapper"

# Order from the hand-off: frames through ICP place. Physics and the editor
# come later and are not stages here.
STAGES: tuple[str, ...] = (
    "frames",
    "poses",
    "splat",
    "masks",
    "association",
    "sam3d",
    "place",
)

_MODULES = {
    "frames": "sceneagent.run4",
    "poses": "sceneagent.run4",
    "splat": "sceneagent.bin_splat",
    "masks": "sceneagent.bin_masks",
    "association": "sceneagent.bin_association",
    "sam3d": "sceneagent.bin_sam3d",
    "place": "sceneagent.bin_place",
}


def stage_commands(out: Path | str, video: Path | str) -> dict[str, list[str]]:
    """Return one argv per stage. The caller does not run them."""
    root = str(out)
    clip = str(video)
    commands: dict[str, list[str]] = {}
    for stage in STAGES:
        module = _MODULES[stage]
        if module == "sceneagent.run4":
            commands[stage] = ["python", "-m", module, clip, "--out", root]
        else:
            commands[stage] = ["python", "-m", module, "--out", root]
    return commands


def video_plan(
    scene: str, out: Path | str, video: Path | str | None = None
) -> dict[str, Any]:
    """Build the Mailbox plan. Any other scene name is refused."""
    if scene != FIRST_SCENE:
        raise ValueError(f"the thin slice runs {FIRST_SCENE} only, got {scene!r}")
    clip = Path(VIDEO_RELATIVE if video is None else video)
    root = Path(out)
    return {
        "scene": FIRST_SCENE,
        "video": str(clip),
        "frame_rule": "sharpest_per_bin",
        "frame_bins": BIN_COUNT,
        "matching": MATCHING,
        "mapper": MAPPER,
        "stages": list(STAGES),
        "commands": stage_commands(root, clip),
        "executed": False,
    }


def write_plan(out: Path | str, scene: str = FIRST_SCENE) -> Path:
    """Write ``plan.json`` under ``out``. Does not create the scene or run a stage."""
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "plan.json"
    path.write_text(
        json.dumps(video_plan(scene, root), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
