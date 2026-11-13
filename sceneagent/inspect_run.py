# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SF-22 (KAN-23) on a run directory: the inspector's views, after the settle.

    python -m sceneagent.inspect_run --out /workspace/sf4-smoke/<video>

The box side of :mod:`sceneagent.inspector`. It draws the views of the
exported scene (the objects that passed the placement gate, in the export's
gravity-aligned frame, in metres) and writes the settle summary from
``settle.json``, the Isaac Lab 3 settle :mod:`sceneagent.settle_run` already
ran, rather than running a second simulation. The sheet, the views zip and
the summary are copied to the run directory, where gpu-smoke publishes them.
The fix report is written on the GitHub runner, which holds the Gemini key
(workflow ``sceneagent-inspector-report``); this module never calls a model.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

ITEM = "inspection views"
FOLDER = "inspection"
MANIFEST = "scene_manifest.json"


def manifest(out: Path) -> dict[str, Any]:
    """The inspector's scene manifest for the exported objects of ``out``."""
    from sceneagent.physics_props import object_names

    placement = json.loads((out / "bin_score.json").read_text(encoding="utf-8"))
    association = json.loads(
        (out / "association_evidence.json").read_text(encoding="utf-8")
    )
    export = json.loads((out / "export_evidence.json").read_text(encoding="utf-8"))
    world = np.asarray(export["support"]["world_from_scene"], dtype=float)
    metres = float(association["splat_unit_m"])
    names = object_names(association)
    objects = []
    for record in placement["icp"]["objects"]:
        if not record.get("placed") or record.get("gate") != "accept":
            continue
        number = int(record["object"])
        similarity = record["similarity"]
        rotation = np.asarray(similarity["rotation"], dtype=float)
        translation = np.asarray(similarity["translation"], dtype=float)
        # The export's frame: x_world = W (s R x + t) + w, then metres.
        objects.append(
            {
                "name": f"object_{number} ({names.get(number, 'object')})",
                "mesh": str(record["glb"]),
                "scale": float(similarity["scale"]),
                "rotation": (world[:3, :3] @ rotation).tolist(),
                "translation": (world[:3, :3] @ translation + world[:3, 3]).tolist(),
                "metres_per_unit": metres,
            }
        )
    if not objects:
        raise ValueError("no exported object to inspect")
    return {"objects": objects, "frame": "export stage: Z up, metres"}


def settle_summary(out: Path) -> dict[str, Any]:
    """The settle that ran (settle_run), in the shape the inspector reports on."""
    result = json.loads((out / "settle.json").read_text(encoding="utf-8"))
    objects = result.get("objects", {})
    return {
        "simulator": "isaac_lab",
        "ran": True,
        "settled": result.get("decision") == "accept",
        "reason": result.get("reason", ""),
        "seconds": result.get("seconds"),
        "bodies": [
            {
                "name": name,
                "speed_m_s": motion.get("speed_m_s"),
                "centre_z_m": motion.get("centre_z_m"),
            }
            for name, motion in sorted(objects.items())
        ],
        "source": "settle.json (sceneagent.settle_run, Isaac Lab 3)",
    }


def run_inspect(out: Path | str, *, render=None) -> int:
    """Draw the views and write the summary; one ledger item."""
    from sceneagent import inspector

    render = render or inspector.render_inspection
    out = Path(out)
    ledger_path = out / "ledger.json"
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        ledger = {"entries": []}
    entries = [e for e in ledger.get("entries", []) if e.get("item") != ITEM]
    entry: dict[str, Any] = {"item": ITEM, "use_gpu": False}
    try:
        folder = out / FOLDER
        folder.mkdir(parents=True, exist_ok=True)
        scene = folder / MANIFEST
        scene.write_text(json.dumps(manifest(out), indent=2) + "\n", encoding="utf-8")
        result = render(scene, folder)
        summary = {
            **settle_summary(out),
            "objects": result["settle"].get("objects"),
            "views": result["settle"].get("views"),
            "renderer": result["settle"].get("renderer"),
        }
        (folder / inspector.SETTLE_NAME).write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        for name in (inspector.SHEET_NAME, inspector.VIEWS_ZIP, inspector.SETTLE_NAME):
            shutil.copyfile(folder / name, out / name)
        entry.update(
            decision="accept",
            objects=len(summary["objects"] or []),
            settled=summary["settled"],
        )
    except (OSError, ValueError, KeyError) as exc:
        entry.update(decision="reject", reason=f"{type(exc).__name__}: {exc}"[:800])
    entries.append(entry)
    ledger["entries"] = entries
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    return 0 if entry["decision"] == "accept" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    return run_inspect(args.out)


if __name__ == "__main__":
    sys.exit(main())
