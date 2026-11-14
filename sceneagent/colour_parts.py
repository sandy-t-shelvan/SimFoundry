# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Split a scanned object in two where its colour changes most sharply (KAN-42).

    python -m sceneagent.colour_parts --mesh RUN/sam3d/object_N.glb --out PARTS_DIR

PartSAM cuts by geometry. A part that moves is often a different colour from
the body it moves against -- a marker's cap, a coloured lid, a drawer front --
and on the PutMarkerInCup marker PartSAM cut along the marker's length, so no
part was the whole cap. This looks for one plane across one of the mesh's
axes where the colour changes most sharply: the vertices are binned into
``SLICES`` slices along each axis, and every cut between slices is scored by
the colour distance between the two sides over their spread (a two-class
Fisher ratio). The best cut is used only when the two sides really differ
(``MIN_CONTRAST``) and the smaller side is a real part (``MIN_SHARE``);
otherwise nothing is written, so a one-colour object is never split.

The output is ``parts.json`` in the shape ``partsam_worker`` writes (larger
side ``part_0``, smaller ``part_1``), so the articulation loop can use it as
a second part set when PartSAM's parts give it no joint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SLICES = 48
MIN_SHARE = 0.05  # the smaller side holds at least this share of the faces
MIN_CONTRAST = 3.0  # Fisher ratio: squared mean distance over pooled variance
MIN_COLOUR_GAP = 40.0  # and the mean colours differ by this much (0-255 RGB)
NAMES = "XYZ"


def best_cut(points: np.ndarray, colours: np.ndarray) -> dict[str, Any] | None:
    """The axis-aligned cut with the sharpest colour change, or None."""
    best: dict[str, Any] | None = None
    rgb = colours[:, :3].astype(float)
    for axis in range(3):
        coord = points[:, axis]
        lo, hi = float(coord.min()), float(coord.max())
        if hi - lo <= 0:
            continue
        edges = np.linspace(lo, hi, SLICES + 1)[1:-1]
        for cut in edges:
            low = coord < cut
            share = min(low.mean(), 1 - low.mean())
            if share < MIN_SHARE:
                continue
            a, b = rgb[low], rgb[~low]
            gap = float(np.linalg.norm(a.mean(0) - b.mean(0)))
            spread = float(
                (a.var(0).sum() * len(a) + b.var(0).sum() * len(b)) / len(rgb)
            )
            contrast = gap**2 / max(spread, 1e-9)
            if (
                gap >= MIN_COLOUR_GAP
                and contrast >= MIN_CONTRAST
                and (best is None or contrast > best["contrast"])
            ):
                best = {
                    "axis": NAMES[axis],
                    "at": float(cut),
                    "contrast": round(contrast, 2),
                    "colour_gap": round(gap, 1),
                    "share": round(float(share), 3),
                    "colours": [
                        [round(float(v)) for v in a.mean(0)],
                        [round(float(v)) for v in b.mean(0)],
                    ],
                }
    return best


def split(mesh_path: Path, out: Path) -> dict[str, Any] | None:
    """Write the two colour parts and their ``parts.json``; None when the colour gives no cut."""
    import trimesh

    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    colours = getattr(mesh.visual, "vertex_colors", None)
    if colours is None or len(colours) != len(mesh.vertices):
        return None
    cut = best_cut(np.asarray(mesh.vertices), np.asarray(colours))
    if cut is None:
        return None
    axis = NAMES.index(cut["axis"])
    centres = np.asarray(mesh.triangles_center)[:, axis]
    low = centres < cut["at"]
    sides = sorted([low, ~low], key=lambda m: -int(m.sum()))  # larger first: the body
    (out / "parts").mkdir(parents=True, exist_ok=True)
    parts = []
    for index, faces in enumerate(sides):
        piece = mesh.submesh([np.nonzero(faces)[0]], append=True)
        path = out / "parts" / f"part_{index}.glb"
        piece.export(path)
        parts.append(
            {
                "name": f"part_{index}",
                "parent_object": None,
                "partsam_label": None,
                "faces": int(faces.sum()),
                "face_share": float(faces.mean()),
                "bbox_min": [float(v) for v in piece.bounds[0]],
                "bbox_max": [float(v) for v in piece.bounds[1]],
                "centroid": [float(v) for v in piece.centroid],
                "mesh": str(path),
            }
        )
    record = {
        "source_mesh": Path(mesh_path).name,
        "method": "colour_cut",
        "cut": cut,
        "parts": parts,
        "dropped": [],
    }
    (out / "parts.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    record = split(args.mesh, args.out)
    print(json.dumps(record["cut"] if record else {"cut": None}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
