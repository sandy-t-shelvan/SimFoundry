# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attach existing object vectors to the Gaussians of that object.

The spec is ``docs/feature-lift.md``. This module does not run an encoder,
does not train, and does not decide which Gaussian belongs to which object.
The assignment array is an input.

In a run, :func:`run_feature_lift` reads the association's object sets and
the ``prior_input`` packs (whose DINOv2 and OpenCLIP columns the prior stage
already computed) and writes ``features/features.npz``. The prior stage calls
it once its packs are written; it can also be run on a finished run:

    python -m sceneagent.feature_lift --out RUN [--dest DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from sceneagent.mlp.contract import CLIP_DIM, DINO_DIM

FEATURE_DIM = DINO_DIM + CLIP_DIM


def lift_object_vectors(assignments, dino, clip):
    """Repeat each object's vectors onto its Gaussians.

    ``assignments`` is one integer per Gaussian. ``-1`` is unassigned.
    ``dino`` is ``(K, 384)`` and ``clip`` is ``(K, 512)``, one row per object.
    The result is ``(N, 896)``: DINOv2, then OpenCLIP. Unassigned rows are NaN.
    """
    assigned = np.asarray(assignments, dtype=np.int64)
    if assigned.ndim != 1:
        raise ValueError("assignments must be one index per Gaussian")
    dino_rows = np.asarray(dino, dtype=np.float64)
    clip_rows = np.asarray(clip, dtype=np.float64)
    if dino_rows.ndim != 2 or dino_rows.shape[1] != DINO_DIM:
        raise ValueError(f"dino must have shape (K, {DINO_DIM})")
    if clip_rows.shape != (dino_rows.shape[0], CLIP_DIM):
        raise ValueError(f"clip must have shape (K, {CLIP_DIM}) with the same K")
    objects = dino_rows.shape[0]
    if np.any((assigned < -1) | (assigned >= objects)):
        raise ValueError("an assignment index is outside the object table")
    features = np.full((assigned.shape[0], FEATURE_DIM), np.nan, dtype=np.float64)
    known = assigned >= 0
    features[known, :DINO_DIM] = dino_rows[assigned[known]]
    features[known, DINO_DIM:] = clip_rows[assigned[known]]
    return features


def shape_line(features) -> str:
    """One line naming the array shape, for a run log."""
    array = np.asarray(features)
    if array.ndim != 2 or array.shape[1] != FEATURE_DIM:
        raise ValueError(f"features must have shape (N, {FEATURE_DIM})")
    return f"features shape {array.shape[0]} x {FEATURE_DIM}"


def write_feature_file(path, features) -> Path:
    """Write ``features.npz`` with array ``features`` of shape ``(N, 896)``."""
    array = np.asarray(features, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != FEATURE_DIM:
        raise ValueError(f"features must have shape (N, {FEATURE_DIM})")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, features=array)
    return destination


FEATURE_DIR = "features"
FEATURE_FILE = "features.npz"
EVIDENCE = "features_evidence.json"


def object_vectors(pack) -> tuple[np.ndarray, np.ndarray]:
    """One object's DINOv2 and OpenCLIP vectors from its ``prior_input`` pack.

    The pack holds one row per Gaussian of the object's fitted splat, each
    with the patch tokens of the views that see it. The object's vector is
    their mean, ignoring non-finite entries: a pooling of vectors the prior
    stage already computed, not a new encoding.
    """
    rows = np.asarray(pack, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] < FEATURE_DIM or rows.shape[0] == 0:
        raise ValueError(f"a pack must have shape (M, >= {FEATURE_DIM})")
    head = rows[:, :FEATURE_DIM]
    finite = np.isfinite(head)
    if not finite.any(axis=0).all():
        raise ValueError("a feature column has no finite value")
    mean = np.where(finite, head, 0.0).sum(axis=0) / finite.sum(axis=0)
    return mean[:DINO_DIM], mean[DINO_DIM:]


def assignments_from_sets(sets: dict[int, Any], count: int, table: list[int]):
    """One object row per Gaussian from the association's index sets.

    ``table`` lists the objects that have vectors, in row order. A Gaussian in
    no set, in the set of an object without vectors, or in two sets stays
    ``-1``: the lift never picks between objects.
    """
    owner = np.full(int(count), -1, dtype=np.int64)
    hits = np.zeros(int(count), dtype=np.int64)
    row_of = {number: row for row, number in enumerate(table)}
    for number, ids in sets.items():
        index = np.asarray(ids, dtype=np.int64)
        if index.size and (index.min() < 0 or index.max() >= count):
            raise ValueError(f"object_{number} names a Gaussian outside 0..{count - 1}")
        hits[index] += 1
        if number in row_of:
            owner[index] = row_of[number]
    owner[hits > 1] = -1
    return owner, int((hits > 1).sum())


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_feature_lift(out: Path | str, dest: Path | str | None = None) -> dict[str, Any]:
    """Lift a run's object vectors onto its scene Gaussians; return the evidence.

    Reads ``association/objects.npz``, ``association_evidence.json`` (the
    splat's Gaussian count) and ``prior_input/object_<n>.npy``. Writes
    ``features/features.npz`` and ``features_evidence.json`` under ``dest``
    (default: the run itself). Raises when there is nothing to lift.
    """
    out = Path(out)
    dest = Path(dest) if dest is not None else out
    association = json.loads(
        (out / "association_evidence.json").read_text(encoding="utf-8")
    )
    count = int(association["inputs"]["positions"])
    sets_path = out / "association" / "objects.npz"
    with np.load(sets_path) as archive:
        sets = {int(k.split("_")[1]): archive[k] for k in archive.files}
    table: list[int] = []
    dino, clip, packs = [], [], {}
    for number in sorted(sets):
        pack_path = out / "prior_input" / f"object_{number}.npy"
        if not pack_path.is_file():
            continue
        d, c = object_vectors(np.load(pack_path))
        table.append(number)
        dino.append(d)
        clip.append(c)
        packs[f"object_{number}"] = _sha256(pack_path)
    if not table:
        raise ValueError("no object has a prior_input pack to lift")
    owner, shared = assignments_from_sets(sets, count, table)
    features = lift_object_vectors(owner, np.stack(dino), np.stack(clip))
    path = write_feature_file(dest / FEATURE_DIR / FEATURE_FILE, features)
    evidence = {
        "shape_line": shape_line(features),
        "features": str(path),
        "columns": {"dinov2": [0, DINO_DIM], "openclip": [DINO_DIM, FEATURE_DIM]},
        "object_vector": "mean of the prior_input pack's DINOv2 and OpenCLIP columns",
        "objects": {
            f"object_{n}": int((owner == row).sum()) for row, n in enumerate(table)
        },
        "objects_without_pack": [f"object_{n}" for n in sorted(sets) if n not in table],
        "gaussians": count,
        "assigned": int((owner >= 0).sum()),
        "unassigned_nan": int((owner < 0).sum()),
        "in_two_or_more_sets": shared,
        "inputs": {
            "association": str(sets_path),
            "association_sha256": _sha256(sets_path),
            "prior_input_sha256": packs,
        },
    }
    (dest / EVIDENCE).write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="a run directory")
    parser.add_argument("--dest", type=Path, default=None, help="default: the run")
    args = parser.parse_args(argv)
    try:
        evidence = run_feature_lift(args.out, args.dest)
    except (OSError, ValueError, KeyError) as exc:
        print(f"feature lift failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(evidence["shape_line"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
