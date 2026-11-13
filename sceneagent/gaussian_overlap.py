# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record how many consensus Gaussians two objects share.

Reads ``association/objects.npz`` and writes ``gaussian_overlap.json``.
It does not group masks, change a gate, or decide that one object is a
duplicate of another. The fraction is ``shared / min(size_a, size_b)``.

    python -m sceneagent.gaussian_overlap --out /workspace/sf4-smoke/mailbox
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# The Mailbox groups from run 43370600333. Measured, not classified.
PAIRS = ((2, 1), (0, 1), (0, 2))
OUTPUT = "gaussian_overlap.json"


def _share(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    common = int(np.intersect1d(left, right).size)
    size_a, size_b = int(left.size), int(right.size)
    smaller = min(size_a, size_b)
    return {
        "shared": common,
        "size_a": size_a,
        "size_b": size_b,
        "share_of_smaller": None if smaller == 0 else round(common / smaller, 6),
    }


def measure(out: Path) -> dict[str, Any]:
    """Shared-Gaussian fractions for the fixed pairs. Missing ids are recorded."""
    path = Path(out) / "association" / "objects.npz"
    document: dict[str, Any] = {"npz": str(path), "pairs": []}
    if not path.is_file():
        document["error"] = "missing objects.npz"
        return document
    sets = np.load(path)
    for a, b in PAIRS:
        key_a, key_b = f"object_{a}", f"object_{b}"
        row: dict[str, Any] = {"a": key_a, "b": key_b}
        if key_a not in sets.files or key_b not in sets.files:
            row["present"] = False
        else:
            row["present"] = True
            row.update(_share(np.asarray(sets[key_a]), np.asarray(sets[key_b])))
        document["pairs"].append(row)
    return document


def write_measure(out: Path | str) -> dict[str, Any]:
    """Write the measure next to the run and return it. Always succeeds."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    document = measure(out)
    (out / OUTPUT).write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(write_measure(args.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
