# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the held-out alignment cutoffs from the checked-in score files.

Writes ``docs/results/heldout_candidates.csv`` and prints the four cutoffs.
The held-out half is scored after the cutoffs are chosen. It does not move them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sceneagent.heldout_thresholds import RESULTS, derive, load_candidates, write_csv

CSV_PATH = RESULTS / "heldout_candidates.csv"


def main(argv=None):
    rows = load_candidates()
    if not rows:
        print("no labelled candidates", file=sys.stderr)
        return 1
    write_csv(rows, CSV_PATH)
    report = derive(rows)
    limits = report["limits"]
    print(
        json.dumps(
            {
                "csv": str(CSV_PATH),
                "objects": report["objects"],
                "candidates": report["candidates"],
                "derivation_candidates": report["derivation_candidates"],
                "heldout_candidates": report["heldout_candidates"],
                "normalized_median_max": limits["normalized_median"],
                "p90_max": limits["p90"],
                "coverage_min": limits["coverage"],
                "silhouette_min": limits["silhouette"],
                "heldout_accept_rate": report["accept_rate"],
                "heldout_precision": report["precision"],
                "heldout_recall": report["recall"],
                "heldout_accepted": report["accepted"],
                "heldout_good": report["good"],
                "heldout_true_positive": report["true_positive"],
                "heldout_n": report["n"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
