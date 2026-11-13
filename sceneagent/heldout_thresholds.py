# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Held-out cutoffs for the alignment gate. Ours, not published in the paper.

Candidates come from the checked-in GraspClutter6D bin score files. A
candidate is good when its F1 at 1 cm is above that run's median F1. The
split is by object id. Thresholds are chosen on the derivation half only.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from sceneagent.alignment import RESIDUAL_NAMES

SEED = 0
RESULTS = Path(__file__).resolve().parents[1] / "docs" / "results"
# Lower is better for the two distances. Higher is better for the two agreements.
HIGHER_IS_BETTER = {
    "normalized_median": False,
    "p90": False,
    "coverage": True,
    "silhouette": True,
}


def load_candidates(results=RESULTS):
    """One row per placed object that has residuals and a matched F1."""
    rows = []
    for path in sorted(Path(results).glob("bin_000091_score_*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        objects = (document.get("icp") or {}).get("objects") or []
        matches = (document.get("score") or {}).get("matches") or []
        by_object = {}
        for match in matches:
            delivered = str(match.get("delivered") or "")
            if not delivered.startswith("object_"):
                continue
            by_object[int(delivered.split("_", 1)[1])] = match
        scored = [float(match["f1"]) for match in by_object.values()]
        if not scored:
            continue
        median = float(np.median(scored))
        for obj in objects:
            number = obj.get("object")
            match = by_object.get(number)
            residuals = obj.get("residuals") or {}
            if match is None or any(
                residuals.get(name) is None for name in RESIDUAL_NAMES
            ):
                continue
            f1 = float(match["f1"])
            rows.append(
                {
                    "run": path.stem,
                    "object_id": int(number),
                    "normalized_median": float(residuals["normalized_median"]),
                    "p90": float(residuals["p90"]),
                    "coverage": float(residuals["coverage"]),
                    "silhouette": float(residuals["silhouette"]),
                    "f1": f1,
                    "chamfer_m": float(match["chamfer_m"]),
                    "good": f1 > median,
                }
            )
    return rows


def split_by_object(rows, seed=SEED):
    """Derivation and held-out halves. An object id is in only one half."""
    ids = sorted({int(row["object_id"]) for row in rows})
    if len(ids) < 2:
        raise ValueError("need at least two object ids to split")
    order = np.random.default_rng(seed).permutation(len(ids))
    half = len(ids) // 2
    derivation_ids = {ids[int(index)] for index in order[:half]}
    derivation = [row for row in rows if row["object_id"] in derivation_ids]
    heldout = [row for row in rows if row["object_id"] not in derivation_ids]
    return derivation, heldout, ids


def _balanced_accuracy(good, predicted):
    positive = int(np.sum(good))
    negative = len(good) - positive
    if positive == 0 or negative == 0:
        return None
    true_positive = int(np.sum(good & predicted))
    true_negative = int(np.sum(~good & ~predicted))
    return 0.5 * (true_positive / positive + true_negative / negative)


def _stricter(candidate, current, higher_is_better):
    if higher_is_better:
        return candidate > current
    return candidate < current


def best_threshold(rows, name):
    """Cutoff on this residual with the best balanced accuracy.

    A tie takes the stricter cutoff. Held-out rows are not read.
    """
    higher = HIGHER_IS_BETTER[name]
    good = np.array([bool(row["good"]) for row in rows])
    if good.size == 0 or len(set(good.tolist())) < 2:
        raise ValueError(f"{name} has no good and bad candidates to separate")
    chosen = None
    score = None
    for raw in sorted({float(row[name]) for row in rows}):
        if higher:
            predicted = np.array([float(row[name]) >= raw for row in rows])
        else:
            predicted = np.array([float(row[name]) <= raw for row in rows])
        accuracy = _balanced_accuracy(good, predicted)
        if accuracy is None:
            continue
        if (
            score is None
            or accuracy > score + 1e-12
            or (abs(accuracy - score) <= 1e-12 and _stricter(raw, chosen, higher))
        ):
            chosen = raw
            score = accuracy
    if chosen is None:
        raise ValueError(f"{name} produced no cutoff")
    return chosen, score


def derive(rows, seed=SEED):
    """Four cutoffs from the derivation half, then the held-out counts."""
    derivation, heldout, ids = split_by_object(rows, seed)
    limits = {}
    scores = {}
    for name in RESIDUAL_NAMES:
        cutoff, accuracy = best_threshold(derivation, name)
        limits[name] = cutoff
        scores[name] = accuracy
    report = evaluate(heldout, limits)
    report["derivation_gate"] = evaluate(derivation, limits)
    report.update(
        {
            "seed": seed,
            "objects": len(ids),
            "candidates": len(rows),
            "derivation_candidates": len(derivation),
            "heldout_candidates": len(heldout),
            "limits": limits,
            "derivation_balanced_accuracy": scores,
        }
    )
    return report


def evaluate(rows, limits):
    """Accept rate, precision, and recall of the four-way AND gate."""
    good = [bool(row["good"]) for row in rows]
    accepted = [passes(row, limits) for row in rows]
    true_positive = sum(ok and label for ok, label in zip(accepted, good, strict=True))
    n_accepted = sum(accepted)
    n_good = sum(good)
    return {
        "n": len(rows),
        "accepted": n_accepted,
        "good": n_good,
        "true_positive": true_positive,
        "accept_rate": n_accepted / len(rows) if rows else None,
        "precision": true_positive / n_accepted if n_accepted else None,
        "recall": true_positive / n_good if n_good else None,
    }


def passes(row, limits):
    return (
        float(row["normalized_median"]) <= float(limits["normalized_median"])
        and float(row["p90"]) <= float(limits["p90"])
        and float(row["coverage"]) >= float(limits["coverage"])
        and float(row["silhouette"]) >= float(limits["silhouette"])
    )


def write_csv(rows, path):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run",
        "object_id",
        "normalized_median",
        "p90",
        "coverage",
        "silhouette",
        "f1",
        "chamfer_m",
        "good",
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})
    return destination
