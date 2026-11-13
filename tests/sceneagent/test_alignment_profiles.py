# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Held-out alignment profile. Iteration-1 and placement constants stay put."""

from __future__ import annotations

from pathlib import Path

import pytest

from sceneagent.alignment import (
    FRUITS_ICP_RESIDUAL,
    HELDOUT_COVERAGE_MIN,
    HELDOUT_GATE_LABEL,
    HELDOUT_NORMALIZED_MEDIAN_MAX,
    HELDOUT_P90_MAX,
    HELDOUT_SILHOUETTE_MIN,
    OURS_COVERAGE_MIN,
    OURS_NORMALIZED_MEDIAN_MAX,
    OURS_P90_MAX,
    OURS_SILHOUETTE_MIN,
    PLACEMENT_P90_MAX_M,
    PLACEMENT_SILHOUETTE_MIN,
    accept_alignment,
    ours_heldout_v1_thresholds,
)
from sceneagent.heldout_thresholds import derive, load_candidates

ROOT = Path(__file__).resolve().parents[2]


def test_ours_iteration_1_constants_are_unchanged():
    assert OURS_NORMALIZED_MEDIAN_MAX == 0.05
    assert OURS_P90_MAX == 0.25
    assert OURS_COVERAGE_MIN == 0.5
    assert OURS_SILHOUETTE_MIN == 0.5
    assert PLACEMENT_P90_MAX_M == 0.05
    assert PLACEMENT_SILHOUETTE_MIN == 0.5


def test_the_heldout_profile_is_selected_only_when_named():
    undecided = accept_alignment(FRUITS_ICP_RESIDUAL)
    assert undecided["decision"] == "undecided"
    assert "profile" not in undecided
    iteration = accept_alignment(FRUITS_ICP_RESIDUAL, "ours_iteration_1")
    assert iteration["gate"] == "ours_iteration_1"
    assert iteration["decision"] == "reject"
    assert "profile" not in iteration
    held = accept_alignment(FRUITS_ICP_RESIDUAL, profile=HELDOUT_GATE_LABEL)
    assert held["profile"] == "ours_heldout_v1"
    assert held["thresholds_published"] is False
    assert held["gate"] == "ours_heldout_v1"
    assert held["paper_published"] is False
    assert set(held["failed"]) == {"p90", "coverage", "silhouette"}
    with pytest.raises(ValueError, match="unknown alignment profile"):
        accept_alignment(FRUITS_ICP_RESIDUAL, profile="paper")


def test_the_profile_matches_a_fresh_derivation_and_the_doc_label():
    report = derive(load_candidates())
    profile = ours_heldout_v1_thresholds()
    assert profile["normalized_median"]["max"] == HELDOUT_NORMALIZED_MEDIAN_MAX
    assert profile["normalized_median"]["max"] == report["limits"]["normalized_median"]
    assert profile["p90"]["max"] == HELDOUT_P90_MAX == report["limits"]["p90"]
    assert (
        profile["coverage"]["min"]
        == HELDOUT_COVERAGE_MIN
        == report["limits"]["coverage"]
    )
    assert profile["silhouette"]["min"] == HELDOUT_SILHOUETTE_MIN
    assert profile["silhouette"]["min"] == report["limits"]["silhouette"]
    text = (ROOT / "docs" / "heldout-thresholds.md").read_text(encoding="utf-8")
    assert "ours, not published in the paper" in text
    assert str(HELDOUT_NORMALIZED_MEDIAN_MAX) in text
