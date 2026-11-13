# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Object vectors copied onto Gaussians. No encoder is loaded."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from sceneagent.feature_lift import (
    FEATURE_DIM,
    lift_object_vectors,
    shape_line,
    write_feature_file,
)

ROOT = Path(__file__).resolve().parents[2]


def test_the_lift_copies_each_object_vector_and_leaves_the_rest_empty():
    dino = np.arange(2 * 384, dtype=float).reshape(2, 384)
    clip = np.arange(2 * 512, dtype=float).reshape(2, 512) + 10
    features = lift_object_vectors([0, 0, 1, -1], dino, clip)
    assert features.shape == (4, FEATURE_DIM)
    assert shape_line(features) == "features shape 4 x 896"
    assert np.array_equal(features[0, :384], dino[0])
    assert np.array_equal(features[0, 384:], clip[0])
    assert np.array_equal(features[1], features[0])
    assert np.array_equal(features[2, :384], dino[1])
    assert np.isnan(features[3]).all()


def test_a_bad_width_or_index_is_refused():
    dino = np.zeros((1, 384))
    clip = np.zeros((1, 512))
    with pytest.raises(ValueError, match="384"):
        lift_object_vectors([0], np.zeros((1, 16)), clip)
    with pytest.raises(ValueError, match="outside"):
        lift_object_vectors([1], dino, clip)


def test_the_npz_round_trip_keeps_the_shape(tmp_path):
    features = lift_object_vectors([0], np.ones((1, 384)), np.full((1, 512), 2.0))
    path = write_feature_file(tmp_path / "features.npz", features)
    loaded = np.load(path)["features"]
    assert loaded.shape == (1, 896)
    assert shape_line(loaded) == "features shape 1 x 896"


def test_the_spec_refuses_membership_and_a_new_encoder():
    text = (ROOT / "docs" / "feature-lift.md").read_text(encoding="utf-8")
    assert "does not decide membership" in text
    assert "N_gaussians" in text or "896" in text
    source = inspect.getsource(__import__("sceneagent.feature_lift", fromlist=["*"]))
    assert "open_clip" not in source and "torch" not in source
