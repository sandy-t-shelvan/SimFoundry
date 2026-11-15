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


def test_an_object_vector_is_the_mean_of_its_packs_finite_feature_columns():
    from sceneagent.feature_lift import object_vectors

    pack = np.zeros((3, 912))
    pack[:, 0] = [1.0, 3.0, np.nan]
    pack[:, 400] = [2.0, 4.0, 6.0]
    pack[:, 900] = 99.0  # geometry columns are not features
    dino, clip = object_vectors(pack)
    assert dino.shape == (384,) and clip.shape == (512,)
    assert dino[0] == 2.0 and clip[400 - 384] == 4.0
    pack[:, 1] = np.nan
    with pytest.raises(ValueError, match="no finite"):
        object_vectors(pack)


def test_a_gaussian_in_two_sets_or_without_a_vector_stays_unassigned():
    from sceneagent.feature_lift import assignments_from_sets

    sets = {0: [0, 1], 1: [1, 2], 2: [3]}
    owner, shared = assignments_from_sets(sets, 5, table=[0, 1])
    assert owner.tolist() == [0, -1, 1, -1, -1] and shared == 1
    with pytest.raises(ValueError, match="outside"):
        assignments_from_sets({0: [5]}, 5, table=[0])


def test_a_run_is_lifted_into_its_own_feature_file(tmp_path):
    import json

    from sceneagent.feature_lift import main, run_feature_lift

    run = tmp_path / "run"
    (run / "association").mkdir(parents=True)
    (run / "prior_input").mkdir()
    np.savez_compressed(
        run / "association" / "objects.npz",
        object_0=np.array([0, 2], np.int32),
        object_1=np.array([1], np.int32),
    )
    (run / "association_evidence.json").write_text(
        json.dumps({"inputs": {"positions": 4}})
    )
    np.save(run / "prior_input" / "object_0.npy", np.ones((5, 912)))
    evidence = run_feature_lift(run, tmp_path / "dest")
    assert evidence["shape_line"] == "features shape 4 x 896"
    assert evidence["objects"] == {"object_0": 2}
    assert evidence["objects_without_pack"] == ["object_1"]
    features = np.load(tmp_path / "dest" / "features" / "features.npz")["features"]
    assert np.isnan(features[[1, 3]]).all() and (features[[0, 2]] == 1.0).all()
    assert not (run / "features").exists()
    (run / "prior_input" / "object_0.npy").unlink()
    assert main(["--out", str(run)]) == 1
