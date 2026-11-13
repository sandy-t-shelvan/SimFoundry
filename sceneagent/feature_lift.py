# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attach existing object vectors to the Gaussians of that object.

The spec is ``docs/feature-lift.md``. This module does not run an encoder,
does not train, and does not decide which Gaussian belongs to which object.
The assignment array is an input.
"""

from __future__ import annotations

from pathlib import Path

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
