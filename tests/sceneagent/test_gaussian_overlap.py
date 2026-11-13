# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KAN-54: record shared Gaussians. No cutoff and no association change."""

from __future__ import annotations

import json

import numpy as np

from sceneagent import gaussian_overlap
from sceneagent.vast_smoke import (
    FETCH_NAMES,
    PUBLISHED_NAMES,
    build_remote_shell,
)


def test_share_of_the_smaller_set(tmp_path):
    folder = tmp_path / "association"
    folder.mkdir()
    np.savez(
        folder / "objects.npz",
        object_0=np.array([1, 2, 3], dtype=np.int32),
        object_1=np.array([1, 2, 9, 10], dtype=np.int32),
        object_2=np.array([1, 2, 3, 4], dtype=np.int32),
    )
    document = gaussian_overlap.write_measure(tmp_path)
    by_pair = {(row["a"], row["b"]): row for row in document["pairs"]}
    # object_2 and object_1 share 2 of 4 ids.
    assert by_pair[("object_2", "object_1")]["share_of_smaller"] == 0.5
    assert by_pair[("object_0", "object_1")]["shared"] == 2
    assert by_pair[("object_0", "object_2")]["share_of_smaller"] == 1.0
    saved = json.loads((tmp_path / "gaussian_overlap.json").read_text())
    assert saved["pairs"][0]["a"] == "object_2"
    assert "duplicate" not in json.dumps(saved)


def test_a_missing_set_is_recorded_and_the_command_exits_zero(tmp_path):
    document = gaussian_overlap.write_measure(tmp_path)
    assert document["error"] == "missing objects.npz"
    assert gaussian_overlap.main(["--out", str(tmp_path)]) == 0


def test_only_the_mailbox_shell_records_the_overlap():
    mailbox = build_remote_shell(
        "/workspace/r", "/workspace/sf4-smoke/mailbox", pipeline="mailbox"
    )
    other = build_remote_shell("/workspace/r", "/workspace/o", pipeline="putcupinbowl")
    assert "sceneagent.gaussian_overlap" in mailbox
    assert "sceneagent.gaussian_overlap" not in other
    assert "gaussian_overlap.json" in FETCH_NAMES
    assert "gaussian_overlap.json" in PUBLISHED_NAMES
