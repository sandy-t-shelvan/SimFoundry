# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""soft_press fails fast when PhysGaussian is absent and does not install it."""

from __future__ import annotations

from sceneagent.soft_press_drive import MISSING, deps_ready
from sceneagent.vast_smoke import (
    FETCH_NAMES,
    PIPELINES,
    PUBLISHED_NAMES,
    build_remote_command,
)


def test_missing_physgaussian_is_a_clear_refusal():
    assert deps_ready(importer=_missing, root=__file__) is False
    assert "does not install" in MISSING


def test_soft_press_command_does_not_name_the_key():
    assert "soft_press" in PIPELINES
    argv = build_remote_command(
        "/w/repo", "/workspace/sf4-smoke/soft_press", pipeline="soft_press"
    )
    text = " ".join(argv).lower()
    assert "soft_press_drive.py" in text
    assert "gemini" not in text
    assert "pip" not in text
    for name in ("soft_press_params.json", "soft_press_status.txt"):
        assert name in FETCH_NAMES and name in PUBLISHED_NAMES


def _missing(name):
    raise ImportError(name)
