# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Isaac Lab render check's pure parts, and its gpu-smoke wiring."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

from sceneagent import isaaclab_headless_check as check
from sceneagent.vast_smoke import (
    FETCH_NAMES,
    ISAACLAB_PYTHON,
    PIPELINES,
    PUBLISHED_NAMES,
    REMOTE_ISAACLAB_OUT,
    build_remote_command,
    remote_out,
)


def test_png_is_a_valid_rgb_image():
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    rgb[1, 2] = (200, 10, 10)
    data = check.png_bytes(rgb)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    _length, kind = struct.unpack(">I4s", data[8:16])
    width, height, depth, color = struct.unpack(">IIBB", data[16:26])
    assert (kind, width, height, depth, color) == (b"IHDR", 5, 4, 8, 2)
    idat = data.index(b"IDAT")
    size = struct.unpack(">I", data[idat - 4 : idat])[0]
    raw = zlib.decompress(data[idat + 4 : idat + 4 + size])
    rows = [raw[r * 16 : (r + 1) * 16] for r in range(4)]
    assert all(row[0] == 0 for row in rows)
    assert rows[1][1 + 2 * 3 : 1 + 3 * 3] == bytes((200, 10, 10))


def test_judge_needs_a_resting_cube_that_is_visible():
    assert check.judge(0.0999, 0.05)[0] == "accept"
    # Fell through the ground, or never landed.
    assert check.judge(-0.5, 0.05)[0] == "reject"
    assert check.judge(0.5, 0.05)[0] == "reject"
    # A lit ground plane with no red cube in view.
    grey = np.full((8, 8, 3), 128, dtype=np.uint8)
    assert check.red_fraction(grey) == 0.0
    assert check.judge(0.1, check.red_fraction(grey))[0] == "reject"


def test_red_fraction_counts_only_red_pixels():
    rgb = np.full((10, 10, 3), 128, dtype=np.uint8)
    rgb[:2] = (220, 30, 30)
    assert check.red_fraction(rgb) == 0.2


def test_the_scene_uses_the_lab_3_spawn_api():
    text = Path(check.__file__).read_text(encoding="utf-8")
    assert "launch_simulation" in text
    assert "UsdPhysicsRigidBodyCfg" in text
    assert "UsdPhysicsCollisionCfg" in text
    assert "MassCfg(mass=1.0)" in text
    assert "root_pos_w.torch" in text
    assert "IsaacRtxRendererCfg" in text
    assert "RigidBodyPropertiesCfg" not in text
    assert "MassPropertiesCfg" not in text
    assert "CollisionPropertiesCfg" not in text
    assert "AppLauncher" not in text


def test_pipeline_runs_the_isaaclab_interpreter_in_its_own_dir():
    assert PIPELINES["isaaclab_render"][0] == (check.ITEM,)
    assert remote_out("isaaclab_render") == REMOTE_ISAACLAB_OUT
    argv = build_remote_command(
        "/w/repo", REMOTE_ISAACLAB_OUT, pipeline="isaaclab_render"
    )
    assert argv[:4] == ["env", "-u", "PYTHONPATH", ISAACLAB_PYTHON]
    assert argv[4:] == [
        "/w/repo/sceneagent/isaaclab_headless_check.py",
        "--render",
        REMOTE_ISAACLAB_OUT,
    ]
    for name in (check.PNG_NAME, check.LOG_NAME):
        assert name in FETCH_NAMES
        assert name in PUBLISHED_NAMES
