# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""soft_press fails fast when PhysGaussian is absent and does not install it."""

from __future__ import annotations

import json

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
    from sceneagent.vast_smoke import PHYSGAUSSIAN_PYTHON

    assert "soft_press" in PIPELINES
    argv = build_remote_command(
        "/w/repo", "/workspace/sf4-smoke/soft_press", pipeline="soft_press"
    )
    text = " ".join(argv).lower()
    assert "soft_press_drive.py" in text
    assert PHYSGAUSSIAN_PYTHON in argv
    assert "gemini" not in text
    assert "pip" not in text
    for name in (
        "soft_press_params.json",
        "soft_press_status.txt",
        "soft_press_frames.zip",
    ):
        assert name in FETCH_NAMES and name in PUBLISHED_NAMES


def _missing(name):
    raise ImportError(name)


def test_unresolved_elastic_does_not_start_the_solver(tmp_path, monkeypatch):
    import sceneagent.soft_press_drive as drive_mod

    monkeypatch.setattr(drive_mod, "deps_ready", lambda: True)
    params = tmp_path / "params.json"
    params.write_text(
        json.dumps(
            [
                {
                    "particle_mass": 5e-5,
                    "youngs_modulus_pa": None,
                    "poisson_ratio": None,
                }
            ]
        ),
        encoding="utf-8",
    )
    called = []
    code = drive_mod.drive(
        tmp_path / "out", params, runner=lambda record, out: called.append(record)
    )
    assert code == 1
    assert called == []
    status = (tmp_path / "out" / "soft_press_status.txt").read_text(encoding="utf-8")
    assert "does not invent" in status


def test_a_traced_pair_is_simulated_into_png_frames(tmp_path, monkeypatch):
    import sceneagent.soft_press_drive as drive_mod

    monkeypatch.setattr(drive_mod, "deps_ready", lambda: True)
    params = tmp_path / "params.json"
    params.write_text(
        json.dumps(
            [
                {
                    "particle_mass": 5e-5,
                    "youngs_modulus_pa": 1.2e5,
                    "poisson_ratio": 0.3,
                    "log": ["n_gaussians: 1000"],
                }
            ]
        ),
        encoding="utf-8",
    )
    seen = {}

    def runner(record, out):
        seen["E"] = record["youngs_modulus_pa"]
        seen["nu"] = record["poisson_ratio"]
        path = out / "soft_press_00.png"
        path.write_bytes(drive_mod.raster_png([[0.5, 0.5, 0.6], [0.2, 0.2, 0.3]]))
        return [path]

    code = drive_mod.drive(tmp_path / "out", params, runner=runner)
    assert code == 0
    assert seen == {"E": 1.2e5, "nu": 0.3}
    frame = (tmp_path / "out" / "soft_press_00.png").read_bytes()
    assert frame.startswith(b"\x89PNG\r\n\x1a\n")
    assert (tmp_path / "out" / "soft_press_frames.zip").is_file()
