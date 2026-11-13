# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Package folder from a fake run. No simulator is started."""

from __future__ import annotations

import json

import pytest

from sceneagent.packager import build_package, verify_package


def test_a_partial_run_packages_and_names_what_is_missing(tmp_path):
    run = tmp_path / "run"
    (run / "export").mkdir(parents=True)
    (run / "export" / "scene.usda").write_text("#usda 1.0\n", encoding="utf-8")
    (run / "export" / "scene.xml").write_text("<mujoco/>\n", encoding="utf-8")
    (run / "sam3d").mkdir()
    (run / "sam3d" / "object_0.glb").write_bytes(b"glb")
    manifest = build_package(run, run / "package")
    names = [item["path"] for item in manifest["files"]]
    assert "scene.usda" in names
    assert "scene.xml" in names
    assert "assets/object_0/mesh.glb" in names
    assert "background" in manifest["missing"]
    assert "joints" in manifest["missing"]
    text = (run / "package" / "package_manifest.json").read_text(encoding="utf-8")
    assert str(run) not in text
    assert '"/' not in text
    verify_package(run / "package")
    mesh = run / "package" / "assets" / "object_0" / "mesh.glb"
    mesh.write_bytes(b"glX")
    with pytest.raises(ValueError, match="hash changed"):
        verify_package(run / "package")


def test_the_manifest_matches_the_bytes(tmp_path):
    run = tmp_path / "run"
    (run / "export").mkdir(parents=True)
    (run / "export" / "scene.usda").write_text("x", encoding="utf-8")
    build_package(run, tmp_path / "pkg")
    document = json.loads(
        (tmp_path / "pkg" / "package_manifest.json").read_text(encoding="utf-8")
    )
    usda = next(item for item in document["files"] if item["path"] == "scene.usda")
    assert usda["size"] == 1
    assert len(usda["sha256"]) == 64
