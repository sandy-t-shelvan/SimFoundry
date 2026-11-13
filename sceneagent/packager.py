# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Assemble one scene folder from the files a run already wrote.

Paths in the manifest are relative to the package directory. A stage that
did not write its file is named under ``missing`` and is not invented.
``verify_package`` recomputes every hash.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

# Source path relative to the run, then path inside the package.
_FILES = (
    ("export/scene.usda", "scene.usda"),
    ("export/scene.xml", "scene.xml"),
    ("scene_state.json", "scene_state.json"),
    ("background/orientation_sweep.json", "background/orientation_sweep.json"),
    ("joints.json", "joints.json"),
    ("plan.json", "reports/plan.json"),
    ("ledger.json", "reports/ledger.json"),
    ("step_verdicts.json", "reports/step_verdicts.json"),
    ("inspection/isaac_lab.json", "reports/isaac_lab.json"),
)
# Named in ``missing`` when absent, even if other files were copied.
_REQUIRED = ("scene.usda", "scene.xml", "background", "joints", "scene_state.json")


def build_package(run_dir, out_dir) -> dict[str, Any]:
    """Copy the run's outputs into ``out_dir`` and write the manifest."""
    run = Path(run_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    present = {name: False for name in _REQUIRED}
    copied: list[str] = []
    for source_rel, dest_rel in _files(run):
        source = run / source_rel
        if not source.is_file():
            continue
        destination = out / dest_rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied.append(dest_rel)
        for name in _REQUIRED:
            if dest_rel == name or dest_rel.startswith(name + "/"):
                present[name] = True
    files = [_entry(out, relative) for relative in sorted(copied)]
    manifest = {
        "files": files,
        "missing": [name for name, ok in present.items() if not ok],
    }
    text = json.dumps(manifest, indent=2) + "\n"
    if str(run.resolve()) in text:
        raise ValueError("package manifest contains an absolute run path")
    (out / "package_manifest.json").write_text(text, encoding="utf-8")
    top = run / "package_manifest.json"
    if top.resolve() != (out / "package_manifest.json").resolve():
        top.write_text(text, encoding="utf-8")
    return manifest


def verify_package(directory) -> None:
    """Recompute every hash. Raise if a listed file is missing or changed."""
    root = Path(directory)
    manifest = json.loads((root / "package_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        relative = str(item["path"])
        if relative.startswith("/") or ".." in Path(relative).parts:
            raise ValueError(f"package path is not relative: {relative}")
        path = root / relative
        if not path.is_file():
            raise ValueError(f"missing {relative}")
        if path.stat().st_size != int(item["size"]):
            raise ValueError(f"size changed for {relative}")
        if _sha256(path) != item["sha256"]:
            raise ValueError(f"hash changed for {relative}")


def _files(run: Path):
    found = list(_FILES)
    sam = run / "sam3d"
    if sam.is_dir():
        for glb in sorted(sam.glob("object_*.glb")):
            found.append((str(glb.relative_to(run)), f"assets/{glb.stem}/mesh.glb"))
    physics = run / "physics"
    if physics.is_dir():
        for record in sorted(physics.glob("object_*.json")):
            found.append(
                (str(record.relative_to(run)), f"assets/{record.stem}/physics.json")
            )
    return found


def _entry(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    return {
        "path": relative,
        "size": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()
