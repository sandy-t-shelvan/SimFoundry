# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep published gap counts and the CI dependency list tied to the tree.

``sceneagent.catalog.counts`` is the tally. ``to_markdown`` is what
``docs/GAP_MATRIX.md`` is regenerated from. The canvas block rows use the
same status words. A hand-edited header that lags a partial row is how
``17 / 10 / 15`` shipped after Articulate Anything moved.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from sceneagent.catalog import counts, to_markdown

ROOT = Path(__file__).resolve().parents[2]

LIVING_DOCS = (
    ROOT / "docs" / "GAP_MATRIX.md",
    ROOT / "docs" / "DESIGN.md",
    ROOT / "docs" / "EXPANSION_PLAN.md",
    ROOT / "docs" / "SDLC.md",
    ROOT / "docs" / "canvases" / "sceneagent-paper-gaps.canvas.tsx",
)

STALE_SPLITS = (
    "17/10/15",
    "17 / 10 / 15",
    "17 done, 10 partial, 15 missing",
)


def test_gap_matrix_is_the_catalog_markdown():
    tally = counts()
    text = (ROOT / "docs" / "GAP_MATRIX.md").read_text(encoding="utf-8")
    assert text == to_markdown()
    header = (
        f"Counts: {tally['done']} done, {tally['partial']} partial, "
        f"{tally['missing']} missing, {tally['total']} rows."
    )
    assert header in text
    assert tally["done"] + tally["partial"] + tally["missing"] == tally["total"]


def test_living_docs_do_not_keep_the_old_split():
    for path in LIVING_DOCS:
        text = path.read_text(encoding="utf-8")
        for needle in STALE_SPLITS:
            assert needle not in text, f"{needle} in {path.name}"
    for path in (ROOT / "docs").rglob("*"):
        if path.suffix not in {".md", ".tsx"} or not path.is_file():
            continue
        assert "17/10/15" not in path.read_text(encoding="utf-8"), path


def test_canvas_block_statuses_match_catalog_counts():
    tally = counts()
    canvas = (ROOT / "docs" / "canvases" / "sceneagent-paper-gaps.canvas.tsx").read_text(encoding="utf-8")
    block = canvas[canvas.index("const BLOCKS"):canvas.index("const PAPER_NODES")]
    found = re.findall(r'status: "(done|partial|missing)"', block)
    seen = Counter(found)
    assert seen["done"] == tally["done"]
    assert seen["partial"] == tally["partial"]
    assert seen["missing"] == tally["missing"]
    assert len(found) == tally["total"]
    slash = f"{tally['done']} / {tally['partial']} / {tally['missing']}"
    words = f"{tally['done']} done, {tally['partial']} partial, {tally['missing']} missing"
    assert slash in canvas
    assert words in canvas


def test_design_opening_counts_match_catalog():
    tally = counts()
    design = (ROOT / "docs" / "DESIGN.md").read_text(encoding="utf-8")
    sentence = (
        f"Current counts: {tally['done']} done, {tally['partial']} partial, "
        f"{tally['missing']} missing, {tally['total']} rows."
    )
    assert sentence in design
    assert "Current counts: 4 done, 11 partial, 23 missing, 38 rows." not in design


def test_expansion_plan_drops_the_named_stale_sentences():
    plan = (ROOT / "docs" / "EXPANSION_PLAN.md").read_text(encoding="utf-8")
    assert "after meshes exist. Not run." not in plan
    assert "accept or reject. Not run." not in plan
    section = plan.split("## 3.3 Object reconstruction", 1)[1].split("## 3.4", 1)[0]
    assert "SAM has not run" not in section
    assert "Missing. No splat." not in section
    assert "All missing" not in section


def _sceneagent_ci_pins():
    """Package pins installed by sceneagent CI, ignoring comments."""
    text = (ROOT / "requirements-sceneagent-ci.txt").read_text(encoding="utf-8")
    pins = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            pins.append(stripped)
    return pins


def test_sdlc_dependency_list_matches_sceneagent_ci():
    pins = _sceneagent_ci_pins()
    assert [pin.split("==", 1)[0] for pin in pins] == [
        "ruff",
        "mypy",
        "numpy",
        "pytest",
        "trimesh",
        "pillow",
    ]
    workflow = (ROOT / ".github" / "workflows" / "sceneagent-ci.yml").read_text(encoding="utf-8")
    assert workflow.count("python -m pip install -r requirements-sceneagent-ci.txt") == 3
    assert "pip install numpy pytest trimesh pillow" not in workflow
    sdlc = (ROOT / "docs" / "SDLC.md").read_text(encoding="utf-8")
    for pin in pins:
        assert pin in sdlc
    assert "numpy` and `pytest` only" not in sdlc
