# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU plane fit and hole fill. Images in these tests are written by the code."""

from __future__ import annotations

import builtins
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sceneagent.infill import (
    PlaneFitError,
    complete_background,
    fill_hole,
    fit_plane,
    plane_axes,
    plane_fill_from_association,
)

RED = (10, 20, 30)
BLUE = (200, 10, 10)
GREEN = (0, 180, 40)
WHITE = (255, 255, 255)


def _scene():
    """Four coplanar samples and a 4x5 hole whose pixels are known exactly.

    The +Z basis is world X, Y. Pixel (x, y) is the normalized coordinate
    times (width - 1, height - 1). The white sample projects into the hole
    and must not become a color source.
    """
    points = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.25, 1.0, 0.0],
        [0.5, 0.0, 0.0],
    ]
    colors = [RED, BLUE, GREEN, WHITE]
    hole = np.zeros((4, 5), dtype=bool)
    hole[0, 2] = True
    hole[1, 2] = True
    hole[2, 2] = True
    expected = np.zeros((4, 5, 3), dtype=np.uint8)
    expected[0, 0] = RED
    expected[0, 2] = RED
    expected[0, 4] = BLUE
    expected[1, 2] = RED
    expected[2, 2] = GREEN
    expected[3, 1] = GREEN
    return points, colors, hole, expected


def _unit_normal(vector):
    array = np.asarray(vector, dtype=np.float64)
    array = array / np.linalg.norm(array)
    axis = int(np.argmax(np.abs(array)))
    if array[axis] < 0.0:
        array = -array
    return array


def test_fit_plane_recovers_axis_aligned_and_tilted_planes():
    u_axis, v_axis = plane_axes([0.0, 0.0, 1.0])
    assert np.allclose(u_axis, [1.0, 0.0, 0.0])
    assert np.allclose(v_axis, [0.0, 1.0, 0.0])

    xs, ys = np.meshgrid(np.linspace(-1.0, 2.0, 5), np.linspace(0.0, 3.0, 4))
    flat = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, 1.5)])
    fitted = fit_plane(flat)
    assert np.allclose(fitted["normal"], [0.0, 0.0, 1.0], atol=1e-8)
    assert fitted["n_inliers"] == len(flat)
    assert fitted["residual_rms"] < 1e-8
    assert abs(fitted["origin"][2] - 1.5) < 1e-8

    xx, yy = np.meshgrid(np.linspace(0.0, 1.0, 6), np.linspace(0.0, 2.0, 5))
    tilted = np.column_stack(
        [xx.ravel(), yy.ravel(), 0.3 * xx.ravel() + 0.4 * yy.ravel() + 1.0]
    )
    true = _unit_normal([-0.3, -0.4, 1.0])
    got = fit_plane(tilted)
    assert float(np.dot(got["normal"], true)) > 1.0 - 1e-8
    assert got["n_inliers"] == len(tilted)
    assert got["residual_rms"] < 1e-8


def test_fit_plane_recovers_a_noisy_plane():
    rng = np.random.default_rng(0)
    xx, yy = np.meshgrid(np.linspace(0.0, 1.0, 10), np.linspace(0.0, 1.0, 10))
    true = _unit_normal([-0.3, -0.4, 1.0])
    base = np.column_stack(
        [xx.ravel(), yy.ravel(), 0.3 * xx.ravel() + 0.4 * yy.ravel() + 1.0]
    )
    noisy = base + rng.normal(0.0, 0.01, size=base.shape) * true
    fitted = fit_plane(noisy)
    assert float(np.dot(fitted["normal"], true)) > 0.9999
    assert fitted["n_inliers"] >= 95
    assert 0.0 < fitted["residual_rms"] < 0.015


def test_fit_plane_ignores_outliers():
    rng = np.random.default_rng(1)
    xs, ys = np.meshgrid(np.linspace(0.0, 1.0, 8), np.linspace(0.0, 1.0, 8))
    inliers = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])
    outliers = np.column_stack(
        [rng.uniform(0.0, 1.0, 16), rng.uniform(0.0, 1.0, 16), np.full(16, 5.0)]
    )
    points = np.vstack([inliers, outliers])
    fitted = fit_plane(points)
    assert float(np.dot(fitted["normal"], [0.0, 0.0, 1.0])) > 0.999
    assert fitted["n_inliers"] == len(inliers)
    assert int(fitted["inliers"].max()) < len(inliers)
    assert fitted["residual_rms"] < 1e-8


def test_fit_plane_raises_on_too_few_points():
    with pytest.raises(PlaneFitError, match="too few"):
        fit_plane([])
    with pytest.raises(PlaneFitError, match="too few"):
        fit_plane([[0.0, 0.0, 0.0]])
    with pytest.raises(PlaneFitError, match="too few"):
        fit_plane([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])


def test_fit_plane_raises_on_collinear_points():
    line = [[float(index), 2.0 * index, -float(index)] for index in range(8)]
    with pytest.raises(PlaneFitError, match="collinear"):
        fit_plane(line)
    with pytest.raises(PlaneFitError, match="collinear"):
        fit_plane([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    with pytest.raises(PlaneFitError, match="collinear"):
        fit_plane([[1.0, 2.0, 3.0]] * 5)
    with pytest.raises(PlaneFitError, match="finite"):
        fit_plane([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, float("nan")]])


def test_fill_hole_shape_and_nearest_source_colors():
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    known = np.zeros((4, 5), dtype=bool)
    image[0, 0] = RED
    image[0, 4] = BLUE
    image[3, 1] = GREEN
    known[0, 0] = True
    known[0, 4] = True
    known[3, 1] = True
    hole = np.zeros((4, 5), dtype=bool)
    hole[0, 2] = True
    hole[1, 2] = True
    hole[2, 2] = True
    # A color sitting in the hole is the removed object, not a source.
    image[1, 2] = WHITE
    known[1, 2] = True
    filled = fill_hole(image, known, hole)
    assert filled.shape == (4, 5, 3)
    assert filled.dtype == np.uint8
    assert tuple(filled[0, 2]) == RED
    assert tuple(filled[1, 2]) == RED
    assert tuple(filled[2, 2]) == GREEN
    assert tuple(filled[0, 0]) == RED
    assert tuple(filled[0, 4]) == BLUE
    assert tuple(filled[3, 1]) == GREEN
    assert tuple(filled[0, 1]) == (0, 0, 0)
    assert tuple(filled[2, 0]) == (0, 0, 0)
    empty = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(PlaneFitError, match="no plane-surface color"):
        fill_hole(empty, np.zeros((2, 2), dtype=bool), np.ones((2, 2), dtype=bool))


def test_complete_background_writes_rendered_hole_fill(tmp_path):
    points, colors, hole, expected = _scene()
    destination = tmp_path / "background.png"
    result = complete_background(
        points=points, colors=colors, hole=hole, out_path=destination
    )
    assert result["ok"] is True
    assert result["emitted"] is True
    assert result["forged"] is False
    assert result["method"] == "plane"
    assert result["plane_completion"] == "fitted_plane"
    assert result["lama"] == "not_requested"
    assert result["use_lama"] is False
    assert result["shape"] == [4, 5, 3]
    assert Path(result["artifact"]) == destination
    assert destination.is_file()
    assert destination.stat().st_size > 0
    decoded = np.asarray(Image.open(destination))
    assert decoded.shape == (4, 5, 3)
    assert np.array_equal(decoded, expected)
    assert tuple(result["plane"]["normal"]) == pytest.approx((0.0, 0.0, 1.0), abs=1e-8)
    assert result["plane"]["n_inliers"] == 4


def test_background_ledger_records_method_inputs_and_outputs(tmp_path):
    points, colors, hole, _expected = _scene()
    destination = tmp_path / "background.png"
    result = complete_background(
        points=points, colors=colors, hole=hole, out_path=destination
    )
    ledger_path = tmp_path / "ledger.json"
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert payload["section"] == "3.3"
    assert len(payload["entries"]) == 1
    entry = payload["entries"][0]
    assert entry == result["ledger"]
    assert entry["item"] == "background"
    assert entry["section"] == "3.3"
    assert entry["method"] == "plane"
    assert entry["decision"] == "accept"
    assert entry["inputs"]["n_points"] == 4
    assert entry["inputs"]["point_source"] == "points"
    assert entry["inputs"]["n_hole_pixels"] == 3
    assert entry["inputs"]["image_size"] == [4, 5]
    assert entry["inputs"]["use_lama"] is False
    assert entry["inputs"]["hole_image"] is None
    assert entry["outputs"]["artifact"] == str(destination)
    assert entry["outputs"]["width"] == 5
    assert entry["outputs"]["height"] == 4
    assert entry["outputs"]["n_inliers"] == 4
    assert entry["outputs"]["forged"] is False
    assert entry["outputs"]["plane_normal"] == pytest.approx([0.0, 0.0, 1.0], abs=1e-8)
    assert Path(entry["outputs"]["artifact"]).is_file()

    via_association = complete_background(
        association={
            "background_points": points,
            "colors": colors,
            "hole": hole.tolist(),
            "centers": [[9.0, 9.0, 9.0], [8.0, 8.0, 8.0], [7.0, 7.0, 7.0]],
        },
        out_path=tmp_path / "assoc.png",
    )
    assert via_association["method"] == "plane"
    assert (
        via_association["ledger"]["inputs"]["point_source"]
        == "association.background_points"
    )
    reread = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert len(reread["entries"]) == 2
    assert reread["entries"][0] == entry
    assert reread["entries"][1]["method"] == "plane"
    assert reread["entries"][1]["outputs"]["artifact"].endswith("assoc.png")


def test_complete_background_raises_on_degenerate_points_without_writing(tmp_path):
    hole = np.zeros((4, 4), dtype=bool)
    hole[1, 1] = True
    colors = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 1, 1]]
    destination = tmp_path / "bad.png"
    with pytest.raises(PlaneFitError, match="collinear"):
        complete_background(
            points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            colors=colors,
            hole=hole,
            out_path=destination,
        )
    assert not destination.exists()
    assert not (tmp_path / "ledger.json").exists()
    with pytest.raises(PlaneFitError, match="too few"):
        complete_background(
            points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            colors=[[1, 2, 3], [4, 5, 6]],
            hole=hole,
            out_path=destination,
        )
    assert not destination.exists()


def test_gaussian_centers_stay_on_the_placeholder_path(tmp_path):
    centers = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    placeholder = plane_fill_from_association(
        {"kept": 3, "total": 4, "centers": centers},
        tmp_path / "plane.png",
    )
    assert placeholder["kind"] == "plane_fill_placeholder"
    followed = complete_background(
        association={"centers": centers, "n_passed_equation_1": 3, "n_gaussians": 4},
        out_path=tmp_path / "out.png",
    )
    assert followed["plane_completion"] == "placeholder"
    assert followed.get("method") is None
    assert "ledger" not in followed


def test_lama_flag_defaults_off_and_is_not_imported(tmp_path, monkeypatch):
    signature = inspect.signature(complete_background)
    assert signature.parameters["use_lama"].default is False
    points, colors, hole, _expected = _scene()
    seen: list[str] = []
    real_import = builtins.__import__

    def guard(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".", 1)[0] == "simple_lama_inpainting":
            seen.append(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guard)
    result = complete_background(
        points=points, colors=colors, hole=hole, out_path=tmp_path / "background.png"
    )
    assert result["method"] == "plane"
    assert result["use_lama"] is False
    assert result["lama"] == "not_requested"
    assert seen == []


def test_requested_lama_imports_and_does_not_forge_when_weights_are_absent(
    tmp_path, monkeypatch
):
    points, colors, hole, expected = _scene()
    seen: list[str] = []
    real_import = builtins.__import__

    def guard(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".", 1)[0] == "simple_lama_inpainting":
            seen.append(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guard)
    destination = tmp_path / "inpaint.png"
    result = complete_background(
        points=points,
        colors=colors,
        hole=hole,
        out_path=destination,
        use_lama=True,
    )
    assert seen
    assert result["forged"] is False
    plane_path = tmp_path / "plane_background.png"
    assert plane_path.is_file()
    assert np.array_equal(np.asarray(Image.open(plane_path)), expected)
    entry = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))[
        "entries"
    ][0]
    assert entry["inputs"]["use_lama"] is True
    if result["method"] == "lama":
        assert result["emitted"] is True
        assert Path(result["artifact"]).is_file()
        assert entry["method"] == "lama"
        assert entry["outputs"]["plane_artifact"] == str(plane_path)
    else:
        assert result["method"] == "plane"
        assert result["lama"] == "weights_absent"
        assert result["artifact"] == str(plane_path)
        assert not destination.exists()
        assert entry["method"] == "plane"
        assert entry["outputs"]["artifact"] == str(plane_path)
        assert entry["outputs"]["lama"] == "weights_absent"
