# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SF-19 reviewer: the eleven views are drawn and measured, and the retry follows from them."""

from __future__ import annotations

import json

import numpy as np
import pytest

from sceneagent import bin_review as br
from sceneagent.reviewer_views import REQUIRED_VIEWS
from sceneagent.similarity_icp import apply_similarity

pytest.importorskip("scipy")
pytest.importorskip("PIL")


def _sphere(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(n, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points


def test_isolated_view_agrees_when_the_gaussians_cover_the_mesh():
    mesh = _sphere()
    image, iou = br.render_isolated(mesh, mesh, mesh[:2000], np.array([1.0, 0, 0]))
    assert image.shape == (br.TILE_PX, br.TILE_PX, 3)
    assert iou > 0.8
    _, off = br.render_isolated(
        mesh, mesh, mesh[:2000] + np.array([0, 3.0, 0]), np.array([1.0, 0, 0])
    )
    assert off < 0.1


def test_the_retry_follows_from_the_measured_views():
    assert br.choose_retry("accept", 0.1, 0.1, True) == "keep"
    assert br.choose_retry("reject", 0.9, 0.2, True) == "icp_flip"
    assert br.choose_retry("reject", 0.2, 0.8, True) == "sam3d_next_view"
    assert br.choose_retry("reject", 0.2, 0.8, False) == "keep"
    assert br.choose_retry("reject", 0.7, 0.8, True) == "keep"


def test_half_turns_keep_the_mesh_centre_in_place():
    rng = np.random.default_rng(1)
    source = rng.normal(size=(200, 3)) + np.array([0.3, -0.2, 0.1])
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotation = q * np.sign(np.linalg.det(q))
    similarity = {
        "scale": 1.7,
        "rotation": rotation.tolist(),
        "translation": [1.0, 2.0, 3.0],
    }
    centre = apply_similarity(
        source.mean(axis=0, keepdims=True), 1.7, rotation, [1.0, 2.0, 3.0]
    )
    for scale, r, t in br.flip_starts(similarity, source).values():
        moved = apply_similarity(source.mean(axis=0, keepdims=True), scale, r, t)
        assert np.allclose(moved, centre) and np.isclose(np.linalg.det(r), 1.0)
        assert not np.allclose(r, rotation)


def test_overlay_marks_the_outline_green_and_the_render_red():
    photo = np.full((40, 40, 3), 200, dtype=np.uint8)
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True
    rendered = np.zeros((40, 40), dtype=bool)
    rendered[18:22, 18:22] = True
    out = br.draw_overlay(photo, mask, rendered)
    assert tuple(out[10, 20]) == (40, 255, 40)
    assert out[20, 20, 0] > out[20, 20, 1]


def test_all_eleven_views_are_written_and_the_manifest_finds_them(tmp_path):
    views = {name: np.zeros((30, 40, 3), dtype=np.uint8) for name, _ in REQUIRED_VIEWS}
    sheet = br.save_views(views, tmp_path / "object_3", "object_3")
    assert sheet.is_file()
    from sceneagent.reviewer_views import write_view_manifest

    manifest = write_view_manifest(tmp_path / "object_3", object_name="object_3")
    assert manifest["status"] == "present" and manifest["missing"] == []
    assert (
        json.loads((tmp_path / "object_3" / "reviewer_views.json").read_text())[
            "status"
        ]
        == "present"
    )


def test_next_view_skips_the_one_already_used():
    association = {
        "object_list": [
            {
                "object": 0,
                "masks": [
                    {"view": "a.png", "mask_index": 0},
                    {"view": "b.png", "mask_index": 1},
                    {"view": "c.png", "mask_index": 0},
                ],
            },
            {"object": 1, "masks": [{"view": "a.png", "mask_index": 1}]},
        ]
    }
    detections = {
        "a.png": [
            {"kept": True, "mask_index": 0, "mask_area_px": 900},
            {"kept": True, "mask_index": 1, "mask_area_px": 50},
        ],
        "b.png": [{"kept": True, "mask_index": 1, "mask_area_px": 400}],
        "c.png": [{"kept": True, "mask_index": 0, "mask_area_px": 300}],
    }
    chosen = br.next_views(association, detections, {0: "a.png", 1: "a.png"})
    assert chosen == {0: {"view": "b.png", "mask_index": 1}}


def test_review_is_a_bin_pipeline():
    from sceneagent.vast_smoke import (
        FETCH_NAMES,
        PIPELINES,
        REMOTE_BIN_OUT,
        build_remote_command,
        remote_out,
    )

    argv = build_remote_command("/workspace/r", "/workspace/o", pipeline="review")
    assert argv[1:] == ["-m", "sceneagent.bin_review", "--out", "/workspace/o"]
    assert remote_out("review") == REMOTE_BIN_OUT
    assert PIPELINES["review"][0][-1] == "bin score (reviewed)"
    assert {"bin_review.json", "review_sheets.zip"} <= set(FETCH_NAMES)
