# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SF-43: background completion and orientation on synthetic splats (CPU)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from sceneagent import background_scene as bs
from sceneagent import infill

GREY = np.array([0.5, 0.5, 0.5])


def _tilt(deg_x: float, deg_y: float) -> np.ndarray:
    ax, ay = np.radians(deg_x), np.radians(deg_y)
    rx = np.array(
        [[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]]
    )
    ry = np.array(
        [[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]]
    )
    return ry @ rx


def _scene(tilt=(12.0, -7.0), offset=(0.3, -0.2, 1.5), seed=0):
    """A floor grid, a box standing on it (its footprint has no floor points),
    a wall at one side, and a few floaters. Returns world points, colours,
    the object's index set, and the rotation used."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(-1.0, 1.0001, 0.05), np.arange(-1.0, 1.0001, 0.05))
    floor = np.c_[xs.ravel(), ys.ravel(), np.zeros(xs.size)]
    under = (np.abs(floor[:, 0]) < 0.2) & (np.abs(floor[:, 1]) < 0.2)
    floor = floor[~under]
    floor[:, 2] += rng.normal(0, 0.002, len(floor))
    wall_y, wall_z = np.meshgrid(
        np.arange(-1.0, 1.0001, 0.1), np.arange(0.05, 0.5, 0.1)
    )
    wall = np.c_[np.full(wall_y.size, 1.05), wall_y.ravel(), wall_z.ravel()]
    bx, by, bz = np.meshgrid(
        *(np.linspace(-0.18, 0.18, 7),) * 2, np.linspace(0.01, 0.3, 7)
    )
    box = np.c_[bx.ravel(), by.ravel(), bz.ravel()]
    floaters = rng.uniform(-1, 1, (5, 3)) + np.array([0, 0, 2.0])
    points = np.vstack([floor, wall, box, floaters])
    colors = np.vstack(
        [
            np.tile([0.8, 0.2, 0.2], (len(floor), 1)),
            np.tile([0.1, 0.1, 0.9], (len(wall), 1)),
            np.tile([0.2, 0.9, 0.2], (len(box), 1)),
            np.tile([1.0, 1.0, 1.0], (len(floaters), 1)),
        ]
    )
    start = len(floor) + len(wall)
    box_set = np.arange(start, start + len(box))
    rot = _tilt(*tilt)
    world = points @ rot.T + np.asarray(offset)
    return world, colors, box_set, rot, int(under.sum())


def test_the_foreground_mask_is_the_union_of_object_sets():
    mask = bs.foreground_mask(6, [np.array([0, 2]), np.array([2, 5])])
    assert mask.tolist() == [True, False, True, False, False, True]
    with pytest.raises(ValueError):
        bs.foreground_mask(3, [np.array([3])])


def test_dc3_the_support_plane_is_the_floor_not_the_wall():
    world, _colors, box_set, rot, _ = _scene()
    fg = bs.foreground_mask(len(world), [box_set])
    plane = bs.fit_support_plane(world[~fg], world[fg])
    true_up = rot @ np.array([0.0, 0.0, 1.0])
    angle = np.degrees(np.arccos(np.clip(np.dot(plane["normal"], true_up), -1, 1)))
    assert angle < 1.0  # floor normal, pointing at the box
    assert plane["n_planes"] >= 2  # the wall was peeled too


def test_dc3_holes_are_filled_where_the_object_stood_and_nowhere_else():
    offset = (0.3, -0.2, 1.5)
    world, colors, box_set, rot, removed = _scene(offset=offset)
    fg = bs.foreground_mask(len(world), [box_set])
    plane = bs.fit_support_plane(world[~fg], world[fg])
    filled = bs.fill_support_holes(world[~fg], colors[~fg], plane, world[fg], cell=0.05)
    # The box hid a 0.4 m square of floor (8 x 8 samples). On a 0.05 m grid
    # rotated against the samples, the boundary cells still catch a floor
    # neighbour, so about 7 x 7 cells are empty. The box's own samples are
    # 0.06 m apart, which also checks the closing (without it: 6 x 6).
    assert removed == 64
    assert 7 * 7 - 4 <= filled["n_filled"] <= 8 * 8
    # Nowhere else: every new Gaussian sits inside the box footprint.
    local = (filled["positions"] - np.asarray(offset)) @ rot
    assert np.abs(local[:, :2]).max() < 0.2 + 0.05
    assert filled["n_filled"] == len(filled["positions"])
    # New Gaussians lie on the plane, inside the box footprint, in floor red.
    heights = (filled["positions"] - plane["origin"]) @ plane["normal"]
    assert np.abs(heights).max() < 1e-9
    assert np.allclose(filled["colors"], [0.8, 0.2, 0.2], atol=1 / 255 + 1e-9)
    assert filled["color_method"] == "nearest_plane_sample"
    assert filled["lama"] == "not_requested"


def test_no_object_means_no_hole():
    world, colors, box_set, _rot, _ = _scene()
    fg = bs.foreground_mask(len(world), [box_set])
    plane = bs.fit_support_plane(world[~fg], world[fg])
    filled = bs.fill_support_holes(
        world[~fg], colors[~fg], plane, np.zeros((0, 3)), cell=0.05
    )
    assert filled["n_filled"] == 0 and len(filled["positions"]) == 0


def test_dc3_alignment_puts_the_floor_at_z0_with_a_measured_residual():
    world, _colors, box_set, _rot, _ = _scene(tilt=(20.0, 9.0))
    fg = bs.foreground_mask(len(world), [box_set])
    plane = bs.fit_support_plane(world[~fg], world[fg])
    aligned = bs.gravity_alignment(plane, world[~fg][plane["inliers"]], world[fg])
    assert aligned["residual_tilt_deg"] < 0.5
    assert abs(aligned["plane_height_after"]) < 0.01
    assert aligned["foreground_above_share"] == 1.0  # the box stands on +Z
    assert 15.0 < aligned["tilt_deg"] < 30.0  # the frame was tilted before
    rot = aligned["transform"][:3, :3]
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(rot), 1.0)


def test_rotation_onto_z_handles_both_poles():
    assert np.allclose(bs.rotation_onto_z(np.array([0, 0, 1.0])), np.eye(3))
    down = bs.rotation_onto_z(np.array([0, 0, -1.0]))
    assert np.allclose(down @ np.array([0, 0, -1.0]), [0, 0, 1.0])


def test_the_sweep_hook_records_not_run_without_the_box_and_picks_the_least_drift():
    transform = np.eye(4)
    off = bs.refine_with_sweep(transform)
    assert off["sweep"] == "not_run" and "GPU box" in off["reason"]
    assert off["transform"] is transform
    # A stub settle that drifts least at +1 deg about X.
    target = bs._tilted(transform, 1.0, 0.0)
    on = bs.refine_with_sweep(transform, lambda t: float(np.abs(t - target).sum()))
    assert on["sweep"] == "run" and np.allclose(on["transform"], target)
    assert len(on["scores"]) == len(bs.SWEEP_TILTS_DEG) ** 2


def test_lama_is_recorded_and_the_plane_fill_stays_when_it_writes_nothing(
    tmp_path, monkeypatch
):
    world, colors, box_set, _rot, _ = _scene()
    fg = bs.foreground_mask(len(world), [box_set])
    plane = bs.fit_support_plane(world[~fg], world[fg])
    monkeypatch.setattr(
        infill,
        "lama_inpaint",
        lambda *a, **k: {"emitted": False, "artifact": None, "lama": "weights_absent"},
    )
    filled = bs.fill_support_holes(
        world[~fg],
        colors[~fg],
        plane,
        world[fg],
        use_lama=True,
        work_dir=tmp_path,
        cell=0.05,
    )
    assert filled["lama"] == "weights_absent"
    assert filled["color_method"] == "nearest_plane_sample"
    assert filled["n_filled"] > 0


def test_the_whole_stage_writes_its_record_and_renders(tmp_path):
    world, colors, box_set, _rot, _ = _scene()
    record = bs.complete_scene_background(world, colors, [box_set], tmp_path)
    saved = json.loads((tmp_path / "background_completion.json").read_text())
    assert saved == json.loads(json.dumps(record))
    assert record["n_foreground_removed"] == len(box_set)
    assert record["fill"]["n_filled"] > 0
    assert record["orientation"]["sweep"] == "not_run"
    assert record["orientation"]["residual_tilt_deg"] < 0.5
    for name in record["artifacts"]:
        assert (tmp_path / name).stat().st_size > 0
    with np.load(tmp_path / "background_filled.npz") as data:
        assert data["new_positions"].shape == (record["fill"]["n_filled"], 3)
        assert data["background_mask"].sum() == len(world) - len(box_set)


def test_a_3dgs_ply_is_read_with_its_dc_colour(tmp_path):
    rows = np.zeros(
        2, dtype=[(n, "<f4") for n in ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2")]
    )
    rows["x"] = [1, 2]
    rows["f_dc_0"] = [0.0, 1.0]
    path = tmp_path / "s.ply"
    header = (
        "ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
        + "".join(f"property float {n}\n" for n in rows.dtype.names)
        + "end_header\n"
    )
    path.write_bytes(header.encode() + rows.tobytes())
    xyz, rgb = bs.read_ply_xyz_rgb(path)
    assert xyz[:, 0].tolist() == [1.0, 2.0]
    assert np.allclose(rgb[0], GREY) and np.isclose(rgb[1, 0], 0.5 + bs.SH_C0)
