# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase 3 association on a synthetic splat with known answers: two cubes before a wall."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from sceneagent import bin_association as ba
from sceneagent.association import gaussian_belongs
from sceneagent.bin_run import SCENE, SOURCE

W, H = 640, 480
PINHOLE = [500.0, 500.0, 320.0, 240.0]
SCALE = 2.0
# A quarter turn about x plus an offset: the splat frame differs from COLMAP's.
TRANSFORM = np.array([[1.0, 0, 0, 0.3], [0, 0, -1.0, 0.1], [0, 1.0, 0, -0.2]])
VIEW_X = {"000001.png": 0.0, "000005.png": 0.3}


def _cube(center, side=0.1, n=26):
    axis = np.linspace(-side / 2, side / 2, n)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    return grid + np.asarray(center)


def _scene():
    """COLMAP-world points: cube A, cube B, and a wall behind both."""
    a = _cube((-0.15, 0.0, 2.0))
    b = _cube((0.15, 0.0, 2.0))
    xs, ys = np.meshgrid(np.linspace(-1.5, 1.5, 300), np.linspace(-1.0, 1.0, 200))
    wall = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, 3.0)], 1)
    world = np.concatenate([a, b, wall])
    labels = np.repeat([0, 1, 2], [len(a), len(b), len(wall)])
    return world, labels


def _pixels(world, view):
    cam = world - np.array([VIEW_X[view], 0.0, 0.0])
    u = PINHOLE[0] * cam[:, 0] / cam[:, 2] + PINHOLE[2]
    v = PINHOLE[1] * cam[:, 1] / cam[:, 2] + PINHOLE[3]
    return np.floor(u).astype(int), np.floor(v).astype(int)


def _write_colmap(model: Path):
    model.mkdir(parents=True)
    cams = struct.pack("<Q", 1) + struct.pack("<iiQQ", 1, 1, W, H)
    cams += struct.pack("<4d", *PINHOLE)
    (model / "cameras.bin").write_bytes(cams)
    blob = struct.pack("<Q", len(VIEW_X))
    for image_id, (name, x) in enumerate(VIEW_X.items(), 1):
        # Identity rotation; t = -R C, so a camera at (x, 0, 0) has t = (-x, 0, 0).
        blob += struct.pack("<i4d3di", image_id, 1, 0, 0, 0, -x, 0, 0, 1)
        blob += name.encode() + b"\x00" + struct.pack("<Q", 1)
        blob += struct.pack("<ddq", 1.0, 2.0, -1)
    (model / "images.bin").write_bytes(blob)


def _write_ply(path: Path, points: np.ndarray):
    path.parent.mkdir(parents=True)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float opacity\nend_header\n"
    )
    data = np.zeros(
        len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("o", "<f4")]
    )
    data["x"], data["y"], data["z"] = points.T
    path.write_bytes(header.encode() + data.tobytes())


def _fake_depth(metres_per_splat_unit=ba.PAPER_METRES_PER_UNIT, half=True):
    """A DA3 stand-in whose output encodes a known metres-per-unit.

    It writes what DA3METRIC-LARGE would for this scene if the splat unit
    were ``metres_per_splat_unit`` metres: the nearest surface's metric
    depth times 300 over the focal length at the processed resolution.
    """
    world, _ = _scene()
    factor = 0.5 if half else 1.0
    width_p, height_p = int(W * factor), int(H * factor)
    focal_p = PINHOLE[0] * factor
    calls = []

    def depth(images_dir, depth_dir):
        calls.append(images_dir)
        depth_dir.mkdir(parents=True, exist_ok=True)
        for view in VIEW_X:
            u, v = _pixels(world, view)
            col, row = (u * factor).astype(int), (v * factor).astype(int)
            ok = (col >= 0) & (col < width_p) & (row >= 0) & (row < height_p)
            nearest = np.full((height_p, width_p), 3.0)  # the wall's plane
            np.minimum.at(nearest, (row[ok], col[ok]), world[ok, 2])
            metres = nearest * SCALE * metres_per_splat_unit
            out = (metres * 300.0 / focal_p).astype(np.float32)
            np.savez_compressed(depth_dir / f"{Path(view).stem}.npz", depth=out)
        return {"model": "fake-da3"}

    depth.calls = calls
    return depth


def _run_dir(tmp_path: Path, scene_gt: dict | None = None):
    world, labels = _scene()
    splat = SCALE * (world @ TRANSFORM[:, :3].T + TRANSFORM[:, 3])
    out = tmp_path / "bin"
    _write_colmap(out / "sparse" / "0")
    _write_ply(out / "splat" / "export" / "splat.ply", splat)
    run = out / "splat" / "outputs" / "processed" / "splatfacto" / "t1"
    run.mkdir(parents=True)
    (run / "dataparser_transforms.json").write_text(
        json.dumps({"transform": TRANSFORM.tolist(), "scale": SCALE})
    )
    (out / "bin_evidence.json").write_text(
        json.dumps({"quality_bar": {"model": "0", "inputs": 13, "pass": True}})
    )
    (out / "splat_evidence.json").write_text(
        json.dumps({"ply": str(out / "splat" / "export" / "splat.ply")})
    )
    masks_dir = out / "masks"
    masks_dir.mkdir()
    detections = {}
    for view in VIEW_X:
        u, v = _pixels(world, view)
        stack = np.zeros((2, H, W), dtype=bool)
        for k in (0, 1):
            pick = labels == k
            stack[k, v[pick], u[pick]] = True
        np.savez_compressed(masks_dir / f"{Path(view).stem}.npz", masks=stack)
        detections[view] = [
            {"phrase": "box", "kept": True, "mask_index": 0},
            {"phrase": "can", "kept": True, "mask_index": 1},
            {"phrase": "bin", "kept": False, "reason": "box area"},
        ]
    (masks_dir / "detections.json").write_text(json.dumps(detections))
    entries = [
        {"item": item, "decision": "accept"}
        for item in (
            "benchmark scene",
            "COLMAP camera poses",
            "full-scene splat",
            "SAM masks",
        )
    ]
    (out / "ledger.json").write_text(
        json.dumps(
            {"scene_name": f"{SOURCE} {SCENE}", "source": SOURCE, "entries": entries}
        )
    )
    data = tmp_path / "data"
    if scene_gt is not None:
        gt_dir = data / "scenes" / SCENE
        gt_dir.mkdir(parents=True)
        (gt_dir / "scene_gt.json").write_text(json.dumps(scene_gt))
    return out, data, world, labels


def test_splat_frame_round_trips_to_colmap():
    world, _ = _scene()
    splat = SCALE * (world @ TRANSFORM[:, :3].T + TRANSFORM[:, 3])
    assert np.allclose(ba.splat_to_colmap(splat, TRANSFORM, SCALE), world)


def test_vectorised_front_test_matches_the_scalar_equation():
    rng = np.random.default_rng(0)
    z_hat = rng.uniform(0.1, 10, 500)
    depth = z_hat + rng.uniform(-0.1, 0.4, 500)
    limit = z_hat + np.maximum(ba.EPS_A, ba.EPS_R * z_hat)
    expected = [
        gaussian_belongs(True, d, zh) for d, zh in zip(depth, z_hat, strict=True)
    ]
    assert list(depth <= limit) == expected


def test_each_mask_keeps_its_cube_front_and_drops_the_back_and_the_wall(tmp_path):
    out, data, world, labels = _run_dir(tmp_path)
    result = ba.associate(out, data, depth=_fake_depth())
    assert result["views"]["000001.png"]["cell_px"] == 2  # round(480/192)
    by_view = {(m["view"], m["mask_index"]): m["gaussians"] for m in result["masks"]}
    for view in VIEW_X:
        for k in (0, 1):
            ids = by_view[(view, k)]
            assert ids.size and np.all(labels[ids] == k)
            kept = world[ids]
            assert np.isclose(kept[:, 2].min(), 1.95)  # the front face is in
            # Back-face points hidden behind the front face are 0.2 units
            # behind z_hat (> max(0.08, 0.03 z_hat) = 0.117) and are out. Points
            # on the silhouette's edge are not hidden: their cell's front is
            # the oblique side face, so they stay, as equation (1) says.
            center = -0.15 if k == 0 else 0.15
            cube = world[labels == k]
            hidden = (
                np.isclose(cube[:, 2], 2.05)
                & (np.abs(cube[:, 0] - center) < 0.04)
                & (np.abs(cube[:, 1]) < 0.04)
            )
            assert hidden.sum() > 100
            kept_rows = {tuple(r) for r in np.round(kept, 6)}
            assert not any(tuple(r) in kept_rows for r in np.round(cube[hidden], 6))


def test_two_views_of_two_cubes_group_into_two_objects(tmp_path):
    out, data, _, labels = _run_dir(tmp_path)
    result = ba.associate(out, data, depth=_fake_depth())
    assert len(result["objects"]) == 2
    for obj in result["objects"]:
        assert obj["views"] == 2
        assert len({m["phrase"] for m in obj["masks"]}) == 1
        assert len(set(labels[obj["ids"]])) == 1
    assert len(result["merged_pairs"]) == 2


def test_masks_in_one_view_never_merge_directly():
    positions = np.zeros((4, 3))
    shared = np.array([0, 1, 2, 3])
    masks = [{"view": "a", "gaussians": shared}, {"view": "a", "gaussians": shared}]
    components, pairs = ba.group_masks(masks, positions)
    assert components == [[0], [1]] and pairs == []


def test_metres_per_unit_from_metric_camera_centers():
    colmap = {"a": np.zeros(3), "b": np.array([2.0, 0, 0]), "c": np.array([0, 4.0, 0])}
    metric = {k: v * 0.25 for k, v in colmap.items()}
    # 0.25 m per COLMAP unit, and the splat is 2x COLMAP: 0.125 m per splat unit.
    assert ba.metres_per_unit(colmap, metric, 2.0) == pytest.approx(0.125)


def test_metric_centers_come_from_an_object_seen_in_every_view():
    pose = {"cam_R_m2c": np.eye(3).ravel().tolist(), "cam_t_m2c": [0, 0, 1000.0]}
    gt = {
        "1": [{"obj_id": 7, **pose}, {"obj_id": 3, **pose}],
        "5": [{"obj_id": 7, **pose}],
    }
    centers = ba.camera_centers_metric(gt, ["000001.png", "000005.png"])
    assert np.allclose(centers["000001.png"], [0, 0, -1.0])


def test_runner_writes_ledger_evidence_and_object_sets(tmp_path):
    out, data, _, _ = _run_dir(tmp_path)
    code = ba.run_bin_association(out, data, depth=_fake_depth())
    # Only 2 of the 13 views exist in this synthetic run, so it is rejected
    # while still writing the evidence.
    assert code == 1
    entry = json.loads((out / "ledger.json").read_text())["entries"][-1]
    assert entry["item"] == ba.ITEM and entry["decision"] == "reject"
    assert entry["reason"].startswith("2 of 13 views posed")
    evidence = json.loads((out / "association_evidence.json").read_text())
    assert evidence["objects"] == 2 and evidence["gt_objects"] == 20
    assert evidence["rule"]["grouping"]["e"].startswith("ours")
    sets = np.load(out / "association" / "objects.npz")
    assert sorted(sets.files) == ["object_0", "object_1"]


def test_runner_accepts_when_every_view_is_posed(tmp_path, monkeypatch):
    out, data, _, _ = _run_dir(tmp_path)
    monkeypatch.setattr(ba, "VIEWS", 2)
    assert ba.run_bin_association(out, data, depth=_fake_depth()) == 0
    entry = json.loads((out / "ledger.json").read_text())["entries"][-1]
    assert entry["decision"] == "accept"
    assert entry["objects"] == 2 and entry["multi_view_objects"] == 2


def test_missing_splat_is_a_reject_not_a_crash(tmp_path):
    out, data, _, _ = _run_dir(tmp_path)
    (out / "splat" / "export" / "splat.ply").unlink()
    assert ba.run_bin_association(out, data, depth=_fake_depth()) == 1
    entry = json.loads((out / "ledger.json").read_text())["entries"][-1]
    assert entry["decision"] == "reject" and "splat.ply" in entry["reason"]


def test_pair_outcomes_say_which_cut_kept_views_apart():
    pairs = [
        {
            "merged": True,
            "close": True,
            "overlap": True,
            "centroid_distance": 0.01,
            "centroid_limit": 0.1,
        },
        {
            "merged": False,
            "close": False,
            "overlap": True,
            "centroid_distance": 0.3,
            "centroid_limit": 0.1,
        },
        {
            "merged": False,
            "close": False,
            "overlap": True,
            "centroid_distance": 0.2,
            "centroid_limit": 0.05,
        },
        {
            "merged": False,
            "close": True,
            "overlap": False,
            "centroid_distance": 0.02,
            "centroid_limit": 0.1,
        },
    ]
    outcome = ba.pair_outcomes(pairs)
    assert outcome["merged"] == 1 and outcome["too_far"] == 2
    assert outcome["too_little_overlap"] == 1 and outcome["neither"] == 0
    assert outcome["too_far_distance_units"] == {"min": 0.2, "median": 0.3, "max": 0.3}
    assert outcome["too_far_limit_units_max"] == 0.1


def test_da3_scale_recovers_the_splat_unit_in_metres(tmp_path):
    out, data, _, _ = _run_dir(tmp_path)
    for truth in (0.2, 0.915):
        result = ba.associate(out, data, depth=_fake_depth(truth))
        unit = result["unit"]
        assert unit["metres_per_unit"] == pytest.approx(truth, rel=0.03)
        assert unit["to_paper_units"] == pytest.approx(truth / 0.915, rel=0.03)
        assert unit["rule"].startswith("ours") and unit["model"] == "fake-da3"


def test_a_tighter_unit_keeps_the_back_face_once_lengths_are_in_paper_units(tmp_path):
    # At 0.2 m per splat unit the cube's 0.2-unit depth is 4.4 cm, inside the
    # paper's 0.08-unit (7.3 cm) slack, so the hidden back face now belongs.
    out, data, world, labels = _run_dir(tmp_path)
    result = ba.associate(out, data, depth=_fake_depth(0.2))
    ids = next(m["gaussians"] for m in result["masks"] if m["view"] == "000001.png")
    assert np.isclose(world[ids, 2].max(), 2.05)


def test_cell_ratios_on_a_flat_wall():
    u = np.repeat(np.arange(0.5, 64, 1.0), 48)
    v = np.tile(np.arange(0.5, 48, 1.0), 64)
    z_units = np.full(u.size, 4.0)
    focal = 100.0
    # 32 x 24 processed image, focal 50 there; a wall 2 m away everywhere.
    stored = np.full((24, 32), 2.0 * 300.0 / 50.0, dtype=np.float32)
    ratios = ba.cell_ratios(u, v, z_units, stored, (64, 48), focal)
    assert ratios.size == (32 // 4) * (24 // 4)
    assert np.allclose(ratios, 0.5)


def test_a_failed_depth_worker_rejects_with_its_last_lines(tmp_path):
    from types import SimpleNamespace

    from sceneagent import bin_scale

    def runner(argv, **kwargs):
        assert argv[0] == bin_scale.DEPTH_PYTHON
        # Our own cache: 58301075's root cache holds an incomplete DA3 model.
        assert kwargs["env"]["HF_HUB_CACHE"] == bin_scale.HF_CACHE
        assert "HF_HUB_OFFLINE" not in kwargs["env"]
        return SimpleNamespace(
            returncode=1, stdout="", stderr="loading\nNo module named depth_anything_3"
        )

    out, data, _, _ = _run_dir(tmp_path)

    def depth(images, depth_dir):
        return bin_scale.run_depth(images, depth_dir, runner=runner)

    assert ba.run_bin_association(out, data, depth=depth) == 1
    entry = json.loads((out / "ledger.json").read_text())["entries"][-1]
    assert entry["decision"] == "reject"
    assert "No module named depth_anything_3" in entry["reason"]


def test_consensus_keeps_gaussians_two_views_agree_on():
    masks = [
        {
            "view": "a.png",
            "gaussians": np.array([1, 2, 3, 9]),
        },  # 9: a floater only a sees
        {"view": "a.png", "gaussians": np.array([4])},
        {"view": "b.png", "gaussians": np.array([1, 2, 3, 4])},
    ]
    kept, union = ba.consensus_gaussians(masks, 2)
    assert kept.tolist() == [1, 2, 3, 4] and union == 5
    # One view only: min(kappa, 1) = 1 keeps the union.
    kept, union = ba.consensus_gaussians(masks[:2], 2)
    assert kept.tolist() == [1, 2, 3, 4, 9] and union == 5
    assert ba.consensus_gaussians(masks, 1)[0].size == 5
