# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KAN-52: COLMAP flags for 3.11.1, stale pose state, frame selection, diagnosis."""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import numpy as np
import pytest

from sceneagent.bin_run import clear_pose_state
from sceneagent.colmap_stage import MAPPER_LADDER, mapper_ladder, pose_commands
from sceneagent.focal_prior import estimate_focal, self_calibrate
from sceneagent.pose_diagnosis import (
    camera_is_plausible,
    diagnose,
    match_stats,
    read_cameras_bin,
)
from sceneagent.run4_sampling import (
    LONG_SIDE_FILTER,
    bin_count,
    decode_all_command,
    extract_selected_command,
    select_sharpest_per_bin,
)

# Option names this tree passes, each read from ``colmap <command> -h`` of
# COLMAP 3.11.1 (commit 682ea9a, CUDA build) on box 60254439, 2026-11-13.
# 3.11.1 refuses an unknown option ("Failed to parse options - unrecognised
# option '--FeatureMatching.use_gpu'", exit 1), so a renamed flag fails the
# step rather than being ignored; this list keeps the names honest offline.
COLMAP_3_11_OPTIONS = {
    "feature_extractor": {
        "--log_to_stderr",
        "--database_path",
        "--image_path",
        "--ImageReader.single_camera",
        "--SiftExtraction.use_gpu",
        "--SiftExtraction.max_image_size",
    },
    "sequential_matcher": {
        "--log_to_stderr",
        "--database_path",
        "--SiftMatching.use_gpu",
        "--SequentialMatching.overlap",
    },
    "exhaustive_matcher": {"--log_to_stderr", "--database_path", "--SiftMatching.use_gpu"},
    "mapper": {
        "--log_to_stderr",
        "--database_path",
        "--image_path",
        "--output_path",
        "--Mapper.init_min_num_inliers",
        "--Mapper.init_min_tri_angle",
    },
}


def _flags(argv):
    return {str(a) for a in argv if str(a).startswith("--")}


@pytest.mark.parametrize("matcher", ["sequential_matcher", "exhaustive_matcher"])
def test_every_flag_for_colmap_3_is_a_3_11_1_option(matcher):
    commands = pose_commands("db", "images", "sparse", True, major=3, matcher=matcher)
    for argv in commands:
        assert _flags(argv) <= COLMAP_3_11_OPTIONS[argv[1]], argv[1]
    for _rung, flags in mapper_ladder(matcher):
        assert _flags(flags) <= COLMAP_3_11_OPTIONS["mapper"]


def test_the_sift_device_flag_follows_the_banner_on_both_sift_steps():
    for use_gpu, value in ((True, "1"), (False, "0")):
        extract, match, _mapper = pose_commands("db", "i", "s", use_gpu, major=3)
        assert extract[extract.index("--SiftExtraction.use_gpu") + 1] == value
        assert match[match.index("--SiftMatching.use_gpu") + 1] == value


def test_colmap_4_uses_the_renamed_feature_options():
    extract, match, _mapper = pose_commands("db", "i", "s", True, major=4)
    assert "--FeatureExtraction.use_gpu" in extract
    assert "--FeatureMatching.use_gpu" in match
    assert "--SiftExtraction.use_gpu" not in extract
    assert "--log_to_stderr" not in extract


def test_the_ladder_relaxes_the_initial_pair_and_never_holds_intrinsics():
    assert mapper_ladder("sequential_matcher") is MAPPER_LADDER
    assert mapper_ladder("exhaustive_matcher") is MAPPER_LADDER
    assert [r for r, _ in MAPPER_LADDER] == ["colmap_default", "relaxed_init", "relaxed_init_more"]
    for _rung, flags in MAPPER_LADDER:
        assert not any("refine" in flag for flag in flags)
    with pytest.raises(ValueError):
        mapper_ladder("vocab_tree_matcher")


def test_clear_pose_state_leaves_nothing_a_rerun_could_reuse(tmp_path):
    for name in ("images", "sparse/0", "reconstruction_gate"):
        (tmp_path / name).mkdir(parents=True)
    (tmp_path / "images" / "0001.jpg").write_bytes(b"x")
    (tmp_path / "sparse" / "0" / "images.bin").write_bytes(b"x")
    stale = (
        "database.db",
        "database.db-wal",
        "database.db-shm",
        "colmap_poses.json",
        "run4_evidence.json",
    )
    for name in stale:
        (tmp_path / name).write_text("old")
    (tmp_path / "scene.json").write_text("{}")
    clear_pose_state(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["scene.json"]
    clear_pose_state(tmp_path)  # a second clear on an empty dir is fine


def _frames(count, fps=24.0, jitter=0.0):
    return [
        {"index": i, "time": i / fps + (jitter if i % 7 == 3 else 0.0), "score": float(i % 5)}
        for i in range(count)
    ]


def test_a_clip_with_fewer_frames_than_bins_keeps_every_frame():
    # 10 s at 24 fps is 240 frames; a jittered timestamp used to leave a bin
    # empty and fail the clip.
    frames = _frames(240, jitter=0.03)
    n = bin_count(len(frames))
    assert n == 240
    rows = select_sharpest_per_bin(frames, 10.0, n)
    assert [r["chosen_index"] for r in rows] == list(range(240))
    assert bin_count(1072) == 300
    with pytest.raises(ValueError):
        bin_count(0)


def test_a_long_clip_still_gets_300_sharpest_bins():
    frames = _frames(1440, fps=24.0)
    rows = select_sharpest_per_bin(frames, 60.0, bin_count(len(frames)))
    assert len(rows) == 300
    assert all(r["candidates"] >= 4 for r in rows)


def test_frames_are_decoded_and_staged_with_a_capped_long_side():
    decode = decode_all_command("ffmpeg", "v.mp4", "%06d.png")
    assert decode[decode.index("-vf") + 1] == LONG_SIDE_FILTER
    extract = extract_selected_command("ffmpeg", "v.mp4", [3, 9], "%04d.jpg")
    chain = extract[extract.index("-vf") + 1]
    assert chain.startswith("select=") and chain.endswith("," + LONG_SIDE_FILTER)
    assert "min(iw,1920)" in LONG_SIDE_FILTER and "min(ih,1920)" in LONG_SIDE_FILTER


def _camera_blob(model_id, width, height, params):
    return struct.pack("<Qiiqq", 1, 1, model_id, width, height) + struct.pack(
        f"<{len(params)}d", *params
    )


def test_cameras_bin_is_read_and_judged_by_colmaps_bogus_rule():
    # The live dead-end camera (k 1.66 > 1) and the refined one.
    dead = read_cameras_bin(_camera_blob(2, 1920, 1080, [18182.6, 960, 540, 1.6567]))[0]
    assert dead["model"] == "SIMPLE_RADIAL"
    assert dead["focal_px"] == [18182.6] and dead["extra"] == [1.6567]
    assert camera_is_plausible(dead) is False
    good = read_cameras_bin(_camera_blob(2, 1920, 1080, [2354.9, 960, 540, 0.0464]))[0]
    assert camera_is_plausible(good) is True
    opencv = read_cameras_bin(_camera_blob(4, 1440, 1920, [1500, 1490, 720, 960, 0.1, -0.2, 0, 0]))[0]
    assert opencv["focal_px"] == [1500, 1490] and len(opencv["extra"]) == 4
    assert camera_is_plausible(opencv) is True
    tiny = read_cameras_bin(_camera_blob(0, 1920, 1080, [100.0, 960, 540]))[0]
    assert camera_is_plausible(tiny) is False  # 100 / 1920 < 0.1
    with pytest.raises(ValueError):
        read_cameras_bin(_camera_blob(2, 1920, 1080, [1.0, 2.0]))


def _database(path: Path, keypoints, pairs):
    """``pairs``: {(i, j): (rows, config)} over 1-based image ids named in order."""
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE images (image_id INTEGER, name TEXT)")
    con.execute("CREATE TABLE keypoints (image_id INTEGER, rows INTEGER)")
    con.execute("CREATE TABLE two_view_geometries (pair_id INTEGER, rows INTEGER, config INTEGER)")
    for i, count in enumerate(keypoints, start=1):
        con.execute("INSERT INTO images VALUES (?, ?)", (i, f"{i:04d}.jpg"))
        con.execute("INSERT INTO keypoints VALUES (?, ?)", (i, count))
    for (a, b), (rows, config) in pairs.items():
        con.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?)",
            (a * 2147483647 + b, rows, config),
        )
    con.commit()
    con.close()
    return path


def _chain(n, rows=400, config=3, skip=()):
    return {(i, i + 1): (rows, config) for i in range(1, n) if i not in skip}


def test_diagnosis_names_parallax_when_neighbours_are_planar_or_panoramic(tmp_path):
    db = _database(tmp_path / "d.db", [4000] * 20, _chain(20, config=6))
    stats = match_stats(db)
    assert stats["planar_share"] == 1.0 and stats["connected_share"] == 1.0
    assert diagnose(db, []).startswith("not enough parallax")


def test_diagnosis_names_a_broken_video(tmp_path):
    db = _database(tmp_path / "d.db", [4000] * 20, _chain(20, skip={6, 13}))
    assert match_stats(db)["connected_share"] == pytest.approx(7 / 20)
    assert diagnose(db, []).startswith("the video falls apart")


def test_diagnosis_names_textureless_or_blurry_frames(tmp_path):
    db = _database(tmp_path / "d.db", [120] * 20, _chain(20, rows=20))
    assert diagnose(db, []).startswith("too few features")


def test_diagnosis_names_weak_neighbour_matches(tmp_path):
    db = _database(tmp_path / "d.db", [4000] * 20, _chain(20, rows=30))
    assert diagnose(db, []).startswith("neighbouring frames barely match")


def test_diagnosis_names_diverged_intrinsics_with_the_camera(tmp_path):
    db = _database(tmp_path / "d.db", [4000] * 20, _chain(20))
    camera = {"width": 1920, "height": 1080, "focal_px": [18182.6], "extra": [1.66]}
    cause = diagnose(db, [{"rung": "colmap_default", "camera": camera}])
    assert cause.startswith("camera intrinsics diverged during mapping (focal 18183 px")


def test_diagnosis_never_raises_on_a_missing_database(tmp_path):
    assert diagnose(tmp_path / "missing.db", []).startswith("match database unreadable")


def _rotation(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def _orbit_fundamentals(focal, width=1920, height=1080, count=60, seed=0):
    """F between neighbouring views of a camera orbiting a point 1 m away."""
    rng = np.random.default_rng(seed)
    k = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]])
    k_inv = np.linalg.inv(k)
    out = []
    for i in range(count):
        r = _rotation(rng.normal(size=3), np.radians(rng.uniform(2, 8)))
        t = rng.normal(size=3)
        t /= np.linalg.norm(t)
        t_cross = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
        f = k_inv.T @ (t_cross @ r) @ k_inv
        out.append(f / np.linalg.norm(f))
    return np.array(out), np.full(count, 200.0)


@pytest.mark.parametrize("focal", [874.0, 1663.0, 2600.0])
def test_the_focal_is_recovered_from_fundamental_matrices(focal):
    fundamentals, weights = _orbit_fundamentals(focal)
    record = estimate_focal(fundamentals, weights, 1920, 1080)
    assert record["ok"] is True
    assert record["focal_px"] == pytest.approx(focal, rel=0.01)
    assert record["cost"] < 1e-3


def test_too_few_pairs_or_an_edge_minimum_is_not_a_focal():
    fundamentals, weights = _orbit_fundamentals(1663.0, count=5)
    record = estimate_focal(fundamentals, weights, 1920, 1080)
    assert record["ok"] is False and "needs 20" in record["why"]
    fundamentals, weights = _orbit_fundamentals(100.0)  # far below 0.3 * 1920
    record = estimate_focal(fundamentals, weights, 1920, 1080)
    assert record["ok"] is False and "edge" in record["why"]


def _colmap_like_database(path, fundamentals, weights):
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE cameras (camera_id INTEGER, model INTEGER, width INTEGER,"
        " height INTEGER, params BLOB, prior_focal_length INTEGER)"
    )
    con.execute(
        "INSERT INTO cameras VALUES (1, 2, 1920, 1080, ?, 0)",
        (np.array([2304.0, 960.0, 540.0, 0.3]).tobytes(),),
    )
    con.execute("CREATE TABLE two_view_geometries (pair_id INTEGER, rows INTEGER, config INTEGER, F BLOB)")
    for i, (f, w) in enumerate(zip(fundamentals, weights, strict=True)):
        con.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, 3, ?)",
            (i, int(w), np.ascontiguousarray(f).tobytes()),
        )
    # A planar pair and a weak pair are ignored.
    con.execute("INSERT INTO two_view_geometries VALUES (998, 500, 6, ?)", (np.eye(3).tobytes(),))
    con.execute("INSERT INTO two_view_geometries VALUES (999, 10, 3, ?)", (np.eye(3).tobytes(),))
    con.commit()
    con.close()
    return path


def test_self_calibration_writes_the_focal_as_the_cameras_prior(tmp_path):
    fundamentals, weights = _orbit_fundamentals(1663.0)
    db = _colmap_like_database(tmp_path / "database.db", fundamentals, weights)
    record = self_calibrate(db)
    assert record["ok"] is True and record["applied"] is True
    assert record["pairs"] == 60
    con = sqlite3.connect(db)
    params, prior = con.execute("SELECT params, prior_focal_length FROM cameras").fetchone()
    values = np.frombuffer(params, dtype=np.float64)
    assert values[0] == pytest.approx(1663.0, rel=0.01)
    assert list(values[1:]) == [960.0, 540.0, 0.0]
    assert prior == 1


def test_self_calibration_leaves_the_camera_alone_when_it_cannot_estimate(tmp_path):
    fundamentals, weights = _orbit_fundamentals(1663.0, count=3)
    db = _colmap_like_database(tmp_path / "database.db", fundamentals, weights)
    record = self_calibrate(db)
    assert record["ok"] is False
    params, prior = sqlite3.connect(db).execute("SELECT params, prior_focal_length FROM cameras").fetchone()
    assert np.frombuffer(params, dtype=np.float64)[0] == 2304.0 and prior == 0
    assert self_calibrate(tmp_path / "missing.db")["ok"] is False
