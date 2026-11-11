# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Incremental vs global mapping comparison with a fake COLMAP 4: steps, stats, pose accuracy."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sceneagent import mapper_compare as mc

pytest.importorskip("scipy")

BIN_VIEWS = [f"{i:06d}.png" for i in (1, 5, 9, 13)]
FRUITS = [f"frame_{i:05d}.jpg" for i in range(1, 7)]


def _world_cameras():
    """Camera-from-world poses (identity rotations) at distinct centers."""
    centers = {1: [0.0, 0, 0], 5: [0.3, 0, 0], 9: [0.0, 0.3, 0], 13: [0.2, 0.2, 0.1]}
    return {k: np.asarray(c) for k, c in centers.items()}


def _scene_gt():
    # One object at the world origin; camera k sees it at -C_k (identity R).
    return {
        str(k): [
            {
                "obj_id": 3,
                "cam_R_m2c": np.eye(3).ravel().tolist(),
                "cam_t_m2c": (-c * 1000).tolist(),
            }
        ]
        for k, c in _world_cameras().items()
    }


class FakeColmap:
    """Writes what each COLMAP 4 step would; ``fail`` names a mapper that exits 1."""

    def __init__(self, fail=None):
        self.fail, self.calls = fail, []

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        step = argv[1]
        arg = dict(zip(argv[2::2], argv[3::2], strict=False))
        if step == "feature_extractor":
            Path(arg["--database_path"]).write_text("db")
        if step in ("mapper", "global_mapper"):
            if step == self.fail:
                return SimpleNamespace(
                    returncode=1, stdout="", stderr="no initial pair"
                )
            (Path(arg["--output_path"]) / "0").mkdir()
        if step == "model_converter":
            out = Path(arg["--output_path"])
            names = sorted(
                p.name for p in Path(arg["--input_path"]).parents[1].glob("images/*")
            )
            lines = []
            for i, name in enumerate(names, 1):
                key = int(Path(name).stem) if name.endswith(".png") else i
                # COLMAP stores t = -R C; scale the scene by 3 to make a similarity.
                c = 3.0 * _world_cameras().get(key, np.array([i, 0.0, 0.0]))
                lines.append(f"{i} 1 0 0 0 {-c[0]} {-c[1]} {-c[2]} 1 {name}")
                lines.append("1.0 2.0 7")
            (out / "images.txt").write_text("# header\n" + "\n".join(lines) + "\n")
            points = [
                "1 0 0 1 255 255 255 0.5 1 0 2 0",
                "2 0 1 1 255 255 255 1.5 1 1 3 1 4 1",
            ]
            (out / "points3D.txt").write_text("# header\n" + "\n".join(points) + "\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")


def _dirs(tmp_path):
    out = tmp_path / "bin"
    (out / "images").mkdir(parents=True)
    for name in BIN_VIEWS:
        (out / "images" / name).write_bytes(b"png")
    fruits = tmp_path / "run_images"
    fruits.mkdir()
    for name in FRUITS:
        (fruits / name).write_bytes(b"jpg")
    (fruits / "000001.png").write_bytes(b"stray bin view")
    data = tmp_path / "data"
    (data / "scenes" / "000091").mkdir(parents=True)
    (data / "scenes" / "000091" / "scene_gt.json").write_text(json.dumps(_scene_gt()))
    return out, data, fruits


def test_every_variant_runs_on_the_same_binary_and_the_bin_scores_its_poses(tmp_path):
    out, data, fruits = _dirs(tmp_path)
    fake = FakeColmap()
    assert mc.run_mapper_compare(out, data, runner=fake, fruits_frames=fruits) == 0
    assert {argv[0] for argv in fake.calls} == {mc.COLMAP4}
    evidence = json.loads((out / "mapper_compare.json").read_text())
    assert evidence["inputs"] == {
        "bin": 4,
        "fruits": 6,
    }  # the stray PNG is not a Fruits frame
    variants = evidence["variants"]
    assert [(v["dataset"], v["matcher"], v["mapper"]) for v in variants] == list(
        mc.VARIANTS
    )
    bin_runs = [v for v in variants if v["dataset"] == "bin"]
    for v in bin_runs:
        assert v["registered"] == 4 and v["points"] == 2
        assert v["mean_reproj_err_px"] == pytest.approx(1.0)
        assert v["mean_obs_per_image"] == pytest.approx(5 / 4)
        assert v["pose_accuracy"]["camera_rms_m"] < 1e-9
    assert all("pose_accuracy" not in v for v in variants if v["dataset"] == "fruits")
    # Matching runs once per (dataset, matcher), not once per mapper.
    matches = [argv for argv in fake.calls if argv[1].endswith("_matcher")]
    assert len(matches) == 3
    entry = json.loads((out / "ledger.json").read_text())["entries"][-1]
    assert entry == {
        "item": "mapper comparison",
        "decision": "accept",
        "use_gpu": True,
        "variants": 5,
    }


def test_a_failed_mapper_is_recorded_and_the_rest_still_run(tmp_path):
    out, data, fruits = _dirs(tmp_path)
    assert (
        mc.run_mapper_compare(
            out, data, runner=FakeColmap(fail="global_mapper"), fruits_frames=fruits
        )
        == 1
    )
    variants = json.loads((out / "mapper_compare.json").read_text())["variants"]
    assert [v.get("error") is None for v in variants] == [
        True,
        False,
        True,
        False,
        False,
    ]
    entry = json.loads((out / "ledger.json").read_text())["entries"][-1]
    assert entry["reason"].startswith(
        "3 variant(s) failed; first: bin exhaustive_matcher global_mapper"
    )


def test_sequential_matching_keeps_the_five_frame_window():
    argv = mc.match_command(Path("db"), "sequential_matcher")
    assert argv[-2:] == ["--SequentialMatching.overlap", "5"]
    assert "--SequentialMatching.overlap" not in mc.match_command(
        Path("db"), "exhaustive_matcher"
    )


def test_mapper_is_a_bin_pipeline():
    from sceneagent.vast_smoke import (
        FETCH_NAMES,
        PIPELINES,
        REMOTE_BIN_OUT,
        build_remote_command,
        remote_out,
    )

    argv = build_remote_command("/workspace/r", "/workspace/o", pipeline="mapper")
    assert argv[1:] == ["-m", "sceneagent.mapper_compare", "--out", "/workspace/o"]
    assert remote_out("mapper") == REMOTE_BIN_OUT
    assert PIPELINES["mapper"][0] == ("mapper comparison",)
    assert "mapper_compare.json" in FETCH_NAMES
