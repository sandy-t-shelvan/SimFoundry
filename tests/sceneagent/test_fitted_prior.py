# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SF-33: the fitted-splat prior input. No GPU, no network: the prior's driver is stubbed."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from sceneagent import fitted_prior as fp


def _pack(n: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    packed = rng.normal(size=(n, fp.PRIOR_WIDTH))
    packed[:, fp.OPACITY_COLUMN] = rng.uniform(0.2, 0.95, size=n)
    return packed


def test_the_layout_is_the_priors():
    assert (fp.DINO_DIM, fp.CLIP_DIM, fp.GEOM_DIM) == (384, 512, 16)
    assert fp.PRIOR_WIDTH == 912
    # x, y, z, three scales, then opacity: geometry column 6.
    assert fp.OPACITY_COLUMN == 896 + 6
    assert fp.DINO_ID == "vit_small_patch14_dinov2.lvd142m"
    assert fp.CLIP_ID == "ViT-B-16 openai"


def test_a_varied_pack_passes_and_reports_n_and_spread():
    checked = fp.validate_pack(_pack(80))
    assert checked["n"] == 80 and checked["opacity_std"] > 1e-3


def test_a_constant_opacity_pack_is_refused():
    constant = _pack()
    constant[:, fp.OPACITY_COLUMN] = 0.99
    with pytest.raises(fp.PackError, match="constant"):
        fp.validate_pack(constant)
    flat = _pack()
    flat[:, fp.OPACITY_COLUMN] = 0.5 + 1e-5 * np.arange(len(flat)) / len(flat)
    with pytest.raises(fp.PackError, match="constant"):
        fp.validate_pack(flat)


@pytest.mark.parametrize(
    "bad, match",
    [
        (np.zeros((64, 911)), "width"),
        (np.zeros(912), "rank 2"),
        (_pack(10), "only 10"),
    ],
)
def test_the_wrong_shape_is_refused(bad, match):
    with pytest.raises(fp.PackError, match=match):
        fp.validate_pack(bad)


def test_non_finite_values_and_a_raw_opacity_are_refused():
    broken = _pack()
    broken[3, 7] = np.nan
    with pytest.raises(fp.PackError, match="non-finite"):
        fp.validate_pack(broken)
    raw = _pack()
    raw[:, fp.OPACITY_COLUMN] = np.linspace(-3, 3, len(raw))  # logits, not sigmoid
    with pytest.raises(fp.PackError, match="outside"):
        fp.validate_pack(raw)


def _vertex_coloured_box(path: Path) -> np.ndarray:
    trimesh = pytest.importorskip("trimesh")
    box = trimesh.creation.box(extents=(1, 2, 3))
    rgb = np.random.default_rng(1).integers(0, 255, size=(len(box.vertices), 3))
    box.visual = trimesh.visual.ColorVisuals(
        box, vertex_colors=np.c_[rgb, np.full(len(rgb), 255)].astype(np.uint8)
    )
    box.export(path)
    return rgb


def test_vertex_colours_become_a_per_face_atlas_the_prior_can_sample(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("PIL")
    rgb = _vertex_coloured_box(tmp_path / "object_0.glb")
    out, note = fp.ensure_uv_texture(tmp_path / "object_0.glb", tmp_path / "t.glb")
    assert note == fp.ATLAS_NOTE and out == tmp_path / "t.glb"
    loaded = trimesh.load(out, force="scene", process=False)
    (mesh,) = list(loaded.geometry.values())
    uv = np.asarray(mesh.visual.uv)
    image = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB"))
    assert len(uv) == len(mesh.vertices)  # what load_textured_mesh requires
    # The prior's lookup: x = u (w - 1), y = (1 - v) (h - 1). Each corner of a
    # face lands on that face's pixel, holding the mean of its vertex colours.
    h, w = image.shape[:2]
    x = np.rint(uv[:, 0] * (w - 1)).astype(int)
    y = np.rint((1 - uv[:, 1]) * (h - 1)).astype(int)
    assert np.allclose(uv[:, 0] * (w - 1), x) and np.allclose(
        (1 - uv[:, 1]) * (h - 1), y
    )
    sampled = image[y, x].reshape(-1, 3, 3)
    assert (sampled == sampled[:, :1]).all()  # flat per face
    source = trimesh.load(tmp_path / "object_0.glb", force="mesh", process=False)
    expected = np.rint(rgb[np.asarray(source.faces)].mean(axis=1))
    assert np.abs(sampled[:, 0].astype(int) - expected).max() <= 1


def test_a_mesh_without_any_colour_is_refused(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    trimesh.creation.box().export(tmp_path / "plain.glb")
    with pytest.raises(ValueError, match="neither"):
        fp.ensure_uv_texture(tmp_path / "plain.glb", tmp_path / "t.glb")


def test_the_driver_calls_the_priors_own_conversion():
    compile(fp.DRIVER, "<driver>", "exec")
    assert (
        "fitted_splat as fs" in fp.DRIVER
        and "fs._convert_one(row, encoders)" in fp.DRIVER
    )
    assert "FeatureEncoders().load()" in fp.DRIVER
    env = fp.driver_env({"PATH": "/usr/bin", "PYTHONPATH": "/w/repo"}, Path("/p"))
    assert env["PYTHONPATH"] == "/p"  # our sceneagent must not shadow the prior's
    parts = env["PATH"].split(":")
    assert parts[0] == str(fp.COLMAP_WRAP)
    assert parts[2] == fp.COLMAP_BIN
    wrapper = fp.COLMAP_WRAP / "colmap"
    text = wrapper.read_text(encoding="utf-8")
    assert '[ "$arg" = "--SiftMatching.min_num_inliers" ]' in text
    assert 'arg="--TwoViewGeometry.min_num_inliers"' in text


class _Box:
    """Stands in for git and the prior's driver."""

    def __init__(self, commit: str, packs: dict[int, np.ndarray | None]):
        self.commit = commit
        self.packs = packs
        self.calls: list[list[str]] = []

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        self.cwd = kwargs.get("cwd")
        if argv[0] == "git":
            return subprocess.CompletedProcess(argv, 0, self.commit + "\n", "")
        jobs = json.loads(Path(argv[-1]).read_text(encoding="utf-8"))
        results = {
            "encoders": {
                "dino": {"id": fp.DINO_ID, "state_sha256": "a" * 64},
                "clip": {"id": fp.CLIP_ID, "state_sha256": "b" * 64},
            },
            "objects": [],
        }
        for job in jobs["objects"]:
            pack = self.packs.get(job["object"])
            if pack is None:
                results["objects"].append(
                    {
                        "object": job["object"],
                        "ok": False,
                        "error": "RuntimeError: only 3 views",
                    }
                )
                continue
            directory = Path(job["directory"])
            directory.mkdir(parents=True, exist_ok=True)
            np.save(directory / "prior_input.npy", pack)
            results["objects"].append(
                {
                    "object": job["object"],
                    "ok": True,
                    "summary": {
                        "prior_input": str(directory / "prior_input.npy"),
                        "orbit_views": 16,
                        "splat_iterations": 600,
                    },
                }
            )
        Path(jobs["results"]).write_text(json.dumps(results), encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, "", "")


def _bin(tmp_path: Path, numbers=(3, 5)) -> Path:
    out = tmp_path / "bin"
    (out / "sam3d").mkdir(parents=True)
    placed = []
    for n in numbers:
        glb = out / "sam3d" / f"object_{n}.glb"
        _vertex_coloured_box(glb)
        placed.append({"object": n, "placed": True, "gate": "accept", "glb": str(glb)})
    # Meshed, but the fit check rejected it: not an input to the prior.
    rejected = out / "sam3d" / "object_9.glb"
    _vertex_coloured_box(rejected)
    placed.append({"object": 9, "placed": True, "gate": "reject", "glb": str(rejected)})
    (out / "bin_score.json").write_text(json.dumps({"icp": {"objects": placed}}))
    return out


def test_every_meshed_object_is_packed_and_recorded(tmp_path):
    pytest.importorskip("trimesh")
    out = _bin(tmp_path)
    box = _Box(fp.PRIOR_COMMIT, {3: _pack(64, 3), 5: _pack(100, 5)})
    assert fp.run_fitted_prior(out, runner=box, prior_repo=tmp_path / "prior") == 0
    evidence = json.loads((out / "prior_input_ledger.json").read_text())
    assert evidence["prior"]["commit"] == fp.PRIOR_COMMIT
    assert evidence["encoders"]["clip"]["id"] == fp.CLIP_ID
    by = {o["object"]: o for o in evidence["objects"]}
    assert sorted(by) == [3, 5]  # object 9 had no mesh
    assert by[5]["n"] == 100 and by[5]["opacity_std"] > 1e-3
    assert (out / "prior_input" / "object_5.npy").is_file()
    assert not (out / "prior_input" / "object_9.npy").exists()
    assert by[3]["texture"] == fp.ATLAS_NOTE and len(by[3]["prior_input_sha256"]) == 64
    entry = json.loads((out / "ledger.json").read_text())["entries"][-1]
    assert entry["item"] == fp.ITEM and entry["decision"] == "accept"
    assert entry["packed"] == 2 and entry["prior_commit"] == fp.PRIOR_COMMIT
    driver = box.calls[-1]
    assert driver[:2] == [fp.FIT_PYTHON, "-c"]
    assert box.cwd == str(out / "fitted")  # not the prior checkout


def test_a_failed_fit_is_a_fallback_and_the_stage_still_accepts(tmp_path):
    pytest.importorskip("trimesh")
    out = _bin(tmp_path)
    constant = _pack()
    constant[:, fp.OPACITY_COLUMN] = 0.99
    box = _Box(fp.PRIOR_COMMIT, {3: None, 5: constant})
    assert fp.run_fitted_prior(out, runner=box, prior_repo=tmp_path / "prior") == 0
    entry = json.loads((out / "ledger.json").read_text())["entries"][-1]
    assert entry["decision"] == "accept" and entry["failed"] == 2
    assert "only 3 views" in entry["reason"] and "constant" in entry["reason"]
    assert [row["plan_source"] for row in entry["fallbacks"]] == [
        "fallback",
        "fallback",
    ]
    assert all(row["reason"] for row in entry["fallbacks"])


def test_a_checkpoint_with_the_wrong_hash_is_refused(tmp_path):
    (tmp_path / "prior.npz").write_bytes(b"not the prior")
    with pytest.raises(ValueError, match="not the pinned"):
        fp.checked_checkpoint(tmp_path / "prior.npz")


def test_an_accepted_object_without_a_mesh_rejects_before_the_fit(tmp_path):
    out = tmp_path / "bin"
    out.mkdir()
    (out / "bin_score.json").write_text(
        json.dumps(
            {"icp": {"objects": [{"object": 1, "placed": True, "gate": "accept"}]}}
        )
    )
    box = _Box(fp.PRIOR_COMMIT, {})
    assert fp.run_fitted_prior(out, runner=box, prior_repo=tmp_path / "prior") == 1
    assert [c[0] for c in box.calls] == ["git"]


def test_an_unpinned_prior_checkout_is_refused_before_any_fit(tmp_path):
    pytest.importorskip("trimesh")
    out = _bin(tmp_path)
    box = _Box("0" * 40, {3: _pack()})
    assert fp.run_fitted_prior(out, runner=box, prior_repo=tmp_path / "prior") == 1
    assert [c[0] for c in box.calls] == ["git"]  # the driver never ran
    entry = json.loads((out / "ledger.json").read_text())["entries"][-1]
    assert "not the pinned" in entry["reason"]


def test_the_prior_stage_lifts_its_packs_onto_the_scene_gaussians(tmp_path):
    pytest.importorskip("trimesh")
    out = _bin(tmp_path)
    (out / "association").mkdir()
    np.savez_compressed(
        out / "association" / "objects.npz",
        object_3=np.array([0, 1], np.int32),
        object_5=np.array([2], np.int32),
    )
    (out / "association_evidence.json").write_text(
        json.dumps({"inputs": {"positions": 4}})
    )
    box = _Box(fp.PRIOR_COMMIT, {3: _pack(64, 3), 5: _pack(100, 5)})
    assert fp.run_fitted_prior(out, runner=box, prior_repo=tmp_path / "prior") == 0
    entry = json.loads((out / "ledger.json").read_text())["entries"][-1]
    assert entry["features"] == "features shape 4 x 896"
    features = np.load(out / "features" / "features.npz")["features"]
    assert features.shape == (4, 896) and np.isnan(features[3]).all()
    lift = json.loads((out / "prior_input_ledger.json").read_text())["feature_lift"]
    assert lift["objects"] == {"object_3": 2, "object_5": 1}


def test_a_lift_without_association_is_recorded_and_the_packs_stay(tmp_path):
    pytest.importorskip("trimesh")
    out = _bin(tmp_path)
    box = _Box(fp.PRIOR_COMMIT, {3: _pack(64, 3), 5: _pack(100, 5)})
    assert fp.run_fitted_prior(out, runner=box, prior_repo=tmp_path / "prior") == 0
    evidence = json.loads((out / "prior_input_ledger.json").read_text())
    assert "FileNotFoundError" in evidence["feature_lift"]["error"]
    assert (
        "features" not in json.loads((out / "ledger.json").read_text())["entries"][-1]
    )
