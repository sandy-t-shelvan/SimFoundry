# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SF-15 orchestration and filtering, with a fake mask worker: no GPU."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sceneagent import bin_masks, mask_filter
from sceneagent.bin_run import SCENE, SOURCE, VIEWS

W, H = 1920, 1080


def _det(phrase, logit, box):
    return {"phrase": phrase, "logit": logit, "box_xyxy": box}


def test_filter_drops_off_prompt_huge_and_duplicate_boxes():
    dets = [
        _det("object", 0.9, [100, 100, 300, 300]),
        _det("object", 0.5, [105, 105, 300, 300]),  # same object, lower score
        _det("object", 0.8, [0, 0, 1900, 1000]),  # the bin
        _det("table", 0.7, [500, 500, 600, 600]),
        _det("object", 0.6, [800, 800, 900, 900]),
    ]
    out = mask_filter.filter_detections(dets, W, H, "object .")
    assert [d["kept"] for d in out] == [True, False, False, False, True]
    assert [d["drop_reason"] for d in out] == [
        None,
        "duplicate of a higher-scoring box",
        "box covers most of the view",
        "phrase not in prompt",
        None,
    ]


def test_prompt_terms_split_on_dots():
    assert mask_filter.prompt_terms("Object . red can .") == ["object", "red can"]


def _bin_images(out: Path, n: int = VIEWS) -> None:
    images = out / "images"
    images.mkdir(parents=True)
    for i in range(n):
        (images / f"{4 * i + 1:06d}.png").write_bytes(b"png")
    (out / "ledger.json").write_text(
        json.dumps(
            {
                "scene_name": f"{SOURCE} {SCENE}",
                "source": SOURCE,
                "entries": [{"item": "benchmark scene", "decision": "accept"}],
            }
        )
    )


class FakeMaskEnv:
    """segment_anything import check, optional pip, then the worker."""

    def __init__(self, kept_per_view=2, has_sam=True, worker_rc=0):
        self.calls = []
        self.kept_per_view = kept_per_view
        self.has_sam = has_sam
        self.worker_rc = worker_rc

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if argv[1:3] == ["-c", "import segment_anything"]:
            return SimpleNamespace(
                returncode=0 if self.has_sam else 1, stdout="", stderr=""
            )
        if argv[1:4] == ["-m", "pip", "install"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        assert argv[1:3] == ["-m", "sceneagent.mask_worker"]
        if self.worker_rc:
            return SimpleNamespace(
                returncode=self.worker_rc, stdout="", stderr="cuda oom"
            )
        images = Path(argv[argv.index("--images") + 1])
        masks = Path(argv[argv.index("--out") + 1])
        masks.mkdir(parents=True)
        report = {}
        for i, png in enumerate(sorted(images.glob("*.png"))):
            kept = self.kept_per_view if i else min(self.kept_per_view, 1)
            report[png.name] = [
                {
                    "phrase": "object",
                    "logit": 0.9,
                    "box_xyxy": [0, 0, 10, 10],
                    "kept": True,
                }
                for _ in range(kept)
            ] + [
                {
                    "phrase": "table",
                    "kept": False,
                    "drop_reason": "phrase not in prompt",
                }
            ]
        (masks / "detections.json").write_text(json.dumps(report))
        return SimpleNamespace(returncode=0, stdout="", stderr="")


def _weights():
    return {
        "path": bin_masks.SAM_WEIGHTS,
        "sha256": "4b8939" + "0" * 58,
        "downloaded": False,
    }


def _never_poses(out, data):
    raise AssertionError("images were already staged")


def test_all_views_with_masks_are_accepted_and_recorded(tmp_path):
    _bin_images(tmp_path)
    fake = FakeMaskEnv()
    rc = bin_masks.run_bin_masks(
        tmp_path, poses=_never_poses, runner=fake, weights=_weights
    )
    assert rc == 0
    entry = json.loads((tmp_path / "ledger.json").read_text())["entries"][-1]
    assert entry["item"] == "SAM masks" and entry["decision"] == "accept"
    assert entry["sam"] == "vit_h" and entry["prompt"] == mask_filter.PROMPT
    assert entry["masks"] == 1 + 2 * (VIEWS - 1)
    assert entry["kept_per_view"] == [1, 2] and entry["gt_objects"] == 20
    evidence = json.loads((tmp_path / "masks_evidence.json").read_text())
    assert evidence["summary"]["views_with_masks"] == VIEWS
    assert (
        evidence["filter"]["max_box_area_fraction"] == mask_filter.MAX_BOX_AREA_FRACTION
    )
    worker_argv, worker_kwargs = fake.calls[-1]
    assert worker_argv[0] == bin_masks.MASK_PYTHON
    assert worker_argv[worker_argv.index("--sam-weights") + 1] == bin_masks.SAM_WEIGHTS
    assert worker_kwargs["env"]["PYTHONPATH"].endswith(bin_masks.MASK_PYDEPS)
    # Grounded-SAM-2 imports its GroundingDINO as grounding_dino.*.
    assert bin_masks.GSAM_ROOT in worker_kwargs["env"]["PYTHONPATH"].split(":")


def test_a_view_without_masks_rejects(tmp_path):
    _bin_images(tmp_path)
    rc = bin_masks.run_bin_masks(
        tmp_path,
        poses=_never_poses,
        runner=FakeMaskEnv(kept_per_view=0),
        weights=_weights,
    )
    assert rc == 1
    entry = json.loads((tmp_path / "ledger.json").read_text())["entries"][-1]
    assert entry["reason"] == f"0 of {VIEWS} views kept a mask"


def test_missing_segment_anything_installs_into_its_own_target(tmp_path):
    _bin_images(tmp_path)
    fake = FakeMaskEnv(has_sam=False)
    bin_masks.run_bin_masks(tmp_path, poses=_never_poses, runner=fake, weights=_weights)
    pip = next(argv for argv, _ in fake.calls if "pip" in argv)
    assert pip[-3:] == ["--target", bin_masks.MASK_PYDEPS, bin_masks.SAM_PIP]


def test_a_worker_failure_rejects_with_its_exit_code(tmp_path):
    _bin_images(tmp_path)
    rc = bin_masks.run_bin_masks(
        tmp_path, poses=_never_poses, runner=FakeMaskEnv(worker_rc=3), weights=_weights
    )
    assert rc == 1
    entry = json.loads((tmp_path / "ledger.json").read_text())["entries"][-1]
    assert "mask worker exited 3: cuda oom" in entry["reason"]
    evidence = json.loads((tmp_path / "masks_evidence.json").read_text())
    assert evidence["worker"]["output_tail"] == "cuda oom"


def test_missing_images_run_the_bin_poses_first(tmp_path):
    called = []

    def poses(out, data):
        called.append(out)
        _bin_images(out)
        return 0

    rc = bin_masks.run_bin_masks(
        tmp_path, poses=poses, runner=FakeMaskEnv(), weights=_weights
    )
    assert rc == 0 and called == [tmp_path]


def test_weights_are_downloaded_once_and_hash_checked(tmp_path, monkeypatch):
    payload = b"vit_h"
    monkeypatch.setattr(bin_masks, "SAM_SHA256", hashlib.sha256(payload).hexdigest())
    target = tmp_path / "sam" / "w.pth"
    fetched = []

    def download(url, path):
        fetched.append(url)
        path.write_bytes(payload)

    first = bin_masks.ensure_sam_weights(target, "https://x/w.pth", download=download)
    second = bin_masks.ensure_sam_weights(target, "https://x/w.pth", download=download)
    assert first["downloaded"] is True and second["downloaded"] is False
    assert fetched == ["https://x/w.pth"]


def test_a_wrong_hash_on_an_existing_file_raises_and_leaves_it(tmp_path):
    target = tmp_path / "w.pth"
    target.write_bytes(b"not sam")
    with pytest.raises(ValueError, match="is not the published"):
        bin_masks.ensure_sam_weights(target, "https://x/w.pth")
    assert target.read_bytes() == b"not sam"  # someone else's checkout


def test_a_wrong_hash_on_a_download_removes_it(tmp_path):
    target = tmp_path / "w.pth"

    def download(url, path):
        path.write_bytes(b"truncated")

    with pytest.raises(ValueError, match="is not the published"):
        bin_masks.ensure_sam_weights(target, "https://x/w.pth", download=download)
    assert not target.exists()


def test_mask_worker_imports_without_torch_or_numpy():
    import subprocess
    import sys

    code = (
        "import sys, sceneagent.mask_worker; "
        "bad = [m for m in ('numpy', 'torch', 'trimesh') if m in sys.modules]; "
        "sys.exit(1 if bad else 0)"
    )
    root = Path(bin_masks.__file__).resolve().parent.parent
    done = subprocess.run([sys.executable, "-c", code], cwd=root, check=False)
    assert done.returncode == 0


def test_the_bin_prompt_names_object_categories_and_says_where_they_come_from():
    terms = mask_filter.prompt_terms(mask_filter.PROMPT)
    assert len(terms) >= 10 and "can" in terms and "bottle" in terms
    assert "ours" in mask_filter.PROMPT_SOURCE
    dets = [
        _det("bottle", 0.6, [10, 10, 60, 200]),
        _det("wall", 0.9, [300, 300, 400, 400]),
    ]
    out = mask_filter.filter_detections(dets, W, H)
    assert [d["kept"] for d in out] == [True, False]
