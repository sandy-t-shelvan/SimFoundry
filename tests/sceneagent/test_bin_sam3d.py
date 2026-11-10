# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SF-29 orchestration with a fake SAM 3D worker: view choice, ledger, failures."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sceneagent import bin_sam3d
from sceneagent.bin_run import SCENE, SOURCE
from sceneagent.vast_smoke import (
    FETCH_NAMES,
    PIPELINES,
    REMOTE_BIN_OUT,
    build_remote_command,
    remote_out,
)

OBJECTS = [
    {
        "object": 0,
        "masks": [
            {"view": "000001.png", "mask_index": 0, "phrase": "jar"},
            {"view": "000005.png", "mask_index": 1, "phrase": "jar"},
        ],
    },
    {"object": 1, "masks": [{"view": "000005.png", "mask_index": 0, "phrase": "box"}]},
]
DETECTIONS = {
    "000001.png": [{"kept": True, "mask_index": 0, "mask_area_px": 900}],
    "000005.png": [
        {"kept": True, "mask_index": 0, "mask_area_px": 5000},
        {"kept": True, "mask_index": 1, "mask_area_px": 1200},
        {"kept": False, "reason": "box area"},
    ],
}


def _run_dir(tmp_path: Path) -> Path:
    out = tmp_path / "bin"
    (out / "masks").mkdir(parents=True)
    (out / "masks" / "detections.json").write_text(json.dumps(DETECTIONS))
    (out / "association_evidence.json").write_text(json.dumps({"object_list": OBJECTS}))
    entries = [
        {"item": item, "decision": "accept"}
        for item in (
            "benchmark scene",
            "COLMAP camera poses",
            "full-scene splat",
            "SAM masks",
            "Gaussian association",
        )
    ]
    (out / "ledger.json").write_text(
        json.dumps(
            {"scene_name": f"{SOURCE} {SCENE}", "source": SOURCE, "entries": entries}
        )
    )
    return out


class FakeWorker:
    """Writes the results.json the real worker would, failing ``fail`` objects."""

    def __init__(self, fail=(), write=True):
        self.fail, self.write, self.calls = set(fail), write, []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        jobs = json.loads(Path(argv[argv.index("--jobs") + 1]).read_text())
        out = Path(argv[argv.index("--out") + 1])
        objects = []
        for job in jobs:
            if job["object"] in self.fail:
                objects.append(
                    {"object": job["object"], "ok": False, "error": "CUDA OOM"}
                )
            else:
                pose = {
                    "rotation": [1, 0, 0, 0],
                    "translation": [0, 0, 1],
                    "scale": [1, 1, 1],
                }
                objects.append(
                    {"object": job["object"], "ok": True, "glb_bytes": 10, "pose": pose}
                )
        if self.write:
            (out / "results.json").write_text(
                json.dumps({"load_seconds": 3.0, "objects": objects})
            )
        return SimpleNamespace(
            returncode=0 if self.write else 1, stdout="", stderr="boom"
        )


def _config(tmp_path):
    def config():
        return tmp_path / "pipeline.yaml"

    return config


def test_each_object_uses_its_largest_mask():
    chosen = bin_sam3d.choose_views(OBJECTS, DETECTIONS)
    assert [(c["object"], c["view"], c["mask_index"]) for c in chosen] == [
        (0, "000005.png", 1),
        (1, "000005.png", 0),
    ]
    assert chosen[0]["mask_area_px"] == 1200


def test_all_objects_meshed_is_an_accept(tmp_path):
    out = _run_dir(tmp_path)
    fake = FakeWorker()
    assert bin_sam3d.run_bin_sam3d(out, runner=fake, config=_config(tmp_path)) == 0
    argv, kwargs = fake.calls[0]
    assert argv[0] == bin_sam3d.SAM3D_PYTHON and argv[2] == "sceneagent.sam3d_worker"
    assert kwargs["cwd"] == bin_sam3d.SAM3D_ROOT
    assert kwargs["env"]["CONDA_PREFIX"] == bin_sam3d.SAM3D_ENV
    jobs = json.loads((out / "sam3d" / "jobs.json").read_text())
    assert jobs[0]["image"].endswith("images/000005.png")
    assert jobs[0]["masks"].endswith("masks/000005.npz") and jobs[0]["mask_index"] == 1
    entry = json.loads((out / "ledger.json").read_text())["entries"][-1]
    assert entry == {
        "item": "SAM 3D meshes",
        "decision": "accept",
        "use_gpu": True,
        "objects": 2,
    }
    evidence = json.loads((out / "sam3d_evidence.json").read_text())
    assert evidence["meshed"] == 2 and evidence["view_rule"].startswith("ours")


def test_one_failed_object_rejects_and_nothing_is_substituted(tmp_path):
    out = _run_dir(tmp_path)
    assert (
        bin_sam3d.run_bin_sam3d(
            out, runner=FakeWorker(fail={1}), config=_config(tmp_path)
        )
        == 1
    )
    entry = json.loads((out / "ledger.json").read_text())["entries"][-1]
    assert entry["decision"] == "reject"
    assert entry["reason"] == "1 of 2 objects meshed; first failure: CUDA OOM"
    evidence = json.loads((out / "sam3d_evidence.json").read_text())
    assert [o.get("ok") for o in evidence["objects"]] == [True, False]


def test_a_worker_that_wrote_nothing_rejects_with_its_tail(tmp_path):
    out = _run_dir(tmp_path)
    assert (
        bin_sam3d.run_bin_sam3d(
            out, runner=FakeWorker(write=False), config=_config(tmp_path)
        )
        == 1
    )
    entry = json.loads((out / "ledger.json").read_text())["entries"][-1]
    assert entry["reason"].startswith("RuntimeError: sam3d worker exited 1: boom")


def test_config_search_takes_the_first_pipeline_yaml(tmp_path):
    late = tmp_path / "snap" / "abc" / "checkpoints" / "pipeline.yaml"
    late.parent.mkdir(parents=True)
    late.write_text("x")
    missing = str(tmp_path / "hf" / "pipeline.yaml")
    assert bin_sam3d.find_config((missing,), str(tmp_path / "snap")) == late
    with pytest.raises(FileNotFoundError):
        bin_sam3d.find_config((missing,), str(tmp_path / "none"))


def test_sam3d_is_a_bin_pipeline():
    argv = build_remote_command("/workspace/r", "/workspace/o", pipeline="sam3d")
    assert argv[1:] == ["-m", "sceneagent.bin_sam3d", "--out", "/workspace/o"]
    assert remote_out("sam3d") == REMOTE_BIN_OUT
    assert PIPELINES["sam3d"][0][-1] == "SAM 3D meshes"
    assert "sam3d_evidence.json" in FETCH_NAMES
