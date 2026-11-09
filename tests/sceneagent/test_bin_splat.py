# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bin splat orchestration with a fake nerfstudio: no GPU, no ns-* installed."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from sceneagent import bin_splat
from sceneagent.bin_run import SCENE, SOURCE


def _passing_bin(out: Path) -> None:
    (out / "sparse" / "0").mkdir(parents=True)
    (out / "sparse" / "0" / "cameras.bin").write_bytes(b"x")
    (out / "images").mkdir()
    (out / "bin_evidence.json").write_text(
        json.dumps({"quality_bar": {"model": "0", "pass": True}})
    )
    (out / "ledger.json").write_text(
        json.dumps(
            {
                "scene_name": f"{SOURCE} {SCENE}",
                "source": SOURCE,
                "entries": [
                    {"item": "benchmark scene", "decision": "accept"},
                    {"item": "COLMAP camera poses", "decision": "accept"},
                ],
            }
        )
    )


class FakeNerfstudio:
    """Writes what each ns tool would leave behind, and records every call."""

    def __init__(self, fail: str | None = None, gaussians: int = 5) -> None:
        self.calls: list[tuple[list[str], dict]] = []
        self.fail = fail
        self.gaussians = gaussians

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        tool = argv[0]
        if tool == self.fail:
            return SimpleNamespace(returncode=1, stdout="", stderr="boom")
        if tool == "ns-train":
            run = Path(argv[argv.index("--output-dir") + 1]) / "processed/splatfacto/t1"
            run.mkdir(parents=True)
            (run / "config.yml").write_text("method_name: splatfacto\n")
        elif tool == "ns-eval":
            Path(argv[argv.index("--output-path") + 1]).write_text(
                json.dumps({"results": {"psnr": 25.0, "ssim": 0.8, "lpips": 0.2}})
            )
        elif tool == "ns-export":
            export = Path(argv[argv.index("--output-dir") + 1])
            export.mkdir(parents=True)
            header = f"ply\nformat binary_little_endian 1.0\nelement vertex {self.gaussians}\nend_header\n"
            (export / "splat.ply").write_bytes(header.encode())
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")


def _never_poses(out, data):
    raise AssertionError("poses should have been reused")


def test_reuses_passing_poses_and_accepts_a_nonempty_splat(tmp_path):
    _passing_bin(tmp_path)
    fake = FakeNerfstudio()
    assert bin_splat.run_bin_splat(tmp_path, poses=_never_poses, runner=fake) == 0
    tools = [argv[0] for argv, _ in fake.calls]
    assert tools == ["ns-process-data", "ns-train", "ns-eval", "ns-export"]
    process = fake.calls[0][0]
    assert "--skip-colmap" in process
    assert process[process.index("--colmap-model-path") + 1] == str(
        (tmp_path / "sparse/0").resolve()
    )
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert [e["item"] for e in ledger["entries"]] == [
        "benchmark scene",
        "COLMAP camera poses",
        "full-scene splat",
    ]
    splat = ledger["entries"][-1]
    assert splat["decision"] == "accept" and splat["gaussians"] == 5
    assert splat["psnr"] == 25.0
    evidence = json.loads((tmp_path / "splat_evidence.json").read_text())
    assert evidence["poses_reused"] is True
    assert json.loads((tmp_path / "ns_eval.json").read_text())["ok"] is True


def test_ns_tools_run_from_the_nerf_env_without_our_pythonpath(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/workspace/sf4-smoke/repo")
    _passing_bin(tmp_path)
    fake = FakeNerfstudio()
    bin_splat.run_bin_splat(tmp_path, poses=_never_poses, runner=fake)
    for _, kwargs in fake.calls:
        env = kwargs["env"]
        assert "PYTHONPATH" not in env
        assert env["PATH"].split(":")[:2] == [
            bin_splat.NS_BIN,
            f"{bin_splat.CUDA_HOME}/bin",
        ]
        assert env["CUDA_HOME"] == bin_splat.CUDA_HOME


def test_a_failed_train_stops_before_export_and_rejects(tmp_path):
    _passing_bin(tmp_path)
    fake = FakeNerfstudio(fail="ns-train")
    assert bin_splat.run_bin_splat(tmp_path, poses=_never_poses, runner=fake) == 1
    assert [argv[0] for argv, _ in fake.calls] == ["ns-process-data", "ns-train"]
    entry = json.loads((tmp_path / "ledger.json").read_text())["entries"][-1]
    assert entry == {
        "item": "full-scene splat",
        "method": "splatfacto",
        "decision": "reject",
        "failed_step": "train",
    }


def test_an_empty_export_is_rejected(tmp_path):
    _passing_bin(tmp_path)
    fake = FakeNerfstudio(gaussians=0)
    assert bin_splat.run_bin_splat(tmp_path, poses=_never_poses, runner=fake) == 1
    evidence = json.loads((tmp_path / "splat_evidence.json").read_text())
    assert evidence["failed_step"] == "export"


def test_a_timeout_is_recorded_as_124(tmp_path):
    _passing_bin(tmp_path)

    def slow(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    assert bin_splat.run_bin_splat(tmp_path, poses=_never_poses, runner=slow) == 1
    step = json.loads((tmp_path / "splat_evidence.json").read_text())["steps"][0]
    assert step["returncode"] == 124 and step["step"] == "process-data"


def test_runs_poses_when_the_ledger_belongs_to_another_pipeline(tmp_path):
    _passing_bin(tmp_path)
    (tmp_path / "ledger.json").write_text(
        json.dumps({"scene_name": "Fruits", "entries": [{"item": "video"}]})
    )
    called = []

    def poses(out, data):
        called.append(out)
        _rewrite = json.loads((out / "ledger.json").read_text())
        _rewrite.update(
            scene_name=f"{SOURCE} {SCENE}",
            entries=[
                {"item": "benchmark scene", "decision": "accept"},
                {"item": "COLMAP camera poses", "decision": "accept"},
            ],
        )
        (out / "ledger.json").write_text(json.dumps(_rewrite))
        return 0

    assert bin_splat.run_bin_splat(tmp_path, poses=poses, runner=FakeNerfstudio()) == 0
    assert called == [tmp_path]
    assert (
        json.loads((tmp_path / "splat_evidence.json").read_text())["poses_reused"]
        is False
    )


def test_failed_poses_reject_the_splat_without_running_nerfstudio(tmp_path):
    fake = FakeNerfstudio()

    def poses(out, data):
        (out / "ledger.json").write_text(
            json.dumps({"entries": [{"item": "benchmark scene", "decision": "reject"}]})
        )
        return 1

    assert bin_splat.run_bin_splat(tmp_path, poses=poses, runner=fake) == 1
    assert fake.calls == []
    entry = json.loads((tmp_path / "ledger.json").read_text())["entries"][-1]
    assert entry["decision"] == "reject" and entry["reason"] == "bin poses did not pass"


def test_ply_vertex_count_reads_only_the_header(tmp_path):
    ply = tmp_path / "a.ply"
    ply.write_bytes(b"ply\nformat ascii 1.0\nelement vertex 42\nend_header\n\x00\x01")
    assert bin_splat.ply_vertex_count(ply) == 42
    assert bin_splat.ply_vertex_count(tmp_path / "missing.ply") == 0
