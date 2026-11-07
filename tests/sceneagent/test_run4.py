# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the SF-4 run4 gates. No GPU, SSH, or COLMAP binary."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

from sceneagent.gpu_headroom import (
    COLMAP_MIN_FREE_MIB,
    NVIDIA_SMI_QUERY,
    SPLATFACTO_MIN_FREE_MIB,
    GpuCommandResult,
    GpuHeadroomError,
    require_gpu_headroom,
)
from sceneagent.gpu_smoke_trigger import workflow_safety_findings
from sceneagent.run4 import main, run_colmap_on_images, run_run4
from sceneagent.run4_report import registered_image_names, unregistered_frame_report
from sceneagent.run4_sampling import (
    BIN_COUNT,
    FRUITS_BYTES,
    FRUITS_FRAME_COUNT,
    FRUITS_MVHD_DURATION_S,
    FRUITS_PROBE_DURATION_S,
    FRUITS_SHA256,
    decode_all_command,
    extract_selected_command,
    laplacian_variance,
    select_expression,
    select_sharpest_per_bin,
)
from sceneagent.simfoundry_python import (
    SIMFOUNDRY_PYTHON,
    PinnedInterpreterMissing,
    require_simfoundry_python,
    stage_environment,
)
from sceneagent.vast_smoke import (
    REMOTE_PYTHON,
    InstanceNotApproved,
    build_remote_command,
    build_remote_shell,
    run_smoke,
)
from sceneagent.video import inspect_video

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "sceneagent-gpu-smoke.yml"
FRUITS = ROOT / "docs/assets/example_videos/Fruits.mp4"
SHA = "a" * 40


def _python(tmp_path: Path) -> str:
    path = tmp_path / "python3"
    path.write_bytes(b"")
    path.chmod(0o755)
    return str(path)


def _gpu(stdout: str, returncode: int = 0, stderr: str = "") -> GpuCommandResult:
    return GpuCommandResult(returncode, stdout, stderr)


def _frames(count: int = FRUITS_FRAME_COUNT, duration: float = FRUITS_PROBE_DURATION_S):
    return [
        {"index": i, "time": duration * i / count, "score": float(i)} for i in range(count)
    ]


def _images_bin(names: list[str]) -> bytes:
    chunks = [struct.pack("<Q", len(names))]
    for name in names:
        chunks.append(struct.pack("<i4d3di", 1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1))
        chunks.append(name.encode("utf-8") + b"\x00")
        chunks.append(struct.pack("<Q", 0))
    return b"".join(chunks)


def _analyzer_text(
    registered: int = 15, points: int = 0, observations: str = "30.000000", reprojection: str = "4.0"
) -> str:
    return "\n".join(
        [
            f"Registered images: {registered}",
            f"] Points: {points}",
            f"Mean observations per image: {observations}",
            f"Mean reprojection error: {reprojection}px",
        ]
    )


def test_pinned_interpreter_is_the_simfoundry_env_and_names_a_missing_file(tmp_path):
    assert SIMFOUNDRY_PYTHON == "/root/miniforge3/envs/simfoundry/bin/python3"
    assert REMOTE_PYTHON == SIMFOUNDRY_PYTHON
    missing = tmp_path / "missing-python"
    with pytest.raises(PinnedInterpreterMissing, match=f"pinned interpreter missing: {missing}"):
        require_simfoundry_python(str(missing))
    pinned = _python(tmp_path)
    assert require_simfoundry_python(pinned) == pinned
    env = stage_environment(pinned)
    assert env["PATH"].split(":")[0] == str(Path(pinned).parent)


def test_main_exits_127_when_the_pinned_interpreter_is_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "sceneagent.simfoundry_python.SIMFOUNDRY_PYTHON", str(tmp_path / "no-python")
    )

    def explode(*_args, **_kwargs):
        raise AssertionError("load_frames ran")

    monkeypatch.setattr("sceneagent.run4.load_frames", explode)
    code = main([str(tmp_path / "clip.mp4"), "--out", str(tmp_path / "out")])
    assert code == 127
    assert capsys.readouterr().err.strip() == (
        f"pinned interpreter missing: {tmp_path / 'no-python'}"
    )


def test_gpu_check_records_the_real_error_and_does_not_retry():
    calls = []

    def query(argv):
        calls.append(argv)
        return _gpu("", returncode=9, stderr="NVML: driver refused the query")

    with pytest.raises(GpuHeadroomError, match="rc=9: NVML: driver refused the query") as caught:
        require_gpu_headroom("colmap", query=query)
    assert calls == [NVIDIA_SMI_QUERY]
    assert caught.value.returncode == 9
    assert caught.value.stderr == "NVML: driver refused the query"
    assert caught.value.retries == 0
    assert "utilization" not in " ".join(NVIDIA_SMI_QUERY)
    assert "compute-apps" not in " ".join(NVIDIA_SMI_QUERY)

    def unused(_argv):
        raise AssertionError("retried")

    with pytest.raises(GpuHeadroomError, match="retries are not allowed"):
        require_gpu_headroom("colmap", query=unused, retries=1)


@pytest.mark.parametrize(
    ("stage", "free", "ok"),
    [
        ("colmap", 6144, True),
        ("colmap", 6143, False),
        ("splatfacto", 16384, True),
        ("splatfacto", 16383, False),
    ],
)
def test_memory_floors_are_6144_and_16384(stage, free, ok):
    assert COLMAP_MIN_FREE_MIB == 6144
    assert SPLATFACTO_MIN_FREE_MIB == 16384

    def query(_argv):
        return _gpu(f"0, 24564, {free}")

    if ok:
        sample = require_gpu_headroom(stage, query=query)
        assert sample["free_mib"] == free
        assert sample["retries"] == 0
        assert sample["passed"] is True
    else:
        with pytest.raises(GpuHeadroomError, match="no retry") as caught:
            require_gpu_headroom(stage, query=query)
        assert caught.value.sample["free_mib"] == free
        assert caught.value.retries == 0


def test_unreadable_gpu_sample_fails_closed():
    def query(_argv):
        return _gpu("0, inf, 24564\n1, 24564, 20000")

    with pytest.raises(GpuHeadroomError, match="exactly one GPU"):
        require_gpu_headroom("colmap", query=query)

    def nan_query(_argv):
        return _gpu("0, nan, 6144")

    with pytest.raises(GpuHeadroomError, match="invalid aggregate memory sample"):
        require_gpu_headroom("splatfacto", query=nan_query)


def test_sharpest_bins_cover_the_whole_fruits_length_and_do_not_truncate():
    chosen = select_sharpest_per_bin(_frames(), FRUITS_PROBE_DURATION_S)
    indexes = [row["chosen_index"] for row in chosen]
    assert len(indexes) == BIN_COUNT == 300
    assert indexes[-1] == FRUITS_FRAME_COUNT - 1
    assert max(indexes) > 299
    assert indexes != list(range(300))
    command = decode_all_command("ffmpeg", "Fruits.mp4", "frames/%06d.png")
    assert "-frames:v" not in command
    assert not any(part.startswith("fps=") for part in command)
    extract = extract_selected_command("ffmpeg", "Fruits.mp4", indexes, "images/%04d.jpg")
    assert "-frames:v" in extract
    assert extract[extract.index("-frames:v") + 1] == "300"
    expression = extract[extract.index("-vf") + 1]
    assert f"eq(n\\,{FRUITS_FRAME_COUNT - 1})" in expression
    assert expression.startswith("select=(")


def test_empty_bin_tie_and_nonfinite_score_fail_closed():
    with pytest.raises(ValueError, match="empty time bin"):
        select_sharpest_per_bin(
            [
                {"index": 0, "time": 0.0, "score": 1.0},
                {"index": 1, "time": 0.01, "score": 1.0},
            ],
            10.0,
            n=4,
        )
    chosen = select_sharpest_per_bin(
        [
            {"index": 0, "time": 0.0, "score": 3.0},
            {"index": 1, "time": 0.5, "score": 3.0},
        ],
        1.0,
        n=1,
    )
    assert chosen[0]["chosen_index"] == 0
    with pytest.raises(ValueError, match="invalid sharpness"):
        select_sharpest_per_bin(
            [{"index": 0, "time": 0.0, "score": float("nan")}], 1.0, n=1
        )
    assert laplacian_variance([[0, 0, 0], [0, 0, 0], [0, 0, 0]]) == 0
    assert laplacian_variance(
        [
            [0, 0, 0, 0],
            [0, 0, 9, 0],
            [0, 4, 0, 0],
            [0, 0, 0, 0],
        ]
    ) > 0
    with pytest.raises(ValueError, match="frame too small"):
        laplacian_variance([[1, 2], [3, 4]])


def test_registered_names_and_report_do_not_treat_a_missing_read_as_empty():
    names = ["0001.jpg", "0002.jpg"]
    assert registered_image_names(_images_bin(names)) == names
    with pytest.raises(ValueError, match="truncated"):
        registered_image_names(_images_bin(names)[:-1])
    report = unregistered_frame_report(
        ["0001.jpg", "0002.jpg", "0003.jpg"],
        {"0": ["0001.jpg"], "1": None},
        {"1": "images.bin missing"},
        "0",
    )
    assert report["report_only"] is True
    assert report["per_model_unregistered"]["0"] == ["0002.jpg", "0003.jpg"]
    assert report["per_model_unregistered"]["1"] is None
    assert report["largest_model_unregistered"] == ["0002.jpg", "0003.jpg"]
    missing = unregistered_frame_report(["0001.jpg"], {"0": None}, {}, "0")
    assert missing["largest_model_unregistered"] is None
    assert missing["largest_model_unregistered_count"] is None


def test_sequential_matcher_is_unchanged_and_points_do_not_gate(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for index in range(1, 16):
        (image_dir / f"{index:04d}.jpg").write_bytes(b"")
    calls = []
    checks = []

    def gpu_check(stage):
        checks.append(stage)
        return {"free_mib": 6144, "retries": 0}

    def runner(argv):
        calls.append(list(argv))
        if argv[1] == "-h":
            return {"returncode": 0, "stdout": "COLMAP 3.11.1", "stderr": ""}
        if argv[1] == "mapper":
            model = Path(argv[argv.index("--output_path") + 1]) / "0"
            model.mkdir(parents=True)
            kept = [f"{index:04d}.jpg" for index in range(1, 15)]
            (model / "images.bin").write_bytes(_images_bin(kept))
        if argv[1] == "feature_extractor":
            return {"returncode": 0, "stdout": "", "stderr": ""}
        return {"returncode": 0, "stdout": "", "stderr": ""}

    def analyzer(_model):
        return {
            "returncode": 0,
            "stdout": _analyzer_text(points=0),
            "stderr": "",
        }

    result = run_colmap_on_images(
        image_dir,
        tmp_path / "work",
        gpu_check=gpu_check,
        command_runner=runner,
        analyzer=analyzer,
    )
    commands = [argv[1] for argv in calls]
    assert "vocab_tree_matcher" not in commands
    assert "sequential_matcher" in commands
    assert checks == ["colmap", "colmap", "colmap"]
    assert result["ok"] is True
    assert result["gate_document"]["quality_bar"]["points_info_only"] == 0
    assert result["gate_document"]["quality_bar"]["pass"] is True
    assert "0015.jpg" not in result["registered_names"]["0"]
    assert len(result["registered_names"]["0"]) == 14


def test_nonfinite_observation_still_fails_the_largest_model(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for index in range(1, 16):
        (image_dir / f"{index:04d}.jpg").write_bytes(b"")

    def runner(argv):
        if argv[1] == "-h":
            return {"returncode": 0, "stdout": "COLMAP 3.11.1", "stderr": ""}
        if argv[1] == "mapper":
            model = Path(argv[argv.index("--output_path") + 1]) / "0"
            model.mkdir(parents=True)
            (model / "images.bin").write_bytes(b"")
        return {"returncode": 0, "stdout": "", "stderr": ""}

    def analyzer(_model):
        return {
            "returncode": 0,
            "stdout": _analyzer_text(observations="nan"),
            "stderr": "",
        }

    result = run_colmap_on_images(
        image_dir,
        tmp_path / "work",
        gpu_check=lambda _stage: {"ok": True},
        command_runner=runner,
        analyzer=analyzer,
    )
    assert result["ok"] is False
    assert result["gate_document"]["quality_bar"]["mean_obs_per_image"]["value"] is None
    assert result["name_errors"]["0"]
    assert result["registered_names"]["0"] is None


def test_gpu_failure_does_not_start_colmap_or_retry(tmp_path):
    calls = []

    def query(argv):
        calls.append(argv)
        return _gpu("", returncode=1, stderr="real nvml failure")

    def colmap(_names, _selection):
        raise AssertionError("colmap ran")

    result = run_run4(
        tmp_path / "Fruits.mp4",
        tmp_path / "out",
        commit_sha=SHA,
        interpreter=_python(tmp_path),
        frames=_frames(),
        duration_s=FRUITS_PROBE_DURATION_S,
        gpu_query=query,
        run_colmap=colmap,
        inspect=lambda _path: {"name": "Fruits.mp4", "bytes": FRUITS_BYTES, "duration_s": 15.067},
    )
    assert calls == [NVIDIA_SMI_QUERY]
    assert result["code"] == 1
    evidence = json.loads((tmp_path / "out" / "run4_evidence.json").read_text(encoding="utf-8"))
    assert evidence["checked_out_commit"] == SHA
    assert evidence["gpu"]["colmap_error"]["stderr"] == "real nvml failure"
    assert evidence["gpu"]["colmap_error"]["returncode"] == 1
    assert evidence["gpu"]["colmap_error"]["retries"] == 0
    assert evidence["gpu"]["splatfacto_status"] == "not_run"
    assert evidence["gpu"]["splatfacto_min_free_mib"] == 16384
    assert evidence["retries"] == 0
    ledger = json.loads((tmp_path / "out" / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["entries"][1]["decision"] == "reject"
    assert "real nvml failure" in ledger["entries"][1]["reason"]


def test_run4_evidence_reports_unregistered_names_without_changing_the_pass(tmp_path):
    def query(_argv):
        return _gpu("0, 24564, 6144")

    def colmap(names, _selection):
        return {
            "ok": True,
            "use_gpu": True,
            "reason": "",
            "failed_step": None,
            "stderr": "",
            "returncode": 0,
            "gate_document": {
                "chosen_model": "0",
                "quality_bar": {"pass": True, "points_info_only": 1},
            },
            "registered_names": {"0": names[:-1]},
            "name_errors": {},
        }

    result = run_run4(
        tmp_path / "Fruits.mp4",
        tmp_path / "out",
        commit_sha=SHA,
        interpreter=_python(tmp_path),
        frames=_frames(),
        duration_s=FRUITS_PROBE_DURATION_S,
        gpu_query=query,
        run_colmap=colmap,
        inspect=lambda _path: {"name": "Fruits.mp4", "bytes": FRUITS_BYTES, "duration_s": 15.067},
    )
    assert result["code"] == 0
    evidence = result["evidence"]
    report = evidence["unregistered_frames"]
    assert report["report_only"] is True
    assert report["largest_model"] == "0"
    assert report["largest_model_unregistered"] == ["0300.jpg"]
    assert evidence["matcher"] == "sequential_matcher"
    assert evidence["sampling"]["decoded_frames"] == FRUITS_FRAME_COUNT
    assert evidence["sampling"]["selected_indices"][-1] == FRUITS_FRAME_COUNT - 1
    assert evidence["quality_bar"]["points_info_only"] == 1


def test_bad_commit_sha_is_not_replaced(tmp_path):
    def colmap(_names, _selection):
        raise AssertionError("colmap ran")

    result = run_run4(
        tmp_path / "Fruits.mp4",
        tmp_path / "out",
        commit_sha="not-a-sha",
        interpreter=_python(tmp_path),
        frames=_frames(count=8, duration=1.0),
        duration_s=1.0,
        gpu_query=lambda _argv: _gpu("0, 24564, 6144"),
        run_colmap=colmap,
        inspect=lambda _path: {"bytes": 1, "duration_s": 1.0},
    )
    assert result["code"] == 1
    assert result["evidence"]["checked_out_commit"] is None
    assert "not 40 hex" in result["evidence"]["fail_reasons"][0]


def test_remote_run4_command_keeps_the_pinned_interpreter(tmp_path):
    default = build_remote_command("/workspace/sf4-smoke/repo", "/workspace/sf4-smoke/run")
    assert default[0] == REMOTE_PYTHON
    assert default[1:3] == ["-m", "sceneagent"]
    argv = build_remote_command(
        "/workspace/sf4-smoke/repo",
        "/workspace/sf4-smoke/run",
        pipeline="run4",
        commit_sha=SHA,
    )
    assert argv[0] == REMOTE_PYTHON
    assert argv[1:3] == ["-m", "sceneagent.run4"]
    assert argv[-2:] == ["--commit-sha", SHA]
    script = build_remote_shell(
        "/workspace/sf4-smoke/repo",
        "/workspace/sf4-smoke/run",
        pipeline="run4",
        commit_sha=SHA,
    )
    assert "sceneagent.run4" in script
    assert "command -v python3" not in script
    assert script.count(REMOTE_PYTHON) >= 2
    with pytest.raises(InstanceNotApproved, match="refusing commit sha"):
        build_remote_command(
            "/workspace/sf4-smoke/repo",
            "/workspace/sf4-smoke/run",
            pipeline="run4",
            commit_sha="nope",
        )
    code = run_smoke(
        59601312,
        tmp_path,
        tmp_path / "artifacts",
        "",
        pipeline="not-a-pipeline",
    )
    assert code == 2


def test_workflow_runs_run4_and_records_the_checked_out_commit():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert workflow_safety_findings(text) == []
    assert text.count("secrets.VAST_SSH_PRIVATE_KEY") == 1
    assert "git rev-parse HEAD" in text
    assert "--pipeline run4" in text
    assert "--commit-sha \"$CHECKED_OUT_SHA\"" in text
    assert "StrictHostKeyChecking" not in text
    keyed = text.split("name: Vast GPU smoke on Fruits.mp4", 1)[1].split("name: Upload", 1)[0]
    run = keyed.split("run: |", 1)[1]
    assert "VAST_SSH_PRIVATE_KEY" not in run
    assert "secrets.VAST_SSH_PRIVATE_KEY" in keyed


def test_committed_fruits_file_matches_the_recorded_pin():
    data = FRUITS.read_bytes()
    assert len(data) == FRUITS_BYTES
    assert hashlib.sha256(data).hexdigest() == FRUITS_SHA256
    info = inspect_video(FRUITS)
    assert info["duration_s"] == pytest.approx(FRUITS_MVHD_DURATION_S)
