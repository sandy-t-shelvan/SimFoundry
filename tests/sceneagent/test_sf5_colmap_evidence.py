# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the SF-5 native-frame COLMAP evidence package.

No GPU, no SSH, and no launch. The dry run uses a synthetic video and a
stub colmap. The pinned Fruits.mp4 is hashed and probed, not reconstructed.
"""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from sceneagent.colmap_stage import run_poses, sample_frame_command
from sceneagent.reconstruction_gate import (
    MAX_REPROJ_PX,
    MIN_OBS_PER_IMAGE,
    MIN_REG_FLOOR,
    REG_FRACTION,
)
from sceneagent.sf5_colmap_evidence import (
    APPROVED_BYTES,
    APPROVED_DURATION_ROUNDED_S,
    APPROVED_FRAMES,
    APPROVED_HEIGHT,
    APPROVED_RELATIVE,
    APPROVED_SHA256,
    APPROVED_WIDTH,
    GPU_FREE_ARGV,
    INSTANCE_ID,
    LEGACY_MODEL_ROOT,
    PINNED_BASE_SHA,
    REQUIRED_COLMAP_VERSION,
    SCENEAGENT_ROOT,
    EvidenceError,
    ForbiddenPathError,
    assert_native_argv,
    assert_no_symlink,
    build_manifest,
    counts_as_fruits_evidence,
    evidence_kwargs,
    execute,
    fruits_probe_matches,
    gate_accepted,
    gpu_query_is_aggregate,
    interpret_gpu_free,
    launch_ready,
    main,
    output_allowed,
    parse_colmap_banner,
    path_is_forbidden,
    pinned_file_bytes_match,
    query_gpu_free,
    render_launch_command,
    required_output_prefix,
    reuse_refused,
    write_package_sums,
    _ffprobe,
)

ROOT = Path(__file__).resolve().parents[2]
FRUITS = ROOT / APPROVED_RELATIVE
CODE_SHA = "a" * 40
_STUB = """#!/usr/bin/env python3
import os
import sys
from pathlib import Path

def flag(name):
    return sys.argv[sys.argv.index(name) + 1]

args = sys.argv[1:]
if not args or args[0] == "-h":
    sys.stdout.write("COLMAP 3.11.1 -- Structure-from-Motion\\n")
    raise SystemExit(0)
cmd = args[0]
if cmd == "mapper":
    model = Path(flag("--output_path")) / "0"
    model.mkdir(parents=True, exist_ok=True)
    (model / "cameras.bin").write_bytes(b"stub-model")
    raise SystemExit(0)
if cmd == "model_analyzer":
    model = Path(flag("--path"))
    images = model.parent.parent / "images"
    count = len(list(images.glob("*.jpg")))
    reprojection = "9.000000" if os.environ.get("SF5_STUB_REPROJ") == "fail" else "1.000000"
    sys.stdout.write(
        "I20261106 00:00:00.000000 1 model.cc:448] Registered images: %d\\n"
        "I20261106 00:00:00.000000 1 model.cc:450] Points: 100\\n"
        "I20261106 00:00:00.000000 1 model.cc:455] Mean observations per image: 40.000000\\n"
        "I20261106 00:00:00.000000 1 model.cc:458] Mean reprojection error: %spx\\n"
        % (count, reprojection)
    )
    raise SystemExit(0)
if cmd in {"feature_extractor", "sequential_matcher"}:
    raise SystemExit(0)
sys.stderr.write("unexpected %s\\n" % cmd)
raise SystemExit(1)
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _synthetic_video(directory: Path) -> tuple[Path, str]:
    path = directory / "synthetic.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=10:duration=2",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path, _sha256(path)


def _install_stub(
    directory: Path, banner: str = "COLMAP 3.11.1 -- Structure-from-Motion\n"
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stub = directory / "colmap"
    needle = 'sys.stdout.write("COLMAP 3.11.1 -- Structure-from-Motion\\n")'
    text = _STUB if banner == "COLMAP 3.11.1 -- Structure-from-Motion\n" else _STUB.replace(
        needle, "sys.stdout.write(%r)" % banner, 1
    )
    stub.write_text(text, encoding="utf-8")
    stub.chmod(0o755)
    return stub


def _dry_request(tmp_path: Path, video: Path, digest: str, **overrides) -> dict:
    output = tmp_path / "out" / "run"
    request = {
        "repo": str(ROOT),
        "input": str(video),
        "input_sha256": digest,
        "output_prefix": str(output),
        "expect_code_sha": None,
        "instance_id": None,
        "dry_run": True,
    }
    request.update(overrides)
    return request


def test_repo_gate_thresholds_stay_the_published_bar():
    assert MIN_REG_FLOOR == 10
    assert REG_FRACTION == 0.8
    assert MIN_OBS_PER_IMAGE == 30.0
    assert MAX_REPROJ_PX == 4.0
    source = (ROOT / "sceneagent" / "sf5_colmap_evidence.py").read_text(encoding="utf-8")
    assert "MIN_REG_FLOOR" not in source
    assert "MAX_REPROJ_PX" not in source


def test_pinned_fruits_bytes_match_this_checkout():
    assert FRUITS.is_file()
    assert not FRUITS.is_symlink()
    assert FRUITS.stat().st_size == APPROVED_BYTES == 1733022
    assert _sha256(FRUITS) == APPROVED_SHA256
    assert pinned_file_bytes_match(ROOT) is True
    probed = _ffprobe(FRUITS)
    assert fruits_probe_matches(probed) is True
    assert probed["frames"] == APPROVED_FRAMES == 452
    assert probed["width"] == APPROVED_WIDTH == 1280
    assert probed["height"] == APPROVED_HEIGHT == 720
    assert f"{probed['duration_s']:.3f}" == APPROVED_DURATION_ROUNDED_S == "15.067"


def test_native_command_is_passthrough_and_the_old_api_is_unchanged():
    native = sample_frame_command(
        "ffmpeg", "clip.mp4", "images/%04d.jpg", fps=0, max_frames=0, native_frames=True
    )
    assert native == [
        "ffmpeg",
        "-y",
        "-i",
        "clip.mp4",
        "-fps_mode",
        "passthrough",
        "images/%04d.jpg",
    ]
    assert_native_argv(native)
    omitted = sample_frame_command(
        "ffmpeg", "clip.mp4", "images/%04d.jpg", fps=0, max_frames=0
    )
    assert omitted == ["ffmpeg", "-y", "-i", "clip.mp4", "images/%04d.jpg"]
    assert "-fps_mode" not in omitted
    default = sample_frame_command("ffmpeg", "clip.mp4", "images/%04d.jpg")
    assert "fps=1" in default
    assert default[default.index("-frames:v") + 1] == "16"
    kwargs = evidence_kwargs()
    assert kwargs == {
        "fps": 0,
        "max_frames": 0,
        "native_frames": True,
        "color_calibrate": True,
    }
    with pytest.raises(ValueError):
        sample_frame_command(
            "ffmpeg",
            "clip.mp4",
            "images/%04d.jpg",
            fps=30,
            max_frames=0,
            native_frames=True,
        )


def test_run_poses_forwards_native_frames_and_rejects_a_filter(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sceneagent.colmap_stage.shutil.which", lambda name: f"/usr/bin/{name}"
    )
    seen = []

    def fake_run(command, timeout_s=600):
        seen.append(list(command))
        return {
            "command": command,
            "returncode": 1,
            "timed_out": False,
            "stdout_tail": "",
            "stderr_tail": "",
        }

    monkeypatch.setattr("sceneagent.colmap_stage._run", fake_run)
    result = run_poses(
        tmp_path / "clip.mp4",
        tmp_path / "work",
        fps=0,
        max_frames=0,
        native_frames=True,
    )
    assert result["native_frames"] is True
    assert result["full_rate"] is True
    assert result["color_calibrate"] is True
    assert_native_argv(seen[0])
    with pytest.raises(ValueError):
        run_poses(tmp_path / "clip.mp4", tmp_path / "work", fps=1, max_frames=0, native_frames=True)


def test_forbidden_paths_are_rejected_before_stat(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("filesystem access")

    monkeypatch.setattr(os.path, "lexists", boom)
    monkeypatch.setattr(os.path, "islink", boom)
    monkeypatch.setattr(os, "readlink", boom)
    monkeypatch.setattr(os, "listdir", boom)
    for raw in (
        f"{SCENEAGENT_ROOT}/secret",
        f"{LEGACY_MODEL_ROOT}/sparse/0",
        "/tmp/ok/../../workspace/sceneagent-simfoundry/file",
        "/tmp/fitted_splat/model",
    ):
        assert path_is_forbidden(raw) is True
        with pytest.raises(ForbiddenPathError):
            assert_no_symlink(Path(os.path.normpath(raw)))


def test_symlink_to_a_forbidden_root_is_rejected_before_open(tmp_path, monkeypatch):
    link = tmp_path / "link"
    link.symlink_to(f"{SCENEAGENT_ROOT}/Fruits.mp4")
    opened = []

    def boom(*_args, **_kwargs):
        opened.append(True)
        raise AssertionError("opened")

    monkeypatch.setattr(Path, "open", boom)
    with pytest.raises(ForbiddenPathError):
        assert_no_symlink(link)
    assert opened == []
    benign = tmp_path / "benign"
    benign.symlink_to(tmp_path)
    with pytest.raises(ForbiddenPathError):
        assert_no_symlink(benign)


def test_gpu_query_is_aggregate_and_fails_closed_once():
    assert gpu_query_is_aggregate(GPU_FREE_ARGV) is True
    calls = []

    def fail(argv):
        calls.append(list(argv))
        return {"returncode": 1, "stdout": "", "stderr": "unavailable"}

    with pytest.raises(EvidenceError):
        query_gpu_free(fail)
    assert calls == [list(GPU_FREE_ARGV)]
    assert "pid" not in " ".join(GPU_FREE_ARGV)
    with pytest.raises(EvidenceError):
        interpret_gpu_free("0\n", 0)
    assert interpret_gpu_free("100\n40\n", 0)["aggregate_free_mib"] == 140


def test_launch_command_is_exact_and_blocked():
    assert launch_ready() is False
    text = render_launch_command(CODE_SHA)
    assert text.startswith("STATUS: BLOCKED\nready_for_execution: false\n")
    assert APPROVED_RELATIVE in text
    assert APPROVED_SHA256 in text
    assert CODE_SHA in text
    assert str(required_output_prefix(CODE_SHA)) in text
    assert INSTANCE_ID in text
    assert "--dry-run" not in text
    assert "--max-frames" not in text
    assert "fps=" not in text
    assert "ssh " not in text
    assert "<" not in text
    assert "TODO" not in text
    assert "placeholder" not in text.lower()
    assert counts_as_fruits_evidence(True, "accept") is False
    with pytest.raises(EvidenceError):
        render_launch_command("not-a-sha")


def test_manifest_has_no_self_hash(tmp_path):
    blob = b"package-bytes\n"
    (tmp_path / "runner.py").write_bytes(blob)
    manifest = build_manifest({"runner.py": blob}, code_sha=CODE_SHA)
    assert manifest["ready_for_execution"] is False
    assert manifest["instance_id"] == INSTANCE_ID
    assert manifest["code_sha"] == CODE_SHA
    assert manifest["code_short_sha"] == "aaaaaaa"
    assert manifest["inputs"]["sha256"] == APPROVED_SHA256
    assert manifest["inputs"]["status"].startswith("pinned")
    assert manifest["files"] == {"runner.py": hashlib.sha256(blob).hexdigest()}
    assert "manifest.json" not in manifest["files"]
    assert "self_sha256" not in manifest
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_package_sums(tmp_path)
    sums = (tmp_path / "PACKAGE_SHA256SUMS").read_text(encoding="utf-8")
    assert "manifest.json" in sums
    assert "PACKAGE_SHA256SUMS" not in sums
    assert hashlib.sha256(blob).hexdigest() in sums


def test_real_run_rejects_a_wrong_target_without_creating_the_prefix():
    code = main(
        [
            "--repo",
            str(ROOT),
            "--input",
            APPROVED_RELATIVE,
            "--input-sha256",
            APPROVED_SHA256,
            "--output-prefix",
            str(required_output_prefix(PINNED_BASE_SHA)),
            "--expect-code-sha",
            PINNED_BASE_SHA,
            "--instance-id",
            "999",
        ]
    )
    assert code == 2
    assert not (ROOT / "cut1-gpu").exists()
    assert not Path("/workspace/cut1-gpu").exists() or os.path.islink(
        "/workspace"
    )


def test_missing_hash_and_other_paths_are_rejected(tmp_path):
    video, digest = _synthetic_video(tmp_path)
    wrong = _dry_request(tmp_path, video, "0" * 64)
    assert execute(wrong) == 2
    ledger = json.loads((tmp_path / "out" / "run" / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["input"]["hash_match"] is False
    assert ledger["decision"] == "reject"
    assert ledger["counts_as_fruits_evidence"] is False
    assert not (tmp_path / "out" / "run" / "colmap").exists()
    recorded = (tmp_path / "out" / "run" / "ledger.sha256").read_text(encoding="utf-8").split()[0]
    body = (tmp_path / "out" / "run" / "ledger.json").read_bytes()
    assert recorded == hashlib.sha256(body).hexdigest()
    refused = tmp_path / "foreign"
    assert (
        execute(
            _dry_request(
                tmp_path,
                video,
                digest,
                input=f"{SCENEAGENT_ROOT}/Fruits.mp4",
                output_prefix=str(refused),
            )
        )
        == 2
    )
    assert not refused.exists()
    assert (
        execute(
            _dry_request(
                tmp_path,
                video,
                digest,
                input=str(Path(LEGACY_MODEL_ROOT) / "sparse" / "0"),
                output_prefix=str(tmp_path / "legacy-out"),
            )
        )
        == 2
    )


def test_reuse_and_version_and_gate_failure(tmp_path, monkeypatch):
    video, digest = _synthetic_video(tmp_path)
    stale = tmp_path / "stale"
    stale.mkdir()
    planted = stale / "model_analyzer_0.txt"
    planted.write_text("old-host-model\n", encoding="utf-8")
    assert reuse_refused(stale) is True
    assert execute(_dry_request(tmp_path, video, digest, output_prefix=str(stale))) == 2
    assert planted.read_text(encoding="utf-8") == "old-host-model\n"
    _install_stub(tmp_path / "bin", banner="COLMAP 4.2.1 -- Structure-from-Motion\n")
    monkeypatch.setenv(
        "PATH", str(tmp_path / "bin") + os.pathsep + os.environ["PATH"]
    )
    wrong_version = tmp_path / "version-out"
    assert execute(_dry_request(tmp_path, video, digest, output_prefix=str(wrong_version))) == 2
    assert not (wrong_version / "colmap").exists()
    ledger = json.loads((wrong_version / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["colmap_version"] == "4.2.1"
    assert ledger["legacy_model_reused"] is False
    assert parse_colmap_banner("COLMAP 4.2.1 and COLMAP 3.11.1\n") is None


def test_offline_dry_run_registers_every_native_frame(tmp_path, monkeypatch):
    video, digest = _synthetic_video(tmp_path)
    probed = _ffprobe(video)
    assert probed["frames"] > 16
    _install_stub(tmp_path / "bin")
    monkeypatch.setenv("PATH", str(tmp_path / "bin") + os.pathsep + os.environ["PATH"])
    monkeypatch.delenv("SF5_STUB_REPROJ", raising=False)
    output = tmp_path / "dry" / "run"
    code = execute(_dry_request(tmp_path, video, digest, output_prefix=str(output)))
    assert code == 0
    ledger = json.loads((output / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["decision"] == "accept"
    assert ledger["dry_run"] is True
    assert ledger["evidence_class"] == "synthetic_dry_run"
    assert ledger["counts_as_fruits_evidence"] is False
    assert ledger["legacy_model_reused"] is False
    assert ledger["config"]["native_frames"] is True
    assert ledger["config"]["fps"] == 0
    assert ledger["config"]["max_frames"] == 0
    assert ledger["config"]["color_calibrate"] is True
    assert ledger["config"]["colmap_required"] == REQUIRED_COLMAP_VERSION
    assert ledger["colmap_version"] == REQUIRED_COLMAP_VERSION
    assert ledger["extracted_frames"] == probed["frames"]
    assert ledger["extracted_frames"] > 16
    assert "-fps_mode" in ledger["ffmpeg_command"]
    assert "passthrough" in ledger["ffmpeg_command"]
    assert "-vf" not in ledger["ffmpeg_command"]
    assert "-frames:v" not in ledger["ffmpeg_command"]
    assert ledger["exit_statuses"]["ffmpeg"] == 0
    assert ledger["exit_statuses"]["feature_extractor"] == 0
    assert ledger["exit_statuses"]["sequential_matcher"] == 0
    assert ledger["exit_statuses"]["mapper"] == 0
    assert ledger["exit_statuses"]["reconstruction_gate_pass"] is True
    assert ledger["process_exit"] == 0
    assert ledger["code_sha"] == subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    assert ledger["base_sha"] == PINNED_BASE_SHA
    evidence = ledger["reconstruction_evidence"]
    assert evidence["gate_pass"] is True
    assert evidence["verdict"] == "pass"
    names = set(evidence["checksums"])
    assert "reconstruction_gate.json" in names
    assert "SHA256SUMS" in names
    assert any(name.startswith("model_analyzer_") for name in names)
    evidence_dir = output / "colmap" / "reconstruction_gate"
    sums = (evidence_dir / "SHA256SUMS").read_text(encoding="utf-8")
    for name, digest_hex in evidence["checksums"].items():
        if name == "SHA256SUMS":
            continue
        assert f"{digest_hex}  {name}" in sums
        assert hashlib.sha256((evidence_dir / name).read_bytes()).hexdigest() == digest_hex
    recorded = (output / "ledger.sha256").read_text(encoding="utf-8").split()[0]
    assert recorded == hashlib.sha256((output / "ledger.json").read_bytes()).hexdigest()
    assert gate_accepted({"ok": True, "reconstruction_gate": {"pass": True}}) is True
    assert output_allowed(output, dry_run=True, code_sha=None, repo=ROOT) is True
    assert (
        output_allowed(
            Path("/workspace/cut1-gpu/KAN-5-aaaaaaa/run"),
            dry_run=True,
            code_sha=None,
            repo=ROOT,
        )
        is False
    )


def test_gate_failure_exits_nonzero(tmp_path, monkeypatch):
    video, digest = _synthetic_video(tmp_path)
    _install_stub(tmp_path / "bin")
    monkeypatch.setenv("PATH", str(tmp_path / "bin") + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("SF5_STUB_REPROJ", "fail")
    output = tmp_path / "fail" / "run"
    code = execute(_dry_request(tmp_path, video, digest, output_prefix=str(output)))
    assert code == 1
    ledger = json.loads((output / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["decision"] == "reject"
    assert ledger["process_exit"] == 1
    assert ledger["counts_as_fruits_evidence"] is False
    assert ledger["exit_statuses"]["reconstruction_gate_pass"] is False
    assert ledger["failed_step"] == "reconstruction_gate"
    assert "mean_reproj_err_px" in ledger["reason"]
    evidence = output / "colmap" / "reconstruction_gate"
    assert (evidence / "model_analyzer_0.txt").is_file()
    assert (evidence / "reconstruction_gate.json").is_file()
    assert (evidence / "SHA256SUMS").is_file()
    assert gate_accepted({"ok": False, "reconstruction_gate": {"pass": False}}) is False


def test_package_source_does_not_control_other_processes():
    source = (ROOT / "sceneagent" / "sf5_colmap_evidence.py").read_text(encoding="utf-8")
    for banned in ("os.kill", "pkill", "shell=True", "query-compute-apps"):
        assert banned not in source
    assert shutil.which("nvidia-smi") is None or gpu_query_is_aggregate(GPU_FREE_ARGV)
