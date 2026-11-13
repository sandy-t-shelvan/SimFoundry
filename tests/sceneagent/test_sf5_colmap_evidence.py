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
import sys
import tempfile
import time
from pathlib import Path

import pytest

from sceneagent.colmap_stage import (
    command_uses_gpu,
    poll_gpu_stage,
    pose_commands,
    run_poses,
    sample_frame_command,
    _run,
    _stop_our_child,
)
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
    FAQ_PINNED_TO_COLMAP_3_11_1,
    GPU_FREE_ARGV,
    GPU_SAMPLE_INTERVAL_S,
    INSTANCE_ID,
    PROPOSED_MIN_FREE_MIB,
    LEGACY_MODEL_ROOT,
    PINNED_BASE_SHA,
    REQUIRED_COLMAP_VERSION,
    SCENEAGENT_ROOT,
    EvidenceError,
    ForbiddenPathError,
    GpuBudgetError,
    GpuQueryTimeout,
    assert_native_argv,
    assert_no_symlink,
    build_manifest,
    counts_as_fruits_evidence,
    during_sample_is_terminal,
    evidence_kwargs,
    execute,
    fruits_probe_matches,
    gate_accepted,
    accepted_free_mib_budget,
    gpu_query_is_aggregate,
    memory_meets_budget,
    parse_gpu_memory_csv,
    main,
    output_allowed,
    parse_colmap_banner,
    path_is_forbidden,
    pinned_file_bytes_match,
    query_gpu_memory,
    _nvidia_smi,
    sift_match_bytes,
    summarize_free_samples,
    _step_statuses,
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
if cmd == "feature_extractor":
    log = os.environ.get("SF5_STUB_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as handle:
            handle.write("feature_extractor\\n")
    raise SystemExit(0)
if cmd == "sequential_matcher":
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


_MEDIA_FRAMES = 20


def _synthetic_video(directory: Path) -> tuple[Path, str]:
    path = directory / "synthetic.mp4"
    path.write_bytes(b"synthetic-video-bytes")
    return path, _sha256(path)


def _install_media_stubs(directory: Path, frames: int = _MEDIA_FRAMES) -> None:
    """ffmpeg and ffprobe stand-ins so the unit job does not need the binaries."""
    directory.mkdir(parents=True, exist_ok=True)
    ffmpeg = directory / "ffmpeg"
    ffmpeg.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "from pathlib import Path\n"
        "from PIL import Image\n"
        "if '-h' in sys.argv:\n"
        "    sys.stdout.write('-fps_mode passthrough\\n')\n"
        "    raise SystemExit(0)\n"
        "pattern = Path(sys.argv[-1])\n"
        "pattern.parent.mkdir(parents=True, exist_ok=True)\n"
        "image = Image.new('RGB', (8, 8), (40, 80, 120))\n"
        "import os\n"
        f"count = int(os.environ.get('SF5_EXTRACT_FRAMES', '{frames}'))\n"
        "for index in range(1, count + 1):\n"
        "    image.save(pattern.parent / f'{index:04d}.jpg', quality=95)\n",
        encoding="utf-8",
    )
    ffmpeg.chmod(0o755)
    probe = directory / "ffprobe"
    probe.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "print(json.dumps({'streams': [{'nb_read_frames': '%s', 'nb_frames': '%s', "
        "'width': 8, 'height': 8, 'duration': '2.0'}]}))\n" % (frames, frames),
        encoding="utf-8",
    )
    probe.chmod(0o755)


def _install_stub(
    directory: Path, banner: str = "COLMAP 3.11.1 -- Structure-from-Motion\n"
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    _install_media_stubs(directory)
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
        "gpu_free_mib_min": 6144,
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
    if shutil.which("ffprobe"):
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


def test_gpu_budget_is_explicit_and_fail_closed():
    assert GPU_FREE_ARGV == (
        "nvidia-smi",
        "--query-gpu=memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    )
    assert gpu_query_is_aggregate(GPU_FREE_ARGV) is True
    assert "pid" not in " ".join(GPU_FREE_ARGV)
    assert "compute-apps" not in " ".join(GPU_FREE_ARGV)
    assert PROPOSED_MIN_FREE_MIB == 6144
    assert GPU_SAMPLE_INTERVAL_S == 10
    assert FAQ_PINNED_TO_COLMAP_3_11_1 is False
    assert accepted_free_mib_budget(6144) == 6144
    assert accepted_free_mib_budget("6144") == 6144
    for bad in (None, 0, -1, 1, 6143, 16384, True, 6144.0, "0"):
        with pytest.raises(GpuBudgetError):
            accepted_free_mib_budget(bad)
    assert sift_match_bytes(32768) == 4328521728
    assert sift_match_bytes(32768) // (1024 * 1024) == 4128
    assert sift_match_bytes(8192) == 276824064
    assert sift_match_bytes(8192) // (1024 * 1024) == 264
    assert PROPOSED_MIN_FREE_MIB != 4128
    calls = []

    def fail(argv):
        calls.append(list(argv))
        return {"returncode": 1, "stdout": "", "stderr": "unavailable"}

    with pytest.raises(EvidenceError):
        query_gpu_memory(fail, 6144)
    assert calls == [list(GPU_FREE_ARGV)]
    with pytest.raises(EvidenceError):
        parse_gpu_memory_csv("100\n40\n", 0)
    with pytest.raises(EvidenceError):
        parse_gpu_memory_csv("", 0)
    with pytest.raises(EvidenceError):
        parse_gpu_memory_csv("8192, 100, 9000\n", 0)
    short = parse_gpu_memory_csv("8192, 2048, 6143\n8192, 100, 8000\n", 0)
    assert memory_meets_budget(short, 6144) is False
    exact = parse_gpu_memory_csv("16384, 10240, 6144\n", 0)
    assert memory_meets_budget(exact, 6144) is True
    both = parse_gpu_memory_csv("16384, 10240, 6144\n16384, 10240, 6144\n", 0)
    assert memory_meets_budget(both, 6144) is True
    assert memory_meets_budget([], 6144) is False
    summary = summarize_free_samples([7000, 6200, 6500])
    assert summary["lowest_free_mib_observed"] == 6200
    assert summary["exact_true_peak"] is False
    assert summary["retries"] == 0
    assert summary["sample_interval_s"] == 10


def test_launch_command_is_exact():
    text = render_launch_command(CODE_SHA)
    assert "BLOCKED" not in text
    assert INSTANCE_ID == "60254439"
    assert APPROVED_RELATIVE in text
    assert APPROVED_SHA256 in text
    assert CODE_SHA in text
    assert str(required_output_prefix(CODE_SHA)) in text
    assert INSTANCE_ID in text
    assert "--gpu-free-mib-min 6144" in text
    assert "gpu_budget_approved: false" in text
    assert "gpu_budget_status: provisional" in text
    assert "6:47 PM ET" in text
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
    assert "ready_for_execution" not in manifest
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
    _install_stub(tmp_path / "bin")
    monkeypatch.setenv("PATH", str(tmp_path / "bin") + os.pathsep + os.environ["PATH"])
    probed = _ffprobe(video)
    assert probed["frames"] > 16
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
    assert ledger["config"]["gpu_free_mib_min"] == 6144
    assert ledger["config"]["gpu_budget_approved"] is False
    before = [row for row in ledger["gpu_memory"] if row["when"] == "before"]
    assert [row["stage"] for row in before] == ["mapper"]
    assert all(row["source"] == "offline_fixture" for row in before)
    assert all(row["min_free_mib"] >= 6144 for row in before)
    assert all(row["exact_true_peak"] is False for row in before)
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


def test_absent_budget_stops_before_the_video_probe(tmp_path, monkeypatch):
    video, digest = _synthetic_video(tmp_path)

    def boom(_path):
        raise AssertionError("ffprobe")

    monkeypatch.setattr("sceneagent.sf5_colmap_evidence._ffprobe", boom)
    code = execute(_dry_request(tmp_path, video, digest, gpu_free_mib_min=None))
    assert code == 4
    ledger = json.loads((tmp_path / "out" / "run" / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["process_exit"] == 4
    assert ledger["config"]["gpu_budget_approved"] is False
    assert "6144" in ledger["reason"]


def test_short_free_memory_stops_before_the_mapper(tmp_path, monkeypatch):
    video, digest = _synthetic_video(tmp_path)
    _install_stub(tmp_path / "bin")
    monkeypatch.setenv("PATH", str(tmp_path / "bin") + os.pathsep + os.environ["PATH"])
    stage_log = tmp_path / "stages.log"
    monkeypatch.setenv("SF5_STUB_LOG", str(stage_log))
    output = tmp_path / "short" / "run"
    code = execute(
        _dry_request(
            tmp_path,
            video,
            digest,
            output_prefix=str(output),
            memory_stdout="8192, 7000, 100\n8192, 100, 8000\n",
        )
    )
    assert code == 4
    assert stage_log.read_text(encoding="utf-8") == "feature_extractor\n"
    ledger = json.loads((output / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["process_exit"] == 4
    assert ledger["gpu_memory"][0]["stage"] == "mapper"
    assert ledger["gpu_memory"][0]["when"] == "before"
    assert ledger["gpu_memory"][0]["min_free_mib"] == 100
    assert ledger["gpu_memory"][0]["meets_budget"] is False
    assert "mapper" not in ledger.get("exit_statuses", {})


def test_mapper_gpu_check_follows_the_ba_flag():
    commands = pose_commands("/tmp/db", "/tmp/images", "/tmp/sparse", True, major=3)
    by_name = {argv[1]: argv for argv in commands}
    assert by_name["feature_extractor"][by_name["feature_extractor"].index("--SiftExtraction.use_gpu") + 1] == "0"
    assert by_name["sequential_matcher"][by_name["sequential_matcher"].index("--SiftMatching.use_gpu") + 1] == "0"
    assert command_uses_gpu(by_name["feature_extractor"], cuda_build=True) is False
    assert command_uses_gpu(by_name["sequential_matcher"], cuda_build=True) is False
    assert command_uses_gpu(by_name["mapper"], cuda_build=True) is True
    assert command_uses_gpu(by_name["mapper"], cuda_build=False) is False
    cpu_mapper = list(by_name["mapper"]) + ["--Mapper.ba_use_gpu", "0"]
    assert command_uses_gpu(cpu_mapper, cuda_build=True) is False
    gpu_mapper = list(by_name["mapper"]) + ["--Mapper.ba_use_gpu", "1"]
    assert command_uses_gpu(gpu_mapper, cuda_build=False) is True
    cpu_extract = pose_commands("/tmp/db", "/tmp/images", "/tmp/sparse", False, major=3)
    assert command_uses_gpu(cpu_extract[0], cuda_build=False) is False


def test_during_stage_samples_use_the_ten_second_interval():
    clock = {"t": 0.0}

    def monotonic():
        return clock["t"]

    def sleeper(seconds):
        clock["t"] += seconds

    def poll():
        if clock["t"] >= 25:
            return 0
        return None

    calls = []

    def on_sample():
        calls.append(clock["t"])

    watched = poll_gpu_stage(
        poll,
        on_sample,
        interval_s=10,
        timeout_s=100,
        sleeper=sleeper,
        monotonic=monotonic,
    )
    assert watched["status"] == "exited"
    assert calls == [10.0, 20.0]


def test_native_timeout_keeps_the_16_frame_ceiling_and_scales():
    import inspect

    signature = inspect.signature(run_poses)
    assert signature.parameters["ffmpeg_timeout_s"].default == 180
    assert signature.parameters["colmap_timeout_s"].default == 900
    from sceneagent.sf5_colmap_evidence import native_timeouts

    assert native_timeouts(16) == {"ffmpeg_timeout_s": 180, "colmap_timeout_s": 900}
    assert native_timeouts(452)["colmap_timeout_s"] == 900 * 29
    assert native_timeouts(452)["ffmpeg_timeout_s"] == 180 * 29
    assert native_timeouts(452)["colmap_timeout_s"] > 900


def test_native_run_passes_the_scaled_timeout(tmp_path, monkeypatch):
    from sceneagent.sf5_colmap_evidence import native_timeouts

    video, digest = _synthetic_video(tmp_path)
    _install_stub(tmp_path / "bin")
    monkeypatch.setenv("PATH", str(tmp_path / "bin") + os.pathsep + os.environ["PATH"])
    probed = _ffprobe(video)
    seen = {}

    def fake_run_poses(*_args, **kwargs):
        seen.update(kwargs)
        raise EvidenceError("stop after reading the timeout")

    monkeypatch.setattr("sceneagent.sf5_colmap_evidence.run_poses", fake_run_poses)
    code = execute(_dry_request(tmp_path, video, digest, output_prefix=str(tmp_path / "timeout" / "run")))
    assert code == 2
    assert seen["colmap_timeout_s"] == native_timeouts(probed["frames"])["colmap_timeout_s"]
    assert seen["ffmpeg_timeout_s"] == native_timeouts(probed["frames"])["ffmpeg_timeout_s"]
    assert seen["colmap_timeout_s"] > 900


def test_calibration_failure_stays_a_reject(tmp_path, monkeypatch):
    video, digest = _synthetic_video(tmp_path)
    _install_stub(tmp_path / "bin")
    monkeypatch.setenv("PATH", str(tmp_path / "bin") + os.pathsep + os.environ["PATH"])

    def boom(_directory):
        raise ValueError("calibration boom")

    monkeypatch.setattr(
        "sceneagent.colmap_stage.apply_per_image_color_calibration", boom
    )
    output = tmp_path / "cal" / "run"
    code = execute(_dry_request(tmp_path, video, digest, output_prefix=str(output)))
    assert code == 1
    ledger = json.loads((output / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["decision"] == "reject"
    assert ledger["process_exit"] == 1
    assert ledger["failed_step"] == "color_calibration"
    assert ledger["reason"] == "calibration boom"
    assert "was off" not in ledger["reason"]


def assert_stop_signals_only_our_child(stop):
    """``stop`` may signal only the child it is given.

    The sentinel is a process this test spawned and then does not pass to
    ``stop``. A handler records SIGUSR1 (10) and SIGTERM. The test does not
    list processes. Cleanup signals only those two Popen objects.
    """
    log_path = Path(tempfile.mkstemp(prefix="sf5-sentinel-")[1])
    sentinel_code = (
        "import signal, sys, time\n"
        "path = sys.argv[1]\n"
        "def remember(signum, _frame):\n"
        "    handle = open(path, 'a', encoding='utf-8')\n"
        "    handle.write('%s\\n' % signum)\n"
        "    handle.close()\n"
        "signal.signal(signal.SIGUSR1, remember)\n"
        "signal.signal(signal.SIGTERM, remember)\n"
        "sys.stdout.write('ready\\n')\n"
        "sys.stdout.flush()\n"
        "while True:\n"
        "    time.sleep(0.2)\n"
    )
    sentinel = subprocess.Popen(
        [sys.executable, "-c", sentinel_code, str(log_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    previous = os.environ.get("SF5_UNRELATED_SENTINEL")
    os.environ["SF5_UNRELATED_SENTINEL"] = str(sentinel.pid)
    try:
        ready = sentinel.stdout.readline()
        assert ready.strip() == "ready"
        assert sentinel.poll() is None
        stop(child)
        assert child.poll() is not None, "own child was not stopped"
        assert sentinel.poll() is None, "unrelated sentinel was stopped"
        recorded = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        assert recorded == "", recorded
    finally:
        if previous is None:
            os.environ.pop("SF5_UNRELATED_SENTINEL", None)
        else:
            os.environ["SF5_UNRELATED_SENTINEL"] = previous
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        if sentinel.poll() is None:
            sentinel.terminate()
            try:
                sentinel.wait(timeout=2)
            except subprocess.TimeoutExpired:
                sentinel.kill()
                sentinel.wait(timeout=5)
        log_path.unlink(missing_ok=True)


def test_stop_our_child_does_not_signal_an_unrelated_sentinel():
    assert_stop_signals_only_our_child(_stop_our_child)


def test_timed_out_stage_reaps_a_child_that_ignores_terminate():
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
        ]
    )
    _stop_our_child(proc)
    assert proc.poll() is not None
    assert proc.returncode is not None


def test_during_sample_does_not_stop_colmap():
    assert during_sample_is_terminal({"meets_budget": False, "min_free_mib": 1}) is False


def test_gpu_stage_drains_pipes_while_sampling(tmp_path):
    script = tmp_path / "flood.py"
    script.write_text(
        "import sys\n"
        "chunk = b'x' * 4096\n"
        "for _ in range(64):\n"
        "    sys.stdout.buffer.write(chunk)\n"
        "    sys.stderr.buffer.write(chunk)\n",
        encoding="utf-8",
    )
    result = _run(
        [sys.executable, str(script)],
        timeout_s=5,
        on_gpu_sample=lambda: {"logged": True},
        sample_interval_s=10,
    )
    assert result["timed_out"] is False
    assert result["returncode"] == 0
    assert len(result["stdout_tail"]) == 2000
    assert len(result["stderr_tail"]) == 2000


def test_step_status_records_a_timeout_separately_from_the_return_code():
    statuses = _step_statuses(
        {
            "steps": {
                "ffmpeg": {"returncode": 0, "timed_out": False},
                "mapper": {"returncode": None, "timed_out": True},
            },
            "reconstruction_gate": {},
        }
    )
    assert statuses["ffmpeg"] == 0
    assert statuses["ffmpeg_timed_out"] is False
    assert statuses["mapper"] is None
    assert statuses["mapper_timed_out"] is True


def test_timeout_reason_is_written_on_the_ledger(tmp_path, monkeypatch):
    video, digest = _synthetic_video(tmp_path)
    _install_stub(tmp_path / "bin")
    monkeypatch.setenv("PATH", str(tmp_path / "bin") + os.pathsep + os.environ["PATH"])
    probed = _ffprobe(video)

    def fake_run_poses(*_args, **_kwargs):
        return {
            "ran": True,
            "ok": False,
            "failed_step": "mapper",
            "reason": "mapper timed out",
            "color_calibrate": True,
            "color_calibration": {"applied": True},
            "frames": [f"{index:04d}.jpg" for index in range(1, probed["frames"] + 1)],
            "steps": {
                "ffmpeg": {
                    "command": ["ffmpeg", "-fps_mode", "passthrough", str(video)],
                    "returncode": 0,
                    "timed_out": False,
                },
                "mapper": {"returncode": None, "timed_out": True},
            },
        }

    monkeypatch.setattr("sceneagent.sf5_colmap_evidence.run_poses", fake_run_poses)
    output = tmp_path / "timeout-ledger" / "run"
    code = execute(_dry_request(tmp_path, video, digest, output_prefix=str(output)))
    assert code == 1
    ledger = json.loads((output / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["reason"] == "mapper timed out"
    assert ledger["failed_step"] == "mapper"
    assert ledger["exit_statuses"]["mapper"] is None
    assert ledger["exit_statuses"]["mapper_timed_out"] is True


def test_ffmpeg_failure_is_not_recorded_as_a_frame_count(tmp_path, monkeypatch):
    video, digest = _synthetic_video(tmp_path)
    _install_stub(tmp_path / "bin")
    monkeypatch.setenv("PATH", str(tmp_path / "bin") + os.pathsep + os.environ["PATH"])
    probed = _ffprobe(video)
    command = ["ffmpeg", "-fps_mode", "passthrough", str(video)]

    def fake_run_poses(*_args, **_kwargs):
        return {
            "ran": True,
            "ok": False,
            "failed_step": "ffmpeg",
            "reason": "ffmpeg timed out",
            "color_calibrate": True,
            "frames": ["0001.jpg", "0002.jpg"],
            "steps": {
                "ffmpeg": {
                    "command": command,
                    "returncode": None,
                    "timed_out": True,
                }
            },
        }

    monkeypatch.setattr("sceneagent.sf5_colmap_evidence.run_poses", fake_run_poses)
    output = tmp_path / "ffmpeg-timeout" / "run"
    code = execute(_dry_request(tmp_path, video, digest, output_prefix=str(output)))
    assert code == 2
    ledger = json.loads((output / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["failed_step"] == "ffmpeg"
    assert ledger["reason"] == "ffmpeg timed out"
    assert ledger["exit_statuses"]["ffmpeg"] is None
    assert ledger["exit_statuses"]["ffmpeg_timed_out"] is True
    assert ledger["extracted_frames"] == 2
    assert probed["frames"] != 2


def test_frame_count_mismatch_stops_before_colmap(tmp_path, monkeypatch):
    video, digest = _synthetic_video(tmp_path)
    _install_stub(tmp_path / "bin")
    monkeypatch.setenv("PATH", str(tmp_path / "bin") + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("SF5_EXTRACT_FRAMES", "3")
    stage_log = tmp_path / "stages.log"
    monkeypatch.setenv("SF5_STUB_LOG", str(stage_log))
    output = tmp_path / "frames" / "run"
    code = execute(_dry_request(tmp_path, video, digest, output_prefix=str(output)))
    assert code == 2
    assert not stage_log.exists()
    ledger = json.loads((output / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["failed_step"] == "frame_count"
    assert ledger["reason"] == "extracted frame count is not the native count"
    assert ledger["extracted_frames"] == 3
    assert "feature_extractor" not in ledger["exit_statuses"]


def test_gpu_query_timeout_fails_closed_and_stops_only_our_stage_child(
    tmp_path, monkeypatch
):
    query_log = tmp_path / "queries.log"
    query_bin = tmp_path / "bin"
    query_bin.mkdir()
    sleeper = query_bin / "nvidia-smi"
    sleeper.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "import time\n"
        "from pathlib import Path\n"
        "path = Path(os.environ['SF5_GPU_QUERY_LOG'])\n"
        "handle = path.open('a', encoding='utf-8')\n"
        "handle.write('start\\n')\n"
        "handle.close()\n"
        "while True:\n"
        "    handle = path.open('a', encoding='utf-8')\n"
        "    handle.write('beat\\n')\n"
        "    handle.close()\n"
        "    time.sleep(0.05)\n",
        encoding="utf-8",
    )
    sleeper.chmod(0o755)
    monkeypatch.setenv("SF5_GPU_QUERY_LOG", str(query_log))
    monkeypatch.setenv("PATH", str(query_bin) + os.pathsep + os.environ["PATH"])
    monkeypatch.setattr(
        "sceneagent.sf5_colmap_evidence.GPU_QUERY_TIMEOUT_S", 0.2
    )
    stage_stamp = tmp_path / "stage-stamp.txt"
    stage = tmp_path / "stage.py"
    stage.write_text(
        "import sys\n"
        "import time\n"
        "from pathlib import Path\n"
        "path = Path(sys.argv[1])\n"
        "while True:\n"
        "    path.write_text(str(time.time()), encoding='utf-8')\n"
        "    time.sleep(0.05)\n",
        encoding="utf-8",
    )
    log_path = Path(tempfile.mkstemp(prefix="sf5-sentinel-")[1])
    sentinel = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import signal, sys, time\n"
                "path = sys.argv[1]\n"
                "def remember(signum, _frame):\n"
                "    handle = open(path, 'a', encoding='utf-8')\n"
                "    handle.write('%s\\n' % signum)\n"
                "    handle.close()\n"
                "signal.signal(signal.SIGUSR1, remember)\n"
                "signal.signal(signal.SIGTERM, remember)\n"
                "sys.stdout.write('ready\\n')\n"
                "sys.stdout.flush()\n"
                "while True:\n"
                "    time.sleep(0.2)\n"
            ),
            str(log_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        assert sentinel.stdout.readline().strip() == "ready"
        def sample():
            _nvidia_smi(list(GPU_FREE_ARGV))

        with pytest.raises(GpuQueryTimeout, match="^gpu_query_timeout$"):
            _run(
                [sys.executable, str(stage), str(stage_stamp)],
                timeout_s=5,
                on_gpu_sample=sample,
                sample_interval_s=0.05,
            )
        frozen = stage_stamp.read_text(encoding="utf-8")
        time.sleep(0.3)
        assert stage_stamp.read_text(encoding="utf-8") == frozen
        starts = query_log.read_text(encoding="utf-8").count("start\n")
        beats = query_log.read_text(encoding="utf-8")
        time.sleep(0.25)
        assert starts == 1
        assert query_log.read_text(encoding="utf-8") == beats
        assert sentinel.poll() is None
        recorded = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        assert recorded == ""
    finally:
        if sentinel.poll() is None:
            sentinel.terminate()
            try:
                sentinel.wait(timeout=2)
            except subprocess.TimeoutExpired:
                sentinel.kill()
                sentinel.wait(timeout=5)
        log_path.unlink(missing_ok=True)
    video, digest = _synthetic_video(tmp_path)
    _install_stub(tmp_path / "ledger-bin")
    monkeypatch.setenv(
        "PATH", str(tmp_path / "ledger-bin") + os.pathsep + os.environ["PATH"]
    )

    def raise_query_timeout(*_args, **_kwargs):
        raise GpuQueryTimeout("gpu_query_timeout")

    monkeypatch.setattr(
        "sceneagent.sf5_colmap_evidence.run_poses", raise_query_timeout
    )
    output = tmp_path / "query-timeout" / "run"
    code = execute(_dry_request(tmp_path, video, digest, output_prefix=str(output)))
    assert code != 0
    ledger = json.loads((output / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["reason"] == "gpu_query_timeout"


def test_gpu_query_timeout_exits_3_not_2(tmp_path, monkeypatch):
    """KAN-29. GpuQueryTimeout is a resource failure, exit 3.

    The reason string is lowercase, so a classifier that only looks for
    the letters GPU leaves this as identity exit 2.
    """
    video, digest = _synthetic_video(tmp_path)
    _install_stub(tmp_path / "ledger-bin")
    monkeypatch.setenv(
        "PATH", str(tmp_path / "ledger-bin") + os.pathsep + os.environ["PATH"]
    )

    def raise_query_timeout(*_args, **_kwargs):
        raise GpuQueryTimeout("gpu_query_timeout")

    monkeypatch.setattr(
        "sceneagent.sf5_colmap_evidence.run_poses", raise_query_timeout
    )
    output = tmp_path / "query-timeout-exit" / "run"
    code = execute(_dry_request(tmp_path, video, digest, output_prefix=str(output)))
    ledger = json.loads((output / "ledger.json").read_text(encoding="utf-8"))
    assert code == 3
    assert ledger["process_exit"] == 3
    assert ledger["reason"] == "gpu_query_timeout"
    assert code != 2
    assert code != 4


def test_colmap_crash_reason_includes_return_code(tmp_path, monkeypatch):
    video, _digest = _synthetic_video(tmp_path)
    _install_stub(tmp_path / "bin")
    monkeypatch.setenv("PATH", str(tmp_path / "bin") + os.pathsep + os.environ["PATH"])

    def fake_run(command, timeout_s=600, **_kwargs):
        if Path(command[0]).name == "ffmpeg":
            dest = Path(str(command[-1]).replace("%04d", "0001"))
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"frame")
            return {
                "command": command,
                "returncode": 0,
                "timed_out": False,
                "stdout_tail": "",
                "stderr_tail": "",
                "gpu_samples": [],
            }
        assert command[1] == "feature_extractor"
        return {
            "command": command,
            "returncode": 139,
            "timed_out": False,
            "stdout_tail": "",
            "stderr_tail": "crash",
            "gpu_samples": [],
        }

    monkeypatch.setattr("sceneagent.colmap_stage._run", fake_run)
    result = run_poses(
        video,
        tmp_path / "colmap",
        color_calibrate=False,
    )
    assert result["failed_step"] == "feature_extractor"
    assert result["reason"]
    assert "139" in result["reason"]
    assert "timed out" not in result["reason"]


def test_package_source_does_not_control_other_processes():
    source = (ROOT / "sceneagent" / "sf5_colmap_evidence.py").read_text(encoding="utf-8")
    for banned in ("os.kill", "pkill", "shell=True", "query-compute-apps"):
        assert banned not in source
    assert shutil.which("nvidia-smi") is None or gpu_query_is_aggregate(GPU_FREE_ARGV)
