# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mutation checks for the SF-5 evidence package.

Every mutant is loaded and must fail an oracle. A mutant that still passes,
or a pattern that does not occur once, fails this module. Nothing is skipped.
"""

import importlib.util
import tempfile
from pathlib import Path

from sceneagent import sf5_colmap_evidence as package
from sceneagent.colmap_stage import _run, sample_frame_command
from tests.sceneagent.test_sf5_colmap_evidence import assert_stop_signals_only_our_child

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "sceneagent" / "sf5_colmap_evidence.py"
COLMAP = ROOT / "sceneagent" / "colmap_stage.py"
PIN = "a9c099cea15ec73bc203cff0925aa88f166acc2c75e672d9ff89d3bb34169b4d"

MUTANTS = (
    (
        "threshold_bypass",
        "package",
        'return poses.get("ok") is True and gate.get("pass") is True',
        "return True",
    ),
    (
        "ignore_run_status",
        "package",
        'return poses.get("ok") is True and gate.get("pass") is True',
        'return gate.get("pass") is True',
    ),
    (
        "ignore_gate",
        "package",
        'return poses.get("ok") is True and gate.get("pass") is True',
        'return poses.get("ok") is True',
    ),
    ("reject_exit_swallowed", "package", '"reject": 1,', '"reject": 0,'),
    ("identity_exit_swallowed", "package", '"identity": 2,', '"identity": 0,'),
    (
        "gpu_blind_retry",
        "package",
        '"""One query. A failure is not retried."""\n    return 1',
        '"""One query. A failure is not retried."""\n    return 3',
    ),
    (
        "hash_bypass",
        "package",
        "return bool(actual) and actual == expected",
        "return True",
    ),
    (
        "colmap_421_accepted",
        "package",
        "return version == REQUIRED_COLMAP_VERSION",
        "return True",
    ),
    (
        "dry_run_counts_as_evidence",
        "package",
        'return (not dry_run) and decision == "accept"',
        "return True",
    ),
    (
        "fitted_splat_allowed",
        "package",
        'frozenset({"fitted_splat", "sceneagent-simfoundry"})',
        'frozenset({"sceneagent-simfoundry"})',
    ),
    (
        "sceneagent_root_opened",
        "package",
        'SCENEAGENT_ROOT = "/workspace/sceneagent-simfoundry"\n'
        'LEGACY_MODEL_ROOT = "/workspace/fruits-fullrate-colmap"\n'
        'PRODUCTION_ROOT = "/workspace/cut1-gpu"\n'
        "FORBIDDEN_PREFIXES = (SCENEAGENT_ROOT, LEGACY_MODEL_ROOT)\n"
        'FORBIDDEN_COMPONENTS = frozenset({"fitted_splat", "sceneagent-simfoundry"})',
        'SCENEAGENT_ROOT = "/workspace/not-the-real-root"\n'
        'LEGACY_MODEL_ROOT = "/workspace/fruits-fullrate-colmap"\n'
        'PRODUCTION_ROOT = "/workspace/cut1-gpu"\n'
        "FORBIDDEN_PREFIXES = (SCENEAGENT_ROOT, LEGACY_MODEL_ROOT)\n"
        'FORBIDDEN_COMPONENTS = frozenset({"fitted_splat"})',
    ),
    (
        "legacy_model_renamed",
        "package",
        'LEGACY_MODEL_ROOT = "/workspace/fruits-fullrate-colmap"',
        'LEGACY_MODEL_ROOT = "/workspace/not-the-legacy-model"',
    ),
    (
        "reuse_allowed",
        "package",
        "return any(prefix.iterdir())",
        "return False",
    ),
    (
        "native_flag_cleared",
        "package",
        '"native_frames": True,',
        '"native_frames": False,',
    ),
    ("fps_subsampling", "package", '"fps": 0,', '"fps": 1,'),
    ("frame_cap_restored", "package", '"max_frames": 0,', '"max_frames": 16,'),
    (
        "calibration_disabled",
        "package",
        '"color_calibrate": True,',
        '"color_calibrate": False,',
    ),
    (
        "gpu_process_query",
        "package",
        '"--format=csv,noheader,nounits",',
        '"--format=csv,noheader,nounits",\n'
        '    "--query-compute-apps=pid,process_name",',
    ),
    (
        "shell_enabled",
        "package",
        '["git", "-C", str(repo), *args],\n'
        "        capture_output=True,\n"
        "        text=True,\n"
        "        check=False,\n"
        "        shell=False,",
        '["git", "-C", str(repo), *args],\n'
        "        capture_output=True,\n"
        "        text=True,\n"
        "        check=False,\n"
        "        shell=True,",
    ),
    (
        "process_kill_added",
        "package",
        '"""One query. A failure is not retried."""\n    return 1',
        '"""One query. A failure is not retried."""\n    os.kill(1, 9)\n    return 1',
    ),
    ("sf4_frame_count", "package", "APPROVED_FRAMES = 452", "APPROVED_FRAMES = 300"),
    (
        "pin_hash_changed",
        "package",
        f'APPROVED_SHA256 = "{PIN}"',
        'APPROVED_SHA256 = "' + PIN[:-1] + 'e"',
    ),
    (
        "short_output_allowed",
        "package",
        "return len(norm.parts) >= 4",
        "return True",
    ),
    (
        "stale_evidence_path",
        "package",
        "return actual == expected",
        "return True",
    ),
    (
        "checksum_bypass",
        "package",
        "return bool(listed) and listed == actual",
        "return True",
    ),
    (
        "symlink_check_removed",
        "package",
        "if os.path.islink(acc):",
        "if False and os.path.islink(acc):",
    ),
    (
        "short_gpu_hidden_by_another",
        "package",
        'return bool(rows) and all(row["free_mib"] >= budget for row in rows)',
        'return bool(rows) and any(row["free_mib"] >= budget for row in rows)',
    ),
    (
        "budget_not_pinned",
        "package",
        "or not proposed:",
        "or False:",
    ),
    (
        "sample_called_peak",
        "package",
        '        "sample_interval_s": GPU_SAMPLE_INTERVAL_S,\n'
        '        "exact_true_peak": False,\n'
        '        "continuous_minimum": False,',
        '        "sample_interval_s": GPU_SAMPLE_INTERVAL_S,\n'
        '        "exact_true_peak": True,\n'
        '        "continuous_minimum": False,',
    ),
    (
        "proposed_budget_lowered",
        "package",
        "PROPOSED_MIN_FREE_MIB = 6144",
        "PROPOSED_MIN_FREE_MIB = 1",
    ),
    (
        "sift_formula_short",
        "package",
        "return 4 * num_matches * num_matches + 4 * num_matches * 256",
        "return 4 * num_matches * 24",
    ),
    (
        "sample_interval_becomes_recheck",
        "package",
        "GPU_SAMPLE_INTERVAL_S = 10",
        "GPU_SAMPLE_INTERVAL_S = 60",
    ),
    (
        "sample_retry_enabled",
        "package",
        '"retries": 0,',
        '"retries": 1,',
    ),
    (
        "native_timeout_stays_at_16_frames",
        "package",
        '"colmap_timeout_s": SHORT_RUN_COLMAP_TIMEOUT_S * blocks,',
        '"colmap_timeout_s": SHORT_RUN_COLMAP_TIMEOUT_S,',
    ),
    (
        "calibration_failure_called_off",
        "package",
        '    if "color_calibration" not in poses:\n        return "failed"',
        '    if "color_calibration" not in poses:\n        return "off"',
    ),
    (
        "during_sample_stops_colmap",
        "package",
        '    """\n    return False\n',
        '    """\n    return True\n',
    ),
    (
        "gpu_stage_pipes_not_drained",
        "colmap",
        "        thread.start()\n",
        "        pass\n",
    ),
    (
        "outside_signal_sigusr1_twice",
        "colmap",
        '    """\n    if proc.poll() is not None:\n        return\n',
        '    """\n'
        "    try:\n"
        '        _sentinel = os.environ.get("SF5_UNRELATED_SENTINEL")\n'
        "        if _sentinel:\n"
        "            os.kill(int(_sentinel), 10)\n"
        "    except OSError:\n"
        "        pass\n"
        "    try:\n"
        '        _sentinel = os.environ.get("SF5_UNRELATED_SENTINEL")\n'
        "        if _sentinel:\n"
        "            os.kill(int(_sentinel), 10)\n"
        "    except OSError:\n"
        "        pass\n"
        "    if proc.poll() is not None:\n"
        "        return\n",
    ),
    (
        "native_extract_becomes_fps",
        "colmap",
        '"-fps_mode",\n            "passthrough",',
        '"-vf",\n            "fps=30",',
    ),
    (
        "native_extract_gains_a_cap",
        "colmap",
        '"-fps_mode",\n            "passthrough",',
        '"-frames:v",\n            "16",',
    ),
)


def _apply(text: str, old: str, new: str, name: str) -> str:
    count = text.count(old)
    if count != 1:
        raise AssertionError(f"{name} matched {count} times, not once")
    updated = text.replace(old, new, 1)
    if updated == text:
        raise AssertionError(f"{name} did not change the source")
    return updated


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"{name} did not load")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _oracle_package(mod) -> None:
    assert mod.gate_accepted({"ok": False, "reconstruction_gate": {"pass": True}}) is False
    assert mod.gate_accepted({"ok": True, "reconstruction_gate": {"pass": False}}) is False
    assert mod.gate_accepted({"ok": True, "reconstruction_gate": {"pass": True}}) is True
    assert mod.exit_code("reject") == 1
    assert mod.exit_code("identity") == 2
    assert mod.exit_code("resource") == 3
    assert mod.exit_code("accept") == 0
    assert mod.exit_code("gpu_budget") == 4
    assert mod.gpu_attempts() == 1
    assert mod.hashes_match("abc", "def") is False
    assert mod.hashes_match("abc", "abc") is True
    assert mod.colmap_version_allowed("4.2.1") is False
    assert mod.colmap_version_allowed("3.11.1") is True
    assert mod.colmap_version_allowed(None) is False
    assert mod.counts_as_fruits_evidence(True, "accept") is False
    assert mod.counts_as_fruits_evidence(False, "reject") is False
    assert mod.counts_as_fruits_evidence(False, "accept") is True
    assert mod.GPU_FREE_ARGV == (
        "nvidia-smi",
        "--query-gpu=memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    )
    assert mod.gpu_query_is_aggregate(mod.GPU_FREE_ARGV) is True
    assert "pid" not in " ".join(mod.GPU_FREE_ARGV)
    assert mod.PROPOSED_MIN_FREE_MIB == 6144
    assert mod.GPU_SAMPLE_INTERVAL_S == 10
    assert mod.FAQ_PINNED_TO_COLMAP_3_11_1 is False
    assert mod.sift_match_bytes(32768) == 4328521728
    assert mod.sift_match_bytes(8192) == 276824064
    assert mod.accepted_free_mib_budget(6144) == 6144
    for bad in (None, 1, 6143, 16384, True):
        try:
            mod.accepted_free_mib_budget(bad)
        except mod.GpuBudgetError:
            pass
        else:
            raise AssertionError(bad)
    assert (
        mod.memory_meets_budget([{"free_mib": 8000}, {"free_mib": 1000}], 6144) is False
    )
    assert mod.memory_meets_budget([{"free_mib": 6144}], 6144) is True
    assert mod.memory_meets_budget([], 6144) is False
    assert mod.summarize_free_samples([7000, 6100])["exact_true_peak"] is False
    assert mod.summarize_free_samples([7000, 6100])["lowest_free_mib_observed"] == 6100
    assert mod.summarize_free_samples([])["retries"] == 0
    assert mod.native_timeouts(452)["colmap_timeout_s"] == 900 * 29
    assert mod.native_timeouts(16)["colmap_timeout_s"] == 900
    assert (
        mod.calibration_outcome(
            {"color_calibrate": True, "failed_step": "color_calibration"}
        )
        == "failed"
    )
    assert mod.calibration_outcome({"color_calibrate": False}) == "off"
    assert mod.during_sample_is_terminal({"meets_budget": False}) is False
    assert mod.path_is_forbidden("/workspace/sceneagent-simfoundry/secret") is True
    assert mod.path_is_forbidden("/tmp/fitted_splat/a") is True
    assert mod.path_is_forbidden("/workspace/fruits-fullrate-colmap/sparse/0") is True
    assert mod.under_legacy("/workspace/fruits-fullrate-colmap/sparse/0") is True
    assert mod.legacy_model_was_used(["/tmp/owned/run"]) is False
    assert mod.evidence_kwargs() == {
        "fps": 0,
        "max_frames": 0,
        "native_frames": True,
        "color_calibrate": True,
    }
    assert mod.APPROVED_FRAMES == 452
    assert mod.APPROVED_SHA256 == PIN
    assert mod.pinned_file_bytes_match(ROOT) is True
    assert mod.evidence_path_allowed(
        Path("/tmp/owned/run"),
        Path("/workspace/fruits-fullrate-colmap/reconstruction_gate"),
    ) is False
    assert mod.sums_match("a" * 64, "b" * 64) is False
    assert mod.output_allowed(
        Path("/tmp/a"), dry_run=True, code_sha=None, repo=ROOT
    ) is False
    source = Path(mod.__file__).read_text(encoding="utf-8")
    assert "shell=True" not in source
    assert "os.kill" not in source
    assert "pkill" not in source
    assert "query-compute-apps" not in source
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        link = root / "link"
        link.symlink_to("/workspace/sceneagent-simfoundry/secret")
        try:
            mod.assert_no_symlink(link)
        except mod.ForbiddenPathError:
            pass
        else:
            raise AssertionError("symlink to a forbidden root was not rejected")
        benign = root / "benign"
        benign.symlink_to(root / "target")
        (root / "target").mkdir()
        try:
            mod.assert_no_symlink(benign)
        except mod.ForbiddenPathError:
            pass
        else:
            raise AssertionError("symlink component was not rejected")
        planted = root / "planted"
        planted.mkdir()
        (planted / "model_analyzer_0.txt").write_text("stale", encoding="utf-8")
        assert mod.reuse_refused(planted) is True
        assert mod.pinned_file_bytes_match(root) is False


def _oracle_colmap(mod) -> None:
    native = mod.sample_frame_command(
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
    omitted = mod.sample_frame_command(
        "ffmpeg", "clip.mp4", "images/%04d.jpg", fps=0, max_frames=0
    )
    assert omitted == ["ffmpeg", "-y", "-i", "clip.mp4", "images/%04d.jpg"]
    default = mod.sample_frame_command("ffmpeg", "clip.mp4", "images/%04d.jpg")
    assert "fps=1" in default and "16" in default
    if hasattr(mod, "_run"):
        _assert_stage_pipes_drain(mod._run)
    if hasattr(mod, "_stop_our_child"):
        assert_stop_signals_only_our_child(mod._stop_our_child)


def _assert_stage_pipes_drain(run):
    script = (
        "import sys\n"
        "chunk = b'x' * 4096\n"
        "for _ in range(64):\n"
        "    sys.stdout.buffer.write(chunk)\n"
        "    sys.stderr.buffer.write(chunk)\n"
    )
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "flood.py"
        path.write_text(script, encoding="utf-8")
        result = run(
            [__import__("sys").executable, str(path)],
            timeout_s=5,
            on_gpu_sample=lambda: {"logged": True},
            sample_interval_s=10,
        )
    assert result["timed_out"] is False
    assert result["returncode"] == 0


def test_unmutated_oracles_pass():
    _oracle_package(package)
    _oracle_colmap(type("Colmap", (), {"sample_frame_command": staticmethod(sample_frame_command)}))


def test_every_mutant_is_killed():
    package_text = PACKAGE.read_text(encoding="utf-8")
    colmap_text = COLMAP.read_text(encoding="utf-8")
    survived = []
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        for index, (name, kind, old, new) in enumerate(MUTANTS):
            source = package_text if kind == "package" else colmap_text
            updated = _apply(source, old, new, name)
            path = root / f"{index}-{name}.py"
            path.write_text(updated, encoding="utf-8")
            module = _load(path, f"sf5_mutant_{index}")
            try:
                if kind == "package":
                    _oracle_package(module)
                else:
                    _oracle_colmap(module)
            except Exception:
                continue
            survived.append(name)
    assert survived == []
    assert len(MUTANTS) == 40
