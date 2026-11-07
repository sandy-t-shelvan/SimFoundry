# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Held-out ns-eval report. No nerfstudio, GPU, or training subprocess."""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from simfoundry.pipeline import ns_eval_report
from simfoundry.pipeline.ns_eval_report import (
    NS_EVAL_TIMEOUT_S,
    discard_stale_ns_eval_report,
    write_heldout_ns_eval_report,
)

ROOT = Path(__file__).resolve().parents[1]
STAGE_2C = ROOT / "scripts/pipeline/A_reconstruction/stages/2c_train_bg_splat.py"
STAGE_5 = (
    ROOT
    / "scripts/pipeline/A_reconstruction/stages/auto_bg_reconstruction/5_train_bg_splat.py"
)

_NS_EVAL_JSON = {
    "experiment_name": "unnamed",
    "method_name": "splatfacto",
    "checkpoint": "nerfstudio_models/step-000000005.ckpt",
    "results": {
        "psnr": 21.5,
        "psnr_std": 0.4,
        "ssim": 0.81,
        "ssim_std": 0.02,
        "lpips": 0.23,
        "lpips_std": 0.01,
        "fps": 12.0,
    },
}


@pytest.fixture
def restore_cwd():
    cwd = os.getcwd()
    try:
        yield
    finally:
        os.chdir(cwd)


def _config(tmp_path: Path, text: str = "eval_mode: fraction\n") -> Path:
    path = tmp_path / "config.yml"
    path.write_text(text, encoding="utf-8")
    return path


def _write_raw(argv: list[str], body: dict) -> None:
    out = Path(argv[argv.index("--output-path") + 1])
    out.write_text(json.dumps(body), encoding="utf-8")


def _assert_eval_argv(argv: list[str], config: Path, prefix: list[str]) -> None:
    assert argv[: len(prefix)] == prefix
    rest = argv[len(prefix) :]
    assert rest[0] == "ns-eval"
    assert rest[1] == "--load-config"
    assert Path(rest[2]).resolve() == Path(config).resolve()
    assert rest[3] == "--output-path"
    assert Path(rest[4]).is_absolute()
    assert "--render-output-path" not in argv
    assert "train" not in argv
    assert Path(rest[4]) != (config.parent / "ns_eval.json").resolve()


def _success_run(config: Path, prefix: list[str] | None = None):
    prefix = prefix or []
    seen: list[list[str]] = []

    def run(argv, **kwargs):
        seen.append(list(argv))
        assert kwargs["timeout"] == NS_EVAL_TIMEOUT_S
        assert kwargs["check"] is False
        _assert_eval_argv(list(argv), config, prefix)
        _write_raw(list(argv), _NS_EVAL_JSON)
        return subprocess.CompletedProcess(argv, 0, "", "")

    return run, seen


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_success_copies_heldout_metrics_and_replaces_a_stale_report(tmp_path):
    config = _config(tmp_path)
    report = tmp_path / "ns_eval.json"
    report.write_text(json.dumps({"ok": True, "psnr": 1, "ssim": 1, "lpips": 1}), encoding="utf-8")
    run, seen = _success_run(config)
    payload = write_heldout_ns_eval_report(
        load_config=config,
        report_path=report,
        argv_prefix=[],
        cwd=tmp_path,
        env={"KEEP": "1"},
        run=run,
    )
    assert seen
    assert payload["ok"] is True
    assert payload["report_only"] is True
    assert payload["split"] == "eval"
    assert payload["psnr"] == 21.5
    assert payload["ssim"] == 0.81
    assert payload["lpips"] == 0.23
    assert payload["checkpoint"] == "nerfstudio_models/step-000000005.ckpt"
    assert payload["error"] is None
    assert payload["returncode"] == 0
    written = _read(report)
    assert written["psnr"] == 21.5
    assert list(tmp_path.glob(".ns_eval_*")) == []


def test_nonzero_exit_does_not_keep_scores_written_beside_the_failure(tmp_path):
    config = _config(tmp_path)
    report = tmp_path / "ns_eval.json"
    report.write_text(json.dumps({"ok": True, "psnr": 30, "ssim": 1, "lpips": 0}), encoding="utf-8")

    def run(argv, **kwargs):
        _write_raw(argv, _NS_EVAL_JSON)
        return subprocess.CompletedProcess(argv, 2, "", "dataloader empty")

    payload = write_heldout_ns_eval_report(
        load_config=config,
        report_path=report,
        argv_prefix=[],
        cwd=tmp_path,
        env=None,
        run=run,
    )
    assert payload["ok"] is False
    assert payload["psnr"] is None
    assert payload["ssim"] is None
    assert payload["lpips"] is None
    assert payload["returncode"] == 2
    assert "exited 2" in payload["error"]
    assert "dataloader empty" in payload["error"]
    assert _read(report)["psnr"] is None


def test_missing_config_does_not_run_and_clears_a_stale_success(tmp_path):
    report = tmp_path / "ns_eval.json"
    report.write_text(json.dumps({"ok": True, "psnr": 30}), encoding="utf-8")
    called = []

    def run(argv, **kwargs):
        called.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    payload = write_heldout_ns_eval_report(
        load_config=tmp_path / "missing.yml",
        report_path=report,
        argv_prefix=[],
        cwd=tmp_path,
        env=None,
        run=run,
    )
    assert called == []
    assert payload["ok"] is False
    assert payload["argv"] is None
    assert "training config not found" in payload["error"]
    assert _read(report)["psnr"] is None


def test_eval_mode_all_is_not_scored(tmp_path):
    config = _config(tmp_path, "eval_mode: all\n")
    called = []

    def run(argv, **kwargs):
        called.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    payload = write_heldout_ns_eval_report(
        load_config=config,
        report_path=tmp_path / "ns_eval.json",
        argv_prefix=["mamba", "run", "-n", "nerfstudio_simfoundry"],
        cwd=tmp_path,
        env=None,
        run=run,
    )
    assert called == []
    assert payload["ok"] is False
    assert "eval_mode is all" in payload["error"]
    assert payload["psnr"] is None


def test_quoted_eval_mode_all_is_not_scored(tmp_path):
    config = _config(tmp_path, 'eval_mode: "all"\n')
    payload = write_heldout_ns_eval_report(
        load_config=config,
        report_path=tmp_path / "ns_eval.json",
        argv_prefix=[],
        cwd=None,
        env=None,
        run=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ns-eval ran")),
    )
    assert "eval_mode is all" in payload["error"]


def test_malformed_metrics_are_not_invented(tmp_path):
    config = _config(tmp_path)
    bodies = [
        "{",
        {"results": "nope"},
        {"psnr": 10, "ssim": 1, "lpips": 0},
        {"results": {"psnr": True, "ssim": 1, "lpips": 0}},
        {"results": {"psnr": "20", "ssim": 1, "lpips": 0}},
        {"results": {"psnr": float("nan"), "ssim": 1, "lpips": 0}},
        {"results": {"ssim": 0.5, "lpips": 0.2}},
    ]
    for body in bodies:
        def run(argv, body=body, **kwargs):
            out = Path(argv[argv.index("--output-path") + 1])
            if isinstance(body, str):
                out.write_text(body, encoding="utf-8")
            else:
                out.write_text(json.dumps(body), encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0, "", "")

        payload = write_heldout_ns_eval_report(
            load_config=config,
            report_path=tmp_path / "ns_eval.json",
            argv_prefix=[],
            cwd=tmp_path,
            env=None,
            run=run,
        )
        assert payload["ok"] is False
        assert payload["psnr"] is None
        assert payload["ssim"] is None
        assert payload["lpips"] is None
        assert payload["error"]


def test_missing_command_is_a_report_error(tmp_path):
    config = _config(tmp_path)

    def run(argv, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "ns-eval")

    payload = write_heldout_ns_eval_report(
        load_config=config,
        report_path=tmp_path / "ns_eval.json",
        argv_prefix=[],
        cwd=tmp_path,
        env=None,
        run=run,
    )
    assert payload["ok"] is False
    assert payload["returncode"] is None
    assert "failed to start" in payload["error"]
    assert payload["psnr"] is None


def test_timeout_is_a_report_error(tmp_path):
    config = _config(tmp_path)

    def run(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    payload = write_heldout_ns_eval_report(
        load_config=config,
        report_path=tmp_path / "ns_eval.json",
        argv_prefix=[],
        cwd=tmp_path,
        env=None,
        timeout_s=3,
        run=run,
    )
    assert payload["ok"] is False
    assert "timed out after 3s" in payload["error"]
    assert payload["psnr"] is None


def test_run_bounded_kills_the_process_group_on_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        ns_eval_report._run_bounded(
            ["sleep", "30"],
            cwd=None,
            env=None,
            timeout=0.3,
            check=False,
            capture_output=True,
            text=True,
        )


def test_unexpected_runner_error_still_writes_a_report(tmp_path):
    config = _config(tmp_path)

    def run(argv, **kwargs):
        raise RuntimeError("boom")

    payload = write_heldout_ns_eval_report(
        load_config=config,
        report_path=tmp_path / "ns_eval.json",
        argv_prefix=[],
        cwd=tmp_path,
        env=None,
        run=run,
    )
    assert payload["ok"] is False
    assert "RuntimeError" in payload["error"]
    assert _read(tmp_path / "ns_eval.json")["ok"] is False


def test_relative_config_and_cwd_are_absolute_when_ns_eval_runs(tmp_path, monkeypatch):
    """Hydra's auto_bg base_dir is relative to scripts/cfg. ns-eval's cwd is that
    directory, so a relative --load-config would be resolved a second time and miss.
    """
    monkeypatch.chdir(tmp_path)
    config = Path("splat") / "outputs" / "run" / "config.yml"
    config.parent.mkdir(parents=True)
    config.write_text("eval_mode: fraction\n", encoding="utf-8")
    seen: dict = {}

    def run(argv, **kwargs):
        seen["argv"] = list(argv)
        seen["cwd"] = kwargs["cwd"]
        _write_raw(list(argv), _NS_EVAL_JSON)
        return subprocess.CompletedProcess(argv, 0, "", "")

    payload = write_heldout_ns_eval_report(
        load_config=config,
        report_path=Path("splat") / "ns_eval.json",
        argv_prefix=["mamba", "run", "-n", "nerfstudio_simfoundry"],
        cwd=Path("splat"),
        env=None,
        run=run,
    )
    assert payload["ok"] is True
    load_arg = seen["argv"][seen["argv"].index("--load-config") + 1]
    out_arg = seen["argv"][seen["argv"].index("--output-path") + 1]
    assert Path(load_arg).is_absolute()
    assert Path(load_arg) == config.resolve()
    assert Path(seen["cwd"]).is_absolute()
    assert Path(seen["cwd"]) == Path("splat").resolve()
    assert Path(out_arg).is_absolute()
    assert (Path("splat") / "ns_eval.json").resolve().is_file()


def test_keyboard_interrupt_kills_the_eval_group(monkeypatch):
    killed: list[tuple[int, int]] = []

    class _Proc:
        def __init__(self):
            self.pid = 5151
            self.args = ["ns-eval"]
            self.returncode = None
            self.calls = 0

        def communicate(self, timeout=None):
            del timeout
            self.calls += 1
            if self.calls == 1:
                raise KeyboardInterrupt()
            return "", ""

        def poll(self):
            return self.returncode

        def kill(self):
            self.returncode = -9

    proc = _Proc()

    def popen(*_args, **kwargs):
        assert kwargs["start_new_session"] is True
        return proc

    def killpg(pid, sig):
        killed.append((pid, sig))
        proc.returncode = -9

    monkeypatch.setattr(ns_eval_report.subprocess, "Popen", popen)
    monkeypatch.setattr(ns_eval_report.os, "killpg", killpg)
    with pytest.raises(KeyboardInterrupt):
        ns_eval_report._run_bounded(
            ["ns-eval"],
            cwd=None,
            env=None,
            timeout=5,
            check=False,
            capture_output=True,
            text=True,
        )
    assert killed == [(5151, signal.SIGKILL)]
    assert proc.returncode == -9


def test_sigterm_kills_the_eval_group(tmp_path):
    pidfile = tmp_path / "child.pid"
    child_code = (
        "import os, time\n"
        f"open({str(pidfile)!r}, 'w').write(str(os.getpid()))\n"
        "time.sleep(30)\n"
    )
    stage_code = (
        "from simfoundry.pipeline.ns_eval_report import _run_bounded\n"
        "_run_bounded(\n"
        f"    [{sys.executable!r}, '-c', {child_code!r}],\n"
        "    cwd=None, env=None, timeout=20, check=False,\n"
        "    capture_output=True, text=True,\n"
        ")\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    stage = subprocess.Popen([sys.executable, "-c", stage_code], cwd=str(ROOT), env=env)
    child = None
    try:
        deadline = time.time() + 5
        while time.time() < deadline and not pidfile.exists():
            if stage.poll() is not None:
                raise AssertionError(f"stage exited early: {stage.returncode}")
            time.sleep(0.05)
        assert pidfile.is_file()
        child = int(pidfile.read_text(encoding="utf-8"))
        os.kill(stage.pid, signal.SIGTERM)
        stage.wait(timeout=5)
        deadline = time.time() + 2
        while time.time() < deadline:
            try:
                os.kill(child, 0)
            except ProcessLookupError:
                child = None
                break
            time.sleep(0.05)
        else:
            raise AssertionError("eval child survived SIGTERM to the stage")
    finally:
        if stage.poll() is None:
            stage.kill()
            stage.wait(timeout=5)
        if child is not None:
            try:
                os.kill(child, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_discard_removes_only_an_existing_report(tmp_path):
    report = tmp_path / "ns_eval.json"
    discard_stale_ns_eval_report(report)
    report.write_text("{}", encoding="utf-8")
    discard_stale_ns_eval_report(report)
    assert not report.exists()


def _stub_module(monkeypatch: pytest.MonkeyPatch, name: str, **attrs: object) -> None:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)


def _load_stage(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _hydra_main(*_args, **_kwargs):
    def deco(fn):
        return fn

    return deco


def _patch_eval(monkeypatch: pytest.MonkeyPatch, *, returncode: int, body: dict | None, explode: BaseException | None = None):
    seen: list[list[str]] = []

    def run(argv, **kwargs):
        seen.append(list(argv))
        assert kwargs["timeout"] == NS_EVAL_TIMEOUT_S
        if explode is not None:
            raise explode
        if body is not None:
            _write_raw(list(argv), body)
        return subprocess.CompletedProcess(argv, returncode, "", "eval failed" if returncode else "")

    monkeypatch.setattr(ns_eval_report, "_run_bounded", run)
    return seen


def _stage_2c(monkeypatch: pytest.MonkeyPatch):
    _stub_module(monkeypatch, "hydra", main=_hydra_main)
    return _load_stage("stage_2c_ns_eval_under_test", STAGE_2C)


def _stage_5(monkeypatch: pytest.MonkeyPatch):
    _stub_module(monkeypatch, "hydra", main=_hydra_main)
    _stub_module(monkeypatch, "cv2")
    _stub_module(monkeypatch, "plyfile", PlyData=object, PlyElement=object)
    _stub_module(
        monkeypatch,
        "simfoundry.pipeline.stage_utils",
        bootstrap_hydra_workdir=lambda script_file: "/tmp",
    )
    module = _load_stage("stage_5_ns_eval_under_test", STAGE_5)
    module.build_transforms = lambda *args, **kwargs: None
    return module


def _prepare_2c(tmp_path: Path):
    video = tmp_path / "in.mp4"
    video.write_bytes(b"video")
    base = tmp_path / "s2c_gs"
    processed = base / "processed"
    processed.mkdir(parents=True)
    (processed / "transforms.json").write_text("{}", encoding="utf-8")
    cfg = OmegaConf.create(
        {
            "s2c_gs": {
                "out_dir": str(base),
                "processed_dirname": "processed",
                "video_fpath": str(video),
                "method": "splatfacto",
                "max_num_iterations": 5,
                "export_dirname": "export",
            }
        }
    )
    return cfg, base


def _patch_2c_commands(monkeypatch: pytest.MonkeyPatch, module, *, make_config: bool, make_ply: bool, train_rc: int = 0):
    calls: list[list[str]] = []

    def run(argv, cwd=None, env=None, check=False, **kwargs):
        del env, kwargs
        calls.append(list(argv))
        if argv[0] == "ns-train":
            if train_rc:
                raise subprocess.CalledProcessError(train_rc, argv)
            if make_config:
                config = Path(cwd) / "outputs" / "run" / "config.yml"
                config.parent.mkdir(parents=True, exist_ok=True)
                config.write_text("eval_mode: fraction\n", encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0)
        if argv[0] == "ns-export":
            export_dir = Path(argv[argv.index("--output-dir") + 1])
            export_dir.mkdir(parents=True, exist_ok=True)
            if make_ply:
                (export_dir / "splat.ply").write_bytes(b"ply")
            return subprocess.CompletedProcess(argv, 0)
        raise AssertionError(argv)

    monkeypatch.setattr(module.subprocess, "run", run)
    return calls


def test_stage_2c_success_keeps_ply_verdict_and_records_scores(tmp_path, monkeypatch, restore_cwd):
    module = _stage_2c(monkeypatch)
    cfg, base = _prepare_2c(tmp_path)
    calls = _patch_2c_commands(monkeypatch, module, make_config=True, make_ply=True)
    eval_calls = _patch_eval(monkeypatch, returncode=0, body=_NS_EVAL_JSON)
    module.main(cfg)
    assert [cmd[0] for cmd in calls] == ["ns-train", "ns-export"]
    _assert_eval_argv(eval_calls[0], base / "outputs" / "run" / "config.yml", [])
    info = _read(base / "stage_info.json")
    assert info["success"] is True
    assert info["exported_plys"] == [str(base / "export" / "splat.ply")]
    assert "psnr" not in info
    report = _read(base / "ns_eval.json")
    assert report["ok"] is True
    assert report["psnr"] == 21.5
    assert report["ssim"] == 0.81
    assert report["lpips"] == 0.23


def test_stage_2c_eval_failure_leaves_success_true(tmp_path, monkeypatch, restore_cwd):
    module = _stage_2c(monkeypatch)
    cfg, base = _prepare_2c(tmp_path)
    calls = _patch_2c_commands(monkeypatch, module, make_config=True, make_ply=True)
    _patch_eval(
        monkeypatch,
        returncode=1,
        body={"results": {"psnr": 99, "ssim": 1, "lpips": 0}},
    )
    module.main(cfg)
    assert [cmd[0] for cmd in calls] == ["ns-train", "ns-export"]
    info = _read(base / "stage_info.json")
    assert info["success"] is True
    report = _read(base / "ns_eval.json")
    assert report["ok"] is False
    assert report["psnr"] is None
    assert "exited 1" in report["error"]


def test_stage_2c_missing_command_leaves_success_true(tmp_path, monkeypatch, restore_cwd):
    module = _stage_2c(monkeypatch)
    cfg, base = _prepare_2c(tmp_path)
    _patch_2c_commands(monkeypatch, module, make_config=True, make_ply=True)
    _patch_eval(
        monkeypatch,
        returncode=0,
        body=None,
        explode=FileNotFoundError(2, "No such file or directory", "ns-eval"),
    )
    module.main(cfg)
    assert _read(base / "stage_info.json")["success"] is True
    report = _read(base / "ns_eval.json")
    assert report["ok"] is False
    assert "failed to start" in report["error"]
    assert report["psnr"] is None


def test_stage_2c_train_failure_still_fails_and_drops_the_stale_report(tmp_path, monkeypatch, restore_cwd):
    module = _stage_2c(monkeypatch)
    cfg, base = _prepare_2c(tmp_path)
    stale = base / "ns_eval.json"
    stale.write_text(json.dumps({"ok": True, "psnr": 30}), encoding="utf-8")
    _patch_2c_commands(monkeypatch, module, make_config=False, make_ply=False, train_rc=1)
    eval_calls = _patch_eval(monkeypatch, returncode=0, body=_NS_EVAL_JSON)
    with pytest.raises(subprocess.CalledProcessError):
        module.main(cfg)
    assert eval_calls == []
    assert not stale.exists()
    assert not (base / "stage_info.json").exists()


def test_stage_2c_missing_config_still_fails_and_records_why(tmp_path, monkeypatch, restore_cwd):
    module = _stage_2c(monkeypatch)
    cfg, base = _prepare_2c(tmp_path)
    calls = _patch_2c_commands(monkeypatch, module, make_config=False, make_ply=False)
    eval_calls = _patch_eval(monkeypatch, returncode=0, body=_NS_EVAL_JSON)
    with pytest.raises(FileNotFoundError, match="No training config found"):
        module.main(cfg)
    assert [cmd[0] for cmd in calls] == ["ns-train"]
    assert eval_calls == []
    report = _read(base / "ns_eval.json")
    assert report["ok"] is False
    assert "training config not found" in report["error"]
    assert not (base / "stage_info.json").exists()


def test_stage_2c_missing_ply_still_fails_after_a_successful_eval(tmp_path, monkeypatch, restore_cwd):
    module = _stage_2c(monkeypatch)
    cfg, base = _prepare_2c(tmp_path)
    _patch_2c_commands(monkeypatch, module, make_config=True, make_ply=False)
    _patch_eval(monkeypatch, returncode=0, body=_NS_EVAL_JSON)
    with pytest.raises(FileNotFoundError, match="produced no .ply"):
        module.main(cfg)
    assert _read(base / "stage_info.json")["success"] is False
    assert _read(base / "ns_eval.json")["ok"] is True


class _Section:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _prepare_5(tmp_path: Path):
    base = tmp_path / "splat"
    inpainted = tmp_path / "frames"
    da3 = tmp_path / "results.npz"
    inpainted.write_bytes(b"")
    da3.write_bytes(b"")
    section = _Section(
        inpainted_dir=str(inpainted),
        masks_dir=None,
        da3_npz=str(da3),
        processed_dir=str(tmp_path / "ns_data"),
        base_dir=str(base),
        view_only=False,
        no_masks=True,
        no_seed_ply=True,
        seed_ply_path=None,
        use_depth_loss=False,
        skip_train=False,
        no_export=False,
        method="splatfacto-big",
        max_num_iterations=3,
        viewer_port=7007,
        keep_viewer=False,
        camera_optimizer_mode="SO3xR3",
    )
    return _Section(scene_name="desk", s5_train_bg_splat=section), base


def _patch_5_commands(
    monkeypatch: pytest.MonkeyPatch,
    module,
    *,
    make_config: bool,
    make_ply: bool,
    train_rc: int = 0,
    export_rc: int = 0,
):
    calls: list[list[str]] = []

    def run(argv, cwd=None, env=None, check=False, **kwargs):
        del env, kwargs
        calls.append(list(argv))
        if "ns-train" in argv:
            if train_rc:
                raise subprocess.CalledProcessError(train_rc, argv)
            if make_config:
                config = Path(cwd) / "outputs" / "run" / "config.yml"
                config.parent.mkdir(parents=True, exist_ok=True)
                config.write_text("eval_mode: fraction\n", encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0)
        if "python" in argv:
            if export_rc:
                raise subprocess.CalledProcessError(export_rc, argv)
            export_dir = Path(cwd) / "export"
            export_dir.mkdir(parents=True, exist_ok=True)
            if make_ply:
                (export_dir / "splat.ply").write_bytes(b"ply")
            return subprocess.CompletedProcess(argv, 0)
        raise AssertionError(argv)

    monkeypatch.setattr(module.subprocess, "run", run)
    return calls


def test_stage_5_success_exports_ply_and_records_scores(tmp_path, monkeypatch, restore_cwd):
    module = _stage_5(monkeypatch)
    cfg, base = _prepare_5(tmp_path)
    calls = _patch_5_commands(monkeypatch, module, make_config=True, make_ply=True)
    eval_calls = _patch_eval(monkeypatch, returncode=0, body=_NS_EVAL_JSON)
    assert module.main(cfg) is None
    assert calls[0][0:4] == ["mamba", "run", "-n", module.NS_ENV_NAME]
    assert "ns-train" in calls[0]
    assert calls[1][4] == "python"
    _assert_eval_argv(
        eval_calls[0],
        base / "outputs" / "run" / "config.yml",
        ["mamba", "run", "-n", module.NS_ENV_NAME],
    )
    assert (base / "export" / "desk_bg.ply").is_file()
    report = _read(base / "ns_eval.json")
    assert report["ok"] is True
    assert report["psnr"] == 21.5
    assert report["ssim"] == 0.81
    assert report["lpips"] == 0.23


def test_stage_5_eval_failure_still_exports_and_returns(tmp_path, monkeypatch, restore_cwd):
    module = _stage_5(monkeypatch)
    cfg, base = _prepare_5(tmp_path)
    calls = _patch_5_commands(monkeypatch, module, make_config=True, make_ply=True)
    _patch_eval(
        monkeypatch,
        returncode=1,
        body={"results": {"psnr": 99, "ssim": 1, "lpips": 0}},
    )
    assert module.main(cfg) is None
    assert len(calls) == 2
    assert (base / "export" / "desk_bg.ply").is_file()
    report = _read(base / "ns_eval.json")
    assert report["ok"] is False
    assert report["psnr"] is None
    assert "exited 1" in report["error"]


def test_stage_5_train_failure_still_fails_and_drops_the_stale_report(tmp_path, monkeypatch, restore_cwd):
    module = _stage_5(monkeypatch)
    cfg, base = _prepare_5(tmp_path)
    base.mkdir()
    stale = base / "ns_eval.json"
    stale.write_text(json.dumps({"ok": True, "psnr": 30}), encoding="utf-8")
    _patch_5_commands(monkeypatch, module, make_config=False, make_ply=False, train_rc=1)
    eval_calls = _patch_eval(monkeypatch, returncode=0, body=_NS_EVAL_JSON)
    with pytest.raises(subprocess.CalledProcessError):
        module.main(cfg)
    assert eval_calls == []
    assert not stale.exists()
    assert not (base / "export" / "desk_bg.ply").exists()


def test_stage_5_export_failure_still_fails_after_a_successful_eval(tmp_path, monkeypatch, restore_cwd):
    module = _stage_5(monkeypatch)
    cfg, base = _prepare_5(tmp_path)
    _patch_5_commands(monkeypatch, module, make_config=True, make_ply=False, export_rc=1)
    _patch_eval(monkeypatch, returncode=0, body=_NS_EVAL_JSON)
    with pytest.raises(subprocess.CalledProcessError):
        module.main(cfg)
    assert _read(base / "ns_eval.json")["ok"] is True
    assert not (base / "export" / "desk_bg.ply").exists()


def test_stage_5_missing_config_still_exits_and_records_why(tmp_path, monkeypatch, restore_cwd):
    module = _stage_5(monkeypatch)
    cfg, base = _prepare_5(tmp_path)
    calls = _patch_5_commands(monkeypatch, module, make_config=False, make_ply=False)
    eval_calls = _patch_eval(monkeypatch, returncode=0, body=_NS_EVAL_JSON)
    with pytest.raises(SystemExit, match="No training config found"):
        module.main(cfg)
    assert len(calls) == 1
    assert eval_calls == []
    report = _read(base / "ns_eval.json")
    assert report["ok"] is False
    assert "training config not found" in report["error"]
