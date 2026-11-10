# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the Vast Fruits smoke. No network and no API key."""

import base64
import hashlib
import json
import signal
import subprocess
from pathlib import Path

import pytest

from sceneagent.gpu_smoke_trigger import DISPATCH_INPUTS, workflow_safety_findings
from sceneagent.vast_smoke import (
    ALLOWED_INSTANCE_ID,
    BIN_PIPELINES,
    DETACH_LEFTOVER_KEYS,
    FRUITS_RELATIVE,
    REMOTE_OUT,
    REMOTE_PYTHON,
    REMOTE_REPO,
    SSH_READY_MAX_ATTEMPTS,
    SSH_READY_TIMEOUT_S,
    KNOWN_HOST_PATH,
    InstanceNotApproved,
    SmokeError,
    SshRemote,
    assert_request_allowed,
    call_json,
    guard_own_key_delete,
    attached_ssh_keys_from_show,
    await_ssh_after_attach,
    build_fetch_script,
    build_remote_command,
    build_remote_shell,
    child_env,
    classify_status,
    collect_artifact_manifest,
    endpoint_for,
    ephemeral_key_record,
    list_attached_ssh_keys,
    load_known_host_line,
    load_ssh_key_fingerprints,
    main,
    mask_secret,
    parse_instance_id,
    public_instance_view,
    publish_artifacts,
    publish_failure_manifest,
    redact,
    request_json,
    require_approved_instance,
    rsync_invocations,
    assess_secret_forwarding,
    iter_shipped_files,
    payload_import_blockers,
    normalize_run_context,
    run_smoke,
    ssh_base_argv,
    ssh_endpoint,
    ssh_forwarding_problems,
    ssh_key_id_from_attach,
    wait_until_ssh_accepts,
)

ROOT = Path(__file__).resolve().parents[2]
FRUITS = ROOT / "docs" / "assets" / "example_videos" / "Fruits.mp4"
WORKFLOW = ROOT / ".github" / "workflows" / "sceneagent-gpu-smoke.yml"
PUBLIC_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITestKeyForSf4SmokeOnly sf4-smoke"


def _command():
    return build_remote_command(REMOTE_REPO, REMOTE_OUT)


def _texas(status, **extra):
    row = {
        "id": ALLOWED_INSTANCE_ID,
        "actual_status": status,
        "intended_status": "running" if status in {"running", "loading"} else "stopped",
        "geolocation": "Texas, US",
        "ssh_host": "ssh6.vast.ai",
        "ssh_port": 35512,
        "public_ipaddr": "47.186.21.5",
        "ports": {"22/tcp": [{"HostPort": "51823"}]},
        "gpu_name": "RTX 4090",
        "jupyter_token": "do-not-log",
        "extra_env": [["SECRET", "nope"]],
        "onstart": "echo no",
    }
    row.update(extra)
    return row


def _checkout(tmp_path):
    dest = tmp_path / "docs" / "assets" / "example_videos"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "Fruits.mp4").write_bytes(b"not-a-real-video")
    (tmp_path / "sceneagent").mkdir(exist_ok=True)
    return tmp_path


def _ledger(decision="accept", use_gpu=True):
    colmap = {
        "item": "COLMAP camera poses",
        "decision": decision,
        "frames": ["0001.jpg"],
        "model_dir": "colmap/sparse/0",
    }
    if decision == "accept":
        colmap["use_gpu"] = use_gpu
    if decision == "reject":
        colmap["failed_step"] = "mapper"
    if decision == "deferred":
        colmap["reason"] = "colmap or ffmpeg is not on PATH"
    return {
        "scene_name": "Fruits",
        "source": "simfoundry_example_video",
        "entries": [
            {"item": "video", "decision": "accept", "bytes": 12, "duration_s": 15.0},
            colmap,
        ],
    }


def _write_run(run, decision="accept", use_gpu=True, probe=None):
    run.mkdir(parents=True, exist_ok=True)
    (run / "ledger.json").write_text(
        json.dumps(_ledger(decision, use_gpu)) + "\n", encoding="utf-8"
    )
    (run / "video.json").write_text("{}\n", encoding="utf-8")
    (run / "colmap_poses.json").write_text("{}\n", encoding="utf-8")
    if probe is not None:
        (run / "host_probe.json").write_text(json.dumps(probe) + "\n", encoding="utf-8")


class _Remote:
    def __init__(self, writer, rc=0):
        self.writer = writer
        self.rc = rc
        self.closed = False

    def run(self, checkout, local_run):
        local_run.mkdir(parents=True, exist_ok=True)
        self.writer(local_run)
        return self.rc

    def close(self):
        self.closed = True


def test_parse_instance_id_accepts_only_59601312():
    assert ALLOWED_INSTANCE_ID == 59601312
    assert parse_instance_id(59601312) == 59601312
    assert parse_instance_id("59601312") == 59601312
    assert parse_instance_id(" 59601312 ") == 59601312
    for value in (
        58301075,
        "58301075",
        " 58301075 ",
        11111111,
        "59601312.0",
        "",
        None,
        True,
        False,
        59601312.0,
        "59601312abc",
    ):
        with pytest.raises(InstanceNotApproved):
            parse_instance_id(value)


def test_endpoint_allowlist_is_show_and_ssh_on_one_id():
    method, url, body = endpoint_for("show", "59601312")
    assert method == "GET"
    assert url == "https://console.vast.ai/api/v0/instances/59601312/?owner=me"
    assert body is None
    for kind in ("start", "stop", "attach_ssh", "detach_ssh"):
        with pytest.raises(InstanceNotApproved):
            endpoint_for(kind, 59601312)
    for state in ("running", "stopped"):
        with pytest.raises(InstanceNotApproved):
            assert_request_allowed(
                "PUT",
                "https://console.vast.ai/api/v0/instances/59601312/",
                {"state": state},
            )
    for kind in ("create", "rent", "resize", "destroy", "delete", "reboot", "label"):
        with pytest.raises(InstanceNotApproved):
            endpoint_for(kind, 59601312)
    with pytest.raises(InstanceNotApproved):
        endpoint_for("show", 11111111)
    with pytest.raises(InstanceNotApproved):
        assert_request_allowed(
            "POST",
            "https://console.vast.ai/api/v0/instances/59601312/ssh/",
            {"ssh_key": PUBLIC_KEY},
        )
    with pytest.raises(InstanceNotApproved):
        assert_request_allowed(
            "PUT",
            "https://console.vast.ai/api/v0/instances/59601312/",
            {"state": "running", "disk": 500},
        )
    with pytest.raises(InstanceNotApproved):
        assert_request_allowed(
            "DELETE", "https://console.vast.ai/api/v0/instances/59601312/", None
        )
    with pytest.raises(InstanceNotApproved):
        assert_request_allowed(
            "POST", "https://console.vast.ai/api/v0/asks/1/", {"image": "x"}
        )
    with pytest.raises(InstanceNotApproved):
        assert_request_allowed(
            "GET",
            "https://console.vast.ai/api/v0/instances/59601312/?owner=me&api_key=secret",
            None,
        )


def test_request_json_refuses_a_forbidden_url_before_the_network(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("urlopen")

    monkeypatch.setattr("sceneagent.vast_smoke.urllib.request.urlopen", boom)
    with pytest.raises(InstanceNotApproved):
        request_json(
            "POST", "https://console.vast.ai/api/v0/asks/1/", "k", {"image": "x"}
        )


def test_require_approved_instance_checks_id_and_texas():
    row = require_approved_instance({"instances": _texas("running")})
    assert row["id"] == 59601312
    with pytest.raises(InstanceNotApproved, match="missing"):
        require_approved_instance({"instances": None})
    with pytest.raises(InstanceNotApproved, match="not Texas"):
        require_approved_instance(
            {"instances": _texas("running", geolocation="Estonia, EE")}
        )
    with pytest.raises(InstanceNotApproved, match="returned id"):
        require_approved_instance({"instances": _texas("running", id=11111111)})


def test_public_view_drops_tokens():
    view = public_instance_view(_texas("exited"))
    assert view["geolocation"] == "Texas, US"
    assert "jupyter_token" not in view
    assert "extra_env" not in view
    assert "onstart" not in view
    blob = json.dumps(view)
    assert "do-not-log" not in blob


@pytest.mark.parametrize(
    ("actual", "intended", "expected"),
    [
        ("running", "running", "running"),
        ("loading", "running", "starting"),
        ("created", "running", "starting"),
        ("rebooting", "running", "starting"),
        ("stopped", "stopped", "stopped"),
        ("exited", "stopped", "stopped"),
        (None, "stopped", "stopped"),
        (None, "running", "starting"),
        ("offline", "stopped", "unknown"),
        ("unknown", "running", "unknown"),
        ("frozen", "stopped", "unknown"),
        ("", "", "unknown"),
    ],
)
def test_classify_status(actual, intended, expected):
    assert (
        classify_status({"actual_status": actual, "intended_status": intended})
        == expected
    )


def test_redact_removes_every_copy_of_the_key():
    secret = "sf4-test-key"
    assert (
        redact(f"header {secret} and {secret}", secret)
        == "header [redacted] and [redacted]"
    )
    assert redact("plain", "") == "plain"


def test_mask_secret_is_silent_outside_actions(capsys, monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    mask_secret("sf4-test-key")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_add_mask_is_emitted_inside_actions(capsys, monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    mask_secret("sf4-test-key")
    captured = capsys.readouterr()
    assert captured.out == "::add-mask::sf4-test-key\n"
    assert captured.err == ""


def test_child_env_drops_the_api_key(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "super-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "gh-token-value")
    monkeypatch.setenv("GH_TOKEN", "gh-token-value-2")
    env = child_env()
    assert "VAST_API_KEY" not in env
    assert "GITHUB_TOKEN" not in env
    monkeypatch.setenv(
        "VAST_SSH_PRIVATE_KEY", "-----BEGIN OPENSSH PRIVATE KEY-----\nsecret\n"
    )
    env = child_env()
    assert "VAST_SSH_PRIVATE_KEY" not in env
    assert "GH_TOKEN" not in env
    blob = " ".join(env.values())
    assert "super-secret" not in blob
    assert "gh-token-value" not in blob


def test_remote_command_is_the_fruits_entry_point_and_carries_no_secret():
    argv = _command()
    assert argv[0] == REMOTE_PYTHON
    assert argv[1:3] == ["-m", "sceneagent"]
    assert argv[3].endswith(FRUITS_RELATIVE)
    assert argv[4:] == ["--out", REMOTE_OUT]
    script = build_remote_shell(REMOTE_REPO, REMOTE_OUT)
    assert "Fruits.mp4" in script
    assert "PYTHONPATH" in script
    assert "host_probe.json" in script
    assert "command -v python3" not in script
    assert "py=python" not in script
    blob = script + "\n" + build_fetch_script(REMOTE_OUT) + "\n" + " ".join(argv)
    lowered = blob.lower()
    for banned in ("vast_api_key", "bearer", "api_key", "gemini", "meshy"):
        assert banned not in lowered
    assert REMOTE_REPO.startswith("/workspace/sf4-smoke/")
    assert "SimFoundry" not in REMOTE_REPO
    with pytest.raises(InstanceNotApproved):
        build_remote_command("/tmp/../etc", REMOTE_OUT)


def test_missing_pinned_interpreter_exits_127_and_does_not_use_path():
    script = build_remote_shell(REMOTE_REPO, REMOTE_OUT)
    assert script.count(REMOTE_PYTHON) >= 2
    completed = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 127
    assert completed.stderr.strip() == f"pinned interpreter missing: {REMOTE_PYTHON}"
    assert "command -v" not in script


def test_ssh_and_rsync_argv_do_not_carry_the_api_key(tmp_path):
    argv = ssh_base_argv("/tmp/sf4_ed25519", "/tmp/known_hosts", 22)
    invocations = rsync_invocations(
        tmp_path,
        "ssh6.vast.ai",
        22,
        "/tmp/sf4_ed25519",
        "/tmp/known_hosts",
        REMOTE_REPO,
    )
    blob = " ".join(argv)
    for command, _cwd in invocations:
        blob += " " + " ".join(command)
    assert "StrictHostKeyChecking=yes" in argv
    assert "accept-new" not in " ".join(argv)
    assert "VAST_API_KEY" not in blob
    assert "Bearer" not in blob
    assert FRUITS_RELATIVE in blob
    assert "sceneagent" in blob


def test_rsync_creates_the_missing_remote_parent(tmp_path):
    # Run4 on 2026-11-07 failed with `mkdir "/workspace/sf4-smoke/repo" failed`:
    # a fresh box has no /workspace/sf4-smoke, and rsync makes only the last
    # path component unless told otherwise.
    invocations = rsync_invocations(
        tmp_path,
        "ssh6.vast.ai",
        22,
        "/tmp/sf4_ed25519",
        "/tmp/known_hosts",
        REMOTE_REPO,
    )
    assert invocations
    for command, _cwd in invocations:
        assert command[0] == "rsync"
        # --mkpath needs rsync 3.2.3 on both ends; instance 59601312 has
        # 3.1.3 and refused it (2026-11-08). The remote side makes the
        # directory itself instead, which every rsync version accepts.
        assert "--mkpath" not in command
        assert f"--rsync-path=mkdir -p {REMOTE_REPO} && rsync" in command


def test_rsync_path_creates_the_destination_with_a_real_rsync(tmp_path):
    import shutil
    import subprocess

    if shutil.which("rsync") is None:
        pytest.skip("rsync is not installed")
    checkout = _checkout(tmp_path)
    fake_ssh = tmp_path / "fake_ssh"
    # Stands in for ssh: drop the options and host, run the remote command here.
    fake_ssh.write_text(
        '#!/bin/sh\nwhile [ "$1" != "host" ]; do shift; done\nshift\nexec sh -c "$*"\n'
    )
    fake_ssh.chmod(0o755)
    dest = tmp_path / "remote" / "sf4-smoke" / "repo"
    for command, cwd in rsync_invocations(checkout, "host", 22, "/k", "/kh", str(dest)):
        command = [
            str(fake_ssh) if part.startswith("ssh ") else part for part in command
        ]
        command = [part.replace("root@host:", "host:") for part in command]
        subprocess.run(command, cwd=cwd, check=True, capture_output=True)
    assert (dest / "sceneagent").is_dir()
    assert (dest / FRUITS_RELATIVE).is_file()


def test_ssh_key_id_comes_from_the_attach_response_only():
    assert ssh_key_id_from_attach({"ssh_key_id": 17}) == "17"
    assert ssh_key_id_from_attach({"key": {"key_id": "abc"}}) == "abc"
    assert ssh_key_id_from_attach({"id": 17, "success": True}) is None


def test_deferred_ledger_is_not_a_pass_and_is_not_rewritten(tmp_path):
    run = tmp_path / "run"
    _write_run(run, decision="deferred")
    before = (run / "ledger.json").read_bytes()
    manifest = collect_artifact_manifest(run, command=_command(), instance_id=59601312)
    assert (run / "ledger.json").read_bytes() == before
    ledger = json.loads(before.decode("utf-8"))
    assert ledger["entries"][1]["decision"] == "deferred"
    assert manifest["dc2"] == "missing"
    assert manifest["passed"] is False
    assert manifest["complete"] is False
    assert "not a pass" in manifest["failure_reason"]
    assert "plan.json was not written by the pipeline" in manifest["failure_reason"]
    assert not (run / "plan.json").exists()
    assert not (run / "plan.derived.json").exists()
    stage = manifest["stages"][1]
    assert stage["reported"] == "skipped"
    assert stage["counts_as_pass"] is False


def test_reject_is_reported_failed_without_editing_the_ledger(tmp_path):
    run = tmp_path / "run"
    _write_run(run, decision="reject")
    before = (run / "ledger.json").read_bytes()
    manifest = collect_artifact_manifest(run, command=_command(), instance_id=59601312)
    assert (run / "ledger.json").read_bytes() == before
    assert manifest["passed"] is False
    assert manifest["stages"][1]["reported"] == "failed"
    assert manifest["stages"][1]["decision"] == "reject"


def test_cpu_accept_is_not_a_gpu_pass(tmp_path):
    run = tmp_path / "run"
    _write_run(run, decision="accept", use_gpu=False)
    manifest = collect_artifact_manifest(run, command=_command(), instance_id=59601312)
    assert manifest["passed"] is False
    assert manifest["stages"][1]["reported"] == "accept_without_gpu"


def test_collector_copies_pipeline_plan_unchanged_and_fails_when_missing(tmp_path):
    run = tmp_path / "run"
    _write_run(run, decision="accept", use_gpu=True)
    original = b'{"scene_name":"Fruits","from":"pipeline"}\n'
    (run / "plan.json").write_bytes(original)
    manifest = collect_artifact_manifest(run, command=_command(), instance_id=59601312)
    assert manifest["passed"] is True
    assert (run / "plan.json").read_bytes() == original
    assert not (run / "plan.derived.json").exists()
    artifact_dir = tmp_path / "artifacts"
    copied = publish_artifacts(run, artifact_dir)
    assert "plan.json" in copied
    assert (artifact_dir / "plan.json").read_bytes() == original

    bare = tmp_path / "bare"
    _write_run(bare, decision="accept", use_gpu=True)
    missing = collect_artifact_manifest(bare, command=_command(), instance_id=59601312)
    assert missing["passed"] is False
    assert missing["plan"] is None
    assert "plan" in missing["missing"]
    assert "plan.json was not written by the pipeline" in missing["failure_reason"]
    assert not (bare / "plan.json").exists()
    assert not (bare / "plan.derived.json").exists()
    published = publish_artifacts(bare, tmp_path / "bare-artifacts")
    assert "plan.json" not in published


def test_gpu_accept_passes_and_keeps_an_existing_plan(tmp_path):
    run = tmp_path / "run"
    _write_run(run, decision="accept", use_gpu=True)
    original = {
        "scene_name": "Fruits",
        "steps": [{"item": "video", "decision": "accept"}],
    }
    (run / "plan.json").write_text(json.dumps(original) + "\n", encoding="utf-8")
    before = (run / "plan.json").read_bytes()
    manifest = collect_artifact_manifest(
        run, command=_command(), instance_id="59601312"
    )
    assert manifest["passed"] is True
    assert manifest["failure_reason"] is None
    assert (run / "plan.json").read_bytes() == before
    assert (run / "manifest.json").is_file()


def test_cuda_probe_does_not_turn_a_deferred_ledger_into_a_pass(tmp_path):
    run = tmp_path / "run"
    _write_run(
        run, decision="deferred", probe={"colmap_cuda": True, "nvidia_smi_ok": True}
    )
    manifest = collect_artifact_manifest(run, command=_command(), instance_id=59601312)
    assert manifest["passed"] is False
    assert manifest["host_probe"]["colmap_cuda"] is True


def test_missing_ledger_is_not_invented(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    manifest = collect_artifact_manifest(run, command=_command(), instance_id=59601312)
    assert manifest["passed"] is False
    assert "ledger" in manifest["missing"]
    assert manifest["ledger_status"] == "missing"
    assert manifest["dc2"] == "missing"
    assert not (run / "ledger.json").exists()
    assert not (run / "plan.json").exists()
    assert (run / "manifest.json").is_file()


def test_fruits_video_deferral_is_not_a_pass(tmp_path, monkeypatch):
    monkeypatch.setattr("sceneagent.external_tools.shutil.which", lambda _name: None)
    from sceneagent.video import run_video

    run = tmp_path / "fruits"
    run_video(FRUITS, run)
    plan_before = (run / "plan.json").read_bytes()
    before = (run / "ledger.json").read_bytes()
    manifest = collect_artifact_manifest(run, command=_command(), instance_id=59601312)
    assert (run / "ledger.json").read_bytes() == before
    assert (run / "plan.json").read_bytes() == plan_before
    ledger = json.loads(before.decode("utf-8"))
    decisions = {entry["item"]: entry["decision"] for entry in ledger["entries"]}
    assert decisions["video"] == "accept"
    assert decisions["COLMAP camera poses"] == "deferred"
    assert manifest["passed"] is False
    assert (run / "manifest.json").is_file()


def test_run_smoke_on_a_running_instance_does_not_stop(tmp_path):
    calls = []

    def request(method, url, api_key, body=None, timeout=60):
        calls.append((method, body))
        assert api_key == "test-key"
        assert "test-key" not in url
        assert method == "GET"
        return 200, {"instances": _texas("running")}

    def write_passing_run(run):
        _write_run(run, decision="accept", use_gpu=True)
        (run / "plan.json").write_bytes(b'{"scene_name":"Fruits","from":"pipeline"}\n')

    remote = _Remote(write_passing_run)
    code = run_smoke(
        59601312,
        _checkout(tmp_path),
        tmp_path / "artifacts",
        "test-key",
        request=request,
        remote_factory=lambda _row: remote,
        sleep=lambda _seconds: None,
    )
    assert code == 0
    assert remote.closed is True
    assert calls == [("GET", None)]
    assert (tmp_path / "artifacts" / "plan.json").is_file()
    assert (tmp_path / "artifacts" / "ledger.json").is_file()
    assert (tmp_path / "artifacts" / "manifest.json").is_file()
    published = json.loads(
        (tmp_path / "artifacts" / "manifest.json").read_text(encoding="utf-8")
    )
    assert published["passed"] is True


def test_dispatch_refuses_a_stopped_instance_without_lifecycle_calls(tmp_path, capsys):
    calls = []

    def request(method, url, api_key, body=None, timeout=60):
        calls.append((method, url, None if body is None else dict(body)))
        assert "test-key" not in url
        if method == "GET":
            return 200, {"instances": _texas("exited")}
        raise AssertionError((method, body))

    code = run_smoke(
        "59601312",
        _checkout(tmp_path),
        tmp_path / "artifacts",
        "test-key",
        request=request,
        sleep=lambda _seconds: None,
        trigger="dispatch",
        actor="sandy-t-shelvan",
        commit_sha="d" * 40,
    )
    assert code == 2
    assert [call[0] for call in calls] == ["GET"]
    assert _puts(calls) == []
    assert (
        "instance stopped; this workflow does not start or stop instances"
        in capsys.readouterr().err
    )


def test_run_smoke_refuses_the_wrong_instance_before_the_network(tmp_path):
    def boom(*_args, **_kwargs):
        raise AssertionError("network")

    code = run_smoke(11111111, tmp_path, tmp_path / "out", "test-key", request=boom)
    assert code == 2


def test_run_smoke_refuses_a_missing_key_and_a_non_texas_host(tmp_path, capsys):
    code = run_smoke(59601312, _checkout(tmp_path), tmp_path / "out", "")
    assert code == 2
    assert "not set" in capsys.readouterr().err

    def request(method, url, api_key, body=None, timeout=60):
        assert method == "GET"
        return 200, {"instances": _texas("running", geolocation="Quebec, CA")}

    code = run_smoke(
        59601312,
        _checkout(tmp_path),
        tmp_path / "out",
        "test-key",
        request=request,
        sleep=lambda _seconds: None,
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "not Texas" in err
    assert "test-key" not in err


def test_run_smoke_does_not_start_an_offline_instance(tmp_path):
    calls = []

    def request(method, url, api_key, body=None, timeout=60):
        calls.append(method)
        return 200, {"instances": _texas("offline")}

    code = run_smoke(
        59601312,
        _checkout(tmp_path),
        tmp_path / "out",
        "test-key",
        request=request,
        sleep=lambda _seconds: None,
    )
    assert code == 2
    assert calls == ["GET"]


def test_main_has_no_api_key_flag_and_rejects_a_bad_id(monkeypatch, tmp_path):
    source = (ROOT / "sceneagent" / "vast_smoke.py").read_text(encoding="utf-8").lower()
    assert "--api-key" not in source
    for needle in ("api.meshy.ai", "generativelanguage", "/api/v0/asks/", "meshy.com"):
        assert needle not in source

    def boom(*_args, **_kwargs):
        raise AssertionError("network")

    monkeypatch.setattr("sceneagent.vast_smoke.request_json", boom)
    monkeypatch.setenv("VAST_API_KEY", "test-key")
    code = main(
        [
            "--instance-id",
            "1",
            "--checkout",
            str(tmp_path),
            "--artifacts",
            str(tmp_path / "out"),
        ]
    )
    assert code == 2


def test_other_instance_id_is_rejected_before_any_http_call(monkeypatch, tmp_path):
    def boom(*_args, **_kwargs):
        raise AssertionError("HTTP was called")

    monkeypatch.setattr("sceneagent.vast_smoke.urllib.request.urlopen", boom)
    monkeypatch.setenv("VAST_API_KEY", "sf4-not-a-real-key")
    checkout = _checkout(tmp_path)
    for other in (11111111, "1", "58301075", "58301076", 59601313):
        code = main(
            [
                "--instance-id",
                str(other),
                "--checkout",
                str(checkout),
                "--artifacts",
                str(tmp_path / "out"),
            ]
        )
        assert code != 0
        code = run_smoke(
            other, checkout, tmp_path / "out", "sf4-not-a-real-key", request=boom
        )
        assert code != 0
    with pytest.raises(InstanceNotApproved):
        request_json(
            "GET",
            "https://console.vast.ai/api/v0/instances/11111111/?owner=me",
            "sf4-not-a-real-key",
        )


def test_api_key_is_absent_from_logs_and_artifacts(tmp_path, capsys, monkeypatch):
    secret = "sf4-not-a-real-key"
    monkeypatch.setenv("VAST_API_KEY", secret)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)

    def request(method, url, api_key, body=None, timeout=60):
        assert api_key == secret
        assert secret not in url
        assert method == "GET"
        return 200, {"instances": _texas("running")}

    def write_passing_run(run):
        _write_run(run, decision="accept", use_gpu=True)
        (run / "plan.json").write_bytes(b'{"scene_name":"Fruits","from":"pipeline"}\n')

    remote = _Remote(write_passing_run)
    code = run_smoke(
        ALLOWED_INSTANCE_ID,
        _checkout(tmp_path),
        tmp_path / "artifacts",
        secret,
        request=request,
        remote_factory=lambda _row: remote,
        sleep=lambda _seconds: None,
    )
    assert code == 0
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    for name in ("plan.json", "ledger.json", "manifest.json"):
        text = (tmp_path / "artifacts" / name).read_text(encoding="utf-8")
        assert secret not in text
    source = (ROOT / "sceneagent" / "vast_smoke.py").read_text(encoding="utf-8")
    assert 'os.environ.get("VAST_API_KEY"' in source
    assert "--api-key" not in source
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "VAST_API_KEY: ${{ secrets.VAST_API_KEY }}" in workflow
    assert workflow.count("secrets.VAST_API_KEY") == 1


def _run_blocks(text):
    lines = text.splitlines()
    blocks = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.lstrip(" ")
        if not stripped.startswith("run:"):
            index += 1
            continue
        indent = len(line) - len(stripped)
        rest = stripped[len("run:") :]
        chunk = [rest]
        index += 1
        while index < len(lines):
            nxt = lines[index]
            if nxt.strip() == "":
                chunk.append(nxt)
                index += 1
                continue
            leading = len(nxt) - len(nxt.lstrip(" "))
            if leading <= indent:
                break
            chunk.append(nxt)
            index += 1
        blocks.append("\n".join(chunk))
    return blocks


def test_run_steps_do_not_interpolate_expressions():
    blocks = _run_blocks(WORKFLOW.read_text(encoding="utf-8"))
    assert blocks
    for block in blocks:
        assert "${{" not in block
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "--instance-id 59601312" in text
    # The payload and the artifact dir stay under the runner env var.
    # Expressions are kept out of run scripts by the loop above.
    assert '"$GITHUB_WORKSPACE/payload"' in text
    assert '"$GITHUB_WORKSPACE/sf4-artifacts"' in text


def test_workflow_dispatch_takes_only_the_pipeline_choice_and_keeps_the_instance_literal():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in text
    assert DISPATCH_INPUTS in text
    assert workflow_safety_findings(text) == []
    extra = text.replace(
        "        default: run4\n",
        "        default: run4\n      instance_id:\n        type: string\n",
        1,
    )
    assert (
        "workflow_dispatch may only take the run4/bin/splat/masks/association/sam3d/place/mapper/probe pipeline choice"
        in (workflow_safety_findings(extra))
    )
    assert "inputs.instance_id" not in text
    assert "--instance-id 59601312" in text
    assert "pull_request_target:" in text
    for line in text.splitlines():
        assert line.strip() != "pull_request:"
        assert line.strip() != "schedule:"
        assert "continue-on-error" not in line
        if "runs-on:" in line:
            assert "ubuntu-latest" in line
        if "echo" in line:
            assert "VAST_API_KEY" not in line
    assert text.count("secrets.VAST_API_KEY") == 1
    assert "sceneagent-fruits-smoke" in text
    assert "plan.json" in text
    assert "ledger.json" in text
    assert "manifest.json" in text
    assert "/api/v0/asks/" not in text


def test_no_pipeline_plan_records_dc2_missing_and_writes_no_plan_file(tmp_path):
    run = tmp_path / "run"
    _write_run(run, decision="deferred")
    before = (run / "ledger.json").read_bytes()
    manifest = collect_artifact_manifest(run, command=_command(), instance_id=59601312)
    assert list(run.glob("plan*")) == []
    assert (run / "ledger.json").read_bytes() == before
    assert manifest["dc2"] in {"missing", "failed"}
    assert manifest["dc2"] not in {"skipped", "passed", "accept"}
    assert manifest["passed"] is False


def test_key_in_plan_publishes_nothing_and_is_not_logged(tmp_path, capsys, monkeypatch):
    secret = "sf4-planted-key"
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    def request(method, url, api_key, body=None, timeout=60):
        assert method == "GET"
        return 200, {"instances": _texas("running")}

    def write(run):
        _write_run(run, decision="accept", use_gpu=True)
        (run / "plan.json").write_bytes(b'{"note":"sf4-planted-key"}\n')
        ledger = (run / "ledger.json").read_text(encoding="utf-8")
        (run / "ledger.json").write_text(
            ledger.replace("Fruits", "Fruits sf4-planted-key", 1),
            encoding="utf-8",
        )

    plan_before = None

    def write_and_remember(run):
        write(run)
        nonlocal plan_before
        plan_before = (run / "plan.json").read_bytes()

    remote = _Remote(write_and_remember)
    code = run_smoke(
        ALLOWED_INSTANCE_ID,
        _checkout(tmp_path),
        artifacts,
        secret,
        request=request,
        remote_factory=lambda _row: remote,
        sleep=lambda _seconds: None,
    )
    assert code != 0
    assert list(artifacts.iterdir()) == []
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert "plan.json" in captured.err
    assert plan_before == b'{"note":"sf4-planted-key"}\n'


def test_key_in_ledger_publishes_nothing(tmp_path, capsys, monkeypatch):
    secret = "sf4-planted-key"
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    def request(method, url, api_key, body=None, timeout=60):
        assert method == "GET"
        return 200, {"instances": _texas("running")}

    def write(run):
        _write_run(run, decision="accept", use_gpu=True)
        (run / "plan.json").write_bytes(b'{"scene_name":"Fruits","from":"pipeline"}\n')
        ledger = (run / "ledger.json").read_text(encoding="utf-8")
        (run / "ledger.json").write_text(
            ledger.replace("Fruits", "sf4-planted-key", 1), encoding="utf-8"
        )

    remote = _Remote(write)
    code = run_smoke(
        ALLOWED_INSTANCE_ID,
        _checkout(tmp_path),
        artifacts,
        secret,
        request=request,
        remote_factory=lambda _row: remote,
        sleep=lambda _seconds: None,
    )
    assert code != 0
    assert list(artifacts.iterdir()) == []
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert "ledger.json" in captured.err


def test_manifest_key_fails_when_redaction_does_not_remove_it(
    tmp_path, capsys, monkeypatch
):
    secret = "sf4-planted-key"
    monkeypatch.setattr("sceneagent.vast_smoke.redact", lambda text, _secret: text)
    run = tmp_path / "run"
    run.mkdir()
    (run / "plan.json").write_bytes(b'{"scene_name":"Fruits"}\n')
    (run / "ledger.json").write_bytes(b'{"scene_name":"Fruits","entries":[]}\n')
    (run / "manifest.json").write_bytes(b'{"note":"sf4-planted-key"}\n')
    artifacts = tmp_path / "sf4-artifacts"
    with pytest.raises(SmokeError):
        publish_artifacts(run, artifacts, api_key=secret)
    written = (
        [path for path in artifacts.rglob("*") if path.is_file()]
        if artifacts.exists()
        else []
    )
    assert written == []
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert "manifest.json" in captured.err


def test_published_ledger_is_byte_identical_to_the_pipeline(tmp_path):
    run = tmp_path / "run"
    _write_run(run, decision="accept", use_gpu=True)
    (run / "plan.json").write_bytes(b'{"scene_name":"Fruits","from":"pipeline"}\n')
    before = (run / "ledger.json").read_bytes()
    collect_artifact_manifest(run, command=_command(), instance_id=59601312)
    artifacts = tmp_path / "artifacts"
    publish_artifacts(run, artifacts)
    assert (run / "ledger.json").read_bytes() == before
    assert (artifacts / "ledger.json").read_bytes() == before


def test_workflow_permissions_are_contents_read_only():
    text = WORKFLOW.read_text(encoding="utf-8")
    lines = text.splitlines()
    start = next(index for index, line in enumerate(lines) if line == "permissions:")
    grants = []
    for line in lines[start + 1 :]:
        if line and not line.startswith(" "):
            break
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            grants.append(stripped)
    assert grants == ["contents: read"]
    assert "contents: write" not in text
    assert "write" not in "\n".join(grants)


def test_upload_step_runs_only_after_success():
    lines = WORKFLOW.read_text(encoding="utf-8").splitlines()
    start = next(index for index, line in enumerate(lines) if "Upload plan" in line)
    step = "\n".join(lines[start : start + 4])
    assert "if: success()" in step
    assert "always()" not in step
    assert "failure()" not in step


def _ssh_clock():
    clock = {"t": 0.0}
    sleeps = []

    def now():
        return clock["t"]

    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) > SSH_READY_MAX_ATTEMPTS:
            raise AssertionError("ssh readiness slept without a bound")
        clock["t"] += seconds

    return clock, sleeps, now, sleep


def test_ssh_ready_succeeds_after_n_failures(capsys):
    _clock, sleeps, now, sleep = _ssh_clock()
    codes = [255, 255, 0]
    probes = {"n": 0}

    def probe():
        probes["n"] += 1
        if probes["n"] > 6:
            raise AssertionError("ssh readiness probe was not bounded")
        code = codes.pop(0)
        detail = "Permission denied (publickey)" if code != 0 else ""
        return code, detail

    ready = wait_until_ssh_accepts(
        probe,
        sleep=sleep,
        now=now,
        api_key="sf4-planted-key",
        max_attempts=6,
        timeout_s=45,
        backoff_s=2,
    )
    assert ready["accepted"] is True
    assert ready["attempts"] == 3
    assert codes == []
    assert len(sleeps) == 2
    captured = capsys.readouterr()
    assert "sf4-planted-key" not in captured.out
    assert "sf4-planted-key" not in captured.err


def _bounded_calls(counter, key, limit):
    def _tick():
        counter[key] += 1
        if counter[key] > limit:
            raise AssertionError(f"ssh readiness {key} was not bounded")

    return _tick


def test_ssh_ready_exhaustion_raises():
    counter = {"probes": 0, "sleeps": 0}
    probe_tick = _bounded_calls(counter, "probes", 3)
    sleep_tick = _bounded_calls(counter, "sleeps", 3)

    def probe():
        probe_tick()
        return 255, "Permission denied (publickey)"

    def sleep(_seconds):
        sleep_tick()

    with pytest.raises(SmokeError, match="did not accept the configured key"):
        wait_until_ssh_accepts(
            probe,
            sleep=sleep,
            now=lambda: 0.0,
            max_attempts=3,
            timeout_s=45,
            backoff_s=2,
        )
    assert counter["probes"] == 3


def test_ssh_ready_attempt_cap_is_not_an_unbounded_loop():
    probes = {"n": 0}

    def probe():
        probes["n"] += 1
        if probes["n"] > SSH_READY_MAX_ATTEMPTS:
            raise AssertionError("ssh readiness loop was not bounded")
        return 255, "Permission denied (publickey)"

    _clock, sleeps, now, sleep = _ssh_clock()
    with pytest.raises(SmokeError, match="did not accept"):
        wait_until_ssh_accepts(
            probe,
            sleep=sleep,
            now=lambda: 0.0,
            max_attempts=SSH_READY_MAX_ATTEMPTS,
            timeout_s=45,
            backoff_s=2,
        )
    assert probes["n"] == SSH_READY_MAX_ATTEMPTS
    assert len(sleeps) == SSH_READY_MAX_ATTEMPTS - 1


def test_ssh_ready_timeout_stops_before_max_attempts():
    probes = {"n": 0}
    _clock, sleeps, now, sleep = _ssh_clock()

    def probe():
        probes["n"] += 1
        if probes["n"] > 8:
            raise AssertionError("ssh readiness probe was not bounded")
        return 255, "Permission denied (publickey)"

    with pytest.raises(SmokeError, match="did not accept"):
        wait_until_ssh_accepts(
            probe,
            sleep=sleep,
            now=now,
            max_attempts=50,
            timeout_s=5,
            backoff_s=2,
        )
    assert probes["n"] < 50
    assert probes["n"] >= 1
    assert len(sleeps) < 50


def test_ssh_ready_log_omits_the_probe_text(capsys):
    secret = "sf4-planted-key"
    counter = {"probes": 0, "sleeps": 0}
    probe_tick = _bounded_calls(counter, "probes", 1)
    sleep_tick = _bounded_calls(counter, "sleeps", 1)

    def probe():
        probe_tick()
        return 255, f"Permission denied (publickey) {secret}"

    def sleep(_seconds):
        sleep_tick()

    with pytest.raises(SmokeError):
        wait_until_ssh_accepts(
            probe,
            sleep=sleep,
            now=lambda: 0.0,
            api_key=secret,
            max_attempts=1,
            timeout_s=45,
            backoff_s=2,
        )
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert "publickey" in captured.out


def test_attach_without_key_id_is_explicit_and_does_not_detach(capsys):
    codes = [255, 0]
    detached = []
    counter = {"probes": 0, "sleeps": 0}
    probe_tick = _bounded_calls(counter, "probes", 4)
    sleep_tick = _bounded_calls(counter, "sleeps", 4)

    def probe():
        probe_tick()
        code = codes.pop(0)
        return code, "Permission denied (publickey)" if code else ""

    def sleep(_seconds):
        sleep_tick()

    ready = await_ssh_after_attach(
        {"success": True, "msg": "SSH key attached successfully"},
        probe,
        api_key="sf4-planted-key",
        sleep=sleep,
        now=lambda: 0.0,
        detach=lambda key_id: detached.append(key_id),
        max_attempts=4,
        timeout_s=45,
        backoff_s=2,
    )
    assert ready["key_id_returned"] is False
    assert ready["key_id"] is None
    assert ready["accepted"] is True
    assert ready["detach"] == "skipped"
    assert ready["detached"] is False
    assert detached == []
    assert DETACH_LEFTOVER_KEYS is False
    captured = capsys.readouterr()
    assert '"key_id_returned": false' in captured.out
    assert '"detach": "skipped"' in captured.out
    assert "sf4-planted-key" not in captured.out
    assert "sf4-planted-key" not in captured.err
    assert PUBLIC_KEY not in captured.out


def test_attach_without_key_id_exhaustion_names_the_missing_id():
    detached = []
    counter = {"probes": 0, "sleeps": 0}
    probe_tick = _bounded_calls(counter, "probes", 2)
    sleep_tick = _bounded_calls(counter, "sleeps", 2)

    def probe():
        probe_tick()
        return 255, "Permission denied (publickey)"

    def sleep(_seconds):
        sleep_tick()

    with pytest.raises(SmokeError, match="no key id") as raised:
        await_ssh_after_attach(
            {"success": True},
            probe,
            api_key="sf4-planted-key",
            sleep=sleep,
            now=lambda: 0.0,
            detach=detached.append,
            max_attempts=2,
            timeout_s=45,
            backoff_s=2,
        )
    assert detached == []
    message = str(raised.value)
    assert "exactly one match" in message
    assert "not detached" not in message
    assert "sf4-planted-key" not in message
    assert "ssh-ed25519" not in message


def _written(directory):
    if not directory.exists():
        return []
    return [path for path in directory.rglob("*") if path.is_file()]


def test_failure_path_publishes_a_scanned_manifest_only(tmp_path):
    secret = "sf4-planted-key"
    artifacts = tmp_path / "sf4-artifacts"
    published = publish_failure_manifest(
        artifacts,
        api_key=secret,
        instance_id=ALLOWED_INSTANCE_ID,
        failure_reason="rsync failed and the log mentioned sf4-planted-key",
        attached_ssh_keys={"source": "show", "keys": [{"id": "9", "comment": "old"}]},
    )
    names = sorted(path.name for path in _written(artifacts))
    assert names == ["manifest.json"]
    text = (artifacts / "manifest.json").read_text(encoding="utf-8")
    assert secret not in text
    assert "[redacted]" in text
    assert published["passed"] is False
    assert published["plan"] is None
    assert published["ledger"] is None
    assert published["published_on_failure"] == "manifest.json"
    assert "plan.json" not in names
    assert "ledger.json" not in names


def test_failure_path_with_key_in_manifest_publishes_nothing(
    tmp_path, capsys, monkeypatch
):
    secret = "sf4-planted-key"
    monkeypatch.setattr("sceneagent.vast_smoke.redact", lambda text, _secret: text)
    artifacts = tmp_path / "sf4-artifacts"
    with pytest.raises(SmokeError):
        publish_failure_manifest(
            artifacts,
            api_key=secret,
            instance_id=ALLOWED_INSTANCE_ID,
            failure_reason="saw sf4-planted-key",
        )
    assert _written(artifacts) == []
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert "manifest.json" in captured.err


def test_failure_upload_step_is_manifest_only():
    lines = WORKFLOW.read_text(encoding="utf-8").splitlines()
    start = next(
        index for index, line in enumerate(lines) if "Upload redacted failure" in line
    )
    end = next(
        (
            index
            for index, line in enumerate(lines[start + 1 :], start + 1)
            if line.startswith("  ")
            and not line.startswith("   ")
            and line.endswith(":")
        ),
        len(lines),
    )
    step = "\n".join(lines[start:end])
    assert "if: failure()" in step
    assert "sf4-artifacts/manifest.json" in step
    assert "plan.json" not in step
    assert "ledger.json" not in step
    assert "if-no-files-found: error" in step
    assert "always()" not in step
    success = "\n".join(lines[:start])
    assert "if: success()" in success
    assert "sf4-artifacts/plan.json" in success
    assert "sf4-artifacts/ledger.json" in success


def test_run_smoke_ssh_failure_publishes_only_the_failure_manifest(
    tmp_path, capsys, monkeypatch
):
    calls = []
    secret = "sf4-planted-key"
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)

    def request(method, url, api_key, body=None, timeout=60):
        calls.append(method)
        assert method != "DELETE"
        assert method == "GET"
        return 200, {"instances": _texas("running", ssh_keys=[_sample_ssh_key()])}

    class _Boom:
        ssh_report = {
            "key_id_returned": False,
            "detach": "skipped",
            "accepted": False,
        }

        def run(self, checkout, local_run):
            raise SmokeError("rsync failed (255): Permission denied (publickey)")

        def close(self):
            self.closed = True

    remote = _Boom()
    artifacts = tmp_path / "sf4-artifacts"
    code = run_smoke(
        ALLOWED_INSTANCE_ID,
        _checkout(tmp_path),
        artifacts,
        secret,
        request=request,
        remote_factory=lambda _row: remote,
        sleep=lambda _seconds: None,
    )
    assert code == 1
    assert calls == ["GET"]
    names = sorted(path.name for path in _written(artifacts))
    assert names == ["manifest.json"]
    text = (artifacts / "manifest.json").read_text(encoding="utf-8")
    assert secret not in text
    assert PUBLIC_KEY.split()[1] not in text
    manifest = json.loads(text)
    assert manifest["passed"] is False
    assert manifest["plan"] is None
    assert manifest["ledger"] is None
    assert "publickey" in manifest["failure_reason"]
    assert manifest["ssh"]["key_id_returned"] is False
    assert manifest["ssh"]["detach"] == "skipped"
    assert manifest["detach_leftover_keys"] is False
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err


def _sample_ssh_key():
    return {
        "id": 77,
        "public_key": PUBLIC_KEY,
        "created_at": "2026-11-01T00:00:00Z",
    }


def _fingerprint(public_key):
    blob = public_key.split()[1]
    pad = "=" * ((4 - len(blob) % 4) % 4)
    raw = base64.b64decode(blob + pad)
    digest = base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")
    return "SHA256:" + digest.rstrip("=")


def test_listing_attached_ssh_keys_uses_pinned_get_and_fingerprints():
    calls = []

    def request(method, url, api_key, body=None, timeout=60):
        calls.append((method, url, body))
        if method == "DELETE":
            raise AssertionError("detach")
        assert method == "GET"
        assert url == "https://console.vast.ai/api/v0/instances/59601312/ssh/"
        assert body is None
        return 200, {"success": True, "ssh_keys": json.dumps([_sample_ssh_key()])}

    report = list_attached_ssh_keys(request, "test-key", sleep=lambda _seconds: None)
    assert calls and all(method == "GET" for method, _url, _body in calls)
    assert report["count"] == 1
    assert report["fingerprints"] == [_fingerprint(PUBLIC_KEY)]
    assert "keys" not in report
    assert DETACH_LEFTOVER_KEYS is False
    blob = json.dumps(report)
    assert PUBLIC_KEY.split()[1] not in blob
    assert "PRIVATE" not in blob
    assert "77" not in blob
    assert "sf4-smoke" not in blob
    assert "2026-11-01" not in blob


def test_show_body_json_string_keeps_id_and_drops_the_public_key():
    payload = _texas("running", ssh_keys=json.dumps([_sample_ssh_key()]))
    report = attached_ssh_keys_from_show(payload)
    assert report["present_in_show"] is True
    assert report["keys"][0]["id"] == "77"
    assert report["keys"][0]["comment"] == "sf4-smoke"
    assert PUBLIC_KEY.split()[1] not in json.dumps(report)


def test_show_body_without_keys_records_absence_and_makes_no_detach():
    calls = []

    def request(method, url, api_key, body=None, timeout=60):
        calls.append(method)
        assert method == "GET"
        return 200, {"instances": _texas("running")}

    report = list_attached_ssh_keys(request, "test-key", sleep=lambda _seconds: None)
    assert calls == ["GET"]
    assert report == {"count": 0, "fingerprints": []}
    assert "detach_leftover_keys" not in report
    assert DETACH_LEFTOVER_KEYS is False
    direct = attached_ssh_keys_from_show(_texas("running"))
    assert direct["keys"] == []


def test_allowlist_still_rejects_ssh_list_and_instance_mutations():
    for kind in ("create", "rent", "resize", "destroy", "search"):
        with pytest.raises(InstanceNotApproved):
            endpoint_for(kind, 59601312)
    with pytest.raises(InstanceNotApproved):
        assert_request_allowed(
            "DELETE",
            "https://console.vast.ai/api/v0/instances/59601312/",
            None,
        )


def test_run_smoke_does_not_publish_show_body_key_material(tmp_path):
    calls = []

    def request(method, url, api_key, body=None, timeout=60):
        calls.append(method)
        assert method == "GET"
        assert "/ssh/" not in url
        return 200, {"instances": _texas("running", ssh_keys=[_sample_ssh_key()])}

    def write_passing_run(run):
        _write_run(run, decision="accept", use_gpu=True)
        (run / "plan.json").write_bytes(b'{"scene_name":"Fruits","from":"pipeline"}\n')

    code = run_smoke(
        ALLOWED_INSTANCE_ID,
        _checkout(tmp_path),
        tmp_path / "artifacts",
        "test-key",
        request=request,
        remote_factory=lambda _row: _Remote(write_passing_run),
        sleep=lambda _seconds: None,
    )
    assert code == 0
    assert calls == ["GET"]
    assert "DELETE" not in calls
    manifest = json.loads(
        (tmp_path / "artifacts" / "manifest.json").read_text(encoding="utf-8")
    )
    assert "attached_ssh_keys" not in manifest
    assert PUBLIC_KEY.split()[1] not in json.dumps(manifest)

    def list_request(method, url, api_key, body=None, timeout=60):
        assert method == "GET"
        assert url.endswith("/59601312/ssh/")
        assert body is None
        return 200, {"success": True, "ssh_keys": json.dumps([_sample_ssh_key()])}

    summary = list_attached_ssh_keys(
        list_request, "test-key", sleep=lambda _seconds: None
    )
    assert summary == {"count": 1, "fingerprints": [_fingerprint(PUBLIC_KEY)]}
    recorded = json.dumps(summary)
    assert "77" not in recorded
    assert "sf4-smoke" not in recorded


PINNED_SSH_LIST = "https://console.vast.ai/api/v0/instances/59601312/ssh/"


def test_pinned_ssh_list_get_is_allowed():
    method, url, body = endpoint_for("show_ssh_keys", 59601312)
    assert method == "GET"
    assert url == PINNED_SSH_LIST
    assert body is None
    assert_request_allowed(method, url, body)
    with pytest.raises(InstanceNotApproved):
        endpoint_for("show_ssh_keys", 58357498)


@pytest.mark.parametrize(
    ("method", "url", "body"),
    [
        ("POST", PINNED_SSH_LIST, None),
        ("POST", PINNED_SSH_LIST, {}),
        ("POST", PINNED_SSH_LIST, {"ssh_key": PUBLIC_KEY, "extra": True}),
        ("PUT", PINNED_SSH_LIST, None),
        ("PUT", PINNED_SSH_LIST, {"ssh_key": PUBLIC_KEY}),
        ("DELETE", PINNED_SSH_LIST, None),
        ("GET", PINNED_SSH_LIST + "?owner=me", None),
        ("GET", "https://console.vast.ai/api/v0/ssh/", None),
        ("GET", "https://console.vast.ai/ssh/", None),
        ("GET", "https://console.vast.ai/api/v0/instances/ssh/", None),
        ("GET", "https://console.vast.ai/api/v0/instances/1/ssh/", None),
        ("GET", "https://console.vast.ai/api/v0/instances/58357498/ssh/", None),
        ("GET", "https://console.vast.ai/api/v0/instances/59601312/ssh", None),
        ("GET", "https://console.vast.ai/api/v0/instances/59601312/ssh/1/", None),
        ("GET", "https://console.vast.ai/api/v0/instances/59601312/ssh/1/extra/", None),
        (
            "POST",
            "https://console.vast.ai/api/v0/instances/59601312/ssh/1/",
            {"ssh_key": PUBLIC_KEY},
        ),
        ("PUT", "https://console.vast.ai/api/v0/instances/59601312/ssh/1/", None),
        (
            "DELETE",
            "https://console.vast.ai/api/v0/instances/59601312/ssh/1/extra/",
            None,
        ),
    ],
)
def test_other_ssh_paths_and_methods_are_refused(method, url, body):
    with pytest.raises(InstanceNotApproved):
        assert_request_allowed(method, url, body)


def test_ssh_list_manifest_has_count_and_fingerprints_only(tmp_path):
    def request(method, url, api_key, body=None, timeout=60):
        assert method == "GET"
        assert url == PINNED_SSH_LIST
        assert body is None
        listed = {
            "id": 77,
            "name": "sf4-smoke",
            "public_key": PUBLIC_KEY,
            "fingerprint": "SHA256:from-the-server",
            "created_at": "2026-11-01T00:00:00Z",
        }
        return 200, {"success": True, "ssh_keys": json.dumps([listed])}

    summary = load_ssh_key_fingerprints(request, "test-key", lambda _seconds: None)
    assert summary == {"count": 1, "fingerprints": [_fingerprint(PUBLIC_KEY)]}
    published = publish_failure_manifest(
        tmp_path / "sf4-artifacts",
        api_key="test-key",
        instance_id=ALLOWED_INSTANCE_ID,
        failure_reason="ssh list",
        ssh_keys=summary,
    )
    assert published["ssh_keys"] == summary
    text = json.dumps(published)
    assert PUBLIC_KEY.split()[1] not in text
    assert "from-the-server" not in text
    assert "sf4-smoke" not in text
    assert "77" not in text
    assert "2026-11-01" not in text


def test_ssh_list_response_key_does_not_leak_into_artifacts(tmp_path, capsys):
    secret = "sf4-planted-key"

    def request(method, url, api_key, body=None, timeout=60):
        assert method == "GET"
        assert url == PINNED_SSH_LIST
        poisoned = {
            "note": secret,
            "ssh_keys": json.dumps(
                [
                    {
                        "id": 9,
                        "name": secret,
                        "public_key": PUBLIC_KEY,
                        "private_key": "-----BEGIN OPENSSH PRIVATE KEY-----\n" + secret,
                        "fingerprint": "SHA256:from-the-server",
                    },
                    {"public_key": f"ssh-ed25519 {secret} leaked"},
                ]
            ),
        }
        return 200, poisoned

    summary = list_attached_ssh_keys(request, secret, sleep=lambda _seconds: None)
    dirty = {
        "count": 99,
        "fingerprints": summary["fingerprints"],
        "public_key": PUBLIC_KEY,
        "private_key": secret,
        "note": secret,
    }
    artifacts = tmp_path / "sf4-artifacts"
    publish_failure_manifest(
        artifacts,
        api_key=secret,
        instance_id=ALLOWED_INSTANCE_ID,
        failure_reason="ssh list",
        ssh_keys=dirty,
    )
    files = [path for path in artifacts.rglob("*") if path.is_file()]
    assert [path.name for path in files] == ["manifest.json"]
    text = files[0].read_text(encoding="utf-8")
    assert secret not in text
    assert PUBLIC_KEY.split()[1] not in text
    assert "BEGIN" not in text
    assert "from-the-server" not in text
    assert summary["fingerprints"][0] in text
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err


def test_ephemeral_fingerprint_is_checked_against_the_post_attach_list():
    calls = []

    def request(method, url, api_key, body=None, timeout=60):
        calls.append((method, url, body))
        assert method == "GET"
        assert url == PINNED_SSH_LIST
        assert body is None
        if len(calls) == 1:
            return 200, {"success": True, "ssh_keys": "[]"}
        listed = {"id": 3, "public_key": PUBLIC_KEY, "name": "sf4-smoke"}
        return 200, {"success": True, "ssh_keys": json.dumps([listed])}

    before = load_ssh_key_fingerprints(request, "test-key", lambda _seconds: None)
    after = load_ssh_key_fingerprints(request, "test-key", lambda _seconds: None)
    record = ephemeral_key_record(PUBLIC_KEY, after)
    missing = ephemeral_key_record(PUBLIC_KEY, before)
    assert before == {"count": 0, "fingerprints": []}
    assert after["count"] == 1
    assert record["fingerprint"] == _fingerprint(PUBLIC_KEY)
    assert record["listed_after_attach"] is True
    assert missing["listed_after_attach"] is False
    assert len(calls) == 2
    assert all(method == "GET" and body is None for method, _url, body in calls)
    blob = json.dumps({"ssh_keys": after, "ephemeral_ssh_key": record})
    assert PUBLIC_KEY.split()[1] not in blob
    assert "BEGIN" not in blob
    assert DETACH_LEFTOVER_KEYS is False


def _bound_sleep(counter, limit=8):
    def sleep(_seconds):
        counter["sleeps"] += 1
        if counter["sleeps"] > limit:
            raise AssertionError("ssh readiness slept without a bound")

    return sleep


def test_ssh_ready_defaults_are_six_probes_and_forty_five_seconds():
    assert SSH_READY_MAX_ATTEMPTS == 6
    assert SSH_READY_TIMEOUT_S == 45.0
    defaults = wait_until_ssh_accepts.__kwdefaults__
    assert defaults is not None
    assert defaults["max_attempts"] == 6
    assert defaults["timeout_s"] == 45.0
    counter = {"probes": 0, "sleeps": 0}

    def probe():
        counter["probes"] += 1
        if counter["probes"] > 6:
            raise AssertionError("default cap did not stop the loop")
        return 255, "Permission denied (publickey)"

    with pytest.raises(SmokeError, match=r"cap 6, timeout 45s"):
        wait_until_ssh_accepts(probe, sleep=_bound_sleep(counter), now=lambda: 0.0)
    assert counter["probes"] == 6
    assert counter["sleeps"] == 5


def test_ssh_ready_default_timeout_stops_at_forty_five_seconds():
    seen = {"n": 0}

    def now():
        seen["n"] += 1
        return 0.0 if seen["n"] == 1 else 45.0

    counter = {"probes": 0, "sleeps": 0}

    def probe():
        counter["probes"] += 1
        if counter["probes"] > 2:
            raise AssertionError("default timeout did not stop the loop")
        return 255, "Permission denied (publickey)"

    with pytest.raises(SmokeError, match="timeout 45s"):
        wait_until_ssh_accepts(probe, sleep=_bound_sleep(counter), now=now)
    assert counter["probes"] == 1
    assert counter["sleeps"] == 0


def _bare_remote(sleep):
    remote = SshRemote.__new__(SshRemote)
    remote.api_key = "test-key"
    remote.sleep = sleep
    remote.key_id_returned = False
    remote.ssh_report = {"accepted": None}
    remote.host = "ssh6.vast.ai"
    remote.port = 22
    remote.private = Path("/tmp/sf4_ed25519")
    remote.known_hosts = "/tmp/known_hosts"
    remote.public = PUBLIC_KEY
    return remote


def test_ssh_remote_sync_retries_the_probe_then_rsyncs(tmp_path, monkeypatch):
    counter = {"probes": 0, "sleeps": 0, "rsync": 0}
    codes = [255, 255, 0]

    def probe():
        counter["probes"] += 1
        if counter["probes"] > 8:
            raise AssertionError("sync probe loop was not bounded")
        code = codes.pop(0)
        detail = "Permission denied (publickey)" if code else ""
        return code, detail

    def run_rsync(*_args, **_kwargs):
        counter["rsync"] += 1
        completed = subprocess.CompletedProcess(args=[], returncode=0, stderr=b"")
        return completed

    monkeypatch.setattr("sceneagent.vast_smoke.subprocess.run", run_rsync)
    remote = _bare_remote(_bound_sleep(counter))
    remote._probe_ssh = probe
    remote._sync(_checkout(tmp_path))
    assert counter["probes"] == 3
    assert counter["rsync"] == 2
    assert remote.ssh_report["accepted"] is True
    assert "detach" not in remote.ssh_report
    assert codes == []


def test_ssh_remote_sync_fails_cleanly_when_the_probe_bound_runs_out(
    tmp_path, monkeypatch
):
    counter = {"probes": 0, "sleeps": 0}

    def probe():
        counter["probes"] += 1
        if counter["probes"] > 8:
            raise AssertionError("sync probe loop was not bounded")
        return 255, "Permission denied (publickey)"

    def run_rsync(*_args, **_kwargs):
        raise AssertionError("rsync ran after the probe bound")

    monkeypatch.setattr("sceneagent.vast_smoke.subprocess.run", run_rsync)
    remote = _bare_remote(_bound_sleep(counter))
    remote._probe_ssh = probe
    with pytest.raises(SmokeError, match="did not accept the configured key") as raised:
        remote._sync(_checkout(tmp_path))
    message = str(raised.value)
    assert "attach_ssh" not in message
    assert "detach" not in message.lower()
    assert counter["probes"] == 6
    assert remote.ssh_report["accepted"] is False


def test_put_state_body_on_the_ssh_path_is_refused():
    for state in ("running", "stopped"):
        with pytest.raises(InstanceNotApproved):
            assert_request_allowed(
                "PUT",
                PINNED_SSH_LIST,
                {"state": state},
            )
    with pytest.raises(InstanceNotApproved):
        assert_request_allowed(
            "PUT",
            "https://console.vast.ai/api/v0/instances/59601312/",
            {"state": "paused"},
        )
    with pytest.raises(InstanceNotApproved):
        endpoint_for("start", 59601312)
    with pytest.raises(InstanceNotApproved):
        endpoint_for("stop", 59601312)
    with pytest.raises(InstanceNotApproved):
        assert_request_allowed(
            "PUT",
            "https://console.vast.ai/api/v0/instances/59601312/",
            {"state": "running"},
        )
    with pytest.raises(InstanceNotApproved):
        assert_request_allowed(
            "PUT",
            "https://console.vast.ai/api/v0/instances/59601312/",
            {"state": "stopped"},
        )


def test_patch_and_put_on_the_pinned_ssh_path_are_refused():
    for method in ("PATCH", "PUT"):
        with pytest.raises(InstanceNotApproved):
            assert_request_allowed(method, PINNED_SSH_LIST, None)
        with pytest.raises(InstanceNotApproved):
            assert_request_allowed(method, PINNED_SSH_LIST, {"ssh_key": PUBLIC_KEY})


def test_pinned_ssh_get_guard_accepts_only_that_path():
    assert_request_allowed("GET", PINNED_SSH_LIST, None)
    refused = [
        ("GET", PINNED_SSH_LIST + "?owner=me", None),
        ("GET", "https://console.vast.ai/api/v0/instances/1/ssh/", None),
        ("GET", "https://console.vast.ai/api/v0/instances/59601312/ssh/77/", None),
        ("GET", "https://console.vast.ai/api/v0/ssh/", None),
        ("POST", PINNED_SSH_LIST, None),
        ("PATCH", PINNED_SSH_LIST, None),
        ("PUT", PINNED_SSH_LIST, None),
        ("DELETE", PINNED_SSH_LIST, None),
    ]
    for method, url, body in refused:
        with pytest.raises(InstanceNotApproved):
            assert_request_allowed(method, url, body)


def _listing_transport(mode):
    """Mock Vast. ``mode`` is one, zero, or several matches for the posted key."""
    calls = []
    seen = {"key": None, "gets": 0}

    def request(method, url, api_key, body=None, timeout=60):
        calls.append((method, url))
        if method == "DELETE":
            if mode == "http":
                return 500, {"success": False}
            if mode == "raise":
                raise RuntimeError("transport down")
            return 200, {"success": True}
        if method == "POST" and url == PINNED_SSH_LIST:
            seen["key"] = body["ssh_key"]
            return 200, {"success": True}
        if method == "GET" and url == PINNED_SSH_LIST:
            seen["gets"] += 1
            if seen["gets"] == 1 or seen["key"] is None:
                return 200, {"success": True, "ssh_keys": []}
            listed = seen["key"] + "\n"
            key = seen["key"]
            if mode == "zero":
                rows = [{"id": 9, "public_key": PUBLIC_KEY + " other"}]
            elif mode == "several":
                rows = [
                    {"id": 77, "public_key": listed},
                    {"id": 78, "public_key": key + " "},
                ]
            elif mode == "trailing":
                rows = [{"id": 77, "public_key": key + "\n  "}]
            elif mode == "leading":
                rows = [{"id": 77, "public_key": " " + key}]
            elif mode == "substring":
                rows = [{"id": 77, "public_key": key + " extra"}]
            elif mode == "reverse":
                rows = [{"id": 77, "public_key": key[::-1]}]
            elif mode == "case":
                kind, blob, *rest = key.split(" ")
                cased = kind + " " + blob.swapcase()
                if rest:
                    cased += " " + " ".join(rest)
                rows = [{"id": 77, "public_key": cased}]
            else:
                rows = [{"id": 77, "public_key": listed}]
            return 200, {"success": True, "ssh_keys": json.dumps(rows)}
        if method == "GET":
            return 200, {"instances": _texas("running")}
        raise AssertionError((method, url, body))

    return calls, request


def _install_remote_run(monkeypatch, fail):
    def fake_run(self, checkout, local_run):
        if fail:
            raise SmokeError("remote failed before rsync")
        local_run.mkdir(parents=True, exist_ok=True)
        _write_run(local_run, decision="accept", use_gpu=True)
        (local_run / "plan.json").write_bytes(
            b'{"scene_name":"Fruits","from":"pipeline"}\n'
        )
        return 0

    monkeypatch.setattr(SshRemote, "run", fake_run)


def test_running_smoke_does_not_attach_or_delete_remote_keys(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    calls, request = _listing_transport("one")
    _install_remote_run(monkeypatch, fail=False)
    artifacts = tmp_path / "ok"
    code = run_smoke(
        ALLOWED_INSTANCE_ID,
        _checkout(tmp_path),
        artifacts,
        "test-key",
        request=request,
        sleep=lambda _seconds: None,
        ssh_private_key=_FAKE_PRIVATE,
        known_host_file=_host_pin(tmp_path),
    )
    assert code == 0
    methods = [method for method, _url in calls]
    assert "POST" not in methods
    assert "DELETE" not in methods
    text = (artifacts / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(text)
    assert "own_key" not in manifest
    assert _FAKE_PRIVATE.strip() not in text


def test_exit_code_survives_failure_manifest_error(tmp_path, monkeypatch):
    def boom(*_args, **_kwargs):
        raise SmokeError("manifest publish failed")

    monkeypatch.setattr("sceneagent.vast_smoke.publish_failure_manifest", boom)

    def offline(method, url, api_key, body=None, timeout=60):
        assert method == "GET"
        return 200, {"instances": _texas("offline")}

    code = run_smoke(
        ALLOWED_INSTANCE_ID,
        _checkout(tmp_path),
        tmp_path / "offline",
        "test-key",
        request=offline,
        sleep=lambda _seconds: None,
    )
    assert code == 2

    monkeypatch.setattr("sceneagent.vast_smoke.publish_artifacts", boom)

    def running(method, url, api_key, body=None, timeout=60):
        assert method == "GET"
        return 200, {"instances": _texas("running")}

    def write(run):
        _write_run(run, decision="accept", use_gpu=True)
        (run / "plan.json").write_bytes(b'{"scene_name":"Fruits","from":"pipeline"}\n')

    for rc in (2, 139):
        code = run_smoke(
            ALLOWED_INSTANCE_ID,
            _checkout(tmp_path),
            tmp_path / f"rc-{rc}",
            "test-key",
            request=running,
            remote_factory=lambda _row, exit_code=rc: _Remote(write, rc=exit_code),
            sleep=lambda _seconds: None,
        )
        assert code == rc


def test_detach_guard_refuses_every_remote_key_delete(monkeypatch):
    monkeypatch.setattr("sceneagent.vast_smoke.DETACH_LEFTOVER_KEYS", True)
    calls = []

    def request(method, url, api_key, body=None, timeout=60):
        calls.append((method, url))
        return 200, {"success": True}

    urls = (
        "https://console.vast.ai/api/v0/instances/59601312/ssh/77/",
        "https://console.vast.ai/api/v0/instances/59601312/ssh/78/",
        "https://console.vast.ai/api/v0/instances/1/ssh/77/",
    )
    for url in urls:
        with pytest.raises(InstanceNotApproved, match="refusing to delete"):
            guard_own_key_delete("DELETE", url)
        with pytest.raises(InstanceNotApproved, match="refusing to delete"):
            call_json(request, "DELETE", url, "test-key", None, sleep=lambda _s: None)
    assert calls == []


def test_ssh_argv_does_not_forward_environment():
    argv = ssh_base_argv("/tmp/sf4_ed25519", "/tmp/known_hosts", 22)
    assert ssh_forwarding_problems(argv) == []
    assert "ForwardAgent=no" in argv
    assert "ClearAllForwardings=yes" in argv
    assert "/dev/null" in argv
    forwarded = [*argv, "-o", "SendEnv=GITHUB_TOKEN"]
    assert any("SendEnv" in problem for problem in ssh_forwarding_problems(forwarded))


def test_rsync_does_not_copy_git_or_credential_files(tmp_path):
    checkout = _checkout(tmp_path)
    git_dir = checkout / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text(
        "extraheader = AUTHORIZATION: basic planted-git-token\n",
        encoding="utf-8",
    )
    (checkout / "sceneagent" / "credentials.json").write_text(
        "token\n", encoding="utf-8"
    )
    nested = checkout / "sceneagent" / ".git"
    nested.mkdir()
    (nested / "config").write_text("planted-git-token\n", encoding="utf-8")
    shipped = [
        path.relative_to(checkout).as_posix() for path in iter_shipped_files(checkout)
    ]
    assert shipped
    assert all(".git" not in path.split("/") for path in shipped)
    assert "sceneagent/credentials.json" not in shipped
    invocations = rsync_invocations(
        checkout,
        "ssh6.vast.ai",
        22,
        "/tmp/sf4_ed25519",
        "/tmp/known_hosts",
        REMOTE_REPO,
    )
    blob = " ".join(" ".join(command) for command, _cwd in invocations)
    assert "--exclude" in blob
    assert ".git" in blob
    assert "planted-git-token" not in blob


def test_mentioning_the_key_name_is_not_treated_as_forwarding_the_value(tmp_path):
    checkout = _checkout(tmp_path)
    (checkout / "sceneagent" / "note.py").write_text(
        'os.environ["VAST_API_KEY"]\n',
        encoding="utf-8",
    )
    record, problems = assess_secret_forwarding(checkout, "sf4-api-key-value")
    assert problems == []
    assert record["secrets_forwarded"] is False
    assert record["payload_secret_hits"] == 0


def _calls_for_status(status):
    """Record every HTTP call. A start PUT is what the refusal tests forbid.

    After a start PUT the next show is running, so a missing guard fails the
    PUT assertion instead of sitting in the boot poll.
    """
    calls = []
    phase = {"started": False}

    def request(method, url, api_key, body=None, timeout=60):
        recorded = None if body is None else dict(body)
        calls.append((method, url, recorded))
        if method == "GET":
            shown = "running" if phase["started"] else status
            return 200, {"instances": _texas(shown)}
        if method == "PUT" and recorded == {"state": "running"}:
            phase["started"] = True
            return 200, {"success": True}
        if method == "PUT":
            return 200, {"success": True}
        return 200, {"success": True}

    return calls, request


def _puts(calls):
    return [
        call
        for call in calls
        if call[0] == "PUT" or (call[2] or {}).get("state") == "running"
    ]


def test_label_trigger_does_not_start_a_stopped_instance(tmp_path, capsys):
    calls, request = _calls_for_status("exited")
    code = run_smoke(
        ALLOWED_INSTANCE_ID,
        _checkout(tmp_path),
        tmp_path / "artifacts",
        "test-key",
        request=request,
        sleep=lambda _seconds: None,
        trigger="label",
        actor="cursor[bot]",
        commit_sha="a" * 40,
    )
    assert code != 0
    assert _puts(calls) == []
    assert (
        "instance stopped; this workflow does not start or stop instances"
        in capsys.readouterr().err
    )
    manifest = json.loads(
        (tmp_path / "artifacts" / "manifest.json").read_text(encoding="utf-8")
    )
    assert (
        manifest["failure_reason"]
        == "instance stopped; this workflow does not start or stop instances"
    )
    assert manifest["passed"] is False


def test_request_file_trigger_does_not_start_a_stopped_instance(tmp_path, capsys):
    calls, request = _calls_for_status("stopped")
    code = run_smoke(
        ALLOWED_INSTANCE_ID,
        _checkout(tmp_path),
        tmp_path / "artifacts",
        "test-key",
        request=request,
        sleep=lambda _seconds: None,
        trigger="request-file",
        actor="sandy-t-shelvan",
        commit_sha="b" * 40,
    )
    assert code != 0
    assert _puts(calls) == []
    assert (
        "instance stopped; this workflow does not start or stop instances"
        in capsys.readouterr().err
    )


def _run_bot_smoke(tmp_path, status, trigger, actor, commit_sha):
    """Run one bot trigger. A fast clock keeps a missing wait from polling."""
    calls, request = _calls_for_status(status)
    clock = {"t": 0.0}

    def now():
        clock["t"] += 1.0
        return clock["t"]

    code = run_smoke(
        ALLOWED_INSTANCE_ID,
        _checkout(tmp_path),
        tmp_path / "artifacts",
        "test-key",
        request=request,
        sleep=lambda _seconds: None,
        now=now,
        boot_timeout_s=0.5,
        trigger=trigger,
        actor=actor,
        commit_sha=commit_sha,
    )
    return calls, code


# Exact text InstanceNotApproved raises for a starting instance. Do not loosen
# this to a substring: a longer message would still contain the words.
_STARTING_REFUSAL = (
    "instance not running; this workflow does not start or stop instances"
)


def _assert_starting_refusal(tmp_path, capsys, calls, code):
    assert code != 0
    assert _puts(calls) == []
    assert capsys.readouterr().err == _STARTING_REFUSAL + "\n"
    manifest = json.loads(
        (tmp_path / "artifacts" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["failure_reason"] == _STARTING_REFUSAL
    assert manifest["passed"] is False


def test_label_trigger_refuses_a_starting_instance(tmp_path, capsys):
    # Vast reports loading while the instance is coming up. That is starting.
    calls, code = _run_bot_smoke(tmp_path, "loading", "label", "cursor[bot]", "a" * 40)
    _assert_starting_refusal(tmp_path, capsys, calls, code)


def test_request_file_trigger_refuses_a_starting_instance(tmp_path, capsys):
    calls, code = _run_bot_smoke(
        tmp_path, "loading", "request-file", "sandy-t-shelvan", "b" * 40
    )
    _assert_starting_refusal(tmp_path, capsys, calls, code)


def test_bot_trigger_refuses_an_unknown_instance_state(tmp_path, capsys):
    calls, request = _calls_for_status("offline")
    code = run_smoke(
        ALLOWED_INSTANCE_ID,
        _checkout(tmp_path),
        tmp_path / "artifacts",
        "test-key",
        request=request,
        sleep=lambda _seconds: None,
        trigger="label",
        actor="cursor[bot]",
        commit_sha="a" * 40,
    )
    assert code != 0
    assert _puts(calls) == []
    assert (
        "instance state unknown; this workflow does not start or stop instances"
        in capsys.readouterr().err
    )


def test_missing_trigger_does_not_start_a_stopped_instance(tmp_path):
    calls, request = _calls_for_status("exited")
    code = run_smoke(
        ALLOWED_INSTANCE_ID,
        _checkout(tmp_path),
        tmp_path / "artifacts",
        "test-key",
        request=request,
        sleep=lambda _seconds: None,
    )
    assert code != 0
    assert _puts(calls) == []


def test_workflow_dispatch_cannot_start_or_stop(tmp_path):
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in text
    assert "58301075" not in text
    assert "--instance-id 59601312" in text
    assert "may start" not in text.lower()
    assert '{"state": "running"}' not in text
    assert '{"state": "stopped"}' not in text
    source = (ROOT / "sceneagent" / "vast_smoke.py").read_text(encoding="utf-8")
    assert "may_start_instance" not in source
    assert "_change_state" not in source
    calls = []

    def request(method, url, api_key, body=None, timeout=60):
        recorded = None if body is None else dict(body)
        calls.append((method, url, recorded))
        if method != "GET":
            raise AssertionError((method, recorded))
        return 200, {"instances": _texas("stopped")}

    code = run_smoke(
        ALLOWED_INSTANCE_ID,
        _checkout(tmp_path),
        tmp_path / "artifacts",
        "test-key",
        request=request,
        sleep=lambda _seconds: None,
        trigger="dispatch",
        actor="sandy-t-shelvan",
        commit_sha="c" * 40,
    )
    assert code == 2
    assert _puts(calls) == []


def test_orchestrator_refuses_when_payload_is_importable(tmp_path, monkeypatch, capsys):
    secret = "sf4-not-a-real-key"
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setenv("PYTHONPATH", "payload")
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "sitecustomize.py").write_text("print('imported')\n", encoding="utf-8")
    monkeypatch.setenv("PYTHONSTARTUP", str(payload / "sitecustomize.py"))
    monkeypatch.syspath_prepend(str(payload))
    pth_dir = tmp_path / "site"
    pth_dir.mkdir()
    (pth_dir / "extra.pth").write_text("payload\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(pth_dir))

    def request(*_args, **_kwargs):
        raise AssertionError("GPU call")

    blockers = payload_import_blockers()
    assert "PYTHONPATH names payload" in blockers
    assert "PYTHONSTARTUP names payload" in blockers
    assert "sys.path names payload" in blockers
    assert "sitecustomize.py is under payload" in blockers
    assert "a .pth file adds payload" in blockers
    code = run_smoke(
        ALLOWED_INSTANCE_ID,
        _checkout(tmp_path),
        tmp_path / "artifacts",
        secret,
        request=request,
        sleep=lambda _seconds: None,
    )
    assert code == 2
    assert not (tmp_path / "artifacts").exists()
    captured = capsys.readouterr()
    assert "No instance call was made." in captured.err
    assert secret not in captured.out + captured.err


def test_secret_in_payload_is_refused_before_any_gpu_call(
    tmp_path, monkeypatch, capsys
):
    secret = "sf4-payload-secret"
    monkeypatch.setenv("GITHUB_TOKEN", "sf4-github-token-value")
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    checkout = _checkout(tmp_path)
    (checkout / "sceneagent" / "note.txt").write_text(secret + "\n", encoding="utf-8")

    def request(*_args, **_kwargs):
        raise AssertionError("GPU call")

    code = run_smoke(
        ALLOWED_INSTANCE_ID,
        checkout,
        tmp_path / "artifacts",
        secret,
        request=request,
        sleep=lambda _seconds: None,
        trigger="label",
        actor="cursor[bot]",
        commit_sha="a" * 40,
    )
    assert code == 2
    assert not (tmp_path / "artifacts").exists()
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert "sf4-github-token-value" not in captured.out
    assert "sf4-github-token-value" not in captured.err
    assert "No instance call was made." in captured.err


def test_git_credentials_in_the_checkout_are_not_forwarded(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "sf4-git-credential")
    checkout = _checkout(tmp_path)
    git_dir = checkout / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text(
        "[http]\n\textraheader = AUTHORIZATION: basic sf4-git-credential\n",
        encoding="utf-8",
    )
    record, problems = assess_secret_forwarding(checkout, "sf4-api-key-value")
    assert problems == []
    assert record["secrets_forwarded"] is False
    assert record["git_credentials_shipped"] is False


def test_manifest_records_trigger_actor_sha_and_no_secret_forwarded(
    tmp_path, capsys, monkeypatch
):
    secret = "sf4-not-a-real-key"
    token = "sf4-github-token-not-shipped"
    monkeypatch.setenv("VAST_API_KEY", secret)
    monkeypatch.setenv("GITHUB_TOKEN", token)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)

    def request(method, url, api_key, body=None, timeout=60):
        assert api_key == secret
        assert secret not in url
        assert token not in url
        return 200, {"instances": _texas("running")}

    def write_passing_run(run):
        _write_run(run, decision="accept", use_gpu=True)
        (run / "plan.json").write_bytes(b'{"scene_name":"Fruits","from":"pipeline"}\n')

    remote = _Remote(write_passing_run)
    sha = "a" * 40
    code = run_smoke(
        ALLOWED_INSTANCE_ID,
        _checkout(tmp_path),
        tmp_path / "artifacts",
        secret,
        request=request,
        remote_factory=lambda _row: remote,
        sleep=lambda _seconds: None,
        trigger="label",
        actor="cursor[bot]",
        commit_sha=sha,
    )
    assert code == 0
    text = (tmp_path / "artifacts" / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(text)
    assert manifest["trigger"] == "label"
    assert manifest["actor"] == "cursor[bot]"
    assert manifest["commit_sha"] == sha
    assert manifest["secrets_forwarded"] is False
    assert manifest["secret_forwarding"] == {
        "secrets_forwarded": False,
        "ssh_env_forwarded": False,
        "git_credentials_shipped": False,
        "payload_secret_hits": 0,
    }
    assert secret not in text
    assert token not in text
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    assert token not in captured.out + captured.err
    assert "cursor[bot]" in captured.out


def test_failure_manifest_records_trigger_and_no_secret(tmp_path, monkeypatch):
    secret = "sf4-not-a-real-key"
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)

    def request(method, url, api_key, body=None, timeout=60):
        assert method == "GET"
        return 200, {"instances": _texas("offline")}

    code = run_smoke(
        ALLOWED_INSTANCE_ID,
        _checkout(tmp_path),
        tmp_path / "artifacts",
        secret,
        request=request,
        sleep=lambda _seconds: None,
        trigger="dispatch",
        actor="sandy-t-shelvan",
        commit_sha="b" * 40,
    )
    assert code == 2
    artifact = tmp_path / "artifacts"
    assert [path.name for path in artifact.iterdir()] == ["manifest.json"]
    text = (artifact / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(text)
    assert manifest["trigger"] == "dispatch"
    assert manifest["actor"] == "sandy-t-shelvan"
    assert manifest["commit_sha"] == "b" * 40
    assert manifest["secrets_forwarded"] is False
    assert manifest["passed"] is False
    assert secret not in text


def test_manifest_refuses_a_record_that_forwarded_a_secret(tmp_path):
    run = tmp_path / "run"
    _write_run(run, decision="accept", use_gpu=True)
    (run / "plan.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(SmokeError, match="forwarded a secret"):
        collect_artifact_manifest(
            run,
            command=_command(),
            instance_id=ALLOWED_INSTANCE_ID,
            secret_forwarding={
                "secrets_forwarded": True,
                "ssh_env_forwarded": True,
                "git_credentials_shipped": False,
                "payload_secret_hits": 1,
            },
        )
    assert not (run / "manifest.json").exists()


def test_run_context_cannot_change_the_instance_id():
    context = normalize_run_context("request-file", "sandy-t-shelvan", "c" * 40)
    assert context == {
        "trigger": "request-file",
        "actor": "sandy-t-shelvan",
        "commit_sha": "c" * 40,
    }
    assert "instance" not in context
    for trigger, actor, sha in (
        ("rent", "sandy-t-shelvan", "c" * 40),
        ("label", "bad actor", "c" * 40),
        ("label", "cursor[bot]", "not-a-sha"),
        ("label", "a" * 80, "c" * 40),
    ):
        with pytest.raises(InstanceNotApproved):
            normalize_run_context(trigger, actor, sha)


def test_remote_exit_code_propagates(tmp_path):
    def request(method, url, api_key, body=None, timeout=60):
        assert method == "GET"
        return 200, {"instances": _texas("running")}

    def write(run):
        _write_run(run, decision="accept", use_gpu=True)
        (run / "plan.json").write_bytes(b'{"scene_name":"Fruits","from":"pipeline"}\n')

    for rc in (2, 139):
        code = run_smoke(
            ALLOWED_INSTANCE_ID,
            _checkout(tmp_path),
            tmp_path / f"rc-{rc}",
            "test-key",
            request=request,
            remote_factory=lambda _row, exit_code=rc: _Remote(write, rc=exit_code),
            sleep=lambda _seconds: None,
        )
        assert code == rc


_FAKE_PRIVATE = (
    "-----BEGIN OPENSSH PRIVATE KEY-----\n"
    "bm90LWEtcmVhbC1rZXk=\n"
    "-----END OPENSSH PRIVATE KEY-----\n"
)


def _host_pin(directory, host="47.186.21.5", port=51823):
    path = Path(directory) / "known_host"
    path.write_text(
        f"[{host}]:{port} ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITestKeyForSf4SmokeOnly\n",
        encoding="utf-8",
    )
    return path


def test_ssh_endpoint_uses_the_direct_port_and_ignores_the_proxy():
    row = _texas(
        "running",
        ssh_host="ssh5.vast.ai",
        ssh_port=16868,
    )
    assert ssh_endpoint(row) == ("47.186.21.5", 51823)
    pin_line = [
        line.strip()
        for line in KNOWN_HOST_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert pin_line == [
        "[47.186.21.5]:51823 ssh-ed25519 "
        "AAAAC3NzaC1lZDI1NTE5AAAAIFqxlHMUC7xRRWryJ7y1/3LjWJiHFLhPlb9bH05FR3Nr"
    ]
    assert (
        load_known_host_line(KNOWN_HOST_PATH, "47.186.21.5", 51823).strip()
        == (pin_line[0])
    )
    with pytest.raises(InstanceNotApproved, match="does not match"):
        load_known_host_line(KNOWN_HOST_PATH, "ssh5.vast.ai", 16868)
    numeric = _texas("running")
    numeric["ports"] = {"22/tcp": [{"HostPort": 51823}]}
    assert ssh_endpoint(numeric) == ("47.186.21.5", 51823)


def test_ssh_endpoint_refuses_a_missing_or_invalid_direct_endpoint():
    proxy = _texas("running", ssh_host="ssh5.vast.ai", ssh_port=16868)
    missing_ip = dict(proxy)
    missing_ip.pop("public_ipaddr")
    with pytest.raises(SmokeError, match="public_ipaddr is missing"):
        ssh_endpoint(missing_ip)
    bad_ip = dict(proxy)
    bad_ip["public_ipaddr"] = "not a host"
    with pytest.raises(SmokeError, match="public_ipaddr is missing"):
        ssh_endpoint(bad_ip)
    for broken in (None, {}, {"80/tcp": [{"HostPort": "80"}]}):
        row = dict(proxy)
        row["ports"] = broken
        with pytest.raises(SmokeError, match="22/tcp is missing"):
            ssh_endpoint(row)
    for host_port in (None, "", "abc", "0", 0, True, -1, 70000):
        row = dict(proxy)
        row["ports"] = {"22/tcp": [{"HostPort": host_port}]}
        with pytest.raises(SmokeError, match="22/tcp is missing|HostPort is invalid"):
            ssh_endpoint(row)


def test_running_show_without_a_direct_endpoint_does_not_use_the_proxy(
    tmp_path, capsys
):
    def request(method, url, api_key, body=None, timeout=60):
        assert method == "GET"
        row = _texas("running", ssh_host="ssh5.vast.ai", ssh_port=16868)
        row.pop("ports")
        return 200, {"instances": row}

    def factory(_row):
        raise AssertionError("proxy connection")

    code = run_smoke(
        ALLOWED_INSTANCE_ID,
        _checkout(tmp_path),
        tmp_path / "out",
        "test-key",
        request=request,
        remote_factory=factory,
        sleep=lambda _seconds: None,
    )
    err = capsys.readouterr().err
    assert code == 2
    assert "22/tcp is missing" in err
    assert "ssh5.vast.ai" not in err
    assert "16868" not in err


def test_fixed_key_is_mode_0600_and_removed_on_close(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sceneagent.vast_smoke.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ssh")),
    )
    pin = _host_pin(tmp_path)
    remote = SshRemote(
        _texas("running"),
        "test-key",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("api")),
        private_key=_FAKE_PRIVATE,
        known_host_file=pin,
        sleep=lambda _s: None,
    )
    key_path = remote.private
    assert key_path.stat().st_mode & 0o777 == 0o600
    assert "BEGIN OPENSSH PRIVATE KEY" not in remote.known_hosts
    assert "StrictHostKeyChecking=yes" in ssh_base_argv(
        str(key_path), remote.known_hosts, remote.port
    )
    text = key_path.read_text(encoding="utf-8")
    assert (
        text == _FAKE_PRIVATE if _FAKE_PRIVATE.endswith("\n") else _FAKE_PRIVATE + "\n"
    )
    remote.close()
    assert not key_path.exists()


def test_missing_secret_or_host_pin_fails_before_ssh(tmp_path, capsys, monkeypatch):
    def explode(*_args, **_kwargs):
        raise AssertionError("ssh")

    monkeypatch.setattr("sceneagent.vast_smoke.subprocess.run", explode)

    def request(method, url, api_key, body=None, timeout=60):
        assert method == "GET"
        assert "/ssh/" not in url
        return 200, {"instances": _texas("running")}

    code = run_smoke(
        ALLOWED_INSTANCE_ID,
        _checkout(tmp_path),
        tmp_path / "out",
        "test-key",
        request=request,
        sleep=lambda _seconds: None,
        ssh_private_key="",
        known_host_file=_host_pin(tmp_path),
    )
    err = capsys.readouterr().err
    assert code == 2
    assert "VAST_SSH_PRIVATE_KEY is not set." in err
    assert "bm90" not in err
    code = run_smoke(
        ALLOWED_INSTANCE_ID,
        _checkout(tmp_path),
        tmp_path / "out2",
        "test-key",
        request=request,
        sleep=lambda _seconds: None,
        ssh_private_key=_FAKE_PRIVATE,
        known_host_file=tmp_path / "missing-pin",
    )
    err = capsys.readouterr().err
    assert code == 2
    assert "known host pin is not configured" in err
    assert _FAKE_PRIVATE.strip() not in err


def test_sigterm_raises_systemexit_and_finally_closes(monkeypatch, tmp_path):
    installed = {}

    def capture(signum, handler):
        installed[signum] = handler

    monkeypatch.setattr("sceneagent.vast_smoke.signal.signal", capture)
    calls = []

    def request(method, url, api_key, body=None, timeout=60):
        calls.append((method, None if body is None else dict(body)))
        assert method == "GET"
        return 200, {"instances": _texas("running")}

    class _Closing:
        def __init__(self):
            self.closed = False

        def run(self, checkout, local_run):
            installed[signal.SIGTERM](signal.SIGTERM, None)

        def close(self):
            self.closed = True

    remote = _Closing()
    with pytest.raises(SystemExit) as caught:
        run_smoke(
            ALLOWED_INSTANCE_ID,
            _checkout(tmp_path),
            tmp_path / "out",
            "test-key",
            request=request,
            remote_factory=lambda _row: remote,
            sleep=lambda _seconds: None,
            handle_sigterm=True,
        )
    assert caught.value.code == 1
    assert remote.closed is True
    assert calls == [("GET", None)]


def test_probe_is_plain_shell_and_never_prints_the_environment():
    script = build_remote_shell(REMOTE_REPO, REMOTE_OUT, pipeline="probe")
    # It reports whether the pinned env exists; it never requires or runs it.
    assert "pinned interpreter missing" not in script
    assert "exit 127" not in script
    assert f"{REMOTE_PYTHON} -m" not in script
    assert f"{REMOTE_OUT}/inventory.txt" in script
    for banned in ("printenv", "\nenv", " env ", "set\n", "export -p"):
        assert banned not in script


def test_probe_runs_here_and_its_manifest_passes_on_the_inventory(tmp_path):
    import subprocess

    out = tmp_path / "run"
    script = build_remote_shell(REMOTE_REPO, str(out), pipeline="probe")
    done = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr
    text = (out / "inventory.txt").read_text()
    for section in (
        "## gpu",
        "## disk",
        "## conda_envs",
        "## tools",
        "## interpreters",
    ):
        assert section in text
    command = build_remote_command(REMOTE_REPO, str(out), pipeline="probe")
    ok = collect_artifact_manifest(
        out, command=command, instance_id=59601312, pipeline="probe"
    )
    assert ok["passed"] is True
    (out / "inventory.txt").unlink()
    missing = collect_artifact_manifest(
        out, command=command, instance_id=59601312, pipeline="probe"
    )
    assert missing["passed"] is False
    assert missing["failure_reason"] == "inventory.txt was not written"


def test_bin_shell_puts_cuda_colmap_on_path_and_py7zr_outside_the_shared_env():
    from sceneagent.vast_smoke import PY7ZR_PIN, REMOTE_COLMAP_BIN, REMOTE_PYDEPS

    script = build_remote_shell("/workspace/r", "/workspace/o", pipeline="bin")
    bindir = REMOTE_PYTHON.rsplit("/", 1)[0]
    assert f'export PATH={bindir}:{REMOTE_COLMAP_BIN}:"$PATH"' in script
    assert f"export PYTHONPATH=/workspace/r:{REMOTE_PYDEPS}" in script
    assert f"pip install --quiet --target {REMOTE_PYDEPS} {PY7ZR_PIN}" in script
    # The path line comes before anything that resolves colmap.
    assert script.index("export PATH=") < script.index("sceneagent.bin_run")


def test_splat_pipeline_runs_bin_splat_and_needs_three_ledger_items():
    from sceneagent.vast_smoke import PIPELINES, build_remote_command

    argv = build_remote_command("/workspace/r", "/workspace/o", pipeline="splat")
    assert argv == [
        REMOTE_PYTHON,
        "-m",
        "sceneagent.bin_splat",
        "--out",
        "/workspace/o",
    ]
    assert PIPELINES["splat"][0][-1] == "full-scene splat"
    script = build_remote_shell("/workspace/r", "/workspace/o", pipeline="splat")
    assert "py7zr" in script  # poses may need the dataset fetch


def test_masks_pipeline_runs_bin_masks():
    from sceneagent.vast_smoke import FETCH_NAMES, PIPELINES, build_remote_command

    argv = build_remote_command("/workspace/r", "/workspace/o", pipeline="masks")
    assert argv == [
        REMOTE_PYTHON,
        "-m",
        "sceneagent.bin_masks",
        "--out",
        "/workspace/o",
    ]
    assert PIPELINES["masks"][0] == ("benchmark scene", "SAM masks")
    assert "masks_evidence.json" in FETCH_NAMES


def test_association_pipeline_runs_bin_association_on_the_bin_dir():
    from sceneagent.vast_smoke import (
        BIN_PIPELINES,
        FETCH_NAMES,
        PIPELINES,
        REMOTE_BIN_OUT,
        build_remote_command,
        remote_out,
    )

    argv = build_remote_command("/workspace/r", "/workspace/o", pipeline="association")
    assert argv == [
        REMOTE_PYTHON,
        "-m",
        "sceneagent.bin_association",
        "--out",
        "/workspace/o",
    ]
    assert "association" in BIN_PIPELINES
    assert remote_out("association") == REMOTE_BIN_OUT
    assert PIPELINES["association"][0][-1] == "Gaussian association"
    assert "association_evidence.json" in FETCH_NAMES


def test_probe_listings_hide_dotfiles_and_credential_names_and_print_no_file_contents(
    tmp_path,
):
    import subprocess

    from sceneagent.vast_smoke import PROBE_SCRIPT

    # Security review on PR #37: the inventory is a public artifact.
    assert "ls -la" not in PROBE_SCRIPT
    assert "cat /root" not in PROBE_SCRIPT and "tail -15 /root" not in PROBE_SCRIPT
    public = next(
        line for line in PROBE_SCRIPT.splitlines() if line.startswith("public()")
    )
    names = (
        "gemini.key\nmodels\nauthorized_keys\napi_token.txt\nid_ed25519\nsam\n.env\n"
    )
    done = subprocess.run(
        ["bash", "-c", f"{public}\npublic"],
        input=names,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.stdout.split() == ["models", "sam"]
    # No folder trees at all: run 42713577064 listed a file named after a
    # local user path, which no name filter would catch.
    for banned in (
        "section folders",
        "section tree",
        "section sizes",
        "section workspace_tree",
        "find /root",
    ):
        assert banned not in PROBE_SCRIPT, banned


def test_only_dataset_pipelines_see_the_py7zr_install_dir():
    from sceneagent.vast_smoke import REMOTE_PYDEPS

    run4 = build_remote_shell("/workspace/r", "/workspace/o", pipeline="run4")
    assert "export PYTHONPATH=/workspace/r\n" in run4
    assert REMOTE_PYDEPS not in run4
    for pipeline in sorted(BIN_PIPELINES):
        script = build_remote_shell("/workspace/r", "/workspace/o", pipeline=pipeline)
        assert f"export PYTHONPATH=/workspace/r:{REMOTE_PYDEPS}" in script


def test_non_probe_runs_remove_a_stale_inventory_before_running():
    for pipeline in ("run4", "bin"):
        script = build_remote_shell("/workspace/r", "/workspace/o", pipeline=pipeline)
        assert "rm -f /workspace/o/inventory.txt" in script
    probe = build_remote_shell("/workspace/r", "/workspace/o", pipeline="probe")
    assert "rm -f" not in probe
