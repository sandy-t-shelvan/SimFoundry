# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the Vast Fruits smoke. No network and no API key."""

import base64
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from sceneagent.vast_smoke import (
    ALLOWED_INSTANCE_ID,
    DETACH_LEFTOVER_KEYS,
    FRUITS_RELATIVE,
    REMOTE_OUT,
    REMOTE_REPO,
    SSH_READY_MAX_ATTEMPTS,
    SSH_READY_TIMEOUT_S,
    InstanceNotApproved,
    SmokeError,
    SshRemote,
    arm_own_key_delete,
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
    run_smoke,
    ssh_base_argv,
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


def test_parse_instance_id_accepts_only_the_texas_instance():
    assert parse_instance_id(58301075) == 58301075
    assert parse_instance_id("58301075") == 58301075
    assert parse_instance_id(" 58301075 ") == 58301075
    for value in (
        58357498,
        "58301075.0",
        "",
        None,
        True,
        False,
        58301075.0,
        "58301075abc",
    ):
        with pytest.raises(InstanceNotApproved):
            parse_instance_id(value)


def test_endpoint_allowlist_is_show_start_stop_and_ssh_on_one_id():
    method, url, body = endpoint_for("show", "58301075")
    assert method == "GET"
    assert url == "https://console.vast.ai/api/v0/instances/58301075/?owner=me"
    assert body is None
    assert endpoint_for("start", 58301075)[2] == {"state": "running"}
    assert endpoint_for("stop", 58301075)[2] == {"state": "stopped"}
    attached = endpoint_for("attach_ssh", 58301075, public_key=PUBLIC_KEY)
    assert attached[0] == "POST"
    assert attached[1].endswith("/58301075/ssh/")
    assert attached[2] == {"ssh_key": PUBLIC_KEY}
    detached = endpoint_for("detach_ssh", 58301075, ssh_key_id="42")
    assert detached[0] == "DELETE"
    assert detached[1].endswith("/58301075/ssh/42/")
    for kind in ("create", "rent", "resize", "destroy", "delete", "reboot", "label"):
        with pytest.raises(InstanceNotApproved):
            endpoint_for(kind, 58301075)
    with pytest.raises(InstanceNotApproved):
        endpoint_for("show", 58357498)
    with pytest.raises(InstanceNotApproved):
        endpoint_for(
            "attach_ssh",
            58301075,
            public_key="-----BEGIN OPENSSH PRIVATE KEY-----\nsecret",
        )
    with pytest.raises(InstanceNotApproved):
        assert_request_allowed(
            "PUT",
            "https://console.vast.ai/api/v0/instances/58301075/",
            {"state": "running", "disk": 500},
        )
    with pytest.raises(InstanceNotApproved):
        assert_request_allowed(
            "DELETE", "https://console.vast.ai/api/v0/instances/58301075/", None
        )
    with pytest.raises(InstanceNotApproved):
        assert_request_allowed(
            "POST", "https://console.vast.ai/api/v0/asks/1/", {"image": "x"}
        )
    with pytest.raises(InstanceNotApproved):
        assert_request_allowed(
            "GET",
            "https://console.vast.ai/api/v0/instances/58301075/?owner=me&api_key=secret",
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
    assert row["id"] == 58301075
    with pytest.raises(InstanceNotApproved, match="missing"):
        require_approved_instance({"instances": None})
    with pytest.raises(InstanceNotApproved, match="not Texas"):
        require_approved_instance(
            {"instances": _texas("running", geolocation="Estonia, EE")}
        )
    with pytest.raises(InstanceNotApproved, match="returned id"):
        require_approved_instance({"instances": _texas("running", id=58357498)})


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
    env = child_env()
    assert "VAST_API_KEY" not in env
    assert "super-secret" not in " ".join(env.values())


def test_remote_command_is_the_fruits_entry_point_and_carries_no_secret():
    argv = _command()
    assert argv[:3] == ["python3", "-m", "sceneagent"]
    assert argv[3].endswith(FRUITS_RELATIVE)
    assert argv[4:] == ["--out", REMOTE_OUT]
    script = build_remote_shell(REMOTE_REPO, REMOTE_OUT)
    assert "Fruits.mp4" in script
    assert "PYTHONPATH" in script
    assert "host_probe.json" in script
    blob = script + "\n" + build_fetch_script(REMOTE_OUT) + "\n" + " ".join(argv)
    lowered = blob.lower()
    for banned in ("vast_api_key", "bearer", "api_key", "gemini", "meshy"):
        assert banned not in lowered
    assert REMOTE_REPO.startswith("/workspace/sf4-smoke/")
    assert "SimFoundry" not in REMOTE_REPO
    with pytest.raises(InstanceNotApproved):
        build_remote_command("/tmp/../etc", REMOTE_OUT)


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
    assert "VAST_API_KEY" not in blob
    assert "Bearer" not in blob
    assert FRUITS_RELATIVE in blob
    assert "sceneagent" in blob


def test_ssh_key_id_comes_from_the_attach_response_only():
    assert ssh_key_id_from_attach({"ssh_key_id": 17}) == "17"
    assert ssh_key_id_from_attach({"key": {"key_id": "abc"}}) == "abc"
    assert ssh_key_id_from_attach({"id": 17, "success": True}) is None


def test_deferred_ledger_is_not_a_pass_and_is_not_rewritten(tmp_path):
    run = tmp_path / "run"
    _write_run(run, decision="deferred")
    before = (run / "ledger.json").read_bytes()
    manifest = collect_artifact_manifest(run, command=_command(), instance_id=58301075)
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
    manifest = collect_artifact_manifest(run, command=_command(), instance_id=58301075)
    assert (run / "ledger.json").read_bytes() == before
    assert manifest["passed"] is False
    assert manifest["stages"][1]["reported"] == "failed"
    assert manifest["stages"][1]["decision"] == "reject"


def test_cpu_accept_is_not_a_gpu_pass(tmp_path):
    run = tmp_path / "run"
    _write_run(run, decision="accept", use_gpu=False)
    manifest = collect_artifact_manifest(run, command=_command(), instance_id=58301075)
    assert manifest["passed"] is False
    assert manifest["stages"][1]["reported"] == "accept_without_gpu"


def test_collector_copies_pipeline_plan_unchanged_and_fails_when_missing(tmp_path):
    run = tmp_path / "run"
    _write_run(run, decision="accept", use_gpu=True)
    original = b'{"scene_name":"Fruits","from":"pipeline"}\n'
    (run / "plan.json").write_bytes(original)
    manifest = collect_artifact_manifest(run, command=_command(), instance_id=58301075)
    assert manifest["passed"] is True
    assert (run / "plan.json").read_bytes() == original
    assert not (run / "plan.derived.json").exists()
    artifact_dir = tmp_path / "artifacts"
    copied = publish_artifacts(run, artifact_dir)
    assert "plan.json" in copied
    assert (artifact_dir / "plan.json").read_bytes() == original

    bare = tmp_path / "bare"
    _write_run(bare, decision="accept", use_gpu=True)
    missing = collect_artifact_manifest(bare, command=_command(), instance_id=58301075)
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
        run, command=_command(), instance_id="58301075"
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
    manifest = collect_artifact_manifest(run, command=_command(), instance_id=58301075)
    assert manifest["passed"] is False
    assert manifest["host_probe"]["colmap_cuda"] is True


def test_missing_ledger_is_not_invented(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    manifest = collect_artifact_manifest(run, command=_command(), instance_id=58301075)
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
    manifest = collect_artifact_manifest(run, command=_command(), instance_id=58301075)
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
        58301075,
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


def test_run_smoke_starts_and_stops_a_stopped_instance_when_the_remote_fails(tmp_path):
    calls = []
    phase = {"started": False}

    def request(method, url, api_key, body=None, timeout=60):
        calls.append((method, None if body is None else dict(body)))
        assert "test-key" not in url
        if method == "GET":
            status = "running" if phase["started"] else "exited"
            return 200, {"instances": _texas(status)}
        if method == "PUT" and body == {"state": "running"}:
            phase["started"] = True
            return 200, {"success": True}
        if method == "PUT" and body == {"state": "stopped"}:
            phase["started"] = False
            return 200, {"success": True}
        raise AssertionError((method, body))

    remote = _Remote(lambda run: _write_run(run, decision="deferred"), rc=0)
    code = run_smoke(
        "58301075",
        _checkout(tmp_path),
        tmp_path / "artifacts",
        "test-key",
        request=request,
        remote_factory=lambda _row: remote,
        sleep=lambda _seconds: None,
    )
    assert code == 1
    assert calls[0][0] == "GET"
    assert calls[1] == ("PUT", {"state": "running"})
    assert calls[-1] == ("PUT", {"state": "stopped"})
    ledger = json.loads(
        (tmp_path / "artifacts" / "ledger.json").read_text(encoding="utf-8")
    )
    assert ledger["entries"][1]["decision"] == "deferred"
    manifest = json.loads(
        (tmp_path / "artifacts" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["passed"] is False


def test_run_smoke_refuses_the_wrong_instance_before_the_network(tmp_path):
    def boom(*_args, **_kwargs):
        raise AssertionError("network")

    code = run_smoke(58357498, tmp_path, tmp_path / "out", "test-key", request=boom)
    assert code == 2


def test_run_smoke_refuses_a_missing_key_and_a_non_texas_host(tmp_path, capsys):
    code = run_smoke(58301075, _checkout(tmp_path), tmp_path / "out", "")
    assert code == 2
    assert "not set" in capsys.readouterr().err

    def request(method, url, api_key, body=None, timeout=60):
        assert method == "GET"
        return 200, {"instances": _texas("running", geolocation="Quebec, CA")}

    code = run_smoke(
        58301075,
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
        58301075,
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
    for other in (58357498, "1", "58301076"):
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
            "https://console.vast.ai/api/v0/instances/58357498/?owner=me",
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
    assert "--instance-id 58301075" in text
    assert '"$GITHUB_WORKSPACE"' in text


def test_workflow_has_no_inputs_and_only_workflow_dispatch():
    text = WORKFLOW.read_text(encoding="utf-8")
    lines = text.splitlines()
    start = next(index for index, line in enumerate(lines) if line == "on:")
    block = []
    for line in lines[start + 1 :]:
        if line and not line.startswith(" ") and not line.startswith("#"):
            break
        block.append(line)
    trigger = "\n".join(block)
    assert trigger.strip() == "workflow_dispatch:"
    assert "inputs:" not in text
    assert "inputs." not in text
    assert "--instance-id 58301075" in text
    for line in lines:
        stripped = line.strip()
        assert stripped not in {
            "push:",
            "pull_request:",
            "pull_request_target:",
            "schedule:",
        }


def test_workflow_is_manual_dispatch_only():
    text = WORKFLOW.read_text(encoding="utf-8")
    lines = text.splitlines()
    start = next(index for index, line in enumerate(lines) if line == "on:")
    block = []
    for line in lines[start + 1 :]:
        if line and not line.startswith(" ") and not line.startswith("#"):
            break
        block.append(line)
    trigger = "\n".join(block)
    assert "workflow_dispatch:" in trigger
    assert "push" not in trigger
    assert "pull_request" not in trigger
    for line in lines:
        stripped = line.strip()
        assert stripped not in {
            "push:",
            "pull_request:",
            "pull_request_target:",
            "schedule:",
        }
        assert "continue-on-error" not in line
        if "runs-on" in line:
            assert "ubuntu-latest" in line
        if "echo" in line:
            assert "VAST_API_KEY" not in line
    assert text.count("secrets.VAST_API_KEY") == 1
    assert "58301075" in text
    assert "sceneagent-fruits-smoke" in text
    assert "plan.json" in text
    assert "ledger.json" in text
    assert "manifest.json" in text
    assert "/api/v0/asks/" not in text


def test_no_pipeline_plan_records_dc2_missing_and_writes_no_plan_file(tmp_path):
    run = tmp_path / "run"
    _write_run(run, decision="deferred")
    before = (run / "ledger.json").read_bytes()
    manifest = collect_artifact_manifest(run, command=_command(), instance_id=58301075)
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
    collect_artifact_manifest(run, command=_command(), instance_id=58301075)
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

    with pytest.raises(SmokeError, match="did not accept the ephemeral key"):
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
    step = "\n".join(lines[start:])
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
        assert url == "https://console.vast.ai/api/v0/instances/58301075/ssh/"
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
            endpoint_for(kind, 58301075)
    with pytest.raises(InstanceNotApproved):
        assert_request_allowed(
            "DELETE",
            "https://console.vast.ai/api/v0/instances/58301075/",
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
        assert url.endswith("/58301075/ssh/")
        assert body is None
        return 200, {"success": True, "ssh_keys": json.dumps([_sample_ssh_key()])}

    summary = list_attached_ssh_keys(
        list_request, "test-key", sleep=lambda _seconds: None
    )
    assert summary == {"count": 1, "fingerprints": [_fingerprint(PUBLIC_KEY)]}
    recorded = json.dumps(summary)
    assert "77" not in recorded
    assert "sf4-smoke" not in recorded


PINNED_SSH_LIST = "https://console.vast.ai/api/v0/instances/58301075/ssh/"


def test_pinned_ssh_list_get_is_allowed():
    method, url, body = endpoint_for("show_ssh_keys", 58301075)
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
        ("GET", "https://console.vast.ai/api/v0/instances/58301075/ssh", None),
        ("GET", "https://console.vast.ai/api/v0/instances/58301075/ssh/1/", None),
        ("GET", "https://console.vast.ai/api/v0/instances/58301075/ssh/1/extra/", None),
        (
            "POST",
            "https://console.vast.ai/api/v0/instances/58301075/ssh/1/",
            {"ssh_key": PUBLIC_KEY},
        ),
        ("PUT", "https://console.vast.ai/api/v0/instances/58301075/ssh/1/", None),
        (
            "DELETE",
            "https://console.vast.ai/api/v0/instances/58301075/ssh/1/extra/",
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
    with pytest.raises(SmokeError, match="did not accept the ephemeral key"):
        remote._sync(_checkout(tmp_path))
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
            "https://console.vast.ai/api/v0/instances/58301075/",
            {"state": "paused"},
        )
    method, url, body = endpoint_for("start", 58301075)
    assert method == "PUT"
    assert url == "https://console.vast.ai/api/v0/instances/58301075/"
    assert body == {"state": "running"}


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
        ("GET", "https://console.vast.ai/api/v0/instances/58301075/ssh/77/", None),
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


def test_own_key_is_detached_on_success_and_on_failure(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    for fail, expected in ((False, 0), (True, 1)):
        calls, request = _listing_transport("one")
        _install_remote_run(monkeypatch, fail)
        artifacts = tmp_path / ("fail" if fail else "ok")
        code = run_smoke(
            ALLOWED_INSTANCE_ID,
            _checkout(tmp_path),
            artifacts,
            "test-key",
            request=request,
            sleep=lambda _seconds: None,
        )
        assert code == expected
        deletes = [url for method, url in calls if method == "DELETE"]
        assert deletes == ["https://console.vast.ai/api/v0/instances/58301075/ssh/77/"]
        manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["own_key"]["match_count"] == 1
        assert manifest["own_key"]["detached"] is True
        assert "exactly one" in manifest["own_key"]["reason"]
        assert "ssh-ed25519" not in json.dumps(manifest["own_key"])


def test_no_delete_when_zero_or_several_keys_match(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    _install_remote_run(monkeypatch, fail=False)
    expectations = {
        "zero": (0, "no listed key"),
        "several": (2, "more than one"),
    }
    for mode, (count, reason) in expectations.items():
        calls, request = _listing_transport(mode)
        artifacts = tmp_path / mode
        code = run_smoke(
            ALLOWED_INSTANCE_ID,
            _checkout(tmp_path),
            artifacts,
            "test-key",
            request=request,
            sleep=lambda _seconds: None,
        )
        assert code == 0
        assert [url for method, url in calls if method == "DELETE"] == []
        manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["own_key"]["match_count"] == count
        assert manifest["own_key"]["detached"] is False
        assert reason in manifest["own_key"]["reason"]
        assert "ssh-ed25519" not in json.dumps(manifest["own_key"])


def _assert_no_match(tmp_path, monkeypatch, mode):
    calls, request = _listing_transport(mode)
    _install_remote_run(monkeypatch, fail=False)
    artifacts = tmp_path / mode
    code = run_smoke(
        ALLOWED_INSTANCE_ID,
        _checkout(tmp_path),
        artifacts,
        "test-key",
        request=request,
        sleep=lambda _seconds: None,
    )
    manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
    assert code == 0
    assert [url for method, url in calls if method == "DELETE"] == []
    assert manifest["own_key"]["match_count"] == 0
    assert manifest["own_key"]["detached"] is False
    assert "ssh-ed25519" not in json.dumps(manifest["own_key"])


def test_near_miss_listed_keys_do_not_match(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    for mode in ("substring", "reverse", "case"):
        _assert_no_match(tmp_path, monkeypatch, mode)


def test_own_key_match_rstrip_boundary(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    calls, request = _listing_transport("trailing")
    _install_remote_run(monkeypatch, fail=False)
    artifacts = tmp_path / "trailing"
    code = run_smoke(
        ALLOWED_INSTANCE_ID,
        _checkout(tmp_path),
        artifacts,
        "test-key",
        request=request,
        sleep=lambda _seconds: None,
    )
    manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
    assert code == 0
    assert [url for method, url in calls if method == "DELETE"] == [
        "https://console.vast.ai/api/v0/instances/58301075/ssh/77/"
    ]
    assert manifest["own_key"]["match_count"] == 1
    assert manifest["own_key"]["detached"] is True
    _assert_no_match(tmp_path, monkeypatch, "leading")


def test_detach_guard_refuses_any_other_key_id(monkeypatch):
    monkeypatch.setattr("sceneagent.vast_smoke.DETACH_LEFTOVER_KEYS", True)
    calls = []

    def request(method, url, api_key, body=None, timeout=60):
        calls.append((method, url))
        return 200, {"success": True}

    own = "https://console.vast.ai/api/v0/instances/58301075/ssh/77/"
    other = "https://console.vast.ai/api/v0/instances/58301075/ssh/78/"
    elsewhere = "https://console.vast.ai/api/v0/instances/1/ssh/77/"
    arm_own_key_delete(None)
    for url in (own, other, elsewhere):
        with pytest.raises(InstanceNotApproved):
            guard_own_key_delete("DELETE", url)
        with pytest.raises(InstanceNotApproved):
            call_json(request, "DELETE", url, "test-key", None, sleep=lambda _s: None)
    assert calls == []
    arm_own_key_delete("77")
    try:
        with pytest.raises(InstanceNotApproved):
            guard_own_key_delete("DELETE", other)
        with pytest.raises(InstanceNotApproved):
            call_json(request, "DELETE", other, "test-key", None, sleep=lambda _s: None)
        with pytest.raises(InstanceNotApproved):
            guard_own_key_delete("DELETE", elsewhere)
        guard_own_key_delete("DELETE", own)
        call_json(request, "DELETE", own, "test-key", None, sleep=lambda _s: None)
        assert calls == [("DELETE", own)]
    finally:
        arm_own_key_delete(None)


def _assert_detach_agreement(manifest):
    own = manifest["own_key"]
    detach = manifest["ssh"]["detach"]
    if own["detached"]:
        assert own["match_count"] == 1
        assert detach == "detached"
    elif own["match_count"] == 1:
        assert detach == "failed"
    else:
        assert own["detached"] is False
        assert detach == "skipped"
    assert "not detached" not in json.dumps(manifest["ssh"])
    assert "not detached" not in json.dumps(manifest["own_key"])


def test_manifest_ssh_detach_agrees_with_own_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    cases = (
        ("one", False),
        ("one", True),
        ("zero", False),
        ("several", False),
    )
    for mode, fail in cases:
        calls, request = _listing_transport(mode)
        _install_remote_run(monkeypatch, fail)
        artifacts = tmp_path / f"{mode}-{'fail' if fail else 'ok'}"
        run_smoke(
            ALLOWED_INSTANCE_ID,
            _checkout(tmp_path),
            artifacts,
            "test-key",
            request=request,
            sleep=lambda _seconds: None,
        )
        manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
        _assert_detach_agreement(manifest)


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
