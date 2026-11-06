# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the Vast Fruits smoke. No network and no API key."""

import json
from pathlib import Path

import pytest

from sceneagent.vast_smoke import (
    ALLOWED_INSTANCE_ID,
    FRUITS_RELATIVE,
    REMOTE_OUT,
    REMOTE_REPO,
    InstanceNotApproved,
    SmokeError,
    assert_request_allowed,
    build_fetch_script,
    build_remote_command,
    build_remote_shell,
    child_env,
    classify_status,
    collect_artifact_manifest,
    endpoint_for,
    publish_artifacts,
    main,
    mask_secret,
    parse_instance_id,
    public_instance_view,
    redact,
    request_json,
    require_approved_instance,
    rsync_invocations,
    run_smoke,
    ssh_base_argv,
    ssh_key_id_from_attach,
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
