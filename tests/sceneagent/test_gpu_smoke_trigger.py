# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for GPU smoke triggers. No Vast call and no network."""

import json
import subprocess
from pathlib import Path

from sceneagent.gpu_smoke_trigger import (
    LABEL_ACTORS,
    REQUEST_FIELDS,
    REQUEST_FILE,
    check_push_diff,
    classify_push_diff,
    _job_if,
    _jobs,
    label_cleanup_runs,
    label_trigger_allowed,
    paths_from_name_status,
    smoke_trigger_mapping,
    validate_request_document,
    validate_request_text,
    workflow_label_actors,
    workflow_safety_findings,
)

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "sceneagent-gpu-smoke.yml"
SCHEMA = ROOT / "ci" / "gpu-run-request.schema.json"
EXAMPLE = ROOT / "ci" / "gpu-run-request.example.json"
THIS_REPO = "sandy-t-shelvan/SimFoundry"


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _init_repo(repo: Path) -> str:
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "sf14@example.com")
    _git(repo, "config", "user.name", "sf14")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-q", "-m", "base")
    return _git(repo, "rev-parse", "HEAD")


def _label(**overrides: object) -> tuple[bool, str]:
    payload = {
        "event_name": "pull_request_target",
        "action": "labeled",
        "label": "run-gpu-smoke",
        "head_repo": THIS_REPO,
        "repository": THIS_REPO,
        "actor": "cursor[bot]",
    }
    payload.update(overrides)
    return label_trigger_allowed(**payload)  # type: ignore[arg-type]


def test_label_refuses_a_base_that_is_not_an_integration_branch():
    allowed, reason = _label(base_ref="feat/unprotected")
    assert allowed is False
    assert "integration branch" in reason
    assert _label(base_ref="main")[0] is True
    assert _label(base_ref="Main_duplicate")[0] is True


def test_fork_label_does_not_run():
    allowed, reason = _label(head_repo="someone/SimFoundry")
    assert allowed is False
    assert "fork" in reason
    allowed, _reason = _label(head_repo="")
    assert allowed is False


def test_label_requires_the_exact_name_and_an_allowlisted_actor():
    assert _label()[0] is True
    assert _label(actor="sandy-t-shelvan")[0] is True
    assert _label(label="run-gpu-smoke-now")[0] is False
    assert _label(actor="someone-else")[0] is False
    assert _label(event_name="pull_request")[0] is False
    allowed, reason = _label()
    assert reason == "gpu-smoke actor=cursor[bot]"


def test_near_miss_request_paths_are_refused():
    near_misses = (
        ".ci/gpu-run-request.json",
        "./ci/gpu-run-request.json",
        "ci//gpu-run-request.json",
        "CI/gpu-run-request.json",
        "ci/GPU-Run-Request.json",
        "ci/gpu-run-request.json ",
        "ci/gpu-run-request.json.bak",
    )
    for path in near_misses:
        ok, message = classify_push_diff([path])
        assert ok is False, path
        assert "refused" in message
        ok, message = classify_push_diff([REQUEST_FILE, path])
        assert ok is False, path
        assert path in message
    ok, _message = classify_push_diff([REQUEST_FILE])
    assert ok is True
    ok, _message = classify_push_diff([REQUEST_FILE, REQUEST_FILE])
    assert ok is True


def test_name_status_rename_and_delete_are_not_the_request_path():
    deleted = "D\tci/gpu-run-request.json"
    renamed = "R100\tci/gpu-run-request.json\tci/moved.json"
    copied = "C100\tother.json\tci/gpu-run-request.json"
    assert paths_from_name_status(["M\tci/gpu-run-request.json"]) == [REQUEST_FILE]
    assert paths_from_name_status(["A\tci/gpu-run-request.json "]) == [
        "ci/gpu-run-request.json "
    ]
    for raw in (deleted, renamed, copied):
        paths = paths_from_name_status([raw])
        ok, message = classify_push_diff(paths)
        assert ok is False
        assert "refused" in message
        assert raw in message


def test_rename_and_delete_of_the_request_file_are_refused(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _init_repo(repo)
    request_dir = repo / "ci"
    request_dir.mkdir()
    request = request_dir / "gpu-run-request.json"
    request.write_text(
        '{"ticket_id":"SF-14","reason":"standing approval"}\n',
        encoding="utf-8",
    )
    _git(repo, "add", "ci/gpu-run-request.json")
    _git(repo, "commit", "-q", "-m", "add request")
    added = _git(repo, "rev-parse", "HEAD")

    _git(repo, "rm", "ci/gpu-run-request.json")
    _git(repo, "commit", "-q", "-m", "delete request")
    deleted = _git(repo, "rev-parse", "HEAD")
    _assert_push_refused(repo, added, deleted, monkeypatch)

    _git(repo, "reset", "--hard", added)
    _git(repo, "mv", "ci/gpu-run-request.json", "ci/moved.json")
    _git(repo, "commit", "-q", "-m", "rename request")
    renamed = _git(repo, "rev-parse", "HEAD")
    _assert_push_refused(repo, added, renamed, monkeypatch)

    _git(repo, "reset", "--hard", added)
    (repo / "other.txt").write_text("other\n", encoding="utf-8")
    _git(repo, "add", "other.txt")
    _git(repo, "commit", "-q", "-m", "other")
    other = _git(repo, "rev-parse", "HEAD")
    _git(repo, "mv", "-f", "other.txt", "ci/gpu-run-request.json")
    _git(repo, "commit", "-q", "-m", "rename onto request")
    onto = _git(repo, "rev-parse", "HEAD")
    _assert_push_refused(repo, other, onto, monkeypatch)


def _assert_push_refused(repo, before, after, monkeypatch):
    output = repo.parent / f"output-{after[:8]}.txt"
    monkeypatch.chdir(repo)
    monkeypatch.setenv("BEFORE", before)
    monkeypatch.setenv("AFTER", after)
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    assert check_push_diff() == 0
    text = output.read_text(encoding="utf-8")
    assert "proceed=false" in text


def test_mixed_push_is_refused():
    ok, message = classify_push_diff(
        ["ci/gpu-run-request.json", "sceneagent/vast_smoke.py"]
    )
    assert ok is False
    assert "refused" in message
    assert "sceneagent/vast_smoke.py" in message
    ok, message = classify_push_diff(["sceneagent/vast_smoke.py"])
    assert ok is False
    ok, message = classify_push_diff([])
    assert ok is False
    ok, message = classify_push_diff(["ci/gpu-run-request.json"])
    assert ok is True


def test_mixed_commit_is_refused(tmp_path, monkeypatch, capsys):
    repo = tmp_path / "repo"
    before = _init_repo(repo)
    request = repo / "ci"
    request.mkdir()
    (request / "gpu-run-request.json").write_text(
        '{"ticket_id":"SF-14","reason":"mixed"}\n',
        encoding="utf-8",
    )
    (repo / "sceneagent_note.txt").write_text("unreviewed\n", encoding="utf-8")
    _git(repo, "add", "ci/gpu-run-request.json", "sceneagent_note.txt")
    _git(repo, "commit", "-q", "-m", "mixed")
    after = _git(repo, "rev-parse", "HEAD")
    output = tmp_path / "output.txt"
    monkeypatch.chdir(repo)
    monkeypatch.setenv("BEFORE", before)
    monkeypatch.setenv("AFTER", after)
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    assert check_push_diff() == 0
    assert "proceed=false" in output.read_text(encoding="utf-8")
    captured = capsys.readouterr()
    assert "refused" in captured.out
    assert "sceneagent_note.txt" in captured.out


def test_request_file_only_commit_is_allowed(tmp_path, monkeypatch, capsys):
    repo = tmp_path / "repo"
    before = _init_repo(repo)
    request = repo / "ci"
    request.mkdir()
    (request / "gpu-run-request.json").write_text(
        '{"ticket_id":"KAN-14","reason":"standing GPU approval for 59601312"}\n',
        encoding="utf-8",
    )
    _git(repo, "add", "ci/gpu-run-request.json")
    _git(repo, "commit", "-q", "-m", "request only")
    after = _git(repo, "rev-parse", "HEAD")
    output = tmp_path / "output.txt"
    monkeypatch.chdir(repo)
    monkeypatch.setenv("BEFORE", before)
    monkeypatch.setenv("AFTER", after)
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    assert check_push_diff() == 0
    assert output.read_text(encoding="utf-8") == "proceed=true\n"
    assert "accepted" in capsys.readouterr().out


def test_request_file_rejects_instance_id_and_invalid_json():
    errors = validate_request_document(
        {"ticket_id": "SF-14", "reason": "because", "instance_id": 59601312}
    )
    assert any("instance_id" in error for error in errors)
    assert validate_request_text("{") == ["request file is not JSON"]
    assert validate_request_document(["SF-14"]) == [
        "request file must be a JSON object"
    ]
    assert validate_request_text('{"ticket_id":"SF-14","reason":"ok"}\n') == []


def test_request_schema_and_example_cannot_name_an_instance():
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert schema["additionalProperties"] is False
    assert tuple(schema["required"]) == REQUEST_FIELDS
    assert tuple(schema["properties"]) == REQUEST_FIELDS
    assert "instance_id" not in schema["properties"]
    example = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    assert validate_request_document(example) == []
    assert not (ROOT / REQUEST_FILE).exists()


def test_zero_sha_push_does_not_proceed(tmp_path, monkeypatch, capsys):
    output = tmp_path / "output.txt"
    monkeypatch.setenv("BEFORE", "0" * 40)
    monkeypatch.setenv("AFTER", "a" * 40)
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    assert check_push_diff() == 0
    assert "proceed=false" in output.read_text(encoding="utf-8")
    assert "no before SHA" in capsys.readouterr().out


def test_smoke_trigger_mapping_is_pinned_to_the_event():
    """workflow_dispatch, the label, and the request file each keep their name.

    Any other event stays invalid. A remap is a safety finding, so pointing
    pull_request_target or push at dispatch does not pass this check.
    """
    text = WORKFLOW.read_text(encoding="utf-8")
    findings = workflow_safety_findings(text)
    assert [item for item in findings if item.startswith("SMOKE_TRIGGER")] == []
    assert smoke_trigger_mapping(text) == {
        "workflow_dispatch": "dispatch",
        "pull_request_target": "label",
        "push": "request-file",
        "fallback": "invalid",
    }
    assert findings == []
    label_as_dispatch = text.replace(
        "github.event_name == 'pull_request_target' && 'label'",
        "github.event_name == 'pull_request_target' && 'dispatch'",
        1,
    )
    push_as_dispatch = text.replace(
        "github.event_name == 'push' && 'request-file'",
        "github.event_name == 'push' && 'dispatch'",
        1,
    )
    label_findings = workflow_safety_findings(label_as_dispatch)
    push_findings = workflow_safety_findings(push_as_dispatch)
    assert label_findings == [
        "SMOKE_TRIGGER maps pull_request_target to 'dispatch', not 'label'"
    ]
    assert push_findings == [
        "SMOKE_TRIGGER maps push to 'dispatch', not 'request-file'"
    ]
    other = text.replace("|| 'invalid'", "|| 'dispatch'", 1)
    assert workflow_safety_findings(other) == [
        "SMOKE_TRIGGER maps any other event to 'dispatch', not 'invalid'"
    ]


def test_cancelled_run_cannot_reach_the_start_put():
    """A queued run that was cancelled must not satisfy the smoke job if.

    Removing ``!cancelled()`` is a killed mutant. An ``always()`` on the
    step that runs ``vast_smoke`` is the same hole.
    """
    text = WORKFLOW.read_text(encoding="utf-8")
    assert [
        item for item in workflow_safety_findings(text) if "!cancelled()" in item
    ] == []
    mutant = text.replace("      !cancelled() &&\n", "", 1)
    killed = [item for item in workflow_safety_findings(mutant) if "cancelled" in item]
    assert killed == [
        (
            "fruits_gpu_smoke job if uses always() without !cancelled() "
            "on the path to vast_smoke"
        )
    ]
    step_mutant = text.replace(
        "      - name: Vast GPU smoke on Fruits.mp4\n",
        "      - name: Vast GPU smoke on Fruits.mp4\n        if: always()\n",
        1,
    )
    step_killed = [
        item
        for item in workflow_safety_findings(step_mutant)
        if "step uses always()" in item
    ]
    assert step_killed == [
        (
            "fruits_gpu_smoke step uses always() without !cancelled() "
            "on the path to vast_smoke"
        )
    ]


def _cleanup_if() -> str:
    text = WORKFLOW.read_text(encoding="utf-8")
    return _job_if(_jobs(text)["label_cleanup"])


def test_label_cleanup_runs_when_the_smoke_job_is_cancelled():
    """Dropping ``!= 'skipped'`` lets cleanup run after a cancel.

    The rest of the original ``if`` stays: ``pull_request_target``,
    ``labeled``, ``run-gpu-smoke``, and the same repository.
    """
    expression = _cleanup_if()
    assert "needs.fruits_gpu_smoke.result != 'skipped'" not in expression
    assert label_cleanup_runs(expression, smoke_result="cancelled")
    assert label_cleanup_runs(expression, smoke_result="skipped")
    assert [
        item
        for item in workflow_safety_findings(WORKFLOW.read_text(encoding="utf-8"))
        if "label cleanup must run" in item
    ] == []


def test_label_cleanup_does_not_run_for_a_fork_pull_request():
    expression = _cleanup_if()
    assert not label_cleanup_runs(
        expression,
        smoke_result="cancelled",
        head_repo="someone/SimFoundry",
        repository=THIS_REPO,
    )


def test_label_cleanup_does_not_run_for_a_different_label():
    expression = _cleanup_if()
    assert not label_cleanup_runs(
        expression,
        smoke_result="cancelled",
        event_label="some-other-label",
    )


def test_workflow_checker_rejects_retired_id_and_lifecycle_puts():
    text = WORKFLOW.read_text(encoding="utf-8")
    retired = text.replace("--instance-id 60254439", "--instance-id 58301075", 1)
    findings = workflow_safety_findings(retired)
    assert "workflow must not name retired instance 58301075" in findings
    assert "instance id must be the literal 60254439" in findings
    started = text.replace(
        "--instance-id 60254439",
        '--instance-id 60254439 \'{"state": "running"}\'',
        1,
    )
    assert "workflow must not send a Vast state running" in workflow_safety_findings(
        started
    )
    stopped = text.replace(
        "--instance-id 60254439",
        '--instance-id 60254439 \'{"state": "stopped"}\'',
        1,
    )
    assert "workflow must not send a Vast state stopped" in workflow_safety_findings(
        stopped
    )


def test_workflow_safety_contract():
    findings = workflow_safety_findings(WORKFLOW.read_text(encoding="utf-8"))
    assert findings == []
    assert (
        tuple(workflow_label_actors(WORKFLOW.read_text(encoding="utf-8")))
        == LABEL_ACTORS
    )


def test_keyed_step_refuses_pythonpath_payload():
    text = WORKFLOW.read_text(encoding="utf-8")
    broken = text.replace(
        "          VAST_API_KEY: ${{ secrets.VAST_API_KEY }}\n",
        "          VAST_API_KEY: ${{ secrets.VAST_API_KEY }}\n"
        "          PYTHONPATH: payload\n",
        1,
    )
    findings = workflow_safety_findings(broken)
    assert any("PYTHONPATH" in item and "payload" in item for item in findings)
    startup = text.replace(
        "          VAST_API_KEY: ${{ secrets.VAST_API_KEY }}\n",
        "          VAST_API_KEY: ${{ secrets.VAST_API_KEY }}\n"
        "          PYTHONSTARTUP: payload/sitecustomize.py\n",
        1,
    )
    startup_findings = workflow_safety_findings(startup)
    assert any("PYTHONSTARTUP" in item for item in startup_findings)
    assert any("sitecustomize" in item for item in startup_findings)
    pth = text.replace(
        "python -m sceneagent.vast_smoke",
        'python -c \'open("payload.pth", "w").write("payload\\n")\' && '
        "python -m sceneagent.vast_smoke",
        1,
    )
    pth_findings = workflow_safety_findings(pth)
    assert any(".pth" in item for item in pth_findings)
    piped = text.replace(
        "python -m sceneagent.vast_smoke",
        "pip install payload && python -m sceneagent.vast_smoke",
        1,
    )
    pip_findings = workflow_safety_findings(piped)
    assert any("pip-installs from payload" in item for item in pip_findings)


def test_keyed_step_uses_base_orchestrator_and_does_not_execute_payload():
    text = WORKFLOW.read_text(encoding="utf-8")
    broken = text.replace(
        "working-directory: orchestrator",
        "working-directory: payload",
        1,
    )
    findings = workflow_safety_findings(broken)
    assert any("payload" in item for item in findings)
    installed = text.replace(
        "python -m sceneagent.vast_smoke",
        "pip install -e payload && python -m sceneagent.vast_smoke",
        1,
    )
    assert any("pip install" in item for item in workflow_safety_findings(installed))


def test_write_token_job_does_not_checkout_head():
    text = WORKFLOW.read_text(encoding="utf-8")
    broken = text.replace(
        "      - name: Comment the run URL and remove run-gpu-smoke\n",
        "      - uses: actions/checkout@v4\n"
        "        with:\n"
        "          path: payload\n"
        "          ref: ${{ github.event.pull_request.head.sha }}\n"
        "      - name: Comment the run URL and remove run-gpu-smoke\n",
        1,
    )
    findings = workflow_safety_findings(broken)
    assert any(
        "cleanup" in item or "checkout" in item or "head" in item for item in findings
    )


def test_label_gate_drops_a_fork_when_the_condition_is_removed():
    text = WORKFLOW.read_text(encoding="utf-8")
    broken = text.replace(
        "github.event.pull_request.head.repo.full_name == github.repository &&\n",
        "",
        1,
    )
    findings = workflow_safety_findings(broken)
    assert any("head.repo" in item for item in findings)
