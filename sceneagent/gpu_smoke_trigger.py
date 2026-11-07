# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decide which GPU-smoke triggers may run. This module never calls Vast.

The instance id is not a parameter here. ``59601312`` is fixed in the
workflow and in ``sceneagent.vast_smoke``. A request file that names any
other field, including an instance id, is rejected. No trigger starts or
stops that instance.

The label path is ``pull_request_target``. GitHub loads that workflow from
the repository default branch. The request-file path is a plain ``push`` to
``Main_duplicate``, so it uses the workflow on that branch.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess

RUN_GPU_SMOKE_LABEL = "run-gpu-smoke"
REQUEST_FILE = "ci/gpu-run-request.json"
LABEL_ACTORS = ("sandy-t-shelvan", "cursor[bot]")
REQUEST_FIELDS = ("ticket_id", "reason")
_ZERO_SHA = "0" * 40
_SHA_RE = re.compile(r"[0-9a-f]{40}")
_TICKET_RE = re.compile(r"[A-Z][A-Z0-9]+-\d+")
_REASON_RE = re.compile(r"(?=.*\S)[^\n\r]+")
_ACTOR_RE = re.compile(r"github\.actor == '([^']+)'")
_SMOKE_TRIGGER_LINE = re.compile(
    r"(?m)^[ \t]*SMOKE_TRIGGER:[ \t]*\$\{\{[ \t]*(.*?)[ \t]*\}\}[ \t]*$"
)
_EVENT_CLAUSE = re.compile(r"^github\.event_name == '([^']*)' && '([^']*)'$")
_LITERAL_CLAUSE = re.compile(r"^'([^']*)'$")
# Names recorded on the manifest. None of them starts or stops the instance.
# pull_request_target is the label. push is the request file. Anything else
# must stay invalid so a typo cannot be recorded as dispatch.
SMOKE_TRIGGER_MAPPING = {
    "workflow_dispatch": "dispatch",
    "pull_request_target": "label",
    "push": "request-file",
    "fallback": "invalid",
}


def label_trigger_allowed(
    *,
    event_name: str,
    action: str,
    label: str,
    head_repo: str,
    repository: str,
    actor: str,
    base_ref: str = "Main_duplicate",
    allowed_actors: tuple[str, ...] = LABEL_ACTORS,
) -> tuple[bool, str]:
    """Return whether a label event may start a run, and the log line.

    A fork is refused even when the label and the actor would otherwise
    match. ``head_repo`` is ``pull_request.head.repo.full_name``. The
    orchestrator is the PR base SHA, so that base must be ``main`` or
    ``Main_duplicate``. Any other base would run unreviewed Python with
    the Vast secret.
    """
    if event_name != "pull_request_target" or action != "labeled":
        return False, "gpu-smoke label refused: not a pull_request_target labeled event"
    if label != RUN_GPU_SMOKE_LABEL:
        return False, "gpu-smoke label refused: label is not run-gpu-smoke"
    if not head_repo or head_repo != repository:
        return False, "gpu-smoke label refused: fork PRs are not run"
    if base_ref not in {"main", "Main_duplicate"}:
        return False, "gpu-smoke label refused: base is not an integration branch"
    if actor not in allowed_actors:
        return False, "gpu-smoke label refused: actor is not allowlisted"
    return True, f"gpu-smoke actor={actor}"


def classify_push_diff(paths: list[str]) -> tuple[bool, str]:
    """Accept only the raw path set ``{ci/gpu-run-request.json}``.

    Nothing is stripped, case-folded, or slash-folded. ``.ci/gpu-run-request.json``
    and ``./ci/gpu-run-request.json`` are different paths, and a push that
    includes either one does not start the GPU job.
    """
    changed = set(paths)
    if changed == {REQUEST_FILE}:
        return True, f"gpu-smoke request-file accepted: only {REQUEST_FILE} changed"
    shown = ", ".join(paths) if paths else "(no files)"
    return (
        False,
        (
            "gpu-smoke request-file refused: push changes paths other than "
            f"{REQUEST_FILE}: {shown}"
        ),
    )


def paths_from_name_status(lines: list[str]) -> list[str]:
    """Turn ``git diff --name-status`` lines into the paths the guard compares.

    ``A`` and ``M`` contribute the path after the tab, with no stripping.
    A rename, copy, delete, or any other status contributes the raw line.
    That line is not the request path, so the push is refused. A line with
    no tab is already a raw path and is kept unchanged.
    """
    paths: list[str] = []
    for line in lines:
        if "\t" not in line:
            paths.append(line)
            continue
        status, _sep, rest = line.partition("\t")
        if status in {"A", "M"} and "\t" not in rest:
            paths.append(rest)
            continue
        paths.append(line)
    return paths


def validate_request_document(document: object) -> list[str]:
    """Reject every field other than the ticket id and the reason.

    ``additionalProperties`` is false on purpose. An instance id in the file
    cannot select a machine.
    """
    if not isinstance(document, dict):
        return ["request file must be a JSON object"]
    errors: list[str] = []
    extra = sorted(set(document) - set(REQUEST_FIELDS))
    missing = [name for name in REQUEST_FIELDS if name not in document]
    if extra:
        errors.append("unknown fields: " + ", ".join(extra))
    if missing:
        errors.append("missing fields: " + ", ".join(missing))
    ticket = document.get("ticket_id")
    if "ticket_id" in document and (
        not isinstance(ticket, str) or _TICKET_RE.fullmatch(ticket) is None
    ):
        errors.append("ticket_id must look like SF-14")
    reason = document.get("reason")
    if "reason" in document and (
        not isinstance(reason, str)
        or len(reason) > 500
        or _REASON_RE.fullmatch(reason) is None
    ):
        errors.append("reason must be one non-empty line of at most 500 characters")
    return errors


def validate_request_text(text: str) -> list[str]:
    """Parse a request file. Invalid JSON is a refusal, not a GPU run."""
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        return ["request file is not JSON"]
    return validate_request_document(document)


def set_github_output(name: str, value: str) -> None:
    """Append one outputs line. Prints the pair when not running in Actions."""
    path = os.environ.get("GITHUB_OUTPUT")
    line = f"{name}={value}\n"
    if not path:
        print(line, end="")
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line)


def _sha_or_none(value: str) -> str | None:
    if _SHA_RE.fullmatch(value) is None:
        return None
    return value


def check_push_diff() -> int:
    """Refuse a push range that is not exactly the request file.

    Exit 0 with ``proceed=false`` means the GPU job does nothing. Exit 1 is
    a broken git invocation and also skips the GPU job, because the guard
    did not report ``proceed=true``. This function does not call Vast.
    """
    before = os.environ.get("BEFORE", "")
    after = os.environ.get("AFTER", "")
    if _sha_or_none(before) is None or _sha_or_none(after) is None:
        print(
            "gpu-smoke request-file refused: before or after SHA is not 40 hex characters"
        )
        set_github_output("proceed", "false")
        return 0
    if before == _ZERO_SHA:
        print("gpu-smoke request-file refused: push has no before SHA")
        set_github_output("proceed", "false")
        return 0
    diff = subprocess.run(
        ["git", "diff", "--name-status", "--find-renames", before, after],
        check=False,
        capture_output=True,
        text=True,
    )
    if diff.returncode != 0:
        print("gpu-smoke request-file refused: could not diff the push range")
        set_github_output("proceed", "false")
        return 0
    ok, message = classify_push_diff(paths_from_name_status(diff.stdout.splitlines()))
    print(message)
    if not ok:
        set_github_output("proceed", "false")
        return 0
    shown = subprocess.run(
        ["git", "show", f"{after}:{REQUEST_FILE}"],
        check=False,
        capture_output=True,
    )
    if shown.returncode != 0:
        print(
            "gpu-smoke request-file refused: request file is not in the pushed commit"
        )
        set_github_output("proceed", "false")
        return 0
    errors = validate_request_text(shown.stdout.decode("utf-8"))
    if errors:
        print("gpu-smoke request-file refused: " + "; ".join(errors))
        set_github_output("proceed", "false")
        return 0
    print(f"gpu-smoke actor will be recorded for request-file {after}")
    set_github_output("proceed", "true")
    return 0


def workflow_label_actors(text: str) -> list[str]:
    """Actor logins named by ``github.actor == '...'`` in a workflow."""
    return _ACTOR_RE.findall(text)


def _on_block(text: str) -> str:
    lines = text.splitlines()
    start = next((index for index, line in enumerate(lines) if line == "on:"), None)
    if start is None:
        return ""
    block: list[str] = []
    for line in lines[start + 1 :]:
        if line and not line.startswith(" ") and not line.startswith("#"):
            break
        block.append(line)
    return "\n".join(block)


def _jobs(text: str) -> dict[str, str]:
    lines = text.splitlines()
    start = next((index for index, line in enumerate(lines) if line == "jobs:"), None)
    if start is None:
        return {}
    jobs: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    for line in lines[start + 1 :]:
        if (
            line.startswith("  ")
            and not line.startswith("   ")
            and line.endswith(":")
            and not line.strip().startswith("#")
        ):
            if current is not None:
                jobs[current] = "\n".join(buf)
            current = line.strip()[:-1]
            buf = []
            continue
        if current is not None:
            buf.append(line)
    if current is not None:
        jobs[current] = "\n".join(buf)
    return jobs


def _condition_if(block: str) -> str:
    """Return the first ``if:`` expression in a job or step block."""
    lines = block.splitlines()
    collected: list[str] = []
    capturing = False
    if_indent = 0
    for line in lines:
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        if not capturing and stripped.startswith("if:"):
            capturing = True
            if_indent = indent
            collected.append(stripped[len("if:") :].strip())
            continue
        if capturing:
            if stripped == "" or indent > if_indent:
                collected.append(stripped)
                continue
            break
    return "\n".join(collected)


def _job_if(job_text: str) -> str:
    lines = job_text.splitlines()
    collected: list[str] = []
    capturing = False
    for line in lines:
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        if not capturing and stripped.startswith("if:"):
            capturing = True
            collected.append(stripped[len("if:") :].strip())
            continue
        if capturing:
            if stripped == "" or indent > 4:
                collected.append(stripped)
                continue
            break
    return "\n".join(collected)


def _steps(job_text: str) -> list[str]:
    lines = job_text.splitlines()
    steps: list[str] = []
    current: list[str] = []
    in_steps = False
    for line in lines:
        if line.startswith("    steps:"):
            in_steps = True
            continue
        if not in_steps:
            continue
        if line.startswith("    ") and not line.startswith("     ") and line.strip():
            break
        if line.startswith("      - "):
            if current:
                steps.append("\n".join(current))
            current = [line]
            continue
        if current:
            current.append(line)
    if current:
        steps.append("\n".join(current))
    return steps


def _step_run(step: str) -> str:
    lines = step.splitlines()
    collected: list[str] = []
    capturing = False
    run_indent = 0
    for line in lines:
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        if not capturing and stripped.startswith("run:"):
            capturing = True
            run_indent = indent
            collected.append(stripped[len("run:") :].strip())
            continue
        if capturing:
            if stripped == "" or indent > run_indent:
                collected.append(line)
                continue
            break
    return "\n".join(collected)


def _is_keyed_step(step: str) -> bool:
    return "VAST_API_KEY" in step or "secrets.VAST_API_KEY" in step


def _checkout_steps(steps: list[str]) -> list[str]:
    return [step for step in steps if "actions/checkout" in step]


def _line_puts_payload_on_import_path(stripped: str, name: str) -> bool:
    """True when this workflow line assigns ``name`` and the value names payload."""
    markers = (f"{name}:", f"{name}=", f"export {name}")
    if not any(marker in stripped for marker in markers):
        return False
    return "payload" in stripped


def _keyed_import_findings(step: str) -> list[str]:
    """Refuse env and commands that can import the payload tree.

    ``--checkout`` may name ``payload`` as the directory shipped to the GPU.
    That is data. ``PYTHONPATH``, ``PYTHONSTARTUP``, ``sys.path``, a ``.pth``
    file, ``sitecustomize``, and ``pip install`` from payload are not.
    """
    findings: list[str] = []
    for line in step.splitlines():
        stripped = line.strip()
        for name in ("PYTHONPATH", "PYTHONSTARTUP"):
            if _line_puts_payload_on_import_path(stripped, name):
                findings.append(f"keyed step puts payload on {name}")
        if "sitecustomize" in stripped:
            findings.append("keyed step references sitecustomize")
        if ".pth" in stripped:
            findings.append("keyed step references a .pth import path")
        if "sys.path" in stripped or "addsitedir" in stripped:
            findings.append("keyed step edits sys.path")
    run = _step_run(step)
    if "pip install" in run and "payload" in run:
        findings.append("keyed step pip-installs from payload")
    return findings


def smoke_trigger_mapping(text: str) -> dict[str, str] | None:
    """Parse ``SMOKE_TRIGGER`` into event name to trigger name, plus fallback.

    ``None`` means the expression is not a chain of
    ``github.event_name == '…' && '…'`` clauses ending in one literal.
    """
    found = _SMOKE_TRIGGER_LINE.findall(text)
    if len(found) != 1:
        return None
    expression = " ".join(found[0].split())
    clauses = [part.strip() for part in expression.split("||")]
    if len(clauses) < 2:
        return None
    mapping: dict[str, str] = {}
    for clause in clauses[:-1]:
        event = _EVENT_CLAUSE.fullmatch(clause)
        if event is None:
            return None
        name, value = event.group(1), event.group(2)
        if name in mapping or name == "fallback":
            return None
        mapping[name] = value
    literal = _LITERAL_CLAUSE.fullmatch(clauses[-1])
    if literal is None:
        return None
    mapping["fallback"] = literal.group(1)
    return mapping


def _smoke_trigger_findings(text: str) -> list[str]:
    """Findings when SMOKE_TRIGGER is not the pinned event map."""
    mapping = smoke_trigger_mapping(text)
    if mapping == SMOKE_TRIGGER_MAPPING:
        return []
    if mapping is None:
        return [
            "SMOKE_TRIGGER mapping is not dispatch, label, request-file, and invalid"
        ]
    findings: list[str] = []
    for event, want in SMOKE_TRIGGER_MAPPING.items():
        if event == "fallback":
            continue
        got = mapping.get(event)
        if got != want:
            findings.append(f"SMOKE_TRIGGER maps {event} to {got!r}, not {want!r}")
    if mapping.get("fallback") != "invalid":
        findings.append(
            "SMOKE_TRIGGER maps any other event to "
            f"{mapping.get('fallback')!r}, not 'invalid'"
        )
    for name in sorted(set(mapping) - set(SMOKE_TRIGGER_MAPPING)):
        findings.append(f"SMOKE_TRIGGER maps {name} to {mapping[name]!r}")
    return findings or [
        "SMOKE_TRIGGER mapping is not dispatch, label, request-file, and invalid"
    ]


def label_cleanup_runs(
    expression: str,
    *,
    smoke_result: str,
    event_name: str = "pull_request_target",
    action: str = "labeled",
    event_label: str = RUN_GPU_SMOKE_LABEL,
    head_repo: str = "sandy-t-shelvan/SimFoundry",
    repository: str = "sandy-t-shelvan/SimFoundry",
) -> bool:
    """Whether the label-cleanup ``if`` runs for this event and smoke result.

    ``always()`` is true, including after a cancel. The label gate is
    ``github.event.label.name == 'run-gpu-smoke'``. A different label, or a
    head repo that is not ``github.repository``, is false. A
    ``needs.fruits_gpu_smoke.result != '…'`` clause is false when the result
    is that value, so ``!= 'skipped'`` would not run after a cancelled queue.
    """
    flat = " ".join(expression.split())
    if flat.startswith(">-"):
        flat = flat[2:].strip()
    elif flat.startswith((">", "|")):
        flat = flat[1:].strip()
    if "always()" not in flat:
        return False
    saw_label_gate = False
    for raw in flat.split("&&"):
        clause = raw.strip()
        if clause in {"", "always()"}:
            continue
        event_eq = re.fullmatch(r"github\.event_name == '([^']*)'", clause)
        if event_eq is not None:
            if event_name != event_eq.group(1):
                return False
            continue
        action_eq = re.fullmatch(r"github\.event\.action == '([^']*)'", clause)
        if action_eq is not None:
            if action != action_eq.group(1):
                return False
            continue
        label_eq = re.fullmatch(r"github\.event\.label\.name == '([^']*)'", clause)
        if label_eq is not None:
            if event_label != label_eq.group(1):
                return False
            if label_eq.group(1) == RUN_GPU_SMOKE_LABEL:
                saw_label_gate = True
            continue
        repo_eq = re.fullmatch(
            r"github\.event\.pull_request\.head\.repo\.full_name == github\.repository",
            clause,
        )
        if repo_eq is not None:
            if head_repo != repository:
                return False
            continue
        excluded = re.fullmatch(
            r"needs\.fruits_gpu_smoke\.result != '([^']*)'",
            clause,
        )
        if excluded is not None:
            if smoke_result == excluded.group(1):
                return False
            continue
        required = re.fullmatch(
            r"needs\.fruits_gpu_smoke\.result == '([^']*)'",
            clause,
        )
        if required is not None:
            if smoke_result != required.group(1):
                return False
            continue
        return False
    return saw_label_gate


def _always_without_cancelled(expression: str) -> bool:
    """True when ``always()`` is used and ``!cancelled()`` is not in the same if."""
    return "always()" in expression and "!cancelled()" not in expression


def _start_put_cancelled_findings(jobs: dict[str, str]) -> list[str]:
    """Flag ``always()`` on a job or step that can reach ``vast_smoke``.

    ``label_cleanup`` uses ``always()`` and does not call Vast, so it is
    not on this path. A cancelled queued run must not reach
    ``sceneagent.vast_smoke``. That module does not start or stop the instance.
    """
    findings: list[str] = []
    for name, job in jobs.items():
        if "python -m sceneagent.vast_smoke" not in job:
            continue
        if _always_without_cancelled(_job_if(job)):
            findings.append(
                f"{name} job if uses always() without !cancelled() "
                "on the path to vast_smoke"
            )
        for step in _steps(job):
            if _always_without_cancelled(_condition_if(step)):
                findings.append(
                    f"{name} step uses always() without !cancelled() "
                    "on the path to vast_smoke"
                )
            if "python -m sceneagent.vast_smoke" in step:
                break
    return findings


def workflow_safety_findings(text: str) -> list[str]:
    """Static contract for the GPU smoke workflow. Empty means it holds.

    The keyed step must run the orchestrator from the PR base checkout, and
    no step that holds ``VAST_API_KEY`` or a write token may execute the
    payload tree. The request-file guard must diff ``before..after`` using
    the checker from the commit before the push.
    """
    findings: list[str] = []
    findings.extend(_smoke_trigger_findings(text))
    on_block = _on_block(text)
    on_lines = {line.strip() for line in on_block.splitlines()}
    if "workflow_dispatch:" not in on_lines:
        findings.append("workflow_dispatch is missing")
    if "pull_request_target:" not in on_lines:
        findings.append("label path must be pull_request_target")
    for forbidden in ("pull_request:", "schedule:", "issue_comment:", "workflow_run:"):
        if forbidden in on_lines:
            findings.append(f"{forbidden} is not a gpu-smoke trigger")
    if "inputs:" in text or "inputs." in text:
        findings.append("workflow_dispatch must stay free of inputs")
    if "- Main_duplicate" not in on_block:
        findings.append("request-file push must be limited to Main_duplicate")
    if "- main" in on_lines:
        findings.append("request-file push must not include main")
    if "ci/gpu-run-request.json" not in on_block:
        findings.append("request-file push must filter on ci/gpu-run-request.json")
    if "58301075" in text:
        findings.append("workflow must not name retired instance 58301075")
    if re.search(r"""["']state["']\s*:\s*["']running["']""", text):
        findings.append("workflow must not send a Vast state running")
    if re.search(r"""["']state["']\s*:\s*["']stopped["']""", text):
        findings.append("workflow must not send a Vast state stopped")
    if "group: gpu-smoke-59601312" not in text:
        findings.append("concurrency group gpu-smoke-59601312 is missing")
    if "cancel-in-progress: false" not in text:
        findings.append("cancel-in-progress must be false")
    before_jobs = text.split("jobs:", 1)[0]
    if "group: gpu-smoke-59601312" in before_jobs:
        findings.append(
            "concurrency must be on the smoke job so an unrelated label does not take the group"
        )
    if text.count("secrets.VAST_API_KEY") != 1:
        findings.append("VAST_API_KEY must appear in exactly one step")
    if text.count("secrets.VAST_SSH_PRIVATE_KEY") != 1:
        findings.append(
            "VAST_SSH_PRIVATE_KEY must be passed from the Actions secret in exactly one step"
        )
    if "--instance-id 59601312" not in text:
        findings.append("instance id must be the literal 59601312")
    actors = workflow_label_actors(text)
    if tuple(actors) != LABEL_ACTORS:
        findings.append(
            "label actor allowlist must be sandy-t-shelvan and cursor[bot], in that order"
        )
    jobs = _jobs(text)
    findings.extend(_start_put_cancelled_findings(jobs))
    smoke = jobs.get("fruits_gpu_smoke", "")
    guard = jobs.get("request_file_guard", "")
    cleanup = jobs.get("label_cleanup", "")
    if not smoke or not guard or not cleanup:
        findings.append(
            "request_file_guard, fruits_gpu_smoke, and label_cleanup are required"
        )
        return findings
    smoke_if = _job_if(smoke)
    for needle in (
        "github.event_name == 'pull_request_target'",
        "github.event.label.name == 'run-gpu-smoke'",
        "github.event.pull_request.head.repo.full_name == github.repository",
        "github.event.pull_request.base.ref == 'Main_duplicate'",
        "github.event.pull_request.base.ref == 'main'",
        "needs.request_file_guard.outputs.proceed == 'true'",
    ):
        if needle not in smoke_if:
            findings.append(f"smoke if is missing {needle}")
    if "pull-requests: write" in smoke:
        findings.append("the smoke job must not hold pull-requests: write")
    if "contents: read" not in smoke:
        findings.append("the smoke job needs contents: read")
    if "group: gpu-smoke-59601312" not in smoke:
        findings.append("the smoke job must use concurrency group gpu-smoke-59601312")
    if "cancel-in-progress: false" not in smoke:
        findings.append("the smoke job must set cancel-in-progress false")
    guard_run = "\n".join(_step_run(step) for step in _steps(guard))
    detach_at = guard_run.find('git checkout --detach "$BEFORE"')
    checker_at = guard_run.find(
        "python -m sceneagent.gpu_smoke_trigger check-push-diff"
    )
    if detach_at < 0 or checker_at < 0 or detach_at > checker_at:
        findings.append(
            "the diff checker must run from git checkout --detach of the before SHA"
        )
    if "VAST_API_KEY" in guard:
        findings.append("the request-file guard must not hold VAST_API_KEY")
    smoke_steps = _steps(smoke)
    keyed = [step for step in smoke_steps if _is_keyed_step(step)]
    if len(keyed) != 1:
        findings.append("exactly one smoke step may hold VAST_API_KEY")
    else:
        step = keyed[0]
        run = _step_run(step)
        if "working-directory: orchestrator" not in step:
            findings.append("the keyed step must run in the orchestrator checkout")
        if "python -m sceneagent.vast_smoke" not in run:
            findings.append("the keyed step must run sceneagent.vast_smoke")
        if "--instance-id 59601312" not in run:
            findings.append("the keyed step must pass instance 59601312 as a literal")
        if "VAST_SSH_PRIVATE_KEY" in run:
            findings.append(
                "keyed run script must not print or interpolate the ssh private key"
            )
        if "secrets.VAST_SSH_PRIVATE_KEY" not in step:
            findings.append(
                "the keyed step must receive VAST_SSH_PRIVATE_KEY from the Actions secret"
            )
        for banned in (
            "pip install",
            "cd ",
            "payload/sceneagent",
            "python payload",
            "import ",
        ):
            if banned in run:
                findings.append(f"keyed step executes payload code ({banned.strip()})")
        if "working-directory: payload" in step:
            findings.append(
                "keyed step executes payload code (working-directory payload)"
            )
        findings.extend(_keyed_import_findings(step))
        if "${{" in run:
            findings.append("keyed run script must not interpolate expressions")
    for step in _checkout_steps(smoke_steps):
        if "persist-credentials: false" not in step:
            findings.append("every checkout must set persist-credentials: false")
        if "path: payload" in step and _is_keyed_step(step):
            findings.append("payload checkout must not be a keyed step")
    base_checkout = [
        step
        for step in smoke_steps
        if "path: orchestrator" in step and "github.event.pull_request.base.sha" in step
    ]
    head_checkout = [
        step
        for step in smoke_steps
        if "path: payload" in step and "github.event.pull_request.head.sha" in step
    ]
    if len(base_checkout) != 1:
        findings.append(
            "orchestrator checkout ref must be github.event.pull_request.base.sha"
        )
    if len(head_checkout) != 1:
        findings.append("payload checkout ref must be the PR head SHA")
    elif "persist-credentials: false" not in head_checkout[0]:
        findings.append("payload checkout must set persist-credentials: false")
    if "pull-requests: write" not in cleanup:
        findings.append("label cleanup needs pull-requests: write")
    if (
        "contents: write" in cleanup
        or "actions/checkout" in cleanup
        or "path: payload" in cleanup
    ):
        findings.append("label cleanup must not checkout or execute the PR head")
    if "VAST_API_KEY" in cleanup:
        findings.append("label cleanup must not hold VAST_API_KEY")
    cleanup_if = _job_if(cleanup)
    if "always()" not in cleanup_if:
        findings.append("label cleanup must run on failure with if: always()")
    # A cancelled queued smoke is skipped by !cancelled(). Cleanup still has
    # to remove run-gpu-smoke, so the if must stay true for result cancelled
    # and for skipped. The event must still be a same-repo run-gpu-smoke label.
    if not label_cleanup_runs(cleanup_if, smoke_result="cancelled"):
        findings.append(
            "label cleanup must run when the smoke job is cancelled "
            "and the labeled event is run-gpu-smoke on the same repo"
        )
    if not label_cleanup_runs(cleanup_if, smoke_result="skipped"):
        findings.append(
            "label cleanup must run when a cancelled queue skips the smoke job "
            "and the labeled event is run-gpu-smoke on the same repo"
        )
    for job_name, job_text in jobs.items():
        if "timeout-minutes:" not in job_text:
            findings.append(f"{job_name} is missing timeout-minutes")
        if "runs-on: ubuntu-latest" not in job_text:
            findings.append(f"{job_name} must run on ubuntu-latest")
        if "pull-requests: write" in job_text and job_name != "label_cleanup":
            findings.append(f"{job_name} must not hold pull-requests: write")
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="GPU smoke trigger checks. Does not call Vast."
    )
    parser.add_argument("command", choices=["check-push-diff"])
    args = parser.parse_args(argv)
    if args.command == "check-push-diff":
        return check_push_diff()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
