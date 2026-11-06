# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manual Vast smoke for ``python -m sceneagent`` on Fruits.mp4.

The only instance this module will touch is ``58301075`` in Texas. It may
show that instance, start it when it is stopped, and stop it again when this
process was the one that started it. It attaches an ephemeral SSH public key
so the runner can copy the checkout and bring the run files back.

It does not create, rent, resize, or destroy an instance, and the remote
command is only the SceneAgent video entry point. That path does not call
Meshy or Gemini. The API key is read from the environment and is not written
into the remote command, the URL, or the artifact files.

``python -m sceneagent`` on an mp4 writes ``ledger.json``. It does not write
``plan.json``. ``collect_artifact_manifest`` does not build one. A missing
plan fails the smoke. The manifest is this module's record of that fact.
A ``deferred`` or ``reject`` ledger entry stays a failed smoke.
"""

from __future__ import annotations

import io
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

API_ORIGIN = "https://console.vast.ai"
ALLOWED_INSTANCE_ID = 58301075
FRUITS_RELATIVE = "docs/assets/example_videos/Fruits.mp4"
REMOTE_REPO = "/workspace/sf4-smoke/repo"
REMOTE_OUT = "/workspace/sf4-smoke/run"
BOOT_TIMEOUT_S = 600.0
REMOTE_TIMEOUT_S = 3300.0
POLL_SECONDS = 15.0
REQUIRED_LEDGER_ITEMS = ("video", "COLMAP camera poses")
FETCH_NAMES = (
    "ledger.json",
    "plan.json",
    "manifest.json",
    "video.json",
    "colmap_poses.json",
    "tool_probe.json",
    "host_probe.json",
)
PUBLIC_FIELDS = (
    "id",
    "actual_status",
    "intended_status",
    "cur_state",
    "next_state",
    "geolocation",
    "ssh_host",
    "ssh_port",
    "gpu_name",
    "num_gpus",
    "label",
    "image_uuid",
)
_REMOTE_DIR_RE = re.compile(r"^/[A-Za-z0-9_./-]+$")
_INSTANCE_PATH_RE = re.compile(
    r"^/api/v0/instances/(?P<id>\d+)/(?:ssh/(?:(?P<key>[A-Za-z0-9_-]+)/)?)?$"
)
_HOST_RE = re.compile(r"^[A-Za-z0-9.-]+$")

RequestFn = Callable[..., tuple[int, dict[str, Any]]]


class InstanceNotApproved(ValueError):
    """The call is outside the SF-4 allowlist for instance 58301075."""


class SmokeError(RuntimeError):
    """The smoke could not finish. The ledger was not rewritten to hide it."""


def parse_instance_id(value: object) -> int:
    """Return 58301075. Any other id is refused before a network call."""
    parsed: int | None
    if isinstance(value, bool):
        parsed = None
    elif isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
    else:
        parsed = None
    if parsed != ALLOWED_INSTANCE_ID:
        if isinstance(value, str) or (
            isinstance(value, int) and not isinstance(value, bool)
        ):
            shown: object = value
        else:
            shown = type(value).__name__
        raise InstanceNotApproved(
            f"Refusing instance {shown!r}. "
            f"SF-4 may use only the existing Texas instance {ALLOWED_INSTANCE_ID}."
        )
    return ALLOWED_INSTANCE_ID


def require_api_key(value: object) -> str:
    """Return the key when it is a single non-empty line. Do not echo it."""
    if not isinstance(value, str) or not value or any(ch.isspace() for ch in value):
        raise InstanceNotApproved("VAST_API_KEY is not set.")
    return value


def redact(text: str, secret: str) -> str:
    """Remove the API key from a string that is about to be printed."""
    if not secret:
        return text
    return text.replace(secret, "[redacted]")


def mask_secret(value: str) -> None:
    """Do not print the key. Actions already masks ``secrets.VAST_API_KEY``."""
    if not isinstance(value, str):
        return


def child_env() -> dict[str, str]:
    """Environment for ssh and rsync. The API key stays on the runner."""
    env = dict(os.environ)
    env.pop("VAST_API_KEY", None)
    return env


def _emit(message: str, api_key: str) -> None:
    print(redact(message, api_key), file=sys.stderr, flush=True)


def _log(payload: dict[str, Any], api_key: str) -> None:
    print(redact(json.dumps(payload, sort_keys=True), api_key), flush=True)


def _same_id(value: object) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, int):
        return value == ALLOWED_INSTANCE_ID
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip()) == ALLOWED_INSTANCE_ID
    return False


def unwrap_instance(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return the instance object from a show-instance body, or None."""
    if "instances" in payload:
        body = payload["instances"]
        if body is None:
            return None
        if isinstance(body, dict):
            return body
        if isinstance(body, list):
            matches = [
                row for row in body if isinstance(row, dict) and _same_id(row.get("id"))
            ]
            if len(matches) == 1:
                return matches[0]
            return None
        return None
    if isinstance(payload, dict) and _same_id(payload.get("id")):
        return payload
    return None


def require_approved_instance(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the instance when it is 58301075 in Texas."""
    row = unwrap_instance(payload)
    if row is None:
        raise InstanceNotApproved(
            f"instance {ALLOWED_INSTANCE_ID} is missing. Not creating a replacement."
        )
    found = row.get("id")
    if not _same_id(found):
        raise InstanceNotApproved(
            f"show instance returned id {found!r}, not {ALLOWED_INSTANCE_ID}."
        )
    geo = row.get("geolocation")
    if geo is None:
        geo = row.get("geolocode")
    geo_text = "" if geo is None else str(geo)
    if "texas" not in geo_text.lower():
        raise InstanceNotApproved(
            f"instance {ALLOWED_INSTANCE_ID} geolocation is {geo_text!r}, not Texas."
        )
    return row


def public_instance_view(record: dict[str, Any]) -> dict[str, Any]:
    """Fields that are safe to print. Tokens and startup scripts stay out."""
    view: dict[str, Any] = {}
    for name in PUBLIC_FIELDS:
        if name in record:
            view[name] = record[name]
    return view


def classify_status(record: dict[str, Any]) -> str:
    """Map a show-instance row to running, starting, stopped, or unknown.

    ``stopped`` and ``exited`` are startable. ``exited`` is what the Vast
    listing showed for this machine when it was not running. ``offline`` and
    ``unknown`` are not started: the API docs say those states do not become
    ``running``, and this module does not destroy the instance to replace it.
    """
    actual = _status_text(record.get("actual_status"))
    intended = _status_text(record.get("intended_status") or record.get("cur_state"))
    if actual == "running":
        return "running"
    if actual in {"loading", "created", "rebooting"}:
        return "starting"
    if actual in {"offline", "unknown", "frozen"}:
        return "unknown"
    if actual in {"exited", "stopped"}:
        return "stopped"
    if actual == "" and intended == "running":
        return "starting"
    if actual == "" and intended in {"stopped", "exited"}:
        return "stopped"
    return "unknown"


def _status_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _require_public_key(value: object) -> str:
    if not isinstance(value, str):
        raise InstanceNotApproved(
            "refusing to attach a key that is not an ssh public key"
        )
    line = value.strip()
    if "\n" in line or line.startswith("-----") or "PRIVATE" in line:
        raise InstanceNotApproved("refusing to attach a private key")
    if not line.startswith(("ssh-ed25519 ", "ssh-rsa ")):
        raise InstanceNotApproved(
            "refusing to attach a key that is not an ssh public key"
        )
    return line


def _require_ssh_key_id(value: object) -> str:
    if isinstance(value, bool) or value is None:
        raise InstanceNotApproved("refusing to detach without the attached key id")
    text = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", text):
        raise InstanceNotApproved(
            "refusing to detach a key id with an unexpected shape"
        )
    return text


def assert_request_allowed(method: str, url: str, body: dict[str, Any] | None) -> None:
    """Reject every Vast call except show, start, stop, attach, and detach."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.netloc != "console.vast.ai":
        raise InstanceNotApproved("refusing a Vast URL outside https://console.vast.ai")
    if parsed.fragment:
        raise InstanceNotApproved("refusing a URL fragment")
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    for name, _query_value in query:
        if name.lower() in {"api_key", "token", "bearer", "authorization"}:
            raise InstanceNotApproved("refusing to put a credential in the URL")
    match = _INSTANCE_PATH_RE.fullmatch(parsed.path)
    if match is None or int(match.group("id")) != ALLOWED_INSTANCE_ID:
        raise InstanceNotApproved(f"refusing Vast path {parsed.path}")
    key_id = match.group("key")
    is_ssh = "/ssh/" in parsed.path
    method = method.upper()
    if method == "GET" and not is_ssh and query == [("owner", "me")] and body is None:
        return
    if (
        method == "PUT"
        and not is_ssh
        and not query
        and isinstance(body, dict)
        and set(body) == {"state"}
        and body["state"] in {"running", "stopped"}
    ):
        return
    if (
        method == "POST"
        and parsed.path.endswith("/ssh/")
        and key_id is None
        and not query
        and isinstance(body, dict)
        and set(body) == {"ssh_key"}
    ):
        _require_public_key(body.get("ssh_key"))
        return
    if method == "DELETE" and key_id and not query and body is None:
        _require_ssh_key_id(key_id)
        return
    raise InstanceNotApproved(f"refusing Vast {method} {parsed.path}")


def endpoint_for(
    kind: str,
    instance_id: object,
    *,
    public_key: str | None = None,
    ssh_key_id: str | None = None,
) -> tuple[str, str, dict[str, Any] | None]:
    """Build one allowlisted request. Any other kind raises."""
    approved = parse_instance_id(instance_id)
    base = f"{API_ORIGIN}/api/v0/instances/{approved}/"
    if kind == "show":
        method, url, body = "GET", base + "?owner=me", None
    elif kind == "start":
        method, url, body = "PUT", base, {"state": "running"}
    elif kind == "stop":
        method, url, body = "PUT", base, {"state": "stopped"}
    elif kind == "attach_ssh":
        method, url, body = (
            "POST",
            base + "ssh/",
            {"ssh_key": _require_public_key(public_key)},
        )
    elif kind == "detach_ssh":
        key_id = _require_ssh_key_id(ssh_key_id)
        method, url, body = "DELETE", f"{base}ssh/{key_id}/", None
    else:
        raise InstanceNotApproved(f"refusing Vast action {kind!r}")
    assert_request_allowed(method, url, body)
    return method, url, body


def decode_body(raw: str) -> dict[str, Any]:
    """Parse a JSON object. Anything else becomes an empty object."""
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def request_json(
    method: str,
    url: str,
    api_key: str,
    body: dict[str, Any] | None = None,
    timeout: float = 60,
) -> tuple[int, dict[str, Any]]:
    """Perform one allowlisted Vast request. The key travels in the header."""
    assert_request_allowed(method, url, body)
    payload = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method=method)
    req.add_header("Authorization", "Bearer " + api_key)
    req.add_header("Accept", "application/json")
    if payload is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = int(resp.status)
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        raw = exc.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", "request failed")
        raise SmokeError(f"Vast API request failed ({reason})") from None
    return status, decode_body(raw)


def call_json(
    request: RequestFn,
    method: str,
    url: str,
    api_key: str,
    body: dict[str, Any] | None,
    *,
    sleep: Callable[[float], None],
    attempts: int = 5,
    timeout: float = 60,
) -> tuple[int, dict[str, Any]]:
    """Call ``request``, waiting out HTTP 429. Other statuses return immediately."""
    status = 0
    payload: dict[str, Any] = {}
    for attempt in range(attempts):
        status, payload = request(method, url, api_key, body, timeout=timeout)
        if status != 429:
            return status, payload
        if attempt + 1 < attempts:
            sleep(POLL_SECONDS)
    return status, payload


def require_remote_dir(path: str) -> str:
    """Absolute remote path with no ``..`` and no shell metacharacters."""
    if not isinstance(path, str) or _REMOTE_DIR_RE.fullmatch(path) is None:
        raise InstanceNotApproved(f"refusing remote path {path!r}")
    if any(part == ".." for part in path.split("/")):
        raise InstanceNotApproved(f"refusing remote path {path!r}")
    return path


def fruits_path(checkout: Path) -> Path:
    path = Path(checkout) / FRUITS_RELATIVE
    if not path.is_file():
        raise InstanceNotApproved(f"missing {FRUITS_RELATIVE} in the checkout")
    return path


def build_remote_command(repo_dir: str, out_dir: str) -> list[str]:
    """Argv for the Fruits video entry point. No API key and no paid client."""
    repo = require_remote_dir(repo_dir)
    out = require_remote_dir(out_dir)
    fruits = f"{repo}/{FRUITS_RELATIVE}"
    return ["python3", "-m", "sceneagent", fruits, "--out", out]


def host_probe_command(out_dir: str) -> str:
    """Shell fragment that records which GPU tools are on the instance."""
    out = require_remote_dir(out_dir)
    code = "\n".join(
        [
            "import json, pathlib, shutil, subprocess",
            f"out = pathlib.Path({out!r})",
            "out.mkdir(parents=True, exist_ok=True)",
            "probe = {",
            "    'python3': shutil.which('python3'),",
            "    'python': shutil.which('python'),",
            "    'ffmpeg': shutil.which('ffmpeg'),",
            "    'colmap': shutil.which('colmap'),",
            "    'nvidia_smi': shutil.which('nvidia-smi'),",
            "}",
            "colmap = probe['colmap']",
            "if colmap:",
            "    proc = subprocess.run([colmap, '-h'], capture_output=True, text=True)",
            "    banner = (proc.stdout or '') + (proc.stderr or '')",
            "    probe['colmap_banner_tail'] = banner[-400:]",
            "    probe['colmap_cuda'] = 'without CUDA' not in banner",
            "smi = probe['nvidia_smi']",
            "if smi:",
            "    proc = subprocess.run([smi, '-L'], capture_output=True, text=True)",
            "    probe['nvidia_smi_ok'] = proc.returncode == 0",
            "    probe['nvidia_smi_tail'] = (proc.stdout or proc.stderr or '')[-300:]",
            "(out / 'host_probe.json').write_text(json.dumps(probe, indent=2) + '\\n')",
        ]
    )
    return f'"$py" -c {shlex.quote(code)}'


def build_remote_shell(repo_dir: str, out_dir: str) -> str:
    """Bash script that runs the video entry point with the repo on PYTHONPATH."""
    repo = require_remote_dir(repo_dir)
    out = require_remote_dir(out_dir)
    argv = build_remote_command(repo, out)
    quoted_args = " ".join(shlex.quote(part) for part in argv[1:])
    lines = [
        "set -eu",
        f"cd {shlex.quote(repo)}",
        f"export PYTHONPATH={shlex.quote(repo)}",
        f"mkdir -p {shlex.quote(out)}",
        "if command -v python3 >/dev/null 2>&1; then py=python3; else py=python; fi",
        host_probe_command(out),
        f'"$py" {quoted_args}',
    ]
    script = "\n".join(lines) + "\n"
    _reject_secret_text(script)
    return script


def build_fetch_script(out_dir: str) -> str:
    """Tar the small run files to stdout. Missing names are skipped."""
    out = require_remote_dir(out_dir)
    names = " ".join(shlex.quote(name) for name in FETCH_NAMES)
    script = "\n".join(
        [
            "set -eu",
            f"cd {shlex.quote(out)}",
            f"for name in {names}; do",
            '  if [ -f "$name" ]; then printf "%s\\n" "$name"; fi',
            "done | tar -cf - -T -",
        ]
    )
    _reject_secret_text(script)
    return script + "\n"


def _reject_secret_text(text: str) -> None:
    lowered = text.lower()
    for banned in ("vast_api_key", "bearer ", "api_key", "gemini", "meshy"):
        if banned in lowered:
            raise InstanceNotApproved("remote command contains a forbidden token")


def ssh_base_argv(key_path: str, known_hosts: str, port: int) -> list[str]:
    return [
        "ssh",
        "-i",
        key_path,
        "-p",
        str(port),
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "ConnectTimeout=20",
    ]


def ssh_endpoint(record: dict[str, Any]) -> tuple[str, int]:
    host = record.get("ssh_host")
    port = record.get("ssh_port")
    if not isinstance(host, str) or _HOST_RE.fullmatch(host) is None:
        raise SmokeError(f"instance {ALLOWED_INSTANCE_ID} has no ssh_host")
    if isinstance(port, str) and port.isdigit():
        port = int(port)
    if isinstance(port, bool) or not isinstance(port, int) or port <= 0:
        raise SmokeError(f"instance {ALLOWED_INSTANCE_ID} has no ssh_port")
    return host, port


def rsync_invocations(
    checkout: Path,
    host: str,
    port: int,
    key_path: str,
    known_hosts: str,
    repo_dir: str,
) -> list[tuple[list[str], Path]]:
    """Copy ``sceneagent`` and Fruits.mp4. The rest of the checkout stays home."""
    repo = require_remote_dir(repo_dir)
    remote = f"root@{host}:{repo}/"
    ssh_e = shlex.join(ssh_base_argv(key_path, known_hosts, port))
    scene = checkout / "sceneagent"
    return [
        (
            [
                "rsync",
                "-a",
                "--exclude",
                "__pycache__",
                "-e",
                ssh_e,
                str(scene),
                remote,
            ],
            checkout,
        ),
        (
            ["rsync", "-a", "-R", "-e", ssh_e, FRUITS_RELATIVE, remote],
            checkout,
        ),
    ]


def ssh_key_id_from_attach(payload: dict[str, Any]) -> str | None:
    """Read a key id from an attach response. A bare ``id`` is not enough."""
    for name in ("ssh_key_id", "key_id"):
        value = payload.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]+", value):
            return value
    for name in ("ssh_key", "key"):
        nested = payload.get(name)
        if isinstance(nested, dict):
            found = ssh_key_id_from_attach(nested)
            if found:
                return found
    return None


def _stage_report(entry: dict[str, Any]) -> dict[str, Any]:
    decision = entry.get("decision")
    item = entry.get("item")
    use_gpu = entry.get("use_gpu")
    report: dict[str, Any] = {
        "item": item,
        "decision": decision,
        "use_gpu": use_gpu,
        "ok": False,
        "counts_as_pass": False,
        "reason": entry.get("reason"),
        "failed_step": entry.get("failed_step"),
    }
    if decision == "accept" and item == "COLMAP camera poses" and use_gpu is not True:
        report["reported"] = "accept_without_gpu"
        return report
    if decision == "accept":
        report["ok"] = True
        report["counts_as_pass"] = True
        report["reported"] = "accept"
        return report
    if decision == "reject":
        report["reported"] = "failed"
        return report
    report["reported"] = "skipped" if decision == "deferred" else "unreported"
    return report


def _failure_reason(
    stages: list[dict[str, Any]], missing_items: list[str], missing_files: list[str]
) -> str | None:
    reasons: list[str] = []
    if "ledger" in missing_files:
        reasons.append("ledger.json is missing. Not inventing one.")
    if "plan" in missing_files:
        reasons.append(
            "plan.json was not written by the pipeline. Not synthesizing one."
        )
    for name in missing_items:
        reasons.append(f"ledger is missing {name}.")
    for stage in stages:
        if stage["counts_as_pass"]:
            continue
        item = stage["item"]
        decision = stage["decision"]
        if stage["reported"] == "skipped":
            reasons.append(
                f"{item} is deferred in the ledger. A skipped stage is not a pass."
            )
        elif stage["reported"] == "failed":
            reasons.append(
                f"{item} is reject in the ledger. A failed stage is not a pass."
            )
        elif stage["reported"] == "accept_without_gpu":
            reasons.append(
                f"{item} accepted without use_gpu. This is not a GPU smoke pass."
            )
        else:
            reasons.append(
                f"{item} has ledger decision {decision!r}. That is not a pass."
            )
    if not reasons:
        return None
    return " ".join(reasons)


def _host_probe_summary(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "host_probe.json"
    if not path.is_file():
        return None
    try:
        probe = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"present": True, "ok": False}
    if not isinstance(probe, dict):
        return {"present": True, "ok": False}
    return {
        "present": True,
        "colmap_cuda": probe.get("colmap_cuda"),
        "nvidia_smi_ok": probe.get("nvidia_smi_ok"),
    }


def _ledger_entries(ledger: dict[str, Any]) -> list[Any]:
    raw = ledger.get("entries")
    if isinstance(raw, list):
        return raw
    return []


def collect_artifact_manifest(
    run_dir: Path,
    *,
    command: list[str],
    instance_id: object,
) -> dict[str, Any]:
    """Record the run. Never rewrite ``ledger.json`` or invent ``plan.json``.

    ``plan.json`` is copied only when the pipeline already wrote that file.
    A missing plan fails the smoke. ``deferred`` stays ``deferred`` and
    ``reject`` stays ``reject``. ``use_gpu`` must be true on the COLMAP
    accept, because a CPU pose run is not this GPU smoke.
    """
    approved = parse_instance_id(instance_id)
    if any(not isinstance(part, str) for part in command):
        raise InstanceNotApproved("remote command must be a list of strings")
    _reject_secret_text("\n".join(command))
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = run_dir / "ledger.json"
    ledger_bytes = ledger_path.read_bytes() if ledger_path.is_file() else None
    ledger: dict[str, Any] | None = None
    if ledger_bytes is not None:
        parsed = json.loads(ledger_bytes.decode("utf-8"))
        if isinstance(parsed, dict):
            ledger = parsed
    plan_path = run_dir / "plan.json"
    stages: list[dict[str, Any]] = []
    if ledger is not None:
        stages = [
            _stage_report(entry)
            for entry in _ledger_entries(ledger)
            if isinstance(entry, dict)
        ]
    present_items = {stage["item"] for stage in stages}
    missing_items = [
        name for name in REQUIRED_LEDGER_ITEMS if name not in present_items
    ]
    missing_files: list[str] = []
    if ledger_bytes is None:
        missing_files.append("ledger")
    if not plan_path.is_file():
        missing_files.append("plan")
    reason = _failure_reason(
        stages, missing_items if ledger is not None else [], missing_files
    )
    passed = (
        ledger_bytes is not None
        and plan_path.is_file()
        and not missing_items
        and bool(stages)
        and all(stage["counts_as_pass"] for stage in stages)
        and reason is None
    )
    manifest = {
        "instance_id": approved,
        "video": FRUITS_RELATIVE,
        "command": command,
        "plan": "plan.json" if plan_path.is_file() else None,
        "ledger": "ledger.json" if ledger_bytes is not None else None,
        "manifest": "manifest.json",
        "host_probe": _host_probe_summary(run_dir),
        "stages": stages,
        "missing": missing_files,
        "complete": ledger_bytes is not None and plan_path.is_file(),
        "passed": passed,
        "failure_reason": reason,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if ledger_bytes is not None and ledger_path.read_bytes() != ledger_bytes:
        raise SmokeError("refusing to rewrite ledger.json")
    manifest["complete"] = (
        plan_path.is_file()
        and ledger_path.is_file()
        and (run_dir / "manifest.json").is_file()
    )
    if not manifest["complete"]:
        manifest["passed"] = False
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if ledger_bytes is not None and ledger_path.read_bytes() != ledger_bytes:
        raise SmokeError("refusing to rewrite ledger.json")
    return manifest


def publish_artifacts(run_dir: Path, artifact_dir: Path) -> list[str]:
    """Copy plan, ledger, and manifest when they exist. Do not invent the rest."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in ("plan.json", "ledger.json", "manifest.json"):
        src = run_dir / name
        if src.is_file():
            shutil.copyfile(src, artifact_dir / name)
            copied.append(name)
    return copied


def unpack_tar(blob: bytes, dest: Path) -> None:
    """Unpack a fetch tar. Members must be relative files under ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    if not blob:
        return
    try:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:") as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                name = Path(member.name)
                if name.is_absolute() or ".." in name.parts:
                    raise SmokeError("refusing a tar member outside the run directory")
                if member.size > 20 * 1024 * 1024:
                    raise SmokeError(f"refusing oversized run file {name.as_posix()}")
                target = dest / name
                target.parent.mkdir(parents=True, exist_ok=True)
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                target.write_bytes(extracted.read())
    except tarfile.TarError as exc:
        raise SmokeError(f"run archive could not be read ({exc})") from None


def generate_ssh_keypair(directory: Path) -> tuple[Path, str]:
    private = directory / "sf4_ed25519"
    try:
        subprocess.run(
            [
                "ssh-keygen",
                "-t",
                "ed25519",
                "-N",
                "",
                "-f",
                str(private),
                "-C",
                "sf4-smoke",
            ],
            check=True,
            capture_output=True,
            env=child_env(),
        )
    except subprocess.CalledProcessError:
        raise SmokeError("ssh-keygen failed") from None
    private.chmod(0o600)
    public = private.with_suffix(".pub").read_text(encoding="utf-8").strip()
    _require_public_key(public)
    return private, public


class SshRemote:
    """Copy the checkout, run the smoke, and fetch the small JSON files."""

    def __init__(
        self,
        record: dict[str, Any],
        api_key: str,
        request: RequestFn,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_key = api_key
        self.request = request
        self.sleep = sleep
        self.host, self.port = ssh_endpoint(record)
        self.tmp = tempfile.TemporaryDirectory()
        self.private, self.public = generate_ssh_keypair(Path(self.tmp.name))
        self.known_hosts = str(Path(self.tmp.name) / "known_hosts")
        self.key_id: str | None = None
        self._attach()

    def close(self) -> None:
        try:
            self._detach()
        finally:
            self.tmp.cleanup()

    def run(self, checkout: Path, local_run: Path) -> int:
        self._sync(checkout)
        code = self._bash(
            build_remote_shell(REMOTE_REPO, REMOTE_OUT), timeout_s=REMOTE_TIMEOUT_S
        )
        self._fetch(local_run)
        return code

    def _call(self, kind: str, **kwargs: Any) -> tuple[int, dict[str, Any]]:
        method, url, body = endpoint_for(kind, ALLOWED_INSTANCE_ID, **kwargs)
        return call_json(
            self.request,
            method,
            url,
            self.api_key,
            body,
            sleep=self.sleep,
        )

    def _attach(self) -> None:
        status, payload = self._call("attach_ssh", public_key=self.public)
        if status != 200 or payload.get("success") is False:
            raise SmokeError(
                f"attaching an ephemeral ssh key failed (HTTP {status}). The key is not printed."
            )
        self.key_id = ssh_key_id_from_attach(payload)
        if self.key_id is None:
            _log(
                {
                    "action": "attach_ssh",
                    "instance_id": ALLOWED_INSTANCE_ID,
                    "key_id_returned": False,
                },
                self.api_key,
            )

    def _detach(self) -> None:
        if not self.key_id:
            return
        status, payload = self._call("detach_ssh", ssh_key_id=self.key_id)
        if status not in {200, 204} or payload.get("success") is False:
            raise SmokeError(
                f"detaching the ephemeral ssh key failed (HTTP {status}). The key is not printed."
            )
        self.key_id = None

    def _sync(self, checkout: Path) -> None:
        fruits_path(checkout)
        if not (checkout / "sceneagent").is_dir():
            raise SmokeError("checkout is missing sceneagent/")
        for argv, cwd in rsync_invocations(
            checkout,
            self.host,
            self.port,
            str(self.private),
            self.known_hosts,
            REMOTE_REPO,
        ):
            _reject_secret_text(" ".join(argv))
            completed = subprocess.run(
                argv,
                check=False,
                cwd=cwd,
                capture_output=True,
                env=child_env(),
                timeout=600,
            )
            if completed.returncode != 0:
                tail = (completed.stderr or b"").decode("utf-8", errors="replace")[
                    -400:
                ]
                raise SmokeError(f"rsync failed ({completed.returncode}): {tail}")

    def _bash(self, script: str, timeout_s: float) -> int:
        argv = [
            *ssh_base_argv(str(self.private), self.known_hosts, self.port),
            f"root@{self.host}",
            "bash",
            "-s",
        ]
        _reject_secret_text(" ".join(argv))
        last_code = 255
        for attempt in range(5):
            started = time.monotonic()
            try:
                completed = subprocess.run(
                    argv,
                    check=False,
                    input=script.encode("utf-8"),
                    capture_output=True,
                    env=child_env(),
                    timeout=timeout_s,
                )
            except subprocess.TimeoutExpired:
                raise SmokeError("remote smoke timed out") from None
            last_code = int(completed.returncode)
            quick_failure = last_code == 255 and (time.monotonic() - started) < 30
            if quick_failure and attempt < 4:
                self.sleep(10)
                continue
            self._log_process(completed)
            return last_code
        return last_code

    def _fetch(self, local_run: Path) -> None:
        argv = [
            *ssh_base_argv(str(self.private), self.known_hosts, self.port),
            f"root@{self.host}",
            "bash",
            "-s",
        ]
        try:
            completed = subprocess.run(
                argv,
                check=False,
                input=build_fetch_script(REMOTE_OUT).encode("utf-8"),
                capture_output=True,
                env=child_env(),
                timeout=180,
            )
        except subprocess.TimeoutExpired:
            raise SmokeError("fetching the run files timed out") from None
        if completed.returncode != 0:
            tail = completed.stderr.decode("utf-8", errors="replace")[-400:]
            raise SmokeError(
                f"fetching the run files failed ({completed.returncode}): {tail}"
            )
        unpack_tar(completed.stdout, local_run)

    def _log_process(self, completed: subprocess.CompletedProcess[bytes]) -> None:
        stderr = completed.stderr.decode("utf-8", errors="replace")[-1000:]
        stdout = completed.stdout.decode("utf-8", errors="replace")[-1000:]
        _log(
            {
                "action": "remote",
                "returncode": completed.returncode,
                "stdout_tail": stdout,
                "stderr_tail": stderr,
            },
            self.api_key,
        )


def _show_instance(
    request: RequestFn,
    api_key: str,
    sleep: Callable[[float], None],
) -> dict[str, Any]:
    method, url, body = endpoint_for("show", ALLOWED_INSTANCE_ID)
    status, payload = call_json(request, method, url, api_key, body, sleep=sleep)
    if status == 401:
        raise InstanceNotApproved(
            "Vast API rejected VAST_API_KEY (HTTP 401). The key is not printed."
        )
    if status == 404:
        raise InstanceNotApproved(
            f"instance {ALLOWED_INSTANCE_ID} is missing (HTTP 404). Not creating a replacement."
        )
    if status != 200:
        raise SmokeError(f"show instance {ALLOWED_INSTANCE_ID} failed (HTTP {status}).")
    return require_approved_instance(payload)


def _change_state(
    kind: str,
    request: RequestFn,
    api_key: str,
    sleep: Callable[[float], None],
) -> None:
    method, url, body = endpoint_for(kind, ALLOWED_INSTANCE_ID)
    status, payload = call_json(request, method, url, api_key, body, sleep=sleep)
    if status != 200 or payload.get("success") is False:
        raise SmokeError(
            f"{kind} instance {ALLOWED_INSTANCE_ID} failed (HTTP {status})."
        )


def wait_until_running(
    request: RequestFn,
    api_key: str,
    *,
    sleep: Callable[[float], None],
    now: Callable[[], float],
    timeout_s: float,
) -> dict[str, Any]:
    """Poll show until SSH is up. Give up rather than destroy a stuck instance."""
    deadline = now() + timeout_s
    saw_loading = False
    last_status: object = None
    while True:
        row = _show_instance(request, api_key, sleep)
        kind = classify_status(row)
        last_status = row.get("actual_status")
        host_ok = isinstance(row.get("ssh_host"), str) and bool(row.get("ssh_port"))
        if kind == "running" and host_ok:
            return row
        if kind == "starting":
            saw_loading = True
        if kind == "unknown":
            raise SmokeError(
                f"instance {ALLOWED_INSTANCE_ID} status {last_status!r} will not be started. "
                "Not destroying it."
            )
        if kind == "stopped" and saw_loading:
            raise SmokeError(
                f"instance {ALLOWED_INSTANCE_ID} returned to {last_status!r} after starting. "
                "Not destroying it."
            )
        if now() >= deadline:
            raise SmokeError(
                f"instance {ALLOWED_INSTANCE_ID} did not reach running "
                f"(last actual_status {last_status!r}). Not destroying it."
            )
        sleep(POLL_SECONDS)


def run_smoke(
    instance_id: object,
    checkout: Path,
    artifact_dir: Path,
    api_key: str,
    *,
    request: RequestFn = request_json,
    remote_factory: Callable[[dict[str, Any]], Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
    boot_timeout_s: float = BOOT_TIMEOUT_S,
    handle_sigterm: bool = False,
) -> int:
    """Run the Fruits smoke on 58301075 and restore power if this call started it.

    Exit 0 is a GPU accept. Exit 2 is a refusal before the instance is changed.
    Exit 1 is a smoke that ran or started and did not pass, including a ledger
    ``reject`` or ``deferred``.
    """
    try:
        approved = parse_instance_id(instance_id)
    except InstanceNotApproved as exc:
        _emit(str(exc), "")
        return 2
    try:
        key = require_api_key(api_key)
    except InstanceNotApproved as exc:
        _emit(str(exc), "")
        return 2
    mask_secret(key)
    try:
        fruits_path(Path(checkout))
    except InstanceNotApproved as exc:
        _emit(str(exc), key)
        return 2

    started = False
    remote: Any = None
    code = 1

    def restore() -> bool:
        nonlocal started
        if not started:
            return True
        try:
            _change_state("stop", request, key, sleep)
        except (SmokeError, InstanceNotApproved) as exc:
            _emit(str(exc), key)
            return False
        started = False
        _log({"action": "stop", "instance_id": approved}, key)
        return True

    if handle_sigterm:

        def _on_term(_signum: int, _frame: object) -> None:
            restore()
            raise SystemExit(1)

        signal.signal(signal.SIGTERM, _on_term)

    try:
        row = _show_instance(request, key, sleep)
        _log({"action": "show", "instance": public_instance_view(row)}, key)
        kind = classify_status(row)
        if kind == "stopped":
            started = True
            _change_state("start", request, key, sleep)
            _log({"action": "start", "instance_id": approved}, key)
            row = wait_until_running(
                request, key, sleep=sleep, now=now, timeout_s=boot_timeout_s
            )
        elif kind == "starting":
            row = wait_until_running(
                request, key, sleep=sleep, now=now, timeout_s=boot_timeout_s
            )
        elif kind != "running":
            actual = row.get("actual_status")
            raise InstanceNotApproved(
                f"instance {approved} status {actual!r} is not running or stopped. "
                "Not starting it, and not destroying it."
            )
        elif not row.get("ssh_host") or not row.get("ssh_port"):
            row = wait_until_running(
                request, key, sleep=sleep, now=now, timeout_s=boot_timeout_s
            )
        if remote_factory is None:
            remote = SshRemote(row, key, request, sleep=sleep)
        else:
            remote = remote_factory(row)
        with tempfile.TemporaryDirectory() as tmp:
            local_run = Path(tmp) / "run"
            remote_code = int(remote.run(Path(checkout), local_run))
            manifest = collect_artifact_manifest(
                local_run,
                command=build_remote_command(REMOTE_REPO, REMOTE_OUT),
                instance_id=approved,
            )
            publish_artifacts(local_run, Path(artifact_dir))
        _log(
            {
                "action": "conclude",
                "instance_id": approved,
                "passed": manifest["passed"],
                "remote_exit": remote_code,
                "failure_reason": manifest.get("failure_reason"),
            },
            key,
        )
        code = 0 if remote_code == 0 and manifest["passed"] else 1
    except InstanceNotApproved as exc:
        _emit(str(exc), key)
        code = 1 if started else 2
    except SmokeError as exc:
        _emit(str(exc), key)
        code = 1
    finally:
        if remote is not None:
            try:
                remote.close()
            except Exception as exc:
                _emit("ssh cleanup failed: " + str(exc), key)
                code = 1
        if not restore():
            code = 1
    return code


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="SceneAgent Fruits GPU smoke on Vast instance 58301075"
    )
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args(argv)
    return run_smoke(
        args.instance_id,
        args.checkout,
        args.artifacts,
        os.environ.get("VAST_API_KEY", ""),
        handle_sigterm=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
