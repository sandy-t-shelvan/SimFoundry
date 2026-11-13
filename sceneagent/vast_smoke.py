# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manual Vast smoke for ``python -m sceneagent`` on Fruits.mp4.

The only instance this module will touch is ``60254439``. It may show that
instance when it is already running. It does not start or stop an instance.
``workflow_dispatch``, a label, and a request-file trigger all refuse a
stopped, starting, or unknown instance, and they send no state PUT. Starting
or stopping the instance takes two separate user yeses outside this workflow.
It copies the checkout with one fixed SSH private key from the Actions secret
``VAST_SSH_PRIVATE_KEY``. That key is written to a mode-0600 file on the
runner and removed on exit. The smoke does not generate, attach, or delete
a remote key, so keys already on the instance stay. SSH requires
``StrictHostKeyChecking=yes`` and one pinned known-host line in
``ci/vast-ssh-known-host``. The connection uses ``public_ipaddr`` and
``ports['22/tcp'][0].HostPort`` from the show response. The proxy
``ssh_host`` and ``ssh_port`` are not a fallback. A missing or invalid direct
endpoint, secret, or pin fails before any SSH connection, and the values are
not logged.
The fixed key stays authorized until someone removes it on the instance.
Revocation and rotation are manual. Reuse is less isolation than a per-run
key, and it removes the dependency on Vast's key attach and detach API.
Restrict the Actions secret, pin the host, and rotate the key. This module
does not add an approval loop.

It does not create, rent, resize, or destroy an instance, and the remote
command is only the SceneAgent video entry point. ``--pipeline run4``
runs ``python -m sceneagent.run4`` on the same pinned interpreter. That
path does not call
Meshy or Gemini. The API key is read from the environment and is not written
into the remote command, the URL, or the artifact files.

``python -m sceneagent`` on an mp4 writes ``plan.json`` through
``pipeline.write_plan`` and writes ``ledger.json``. The collector does not
build a plan. A missing plan fails the smoke. A ``deferred`` or ``reject``
ledger entry stays a failed smoke.

SSH is probed with a fixed attempt cap and timeout before rsync. The probe
uses the fixed key and the pinned host line. It does not scan the live host
and it does not trust a key on first use. A failed run still publishes a
redacted manifest, and a key that survives redaction publishes nothing.

The runner does not forward ``VAST_API_KEY``, ``GITHUB_TOKEN``, or any
other secret to the instance. SSH ignores user config, the child
environment drops secret variables, and rsync copies ``sceneagent`` and
``Fruits.mp4`` only, never ``.git``. The manifest records
``secrets_forwarded`` false. A trigger name, actor, and commit SHA are
recorded when the workflow passes them. They cannot change the instance id.
"""

from __future__ import annotations

import base64
import fnmatch
import hashlib
import io
import json
import os
import re
import shlex
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
from typing import Any, TypedDict

from sceneagent.simfoundry_python import SIMFOUNDRY_PYTHON

API_ORIGIN = "https://console.vast.ai"
ALLOWED_INSTANCE_ID = 60254439
# The remote script uses this file and no other interpreter. A missing or
# non-executable path exits 127. PATH is not a fallback.
REMOTE_PYTHON = SIMFOUNDRY_PYTHON
# COLMAP 3.11.1 with CUDA on 58301075 (probe run 42908214934); /usr/bin/colmap
# there is 3.6 without CUDA. Placed after the interpreter's bin on PATH, before
# everything else.
REMOTE_COLMAP_BIN = "/opt/colmap-cuda/bin"
# py7zr goes here rather than into the shared ``sa`` env.
REMOTE_PYDEPS = "/workspace/sf4-smoke/pydeps"
FRUITS_RELATIVE = "docs/assets/example_videos/Fruits.mp4"
# KAN-48 thin slice: the one video, besides Fruits, that the box receives.
MAILBOX_RELATIVE = "docs/assets/example_videos/Mailbox.mp4"
REMOTE_REPO = "/workspace/sf4-smoke/repo"
REMOTE_OUT = "/workspace/sf4-smoke/run"
# The GraspClutter6D pipelines' own run dir. Sharing REMOTE_OUT with run4 let
# the bin's COLMAP pose run4's 300 Fruits frames beside its 13 views (bin
# evidence of the 59601312 splat run 42741443604: 313 inputs, 300 registered).
REMOTE_BIN_OUT = "/workspace/sf4-smoke/bin"
REMOTE_SETUP_OUT = "/workspace/sf4-smoke/setup"
# Isaac Lab render check (KAN-41). Its own run dir and its own interpreter:
# the isaaclab env that box_setup.sh s_isaaclab installs, never the sa env.
REMOTE_ISAACLAB_OUT = "/workspace/sf4-smoke/isaaclab"
# The Mailbox slice's own run dir, so the bin's results are never overwritten.
REMOTE_MAILBOX_OUT = "/workspace/sf4-smoke/mailbox"
# KAN-48 / KAN-52: each example video the box may run through video_run,
# by pipeline name. Each has its own run dir (remote_out). PutCupInBowl is
# the development video; PutMarkerInCup is kept for the MVP acceptance run
# and is not tuned on.
VIDEO_PIPELINES = {
    "mailbox": MAILBOX_RELATIVE,
    "putcupinbowl": "docs/assets/example_videos/PutCupInBowl.mp4",
    "putmarkerincup": "docs/assets/example_videos/PutMarkerInCup.mp4",
}
ISAACLAB_PYTHON = "/root/.local/share/mamba/envs/isaaclab/bin/python3"
# The setup stages run detached on the box (setsid nohup), so a dropped
# runner does not stop them; each run waits this long, then reports what is
# done and what is running. A COLMAP CUDA build plus four envs takes longer
# than one run, and the next dispatch resumes (each stage is skipped once it
# has succeeded). 30 minutes, not 100: a stalled stage shows up sooner
# (run 43107275324 sat idle with no way to see why).
SETUP_WAIT_S = 1800
# Where the runner writes the Hugging Face token for the `setup` pipeline only
# (Sandeep, 2026-11-11: send it so SAM 3D's gated weights can be fetched).
# It travels on ssh stdin, never in an argv, an environment or a log.
REMOTE_HF_TOKEN = "/root/.cache/huggingface/token"
HF_TOKEN_ENV = "HF_TOKEN"
BOOT_TIMEOUT_S = 600.0
# Room for run4.STEP_TIMEOUT_S["mapper"] (5400 s) inside the 120 min job.
REMOTE_TIMEOUT_S = 6600.0
POLL_SECONDS = 15.0
# Run 42162484354 was refused ~0.7s after attach. Vast's banner says the key
# can take a few seconds. The cap and the deadline both stop the loop.
SSH_READY_MAX_ATTEMPTS = 6
SSH_READY_TIMEOUT_S = 45.0
SSH_READY_BACKOFF_S = 2.0
SSH_READY_BACKOFF_CAP_S = 8.0
# Remote authorized keys are not deleted. The fixed key stays until an
# operator removes it. This flag does not authorize a detach.
DETACH_LEFTOVER_KEYS = False
REQUIRED_LEDGER_ITEMS = ("video", "COLMAP camera poses")
# What each pipeline must put in its ledger, and the input it records.
PIPELINES = {
    "sceneagent": (REQUIRED_LEDGER_ITEMS, FRUITS_RELATIVE),
    "run4": (REQUIRED_LEDGER_ITEMS, FRUITS_RELATIVE),
    "bin": (("benchmark scene", "COLMAP camera poses"), "GraspClutter6D 000091"),
    "splat": (
        ("benchmark scene", "COLMAP camera poses", "full-scene splat"),
        "GraspClutter6D 000091",
    ),
    "masks": (("benchmark scene", "SAM masks"), "GraspClutter6D 000091"),
    "association": (
        (
            "benchmark scene",
            "COLMAP camera poses",
            "full-scene splat",
            "SAM masks",
            "Gaussian association",
        ),
        "GraspClutter6D 000091",
    ),
    "sam3d": (
        (
            "benchmark scene",
            "COLMAP camera poses",
            "full-scene splat",
            "SAM masks",
            "Gaussian association",
            "SAM 3D meshes",
        ),
        "GraspClutter6D 000091",
    ),
    "place": (
        (
            "benchmark scene",
            "COLMAP camera poses",
            "full-scene splat",
            "SAM masks",
            "Gaussian association",
            "SAM 3D meshes",
            "ICP residuals",
            "bin score",
        ),
        "GraspClutter6D 000091",
    ),
    "review": (
        (
            "benchmark scene",
            "SAM 3D meshes",
            "reviewer views",
            "reviewer retries",
            "bin score (reviewed)",
        ),
        "GraspClutter6D 000091",
    ),
    "review_vlm": (
        (
            "benchmark scene",
            "SAM 3D meshes",
            "reviewer views",
            "reviewer retries (VLM)",
            "bin score (VLM reviewed)",
        ),
        "GraspClutter6D 000091",
    ),
    "review_candidates": (
        ("benchmark scene", "SAM 3D meshes", "review candidates", "review pairs drawn"),
        "GraspClutter6D 000091",
    ),
    "review_accept": (
        (
            "benchmark scene",
            "SAM 3D meshes",
            "reviewer retries (VLM accepted)",
            "bin score (VLM accepted)",
        ),
        "GraspClutter6D 000091",
    ),
    "bin_settle": (
        (
            "benchmark scene",
            "SAM 3D meshes",
            "ICP residuals",
            "USD physics export",
            "Isaac Lab settle",
        ),
        "GraspClutter6D 000091",
    ),
    "mapper": (("mapper comparison",), "GraspClutter6D 000091 and Fruits frames"),
    "probe": ((), "host inventory"),
    "setup": (("box setup",), "GPU stack install (sceneagent/box_setup.sh)"),
    "isaaclab_render": (("Isaac Lab render",), "Isaac Lab cube on a ground plane"),
    "articulate": (("articulated motion",), "proposed joints"),
    **{
        name: (
            (
                "video",
                "COLMAP camera poses",
                "full-scene splat",
                "SAM masks",
                "Gaussian association",
                "SAM 3D meshes",
                "ICP residuals",
                "fitted splat prior input",
                "§3.4 properties",
                "USD physics export",
                "Isaac Lab settle",
                "inspection views",
            ),
            relative,
        )
        for name, relative in VIDEO_PIPELINES.items()
    },
}
# GraspClutter6D pipelines. They share one ledger in REMOTE_BIN_OUT, so each is
# judged on its own items only (a masks run must not fail on an earlier splat
# entry), and they are the only ones that may fetch the dataset with py7zr.
BIN_PIPELINES = frozenset(
    {
        "bin",
        "splat",
        "masks",
        "association",
        "sam3d",
        "place",
        "review",
        "review_vlm",
        "review_candidates",
        "review_accept",
        "bin_settle",
        "mapper",
    }
)
# The VLM verdicts reach the box as a committed file in the shipped checkout:
# sceneagent.vlm_review writes it on a runner that holds GEMINI_API_KEY, and the
# box never sees that key. rsync ships only sceneagent/ and Fruits.mp4, so the
# file lives under sceneagent/ (run 42941668034 looked for ci/ and found
# nothing); widening what is shipped would loosen that boundary.
VLM_VERDICTS_RELATIVE = "sceneagent/verdicts/vlm_verdicts.json"
# Round 2's pair verdicts travel the same way.
ACCEPT_VERDICTS_RELATIVE = "sceneagent/verdicts/accept_verdicts.json"


def remote_out(pipeline: str) -> str:
    """The run dir on the box: the bin pipelines' own, else run4's."""
    if pipeline == "setup":
        return REMOTE_SETUP_OUT
    if pipeline == "isaaclab_render":
        return REMOTE_ISAACLAB_OUT
    if pipeline == "articulate":
        return "/workspace/sf4-smoke/articulate"
    if pipeline == "mailbox":
        return REMOTE_MAILBOX_OUT
    if pipeline in VIDEO_PIPELINES:
        return f"/workspace/sf4-smoke/{pipeline}"
    return REMOTE_BIN_OUT if pipeline in BIN_PIPELINES else REMOTE_OUT


# Read-only host inventory for a new or changed box. Plain shell, so it runs
# before any SceneAgent environment exists. It never prints the environment
# (no env/printenv), so no secret can land in the report.
PROBE_SCRIPT = r"""
section() { printf '\n## %s\n' "$1"; }
# The inventory is uploaded as an Actions artifact, readable beyond the operator.
# It lists tools, envs and model files only: no folder trees (one exposed a
# file named after a local user path), no file contents, and any name that
# looks like a credential is dropped.
public() { grep -viE 'key|token|secret|cred|passw|\.pem|netrc|id_rsa|id_ed25519|\.env'; }
section date; date -u
section gpu; nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv 2>&1
section os; head -3 /etc/os-release 2>&1; uname -r
section cpu_ram; nproc; free -g 2>&1
section disk; df -h / /workspace /root 2>&1
section cuda; ls -d /usr/local/cuda* 2>&1; nvcc --version 2>&1 | tail -2
section python; for t in python python3 conda mamba micromamba; do printf '%s: ' "$t"; command -v "$t" || echo missing; done; python3 --version 2>&1
section conda_envs; for d in /root/miniforge3 /root/miniconda3 /opt/conda /root/anaconda3; do [ -d "$d/envs" ] && { echo "$d/envs:"; ls "$d/envs"; }; done; [ -x /root/miniforge3/envs/simfoundry/bin/python3 ] && echo "simfoundry env: present" || echo "simfoundry env: missing"
section tools; for t in colmap colmap4 glomap ffmpeg ffprobe ns-train ns-eval git git-lfs rsync tmux 7z; do printf '%s: ' "$t"; command -v "$t" || echo missing; done
section colmap; colmap -h 2>&1 | head -3
section torch; python3 -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())' 2>&1 | tail -1
# Globs over the usual conda and mamba roots, not find /: a full-disk walk takes
# minutes on a hosted CI runner, where the probe test also runs.
section interpreters; for d in /usr /opt/* /opt/conda/envs/* /root/.local/share/mamba/envs/* /root/miniforge3/envs/* /root/miniconda3/envs/*; do [ -x "$d/bin/python3" ] && echo "$d/bin/python3"; done
section colmap_binaries; for d in /usr /usr/local /opt/* /opt/conda/envs/* /root/.local/share/mamba/envs/* /root/miniforge3/envs/* /root/miniconda3/envs/*; do c="$d/bin/colmap"; [ -x "$c" ] && echo "$c: $(timeout 20 "$c" -h 2>&1 | head -2 | tr "\n" " ")"; done
section python_imports; for py in /usr/bin/python3 /opt/*/bin/python3 /opt/conda/envs/*/bin/python3 /root/.local/share/mamba/envs/*/bin/python3 /root/miniforge3/envs/*/bin/python3 /root/miniconda3/envs/*/bin/python3; do [ -x "$py" ] || continue; echo "$py: $(timeout 60 "$py" -c 'import sys; v=[sys.version.split()[0]]
for m in ("numpy","scipy","trimesh","PIL","py7zr","torch"):
    try: v.append(m+"="+__import__(m).__version__)
    except Exception as e: v.append(m+"=MISSING")
print(" ".join(v))' 2>&1 | tail -1)"; done
section phase2_packages; for py in /opt/conda/bin/python3 /root/.local/share/mamba/envs/*/bin/python3 /root/miniforge3/envs/*/bin/python3; do [ -x "$py" ] || continue; echo "$py:"; timeout 120 "$py" -m pip list 2>/dev/null | grep -iE '^(nerfstudio|gsplat|groundingdino|segment|sam-2|sam2|transformers|supervision|torch |opencv)' ; ls "$(dirname "$py")" 2>/dev/null | grep -E '^ns-(train|eval|export|process-data)$' | tr '\n' ' '; echo; done
section checkpoints; find -L /root/SceneagentX/third_party /root/.cache/huggingface/hub /root/.cache/torch /workspace/models /root/sam-3d-objects/checkpoints /root/SimFoundry/checkpoints /root/miniforge3/envs/groundingdino -maxdepth 4 \( -name '*.pth' -o -name '*.pt' -o -name '*.safetensors' -o -name '*.ckpt' \) -size +1M 2>/dev/null | public | head -60
section third_party; ls /root/SceneagentX/third_party /root/.cache/huggingface/hub 2>&1 | public | head -60
section sam3d; for S in /root/software/sam-3d-objects /root/sam-3d-objects /root/SceneagentX/third_party/sam-3d-objects; do [ -d "$S" ] || continue; echo "root $S:"; ls "$S" "$S/notebook" 2>&1 | public | head -40; done; find -L /root/.cache/huggingface/hub/models--facebook--sam-3d-objects -maxdepth 6 -type f 2>/dev/null | public | head -40
# Every SAM 3D interpreter against every checkout's notebook: which pair imports.
section sam3d_import; for P in /root/miniforge3/envs/sam3d-objects/bin/python3 /opt/conda/envs/sam3d/bin/python3 /opt/conda/envs/sam3d10/bin/python3; do [ -x "$P" ] || continue; echo "interpreter $P:"; timeout 120 "$P" -m pip list 2>/dev/null | grep -iE '^(sam3d|sam-3d|pytorch3d|kaolin|gsplat|utils3d|moge|open3d|spconv|xformers|flash)'; for S in /root/software/sam-3d-objects /root/sam-3d-objects /root/SceneagentX/third_party/sam-3d-objects; do [ -d "$S/notebook" ] || continue; echo "notebook $S:"; (cd "$S/notebook" && timeout 300 "$P" -c 'import inference, inspect; print("inference import ok"); print(inspect.signature(inference.Inference.__init__))' 2>&1 | public | tail -3); done; done; for S in /root/software/sam-3d-objects /root/sam-3d-objects /root/SceneagentX/third_party/sam-3d-objects; do ls "$S"/checkpoints/*/pipeline.yaml 2>/dev/null; done
# Depth Anything 3: which interpreter imports it, and any source checkout.
section da3; for py in /opt/conda/bin/python3 /opt/conda/envs/*/bin/python3 /root/.local/share/mamba/envs/*/bin/python3 /root/miniforge3/envs/*/bin/python3; do [ -x "$py" ] || continue; r=$(timeout 120 "$py" -c 'import depth_anything_3 as d; print(list(d.__path__))' 2>/dev/null | tail -1); [ -n "$r" ] && echo "$py: $r"; done; ls -d /root/Depth-Anything-3/src /root/*/Depth-Anything-3/src /root/SceneagentX/third_party/Depth-Anything-3/src /workspace/Depth-Anything-3/src /workspace/*/Depth-Anything-3/src 2>/dev/null | public
# Grounded-SAM-2's GroundingDINO imports itself as grounding_dino.*, so it needs its checkout root on the path.
section gdino; G=/root/SceneagentX/third_party/Grounded-SAM-2; P=/root/.local/share/mamba/envs/gsam2/bin/python3; [ -x "$P" ] && (cd /tmp && PYTHONPATH="$G:/workspace/sf4-smoke/pydeps-gsam2" timeout 300 "$P" -c 'from groundingdino.util.inference import load_model; print("gdino import ok"); import segment_anything; print("segment_anything ok")' 2>&1 | public | tail -2)
"""
# gc6d_fetch needs py7zr; installed only when the pinned env lacks it.
PY7ZR_PIN = "py7zr==1.1.4"
FETCH_NAMES = (
    "ledger.json",
    "plan.json",
    "manifest.json",
    "video.json",
    "colmap_poses.json",
    "tool_probe.json",
    "host_probe.json",
    "run4_evidence.json",
    "bin_evidence.json",
    "splat_evidence.json",
    "ns_eval.json",
    "masks_evidence.json",
    "association_evidence.json",
    "sam3d_evidence.json",
    "bin_score.json",
    "bin_review.json",
    "bin_review_vlm.json",
    "review_sheets.zip",
    "candidates.json",
    "review_pairs.zip",
    "bin_review_accept.json",
    "mapper_compare.json",
    "inventory.txt",
    "setup_evidence.json",
    "isaaclab_render.png",
    "isaaclab_render.log",
    "video_run.json",
    "scene.json",
    "export_evidence.json",
    "settle.json",
    "settle_frame.png",
    "inspection_sheet.png",
    "inspection_views.zip",
    "settle_summary.json",
    "prior_input_ledger.json",
    "articulate_frames.zip",
    "articulate_proposal.json",
    "vlm_check.json",
)
# The fetched files that become the workflow artifact. One list, so a new
# evidence file cannot be fetched and then silently dropped (run 42942342408
# fetched bin_review_vlm.json and published nothing of it). The workflow's
# upload list must name the same files; a test checks both.
PUBLISHED_NAMES = frozenset(
    {
        "plan.json",
        "ledger.json",
        "manifest.json",
        "run4_evidence.json",
        "bin_evidence.json",
        "splat_evidence.json",
        "ns_eval.json",
        "masks_evidence.json",
        "association_evidence.json",
        "sam3d_evidence.json",
        "bin_score.json",
        "bin_review.json",
        "bin_review_vlm.json",
        "review_sheets.zip",
        "candidates.json",
        "review_pairs.zip",
        "bin_review_accept.json",
        "mapper_compare.json",
        "inventory.txt",
        "setup_evidence.json",
        "isaaclab_render.png",
        "isaaclab_render.log",
        "video_run.json",
        "scene.json",
        "export_evidence.json",
        "settle.json",
        "settle_frame.png",
        "inspection_sheet.png",
        "inspection_views.zip",
        "settle_summary.json",
        "prior_input_ledger.json",
        "articulate_frames.zip",
        "articulate_proposal.json",
        "vlm_check.json",
    }
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
    """The call is outside the SF-4 allowlist for instance 60254439."""


class SmokeError(RuntimeError):
    """The smoke could not finish. The ledger was not rewritten to hide it."""


def parse_instance_id(value: object) -> int:
    """Return 60254439. Any other id, including 58301075, is refused."""
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
            f"SF-4 may use only the existing United Kingdom instance {ALLOWED_INSTANCE_ID}."
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
    """Register the key with GitHub's log masker. No-op outside Actions."""
    if os.environ.get("GITHUB_ACTIONS") == "true" and value:
        print(f"::add-mask::{value}", flush=True)


SSH_PRIVATE_KEY_ENV = "VAST_SSH_PRIVATE_KEY"
KNOWN_HOST_PATH = Path(__file__).resolve().parents[1] / "ci" / "vast-ssh-known-host"
_KNOWN_HOST_RE = re.compile(
    r"^(?:\[(?P<bracket>[A-Za-z0-9.-]+)\]:(?P<port>\d+)|(?P<plain>[A-Za-z0-9.-]+))"
    r"\s+(?:ssh-ed25519|ssh-rsa)\s+[A-Za-z0-9+/=]+(?:\s+\S+)?$"
)
_SECRET_ENV_EXACT = {
    "SSH_AUTH_SOCK",
    "SSH_AGENT_PID",
    "GIT_ASKPASS",
    SSH_PRIVATE_KEY_ENV,
}
_SECRET_ENV_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "API_KEY", "APIKEY")
_RSYNC_EXCLUDES = (
    ".git",
    ".github",
    "__pycache__",
    ".env",
    ".env.*",
    "*.pem",
    "*credential*",
    "*secret*",
    "id_rsa",
    "id_ed25519",
)
_TRIGGER_NAMES = {"dispatch", "label", "request-file"}
_ACTOR_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,38}(?:\[bot\])?")
_COMMIT_SHA_RE = re.compile(r"[0-9a-f]{40}")


class SecretForwarding(TypedDict):
    """What the manifest records about secrets and the GPU."""

    secrets_forwarded: bool
    ssh_env_forwarded: bool
    git_credentials_shipped: bool
    payload_secret_hits: int


def _secret_env_name(name: str) -> bool:
    if name in _SECRET_ENV_EXACT:
        return True
    upper = name.upper()
    return any(marker in upper for marker in _SECRET_ENV_MARKERS)


def child_env() -> dict[str, str]:
    """Environment for ssh and rsync. Secrets stay on the runner.

    GitHub injects ``GITHUB_TOKEN`` into every step. OpenSSH does not send
    the process environment unless config says so, and this argv does not.
    Dropping the names here means a later ``SendEnv`` still has nothing to send.
    """
    return {
        key: value for key, value in os.environ.items() if not _secret_env_name(key)
    }


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
    """Return the instance when it is 60254439 in the United Kingdom."""
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
    if "united kingdom" not in geo_text.lower():
        raise InstanceNotApproved(
            f"instance {ALLOWED_INSTANCE_ID} geolocation is {geo_text!r}, not the United Kingdom."
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

    ``stopped`` and ``exited`` are the stopped kind. This module does not
    start them, and ``workflow_dispatch`` does not either. ``exited`` is what
    the Vast listing showed for the previous machine when it was not running.
    ``offline`` and ``unknown`` are not started: the API docs say those
    states do not become ``running``, and this module does not destroy the
    instance to replace it.
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
    """Reject every Vast call except show and the pinned ssh list.

    Attach, detach, and a PUT that sets ``state`` are refused. This module
    does not generate a remote key or delete one.
    """
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
    # One added read: GET /api/v0/instances/60254439/ssh/ and nothing else.
    # The id is the constant, not a caller-supplied template. No query, no
    # body, no sub-path, and no other method on this path. No state PUT.
    if (
        method == "GET"
        and parsed.path == f"/api/v0/instances/{ALLOWED_INSTANCE_ID}/ssh/"
        and key_id is None
        and not query
        and body is None
    ):
        return
    raise InstanceNotApproved(f"refusing Vast {method} {parsed.path}")


def guard_own_key_delete(method: str, url: str) -> None:
    """Refuse every DELETE. This module does not remove remote ssh keys."""
    if method.upper() != "DELETE":
        return
    path = urllib.parse.urlsplit(url).path
    raise InstanceNotApproved(f"refusing to delete a remote ssh key at {path}")


def endpoint_for(
    kind: str,
    instance_id: object,
) -> tuple[str, str, dict[str, Any] | None]:
    """Build one allowlisted request. Any other kind raises."""
    approved = parse_instance_id(instance_id)
    base = f"{API_ORIGIN}/api/v0/instances/{approved}/"
    if kind == "show":
        method, url, body = "GET", base + "?owner=me", None
    elif kind == "show_ssh_keys":
        method, url, body = "GET", base + "ssh/", None
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
    guard_own_key_delete(method, url)
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
    guard_own_key_delete(method, url)
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


def build_remote_command(
    repo_dir: str,
    out_dir: str,
    pipeline: str = "sceneagent",
    commit_sha: str = "",
) -> list[str]:
    """Argv for the Fruits entry point. No API key and no paid client.

    ``sceneagent`` is the 1 fps video path. ``run4`` is ``sceneagent.run4``.
    A measured checkout SHA is appended only for run4. The default command
    stays free of that flag so an empty SHA is not invented.
    """
    repo = require_remote_dir(repo_dir)
    out = require_remote_dir(out_dir)
    fruits = f"{repo}/{FRUITS_RELATIVE}"
    if pipeline == "sceneagent":
        return [REMOTE_PYTHON, "-m", "sceneagent", fruits, "--out", out]
    if pipeline == "bin":
        return [REMOTE_PYTHON, "-m", "sceneagent.bin_run", "--out", out]
    if pipeline == "splat":
        return [REMOTE_PYTHON, "-m", "sceneagent.bin_splat", "--out", out]
    if pipeline == "masks":
        return [REMOTE_PYTHON, "-m", "sceneagent.bin_masks", "--out", out]
    if pipeline == "association":
        return [REMOTE_PYTHON, "-m", "sceneagent.bin_association", "--out", out]
    if pipeline == "sam3d":
        return [REMOTE_PYTHON, "-m", "sceneagent.bin_sam3d", "--out", out]
    if pipeline == "place":
        return [REMOTE_PYTHON, "-m", "sceneagent.bin_place", "--out", out]
    if pipeline == "review":
        return [REMOTE_PYTHON, "-m", "sceneagent.bin_review", "--out", out]
    if pipeline == "review_vlm":
        verdicts = f"{repo}/{VLM_VERDICTS_RELATIVE}"
        return [
            REMOTE_PYTHON,
            "-m",
            "sceneagent.bin_review",
            "--out",
            out,
            "--verdicts",
            verdicts,
        ]
    if pipeline == "review_candidates":
        return [
            REMOTE_PYTHON,
            "-m",
            "sceneagent.bin_review_pairs",
            "build",
            "--out",
            out,
            "--verdicts",
            f"{repo}/{VLM_VERDICTS_RELATIVE}",
        ]
    if pipeline == "review_accept":
        return [
            REMOTE_PYTHON,
            "-m",
            "sceneagent.bin_review_pairs",
            "apply",
            "--out",
            out,
            "--accept",
            f"{repo}/{ACCEPT_VERDICTS_RELATIVE}",
        ]
    if pipeline == "bin_settle":
        return [REMOTE_PYTHON, "-m", "sceneagent.bin_settle", "--out", out]
    if pipeline == "mapper":
        return [REMOTE_PYTHON, "-m", "sceneagent.mapper_compare", "--out", out]
    if pipeline == "probe":
        return ["/bin/sh", "-c", "host inventory", "--out", out]
    if pipeline == "setup":
        return ["/bin/bash", f"{repo}/sceneagent/box_setup.sh", "run", out]
    if pipeline in VIDEO_PIPELINES:
        return [
            REMOTE_PYTHON,
            "-m",
            "sceneagent.video_run",
            f"{repo}/{VIDEO_PIPELINES[pipeline]}",
            "--out",
            out,
        ]
    if pipeline == "isaaclab_render":
        # The repo on PYTHONPATH must not shadow packages in the isaaclab env.
        return [
            "env",
            "-u",
            "PYTHONPATH",
            ISAACLAB_PYTHON,
            f"{repo}/sceneagent/isaaclab_headless_check.py",
            "--render",
            out,
        ]
    if pipeline == "articulate":
        return [
            "env",
            "-u",
            "PYTHONPATH",
            ISAACLAB_PYTHON,
            f"{repo}/sceneagent/articulate_drive.py",
            "--proposal",
            f"{repo}/sceneagent/articulate_proposal.json",
            "--out",
            out,
        ]
    if pipeline != "run4":
        raise InstanceNotApproved("refusing pipeline")
    argv = [REMOTE_PYTHON, "-m", "sceneagent.run4", fruits, "--out", out]
    if commit_sha:
        if re.fullmatch(r"[0-9a-f]{40}", commit_sha) is None:
            raise InstanceNotApproved("refusing commit sha")
        argv.extend(["--commit-sha", commit_sha])
    return argv


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
    return f"{shlex.quote(REMOTE_PYTHON)} -c {shlex.quote(code)}"


def build_remote_shell(
    repo_dir: str,
    out_dir: str,
    pipeline: str = "sceneagent",
    commit_sha: str = "",
) -> str:
    """Bash script that runs the video entry point with the pinned interpreter.

    ``REMOTE_PYTHON`` is the only Python this script runs. If that file is
    missing or not executable, the script prints why and exits 127. It does
    not search ``PATH`` and it does not fall back to ``python`` or ``python3``.
    ``pipeline`` selects ``sceneagent`` or ``sceneagent.run4``.
    """
    repo = require_remote_dir(repo_dir)
    out = require_remote_dir(out_dir)
    if pipeline == "probe":
        # No pinned interpreter yet on a new box; the probe is plain shell.
        script = (
            f"mkdir -p {shlex.quote(out)}\n{{\n{PROBE_SCRIPT}\n}} > "
            f"{shlex.quote(out + '/inventory.txt')} 2>&1\necho inventory written\n"
        )
        _reject_secret_text(script)
        return script
    if pipeline == "setup":
        # Plain shell: no pinned interpreter exists until setup makes one.
        setup = shlex.quote(repo + "/sceneagent/box_setup.sh")
        q_out = shlex.quote(out)
        script = "\n".join(
            [
                "set -u",
                f"mkdir -p {q_out} /workspace/setup/state",
                # A finished earlier run must not end this wait at once.
                "rm -f /workspace/setup/state/done",
                (
                    f"setsid nohup bash {setup} run {q_out} "
                    "> /workspace/setup/run.log 2>&1 < /dev/null &"
                ),
                "waited=0",
                (
                    "while [ ! -f /workspace/setup/state/done ] && "
                    f"[ $waited -lt {SETUP_WAIT_S} ]; do sleep 30; "
                    "waited=$((waited + 30)); done"
                ),
                f"bash {setup} report {q_out}",
            ]
        )
        script += "\n"
        _reject_secret_text(script)
        return script
    argv = build_remote_command(repo, out, pipeline=pipeline, commit_sha=commit_sha)
    quoted = " ".join(shlex.quote(part) for part in argv)
    # Only dataset pipelines see py7zr's install dir; its dependencies would
    # otherwise shadow same-named packages in the pinned env.
    pythonpath = repo + (":" + REMOTE_PYDEPS if pipeline in BIN_PIPELINES else "")
    py = shlex.quote(REMOTE_PYTHON)
    missing = shlex.quote("pinned interpreter missing: " + REMOTE_PYTHON)
    lines = [
        "set -eu",
        f"if [ ! -x {py} ]; then",
        f"  printf '%s\\n' {missing} >&2",
        "  exit 127",
        "fi",
        f"cd {shlex.quote(repo)}",
        f"export PYTHONPATH={shlex.quote(pythonpath)}",
        f'export PATH={shlex.quote(str(Path(REMOTE_PYTHON).parent) + ":" + REMOTE_COLMAP_BIN)}:"$PATH"',
        f"mkdir -p {shlex.quote(out)}",
        # Only a probe run may publish an inventory; never fetch a stale one.
        f"rm -f {shlex.quote(out + '/inventory.txt')}",
        host_probe_command(out),
    ]
    if pipeline in BIN_PIPELINES:
        lines.append(
            f"{py} -c 'import py7zr' 2>/dev/null || {py} -m pip install --quiet "
            f"--target {shlex.quote(REMOTE_PYDEPS)} {PY7ZR_PIN}"
        )
    lines += [
        quoted,
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
    """SSH argv that does not forward the runner environment or an agent.

    ``-F /dev/null`` ignores ssh_config, which is where ``SendEnv`` and
    ``SetEnv`` would otherwise be picked up. The default is to send neither.
    """
    return [
        "ssh",
        "-F",
        "/dev/null",
        "-i",
        key_path,
        "-p",
        str(port),
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "ForwardAgent=no",
        "-o",
        "ClearAllForwardings=yes",
        "-o",
        "ConnectTimeout=20",
    ]


def ssh_forwarding_problems(argv: list[str]) -> list[str]:
    """Return reasons this argv would forward a secret. Values are not included."""
    problems: list[str] = []
    if argv.count("-F") != 1 or "/dev/null" not in argv:
        problems.append("ssh reads config that could set SendEnv")
    if "ForwardAgent=no" not in argv:
        problems.append("ForwardAgent is not disabled")
    if "ClearAllForwardings=yes" not in argv:
        problems.append("ssh forwardings are not cleared")
    for item in argv:
        if item.startswith("SendEnv"):
            problems.append("ssh SendEnv is set")
        if item.startswith("SetEnv"):
            problems.append("ssh SetEnv is set")
    joined = " ".join(argv)
    if "VAST_API_KEY" in joined or "GITHUB_TOKEN" in joined:
        problems.append("ssh argv names a secret")
    return problems


def ssh_endpoint(record: dict[str, Any]) -> tuple[str, int]:
    """Direct address from the show row. The proxy fields are not used.

    ``public_ipaddr`` is the host. ``ports['22/tcp'][0].HostPort`` is the
    port. A missing or invalid value refuses the connection. ``ssh_host`` and
    ``ssh_port`` are not a fallback.
    """
    host = record.get("public_ipaddr")
    if not isinstance(host, str) or _HOST_RE.fullmatch(host) is None:
        raise SmokeError(
            f"instance {ALLOWED_INSTANCE_ID} direct ssh public_ipaddr is missing."
        )
    ports = record.get("ports")
    mapping = ports.get("22/tcp") if isinstance(ports, dict) else None
    first = mapping[0] if isinstance(mapping, list) and mapping else None
    raw_port = first.get("HostPort") if isinstance(first, dict) else None
    if raw_port is None or not isinstance(mapping, list) or not mapping:
        raise SmokeError(
            f"instance {ALLOWED_INSTANCE_ID} direct ssh port mapping 22/tcp is missing."
        )
    if isinstance(raw_port, str) and raw_port.isdigit():
        port = int(raw_port)
    elif isinstance(raw_port, int) and not isinstance(raw_port, bool):
        port = raw_port
    else:
        port = 0
    if port <= 0 or port > 65535:
        raise SmokeError(
            f"instance {ALLOWED_INSTANCE_ID} direct ssh HostPort is invalid."
        )
    return host, port


def rsync_invocations(
    checkout: Path,
    host: str,
    port: int,
    key_path: str,
    known_hosts: str,
    repo_dir: str,
) -> list[tuple[list[str], Path]]:
    """Copy ``sceneagent``, Fruits.mp4 and the VIDEO_PIPELINES videos. The rest stays home."""
    repo = require_remote_dir(repo_dir)
    remote = f"root@{host}:{repo}/"
    ssh_e = shlex.join(ssh_base_argv(key_path, known_hosts, port))
    scene = checkout / "sceneagent"
    excludes = _rsync_exclude_args()
    # A fresh box has no /workspace/sf4-smoke, and rsync creates only the last
    # path component (run4, 2026-11-07). --mkpath needs rsync 3.2.3 on both
    # ends and instance 58301075 had 3.1.3, so the remote side runs mkdir -p
    # first; every rsync version accepts --rsync-path. ``repo`` is a checked
    # absolute path with no shell metacharacters (require_remote_dir).
    make_parent = f"--rsync-path=mkdir -p {repo} && rsync"
    return [
        (
            [
                "rsync",
                "-a",
                make_parent,
                *excludes,
                "-e",
                ssh_e,
                str(scene),
                remote,
            ],
            checkout,
        ),
        (
            [
                "rsync",
                "-a",
                "-R",
                make_parent,
                *excludes,
                "-e",
                ssh_e,
                FRUITS_RELATIVE,
                *[m for m in VIDEO_PIPELINES.values() if (checkout / m).is_file()],
                remote,
            ],
            checkout,
        ),
    ]


def _rsync_exclude_args() -> list[str]:
    args: list[str] = []
    for pattern in _RSYNC_EXCLUDES:
        args.extend(["--exclude", pattern])
    return args


def _ship_excluded(relative: str) -> bool:
    parts = tuple(part for part in relative.split("/") if part not in {"", "."})
    for pattern in _RSYNC_EXCLUDES:
        if fnmatch.fnmatch(relative, pattern):
            return True
        for part in parts:
            if fnmatch.fnmatch(part, pattern):
                return True
    return False


def iter_shipped_files(checkout: Path) -> list[Path]:
    """Files rsync would copy. ``.git`` and credential names are not included."""
    root = Path(checkout)
    shipped: list[Path] = []
    scene = root / "sceneagent"
    if scene.is_dir():
        for path in sorted(item for item in scene.rglob("*") if item.is_file()):
            relative = path.relative_to(scene).as_posix()
            if _ship_excluded(relative):
                continue
            shipped.append(path)
    for relative in (FRUITS_RELATIVE, *VIDEO_PIPELINES.values()):
        video = root / relative
        if video.is_file() and not _ship_excluded(relative):
            shipped.append(video)
    return shipped


def assess_secret_forwarding(
    checkout: Path, api_key: str
) -> tuple[SecretForwarding, list[str]]:
    """Fail closed before a network call when a secret would reach the GPU.

    Problem strings name the check. They do not contain the secret value.
    """
    problems: list[str] = []
    env = child_env()
    leaked = sorted(name for name in env if _secret_env_name(name))
    if leaked:
        problems.append("ssh environment still has " + ", ".join(leaked))
    problems.extend(ssh_forwarding_problems(ssh_base_argv("key", "known_hosts", 22)))
    secret_values = [api_key] if api_key else []
    for name in ("GITHUB_TOKEN", "GH_TOKEN", "VAST_API_KEY", SSH_PRIVATE_KEY_ENV):
        value = os.environ.get(name, "")
        if value and value not in secret_values:
            secret_values.append(value)
    hits = 0
    git_shipped = False
    for path in iter_shipped_files(checkout):
        relative = path.relative_to(checkout).as_posix()
        if ".git" in path.relative_to(checkout).parts:
            git_shipped = True
            problems.append(f"git metadata would be shipped ({relative})")
            continue
        if path.stat().st_size > 20 * 1024 * 1024:
            problems.append(f"shipped file is too large to scan ({relative})")
            hits += 1
            continue
        data = path.read_bytes()
        if any(secret.encode("utf-8") in data for secret in secret_values if secret):
            problems.append(f"a secret value is in a shipped file ({relative})")
            hits += 1
    record: SecretForwarding = {
        "secrets_forwarded": bool(problems),
        "ssh_env_forwarded": any(
            "ssh environment" in item
            or "SendEnv" in item
            or "ForwardAgent" in item
            or "SetEnv" in item
            or "ssh reads config" in item
            for item in problems
        ),
        "git_credentials_shipped": git_shipped,
        "payload_secret_hits": hits,
    }
    return record, problems


def _path_names_payload(value: str) -> bool:
    """True when a path component is exactly ``payload``.

    ``Path`` keeps ``payload`` as a component of ``./payload`` and
    ``payload/``. It does not treat ``.ci`` as ``ci``.
    """
    if value in {"payload", "./payload", "payload/", "./payload/"}:
        return True
    return "payload" in Path(value).parts


def payload_import_blockers() -> list[str]:
    """Ways this process could import the PR-head ``payload/`` tree.

    The keyed step must run the base orchestrator. A ``PYTHONPATH``,
    ``PYTHONSTARTUP``, ``sys.path`` entry, ``.pth`` line, or ``sitecustomize``
    that names ``payload`` is refused before any instance call.
    """
    blockers: list[str] = []
    if Path.cwd().name == "payload":
        blockers.append("cwd is payload")
    for key in ("PYTHONPATH", "PYTHONSTARTUP"):
        raw = os.environ.get(key, "")
        pieces = raw.split(os.pathsep) if key == "PYTHONPATH" else [raw]
        for piece in pieces:
            if piece and _path_names_payload(piece):
                blockers.append(f"{key} names payload")
    for entry in sys.path:
        if entry and _path_names_payload(entry):
            blockers.append("sys.path names payload")
    seen: set[str] = set()
    for entry in sys.path:
        directory = Path(entry) if entry else Path.cwd()
        try:
            resolved = str(directory.resolve())
        except OSError:
            continue
        if resolved in seen or not directory.is_dir():
            continue
        seen.add(resolved)
        if (directory / "sitecustomize.py").is_file() and _path_names_payload(
            str(directory)
        ):
            blockers.append("sitecustomize.py is under payload")
        for pth in directory.glob("*.pth"):
            try:
                text = pth.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line in text.splitlines():
                body = line.strip()
                if not body or body.startswith(("#", "import ")):
                    continue
                if _path_names_payload(body):
                    blockers.append("a .pth file adds payload")
                    break
    return blockers


def normalize_run_context(
    trigger: str, actor: str, commit_sha: str
) -> dict[str, str] | None:
    """Return trigger metadata, or None when the caller did not set any.

    The instance id is not accepted here. A bad actor or SHA is refused
    without echoing it, so a secret passed in that position is not logged.
    """
    if trigger == "" and actor == "" and commit_sha == "":
        return None
    if trigger not in _TRIGGER_NAMES:
        raise InstanceNotApproved("refusing trigger")
    if _ACTOR_RE.fullmatch(actor) is None:
        raise InstanceNotApproved("refusing actor")
    if _COMMIT_SHA_RE.fullmatch(commit_sha) is None:
        raise InstanceNotApproved("refusing commit sha")
    return {"trigger": trigger, "actor": actor, "commit_sha": commit_sha}


def _apply_run_record(
    document: dict[str, Any],
    run_context: dict[str, str] | None,
    secret_forwarding: SecretForwarding | None,
) -> None:
    """Add trigger metadata. Refuse a record that says a secret was forwarded."""
    if secret_forwarding is not None:
        if secret_forwarding.get("secrets_forwarded") is not False:
            raise SmokeError("refusing to publish a manifest that forwarded a secret")
        document["secrets_forwarded"] = False
        document["secret_forwarding"] = {
            "secrets_forwarded": False,
            "ssh_env_forwarded": bool(secret_forwarding["ssh_env_forwarded"]),
            "git_credentials_shipped": bool(
                secret_forwarding["git_credentials_shipped"]
            ),
            "payload_secret_hits": int(secret_forwarding["payload_secret_hits"]),
        }
    if run_context is not None:
        document["trigger"] = run_context["trigger"]
        document["actor"] = run_context["actor"]
        document["commit_sha"] = run_context["commit_sha"]


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


def note_attach_key_id(payload: dict[str, Any], api_key: str) -> dict[str, Any]:
    """Record whether attach named a key id. Do not detach from here.

    The attach call can succeed with ``success`` and no key id. That is not
    treated as a registered id, and it is not a reason to skip the SSH probe.
    """
    key_id = ssh_key_id_from_attach(payload)
    noted: dict[str, Any] = {
        "key_id": key_id,
        "key_id_returned": key_id is not None,
        "detach": "on_close" if key_id is not None else "skipped",
    }
    if key_id is None:
        _log(
            {
                "action": "attach_ssh",
                "instance_id": ALLOWED_INSTANCE_ID,
                "key_id_returned": False,
                "detach": "skipped",
            },
            api_key,
        )
    return noted


def _ssh_failure_class(returncode: int, detail: str) -> str:
    if returncode == 0:
        return "accepted"
    lowered = detail.lower()
    if "permission denied" in lowered and "publickey" in lowered:
        return "publickey"
    if "permission denied" in lowered:
        return "denied"
    return "unavailable"


def _backoff_seconds(attempt: int, backoff_s: float) -> float:
    """Delay before ``attempt``. Attempt 2 uses the base, then doubles, then caps."""
    power = max(0, attempt - 2)
    return min(backoff_s * (2**power), SSH_READY_BACKOFF_CAP_S)


def wait_until_ssh_accepts(
    probe: Callable[[], tuple[int, str]],
    *,
    sleep: Callable[[float], None],
    now: Callable[[], float],
    api_key: str = "",
    max_attempts: int = SSH_READY_MAX_ATTEMPTS,
    timeout_s: float = SSH_READY_TIMEOUT_S,
    backoff_s: float = SSH_READY_BACKOFF_S,
) -> dict[str, Any]:
    """Retry a key-only SSH probe. Stop at ``max_attempts`` or ``timeout_s``.

    The probe's text is classified and not logged, so a key that leaked into
    stderr cannot reach the log line.
    """
    if max_attempts < 1:
        raise SmokeError("ssh readiness needs at least one attempt")
    if timeout_s <= 0:
        raise SmokeError("ssh readiness timeout must be positive")
    deadline = now() + timeout_s
    probes = 0
    last_class = "unavailable"
    last_code = 255
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            if now() >= deadline:
                break
            remaining = deadline - now()
            if remaining <= 0:
                break
            sleep(min(_backoff_seconds(attempt, backoff_s), remaining))
            if now() >= deadline:
                break
        code, detail = probe()
        probes += 1
        last_code = code
        last_class = _ssh_failure_class(code, detail)
        _log(
            {
                "action": "ssh_ready",
                "attempt": probes,
                "max_attempts": max_attempts,
                "returncode": code,
                "result": last_class,
            },
            api_key,
        )
        if code == 0:
            return {
                "accepted": True,
                "attempts": probes,
                "max_attempts": max_attempts,
                "result": "accepted",
            }
    raise SmokeError(
        "ssh did not accept the configured key "
        f"after {probes} attempts "
        f"(cap {max_attempts}, timeout {timeout_s:g}s, last result {last_class}, "
        f"last status {last_code})."
    )


def ssh_ready_or_raise(
    probe: Callable[[], tuple[int, str]],
    *,
    key_id_returned: bool,
    api_key: str,
    sleep: Callable[[float], None],
    now: Callable[[], float],
    max_attempts: int = SSH_READY_MAX_ATTEMPTS,
    timeout_s: float = SSH_READY_TIMEOUT_S,
    backoff_s: float = SSH_READY_BACKOFF_S,
) -> dict[str, Any]:
    """Probe SSH. A missing attach id is named on the error.

    Close still detaches the key when the listing has exactly one match.
    This error does not describe that later result.
    """
    try:
        ready = wait_until_ssh_accepts(
            probe,
            sleep=sleep,
            now=now,
            api_key=api_key,
            max_attempts=max_attempts,
            timeout_s=timeout_s,
            backoff_s=backoff_s,
        )
    except SmokeError as exc:
        if not key_id_returned:
            raise SmokeError(
                f"{exc} attach_ssh returned no key id. "
                "Close detaches it only when the listing has exactly one match."
            ) from None
        raise
    ready["key_id_returned"] = key_id_returned
    ready["detach"] = "on_close" if key_id_returned else "skipped"
    return ready


def await_ssh_after_attach(
    payload: dict[str, Any],
    probe: Callable[[], tuple[int, str]],
    *,
    api_key: str,
    sleep: Callable[[float], None],
    now: Callable[[], float],
    detach: Callable[[str], None] | None = None,
    max_attempts: int = SSH_READY_MAX_ATTEMPTS,
    timeout_s: float = SSH_READY_TIMEOUT_S,
    backoff_s: float = SSH_READY_BACKOFF_S,
) -> dict[str, Any]:
    """Retry SSH after attach. Detach is not called unless leftover detach is on."""
    noted = note_attach_key_id(payload, api_key)
    ready = ssh_ready_or_raise(
        probe,
        key_id_returned=bool(noted["key_id_returned"]),
        api_key=api_key,
        sleep=sleep,
        now=now,
        max_attempts=max_attempts,
        timeout_s=timeout_s,
        backoff_s=backoff_s,
    )
    ready["key_id"] = noted["key_id"]
    # No leftover-detach path. The callback is not invoked. The flag stays
    # false, and a true value refuses rather than deleting a key.
    ready["detached"] = False
    if detach is not None:
        ready["detached"] = False
    if DETACH_LEFTOVER_KEYS:
        raise SmokeError("leftover ssh keys are not detached")
    return ready


def ssh_ready_argv(key_path: str, known_hosts: str, port: int) -> list[str]:
    """Same argv as a copy, with a short connect timeout for the readiness probe."""
    argv = ssh_base_argv(key_path, known_hosts, port)
    return [
        "ConnectTimeout=5" if item == "ConnectTimeout=20" else item for item in argv
    ]


_KEY_LIST_FIELDS = ("ssh_keys", "attached_ssh_keys", "ssh_key_ids")
_SAFE_COMMENT_RE = re.compile(r"[A-Za-z0-9_.:@+-]+")


def _public_key_text(item: object) -> str | None:
    """Return an OpenSSH public key line. Private keys and other text are dropped."""
    if isinstance(item, str) and item.startswith(("ssh-ed25519 ", "ssh-rsa ")):
        return item
    if not isinstance(item, dict):
        return None
    for name in ("public_key", "ssh_key", "key"):
        value = item.get(name)
        if isinstance(value, str) and value.startswith(("ssh-ed25519 ", "ssh-rsa ")):
            return value
    return None


def ssh_fingerprints(payload: dict[str, Any]) -> list[str]:
    """Local ``SHA256:`` fingerprints. The key text is not returned.

    A fingerprint field on the payload is ignored. The digest is SHA256 of
    the decoded OpenSSH blob, base64 without padding, matching ``ssh-keygen -lf``.
    """
    items, bad = _key_items(payload.get("ssh_keys"))
    if bad or items is None:
        return []
    fingerprints: list[str] = []
    for item in items:
        public = _public_key_text(item)
        if public is None:
            continue
        fingerprint = _fingerprint(public)
        if fingerprint is not None:
            fingerprints.append(fingerprint)
    return fingerprints


def _raw_listed_key(item: object) -> str | None:
    """The listed key text, including a leading space. Not filtered by prefix."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for name in ("public_key", "ssh_key", "key"):
            value = item.get(name)
            if isinstance(value, str):
                return value
    return None


def _listed_key_id(item: dict[str, Any]) -> str | None:
    for name in ("id", "key_id", "ssh_key_id"):
        value = item.get(name)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int) or (
            isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]+", value)
        ):
            return str(value)
    return None


def match_own_public_key(
    public_key: str, payload: dict[str, Any]
) -> tuple[dict[str, Any], str | None]:
    """Find listing rows whose public key equals ``public_key``.

    The only normalization is ``str.rstrip`` on both sides, so a trailing
    newline does not hide a match. A different comment or blob does not match.
    The returned record has no key material. The id, when there is exactly
    one, is the second value and is not part of the record.
    """
    wanted = public_key.rstrip()
    items, bad = _key_items(payload.get("ssh_keys"))
    found: list[str] = []
    if not bad and items is not None:
        for item in items:
            listed = _raw_listed_key(item)
            # Equality after rstrip only. No case fold, no substring, no lstrip.
            if not isinstance(listed, str) or listed.rstrip() != wanted:
                continue
            if isinstance(item, dict):
                key_id = _listed_key_id(item)
                if key_id is not None:
                    found.append(key_id)
    fingerprint = _fingerprint(public_key)
    record: dict[str, Any] = {
        "match_count": len(found),
        "detached": False,
        "fingerprint": fingerprint,
    }
    if len(found) == 1:
        record["reason"] = "exactly one listed key matched this run's public key"
        return record, found[0]
    if len(found) == 0:
        record["reason"] = "no listed key matched this run's public key"
        return record, None
    record["reason"] = "more than one listed key matched this run's public key"
    return record, None


def ssh_key_summary(fingerprints: list[str]) -> dict[str, Any]:
    """Count and fingerprints only. Anything else is dropped."""
    clean = [
        item
        for item in fingerprints
        if isinstance(item, str) and re.fullmatch(r"SHA256:[A-Za-z0-9+/]+", item)
    ]
    return {"count": len(clean), "fingerprints": clean}


def ephemeral_key_record(public_key: str, after: dict[str, Any]) -> dict[str, Any]:
    """Fingerprint of the key this run generated, and if the post-attach list has it."""
    fingerprint = _fingerprint(public_key)
    if fingerprint is None:
        raise SmokeError(
            "ephemeral ssh key could not be fingerprinted. The key is not printed."
        )
    listed = fingerprint in after.get("fingerprints", [])
    return {"fingerprint": fingerprint, "listed_after_attach": listed}


def _fingerprint(public_key: str) -> str | None:
    parts = public_key.split()
    if len(parts) < 2 or not re.fullmatch(r"[A-Za-z0-9+/=]+", parts[1]):
        return None
    blob = parts[1]
    pad = "=" * ((4 - len(blob) % 4) % 4)
    try:
        raw = base64.b64decode(blob + pad, validate=True)
    except ValueError:
        return None
    digest = base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")
    return "SHA256:" + digest.rstrip("=")


def _comment_from_public_key(public_key: str) -> str | None:
    parts = public_key.split()
    if len(parts) < 3:
        return None
    comment = parts[-1]
    if len(comment) > 80 or _SAFE_COMMENT_RE.fullmatch(comment) is None:
        return None
    if "AAAA" in comment or "PRIVATE" in comment:
        return None
    return comment


def _safe_time(value: object) -> str | int | float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str) and len(value) <= 40 and "\n" not in value:
        if "AAAA" in value or "PRIVATE" in value or value.startswith("ssh-"):
            return None
        return value
    return None


def _safe_key_record(item: object) -> dict[str, Any] | None:
    """Id, fingerprint or comment, and created time. Never the key material."""
    if isinstance(item, bool) or item is None:
        return None
    if isinstance(item, int):
        return {"id": str(item)}
    if isinstance(item, str) and re.fullmatch(r"[A-Za-z0-9_-]+", item):
        return {"id": item}
    if not isinstance(item, dict):
        return None
    record: dict[str, Any] = {}
    for name in ("id", "key_id", "ssh_key_id"):
        value = item.get(name)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int) or (
            isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]+", value)
        ):
            record["id"] = str(value)
            break
    fingerprint = item.get("fingerprint")
    if (
        isinstance(fingerprint, str)
        and fingerprint.startswith("SHA256:")
        and len(fingerprint) <= 120
        and "\n" not in fingerprint
    ):
        record["fingerprint"] = fingerprint
    for name in ("comment", "name"):
        comment = item.get(name)
        if (
            isinstance(comment, str)
            and 0 < len(comment) <= 80
            and _SAFE_COMMENT_RE.fullmatch(comment) is not None
            and "AAAA" not in comment
            and "PRIVATE" not in comment
        ):
            record["comment"] = comment
            break
    for name in ("created_at", "created", "created_time"):
        created = _safe_time(item.get(name))
        if created is not None:
            record["created"] = created
            break
    public = None
    for name in ("public_key", "ssh_key", "key"):
        value = item.get(name)
        if isinstance(value, str) and value.startswith(("ssh-ed25519 ", "ssh-rsa ")):
            public = value
            break
    if public is not None:
        if "fingerprint" not in record:
            derived = _fingerprint(public)
            if derived is not None:
                record["fingerprint"] = derived
        if "comment" not in record:
            derived_comment = _comment_from_public_key(public)
            if derived_comment is not None:
                record["comment"] = derived_comment
    if not record:
        return None
    return record


def _key_items(value: object) -> tuple[list[object] | None, bool]:
    """Return the key list, and whether the field was present but unreadable."""
    if isinstance(value, list):
        return value, False
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None, True
        if isinstance(parsed, list):
            return parsed, False
        return None, True
    return None, True


def attached_ssh_keys_from_show(payload: dict[str, Any]) -> dict[str, Any]:
    """Non-secret key ids from a show body. This function makes no request."""
    row = unwrap_instance(payload)
    if row is None and _same_id(payload.get("id")):
        row = payload
    if row is None:
        row = payload
    found = False
    unreadable = False
    raw: list[object] = []
    field: str | None = None
    for name in _KEY_LIST_FIELDS:
        if name not in row:
            continue
        found = True
        field = name
        items, bad = _key_items(row.get(name))
        if bad or items is None:
            unreadable = True
            raw = []
        else:
            raw = items
        break
    keys: list[dict[str, Any]] = []
    for item in raw:
        record = _safe_key_record(item)
        if record is not None:
            keys.append(record)
    return {
        "source": "show",
        "present_in_show": found,
        "field": field,
        "unreadable": unreadable,
        "keys": keys,
        "detach_leftover_keys": False,
    }


def _manifest_ssh_keys(value: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep count and ``SHA256:`` fingerprints. Drop key bodies and other fields."""
    if value is None:
        return None
    raw = value.get("fingerprints")
    fingerprints = raw if isinstance(raw, list) else []
    return ssh_key_summary([item for item in fingerprints if isinstance(item, str)])


def _manifest_own_key(value: dict[str, Any] | None) -> dict[str, Any] | None:
    """match_count, detached, reason, and a fingerprint. No key material."""
    if value is None:
        return None
    count = value.get("match_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        count = 0
    detached = value.get("detached") is True
    reason = value.get("reason")
    if not isinstance(reason, str) or "ssh-" in reason or "PRIVATE" in reason:
        reason = "match recorded without key material"
    kept: dict[str, Any] = {
        "match_count": count,
        "detached": detached,
        "reason": reason,
    }
    fingerprint = value.get("fingerprint")
    if isinstance(fingerprint, str) and re.fullmatch(
        r"SHA256:[A-Za-z0-9+/]+", fingerprint
    ):
        kept["fingerprint"] = fingerprint
    return kept


def _manifest_ephemeral(value: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep the local fingerprint and the post-attach membership bit."""
    if value is None:
        return None
    fingerprint = value.get("fingerprint")
    if (
        not isinstance(fingerprint, str)
        or re.fullmatch(r"SHA256:[A-Za-z0-9+/]+", fingerprint) is None
    ):
        fingerprint = None
    listed = value.get("listed_after_attach")
    if not isinstance(listed, bool):
        listed = None
    return {"fingerprint": fingerprint, "listed_after_attach": listed}


def _public_ssh_report(ssh: dict[str, Any]) -> dict[str, Any]:
    kept: dict[str, Any] = {}
    for name in (
        "accepted",
        "attempts",
        "max_attempts",
        "result",
        "key_id_returned",
        "detach",
        "key_id",
    ):
        if name not in ssh:
            continue
        value = ssh[name]
        if name == "key_id":
            if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]+", value):
                kept[name] = value
            continue
        if value is None or isinstance(value, (str, int, bool)):
            kept[name] = value
    return kept


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


def _record_dc2(decision: str) -> str:
    """Dc2 lives in the manifest. The pipeline ledger is not touched."""
    if decision not in {"missing", "failed"}:
        raise SmokeError(f"dc2 decision {decision!r} is not a failure record")
    return decision


def collect_artifact_manifest(
    run_dir: Path,
    *,
    command: list[str],
    instance_id: object,
    api_key: str = "",
    attached_ssh_keys: dict[str, Any] | None = None,
    ssh: dict[str, Any] | None = None,
    ssh_keys: dict[str, Any] | None = None,
    ephemeral_ssh_key: dict[str, Any] | None = None,
    own_key: dict[str, Any] | None = None,
    run_context: dict[str, str] | None = None,
    secret_forwarding: SecretForwarding | None = None,
    pipeline: str = "sceneagent",
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
    judged = stages
    if pipeline in BIN_PIPELINES:
        judged = [s for s in stages if s["item"] in PIPELINES[pipeline][0]]
    present_items = {stage["item"] for stage in stages}
    missing_items = [
        name for name in PIPELINES[pipeline][0] if name not in present_items
    ]
    missing_files: list[str] = []
    if ledger_bytes is None:
        missing_files.append("ledger")
    if not plan_path.is_file():
        missing_files.append("plan")
    reason = _failure_reason(
        judged, missing_items if ledger is not None else [], missing_files
    )
    if pipeline == "probe":
        # A probe has no plan or ledger; its result is the inventory file.
        inventory = run_dir / "inventory.txt"
        missing_files = [] if inventory.is_file() else ["inventory"]
        reason = None if inventory.is_file() else "inventory.txt was not written"
    passed = (pipeline == "probe" and reason is None) or (
        ledger_bytes is not None
        and plan_path.is_file()
        and not missing_items
        and bool(judged)
        and all(stage["counts_as_pass"] for stage in judged)
        and reason is None
    )
    manifest = {
        "instance_id": approved,
        "video": PIPELINES[pipeline][1],
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
    if not plan_path.is_file():
        manifest["dc2"] = _record_dc2("missing")
    if ledger_bytes is None:
        manifest["ledger_status"] = "missing"
    if attached_ssh_keys is not None:
        manifest["attached_ssh_keys"] = attached_ssh_keys
    listed = _manifest_ssh_keys(ssh_keys)
    if listed is not None:
        manifest["ssh_keys"] = listed
    ephemeral = _manifest_ephemeral(ephemeral_ssh_key)
    if ephemeral is not None:
        manifest["ephemeral_ssh_key"] = ephemeral
    own = _manifest_own_key(own_key)
    if own is not None:
        manifest["own_key"] = own
    if ssh is not None:
        manifest["ssh"] = _public_ssh_report(ssh)
    if pipeline == "probe":
        manifest["complete"] = (run_dir / "inventory.txt").is_file()
    else:
        manifest["complete"] = plan_path.is_file() and ledger_path.is_file()
    if not manifest["complete"]:
        manifest["passed"] = False
    _apply_run_record(manifest, run_context, secret_forwarding)
    manifest_text = redact(json.dumps(manifest, indent=2) + "\n", api_key)
    if api_key and api_key in manifest_text:
        raise SmokeError(
            "VAST_API_KEY was found in manifest.json after redaction. "
            "No artifacts were published."
        )
    (run_dir / "manifest.json").write_text(manifest_text, encoding="utf-8")
    if ledger_bytes is None:
        if ledger_path.exists():
            raise SmokeError("refusing to create ledger.json")
    elif ledger_path.read_bytes() != ledger_bytes:
        raise SmokeError("refusing to rewrite ledger.json")
    return manifest


def _key_hit(name: str) -> None:
    """Say which file contained the key. Do not print the value."""
    _emit(f"VAST_API_KEY was found in {name}. No artifacts were published.", "")


def publish_artifacts(
    run_dir: Path, artifact_dir: Path, api_key: str = ""
) -> list[str]:
    """Copy run files after a key scan. ``plan.json`` and ``ledger.json`` stay byte copies.

    The scan reads every file in ``run_dir`` before ``artifact_dir`` is
    created or written. A key in any file other than the collector's
    manifest fails the run and leaves the artifact directory untouched.
    ``manifest.json`` is redacted and scanned again before it is written.
    """
    run_dir = Path(run_dir)
    artifact_dir = Path(artifact_dir)
    secret = api_key.encode("utf-8") if api_key else b""
    # The setup pipeline's Hugging Face token: no published file may hold it.
    hf_token = os.environ.get(HF_TOKEN_ENV, "").encode("utf-8")
    prepared: list[tuple[str, bytes]] = []
    hits: list[str] = []
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(run_dir).as_posix()
        data = path.read_bytes()
        if path.name == "manifest.json":
            if secret and secret in data:
                data = redact(data.decode("utf-8", errors="replace"), api_key).encode(
                    "utf-8"
                )
            if secret and secret in data:
                hits.append(rel)
            prepared.append((rel, data))
            continue
        if (secret and secret in data) or (hf_token and hf_token in data):
            hits.append(rel)
            continue
        prepared.append((rel, data))
    if hits:
        for name in hits:
            _key_hit(name)
        raise SmokeError(
            "VAST_API_KEY was found in "
            + ", ".join(hits)
            + ". No artifacts were published."
        )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for rel, data in prepared:
        if Path(rel).name not in PUBLISHED_NAMES:
            continue
        target = artifact_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        copied.append(rel)
    return copied


def _is_key_scan_failure(exc: BaseException) -> bool:
    text = str(exc)
    return "No artifacts were published" in text or "VAST_API_KEY was found" in text


def publish_failure_manifest(
    artifact_dir: Path,
    *,
    api_key: str,
    instance_id: object,
    failure_reason: str,
    attached_ssh_keys: dict[str, Any] | None = None,
    ssh: dict[str, Any] | None = None,
    ssh_keys: dict[str, Any] | None = None,
    ephemeral_ssh_key: dict[str, Any] | None = None,
    own_key: dict[str, Any] | None = None,
    run_context: dict[str, str] | None = None,
    secret_forwarding: SecretForwarding | None = None,
) -> dict[str, Any]:
    """Write only a redacted manifest after the same scan as a successful publish.

    ``plan.json`` and ``ledger.json`` are not created. The scan runs before
    ``artifact_dir`` is written. A key that survives redaction leaves it empty.
    """
    approved = parse_instance_id(instance_id)
    document: dict[str, Any] = {
        "instance_id": approved,
        "video": FRUITS_RELATIVE,
        "passed": False,
        "complete": False,
        "plan": None,
        "ledger": None,
        "manifest": "manifest.json",
        "failure_reason": failure_reason,
        "published_on_failure": "manifest.json",
        "detach_leftover_keys": DETACH_LEFTOVER_KEYS,
    }
    if attached_ssh_keys is not None:
        document["attached_ssh_keys"] = attached_ssh_keys
    listed = _manifest_ssh_keys(ssh_keys)
    if listed is not None:
        document["ssh_keys"] = listed
    ephemeral = _manifest_ephemeral(ephemeral_ssh_key)
    if ephemeral is not None:
        document["ephemeral_ssh_key"] = ephemeral
    own = _manifest_own_key(own_key)
    if own is not None:
        document["own_key"] = own
    if ssh is not None:
        document["ssh"] = _public_ssh_report(ssh)
    _apply_run_record(document, run_context, secret_forwarding)
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        (run_dir / "manifest.json").write_text(
            json.dumps(document, indent=2) + "\n",
            encoding="utf-8",
        )
        publish_artifacts(run_dir, Path(artifact_dir), api_key=api_key)
    text = (Path(artifact_dir) / "manifest.json").read_text(encoding="utf-8")
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise SmokeError("failure manifest was not a JSON object")
    return parsed


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


def require_ssh_private_key(value: object) -> str:
    """Return the fixed private key. Do not include the value in an error."""
    if not isinstance(value, str) or not value.strip():
        raise InstanceNotApproved(f"{SSH_PRIVATE_KEY_ENV} is not set.")
    text = value.replace("\r\n", "\n").strip()
    if (
        "-----BEGIN " not in text
        or "-----END " not in text
        or "PRIVATE" not in text
        or text.startswith(("ssh-ed25519 ", "ssh-rsa "))
    ):
        raise InstanceNotApproved(f"{SSH_PRIVATE_KEY_ENV} is not a private key.")
    return text + "\n"


def write_ssh_private_key(directory: Path, material: str) -> Path:
    """Write the runner key as mode 0600. The bytes are not logged."""
    path = directory / "sf4_ed25519"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, material.encode("utf-8"))
    finally:
        os.close(fd)
    os.chmod(path, 0o600)
    return path


def load_known_host_line(path: Path, host: str, port: int) -> str:
    """Return the one pinned known_hosts line, or refuse before SSH.

    The error does not quote the file. A comment-only file is not a pin.
    """
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError:
        raise InstanceNotApproved(
            "VAST SSH known host pin is not configured."
        ) from None
    lines = [
        line.strip()
        for line in raw.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if len(lines) != 1 or "PRIVATE" in lines[0] or lines[0].startswith("-----"):
        raise InstanceNotApproved("VAST SSH known host pin is not configured.")
    match = _KNOWN_HOST_RE.fullmatch(lines[0])
    if match is None:
        raise InstanceNotApproved("VAST SSH known host pin is not configured.")
    pinned_host = match.group("bracket") or match.group("plain")
    pinned_port = match.group("port")
    if pinned_host != host or (pinned_port is not None and int(pinned_port) != port):
        raise InstanceNotApproved(
            "VAST SSH known host pin does not match the instance ssh endpoint."
        )
    return lines[0] + "\n"


def install_known_host_pin(directory: Path, source: Path, host: str, port: int) -> str:
    """Copy the validated pin into the runner temp dir. No keyscan."""
    line = load_known_host_line(source, host, port)
    path = directory / "known_hosts"
    path.write_text(line, encoding="utf-8")
    path.chmod(0o644)
    return str(path)


def _advise_motion(checkout: Path, local_run: Path) -> None:
    """Copy the proposal into the run dir and write an advisory frame check.

    A missing key or a missing zip leaves the motion result unchanged.
    """
    import zipfile

    from sceneagent.articulate_run import PROPOSAL_NAME

    source = Path(checkout) / "sceneagent" / PROPOSAL_NAME
    if source.is_file():
        (local_run / PROPOSAL_NAME).write_bytes(source.read_bytes())
    archive_path = local_run / "articulate_frames.zip"
    if not archive_path.is_file():
        return
    frames: list[bytes] = []
    with zipfile.ZipFile(archive_path) as archive:
        for name in sorted(archive.namelist())[:4]:
            frames.append(archive.read(name))
    if len(frames) < 2:
        return
    try:
        from sceneagent.gemini_client import client_from_environ
        from sceneagent.vlm_check import advise

        report = advise(
            frames,
            "Is the expected joint motion visible in these frames?",
            client_from_environ(),
        )
    except Exception as exc:
        report = {
            "label": "vlm_check",
            "advisory": True,
            "verdict": "unparsed",
            "reason": type(exc).__name__,
        }
    (local_run / "vlm_check.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


class SshRemote:
    """Copy the checkout, run the smoke, and fetch the small JSON files."""

    def __init__(
        self,
        record: dict[str, Any],
        api_key: str,
        request: RequestFn,
        *,
        private_key: str,
        known_host_file: Path,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_key = api_key
        self.request = request
        self.sleep = sleep
        self.host, self.port = ssh_endpoint(record)
        self._private_material = require_ssh_private_key(private_key)
        self.tmp = tempfile.TemporaryDirectory()
        self.ssh_report: dict[str, Any] = {
            "accepted": None,
            "remote_keys_modified": False,
        }
        try:
            root = Path(self.tmp.name)
            self.private = write_ssh_private_key(root, self._private_material)
            self.known_hosts = install_known_host_pin(
                root, known_host_file, self.host, self.port
            )
        except Exception:
            # The constructor does not return, so run_smoke cannot close the
            # mode-0600 file. Remove it here. A cleanup error must not replace
            # the setup error.
            try:
                self.close()
            except Exception as cleanup_error:
                _emit("ssh cleanup failed: " + str(cleanup_error), self.api_key)
            raise

    def close(self) -> None:
        """Remove the runner temp dir, including the private key file."""
        self.tmp.cleanup()

    def run(self, checkout: Path, local_run: Path) -> int:
        self._sync(checkout)
        # Attributes, not new run() arguments: tests inject remotes whose
        # run() takes only the checkout and the local directory.
        pipeline = getattr(self, "pipeline", "sceneagent")
        commit_sha = getattr(self, "commit_sha", "")
        if not isinstance(pipeline, str):
            pipeline = "sceneagent"
        if not isinstance(commit_sha, str):
            commit_sha = ""
        if pipeline == "articulate":
            from sceneagent.articulate_run import prepare_for_box

            prepare_for_box(checkout)
        if pipeline == "setup":
            self.hf_token_sent = self._send_hf_token(os.environ.get(HF_TOKEN_ENV, ""))
        code = self._bash(
            build_remote_shell(
                REMOTE_REPO,
                remote_out(pipeline),
                pipeline=pipeline,
                commit_sha=commit_sha,
            ),
            timeout_s=REMOTE_TIMEOUT_S,
        )
        self._fetch(local_run, pipeline)
        if pipeline == "articulate":
            _advise_motion(checkout, local_run)
        return code

    def _probe_ssh(self) -> tuple[int, str]:
        argv = [
            *ssh_ready_argv(str(self.private), self.known_hosts, self.port),
            f"root@{self.host}",
            "true",
        ]
        _reject_secret_text(" ".join(argv))
        try:
            completed = subprocess.run(
                argv,
                check=False,
                capture_output=True,
                env=child_env(),
                timeout=10,
            )
        except subprocess.TimeoutExpired:
            return 255, "timeout"
        detail = completed.stderr.decode("utf-8", errors="replace")[-300:]
        detail = redact(redact(detail, self.api_key), self._private_material)
        return int(completed.returncode), detail

    def _sync(self, checkout: Path) -> None:
        try:
            ready = wait_until_ssh_accepts(
                self._probe_ssh,
                sleep=self.sleep,
                now=time.monotonic,
                api_key=self.api_key,
            )
        except SmokeError:
            self.ssh_report["accepted"] = False
            raise
        self.ssh_report.update(ready)
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
                tail = redact(redact(tail, self.api_key), self._private_material)
                raise SmokeError(f"rsync failed ({completed.returncode}): {tail}")

    def _send_hf_token(self, token: str) -> bool:
        """Write the Hugging Face token to the box for the setup pipeline.

        Sandeep's call (2026-11-11): SAM 3D's weights are gated, so the setup
        step needs the token on the box. It goes on ssh stdin into a mode-0600
        file; the argv names only the path, and nothing echoes it back.
        """
        if not token or any(ch.isspace() for ch in token):
            return False
        mask_secret(token)
        target = shlex.quote(REMOTE_HF_TOKEN)
        folder = shlex.quote(str(Path(REMOTE_HF_TOKEN).parent))
        argv = [
            *ssh_base_argv(str(self.private), self.known_hosts, self.port),
            f"root@{self.host}",
            f"umask 077 && mkdir -p {folder} && cat > {target}.tmp && mv {target}.tmp {target}",
        ]
        _reject_secret_text(" ".join(argv))
        completed = subprocess.run(
            argv,
            check=False,
            input=token.encode("utf-8"),
            capture_output=True,
            env=child_env(),
            timeout=60,
        )
        return completed.returncode == 0

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

    def _fetch(self, local_run: Path, pipeline: str = "sceneagent") -> None:
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
                input=build_fetch_script(remote_out(pipeline)).encode("utf-8"),
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


def _remote_key_facts(
    remote: Any,
    listing: dict[str, Any] | None,
    ephemeral: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Copy the pre-attach listing and the ephemeral fingerprint off the remote."""
    found_listing = getattr(remote, "ssh_key_listing", None)
    if listing is None and isinstance(found_listing, dict):
        listing = found_listing
    found_ephemeral = getattr(remote, "ephemeral_ssh_key", None)
    if ephemeral is None and isinstance(found_ephemeral, dict):
        ephemeral = found_ephemeral
    return listing, ephemeral


def _remote_own_key(remote: Any) -> dict[str, Any] | None:
    found = getattr(remote, "own_key", None)
    if isinstance(found, dict):
        return found
    return None


def fetch_pinned_ssh_keys(
    request: RequestFn,
    api_key: str,
    sleep: Callable[[float], None],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """GET the pinned ssh list. The summary has no key material."""
    method, url, body = endpoint_for("show_ssh_keys", ALLOWED_INSTANCE_ID)
    status, payload = call_json(request, method, url, api_key, body, sleep=sleep)
    if status != 200:
        raise SmokeError(
            f"listing ssh keys failed (HTTP {status}). The keys are not printed."
        )
    return ssh_key_summary(ssh_fingerprints(payload)), payload


def load_ssh_key_fingerprints(
    request: RequestFn,
    api_key: str,
    sleep: Callable[[float], None],
) -> dict[str, Any]:
    """GET ``/api/v0/instances/60254439/ssh/``. Return count and local fingerprints.

    This does not detach, and it does not return key material.
    """
    summary, _payload = fetch_pinned_ssh_keys(request, api_key, sleep)
    return summary


def list_attached_ssh_keys(
    request: RequestFn,
    api_key: str,
    *,
    sleep: Callable[[float], None] = lambda _seconds: None,
) -> dict[str, Any]:
    """Read the pinned ssh list. Count and fingerprints only. Never detach."""
    return load_ssh_key_fingerprints(request, api_key, sleep)


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
        try:
            ssh_endpoint(row)
            host_ok = True
        except SmokeError:
            host_ok = False
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
    ssh_private_key: str | None = None,
    known_host_file: Path | None = None,
    trigger: str = "",
    actor: str = "",
    commit_sha: str = "",
    pipeline: str = "sceneagent",
) -> int:
    """Run the Fruits smoke on 60254439 when that instance is already running.

    Exit 0 is a GPU accept. Exit 2 is a refusal before the instance is changed.
    Exit 1 is a smoke that ran and did not pass, including a ledger
    ``reject`` or ``deferred``. This function does not start or stop the instance.
    SIGTERM raises ``SystemExit(1)`` so ``finally`` can detach, and it does
    not change instance power.
    """
    try:
        approved = parse_instance_id(instance_id)
    except InstanceNotApproved as exc:
        _emit(str(exc), "")
        return 2
    if pipeline not in PIPELINES:
        _emit("refusing pipeline", "")
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
    try:
        context = normalize_run_context(trigger, actor, commit_sha)
    except InstanceNotApproved as exc:
        _emit(str(exc), key)
        return 2
    import_blockers = payload_import_blockers()
    if import_blockers:
        for blocker in import_blockers:
            _emit(blocker, key)
        _emit(
            "refusing to import the payload checkout. No instance call was made.",
            key,
        )
        return 2
    forwarding, problems = assess_secret_forwarding(Path(checkout), key)
    if problems:
        for problem in problems:
            _emit(problem, key)
        _emit(
            "refusing to forward a secret to the GPU. No instance call was made.",
            key,
        )
        return 2

    if handle_sigterm:

        def _on_term(_signum: int, _frame: object) -> None:
            raise SystemExit(1)

        signal.signal(signal.SIGTERM, _on_term)

    remote: Any = None
    code = 1
    remote_code = 0
    shown = False
    published = False
    scan_failed = False
    failure_reason: str | None = None
    key_report: dict[str, Any] | None = None
    ephemeral_ssh_key: dict[str, Any] | None = None
    own_key: dict[str, Any] | None = None
    ssh_report: dict[str, Any] | None = None

    try:
        row = _show_instance(request, key, sleep)
        shown = True
        _log({"action": "show", "instance": public_instance_view(row)}, key)
        kind = classify_status(row)
        if kind == "stopped":
            raise InstanceNotApproved(
                "instance stopped; this workflow does not start or stop instances"
            )
        if kind == "starting":
            raise InstanceNotApproved(
                "instance not running; this workflow does not start or stop instances"
            )
        if kind != "running":
            raise InstanceNotApproved(
                "instance state unknown; this workflow does not start or stop instances"
            )
        try:
            ssh_endpoint(row)
        except SmokeError as exc:
            raise InstanceNotApproved(str(exc)) from None
        if remote_factory is None:
            material = (
                ssh_private_key
                if ssh_private_key is not None
                else os.environ.get(SSH_PRIVATE_KEY_ENV, "")
            )
            pin = KNOWN_HOST_PATH if known_host_file is None else Path(known_host_file)
            remote = SshRemote(
                row,
                key,
                request,
                private_key=material,
                known_host_file=pin,
                sleep=sleep,
            )
        else:
            remote = remote_factory(row)
        with tempfile.TemporaryDirectory() as tmp:
            local_run = Path(tmp) / "run"
            remote.pipeline = pipeline
            remote.commit_sha = commit_sha
            remote_code = int(remote.run(Path(checkout), local_run))
            found = getattr(remote, "ssh_report", None)
            if isinstance(found, dict):
                ssh_report = found
            key_report, ephemeral_ssh_key = _remote_key_facts(
                remote, key_report, ephemeral_ssh_key
            )
            own_key = _remote_own_key(remote)
            manifest = collect_artifact_manifest(
                local_run,
                command=build_remote_command(
                    REMOTE_REPO,
                    remote_out(pipeline),
                    pipeline=pipeline,
                    commit_sha=commit_sha,
                ),
                instance_id=approved,
                api_key=key,
                ssh=ssh_report,
                ssh_keys=key_report,
                ephemeral_ssh_key=ephemeral_ssh_key,
                own_key=own_key,
                run_context=context,
                secret_forwarding=forwarding,
                pipeline=pipeline,
            )
            publish_artifacts(local_run, Path(artifact_dir), api_key=key)
            published = True
        concluded: dict[str, Any] = {
            "action": "conclude",
            "instance_id": approved,
            "passed": manifest["passed"],
            "remote_exit": remote_code,
            "failure_reason": manifest.get("failure_reason"),
            "secrets_forwarded": False,
        }
        if context is not None:
            concluded["trigger"] = context["trigger"]
            concluded["actor"] = context["actor"]
            concluded["commit_sha"] = context["commit_sha"]
        _log(concluded, key)
        if remote_code == 0 and manifest["passed"]:
            code = 0
        elif remote_code != 0:
            code = remote_code
        else:
            code = 1
    except InstanceNotApproved as exc:
        failure_reason = str(exc)
        _emit(failure_reason, key)
        code = 2
    except SmokeError as exc:
        failure_reason = str(exc)
        scan_failed = _is_key_scan_failure(exc)
        _emit(failure_reason, key)
        code = remote_code if remote_code != 0 else 1
    finally:
        if remote is not None:
            try:
                remote.close()
            except Exception as exc:
                _emit("ssh cleanup failed: " + str(exc), key)
                code = 1
            found = getattr(remote, "ssh_report", None)
            if ssh_report is None and isinstance(found, dict):
                ssh_report = found
            key_report, ephemeral_ssh_key = _remote_key_facts(
                remote, key_report, ephemeral_ssh_key
            )
            own_key = _remote_own_key(remote)
        if (
            shown
            and code != 0
            and not published
            and not scan_failed
            and failure_reason is not None
        ):
            try:
                publish_failure_manifest(
                    Path(artifact_dir),
                    api_key=key,
                    instance_id=approved,
                    failure_reason=failure_reason,
                    ssh=ssh_report,
                    ssh_keys=key_report,
                    ephemeral_ssh_key=ephemeral_ssh_key,
                    own_key=own_key,
                    run_context=context,
                    secret_forwarding=forwarding,
                )
            except SmokeError as exc:
                _emit(str(exc), key)
                if code == 0:
                    code = 1
    return code


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="SceneAgent Fruits GPU smoke on Vast instance 60254439"
    )
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--trigger", choices=sorted(_TRIGGER_NAMES), default="")
    parser.add_argument("--actor", default="")
    parser.add_argument("--commit-sha", default="")
    parser.add_argument("--pipeline", choices=sorted(PIPELINES), default="sceneagent")
    args = parser.parse_args(argv)
    return run_smoke(
        args.instance_id,
        args.checkout,
        args.artifacts,
        os.environ.get("VAST_API_KEY", ""),
        handle_sigterm=True,
        trigger=args.trigger,
        actor=args.actor,
        commit_sha=args.commit_sha,
        pipeline=args.pipeline,
    )


if __name__ == "__main__":
    raise SystemExit(main())
