# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SF-5 dc3/dc4 full-rate COLMAP evidence runner.

PM KAN-5 10314: full rate means every native frame, no subsampling, no cap,
and a fresh COLMAP 3.11.1 reconstruction. The 4.2.1 host model is not
evidence. The quality bar stays ``reconstruction_gate``. This module does
not retune it and it is not the SF-4 300-bin sample.

The runner is not cleared to launch. QA, PM, Projects Manager go, and an
independently verified authorization are outside this file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from sceneagent.colmap_stage import run_poses
from sceneagent.reconstruction_gate import EVIDENCE_DIRNAME

PINNED_BASE_SHA = "4bf04192ad3a0f3c14b5e8cf24f077883cf62919"
APPROVED_RELATIVE = "docs/assets/example_videos/Fruits.mp4"
APPROVED_SHA256 = "a9c099cea15ec73bc203cff0925aa88f166acc2c75e672d9ff89d3bb34169b4d"
APPROVED_BYTES = 1733022
APPROVED_FRAMES = 452
APPROVED_WIDTH = 1280
APPROVED_HEIGHT = 720
APPROVED_DURATION_ROUNDED_S = "15.067"
INSTANCE_ID = "58301075"
REQUIRED_COLMAP_VERSION = "3.11.1"
SCENEAGENT_ROOT = "/workspace/sceneagent-simfoundry"
LEGACY_MODEL_ROOT = "/workspace/fruits-fullrate-colmap"
PRODUCTION_ROOT = "/workspace/cut1-gpu"
FORBIDDEN_PREFIXES = (SCENEAGENT_ROOT, LEGACY_MODEL_ROOT)
FORBIDDEN_COMPONENTS = frozenset({"fitted_splat", "sceneagent-simfoundry"})
# EL engineering recommendation, 2026-11-06 6:47 PM ET. Provisional and
# unmeasured. Not a measured fact and not launch authorization.
PROPOSED_MIN_FREE_MIB = 6144
GPU_SAMPLE_INTERVAL_S = 10
# A wedged nvidia-smi must not sit inside the stage poll forever. One try.
GPU_QUERY_TIMEOUT_S = 30
FAQ_URL = "https://colmap.github.io/faq.html"
FAQ_SECTION = "Feature matching fails due to illegal memory access"
# The live page on 2026-11-06. It is not a snapshot of the COLMAP 3.11.1 docs.
FAQ_PINNED_TO_COLMAP_3_11_1 = False
GPU_FREE_ARGV = (
    "nvidia-smi",
    "--query-gpu=memory.total,memory.used,memory.free",
    "--format=csv,noheader,nounits",
)
AUTHORIZATION_GATES = (
    "exact-hash QA PASS on this package",
    "PM PASS on this package",
    "Projects Manager go",
    "independently verified user authorization",
)
FAIL_CONDITIONS = (
    "input path missing, not the pinned relative path, or a symlink",
    "recomputed sha256 or byte size differs from the pin",
    "probed native frame count, dimensions, or rounded duration differ from the pin",
    "ffmpeg command is not -fps_mode passthrough with no fps filter and no frame cap",
    "extracted frame count differs from the probed native count",
    "COLMAP banner is not exactly 3.11.1",
    "output prefix is not the owned KAN-5 run directory, is non-empty, or is forbidden",
    "reconstruction_gate does not pass",
    "evidence SHA256SUMS do not match the raw analyzer files",
    "GPU free-memory budget is absent, invalid, or not the proposed 6144 MiB",
    "a GPU stage starts while any reported GPU has free memory below 6144 MiB",
    "authorization gates are not recorded here and cannot clear a launch",
)
PURPOSE = (
    "SF-5 dc3/dc4 fresh COLMAP 3.11.1 reconstruction of every native "
    "Fruits.mp4 frame, with per-image color calibration and the repo gate"
)


class EvidenceError(Exception):
    """The evidence run cannot continue."""


class ForbiddenPathError(EvidenceError):
    """A path is forbidden, escapes, or is a symlink."""


class GpuBudgetError(EvidenceError):
    """The explicit free-memory budget is missing or the reading is short."""


class GpuQueryTimeout(EvidenceError):
    """The memory query did not return before its hard timeout."""


# run_poses sizes 180s and 900s for its 16-frame default. A native sequence
# gets one of those ceilings per 16 frames. This is a ceiling, not a measured runtime.
SHORT_RUN_FRAMES = 16
SHORT_RUN_FFMPEG_TIMEOUT_S = 180
SHORT_RUN_COLMAP_TIMEOUT_S = 900


def native_timeouts(frame_count: int) -> dict[str, int]:
    """Step ceilings for a native-frame COLMAP run.

    The 16-frame sample keeps 180s for ffmpeg and 900s for each COLMAP step.
    A longer sequence is allowed that budget once per 16 frames, rounded up.
    """
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count <= 0
    ):
        raise EvidenceError("frame count for the step ceiling is missing")
    blocks = (frame_count + SHORT_RUN_FRAMES - 1) // SHORT_RUN_FRAMES
    return {
        "ffmpeg_timeout_s": SHORT_RUN_FFMPEG_TIMEOUT_S * blocks,
        "colmap_timeout_s": SHORT_RUN_COLMAP_TIMEOUT_S * blocks,
    }


def calibration_outcome(poses: dict[str, Any]) -> str:
    """Distinguish a disabled calibration from one that ran and failed."""
    if not poses.get("color_calibrate"):
        return "off"
    if "color_calibration" not in poses:
        return "failed"
    return "present"


def evidence_kwargs() -> dict[str, Any]:
    """The only extract configuration this package will run."""
    return {
        "fps": 0,
        "max_frames": 0,
        "native_frames": True,
        "color_calibrate": True,
    }


def gate_accepted(poses: dict[str, Any]) -> bool:
    """True only when run_poses and the repo gate both accepted the model."""
    gate = poses.get("reconstruction_gate") or {}
    return poses.get("ok") is True and gate.get("pass") is True


def exit_code(kind: str) -> int:
    """Process status. A reject is not zero and an identity failure is not a reject."""
    codes = {
        "identity": 2,
        "sampling": 2,
        "forbidden": 2,
        "reuse": 2,
        "version": 2,
        "resource": 3,
        "gpu_budget": 4,
        "reject": 1,
        "accept": 0,
    }
    if kind not in codes:
        raise ValueError(kind)
    return codes[kind]


def launch_ready() -> bool:
    """This package does not record QA, PM, or go, so it cannot clear itself."""
    return False


def gpu_attempts() -> int:
    """One query. A failure is not retried."""
    return 1


def hashes_match(actual: str, expected: str) -> bool:
    return bool(actual) and actual == expected


def colmap_version_allowed(version: str | None) -> bool:
    return version == REQUIRED_COLMAP_VERSION


def counts_as_fruits_evidence(dry_run: bool, decision: str) -> bool:
    """A dry run is a rehearsal. It is not the Fruits reconstruction."""
    return (not dry_run) and decision == "accept"


def gpu_query_is_aggregate(argv: tuple[str, ...] | list[str]) -> bool:
    text = " ".join(argv)
    banned = ("pid", "pmon", "compute-apps", "process_name")
    return text == " ".join(GPU_FREE_ARGV) and not any(item in text for item in banned)


def path_is_forbidden(path: os.PathLike[str] | str) -> bool:
    """Lexical check. No filesystem access."""
    raw = os.fspath(path)
    if "\x00" in raw:
        return True
    norm = os.path.normpath(raw)
    parts = Path(norm).parts
    if FORBIDDEN_COMPONENTS.intersection(parts):
        return True
    for prefix in FORBIDDEN_PREFIXES:
        if norm == prefix or norm.startswith(prefix + os.sep):
            return True
    return False


def under_legacy(path: os.PathLike[str] | str) -> bool:
    norm = os.path.normpath(os.fspath(path))
    return norm == LEGACY_MODEL_ROOT or norm.startswith(LEGACY_MODEL_ROOT + os.sep)


def legacy_model_was_used(paths: list[os.PathLike[str] | str]) -> bool:
    return any(under_legacy(path) for path in paths)


def lexical_absolute(path: os.PathLike[str] | str) -> Path:
    text = os.fspath(path)
    if not os.path.isabs(text):
        text = os.path.join(os.getcwd(), text)
    return Path(os.path.normpath(text))


def assert_no_symlink(path: Path) -> None:
    """Reject symlink components before opening the path.

    A link whose target is a forbidden root is rejected from the link text.
    Any other symlink is rejected too, so the target is never followed.
    """
    if path_is_forbidden(path):
        raise ForbiddenPathError(path)
    acc = Path(path.anchor)
    parts = path.parts[1:]
    for part in parts:
        acc = acc / part
        if path_is_forbidden(acc):
            raise ForbiddenPathError(acc)
        if not os.path.lexists(acc):
            # The tail does not exist yet. mkdir creates it after this walk,
            # and a missing component cannot hide a symlink.
            return
        if os.path.islink(acc):
            target = os.readlink(acc)
            joined = (
                target
                if os.path.isabs(target)
                else os.path.normpath(str(acc.parent / target))
            )
            if path_is_forbidden(joined):
                raise ForbiddenPathError(joined)
            raise ForbiddenPathError(acc)


def _contained(root: Path, path: Path) -> bool:
    root_norm = os.path.normpath(str(root))
    path_norm = os.path.normpath(str(path))
    return path_norm == root_norm or path_norm.startswith(root_norm + os.sep)


def required_output_prefix(code_sha: str) -> Path:
    if len(code_sha) < 7:
        raise EvidenceError("code sha is too short")
    return Path(PRODUCTION_ROOT) / f"KAN-5-{code_sha[:7]}" / "run"


def output_allowed(
    prefix: Path, *, dry_run: bool, code_sha: str | None, repo: Path
) -> bool:
    norm = Path(os.path.normpath(str(prefix)))
    if path_is_forbidden(norm) or under_legacy(norm):
        return False
    if dry_run:
        # The production prefix is for the instance, not a rehearsal, and a
        # run must not write into the checkout that holds this package.
        if _contained(Path(PRODUCTION_ROOT), norm) or _contained(repo, norm):
            return False
        return len(norm.parts) >= 4
    if not code_sha:
        return False
    return os.path.normpath(str(norm)) == os.path.normpath(
        str(required_output_prefix(code_sha))
    )


def reuse_refused(prefix: Path) -> bool:
    """True when the prefix already holds a previous run. Nothing is deleted."""
    if not os.path.lexists(prefix):
        return False
    if os.path.islink(prefix) or not os.path.isdir(prefix):
        return True
    return any(prefix.iterdir())


def guard_output(
    prefix: os.PathLike[str] | str, *, dry_run: bool, code_sha: str | None, repo: Path
) -> Path:
    path = lexical_absolute(prefix)
    if path_is_forbidden(path) or not output_allowed(
        path, dry_run=dry_run, code_sha=code_sha, repo=repo
    ):
        raise ForbiddenPathError(path)
    assert_no_symlink(path)
    return path


def resolve_input(repo: Path, relative: str, *, dry_run: bool) -> Path:
    if not dry_run:
        if relative != APPROVED_RELATIVE or Path(relative).is_absolute():
            raise ForbiddenPathError(relative)
        if ".." in Path(relative).parts:
            raise ForbiddenPathError(relative)
        candidate = lexical_absolute(repo / relative)
    else:
        raw = Path(relative)
        if ".." in raw.parts:
            raise ForbiddenPathError(relative)
        candidate = raw if raw.is_absolute() else lexical_absolute(repo / raw)
        candidate = lexical_absolute(candidate)
    if path_is_forbidden(candidate):
        raise ForbiddenPathError(candidate)
    assert_no_symlink(candidate)
    return candidate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pinned_file_bytes_match(repo: Path) -> bool:
    """Recompute the pin. A missing file, a symlink, or a mismatch is false."""
    path = lexical_absolute(repo / APPROVED_RELATIVE)
    if path_is_forbidden(path):
        return False
    try:
        assert_no_symlink(path)
    except (ForbiddenPathError, FileNotFoundError, OSError):
        return False
    if not path.is_file():
        return False
    if path.stat().st_size != APPROVED_BYTES:
        return False
    return hashes_match(sha256_file(path), APPROVED_SHA256)


def assert_native_argv(command: list[Any]) -> None:
    parts = [str(part) for part in command]
    if "-fps_mode" not in parts or parts[parts.index("-fps_mode") + 1] != "passthrough":
        raise EvidenceError("extract is not native-frame passthrough")
    if "-vf" in parts or "-frames:v" in parts:
        raise EvidenceError("extract still subsamples or caps frames")
    if any(part.startswith("fps=") for part in parts):
        raise EvidenceError("extract still has an fps filter")


def parse_colmap_banner(banner: str) -> str | None:
    found = []
    for token in (banner or "").split():
        piece = token.strip(".,;:()[]")
        parts = piece.split(".")
        if len(parts) == 3 and all(part.isdigit() for part in parts):
            found.append(piece)
    if len(found) != 1:
        return None
    return found[0]


def parse_ffprobe_document(payload: dict[str, Any]) -> dict[str, Any]:
    streams = payload.get("streams") or []
    if len(streams) != 1:
        raise EvidenceError("ffprobe did not return one video stream")
    stream = streams[0]
    try:
        frames = int(stream["nb_read_frames"])
        container_frames = int(stream["nb_frames"])
        width = int(stream["width"])
        height = int(stream["height"])
        duration_s = float(stream["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceError("ffprobe document is incomplete") from exc
    return {
        "frames": frames,
        "container_frames": container_frames,
        "width": width,
        "height": height,
        "duration_s": duration_s,
    }


def fruits_probe_matches(probed: dict[str, Any]) -> bool:
    rounded = f"{probed['duration_s']:.3f}"
    return (
        probed["frames"] == APPROVED_FRAMES
        and probed["container_frames"] == APPROVED_FRAMES
        and probed["width"] == APPROVED_WIDTH
        and probed["height"] == APPROVED_HEIGHT
        and rounded == APPROVED_DURATION_ROUNDED_S
    )


def sift_match_bytes(num_matches: int) -> int:
    """Bytes in the current public FAQ's SIFT estimate.

    The expression is ``4 * num_matches * num_matches + 4 * num_matches * 256``.
    It describes that page. It is not a COLMAP 3.11.1 pin, and it is not this
    package's runtime budget.
    """
    return 4 * num_matches * num_matches + 4 * num_matches * 256


def accepted_free_mib_budget(value: Any) -> int:
    """Accept only the explicit proposed 6144 MiB budget. There is no default."""
    proposed = value == PROPOSED_MIN_FREE_MIB or value == str(PROPOSED_MIN_FREE_MIB)
    if isinstance(value, (bool, float)) or not proposed:
        raise GpuBudgetError(
            "GPU free-memory budget is absent or not the proposed 6144 MiB"
        )
    return PROPOSED_MIN_FREE_MIB


def parse_gpu_memory_csv(stdout: str, returncode: int | None) -> list[dict[str, int]]:
    """Parse total, used, free MiB. One row per GPU. No process fields."""
    if returncode != 0:
        raise EvidenceError("GPU memory query failed")
    rows: list[dict[str, int]] = []
    for line in (stdout or "").splitlines():
        text = line.strip()
        if not text:
            continue
        parts = [part.strip() for part in text.split(",")]
        if len(parts) != 3 or any(not part.isdigit() for part in parts):
            raise EvidenceError("GPU memory query is not total, used, free")
        total, used, free = (int(part) for part in parts)
        if used > total or free > total:
            raise EvidenceError("GPU memory query is inconsistent")
        rows.append({"total_mib": total, "used_mib": used, "free_mib": free})
    if not rows:
        raise EvidenceError("GPU memory query returned no GPU")
    return rows


def memory_meets_budget(rows: list[dict[str, Any]], budget: int) -> bool:
    """Every reported GPU must have free memory at or above the budget.

    Processes are not listed, so a short GPU cannot be skipped by picking
    another device.
    """
    return bool(rows) and all(row["free_mib"] >= budget for row in rows)


def during_sample_is_terminal(_reading: dict[str, Any]) -> bool:
    """A mid-stage reading is a log. It does not stop COLMAP.

    6144 MiB is headroom required before a GPU stage starts. COLMAP is
    expected to consume some of that memory while the stage runs.
    """
    return False


def summarize_free_samples(samples: list[int]) -> dict[str, Any]:
    """Lowest free MiB among the samples taken. Not an exact true peak."""
    lowest = min(samples) if samples else None
    return {
        "lowest_free_mib_observed": lowest,
        "sample_count": len(samples),
        "sample_interval_s": GPU_SAMPLE_INTERVAL_S,
        "exact_true_peak": False,
        "continuous_minimum": False,
        "retries": 0,
    }


def query_gpu_memory(runner, budget: int) -> dict[str, Any]:
    """One memory-only query. A failure is not retried."""
    if not gpu_query_is_aggregate(GPU_FREE_ARGV):
        raise EvidenceError("GPU query is not the memory-only query")
    last = None
    for _ in range(gpu_attempts()):
        last = runner(list(GPU_FREE_ARGV))
    if not isinstance(last, dict):
        raise EvidenceError("GPU memory query failed")
    rows = parse_gpu_memory_csv(str(last.get("stdout") or ""), last.get("returncode"))
    return {
        "rows": rows,
        "min_free_mib": min(row["free_mib"] for row in rows),
        "meets_budget": memory_meets_budget(rows, budget),
        "budget_mib": budget,
    }


def evidence_dir_for(output_prefix: Path) -> Path:
    return output_prefix / "colmap" / EVIDENCE_DIRNAME


def evidence_path_allowed(output_prefix: Path, evidence_dir: Path) -> bool:
    expected = os.path.normpath(str(evidence_dir_for(output_prefix)))
    actual = os.path.normpath(str(evidence_dir))
    return actual == expected


def parse_sha256sums(text: str) -> dict[str, str]:
    rows = {}
    for line in text.splitlines():
        if not line:
            continue
        digest, sep, name = line.partition("  ")
        if sep != "  " or len(digest) != 64 or not name or name in rows:
            raise EvidenceError("evidence SHA256SUMS line is not usable")
        rows[name] = digest
    if not rows:
        raise EvidenceError("evidence SHA256SUMS is empty")
    return rows


def sums_match(listed: str, actual: str) -> bool:
    return bool(listed) and listed == actual


def evidence_checksums(evidence_dir: Path) -> dict[str, str]:
    """Recompute every regular file under the evidence directory."""
    if not evidence_dir.is_dir() or evidence_dir.is_symlink():
        raise EvidenceError("reconstruction evidence directory is missing")
    checksums = {}
    for path in sorted(evidence_dir.rglob("*")):
        if path.is_symlink():
            raise EvidenceError("symlink inside reconstruction evidence")
        if not path.is_file():
            continue
        checksums[path.relative_to(evidence_dir).as_posix()] = sha256_file(path)
    sums_path = evidence_dir / "SHA256SUMS"
    if "SHA256SUMS" not in checksums or "reconstruction_gate.json" not in checksums:
        raise EvidenceError("raw gate JSON or SHA256SUMS is missing")
    listed = parse_sha256sums(sums_path.read_text(encoding="utf-8"))
    for name, digest in listed.items():
        if not sums_match(digest, checksums.get(name, "")):
            raise EvidenceError(f"evidence checksum mismatch for {name}")
    analyzer = [name for name in checksums if name.startswith("model_analyzer_")]
    if not analyzer:
        raise EvidenceError("raw model_analyzer output is missing")
    return checksums


def git_text(repo: Path, args: list[str]) -> tuple[int, str]:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
        shell=False,
    )
    return completed.returncode, (completed.stdout or "").strip()


def observed_code_sha(repo: Path) -> str | None:
    code, text = git_text(repo, ["rev-parse", "HEAD"])
    if code != 0 or len(text) != 40:
        return None
    return text


def base_is_ancestor(repo: Path) -> bool:
    code, _text = git_text(
        repo, ["merge-base", "--is-ancestor", PINNED_BASE_SHA, "HEAD"]
    )
    return code == 0


def _ffprobe(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames,nb_frames,width,height,duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
        shell=False,
    )
    if completed.returncode != 0:
        raise EvidenceError("ffprobe failed")
    return parse_ffprobe_document(json.loads(completed.stdout or ""))


def _colmap_banner() -> str:
    binary = shutil_which("colmap")
    if not binary:
        raise EvidenceError("colmap is not on PATH")
    completed = subprocess.run(
        [binary, "-h"],
        capture_output=True,
        text=True,
        check=False,
        shell=False,
    )
    return (completed.stdout or "") + (completed.stderr or "")


def shutil_which(name: str) -> str | None:
    from shutil import which

    return which(name)


def ffmpeg_supports_passthrough() -> bool:
    binary = shutil_which("ffmpeg")
    if not binary:
        raise EvidenceError("ffmpeg is not on PATH")
    completed = subprocess.run(
        [binary, "-hide_banner", "-h", "full"],
        capture_output=True,
        text=True,
        check=False,
        shell=False,
    )
    text = (completed.stdout or "") + (completed.stderr or "")
    return "-fps_mode" in text


def _step_statuses(poses: dict[str, Any]) -> dict[str, Any]:
    statuses: dict[str, Any] = {}
    for name, step in (poses.get("steps") or {}).items():
        if isinstance(step, dict):
            statuses[name] = step.get("returncode")
            statuses[f"{name}_timed_out"] = bool(step.get("timed_out"))
    gate = poses.get("reconstruction_gate") or {}
    statuses["reconstruction_gate_pass"] = gate.get("pass")
    models = []
    for model in gate.get("sparse_models") or []:
        models.append(
            {"model": model.get("model"), "analyzer_rc": model.get("analyzer_rc")}
        )
    statuses["model_analyzer"] = models
    return statuses


def _write_ledger(prefix: Path, document: dict[str, Any]) -> str:
    text = json.dumps(document, indent=2, sort_keys=True) + "\n"
    path = prefix / "ledger.json"
    path.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    (prefix / "ledger.sha256").write_text(digest + "  ledger.json\n", encoding="utf-8")
    return digest


def _finish(
    prefix: Path | None,
    *,
    kind: str,
    document: dict[str, Any],
) -> int:
    code = exit_code(kind)
    document["process_exit"] = code
    document["decision"] = "accept" if kind == "accept" else "reject"
    document["counts_as_fruits_evidence"] = counts_as_fruits_evidence(
        bool(document.get("dry_run")), document["decision"]
    )
    document["legacy_model_reused"] = legacy_model_was_used(
        [document.get("input", {}).get("path", ""), document.get("output_prefix", "")]
    )
    if document["legacy_model_reused"]:
        code = exit_code("reuse")
        document["process_exit"] = code
        document["decision"] = "reject"
        document["counts_as_fruits_evidence"] = False
    if prefix is not None:
        (prefix / "gpu_memory_log.json").write_text(
            json.dumps(document.get("gpu_memory") or [], indent=2) + "\n",
            encoding="utf-8",
        )
        digest = _write_ledger(prefix, document)
        document_path = prefix / "ledger.sha256"
        if document_path.read_text(encoding="utf-8").split()[0] != digest:
            raise EvidenceError("ledger sha256 was not the bytes just written")
    print(
        f"decision={document['decision']} exit={code} "
        f"dry_run={document.get('dry_run')} kind={kind}"
    )
    return code


def execute(request: dict[str, Any]) -> int:
    """Run or refuse one evidence attempt. Writes only under the owned prefix."""
    dry_run = bool(request.get("dry_run"))
    repo = lexical_absolute(request["repo"])
    document: dict[str, Any] = {
        "ticket": "KAN-5",
        "sf": "SF-5",
        "criteria": ["dc3", "dc4"],
        "purpose": PURPOSE,
        "dry_run": dry_run,
        "evidence_class": "synthetic_dry_run" if dry_run else "fruits_native_colmap",
        "base_sha": PINNED_BASE_SHA,
        "instance_id": request.get("instance_id"),
        "config": {
            **evidence_kwargs(),
            "fps_mode": "passthrough",
            "frame_cap": None,
            "colmap_required": REQUIRED_COLMAP_VERSION,
            "sampling_ruling": "PM KAN-5 10314 every native frame, no subsampling, no cap",
            "separate_from": "SF-4 300-bin sampling",
            "thresholds": "sceneagent.reconstruction_gate",
            "gpu_free_mib_min": PROPOSED_MIN_FREE_MIB,
            "gpu_budget_approved": False,
            "gpu_budget_status": "provisional",
            "gpu_query": list(GPU_FREE_ARGV),
            "gpu_sample_interval_s": GPU_SAMPLE_INTERVAL_S,
            "gpu_retries": 0,
            "faq_pinned_to_colmap_3_11_1": FAQ_PINNED_TO_COLMAP_3_11_1,
        },
        "input": {"declared_sha256": request.get("input_sha256")},
        "authorization_gates": list(AUTHORIZATION_GATES),
        "ready_for_execution": launch_ready(),
        "gpu_budget_argument": request.get("gpu_free_mib_min"),
    }
    gpu_log: list[dict[str, Any]] = []
    document["gpu_memory"] = gpu_log
    prefix: Path | None = None
    try:
        if path_is_forbidden(repo):
            raise ForbiddenPathError(repo)
        assert_no_symlink(repo)
        head = observed_code_sha(repo)
        document["code_sha"] = head
        if not dry_run:
            if request.get("instance_id") != INSTANCE_ID:
                raise EvidenceError("instance id is not 58301075")
            if not head or request.get("expect_code_sha") != head:
                raise EvidenceError("checkout SHA does not match expect-code-sha")
            if not base_is_ancestor(repo):
                raise EvidenceError("pinned base is not an ancestor of HEAD")
            if request.get("input_sha256") != APPROVED_SHA256:
                raise EvidenceError("declared sha256 is not the approved pin")
        elif request.get("expect_code_sha") not in (None, head):
            raise EvidenceError("checkout SHA does not match expect-code-sha")
        output = guard_output(
            request["output_prefix"],
            dry_run=dry_run,
            code_sha=head,
            repo=repo,
        )
        document["output_prefix"] = str(output)
        video = resolve_input(repo, str(request["input"]), dry_run=dry_run)
        document["input"]["path"] = str(video)
        if legacy_model_was_used([video, output, repo]):
            raise ForbiddenPathError(video)
        if reuse_refused(output):
            raise EvidenceError("refusing to reuse a non-empty output prefix")
        output.mkdir(parents=True, exist_ok=True)
        prefix = output
        budget = accepted_free_mib_budget(request.get("gpu_free_mib_min"))
        if not video.is_file():
            raise EvidenceError("input file is missing")
        size = video.stat().st_size
        digest = sha256_file(video)
        document["input"]["bytes"] = size
        document["input"]["recomputed_sha256"] = digest
        document["input"]["hash_match"] = hashes_match(
            digest, str(request["input_sha256"])
        )
        if not document["input"]["hash_match"]:
            raise EvidenceError("recomputed sha256 does not match the declared pin")
        if not dry_run and (
            not pinned_file_bytes_match(repo) or size != APPROVED_BYTES
        ):
            raise EvidenceError("input does not match the approved Fruits pin")
        probed = _ffprobe(video)
        document["probed"] = probed
        if not dry_run and not fruits_probe_matches(probed):
            raise EvidenceError("probe does not match the approved Fruits identity")
        if not ffmpeg_supports_passthrough():
            raise EvidenceError("ffmpeg has no -fps_mode passthrough")
        banner = _colmap_banner()
        version = parse_colmap_banner(banner)
        document["colmap_version"] = version
        if not colmap_version_allowed(version):
            raise EvidenceError(
                f"refusing COLMAP {version}; fresh {REQUIRED_COLMAP_VERSION} rerun required"
            )
        source = "offline_fixture" if dry_run else "nvidia-smi"

        def reader(argv):
            if not gpu_query_is_aggregate(argv):
                raise EvidenceError("GPU query is not the memory-only query")
            if dry_run:
                stdout = request.get("memory_stdout") or "24576, 1024, 23552\n"
                return {"returncode": 0, "stdout": stdout, "stderr": ""}
            return _nvidia_smi(argv)

        def before_gpu_stage(stage, _argv):
            reading = query_gpu_memory(reader, budget)
            gpu_log.append(
                {
                    "stage": stage,
                    "when": "before",
                    "source": source,
                    "exact_true_peak": False,
                    "retries": gpu_attempts() - 1,
                    **reading,
                }
            )
            if not reading["meets_budget"]:
                raise GpuBudgetError(
                    f"GPU free memory before {stage} is below {budget} MiB"
                )

        def on_gpu_sample():
            try:
                reading = query_gpu_memory(reader, budget)
            except GpuQueryTimeout:
                raise
            except EvidenceError as exc:
                gpu_log.append(
                    {
                        "stage": "during",
                        "when": "during",
                        "source": source,
                        "query_error": str(exc),
                        "exact_true_peak": False,
                        "stops_the_stage": False,
                    }
                )
                return None
            frees = [row["free_mib"] for row in reading["rows"]]
            gpu_log.append(
                {
                    "stage": "during",
                    "when": "during",
                    "source": source,
                    "stops_the_stage": during_sample_is_terminal(reading),
                    **reading,
                    **summarize_free_samples(frees),
                }
            )
            if during_sample_is_terminal(reading):
                raise GpuBudgetError(
                    "GPU free memory during a stage is below the budget"
                )
            return reading

        timeouts = native_timeouts(int(probed["frames"]))
        document["config"]["ffmpeg_timeout_s"] = timeouts["ffmpeg_timeout_s"]
        document["config"]["colmap_timeout_s"] = timeouts["colmap_timeout_s"]

        def after_extract(frames):
            count = len(frames)
            if count != probed["frames"]:
                return "extracted frame count is not the native count"
            if not dry_run and count != APPROVED_FRAMES:
                return "extracted frame count is not 452"
            if dry_run and count <= 16:
                return "dry run extracted too few frames to show there is no cap"
            return ""

        poses = run_poses(
            video,
            output / "colmap",
            before_gpu_stage=before_gpu_stage,
            on_gpu_sample=on_gpu_sample,
            sample_interval_s=GPU_SAMPLE_INTERVAL_S,
            ffmpeg_timeout_s=timeouts["ffmpeg_timeout_s"],
            colmap_timeout_s=timeouts["colmap_timeout_s"],
            after_extract=after_extract,
            **evidence_kwargs(),
        )
        document["exit_statuses"] = _step_statuses(poses)
        command = ((poses.get("steps") or {}).get("ffmpeg") or {}).get("command")
        if command is None:
            raise EvidenceError("ffmpeg did not run")
        assert_native_argv(list(command))
        if str(video) not in [str(part) for part in command]:
            raise EvidenceError("ffmpeg did not read the checked input")
        document["ffmpeg_command"] = [str(part) for part in command]
        frames = poses.get("frames") or []
        document["extracted_frames"] = len(frames)
        ffmpeg_step = (poses.get("steps") or {}).get("ffmpeg") or {}
        extract_failed = poses.get("failed_step") == "ffmpeg" or bool(
            ffmpeg_step.get("timed_out")
        )
        if extract_failed:
            document["failed_step"] = "ffmpeg"
            if ffmpeg_step.get("timed_out"):
                document["reason"] = poses.get("reason") or "ffmpeg timed out"
            else:
                document["reason"] = poses.get("reason") or "ffmpeg failed"
            return _finish(prefix, kind="sampling", document=document)
        if poses.get("failed_step") == "frame_count" or len(frames) != probed["frames"]:
            document["failed_step"] = "frame_count"
            document["reason"] = (
                poses.get("reason") or "extracted frame count is not the native count"
            )
            return _finish(prefix, kind="sampling", document=document)
        if not dry_run and len(frames) != APPROVED_FRAMES:
            document["failed_step"] = "frame_count"
            document["reason"] = "extracted frame count is not 452"
            return _finish(prefix, kind="sampling", document=document)
        if len(frames) <= 16 and dry_run:
            document["failed_step"] = "frame_count"
            document["reason"] = (
                "dry run extracted too few frames to show there is no cap"
            )
            return _finish(prefix, kind="sampling", document=document)
        outcome = calibration_outcome(poses)
        if outcome == "off":
            raise EvidenceError("color calibration was off")
        if outcome == "failed":
            document["failed_step"] = poses.get("failed_step") or "color_calibration"
            document["reason"] = (
                poses.get("reason") or "color calibration produced no report"
            )
            return _finish(prefix, kind="reject", document=document)
        evidence = evidence_dir_for(output)
        if not evidence_path_allowed(output, evidence):
            raise EvidenceError("evidence path is not under this run")
        if not gate_accepted(poses):
            document["reconstruction_evidence"] = _evidence_or_none(output, evidence)
            document["fail_reasons"] = (poses.get("reconstruction_gate") or {}).get(
                "fail_reasons"
            )
            document["failed_step"] = poses.get("failed_step")
            document["reason"] = poses.get("reason")
            return _finish(prefix, kind="reject", document=document)
        checksums = evidence_checksums(evidence)
        gate_path = evidence / "reconstruction_gate.json"
        gate_document = json.loads(gate_path.read_text(encoding="utf-8"))
        document["reconstruction_evidence"] = {
            "directory": str(evidence),
            "checksums": checksums,
            "gate_pass": gate_document.get("pass"),
            "verdict": gate_document.get("verdict"),
            "fail_reasons": gate_document.get("fail_reasons"),
            "largest_model": gate_document.get("largest_model"),
            "quality_bar": (gate_document.get("quality_bar") or {}),
        }
        if gate_document.get("pass") is not True:
            return _finish(prefix, kind="reject", document=document)
        return _finish(prefix, kind="accept", document=document)
    except (EvidenceError, FileNotFoundError) as exc:
        document["reason"] = str(exc)
        kind = "forbidden" if isinstance(exc, ForbiddenPathError) else "identity"
        if isinstance(exc, GpuBudgetError):
            kind = "gpu_budget"
        elif isinstance(exc, GpuQueryTimeout):
            # The message is lowercase "gpu_query_timeout", so the "GPU"
            # text scan below misses it and would record identity exit 2.
            # A query that does not return is a resource failure, exit 3.
            kind = "resource"
        elif "COLMAP" in str(exc):
            kind = "version"
        elif "GPU" in str(exc):
            kind = "resource"
        elif "reuse" in str(exc):
            kind = "reuse"
        elif "native" in str(exc) or "fps" in str(exc) or "cap" in str(exc):
            kind = "sampling"
        return _finish(prefix, kind=kind, document=document)


def _evidence_or_none(output: Path, evidence: Path) -> dict[str, Any] | None:
    if not evidence.is_dir():
        return None
    try:
        checksums = evidence_checksums(evidence)
    except EvidenceError:
        checksums = {}
    return {
        "directory": str(evidence),
        "checksums": checksums,
        "allowed": evidence_path_allowed(output, evidence),
    }


def _nvidia_smi(argv: list[str], timeout_s: float | None = None) -> dict[str, Any]:
    """One aggregate memory query. A timeout is not retried.

    ``subprocess.run`` stops the query process it started. It does not
    signal the COLMAP stage. The stage child is stopped by ``_run`` when
    this exception leaves the in-stage sample.
    """
    if timeout_s is None:
        timeout_s = GPU_QUERY_TIMEOUT_S
    if not gpu_query_is_aggregate(argv):
        raise EvidenceError("GPU query is not the aggregate free-memory query")
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            check=False,
            shell=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        raise GpuQueryTimeout("gpu_query_timeout") from None
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout or "",
        "stderr": completed.stderr or "",
    }


def render_launch_command(code_sha: str) -> str:
    """Exact command. Not cleared to run, and it contains no placeholders."""
    if launch_ready():
        raise EvidenceError("launch_ready must stay false")
    if len(code_sha) != 40 or any(char not in "0123456789abcdef" for char in code_sha):
        raise EvidenceError("code sha must be 40 lowercase hex characters")
    prefix = required_output_prefix(code_sha)
    command = [
        "python",
        "-m",
        "sceneagent.sf5_colmap_evidence",
        "--repo",
        ".",
        "--input",
        APPROVED_RELATIVE,
        "--input-sha256",
        APPROVED_SHA256,
        "--expect-code-sha",
        code_sha,
        "--output-prefix",
        str(prefix),
        "--instance-id",
        INSTANCE_ID,
        "--gpu-free-mib-min",
        str(PROPOSED_MIN_FREE_MIB),
    ]
    lines = [
        "STATUS: BLOCKED",
        "ready_for_execution: false",
        "gpu_budget_approved: false",
        "gpu_budget_status: provisional",
        "gpu_free_mib_min: 6144",
        "This command is exact. It is not cleared to run.",
        "Blocked until: " + "; ".join(AUTHORIZATION_GATES) + ".",
        (
            "The 6144 MiB figure is an Engineering Lead estimate from "
            "2026-11-06 6:47 PM ET. It is unmeasured and is not launch authorization."
        ),
        "Direct user authorization is also required.",
        "Do not execute this file.",
        "Input is the pinned relative path under the checkout of the code SHA.",
        (
            "The output prefix must be a real directory chain on instance "
            f"{INSTANCE_ID}. Symlink components are rejected before read."
        ),
        "No SSH, no package upload, and no workflow dispatch.",
        "",
        " ".join(command),
        "",
    ]
    return "\n".join(lines)


def build_manifest(
    files: dict[str, bytes], *, code_sha: str, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Hash package files. The manifest body does not contain its own digest."""
    if len(code_sha) != 40:
        raise EvidenceError("code sha must be 40 hex characters")
    body: dict[str, Any] = {
        "ticket": "KAN-5",
        "sf": "SF-5",
        "instance_id": INSTANCE_ID,
        "purpose": PURPOSE,
        "code_sha": code_sha,
        "code_short_sha": code_sha[:7],
        "base_sha": PINNED_BASE_SHA,
        "ready_for_execution": launch_ready(),
        "output_prefix": str(required_output_prefix(code_sha)),
        "inputs": {
            "relative_path": APPROVED_RELATIVE,
            "sha256": APPROVED_SHA256,
            "bytes": APPROVED_BYTES,
            "frames": APPROVED_FRAMES,
            "width": APPROVED_WIDTH,
            "height": APPROVED_HEIGHT,
            "duration_rounded_s": APPROVED_DURATION_ROUNDED_S,
            "status": "pinned; runtime recomputes the sha256 before extraction",
        },
        "resources": {
            "colmap": REQUIRED_COLMAP_VERSION,
            "instance_id": INSTANCE_ID,
            "gpu_query": list(GPU_FREE_ARGV),
            "gpu_free_mib_min": {
                "parameter": "--gpu-free-mib-min",
                "proposed_mib": PROPOSED_MIN_FREE_MIB,
                "approved": False,
                "status": "provisional_el_estimate",
                "provenance": "Engineering Lead, 2026-11-06 6:47 PM ET",
            },
            "gpu_policy": (
                "require the explicit proposed 6144 MiB free on every reported "
                "GPU before each GPU stage; query memory.total,used,free only; "
                "sample every 10s during a stage and log the lowest free, which "
                "is not an exact true peak; do not list processes; do not kill "
                "or renice another process; do not retry"
            ),
        },
        "fail_conditions": list(FAIL_CONDITIONS),
        "authorization_gates": list(AUTHORIZATION_GATES),
        "files": {
            name: hashlib.sha256(blob).hexdigest()
            for name, blob in sorted(files.items())
            if name not in {"manifest.json", "PACKAGE_SHA256SUMS"}
        },
    }
    if extra:
        body.update(extra)
    if "manifest_sha256" in body or "self_sha256" in body:
        raise EvidenceError("manifest must not carry its own digest")
    return body


def write_package_sums(directory: Path) -> None:
    """SHA256SUMS for every other file in ``directory``, including the manifest."""
    destination = directory / "PACKAGE_SHA256SUMS"
    lines = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.is_symlink() or path == destination:
            continue
        relative = path.relative_to(directory).as_posix()
        lines.append(f"{sha256_file(path)}  {relative}")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SF-5 COLMAP evidence runner")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--expect-code-sha")
    parser.add_argument("--instance-id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--gpu-free-mib-min",
        default=None,
        help="Explicit proposed free-memory budget in MiB. No default.",
    )
    args = parser.parse_args(argv)
    return execute(
        {
            "repo": args.repo,
            "input": args.input,
            "input_sha256": args.input_sha256,
            "output_prefix": args.output_prefix,
            "expect_code_sha": args.expect_code_sha,
            "instance_id": args.instance_id,
            "dry_run": args.dry_run,
            "gpu_free_mib_min": args.gpu_free_mib_min,
        }
    )


if __name__ == "__main__":
    raise SystemExit(main())
