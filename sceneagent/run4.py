# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SF-4 run4 pipeline: 300 sharp frames, then the existing COLMAP gate.

The default ``python -m sceneagent`` video path stays 1 fps and 16 frames.
This module is the run4 path. It pins the simfoundry interpreter, refuses
a COLMAP stage that is under 6144 MiB free, and samples the sharpest frame
in each of 300 bins across the whole clip. Acceptance is
``reconstruction_gate`` on the largest model (KAN-4 10239). Points stay
informational. Unregistered frame names are written for the report and do
not change the sequential matcher. splatfacto is not trained; its 16384 MiB
floor is recorded and checked only by :func:`require_gpu_headroom`.

The checked-out commit is whatever ``git rev-parse HEAD`` the workflow
measured. This module does not invent a SHA and does not read ``.git``.
"""

from __future__ import annotations

import io
import json
import math
import shutil
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from sceneagent.colmap_stage import (
    colmap_major,
    init_pairs_from_log,
    mapper_ladder,
    pose_commands,
    use_gpu_from_banner,
)
from sceneagent.focal_prior import self_calibrate
from sceneagent.gpu_headroom import (
    COLMAP_MIN_FREE_MIB,
    SPLATFACTO_MIN_FREE_MIB,
    GpuCommandResult,
    GpuHeadroomError,
    require_gpu_headroom,
)
from sceneagent.pipeline import write_plan
from sceneagent.pose_diagnosis import diagnose, read_cameras_bin
from sceneagent.reconstruction_gate import EVIDENCE_DIRNAME, accept_reconstruction
from sceneagent.run4_report import registered_image_names, unregistered_frame_report
from sceneagent.run4_sampling import (
    BIN_COUNT,
    bin_count,
    decode_all_command,
    extract_selected_command,
    ffprobe_command,
    parse_probe,
    select_frames,
)
from sceneagent.simfoundry_python import (
    PinnedInterpreterMissing,
    require_simfoundry_python,
    stage_environment,
)
from sceneagent.video import VIDEO_SOURCE, inspect_video, video_plan_steps

MATCHER = "sequential_matcher"
# The mapper on 300 frames ran past 900 s on instance 59601312 (Actions run
# 42420122702); extraction and matching had finished in about 4 minutes.
# 5400 s fits inside vast_smoke.REMOTE_TIMEOUT_S and the workflow's 120 min.
STEP_TIMEOUT_S = {"mapper": 5400}
DEFAULT_STEP_TIMEOUT_S = 900
TIMEOUT_RETURNCODE = 124
_COMMIT_RE_TEXT = r"[0-9a-f]{40}"


def checked_out_commit(value: str) -> str | None:
    """Return a 40-hex SHA, or None when the workflow did not measure one.

    Any other string fails. The empty string is not replaced with a guess.
    """
    import re

    if value == "":
        return None
    if re.fullmatch(_COMMIT_RE_TEXT, value) is None:
        raise ValueError("checked-out commit is not 40 hex")
    return value


def _command_text(result: dict[str, Any]) -> str:
    stdout = str(result.get("stdout") or result.get("stdout_tail") or "")
    stderr = str(result.get("stderr") or result.get("stderr_tail") or "")
    return stdout + stderr


def run_colmap_on_images(
    image_dir: Path,
    work_dir: Path,
    *,
    gpu_check: Callable[[str], dict[str, Any]],
    command_runner: Callable[[list[str]], dict[str, Any]],
    analyzer: Callable[[Path], dict[str, Any]] | None = None,
    matcher: str = MATCHER,
) -> dict[str, Any]:
    """Run the COLMAP commands after one memory check each.

    ``pose_commands`` chooses the matcher. This function does not swap in
    vocab-tree matching. A failed check raises :class:`GpuHeadroomError`
    and does not run that command. A non-zero extractor or matcher is
    returned as a failure with its stderr and is not retried. The mapper
    runs once per rung of :func:`mapper_ladder` until the largest model
    passes the gate; every rung is recorded in ``mapper_attempts``, the
    passing one is ``mapper_rung``, and a model that never passes carries a
    likely cause read from the database (:func:`diagnose`).
    """
    image_dir = Path(image_dir)
    work_dir = Path(work_dir)
    sparse = work_dir / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)
    database = work_dir / "database.db"
    banner = command_runner(["colmap", "-h"])
    banner_text = _command_text(banner)
    if banner.get("returncode") != 0:
        detail = banner_text.strip() or "colmap -h failed"
        return {
            "ok": False,
            "use_gpu": None,
            "reason": detail,
            "failed_step": "colmap",
            "stderr": banner_text[-2000:],
            "returncode": banner.get("returncode"),
            "gate_document": None,
            "registered_names": {},
            "name_errors": {},
        }
    use_gpu = use_gpu_from_banner(banner_text)
    major = colmap_major(banner_text)
    commands = pose_commands(
        database, image_dir, sparse, use_gpu, major=major, matcher=matcher
    )
    *prepare, mapper_command = commands
    for argv in prepare:
        if argv[1] not in {matcher, "feature_extractor"}:
            raise ValueError(f"unexpected COLMAP step {argv[1]}")
        gpu_check("colmap")
        step = command_runner([str(part) for part in argv])
        if step.get("returncode") != 0:
            return _step_failure(argv[1], step, use_gpu)
    # Video frames are JPG; GraspClutter6D stills are PNG and are not re-encoded.
    frame_count = len(list(image_dir.glob("*.jpg"))) + len(
        list(image_dir.glob("*.png"))
    )
    focal_prior: dict[str, Any] | None = None
    if matcher == MATCHER:
        # A video has no EXIF focal; start its camera at the focal its own
        # matches imply rather than COLMAP's 1.2 * size guess (focal_prior.py).
        focal_prior = self_calibrate(database)
    mapper_argv = [str(part) for part in mapper_command]
    attempts: list[dict[str, Any]] = []
    gate: dict[str, Any] = {}
    # Every rung maps the same matches from an empty sparse/; only a model
    # that fails the gate goes on to the next rung.
    for rung in mapper_ladder(matcher):
        shutil.rmtree(sparse, ignore_errors=True)
        sparse.mkdir(parents=True)
        gpu_check("colmap")
        step = command_runner([*mapper_argv, *rung[1]])
        if step.get("returncode") != 0:
            if not attempts:
                # The plain mapper itself failed (a crash or a timeout): no
                # model exists to judge, so this is a step failure.
                return _step_failure("mapper", step, use_gpu)
            attempts.append(
                {
                    "rung": rung[0],
                    "flags": list(rung[1]),
                    "returncode": step.get("returncode"),
                    "error": _command_text(step).strip()[-300:],
                }
            )
            break
        gate = accept_reconstruction(
            sparse,
            frame_count,
            work_dir / EVIDENCE_DIRNAME,
            runner=analyzer,
        )
        attempt = _mapper_attempt(rung, gate, _command_text(step), database)
        attempt["camera"] = _largest_camera(sparse)
        attempts.append(attempt)
        if gate["pass"]:
            break
    reason = gate.get("reason") or "no mapper rung produced a model"
    if not gate.get("pass"):
        reason = f"{reason} | likely cause: {diagnose(database, attempts)}"
    if len(attempts) > 1 or not gate.get("pass"):
        reason = f"{reason} | mapper attempts: {attempt_summary(attempts)}"
    registered: dict[str, list[str] | None] = {}
    name_errors: dict[str, str] = {}
    if sparse.is_dir():
        for model in sorted(path for path in sparse.iterdir() if path.is_dir()):
            blob_path = model / "images.bin"
            if not blob_path.is_file():
                registered[model.name] = None
                name_errors[model.name] = "images.bin missing"
                continue
            try:
                registered[model.name] = registered_image_names(blob_path.read_bytes())
            except ValueError as exc:
                registered[model.name] = None
                name_errors[model.name] = str(exc)
    passed = bool(gate.get("pass"))
    winner = next((a["rung"] for a in attempts if a.get("pass")), None)
    return {
        "ok": passed,
        "use_gpu": use_gpu,
        "reason": reason,
        "mapper_attempts": attempts,
        "mapper_rung": winner,
        "focal_prior": focal_prior,
        "failed_step": None if passed else "reconstruction_gate",
        "stderr": "",
        "returncode": 0,
        "gate_document": gate.get("document"),
        "registered_names": registered,
        "name_errors": name_errors,
    }


def _step_failure(name: str, step: dict[str, Any], use_gpu: bool) -> dict[str, Any]:
    detail = _command_text(step).strip() or f"{name} failed"
    return {
        "ok": False,
        "use_gpu": use_gpu,
        "reason": detail,
        "failed_step": name,
        "stderr": _command_text(step)[-2000:],
        "returncode": step.get("returncode"),
        "gate_document": None,
        "registered_names": {},
        "name_errors": {},
    }


def _models_by_size(sparse: Path) -> list[Path]:
    """Model directories, the one with the most registered images first."""
    sized = []
    for model in sorted(path for path in sparse.iterdir() if path.is_dir()):
        try:
            count = len(registered_image_names((model / "images.bin").read_bytes()))
        except (OSError, ValueError):
            continue
        sized.append((-count, model.name, model))
    return [model for _, _, model in sorted(sized)]


def _largest_camera(sparse: Path) -> dict[str, Any] | None:
    """The first camera of the largest model, or None when unreadable."""
    for model in _models_by_size(sparse)[:1]:
        try:
            cameras = read_cameras_bin((model / "cameras.bin").read_bytes())
        except (OSError, ValueError):
            return None
        return cameras[0] if cameras else None
    return None


def _image_names(database: Path) -> dict[int, str]:
    """Image id to file name, read from the COLMAP database; empty if unreadable."""
    import sqlite3

    try:
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as con:
            return {
                int(i): str(n)
                for i, n in con.execute("SELECT image_id, name FROM images")
            }
    except sqlite3.Error:
        return {}


def _mapper_attempt(
    rung: tuple[str, tuple[str, ...]],
    gate: dict[str, Any],
    mapper_text: str,
    database: Path,
) -> dict[str, Any]:
    """One mapper run: its flags, the largest model's numbers and its initial pairs."""
    bar = (gate.get("document") or {}).get("quality_bar") or {}
    names = _image_names(database) if mapper_text else {}
    pairs = init_pairs_from_log(mapper_text)
    return {
        "rung": rung[0],
        "flags": list(rung[1]),
        "pass": bool(gate.get("pass")),
        "registered": (bar.get("registered") or {}).get("value"),
        "inputs": bar.get("inputs"),
        "mean_obs_per_image": (bar.get("mean_obs_per_image") or {}).get("value"),
        "mean_reproj_err_px": (bar.get("mean_reproj_err_px") or {}).get("value"),
        "init_pairs": [
            [names.get(a, str(a)), names.get(b, str(b))] for a, b in pairs[:5]
        ],
        "init_pairs_tried": len(pairs),
        # Why this rung did not end the ladder (None when it passed).
        "gate_reason": None if gate.get("pass") else gate.get("reason"),
    }


def attempt_summary(attempts: list[dict[str, Any]]) -> str:
    """One line per mapper run for the ledger reason, which a failed run publishes."""
    parts = []
    for item in attempts:
        if "error" in item:
            parts.append(f"{item['rung']} exited {item['returncode']}")
            continue
        first = item["init_pairs"][0] if item["init_pairs"] else None
        start = f", first pair {first[0]}/{first[1]}" if first else ""
        parts.append(
            f"{item['rung']} {item['registered']}/{item['inputs']}"
            f"{start}, {item['init_pairs_tried']} pairs tried"
        )
    return "; ".join(parts)


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _blank_evidence(interpreter: str) -> dict[str, Any]:
    return {
        "checked_out_commit": None,
        "checked_out_commit_source": "not measured",
        "python": interpreter,
        "retries": 0,
        "matcher": MATCHER,
        "gpu": {
            "colmap_min_free_mib": COLMAP_MIN_FREE_MIB,
            "splatfacto_min_free_mib": SPLATFACTO_MIN_FREE_MIB,
            "splatfacto_status": "not_run",
            "retries": 0,
        },
        "sampling": None,
        "quality_bar": None,
        "unregistered_frames": None,
        "pass": False,
        "fail_reasons": [],
    }


def run_run4(
    video: Path | str,
    out_dir: Path | str,
    *,
    commit_sha: str = "",
    interpreter: str | None = None,
    frames: Sequence[dict[str, Any]],
    duration_s: float,
    gpu_query: Callable[[list[str]], GpuCommandResult] | None = None,
    run_colmap: Callable[[list[str], list[dict[str, Any]]], dict[str, Any]],
    inspect: Callable[[Path | str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Sample, record the GPU check, and write the run4 evidence.

    ``run_colmap`` receives the selected file names and the bin rows. Tests
    pass a fake. The CLI passes :func:`run_colmap_on_images`. A memory
    failure is stored with the query's return code and stderr and is not
    retried. splatfacto is not started.
    """
    pinned = require_simfoundry_python(interpreter)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    reader = inspect_video if inspect is None else inspect
    info = dict(reader(video))
    write_plan(out, Path(video).stem, VIDEO_SOURCE, video_plan_steps())
    evidence = _blank_evidence(pinned)
    pose_decision = "reject"
    pose_reason = ""
    use_gpu: bool | None = None
    mapper_rung: str | None = None
    failed_step: str | None = None
    try:
        sha = checked_out_commit(commit_sha)
        if sha is not None:
            evidence["checked_out_commit"] = sha
            evidence["checked_out_commit_source"] = (
                "git rev-parse HEAD of the payload checkout"
            )
        gpu = require_gpu_headroom("colmap", query=gpu_query)
        evidence["gpu"]["colmap"] = gpu
        selection, binning = select_frames(
            frames, duration_s, bin_count(len(frames), BIN_COUNT)
        )
        names = [f"{i:04d}.jpg" for i in range(1, len(selection) + 1)]
        evidence["sampling"] = {
            "bin_count": len(selection),
            "binning": binning,
            "duration_s": duration_s,
            "decoded_frames": len(frames),
            "selected_indices": [row["chosen_index"] for row in selection],
            "selected_names": names,
            "bins": selection,
            "definition": (
                "300 equal bins across the full duration (every frame when "
                "the clip has fewer); sharpest Laplacian variance at a long "
                "side of at most 1920 px; earliest index on ties; no truncation"
            ),
        }
        colmap_result = run_colmap(names, selection)
        if isinstance(colmap_result.get("use_gpu"), bool):
            # Recorded on a failure too: run 43469526711 left it null, and
            # nobody could tell whether SIFT had run on the GPU.
            use_gpu = colmap_result["use_gpu"]
        mapper_rung = colmap_result.get("mapper_rung")
        if colmap_result.get("focal_prior") is not None:
            evidence["focal_prior"] = colmap_result["focal_prior"]
        mapper_attempts = colmap_result.get("mapper_attempts")
        if mapper_attempts:
            evidence["mapper_attempts"] = mapper_attempts
        document = colmap_result.get("gate_document")
        if isinstance(document, dict):
            evidence["quality_bar"] = document.get("quality_bar")
            largest = document.get("chosen_model")
        else:
            largest = None
        evidence["unregistered_frames"] = unregistered_frame_report(
            names,
            colmap_result.get("registered_names") or {},
            colmap_result.get("name_errors") or {},
            largest if isinstance(largest, str) else None,
        )
        if colmap_result.get("ok") is True:
            evidence["pass"] = True
            pose_decision = "accept"
            info["reconstruction"] = "colmap"
        else:
            pose_reason = str(
                colmap_result.get("reason") or "reconstruction not accepted"
            )
            failed_step = colmap_result.get("failed_step")
            evidence["fail_reasons"].append(pose_reason)
            if colmap_result.get("stderr"):
                evidence["colmap_stderr"] = colmap_result["stderr"]
    except GpuHeadroomError as exc:
        evidence["gpu"]["colmap_error"] = {
            "message": str(exc),
            "returncode": exc.returncode,
            "stderr": exc.stderr,
            "stdout": exc.stdout,
            "retries": exc.retries,
            "sample": exc.sample,
        }
        evidence["fail_reasons"].append(str(exc))
        pose_reason = str(exc)
        failed_step = "gpu_headroom"
    except ValueError as exc:
        evidence["fail_reasons"].append(str(exc))
        pose_reason = str(exc)
        failed_step = failed_step or "run4"
    code = 0 if evidence["pass"] else 1
    pose_entry: dict[str, Any] = {
        "item": "COLMAP camera poses",
        "decision": pose_decision,
    }
    if evidence.get("mapper_attempts"):
        # Which mapper rung produced the poses, or each one that failed.
        pose_entry["mapper_attempts"] = [
            {
                k: item.get(k)
                for k in ("rung", "pass", "registered", "inputs", "gate_reason")
            }
            for item in evidence["mapper_attempts"]
        ]
        pose_entry["mapper_rung"] = mapper_rung
    if evidence.get("focal_prior") is not None:
        prior = evidence["focal_prior"]
        pose_entry["focal_prior"] = {
            k: prior.get(k) for k in ("ok", "focal_px", "pairs", "why") if k in prior
        }
    pose_entry["use_gpu"] = use_gpu
    if pose_decision == "accept":
        sampling = evidence["sampling"] or {}
        pose_entry["frames"] = sampling.get("selected_names")
    else:
        pose_entry["reason"] = pose_reason
        pose_entry["failed_step"] = failed_step
    ledger = {
        "scene_name": Path(video).stem,
        "source": VIDEO_SOURCE,
        "entries": [
            {
                "item": "video",
                "decision": "accept",
                "bytes": info.get("bytes"),
                "duration_s": info.get("duration_s"),
            },
            pose_entry,
        ],
    }
    _write(out / "video.json", info)
    _write(out / "ledger.json", ledger)
    _write(out / "run4_evidence.json", evidence)
    _write(
        out / "colmap_poses.json",
        {
            "ok": evidence["pass"],
            "matcher": MATCHER,
            "reason": pose_reason,
            "failed_step": failed_step,
            "use_gpu": use_gpu,
            "mapper_rung": mapper_rung,
        },
    )
    return {
        "code": code,
        "pass": evidence["pass"],
        "evidence": evidence,
        "out_dir": str(out),
    }


def load_frames(
    video: Path | str,
    *,
    runner: Callable[..., Any] | None = None,
    score_gray: Callable[[bytes], float] | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """Decode the whole video and score every frame.

    ``runner`` defaults to ``subprocess.run``. The decode command has no
    fps filter and no frame cap. ``score_gray`` defaults to the Laplacian
    variance of a Pillow grayscale image.
    """
    import shutil
    import subprocess
    import tempfile

    ffprobe = shutil.which("ffprobe")
    ffmpeg = shutil.which("ffmpeg")
    if not ffprobe or not ffmpeg:
        raise ValueError("ffmpeg and ffprobe are required; not installing them")
    run = subprocess.run if runner is None else runner
    probe = run(
        ffprobe_command(ffprobe, str(video)),
        check=False,
        capture_output=True,
        text=True,
    )
    if getattr(probe, "returncode", 1) != 0:
        raise ValueError(getattr(probe, "stderr", "") or "ffprobe failed")
    duration, stamps = parse_probe(json.loads(probe.stdout or "{}"))
    with tempfile.TemporaryDirectory() as tmp:
        pattern = str(Path(tmp) / "%06d.png")
        decoded = run(
            decode_all_command(ffmpeg, str(video), pattern),
            check=False,
            capture_output=True,
            text=True,
        )
        if getattr(decoded, "returncode", 1) != 0:
            raise ValueError(getattr(decoded, "stderr", "") or "ffmpeg decode failed")
        frames: list[dict[str, Any]] = []
        for index, stamp in enumerate(stamps):
            path = Path(tmp) / f"{index:06d}.png"
            if not path.is_file():
                raise ValueError("decoded frame count does not match ffprobe")
            raw = path.read_bytes()
            score = _score_png(raw) if score_gray is None else score_gray(raw)
            frames.append({"index": index, "time": stamp, "score": score})
    return frames, duration


def _score_png(raw: bytes) -> float:
    """``laplacian_variance`` of the grayscale frame, computed with NumPy.

    Same stencil and the same population-variance formula; the pure-Python
    loop took seconds per 1080p frame, and Mailbox.mp4 has 1,192 of them.
    """
    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(raw)) as image:
        gray = np.asarray(image.convert("L"), dtype=np.float64)
    if gray.ndim != 2 or gray.shape[0] < 3 or gray.shape[1] < 3:
        raise ValueError("frame too small")
    lap = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    mean = float(lap.mean())
    variance = float((lap * lap).mean()) - mean * mean
    if not math.isfinite(variance) or variance < 0:
        raise ValueError("invalid sharpness")
    return variance


def extract_selected_frames(
    video: Path | str,
    indices: Sequence[int],
    image_dir: Path,
    *,
    runner: Callable[..., Any] | None = None,
) -> None:
    """Write ``0001.jpg`` onward for ``indices`` only."""
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValueError("ffmpeg is required; not installing it")
    image_dir.mkdir(parents=True, exist_ok=True)
    run = subprocess.run if runner is None else runner
    completed = run(
        extract_selected_command(
            ffmpeg, str(video), indices, str(image_dir / "%04d.jpg")
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if getattr(completed, "returncode", 1) != 0:
        raise ValueError(getattr(completed, "stderr", "") or "ffmpeg extract failed")


def _default_colmap(
    video: Path | str,
    image_dir: Path,
    names: list[str],
    selection: list[dict[str, Any]],
) -> dict[str, Any]:
    del names
    extract_selected_frames(
        video, [int(row["chosen_index"]) for row in selection], image_dir
    )

    def gpu_check(stage: str) -> dict[str, Any]:
        return require_gpu_headroom(stage)

    return run_colmap_on_images(
        image_dir, image_dir.parent, gpu_check=gpu_check, command_runner=run_colmap_step
    )


def run_colmap_step(
    argv: list[str], *, runner: Callable[..., Any] | None = None
) -> dict[str, Any]:
    """Run one COLMAP command. A timeout is a failed step, not a crash.

    Without this, a timeout raised past ``run_run4`` and no ledger was
    written, so the run reported a missing ledger instead of the timeout.
    """
    import subprocess

    run = runner or subprocess.run
    step = argv[1] if len(argv) > 1 else ""
    timeout = STEP_TIMEOUT_S.get(step, DEFAULT_STEP_TIMEOUT_S)
    try:
        completed = run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=stage_environment(),
        )
    except subprocess.TimeoutExpired as exc:
        stderr = _as_text(exc.stderr)
        return {
            "command": argv,
            "returncode": TIMEOUT_RETURNCODE,
            "stdout": _as_text(exc.stdout),
            "stderr": f"{stderr.rstrip()}\n{step} timed out after {timeout} s".lstrip(),
        }
    return _run_result(completed)


def _as_text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def _run_result(completed: Any) -> dict[str, Any]:
    return {
        "command": completed.args,
        "returncode": completed.returncode,
        "stdout": completed.stdout or "",
        "stderr": completed.stderr or "",
    }


def main(argv: list[str] | None = None) -> int:
    """Run run4. Exit 127 when the pinned interpreter is missing."""
    import argparse

    parser = argparse.ArgumentParser(description="SF-4 run4 Fruits pipeline")
    parser.add_argument("capture", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--commit-sha", default="")
    args = parser.parse_args(argv)
    try:
        require_simfoundry_python()
    except PinnedInterpreterMissing as exc:
        print(str(exc), file=sys.stderr)
        return 127
    try:
        frames, duration = load_frames(args.capture)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    image_dir = args.out / "images"

    def colmap(names: list[str], selection: list[dict[str, Any]]) -> dict[str, Any]:
        return _default_colmap(args.capture, image_dir, names, selection)

    result = run_run4(
        args.capture,
        args.out,
        commit_sha=args.commit_sha,
        frames=frames,
        duration_s=duration,
        run_colmap=colmap,
    )
    return int(result["code"])


if __name__ == "__main__":
    raise SystemExit(main())
