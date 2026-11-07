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
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from sceneagent.colmap_stage import (
    colmap_major,
    pose_commands,
    use_gpu_from_banner,
)
from sceneagent.gpu_headroom import (
    COLMAP_MIN_FREE_MIB,
    SPLATFACTO_MIN_FREE_MIB,
    GpuCommandResult,
    GpuHeadroomError,
    require_gpu_headroom,
)
from sceneagent.pipeline import write_plan
from sceneagent.reconstruction_gate import EVIDENCE_DIRNAME, accept_reconstruction
from sceneagent.run4_report import registered_image_names, unregistered_frame_report
from sceneagent.run4_sampling import (
    BIN_COUNT,
    decode_all_command,
    extract_selected_command,
    ffprobe_command,
    parse_probe,
    select_sharpest_per_bin,
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
) -> dict[str, Any]:
    """Run the sequential COLMAP commands after one memory check each.

    ``pose_commands`` chooses the matcher. This function does not swap in
    vocab-tree matching. A failed check raises :class:`GpuHeadroomError`
    and does not run that command. A non-zero command is returned as a
    failure with its stderr. It is not retried.
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
    commands = pose_commands(database, image_dir, sparse, use_gpu, major=major)
    for argv in commands:
        if argv[1] != MATCHER and argv[1] not in {
            "feature_extractor",
            "mapper",
        }:
            raise ValueError(f"unexpected COLMAP step {argv[1]}")
        gpu_check("colmap")
        step = command_runner([str(part) for part in argv])
        if step.get("returncode") != 0:
            detail = _command_text(step).strip() or f"{argv[1]} failed"
            return {
                "ok": False,
                "use_gpu": use_gpu,
                "reason": detail,
                "failed_step": argv[1],
                "stderr": _command_text(step)[-2000:],
                "returncode": step.get("returncode"),
                "gate_document": None,
                "registered_names": {},
                "name_errors": {},
            }
    frame_count = len(list(image_dir.glob("*.jpg")))
    gate = accept_reconstruction(
        sparse,
        frame_count,
        work_dir / EVIDENCE_DIRNAME,
        runner=analyzer,
    )
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
    return {
        "ok": bool(gate["pass"]),
        "use_gpu": use_gpu,
        "reason": gate["reason"],
        "failed_step": None if gate["pass"] else "reconstruction_gate",
        "stderr": "",
        "returncode": 0,
        "gate_document": gate["document"],
        "registered_names": registered,
        "name_errors": name_errors,
    }


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
        selection = select_sharpest_per_bin(frames, duration_s, BIN_COUNT)
        names = [f"{i:04d}.jpg" for i in range(1, len(selection) + 1)]
        evidence["sampling"] = {
            "bin_count": len(selection),
            "duration_s": duration_s,
            "decoded_frames": len(frames),
            "selected_indices": [row["chosen_index"] for row in selection],
            "selected_names": names,
            "bins": selection,
            "definition": (
                "300 equal bins across the full duration; sharpest "
                "Laplacian variance; earliest index on ties; no truncation"
            ),
        }
        colmap_result = run_colmap(names, selection)
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
            use_gpu = (
                colmap_result.get("use_gpu")
                if isinstance(colmap_result.get("use_gpu"), bool)
                else None
            )
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
    if pose_decision == "accept":
        pose_entry["use_gpu"] = use_gpu
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
    from PIL import Image

    from sceneagent.run4_sampling import laplacian_variance

    with Image.open(io.BytesIO(raw)) as image:
        gray = image.convert("L")
        width, height = gray.size
        pixels = list(gray.getdata())
    rows = [[float(pixels[y * width + x]) for x in range(width)] for y in range(height)]
    return laplacian_variance(rows)


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
