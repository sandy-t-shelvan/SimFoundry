"""Section 3.2 camera poses, using the COLMAP binary when it is installed.

The PDF names COLMAP and does not publish extractor settings. The commands
below are COLMAP 3.7's sequential pipeline for one video camera. A banner
that says "without CUDA" forces SIFT onto the CPU. A missing binary is not
a reconstruction, and this stage does not train a splat.

The frame sample defaults to 1 fps and 16 frames. That cap is this runner's.
``max_frames`` None or 0 omits it, and ``fps`` None or 0 omits the fps
filter. Both together keep every decoded frame. Omitting the filter is not
the SF-5 evidence command: ``native_frames=True`` asks ffmpeg for
``-fps_mode passthrough`` and rejects an fps filter or a frame cap (PM
KAN-5 10314). Sampled frames are color-calibrated per image before
matching. That scale is our diagonal gain toward
``per_channel_median_of_set``, not the paper's exact method, so the
section 3.2 color gap stays partial.

A mapper that writes a sparse model is not accepted until
``reconstruction_gate`` passes. The gate runs ``model_analyzer`` on every
subdirectory of ``sparse/``, not on ``sparse/0`` alone, and rejects the
reconstruction when the largest model misses the bar. A double tie, the
same registered count and the same points count, fails closed
(``ambiguous_largest_model``) in either directory order. That is the only
departure from collector v9; PM KAN-5 comment 10275.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

from sceneagent.ingest_assets import apply_per_image_color_calibration
from sceneagent.reconstruction_gate import EVIDENCE_DIRNAME, accept_reconstruction


def use_gpu_from_banner(banner):
    """True only when the COLMAP banner does not say the build lacks CUDA."""
    return "without CUDA" not in banner


def colmap_major(banner):
    """Major version from a ``colmap -h`` banner. Unknown banners stay on 3."""
    for token in banner.split():
        if token[:1].isdigit() and "." in token:
            return int(token.split(".", 1)[0])
    return 3


def pose_commands(database, image_dir, sparse_dir, use_gpu, major=3):
    """Three sequential steps. ``use_gpu`` is the SIFT device flag, 0 or 1.

    COLMAP 3.7 names the device ``SiftExtraction.use_gpu``. COLMAP 4.2 names
    it ``FeatureExtraction.use_gpu`` and rejects ``--log_to_stderr``.
    """
    gpu = "1" if use_gpu else "0"
    database = str(database)
    image_dir = str(image_dir)
    sparse_dir = str(sparse_dir)
    if int(major) >= 4:
        return [
            [
                "colmap",
                "feature_extractor",
                "--database_path",
                database,
                "--image_path",
                image_dir,
                "--ImageReader.single_camera",
                "1",
                "--FeatureExtraction.use_gpu",
                gpu,
            ],
            [
                "colmap",
                "sequential_matcher",
                "--database_path",
                database,
                "--FeatureMatching.use_gpu",
                gpu,
                "--SequentialMatching.overlap",
                "5",
            ],
            [
                "colmap",
                "mapper",
                "--database_path",
                database,
                "--image_path",
                image_dir,
                "--output_path",
                sparse_dir,
            ],
        ]
    return [
        [
            "colmap",
            "feature_extractor",
            "--log_to_stderr",
            "1",
            "--database_path",
            database,
            "--image_path",
            image_dir,
            "--ImageReader.single_camera",
            "1",
            "--SiftExtraction.use_gpu",
            gpu,
            "--SiftExtraction.max_image_size",
            "1280",
        ],
        [
            "colmap",
            "sequential_matcher",
            "--log_to_stderr",
            "1",
            "--database_path",
            database,
            "--SiftMatching.use_gpu",
            gpu,
            "--SequentialMatching.overlap",
            "5",
        ],
        [
            "colmap",
            "mapper",
            "--log_to_stderr",
            "1",
            "--database_path",
            database,
            "--image_path",
            image_dir,
            "--output_path",
            sparse_dir,
        ],
    ]


def _flag_value(argv, name):
    try:
        index = list(argv).index(name)
    except ValueError:
        return None
    if index + 1 >= len(argv):
        return None
    return str(argv[index + 1])


def command_uses_gpu(argv, *, cuda_build):
    """Whether this argv is a GPU stage for a memory check.

    Extractor and matcher follow the use_gpu flag they actually carry.
    Mapper follows ``--Mapper.ba_use_gpu`` when the argv sets it. An
    explicit 0 is CPU bundle adjustment. When the flag is absent and the
    banner reported CUDA, the stage is still checked: this tree has not
    verified the COLMAP 3.11.1 default, and current public docs are not
    that pin.
    """
    step = argv[1] if len(argv) > 1 else ""
    if step == "feature_extractor":
        value = _flag_value(argv, "--SiftExtraction.use_gpu")
        if value is None:
            value = _flag_value(argv, "--FeatureExtraction.use_gpu")
        return value == "1"
    if step == "sequential_matcher":
        value = _flag_value(argv, "--SiftMatching.use_gpu")
        if value is None:
            value = _flag_value(argv, "--FeatureMatching.use_gpu")
        return value == "1"
    if step == "mapper":
        ba = _flag_value(argv, "--Mapper.ba_use_gpu")
        if ba == "0":
            return False
        if ba == "1":
            return True
        return bool(cuda_build)
    return False


def poll_gpu_stage(poll, on_sample, *, interval_s, timeout_s, sleeper, monotonic):
    """Poll a running stage. Call ``on_sample`` every ``interval_s``.

    The stage is not retried. ``poll`` returns None while it is running.
    """
    start = monotonic()
    waited = 0.0
    while True:
        state = poll()
        if state is not None:
            return {"status": "exited", "state": state}
        elapsed = monotonic() - start
        if elapsed >= timeout_s:
            return {"status": "timeout", "state": None}
        slice_s = min(0.25, interval_s - waited, timeout_s - elapsed)
        if slice_s <= 0:
            return {"status": "timeout", "state": None}
        sleeper(slice_s)
        waited += slice_s
        if waited + 1e-9 >= interval_s:
            waited = 0.0
            if poll() is None:
                on_sample()


def _no_limit(value):
    """True when the caller asked to keep every frame.

    ``False`` compares equal to ``0``. A boolean has to be rejected before
    that comparison, or ``fps=False`` would silently drop the fps filter.
    """
    if isinstance(value, bool):
        raise ValueError(
            "fps and max_frames must be numbers, or None or 0 for every frame"
        )
    return value is None or value == 0


def _positive_rate(fps):
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps <= 0:
        raise ValueError("fps must be positive, or None or 0 for every decoded frame")
    return fps


def _frame_cap(max_frames):
    if isinstance(max_frames, bool) or not isinstance(max_frames, (int, float)):
        raise ValueError(
            "max_frames must be a positive int, or None or 0 for every frame"
        )
    if isinstance(max_frames, float) and not max_frames.is_integer():
        raise ValueError(
            "max_frames must be a positive int, or None or 0 for every frame"
        )
    cap = int(max_frames)
    if cap <= 0:
        raise ValueError(
            "max_frames must be a positive int, or None or 0 for every frame"
        )
    return cap


def sample_frame_command(
    ffmpeg, video_path, image_pattern, fps=1, max_frames=16, *, native_frames=False
):
    """ffmpeg argv for the frame sample.

    Defaults are this runner's 1 fps and 16-frame cap. The paper does not
    publish either number. ``None`` or ``0`` drops that limit. ffmpeg treats
    ``-frames:v 0`` as an empty extract and ``fps=0`` as an invalid filter,
    so those values omit the flag instead of being forwarded. Omitting both
    keeps every decoded frame on ffmpeg's default image muxer. Omitting only
    the cap still applies ``fps``.

    ``native_frames=True`` is the SF-5 evidence extract (PM KAN-5 10314):
    every native frame, no subsampling, and no cap. ``-fps_mode passthrough``
    is that request. A positive ``fps`` or ``max_frames`` is rejected, so a
    native-rate ``fps=30`` filter cannot stand in for it. The default and
    the omit-only command are unchanged.
    """
    if not isinstance(native_frames, bool):
        raise ValueError("native_frames must be a bool")
    if native_frames:
        if not _no_limit(fps) or not _no_limit(max_frames):
            raise ValueError(
                "native_frames keeps every native frame and rejects an fps "
                "filter or a frame cap"
            )
        return [
            str(ffmpeg),
            "-y",
            "-i",
            str(video_path),
            "-fps_mode",
            "passthrough",
            str(image_pattern),
        ]
    command = [str(ffmpeg), "-y", "-i", str(video_path)]
    if not _no_limit(fps):
        command.extend(["-vf", f"fps={_positive_rate(fps)}"])
    if not _no_limit(max_frames):
        command.extend(["-frames:v", str(_frame_cap(max_frames))])
    command.append(str(image_pattern))
    return command


def validate_frame_limits(fps=1, max_frames=16):
    """Raise ValueError when ``fps`` or ``max_frames`` is not a supported limit."""
    sample_frame_command("ffmpeg", "in", "out.jpg", fps=fps, max_frames=max_frames)


def _request_is_full_rate(fps, max_frames):
    """True only when both the fps filter and the frame cap are off."""
    validate_frame_limits(fps, max_frames)
    return _no_limit(fps) and _no_limit(max_frames)


def run_poses(
    video_path,
    work_dir,
    fps=1,
    max_frames=16,
    color_calibrate=True,
    *,
    analyzer=None,
    native_frames=False,
    before_gpu_stage=None,
    on_gpu_sample=None,
    sample_interval_s=10,
    ffmpeg_timeout_s=180,
    colmap_timeout_s=900,
    after_extract=None,
):
    """Sample frames, calibrate each image, and run COLMAP.

    The 1 fps sample and the 16-frame cap are this runner's, not the paper's.
    ``max_frames`` None or 0 omits ``-frames:v``. ``fps`` None or 0 omits the
    fps filter. Setting both keeps every decoded frame.
    ``native_frames=True`` uses the passthrough extract and rejects a filter
    or a cap. The SF-5 evidence runner sets it. Other callers stay on the
    defaults.

    ``color_calibrate`` scales each extracted image by its own gain before
    ``feature_extractor``. Section 3.2 names per-image color calibration and
    does not publish the correction, so the diagonal scale is our choice and
    the color gap stays partial. ``color_calibration.json`` names the target
    ``per_channel_median_of_set``, records its three values, and maps each
    image name to ``[gR, gG, gB]``. Pass false to leave the sampled pixels
    unchanged.

    The reconstruction is accepted only when the quality bar passes on the
    largest sparse model. ``analyzer``, when passed, is called with each
    model directory and returns ``returncode``, ``stdout``, and ``stderr``.
    The default runs ``colmap model_analyzer``. Tests inject this so the
    gate does not need a COLMAP binary.
    """
    if not isinstance(native_frames, bool):
        raise ValueError("native_frames must be a bool")
    # Reject a contradictory cap before any work directory is deleted.
    if native_frames:
        sample_frame_command(
            "ffmpeg",
            "in",
            "out.jpg",
            fps=fps,
            max_frames=max_frames,
            native_frames=True,
        )
    full_rate = _request_is_full_rate(fps, max_frames)
    calibrated: dict[str, object] = {}

    def _result(**payload):
        payload["fps"] = fps
        payload["max_frames"] = max_frames
        payload["color_calibrate"] = bool(color_calibrate)
        payload["full_rate"] = full_rate
        payload["native_frames"] = native_frames
        if "report" in calibrated:
            payload["color_calibration"] = calibrated["report"]
        return payload

    colmap = shutil.which("colmap")
    ffmpeg = shutil.which("ffmpeg")
    if not colmap or not ffmpeg:
        return _result(
            ran=False,
            ok=False,
            reason="colmap or ffmpeg is not on PATH",
            colmap=colmap,
            ffmpeg=ffmpeg,
        )
    work = Path(work_dir)
    images = work / "images"
    sparse = work / "sparse"
    if images.exists():
        shutil.rmtree(images)
    images.mkdir(parents=True, exist_ok=True)
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir(parents=True, exist_ok=True)
    database = work / "database.db"
    if database.exists():
        database.unlink()
    # A report from an earlier call would describe frames this call just deleted.
    calibration_path = work / "color_calibration.json"
    if calibration_path.exists():
        calibration_path.unlink()
    # A previous acceptance must not survive a run that never reaches the gate.
    evidence_dir = work / EVIDENCE_DIRNAME
    if evidence_dir.exists():
        shutil.rmtree(evidence_dir)
    # JPEG stays. The q95 4:4:4 round trip on the recorded fixture moves
    # channel means by 0.0185546875 levels, under the 1.5-level PNG switch.
    extract = _run(
        sample_frame_command(
            ffmpeg,
            video_path,
            images / "%04d.jpg",
            fps=fps,
            max_frames=max_frames,
            native_frames=native_frames,
        ),
        timeout_s=ffmpeg_timeout_s,
    )
    frames = sorted(path.name for path in images.glob("*.jpg"))
    if extract["returncode"] != 0 or not frames:
        if extract.get("timed_out"):
            reason = "ffmpeg timed out"
        elif extract.get("returncode") not in (0, None):
            reason = "ffmpeg failed"
        else:
            reason = "ffmpeg wrote no frames"
        return _result(
            ran=True,
            ok=False,
            failed_step="ffmpeg",
            frames=frames,
            steps={"ffmpeg": extract},
            reason=reason,
        )
    if after_extract is not None:
        rejection = after_extract(list(frames))
        if rejection:
            return _result(
                ran=True,
                ok=False,
                failed_step="frame_count",
                frames=frames,
                steps={"ffmpeg": extract},
                reason=str(rejection),
            )
    if color_calibrate:
        try:
            report = apply_per_image_color_calibration(images)
            calibration_path.write_text(
                json.dumps(report, indent=2) + "\n",
                encoding="utf-8",
            )
        except (ValueError, OSError) as exc:
            return _result(
                ran=True,
                ok=False,
                failed_step="color_calibration",
                frames=frames,
                steps={"ffmpeg": extract},
                reason=str(exc),
            )
        calibrated["report"] = report
    banner = subprocess.run([colmap, "-h"], check=False, capture_output=True, text=True)
    banner_text = (banner.stdout or "") + (banner.stderr or "")
    use_gpu = use_gpu_from_banner(banner_text)
    major = colmap_major(banner_text)
    steps = {"ffmpeg": extract}
    for argv in pose_commands(database, images, sparse, use_gpu, major=major):
        step_name = argv[1]
        gpu_stage = command_uses_gpu(argv, cuda_build=use_gpu)
        if gpu_stage and before_gpu_stage is not None:
            before_gpu_stage(step_name, list(argv))
        if gpu_stage and on_gpu_sample is not None:
            steps[step_name] = _run(
                argv,
                timeout_s=colmap_timeout_s,
                on_gpu_sample=on_gpu_sample,
                sample_interval_s=sample_interval_s,
            )
        else:
            steps[step_name] = _run(argv, timeout_s=colmap_timeout_s)
        if steps[step_name]["returncode"] != 0:
            if steps[step_name].get("timed_out"):
                reason = f"{step_name} timed out"
            else:
                reason = (
                    f"{step_name} failed with return code "
                    f"{steps[step_name]['returncode']}"
                )
            return _result(
                ran=True,
                ok=False,
                use_gpu=use_gpu,
                colmap_major=major,
                failed_step=step_name,
                frames=frames,
                steps=steps,
                reason=reason,
            )
    # No subdirectory means the mapper wrote nothing. That stays a mapper
    # failure. A model that exists is accepted only by the quality bar, and
    # the bar reads every subdirectory rather than sparse/0 alone.
    wrote_model = any(path.is_dir() for path in sparse.iterdir())
    gate = accept_reconstruction(sparse, len(frames), evidence_dir, runner=analyzer)
    if not gate["pass"]:
        return _result(
            ran=True,
            ok=False,
            use_gpu=use_gpu,
            colmap_major=major,
            failed_step="mapper" if not wrote_model else "reconstruction_gate",
            frames=frames,
            reason=gate["reason"],
            steps=steps,
            reconstruction_gate=gate["document"],
        )
    return _result(
        ran=True,
        ok=True,
        use_gpu=use_gpu,
        colmap_major=major,
        frames=frames,
        model_dir=str(sparse / gate["model"]),
        steps=steps,
        reason="",
        reconstruction_gate=gate["document"],
    )


def _stage_env():
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    return env


def _read_stream(stream, chunks):
    """Drain one pipe so a logging child cannot block on a full buffer."""
    try:
        while True:
            piece = stream.read(4096)
            if not piece:
                break
            chunks.append(piece)
    except (OSError, ValueError):
        return


def _start_stream_readers(proc):
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    threads = (
        threading.Thread(
            target=_read_stream, args=(proc.stdout, stdout_chunks), daemon=True
        ),
        threading.Thread(
            target=_read_stream, args=(proc.stderr, stderr_chunks), daemon=True
        ),
    )
    for thread in threads:
        thread.start()
    return threads, stdout_chunks, stderr_chunks


def _join_streams(threads, stdout_chunks, stderr_chunks):
    for thread in threads:
        thread.join(timeout=5)
    return "".join(stdout_chunks), "".join(stderr_chunks)


def _stop_our_child(proc):
    """Stop the direct child this process started.

    Only ``proc`` is signaled. An unrelated process is not. A grandchild
    that this child has already forked is not addressed: an orphaned
    grandchild can survive after the direct child exits. Process-group
    signaling is not a requirement of this package, and this function does
    not scan the process table or signal any other pid.
    """
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _run(command, timeout_s=600, *, on_gpu_sample=None, sample_interval_s=10):
    env = _stage_env()
    if on_gpu_sample is None:
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            return {
                "command": command,
                "returncode": None,
                "timed_out": True,
                "stdout_tail": (stdout or "")[-2000:],
                "stderr_tail": (stderr or "")[-2000:],
                "gpu_samples": [],
            }
        return {
            "command": command,
            "returncode": completed.returncode,
            "timed_out": False,
            "stdout_tail": (completed.stdout or "")[-2000:],
            "stderr_tail": (completed.stderr or "")[-2000:],
            "gpu_samples": [],
        }
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    readers, stdout_chunks, stderr_chunks = _start_stream_readers(proc)
    samples = []

    def on_sample():
        samples.append(on_gpu_sample())

    try:
        watched = poll_gpu_stage(
            proc.poll,
            on_sample,
            interval_s=sample_interval_s,
            timeout_s=timeout_s,
            sleeper=time.sleep,
            monotonic=time.monotonic,
        )
    except Exception:
        _stop_our_child(proc)
        _join_streams(readers, stdout_chunks, stderr_chunks)
        raise
    timed_out = watched["status"] == "timeout"
    if timed_out:
        _stop_our_child(proc)
    else:
        proc.wait()
    stdout, stderr = _join_streams(readers, stdout_chunks, stderr_chunks)
    return {
        "command": command,
        "returncode": None if timed_out else proc.returncode,
        "timed_out": timed_out,
        "stdout_tail": (stdout or "")[-2000:],
        "stderr_tail": (stderr or "")[-2000:],
        "gpu_samples": samples,
    }
