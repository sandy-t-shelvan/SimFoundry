"""Section 3.2 camera poses, using the COLMAP binary when it is installed.

The PDF names COLMAP and does not publish extractor settings. The commands
below are COLMAP 3.7's sequential pipeline for one video camera. A banner
that says "without CUDA" forces SIFT onto the CPU. A missing binary is not
a reconstruction, and this stage does not train a splat.

The frame sample defaults to 1 fps and 16 frames. That cap is this runner's.
``max_frames`` None or 0 omits it, and ``fps`` None or 0 omits the fps
filter. Both together keep every decoded frame. Sampled frames are
color-calibrated per image before matching. That scale is our diagonal
gain toward ``per_channel_median_of_set``, not the paper's exact method,
so the section 3.2 color gap stays partial.

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


def sample_frame_command(ffmpeg, video_path, image_pattern, fps=1, max_frames=16):
    """ffmpeg argv for the frame sample.

    Defaults are this runner's 1 fps and 16-frame cap. The paper does not
    publish either number. ``None`` or ``0`` drops that limit. ffmpeg treats
    ``-frames:v 0`` as an empty extract and ``fps=0`` as an invalid filter,
    so those values omit the flag instead of being forwarded. Omitting both
    keeps every decoded frame. Omitting only the cap still applies ``fps``.
    """
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
    video_path, work_dir, fps=1, max_frames=16, color_calibrate=True, *, analyzer=None
):
    """Sample frames, calibrate each image, and run COLMAP.

    The 1 fps sample and the 16-frame cap are this runner's, not the paper's.
    ``max_frames`` None or 0 omits ``-frames:v``. ``fps`` None or 0 omits the
    fps filter. Setting both keeps every decoded frame.

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
    full_rate = _request_is_full_rate(fps, max_frames)
    calibrated: dict[str, object] = {}

    def _result(**payload):
        payload["fps"] = fps
        payload["max_frames"] = max_frames
        payload["color_calibrate"] = bool(color_calibrate)
        payload["full_rate"] = full_rate
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
        ),
        timeout_s=180,
    )
    frames = sorted(path.name for path in images.glob("*.jpg"))
    if extract["returncode"] != 0 or not frames:
        return _result(
            ran=True,
            ok=False,
            failed_step="ffmpeg",
            frames=frames,
            steps={"ffmpeg": extract},
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
        steps[step_name] = _run(argv, timeout_s=900)
        if steps[step_name]["returncode"] != 0:
            return _result(
                ran=True,
                ok=False,
                use_gpu=use_gpu,
                colmap_major=major,
                failed_step=step_name,
                frames=frames,
                steps=steps,
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


def _run(command, timeout_s=600):
    try:
        env = os.environ.copy()
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
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
        }
    return {
        "command": command,
        "returncode": completed.returncode,
        "timed_out": False,
        "stdout_tail": (completed.stdout or "")[-2000:],
        "stderr_tail": (completed.stderr or "")[-2000:],
    }
