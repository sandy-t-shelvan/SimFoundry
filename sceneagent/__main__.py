"""Run the phase-1 SceneAgent slice.

python -m sceneagent examples/phase1_place_scene.json --out runs/phase1_place
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sceneagent.pipeline import run_path


def _cli_rate(text):
    """Parse fps. Integral values stay ints so the ffmpeg text stays ``fps=1``."""
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("fps must be a number") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("fps must be >= 0")
    if value == int(value):
        return int(value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description="SceneAgent phase-1 conversion")
    parser.add_argument(
        "capture",
        type=Path,
        help="Capture JSON, SimFoundry scene JSON, or an example mp4",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help=(
            "Phrases for the video pipeline. A video with a committed objects "
            "verdict for its bytes (scene_objects.py) uses that when omitted."
        ),
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Directory for the ledger and report"
    )
    parser.add_argument(
        "--fps",
        type=_cli_rate,
        default=1,
        help=(
            "Video sample rate. 0 keeps every decoded frame. "
            "Default 1 is this runner's, not the paper's. Mp4 captures only."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=16,
        help=(
            "Video frame cap. 0 keeps every sampled frame. "
            "Default 16 is this runner's, not the paper's. Mp4 captures only."
        ),
    )
    parser.add_argument(
        "--no-color-calibration",
        action="store_true",
        help="Do not scale each sampled video frame before COLMAP.",
    )
    args = parser.parse_args(argv)
    if args.capture.suffix.lower() == ".mp4":
        from sceneagent import video_run

        # KAN-52 dc1: a video with a prompt (given, or a committed verdict for
        # its bytes, KAN-56) runs the same stage
        # list and output directory as ``python -m sceneagent.video_run``.
        # Other mp4s stay on the phase-1 sampler.
        from sceneagent.scene_objects import lookup_prompt

        prompt = args.prompt or lookup_prompt(args.capture)
        if prompt:
            if args.fps != 1 or args.max_frames != 16 or args.no_color_calibration:
                parser.error(
                    "--fps, --max-frames, and --no-color-calibration apply to "
                    "the phase-1 sampler, not the video pipeline"
                )
            forwarded = [str(args.capture), "--out", str(args.out)]
            if args.prompt:
                forwarded.extend(["--prompt", args.prompt])
            return video_run.main(forwarded)
        from sceneagent.video import run_video

        summary = run_video(
            args.capture,
            args.out,
            fps=args.fps,
            max_frames=args.max_frames,
            color_calibrate=not args.no_color_calibration,
        )
        print(
            f"video={summary['video']['name']} bytes={summary['video']['bytes']} "
            f"duration_s={summary['video']['duration_s']} colmap={summary['colmap_available']} "
            f"decision={summary['colmap_decision']} reconstruction={summary['reconstruction']} "
            f"out={summary['out_dir']}"
        )
        return 0
    if (
        args.prompt
        or args.fps != 1
        or args.max_frames != 16
        or args.no_color_calibration
    ):
        parser.error(
            "--prompt, --fps, --max-frames, and --no-color-calibration "
            "apply to an mp4 capture"
        )
    summary = run_path(args.capture, args.out)
    print(
        f"scene={summary['scene_name']} objects={summary['n_objects']} sisters={summary['n_sister_rows']} out={summary['out_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
