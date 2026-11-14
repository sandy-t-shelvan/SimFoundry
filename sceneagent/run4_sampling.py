# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sharpest frame in each of 300 bins across the whole clip.

The bins cover ``[0, duration]``. Every bin is half-open except the last,
which includes the right endpoint. Sharpness is the variance of the
interior 4-neighbor Laplacian. A tie keeps the earliest frame index. An
empty bin, a non-finite score, or a timestamp outside the clip fails.
Frames are not duplicated to fill a bin, and the clip is not truncated to
its first 300 frames. ``-frames:v`` on the extract command only bounds the
already chosen set.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from typing import Any

BIN_COUNT = 300
EMPTY_BIN = "empty time bin; no duplication"

# Committed docs/assets/example_videos/Fruits.mp4, read from this tree.
# mvhd timescale 1000 reports 15.067 s. ffprobe on that file reported
# format duration 15.066667 s, 452 video frames, 1280x720. 452 / 30 s is
# that probe duration. This is not a live COLMAP or GPU measurement.
FRUITS_BYTES = 1733022
FRUITS_SHA256 = "a9c099cea15ec73bc203cff0925aa88f166acc2c75e672d9ff89d3bb34169b4d"
FRUITS_MVHD_DURATION_S = 15.067
FRUITS_PROBE_DURATION_S = 15.066667
FRUITS_FRAME_COUNT = 452
FRUITS_FPS = 30


def bin_count(frame_count: int, n: int = BIN_COUNT) -> int:
    """``n`` bins, or one per frame when the clip has fewer frames than that."""
    if isinstance(frame_count, bool) or not isinstance(frame_count, int):
        raise ValueError("frame count is not an integer")
    if frame_count <= 0:
        raise ValueError("no decoded frames")
    return min(n, frame_count)


def bin_edges(duration: float, n: int = BIN_COUNT) -> list[float]:
    """Return ``n + 1`` edges splitting ``[0, duration]`` into equal bins."""
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise ValueError("invalid bin domain")
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("invalid bin domain")
    if not math.isfinite(float(duration)) or float(duration) <= 0:
        raise ValueError("invalid bin domain")
    width = float(duration)
    return [width * i / n for i in range(n + 1)]


def laplacian_variance(gray: Sequence[Sequence[float]]) -> float:
    """Population variance of the interior 4-neighbor Laplacian.

    The border is unused, matching a 3x3 interior stencil with no padding.
    A frame smaller than 3x3, or a non-finite sample, is refused.
    """
    height = len(gray)
    width = len(gray[0]) if height else 0
    if height < 3 or width < 3:
        raise ValueError("frame too small")
    if any(len(row) != width for row in gray):
        raise ValueError("frame rows differ in width")
    total = 0.0
    count = 0
    square = 0.0
    for y in range(1, height - 1):
        row = gray[y]
        up = gray[y - 1]
        down = gray[y + 1]
        for x in range(1, width - 1):
            sample = row[x]
            if isinstance(sample, bool) or not isinstance(sample, (int, float)):
                raise ValueError("invalid sharpness")
            lap = -4.0 * float(sample) + float(up[x]) + float(down[x])
            lap += float(row[x - 1]) + float(row[x + 1])
            if not math.isfinite(lap):
                raise ValueError("invalid sharpness")
            total += lap
            square += lap * lap
            count += 1
    if count == 0:
        raise ValueError("frame too small")
    mean = total / count
    variance = square / count - mean * mean
    if not math.isfinite(variance) or variance < 0:
        raise ValueError("invalid sharpness")
    return variance


def select_sharpest_per_bin(
    frames: Sequence[dict[str, Any]],
    duration: float,
    n: int = BIN_COUNT,
) -> list[dict[str, Any]]:
    """Pick the sharpest frame in each bin. Ties keep the earliest index.

    A clip with no more frames than bins (a 10 s, 24 fps phone video has
    240) keeps every frame, one per bin, rather than failing on a bin that
    timestamp jitter left empty. Use :func:`bin_count` for ``n``.

    ``frames`` must be contiguous indices ``0 .. len-1`` with non-decreasing
    timestamps inside ``[0, duration]``. Every frame belongs to exactly one
    bin. The last bin includes ``duration`` itself.
    """
    edges = bin_edges(duration, n)
    if not frames:
        raise ValueError("no decoded frames")
    indexes = [frame["index"] for frame in frames]
    if indexes != list(range(len(frames))):
        raise ValueError("decoded frame indices must be contiguous from 0")
    times: list[float] = []
    for frame in frames:
        stamp = frame["time"]
        score = frame["score"]
        if (
            isinstance(stamp, bool)
            or not isinstance(stamp, (int, float))
            or not math.isfinite(float(stamp))
        ):
            raise ValueError("frame timestamps outside full duration")
        if float(stamp) < 0 or float(stamp) > float(duration):
            raise ValueError("frame timestamps outside full duration")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or float(score) < 0
        ):
            raise ValueError("invalid sharpness")
        times.append(float(stamp))
    if times != sorted(times):
        raise ValueError("frame timestamps must be in order")
    if len(frames) == n:
        return [
            {
                "bin": index,
                "left": float(frame["time"]),
                "right": float(frame["time"]),
                "last_right_inclusive": True,
                "chosen_index": frame["index"],
                "chosen_time": float(frame["time"]),
                "score": float(frame["score"]),
                "candidates": 1,
            }
            for index, frame in enumerate(frames)
        ]
    chosen: list[dict[str, Any]] = []
    covered: list[int] = []
    for index, (left, right) in enumerate(itertools.pairwise(edges)):
        last = index == n - 1
        candidates = [
            frame
            for frame in frames
            if float(frame["time"]) >= left
            and (
                float(frame["time"]) < right or (last and float(frame["time"]) == right)
            )
        ]
        if not candidates:
            raise ValueError(EMPTY_BIN)
        best = min(
            candidates, key=lambda frame: (-float(frame["score"]), frame["index"])
        )
        chosen.append(
            {
                "bin": index,
                "left": left,
                "right": right,
                "last_right_inclusive": last,
                "chosen_index": best["index"],
                "chosen_time": float(best["time"]),
                "score": float(best["score"]),
                "candidates": len(candidates),
            }
        )
        covered.extend(frame["index"] for frame in candidates)
    if sorted(covered) != list(range(len(frames))):
        raise ValueError("bins do not cover each decoded frame exactly once")
    if len(chosen) != n:
        raise ValueError("selected frame count is not the bin count")
    return chosen


def select_frames(
    frames: Sequence[dict[str, Any]],
    duration: float,
    n: int = BIN_COUNT,
) -> tuple[list[dict[str, Any]], str]:
    """Equal-time bins, or equal-count bins when a timestamp gap empties one.

    Returns the rows and ``"time"`` or ``"index"``. A phone clip can be
    variable-rate: Objectron's laptop video has 303 frames in 10.16 s with
    gaps up to 0.1 s, so 300 equal-time bins left some empty and the clip
    failed before COLMAP. Time bins stay first, so every clip that sampled
    before samples the same frames now. The fallback splits the frames, in
    order, into ``n`` runs whose lengths differ by at most one and keeps the
    sharpest of each; it never duplicates a frame.
    """
    try:
        return select_sharpest_per_bin(frames, duration, n), "time"
    except ValueError as exc:
        if str(exc) != EMPTY_BIN:
            raise
    if n > len(frames):
        raise ValueError("more bins than frames")
    rows: list[dict[str, Any]] = []
    for index in range(n):
        start = index * len(frames) // n
        stop = (index + 1) * len(frames) // n
        group = frames[start:stop]
        best = min(group, key=lambda f: (-float(f["score"]), f["index"]))
        rows.append(
            {
                "bin": index,
                "left": float(group[0]["time"]),
                "right": float(group[-1]["time"]),
                "last_right_inclusive": True,
                "chosen_index": best["index"],
                "chosen_time": float(best["time"]),
                "score": float(best["score"]),
                "candidates": len(group),
            }
        )
    return rows, "index"


def _balanced(terms: Sequence[str]) -> str:
    """Balanced sum. A flat 300-term sum overflows ffmpeg's expression depth."""
    if len(terms) == 1:
        return terms[0]
    middle = len(terms) // 2
    return "(" + _balanced(terms[:middle]) + "+" + _balanced(terms[middle:]) + ")"


def select_expression(indices: Sequence[int]) -> str:
    """ffmpeg ``select`` expression for these source frame indexes."""
    if not indices or isinstance(indices, str):
        raise ValueError("selected indices are empty")
    terms = []
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("selected index is not a frame number")
        terms.append(f"eq(n\\,{index})")
    return "select=" + _balanced(terms)


# Frames are decoded and staged with their long side at most this many
# pixels. A 1080p or smaller video passes through pixel for pixel (the
# scale filter does nothing when the size does not change). A 60 s 4K60
# clip would otherwise decode to about 40 GB of PNG just to be scored, and
# COLMAP reads at most 1280 px (``--SiftExtraction.max_image_size``).
MAX_LONG_SIDE_PX = 1920
LONG_SIDE_FILTER = (
    f"scale=w='if(gte(iw,ih),min(iw,{MAX_LONG_SIDE_PX}),-2)'"
    f":h='if(gte(iw,ih),-2,min(ih,{MAX_LONG_SIDE_PX}))'"
)


def decode_all_command(ffmpeg: str, video: str, pattern: str) -> list[str]:
    """Decode every video frame. No fps filter and no frame cap.

    The long side is capped at :data:`MAX_LONG_SIDE_PX`; nothing else changes.
    """
    return [
        ffmpeg,
        "-y",
        "-i",
        video,
        "-map",
        "0:v:0",
        "-vf",
        LONG_SIDE_FILTER,
        "-vsync",
        "0",
        "-start_number",
        "0",
        pattern,
    ]


def extract_selected_command(
    ffmpeg: str, video: str, indices: Sequence[int], pattern: str
) -> list[str]:
    """Extract the chosen indexes. ``-frames:v`` only bounds that selection."""
    chosen = list(indices)
    return [
        ffmpeg,
        "-y",
        "-i",
        video,
        "-vf",
        f"{select_expression(chosen)},{LONG_SIDE_FILTER}",
        "-vsync",
        "0",
        "-frames:v",
        str(len(chosen)),
        pattern,
    ]


def ffprobe_command(ffprobe: str, video: str) -> list[str]:
    """Probe every video frame timestamp and the container duration."""
    return [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        "frame=best_effort_timestamp_time:format=duration",
        "-of",
        "json",
        video,
    ]


def parse_probe(payload: dict[str, Any]) -> tuple[float, list[float]]:
    """Return ``(duration, timestamps)`` from an ffprobe JSON object.

    A missing duration, a non-finite timestamp, or a timestamp past the
    duration fails. The frame list is the whole stream, not a prefix.
    """
    try:
        duration = float(payload["format"]["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("ffprobe duration missing") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("ffprobe duration missing")
    raw = payload.get("frames")
    if not isinstance(raw, list) or not raw:
        raise ValueError("ffprobe returned no frames")
    stamps: list[float] = []
    for frame in raw:
        if not isinstance(frame, dict) or "best_effort_timestamp_time" not in frame:
            raise ValueError("ffprobe frame timestamp missing")
        try:
            stamp = float(frame["best_effort_timestamp_time"])
        except (TypeError, ValueError) as exc:
            raise ValueError("ffprobe frame timestamp missing") from exc
        if not math.isfinite(stamp) or stamp < 0 or stamp > duration:
            raise ValueError("ffprobe frame timestamp outside the clip")
        stamps.append(stamp)
    return duration, stamps
