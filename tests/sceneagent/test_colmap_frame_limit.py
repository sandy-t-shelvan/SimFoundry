"""CPU tests for the COLMAP frame cap and per-image color calibration.

No COLMAP binary and no GPU. ffmpeg is not invoked; the sample argv is
checked directly, and ``run_poses`` is stubbed at ``_run``.
"""

import json
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from sceneagent.colmap_stage import run_poses, sample_frame_command
from sceneagent.ingest_assets import (
    COLOR_TARGET_NAME,
    JPEG_QUALITY,
    JPEG_SUBSAMPLING,
    OURS_COLOR_HEURISTIC,
    OURS_GAIN_MAX,
    OURS_GAIN_MIN,
    _bounded_channel_gain,
    _mean_packed_rgb,
    _scale_rgb_bytes,
    _write_rgb,
    apply_per_image_color_calibration,
    jpeg_round_trip_mean_error,
    mean_rgb_file,
    per_image_color_gains,
)
from sceneagent.video import run_video

ROOT = Path(__file__).resolve().parents[2]
FRUITS = ROOT / "docs" / "assets" / "example_videos" / "Fruits.mp4"
_PPM_HEADER = b"P6\n2 2\n255\n"


def _ppm(path, rgb, size=2):
    path.write_bytes(_PPM_HEADER + bytes(rgb) * (size * size))


def _ppm_pixels(path):
    data = path.read_bytes()
    assert data.startswith(_PPM_HEADER)
    return data[len(_PPM_HEADER) :]


def _which_binaries(name):
    if name in {"ffmpeg", "colmap"}:
        return f"/usr/bin/{name}"
    return None


def _failed_run(command, timeout_s=600):
    return {
        "command": command,
        "returncode": 1,
        "timed_out": False,
        "stdout_tail": "",
        "stderr_tail": "",
    }


def _banner(*_args, **_kwargs):
    return subprocess.CompletedProcess(
        _args,
        0,
        stdout="COLMAP 3.7 -- Structure-from-Motion\n(Commit Unknown without CUDA)\n",
        stderr="",
    )


def test_default_sample_command_keeps_the_16_frame_cap():
    assert sample_frame_command("ffmpeg", "clip.mp4", "images/%04d.jpg") == [
        "ffmpeg",
        "-y",
        "-i",
        "clip.mp4",
        "-vf",
        "fps=1",
        "-frames:v",
        "16",
        "images/%04d.jpg",
    ]


def test_none_and_zero_keep_every_decoded_frame():
    expected = ["ffmpeg", "-y", "-i", "clip.mp4", "images/%04d.jpg"]
    assert (
        sample_frame_command(
            "ffmpeg", "clip.mp4", "images/%04d.jpg", fps=None, max_frames=None
        )
        == expected
    )
    assert (
        sample_frame_command(
            "ffmpeg", "clip.mp4", "images/%04d.jpg", fps=0, max_frames=0
        )
        == expected
    )
    capped = sample_frame_command(
        "ffmpeg", "clip.mp4", "images/%04d.jpg", fps=None, max_frames=16
    )
    assert "-vf" not in capped
    assert capped[capped.index("-frames:v") + 1] == "16"
    every_at_one_fps = sample_frame_command(
        "ffmpeg", "clip.mp4", "images/%04d.jpg", fps=1, max_frames=0
    )
    assert "fps=1" in every_at_one_fps
    assert "-frames:v" not in every_at_one_fps
    custom = sample_frame_command(
        "ffmpeg", "clip.mp4", "images/%04d.jpg", fps=29.97, max_frames=7
    )
    assert "fps=29.97" in custom
    assert custom[custom.index("-frames:v") + 1] == "7"


def test_bool_and_negative_limits_are_rejected():
    with pytest.raises(ValueError):
        sample_frame_command("ffmpeg", "clip.mp4", "images/%04d.jpg", max_frames=False)
    with pytest.raises(ValueError):
        sample_frame_command("ffmpeg", "clip.mp4", "images/%04d.jpg", fps=False)
    with pytest.raises(ValueError):
        sample_frame_command("ffmpeg", "clip.mp4", "images/%04d.jpg", max_frames=-1)
    with pytest.raises(ValueError):
        sample_frame_command("ffmpeg", "clip.mp4", "images/%04d.jpg", max_frames=2.5)


def test_run_poses_records_a_full_rate_request_without_binaries(tmp_path, monkeypatch):
    monkeypatch.setattr("sceneagent.colmap_stage.shutil.which", lambda _name: None)
    result = run_poses("clip.mp4", tmp_path / "work", fps=None, max_frames=0)
    assert result["ran"] is False
    assert result["ok"] is False
    assert "PATH" in result["reason"]
    assert result["full_rate"] is True
    assert result["fps"] is None
    assert result["max_frames"] == 0
    assert result["color_calibrate"] is True
    assert "color_calibration" not in result
    default = run_poses("clip.mp4", tmp_path / "default")
    assert default["fps"] == 1
    assert default["max_frames"] == 16
    assert default["full_rate"] is False


def test_run_poses_forwards_the_frame_limit_to_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr("sceneagent.colmap_stage.shutil.which", _which_binaries)
    seen = []

    def fake_run(command, timeout_s=600):
        seen.append(list(command))
        return _failed_run(command, timeout_s)

    monkeypatch.setattr("sceneagent.colmap_stage._run", fake_run)
    result = run_poses(tmp_path / "clip.mp4", tmp_path / "work", fps=30, max_frames=None)
    assert result["ran"] is True
    assert result["failed_step"] == "ffmpeg"
    assert result["full_rate"] is False
    assert seen[0] == sample_frame_command(
        "/usr/bin/ffmpeg",
        tmp_path / "clip.mp4",
        tmp_path / "work" / "images" / "%04d.jpg",
        fps=30,
        max_frames=None,
    )
    assert "-frames:v" not in seen[0]
    assert "fps=30" in seen[0]


def test_a_negative_frame_cap_is_rejected_before_the_manifest(tmp_path):
    with pytest.raises(ValueError):
        run_video(tmp_path / "missing.mp4", tmp_path / "out", max_frames=-1)
    assert not (tmp_path / "out").exists()


def test_run_video_forwards_full_rate(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sceneagent.external_tools.shutil.which",
        lambda name: "/usr/bin/colmap" if name == "colmap" else None,
    )
    seen = {}

    def fake_poses(path, work_dir, fps=1, max_frames=16, color_calibrate=True):
        seen["args"] = (fps, max_frames, color_calibrate)
        return {"ran": False, "ok": False, "reason": "stub"}

    monkeypatch.setattr("sceneagent.colmap_stage.run_poses", fake_poses)
    summary = run_video(
        FRUITS, tmp_path / "out", fps=0, max_frames=None, color_calibrate=True
    )
    assert seen["args"] == (0, None, True)
    assert summary["colmap_decision"] == "deferred"


def test_cli_keeps_the_default_cap_and_accepts_full_rate(tmp_path, monkeypatch):
    seen = []

    def fake_run_video(path, out, fps=1, max_frames=16, color_calibrate=True):
        seen.append((fps, max_frames, color_calibrate))
        return {
            "video": {"name": "Fruits.mp4", "bytes": 1, "duration_s": 1.0},
            "colmap_available": False,
            "colmap_decision": "deferred",
            "reconstruction": "not_run",
            "out_dir": str(out),
        }

    monkeypatch.setattr("sceneagent.video.run_video", fake_run_video)
    from sceneagent.__main__ import main

    assert main([str(FRUITS), "--out", str(tmp_path / "default")]) == 0
    assert seen[-1] == (1, 16, True)
    assert (
        main(
            [
                str(FRUITS),
                "--out",
                str(tmp_path / "full"),
                "--fps",
                "0",
                "--max-frames",
                "0",
                "--no-color-calibration",
            ]
        )
        == 0
    )
    assert seen[-1] == (0, 0, False)
    with pytest.raises(SystemExit) as caught:
        main(
            [
                str(tmp_path / "scene.json"),
                "--out",
                str(tmp_path / "json"),
                "--max-frames",
                "0",
            ]
        )
    assert caught.value.code == 2


def test_gains_are_per_image_and_a_zero_channel_stays_at_one():
    matched = per_image_color_gains(
        [("a", (40, 50, 60)), ("b", (40, 50, 60))]
    )
    assert matched["global_correction"] is False
    assert matched["paper_method"] is False
    assert [row["path"] for row in matched["gains"]] == ["a", "b"]
    assert matched["gains"][0]["gain_rgb"] == pytest.approx([1, 1, 1])
    assert matched["gains"][1]["gain_rgb"] == pytest.approx([1, 1, 1])
    dark = per_image_color_gains(
        [("dark", (0, 10, 10)), ("lit", (20, 10, 10))]
    )
    by_path = {row["path"]: row["gain_rgb"] for row in dark["gains"]}
    assert by_path["dark"] == pytest.approx([1.0, 1.0, 1.0])
    assert by_path["lit"][0] == pytest.approx(0.5)
    assert by_path["dark"] != by_path["lit"]


def test_color_calibration_scales_each_image_on_its_own(tmp_path):
    folder = tmp_path / "frames"
    folder.mkdir()
    # Every channel's raw gain is at most 2, inside the 4x fixture limit.
    _ppm(folder / "red.ppm", (80, 40, 20))
    _ppm(folder / "blue.ppm", (20, 40, 80))
    _ppm(folder / "gray.ppm", (40, 40, 40))
    before = {path.name: path.read_bytes() for path in folder.iterdir()}
    out = tmp_path / "calibrated"
    result = apply_per_image_color_calibration(folder, out)
    assert {path.name: path.read_bytes() for path in folder.iterdir()} == before
    assert result["applied"] is True
    assert result["paper_method"] is False
    assert result["global_correction"] is False
    assert result["method"] == "per_image_diagonal_to_median"
    assert "ours" in result["label"]
    assert result["colmap_ran"] is False
    assert result["median_rgb"] == pytest.approx([40, 40, 40])
    assert result["median_rgb"] == pytest.approx(result["input_report"]["median_rgb"])
    assert result["input_report"]["heuristic"] == OURS_COLOR_HEURISTIC
    flagged = {Path(path).name for path in result["input_report"]["flagged_paths"]}
    assert flagged == {"red.ppm", "blue.ppm"}
    by_name = {Path(frame["source"]).name: frame for frame in result["frames"]}
    assert by_name["red.ppm"]["gain_rgb"] == pytest.approx([0.5, 1, 2])
    assert by_name["blue.ppm"]["gain_rgb"] == pytest.approx([2, 1, 0.5])
    assert by_name["gray.ppm"]["gain_rgb"] == pytest.approx([1, 1, 1])
    assert max(by_name["red.ppm"]["gain_rgb"]) <= 4
    assert max(by_name["blue.ppm"]["gain_rgb"]) <= 4
    assert by_name["red.ppm"]["gain_rgb"] != by_name["blue.ppm"]["gain_rgb"]
    target = bytes((40, 40, 40)) * 4
    assert _ppm_pixels(out / "red.ppm") == target
    assert _ppm_pixels(out / "blue.ppm") == target
    assert (out / "gray.ppm").read_bytes() == before["gray.ppm"]
    for name in ("red.ppm", "blue.ppm", "gray.ppm"):
        assert by_name[name]["calibrated_mean_rgb"] == pytest.approx([40, 40, 40])


def test_run_poses_calibrates_each_frame_before_feature_extraction(tmp_path, monkeypatch):
    monkeypatch.setattr("sceneagent.colmap_stage.shutil.which", _which_binaries)
    monkeypatch.setattr("sceneagent.colmap_stage.subprocess.run", _banner)
    work = tmp_path / "colmap"
    originals = {}
    stages = []

    def fake_run(command, timeout_s=600):
        if Path(str(command[0])).name == "ffmpeg":
            stages.append("ffmpeg")
            folder = Path(command[-1]).parent
            for frame_name, color in (
                ("0001.jpg", (200, 10, 10)),
                ("0002.jpg", (10, 10, 200)),
            ):
                path = folder / frame_name
                Image.new("RGB", (2, 2), color).save(path, quality=95, subsampling=0)
                originals[frame_name] = path.read_bytes()
        else:
            stages.append(command[1])
            if command[1] == "feature_extractor":
                assert (work / "color_calibration.json").is_file()
                for frame_name, raw in originals.items():
                    assert (work / "images" / frame_name).read_bytes() != raw
        return {
            "command": command,
            "returncode": 0,
            "timed_out": False,
            "stdout_tail": "",
            "stderr_tail": "",
        }

    monkeypatch.setattr("sceneagent.colmap_stage._run", fake_run)
    result = run_poses(tmp_path / "clip.mp4", work, fps=1, max_frames=2)
    assert result["ok"] is False
    assert result["failed_step"] == "mapper"
    assert stages[0] == "ffmpeg"
    assert stages.index("feature_extractor") < stages.index("sequential_matcher")
    report = result["color_calibration"]
    assert report["global_correction"] is False
    assert report["paper_method"] is False
    by_name = {Path(frame["source"]).name: frame["gain_rgb"] for frame in report["frames"]}
    assert by_name["0001.jpg"][0] < 1
    assert by_name["0002.jpg"][0] > 1
    assert by_name["0001.jpg"] != by_name["0002.jpg"]
    on_disk = json.loads((work / "color_calibration.json").read_text(encoding="utf-8"))
    assert on_disk["global_correction"] is False
    red_mean = mean_rgb_file(work / "images" / "0001.jpg")
    blue_mean = mean_rgb_file(work / "images" / "0002.jpg")
    assert red_mean[0] < 150
    assert blue_mean[0] > 50
    assert abs(red_mean[0] - blue_mean[0]) < 40


def test_run_poses_can_leave_sampled_frames_uncalibrated(tmp_path, monkeypatch):
    monkeypatch.setattr("sceneagent.colmap_stage.shutil.which", _which_binaries)
    monkeypatch.setattr("sceneagent.colmap_stage.subprocess.run", _banner)
    work = tmp_path / "colmap"
    (work / "color_calibration.json").parent.mkdir(parents=True)
    (work / "color_calibration.json").write_text("{}\n", encoding="utf-8")
    originals = {}

    def fake_run(command, timeout_s=600):
        if Path(str(command[0])).name == "ffmpeg":
            folder = Path(command[-1]).parent
            for frame_name, color in (
                ("0001.jpg", (200, 10, 10)),
                ("0002.jpg", (10, 10, 200)),
            ):
                path = folder / frame_name
                Image.new("RGB", (2, 2), color).save(path, quality=95, subsampling=0)
                originals[frame_name] = path.read_bytes()
        return {
            "command": command,
            "returncode": 0,
            "timed_out": False,
            "stdout_tail": "",
            "stderr_tail": "",
        }

    monkeypatch.setattr("sceneagent.colmap_stage._run", fake_run)
    skipped = run_poses(
        tmp_path / "clip.mp4", work, fps=1, max_frames=2, color_calibrate=False
    )
    assert skipped["color_calibrate"] is False
    assert "color_calibration" not in skipped
    assert not (work / "color_calibration.json").exists()
    for frame_name, raw in originals.items():
        assert (work / "images" / frame_name).read_bytes() == raw


def test_nonfinite_means_are_rejected_and_extreme_gains_are_clamped():
    # A zero mean stays at gain 1; test_gains_are_per_image_and_a_zero_channel_stays_at_one
    # already locks that. Non-finite means are rejected, and finite gains outside
    # [OURS_GAIN_MIN, OURS_GAIN_MAX] clamp to that documented interval.
    with pytest.raises(ValueError, match="finite"):
        per_image_color_gains([("a", (float("nan"), 1.0, 1.0)), ("b", (1.0, 1.0, 1.0))])
    with pytest.raises(ValueError, match="finite"):
        per_image_color_gains([("a", (float("inf"), 1.0, 1.0)), ("b", (1.0, 1.0, 1.0))])
    with pytest.raises(ValueError, match="finite"):
        per_image_color_gains([("a", (1.0, float("-inf"), 1.0)), ("b", (1.0, 1.0, 1.0))])
    high = per_image_color_gains(
        [("dim", (1.0, 10.0, 10.0)), ("mid", (80.0, 10.0, 10.0)), ("hot", (80.0, 10.0, 10.0))]
    )
    assert high["gains_by_image"]["dim"][0] == OURS_GAIN_MAX
    low = per_image_color_gains(
        [("a", (1.0, 10.0, 10.0)), ("b", (1.0, 10.0, 10.0)), ("c", (200.0, 10.0, 10.0))]
    )
    assert low["gains_by_image"]["c"][0] == OURS_GAIN_MIN
    # Median red is 1, so c's raw red gain is 1/200 = 0.005, below 0.05.
    assert low["raw_gain_rgb"]["c"][0] == pytest.approx(1.0 / 200.0)
    assert low["clamped"]["c"] is True
    assert low["clamped_channels"]["c"][0] is True
    assert "c" in low["clamped_images"]
    assert low["clamped"]["a"] is False
    assert low["raw_gain_rgb"]["a"] == low["gains_by_image"]["a"]
    assert "a" not in low["clamped_images"]
    assert OURS_GAIN_MIN == 0.05
    assert OURS_GAIN_MAX == 8.0


def test_synthetic_color_cast_is_recovered_with_distinct_gains(tmp_path):
    folder = tmp_path / "cast"
    folder.mkdir()
    _ppm(folder / "neutral.ppm", (80, 80, 80))
    _ppm(folder / "red_cast.ppm", (160, 40, 40))
    _ppm(folder / "blue_cast.ppm", (40, 40, 160))
    result = apply_per_image_color_calibration(folder, tmp_path / "out")
    assert result["target_name"] == COLOR_TARGET_NAME == "per_channel_median_of_set"
    assert result["gap"] == "partial"
    assert result["paper_method"] is False
    assert result["target_rgb"] == pytest.approx([80, 40, 80])
    gains = result["gains_by_image"]
    assert gains["red_cast.ppm"] == pytest.approx([0.5, 1.0, 2.0])
    assert gains["blue_cast.ppm"] == pytest.approx([2.0, 1.0, 0.5])
    assert gains["red_cast.ppm"] != gains["blue_cast.ppm"]
    recovered = bytes((80, 40, 80)) * 4
    assert _ppm_pixels(tmp_path / "out" / "red_cast.ppm") == recovered
    assert _ppm_pixels(tmp_path / "out" / "blue_cast.ppm") == recovered
    assert _ppm_pixels(tmp_path / "out" / "neutral.ppm") == recovered


def test_in_memory_corrected_mean_is_within_half_a_level(tmp_path):
    folder = tmp_path / "cast"
    folder.mkdir()
    _ppm(folder / "neutral.ppm", (80, 80, 80))
    _ppm(folder / "red_cast.ppm", (160, 40, 40))
    _ppm(folder / "blue_cast.ppm", (40, 40, 160))
    result = apply_per_image_color_calibration(folder, tmp_path / "out")
    for frame in result["frames"]:
        for got, want in zip(frame["in_memory_mean_rgb"], result["target_rgb"], strict=True):
            assert abs(got - want) <= 0.5
    # Rounding of the corrected array, measured before any encode.
    pixels = bytes((7, 11, 13)) * 16
    scaled = _scale_rgb_bytes(pixels, (1.2, 0.8, 1.1))
    got = _mean_packed_rgb(scaled, "mem")
    raw = (7 * 1.2, 11 * 0.8, 13 * 1.1)
    for actual, unrounded in zip(got, raw, strict=True):
        assert abs(actual - unrounded) <= 0.5


def test_color_calibration_json_records_target_and_gains(tmp_path, monkeypatch):
    monkeypatch.setattr("sceneagent.colmap_stage.shutil.which", _which_binaries)
    monkeypatch.setattr("sceneagent.colmap_stage.subprocess.run", _banner)

    def fake_run(command, timeout_s=600):
        if Path(str(command[0])).name == "ffmpeg":
            folder = Path(command[-1]).parent
            from PIL import Image

            Image.new("RGB", (2, 2), (160, 40, 40)).save(
                folder / "0001.jpg", quality=JPEG_QUALITY, subsampling=JPEG_SUBSAMPLING
            )
            Image.new("RGB", (2, 2), (40, 40, 160)).save(
                folder / "0002.jpg", quality=JPEG_QUALITY, subsampling=JPEG_SUBSAMPLING
            )
        return {
            "command": command,
            "returncode": 0,
            "timed_out": False,
            "stdout_tail": "",
            "stderr_tail": "",
        }

    monkeypatch.setattr("sceneagent.colmap_stage._run", fake_run)
    work = tmp_path / "colmap"
    run_poses(tmp_path / "clip.mp4", work, max_frames=2)
    manifest = json.loads((work / "color_calibration.json").read_text(encoding="utf-8"))
    assert manifest["target_name"] == "per_channel_median_of_set"
    assert manifest["gap"] == "partial"
    assert manifest["paper_method"] is False
    assert len(manifest["target_rgb"]) == 3
    gains = manifest["gains_by_image"]
    assert set(gains) == {"0001.jpg", "0002.jpg"}
    assert gains["0001.jpg"] != gains["0002.jpg"]
    for triple in gains.values():
        assert len(triple) == 3
        for channel in triple:
            assert OURS_GAIN_MIN <= channel <= OURS_GAIN_MAX


# Measured on the 32x32 fixture below with Pillow JPEG quality 95, subsampling 0.
# This is the error of that fixture, not a widened allowance. It is under 1.5
# levels, so sampled frames stay JPEG rather than moving to PNG.
JPEG_FIXTURE_ROUND_TRIP_MEAN_ERROR = 0.0185546875


def _jpeg_fixture_pixels():
    width, height = 32, 32
    raw = bytearray()
    for y in range(height):
        for x in range(width):
            raw.append((40 + 3 * x + y) % 256)
            raw.append((20 + x + 4 * y) % 256)
            raw.append((90 + 2 * x + 2 * y) % 256)
    return width, height, bytes(raw)


def _ascii_p3(path, rgb, size=2):
    rows = [
        " ".join(" ".join(str(channel) for channel in rgb) for _ in range(size))
        for _ in range(size)
    ]
    path.write_text(f"P3\n{size} {size}\n255\n" + "\n".join(rows) + "\n", encoding="ascii")


def test_ascii_p3_ppm_calibrates_through_the_temp_file(tmp_path):
    folder = tmp_path / "p3"
    folder.mkdir()
    _ascii_p3(folder / "red_cast.ppm", (160, 40, 40))
    _ascii_p3(folder / "blue_cast.ppm", (40, 40, 160))
    out = tmp_path / "out"
    result = apply_per_image_color_calibration(folder, out)
    assert result["applied"] is True
    assert result["gains_by_image"]["red_cast.ppm"] != result["gains_by_image"]["blue_cast.ppm"]
    assert list(out.glob("*.calibrating")) == []
    for name in ("red_cast.ppm", "blue_cast.ppm"):
        written = out / name
        assert written.read_bytes().startswith(b"P6")
        assert mean_rgb_file(written) == pytest.approx((100, 40, 100))


@pytest.mark.parametrize(
    "suffix",
    [".jpg", ".jpeg", ".png", ".ppm", ".pgm", ".pnm", ".bmp", ".tif", ".tiff", ".webp"],
)
def test_write_rgb_uses_the_final_extension_not_the_temp_suffix(tmp_path, suffix):
    destination = tmp_path / f"frame{suffix}"
    pixels = bytes((10, 20, 30)) * 4
    _write_rgb(destination, "pillow", 2, 2, b"", pixels)
    assert destination.is_file()
    assert list(tmp_path.glob("*.calibrating")) == []
    image = Image.open(destination).convert("RGB")
    assert image.size == (2, 2)
    if suffix not in {".jpg", ".jpeg", ".webp"}:
        assert image.tobytes() == pixels


def _p6_bytes(rgb, size=2):
    return f"P6\n{size} {size}\n255\n".encode("ascii") + bytes(rgb) * (size * size)


def test_gain_above_eight_is_clamped_in_color_calibration_json(tmp_path, monkeypatch):
    monkeypatch.setattr("sceneagent.colmap_stage.shutil.which", _which_binaries)
    monkeypatch.setattr("sceneagent.colmap_stage.subprocess.run", _banner)

    def fake_run(command, timeout_s=600):
        if Path(str(command[0])).name == "ffmpeg":
            folder = Path(command[-1]).parent
            # Median red is 40. The dim frame's raw gain is 40/4 = 10, above 8.
            (folder / "0001.jpg").write_bytes(_p6_bytes((4, 40, 40)))
            (folder / "0002.jpg").write_bytes(_p6_bytes((40, 40, 40)))
            (folder / "0003.jpg").write_bytes(_p6_bytes((40, 40, 40)))
        return {
            "command": command,
            "returncode": 0,
            "timed_out": False,
            "stdout_tail": "",
            "stderr_tail": "",
        }

    monkeypatch.setattr("sceneagent.colmap_stage._run", fake_run)
    work = tmp_path / "colmap"
    run_poses(tmp_path / "clip.mp4", work, max_frames=3)
    manifest = json.loads((work / "color_calibration.json").read_text(encoding="utf-8"))
    assert manifest["gains_by_image"]["0001.jpg"][0] == 8.0
    assert manifest["raw_gain_rgb"]["0001.jpg"][0] == 10.0
    assert manifest["clamped"]["0001.jpg"] is True
    assert manifest["clamped_channels"]["0001.jpg"] == [True, False, False]
    assert manifest["clamped_images"] == ["0001.jpg"]
    assert manifest["gains_by_image"]["0002.jpg"][0] == pytest.approx(1.0)
    assert manifest["gains_by_image"]["0003.jpg"][0] == pytest.approx(1.0)
    for name in ("0002.jpg", "0003.jpg"):
        assert manifest["clamped"][name] is False
        assert manifest["raw_gain_rgb"][name] == manifest["gains_by_image"][name]
        assert name not in manifest["clamped_images"]


def test_gain_of_eight_is_kept_and_anything_above_is_clamped():
    exact = per_image_color_gains(
        [
            ("dim", (5.0, 10.0, 10.0)),
            ("mid", (40.0, 10.0, 10.0)),
            ("hot", (40.0, 10.0, 10.0)),
        ]
    )
    # 40/5 is exactly 8, so the bound does not move it.
    assert exact["gains_by_image"]["dim"][0] == 8.0
    assert exact["raw_gain_rgb"]["dim"][0] == 8.0
    assert exact["clamped"]["dim"] is False
    assert exact["clamped_channels"]["dim"] == [False, False, False]
    assert "dim" not in exact["clamped_images"]
    assert exact["raw_gain_rgb"]["dim"] == exact["gains_by_image"]["dim"]
    assert exact["gains_by_image"]["dim"][0] == pytest.approx(40.0 / 5.0)
    above = per_image_color_gains(
        [
            ("dim", (40.0 / 8.001, 10.0, 10.0)),
            ("mid", (40.0, 10.0, 10.0)),
            ("hot", (40.0, 10.0, 10.0)),
        ]
    )
    raw = 40.0 / (40.0 / 8.001)
    assert raw > 8.0
    assert above["gains_by_image"]["dim"][0] == 8.0
    assert above["raw_gain_rgb"]["dim"][0] == pytest.approx(raw)
    assert above["clamped"]["dim"] is True
    assert above["clamped_channels"]["dim"][0] is True
    assert "dim" in above["clamped_images"]
    assert above["clamped"]["mid"] is False
    assert "mid" not in above["clamped_images"]
    assert above["raw_gain_rgb"]["mid"] == above["gains_by_image"]["mid"]
    assert above["gains_by_image"]["dim"][0] != pytest.approx(raw)


def test_bounded_channel_gain_rejects_nonfinite_and_keeps_unit_gain_at_zero():
    # Direct call. The set-level finite check in per_image_color_gains does not
    # run here, so these assertions see the guards inside _bounded_channel_gain.
    for mean, center in (
        (float("nan"), 40.0),
        (float("inf"), 40.0),
        (float("-inf"), 40.0),
        (40.0, float("nan")),
        (40.0, float("inf")),
        (40.0, float("-inf")),
        (float("nan"), float("nan")),
        (float("inf"), float("-inf")),
    ):
        with pytest.raises(ValueError, match="finite"):
            _bounded_channel_gain(mean, center)
    # A non-positive mean is raw 1 and applied 1. That is not a clamp.
    for mean, center in ((0.0, 40.0), (0.0, 0.0), (-1.0, 40.0)):
        raw_gain, applied = _bounded_channel_gain(mean, center)
        assert raw_gain == 1.0
        assert applied == 1.0


def test_jpeg_round_trip_mean_error_is_the_recorded_fixture_bound():
    assert JPEG_QUALITY == 95
    assert JPEG_SUBSAMPLING == 0
    width, height, pixels = _jpeg_fixture_pixels()
    error = jpeg_round_trip_mean_error(width, height, pixels)
    assert error == JPEG_FIXTURE_ROUND_TRIP_MEAN_ERROR
    assert JPEG_FIXTURE_ROUND_TRIP_MEAN_ERROR < 1.5
    command = sample_frame_command("ffmpeg", "clip.mp4", "images/%04d.jpg")
    assert command[-1].endswith(".jpg")
