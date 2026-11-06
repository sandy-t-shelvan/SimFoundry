# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the COLMAP reconstruction acceptance gate.

No COLMAP binary and no GPU. Analyzer text is fed in through the runner
``run_poses`` and ``accept_reconstruction`` take. Expected verdicts for the
parity cases are literals from collector v9's ``quality_bar`` and
``pick_largest``. This module does not import the collector.
"""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from sceneagent.colmap_stage import run_poses
from sceneagent.reconstruction_gate import (
    EVIDENCE_JSON_NAME,
    MAX_REPROJ_PX,
    MIN_OBS_PER_IMAGE,
    MIN_REG_FLOOR,
    REG_FRACTION,
    accept_reconstruction,
    run_model_analyzer,
)
from sceneagent.video import run_video

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).parent / "fixtures" / "colmap_3_11_1_model_analyzer_0.txt"
FRUITS = ROOT / "docs" / "assets" / "example_videos" / "Fruits.mp4"
_REGISTERED_RULE = ">= max(10, ceil(0.8*inputs))"


def _log(code, message):
    return f"I20261106 07:39:22.000000 1 model.cc:{code}] {message}"


def analyzer_text(
    *,
    registered=15,
    points=100,
    observations="40.000000",
    reprojection="1.000000",
    repeat_registered=False,
):
    """COLMAP 3.11.1-shaped model_analyzer text. None omits that label."""
    lines = [_log(446, "Cameras: 1"), _log(447, "Images: 99")]
    if registered is not None:
        lines.append(_log(448, f"Registered images: {registered}"))
    if repeat_registered:
        lines.append(_log(448, "Registered images: 7"))
    if points is not None:
        lines.append(_log(450, f"Points: {points}"))
    lines.append(_log(451, "Observations: 96"))
    lines.append(_log(453, "Mean track length: 2.000000"))
    if observations is not None:
        lines.append(_log(455, f"Mean observations per image: {observations}"))
    if reprojection is not None:
        lines.append(_log(458, f"Mean reprojection error: {reprojection}px"))
    return "\n".join(lines) + "\n"


def _runner(outputs):
    def run(model):
        payload = outputs[model.name]
        if isinstance(payload, BaseException):
            raise payload
        returncode, stdout = payload[0], payload[1]
        stderr = payload[2] if len(payload) > 2 else ""
        return {"returncode": returncode, "stdout": stdout, "stderr": stderr}

    return run


def _evaluate(tmp_path, inputs, models):
    sparse = tmp_path / "sparse"
    sparse.mkdir(parents=True)
    outputs = {}
    for name, payload in models:
        (sparse / name).mkdir()
        outputs[name] = payload
    evidence = tmp_path / "evidence"
    result = accept_reconstruction(sparse, inputs, evidence, runner=_runner(outputs))
    result["evidence"] = evidence
    return result


def _bar(result):
    return result["document"]["quality_bar"]


def _passing(
    name="0",
    registered=15,
    points=100,
    observations="40.000000",
    reprojection="1.000000",
):
    return (
        name,
        (
            0,
            analyzer_text(
                registered=registered,
                points=points,
                observations=observations,
                reprojection=reprojection,
            ),
        ),
    )


def test_thresholds_are_named_once_and_cited():
    assert MIN_REG_FLOOR == 10
    assert REG_FRACTION == 0.8
    assert MIN_OBS_PER_IMAGE == 30.0
    assert MAX_REPROJ_PX == 4.0
    source = (ROOT / "sceneagent" / "reconstruction_gate.py").read_text(
        encoding="utf-8"
    )
    assert "10236" in source
    assert "10237" in source
    assert "10239" in source
    assert source.count("MIN_REG_FLOOR = ") == 1
    assert source.count("REG_FRACTION = ") == 1
    assert source.count("MIN_OBS_PER_IMAGE = ") == 1
    assert source.count("MAX_REPROJ_PX = ") == 1
    stage = (ROOT / "sceneagent" / "colmap_stage.py").read_text(encoding="utf-8")
    assert "MIN_REG_FLOOR" not in stage
    assert "math.ceil" not in stage


def test_pass_when_every_rule_holds(tmp_path):
    result = _evaluate(tmp_path, 15, [_passing(points=1)])
    bar = _bar(result)
    assert result["pass"] is True
    assert result["document"]["verdict"] == "pass"
    assert result["document"]["chosen_model"] == "0"
    assert result["document"]["fail_reasons"] == []
    assert bar["pass"] is True
    assert bar["registered"]["pass"] is True
    assert bar["registered"]["value"] == 15
    assert bar["registered"]["threshold"] == 12
    assert bar["registered"]["rule"] == _REGISTERED_RULE
    assert bar["mean_obs_per_image"]["pass"] is True
    assert bar["mean_obs_per_image"]["value"] == 40.0
    assert bar["mean_obs_per_image"]["rule"] == ">= 30"
    assert bar["mean_reproj_err_px"]["pass"] is True
    assert bar["mean_reproj_err_px"]["value"] == 1.0
    assert bar["mean_reproj_err_px"]["rule"] == "<= 4.0"
    # One point would lose a 100-times-registered rule. There is no such rule.
    assert bar["points_info_only"] == 1


def test_two_of_fifteen_registered_is_rejected(tmp_path):
    text = FIXTURE.read_text(encoding="utf-8")
    result = _evaluate(tmp_path, 15, [("0", (0, text))])
    bar = _bar(result)
    assert result["pass"] is False
    assert result["document"]["verdict"] == "fail"
    assert result["document"]["chosen_model"] == "0"
    assert bar["registered"] == {
        "value": 2,
        "threshold": 12,
        "rule": _REGISTERED_RULE,
        "pass": False,
    }
    assert bar["mean_obs_per_image"]["value"] == 48.0
    assert bar["mean_obs_per_image"]["pass"] is True
    assert bar["mean_reproj_err_px"]["value"] == 0.609572
    assert bar["mean_reproj_err_px"]["pass"] is True
    assert bar["points_info_only"] == 48
    assert bar["fail_reasons"] == ["registered fails or missing"]
    assert "registered fails or missing" in result["reason"]


def test_registered_rule_fails_on_its_own(tmp_path):
    result = _evaluate(
        tmp_path,
        15,
        [_passing(registered=11, observations="30.000000", reprojection="4.0")],
    )
    bar = _bar(result)
    assert bar["registered"]["pass"] is False
    assert bar["registered"]["threshold"] == 12
    assert bar["mean_obs_per_image"]["pass"] is True
    assert bar["mean_reproj_err_px"]["pass"] is True
    assert bar["fail_reasons"] == ["registered fails or missing"]
    assert result["pass"] is False


def test_observations_rule_fails_on_its_own(tmp_path):
    result = _evaluate(
        tmp_path,
        15,
        [_passing(registered=12, observations="29.99", reprojection="4.0")],
    )
    bar = _bar(result)
    assert bar["mean_obs_per_image"]["value"] == 29.99
    assert bar["mean_obs_per_image"]["pass"] is False
    assert bar["registered"]["pass"] is True
    assert bar["mean_reproj_err_px"]["pass"] is True
    assert bar["fail_reasons"] == ["mean_obs_per_image fails or missing"]
    assert result["pass"] is False


def test_reprojection_rule_fails_on_its_own(tmp_path):
    result = _evaluate(
        tmp_path,
        15,
        [_passing(registered=12, observations="30.000000", reprojection="4.01")],
    )
    bar = _bar(result)
    assert bar["mean_reproj_err_px"]["value"] == 4.01
    assert bar["mean_reproj_err_px"]["pass"] is False
    assert bar["registered"]["pass"] is True
    assert bar["mean_obs_per_image"]["pass"] is True
    assert bar["fail_reasons"] == ["mean_reproj_err_px fails or missing"]
    assert result["pass"] is False


@pytest.mark.parametrize(
    ("inputs", "registered", "threshold", "passed"),
    [
        (15, 12, 12, True),
        (15, 11, 12, False),
        (13, 11, 11, True),
        (13, 10, 11, False),
        (14, 12, 12, True),
        (14, 11, 12, False),
        (5, 10, 10, True),
        (5, 9, 10, False),
    ],
)
def test_registered_boundary(tmp_path, inputs, registered, threshold, passed):
    result = _evaluate(
        tmp_path,
        inputs,
        [_passing(registered=registered, observations="30.000000", reprojection="4.0")],
    )
    rule = _bar(result)["registered"]
    assert rule["threshold"] == threshold
    assert rule["value"] == registered
    assert rule["pass"] is passed
    assert result["pass"] is passed


@pytest.mark.parametrize(
    ("observations", "passed"),
    [("30.000000", True), ("30", True), ("29.99", False)],
)
def test_observations_boundary(tmp_path, observations, passed):
    result = _evaluate(
        tmp_path,
        15,
        [_passing(registered=15, observations=observations, reprojection="1.0")],
    )
    rule = _bar(result)["mean_obs_per_image"]
    assert rule["threshold"] == 30.0
    assert rule["pass"] is passed
    assert result["pass"] is passed


@pytest.mark.parametrize(
    ("reprojection", "passed"),
    [("4.0", True), ("4.000000", True), ("4.01", False)],
)
def test_reprojection_boundary(tmp_path, reprojection, passed):
    result = _evaluate(
        tmp_path,
        15,
        [_passing(registered=15, observations="40.000000", reprojection=reprojection)],
    )
    rule = _bar(result)["mean_reproj_err_px"]
    assert rule["threshold"] == 4.0
    assert rule["pass"] is passed
    assert result["pass"] is passed


def test_largest_model_is_not_sparse_zero(tmp_path):
    result = _evaluate(
        tmp_path,
        15,
        [
            _passing(name="0", registered=12, points=99999),
            _passing(name="1", registered=14, points=10),
            _passing(name="2", registered=11, points=100000),
        ],
    )
    assert result["pass"] is True
    assert result["document"]["chosen_model"] == "1"
    assert result["document"]["largest_model"] == "1"
    assert _bar(result)["registered"]["value"] == 14
    assert _bar(result)["points_info_only"] == 10


def test_points_break_a_registered_tie(tmp_path):
    result = _evaluate(
        tmp_path,
        15,
        [
            _passing(name="0", registered=12, points=10, observations="90.000000"),
            _passing(name="1", registered=12, points=80, observations="31.000000"),
        ],
    )
    assert result["pass"] is True
    assert result["document"]["chosen_model"] == "1"
    assert _bar(result)["points_info_only"] == 80
    assert _bar(result)["mean_obs_per_image"]["value"] == 31.0


def test_points_missing_on_the_unique_model_still_passes(tmp_path):
    result = _evaluate(
        tmp_path,
        15,
        [
            (
                "0",
                (
                    0,
                    analyzer_text(registered=15, points=None, observations="40.000000"),
                ),
            )
        ],
    )
    assert result["pass"] is True
    assert _bar(result)["points_info_only"] is None
    assert result["document"]["selection_error"] is None


def test_no_models_fails_closed(tmp_path):
    result = _evaluate(tmp_path, 15, [])
    assert result["pass"] is False
    assert result["document"]["chosen_model"] is None
    assert result["document"]["largest_model"] is None
    assert result["document"]["selection_error"] == "no sparse model"
    assert result["document"]["fail_reasons"][0] == "no sparse model"
    assert result["document"]["verdict"] == "fail"


def test_unparsable_registered_fails_closed(tmp_path):
    text = analyzer_text(registered=15, repeat_registered=True)
    result = _evaluate(tmp_path, 15, [("0", (0, text))])
    assert result["pass"] is False
    assert result["document"]["chosen_model"] is None
    assert "registered missing/unparsable" in result["document"]["selection_error"]
    assert result["document"]["sparse_models"][0]["registered"] is None


def test_missing_registered_on_any_model_fails_closed(tmp_path):
    good = _passing(name="0", registered=15, points=400)
    blank = (
        "1",
        (0, analyzer_text(registered=None, points=400, observations="40.000000")),
    )
    result = _evaluate(tmp_path, 15, [good, blank])
    assert result["pass"] is False
    assert result["document"]["chosen_model"] is None
    assert result["document"]["largest_model"] is None
    assert "registered missing/unparsable" in result["document"]["selection_error"]
    assert "'1'" in result["document"]["selection_error"]


def test_registered_tie_with_missing_points_fails_closed(tmp_path):
    text = analyzer_text(registered=15, points=None, observations="40.000000")
    result = _evaluate(tmp_path, 15, [("0", (0, text)), ("1", (0, text))])
    assert result["pass"] is False
    assert result["document"]["chosen_model"] is None
    error = result["document"]["selection_error"]
    assert "points missing/unparsable" in error
    assert "registered tie at 15" in error


def test_points_tie_has_no_rule_and_fails_closed(tmp_path):
    tied = _evaluate(
        tmp_path,
        15,
        [
            _passing(name="0", registered=15, points=40),
            _passing(name="1", registered=15, points=40),
        ],
    )
    assert tied["pass"] is False
    assert tied["document"]["chosen_model"] is None
    assert "no rule" in tied["document"]["selection_error"]
    assert "points tied at 40" in tied["document"]["selection_error"]
    shared = _evaluate(
        tmp_path / "shared",
        15,
        [
            _passing(name="0", registered=15, points=10),
            _passing(name="1", registered=15, points=50),
            _passing(name="2", registered=15, points=50),
        ],
    )
    assert shared["pass"] is False
    assert shared["document"]["chosen_model"] is None
    assert "points tied at 50" in shared["document"]["selection_error"]
    assert "no rule" in shared["document"]["selection_error"]


def test_analyzer_nonzero_exit_fails_closed(tmp_path):
    text = analyzer_text(registered=15, points=400, observations="40.000000")
    result = _evaluate(tmp_path, 15, [("0", (1, text))])
    model = result["document"]["sparse_models"][0]
    assert result["pass"] is False
    assert result["document"]["chosen_model"] is None
    assert model["analyzer_rc"] == 1
    assert model["registered"] is None
    assert model["points"] is None
    assert "model_analyzer failed" in result["document"]["selection_error"]
    assert "'0'" in result["document"]["selection_error"]


def test_garbage_analyzer_output_fails_closed(tmp_path):
    result = _evaluate(tmp_path, 15, [("0", (0, "not a reconstruction\n"))])
    assert result["pass"] is False
    assert result["document"]["chosen_model"] is None
    assert result["document"]["sparse_models"][0]["registered"] is None
    assert "registered missing/unparsable" in result["document"]["selection_error"]


def test_analyzer_crash_fails_closed(tmp_path):
    result = _evaluate(tmp_path, 15, [("0", OSError("analyzer crashed"))])
    assert result["pass"] is False
    assert result["document"]["chosen_model"] is None
    assert result["document"]["sparse_models"][0]["analyzer_rc"] is None
    assert "model_analyzer failed" in result["document"]["selection_error"]
    transcript = (result["evidence"] / "model_analyzer_0.txt").read_text(
        encoding="utf-8"
    )
    assert "analyzer crashed" in transcript


def test_metrics_on_stderr_are_read(tmp_path):
    text = analyzer_text(registered=15, points=20, observations="40.000000")
    result = _evaluate(tmp_path, 15, [("0", (0, "", text))])
    assert result["pass"] is True
    assert _bar(result)["registered"]["value"] == 15
    assert _bar(result)["points_info_only"] == 20


def test_nonfinite_observation_is_unparsable(tmp_path):
    result = _evaluate(
        tmp_path,
        15,
        [_passing(observations="nan", reprojection="1.0")],
    )
    assert result["pass"] is False
    assert _bar(result)["mean_obs_per_image"]["value"] is None
    assert "mean_obs_per_image fails or missing" in _bar(result)["fail_reasons"]
    stored = (result["evidence"] / EVIDENCE_JSON_NAME).read_text(encoding="utf-8")
    json.loads(stored)


@pytest.mark.parametrize("inputs", [None, 0, True, "15"])
def test_missing_input_count_fails_closed(tmp_path, inputs):
    result = _evaluate(tmp_path, inputs, [_passing(registered=15, points=400)])
    assert result["pass"] is False
    assert result["document"]["inputs"] is None
    assert _bar(result)["registered"]["threshold"] is None
    assert "input frame count missing/ambiguous" in result["document"]["fail_reasons"]
    assert result["document"]["chosen_model"] == "0"


def test_verdict_matches_v9_quality_bar_and_pick_largest(tmp_path):
    # Literals below are what sf4_collect_v9 quality_bar / pick_largest return
    # for these analyzer texts. The collector is not imported or executed.
    sample = FIXTURE.read_text(encoding="utf-8")
    rejected = _evaluate(tmp_path / "sample", 15, [("0", (0, sample))])
    assert rejected["document"]["largest_model"] == "0"
    assert rejected["document"]["chosen_model"] == "0"
    assert _bar(rejected) == {
        "model": "0",
        "inputs": 15,
        "registered": {
            "value": 2,
            "threshold": 12,
            "rule": ">= max(10, ceil(0.8*inputs))",
            "pass": False,
        },
        "mean_obs_per_image": {
            "value": 48.0,
            "threshold": 30.0,
            "rule": ">= 30",
            "label": "Mean observations per image:",
            "pass": True,
        },
        "points_info_only": 48,
        "mean_reproj_err_px": {
            "value": 0.609572,
            "threshold": 4.0,
            "rule": "<= 4.0",
            "pass": True,
        },
        "pass": False,
        "fail_reasons": ["registered fails or missing"],
    }
    by_registered = _evaluate(
        tmp_path / "registered",
        15,
        [
            _passing(
                name="0",
                registered=12,
                points=500,
                observations="40.0",
                reprojection="1.0",
            ),
            _passing(
                name="1",
                registered=13,
                points=20,
                observations="31.5",
                reprojection="2.5",
            ),
        ],
    )
    assert by_registered["document"]["largest_model"] == "1"
    assert by_registered["document"]["chosen_model"] == "1"
    assert _bar(by_registered) == {
        "model": "1",
        "inputs": 15,
        "registered": {
            "value": 13,
            "threshold": 12,
            "rule": ">= max(10, ceil(0.8*inputs))",
            "pass": True,
        },
        "mean_obs_per_image": {
            "value": 31.5,
            "threshold": 30.0,
            "rule": ">= 30",
            "label": "Mean observations per image:",
            "pass": True,
        },
        "points_info_only": 20,
        "mean_reproj_err_px": {
            "value": 2.5,
            "threshold": 4.0,
            "rule": "<= 4.0",
            "pass": True,
        },
        "pass": True,
        "fail_reasons": [],
    }
    by_points = _evaluate(
        tmp_path / "points",
        15,
        [
            _passing(
                name="0",
                registered=12,
                points=7,
                observations="45.0",
                reprojection="0.5",
            ),
            _passing(
                name="1",
                registered=12,
                points=8,
                observations="33.0",
                reprojection="3.0",
            ),
        ],
    )
    assert by_points["document"]["largest_model"] == "1"
    assert by_points["document"]["chosen_model"] == "1"
    assert _bar(by_points) == {
        "model": "1",
        "inputs": 15,
        "registered": {
            "value": 12,
            "threshold": 12,
            "rule": ">= max(10, ceil(0.8*inputs))",
            "pass": True,
        },
        "mean_obs_per_image": {
            "value": 33.0,
            "threshold": 30.0,
            "rule": ">= 30",
            "label": "Mean observations per image:",
            "pass": True,
        },
        "points_info_only": 8,
        "mean_reproj_err_px": {
            "value": 3.0,
            "threshold": 4.0,
            "rule": "<= 4.0",
            "pass": True,
        },
        "pass": True,
        "fail_reasons": [],
    }


def test_evidence_json_and_sha256sums(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "stale.txt").write_text("previous pass\n", encoding="utf-8")
    text_0 = analyzer_text(registered=12, points=10, observations="40.000000")
    text_1 = analyzer_text(
        registered=14, points=4, observations="36.000000", reprojection="0.500000"
    )
    result = _evaluate(
        tmp_path,
        15,
        [("0", (0, text_0)), ("1", (0, text_1))],
    )
    assert not (result["evidence"] / "stale.txt").exists()
    document_path = result["evidence"] / EVIDENCE_JSON_NAME
    document = json.loads(document_path.read_text(encoding="utf-8"))
    assert document["inputs"] == 15
    assert document["chosen_model"] == "1"
    assert document["largest_model"] == "1"
    assert document["verdict"] == "pass"
    assert document["pass"] is True
    assert document["fail_reasons"] == []
    assert [model["model"] for model in document["sparse_models"]] == ["0", "1"]
    for model, registered, points in (
        (document["sparse_models"][0], 12, 10),
        (document["sparse_models"][1], 14, 4),
    ):
        assert model["registered"] == registered
        assert model["points"] == points
        assert model["mean_obs_per_image"] > 30
        assert model["mean_reproj_err_px"] <= 4.0
        assert model["analyzer_rc"] == 0
        raw = (result["evidence"] / model["analyzer_file"]).read_text(encoding="utf-8")
        assert f"Registered images: {registered}" in raw
        assert "# ---- stdout ----" in raw
        assert "# ---- stderr ----" in raw
    bar = document["quality_bar"]
    assert bar["registered"]["value"] == 14
    assert bar["registered"]["threshold"] == 12
    assert bar["registered"]["rule"] == _REGISTERED_RULE
    assert bar["registered"]["pass"] is True
    assert bar["mean_obs_per_image"]["threshold"] == 30.0
    assert bar["mean_obs_per_image"]["rule"] == ">= 30"
    assert bar["mean_obs_per_image"]["pass"] is True
    assert bar["mean_reproj_err_px"]["threshold"] == 4.0
    assert bar["mean_reproj_err_px"]["rule"] == "<= 4.0"
    assert bar["mean_reproj_err_px"]["pass"] is True
    assert bar["points_info_only"] == 4
    sums_path = result["evidence"] / "SHA256SUMS"
    sums = sums_path.read_text(encoding="utf-8")
    assert sums.endswith("\n")
    lines = sums.splitlines()
    listed = []
    for line in lines:
        digest, relative = line.split("  ", 1)
        assert len(digest) == 64
        assert digest == digest.lower()
        assert relative != "SHA256SUMS"
        payload = (result["evidence"] / relative).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == digest
        listed.append(relative)
    assert listed == sorted(listed)
    on_disk = {
        path.relative_to(result["evidence"]).as_posix()
        for path in result["evidence"].rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    assert set(listed) == on_disk
    assert EVIDENCE_JSON_NAME in listed
    assert "model_analyzer_0.txt" in listed
    assert "model_analyzer_1.txt" in listed


def test_missing_colmap_binary_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sceneagent.reconstruction_gate.shutil.which", lambda _name: None
    )
    sparse = tmp_path / "sparse"
    (sparse / "0").mkdir(parents=True)
    result = accept_reconstruction(sparse, 15, tmp_path / "evidence")
    assert result["pass"] is False
    assert "model_analyzer failed" in result["document"]["selection_error"]
    assert run_model_analyzer(sparse / "0")["returncode"] is None


def _install_binaries(monkeypatch):
    # Both modules bind the stdlib shutil module, so the two setattr calls
    # are one patch. The function has to answer ffmpeg and colmap.
    def which(name):
        if name in {"ffmpeg", "colmap"}:
            return f"/usr/bin/{name}"
        return None

    monkeypatch.setattr("sceneagent.colmap_stage.shutil.which", which)
    monkeypatch.setattr("sceneagent.external_tools.shutil.which", which)

    def banner(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="COLMAP 3.11.1 -- Structure-from-Motion\n",
            stderr="",
        )

    monkeypatch.setattr("sceneagent.colmap_stage.subprocess.run", banner)


def _pose_run(monkeypatch, work, frames, models, analyzer_outputs):
    """Drive run_poses. ``_run`` never returns model_analyzer text."""
    seen = []

    def fake_run(command, timeout_s=600):
        seen.append(list(command))
        if Path(str(command[0])).name == "ffmpeg":
            folder = Path(command[-1]).parent
            for index in range(1, frames + 1):
                (folder / f"{index:04d}.jpg").write_bytes(b"frame")
        if len(command) > 1 and command[1] == "mapper":
            output = Path(command[command.index("--output_path") + 1])
            for name in models:
                (output / name).mkdir(parents=True)
        return {
            "command": command,
            "returncode": 0,
            "timed_out": False,
            "stdout_tail": analyzer_text(
                registered=frames, points=9999, observations="80.000000"
            ),
            "stderr_tail": "",
        }

    monkeypatch.setattr("sceneagent.colmap_stage._run", fake_run)

    def analyzer(model):
        return {"returncode": 0, "stdout": analyzer_outputs[model.name], "stderr": ""}

    result = run_poses(
        work / "clip.mp4", work / "colmap", color_calibrate=False, analyzer=analyzer
    )
    assert all(
        "model_analyzer" not in str(part) for command in seen for part in command
    )
    return result


def test_run_poses_rejects_two_of_fifteen(tmp_path, monkeypatch):
    _install_binaries(monkeypatch)
    text = FIXTURE.read_text(encoding="utf-8")
    result = _pose_run(monkeypatch, tmp_path, 15, ["0"], {"0": text})
    assert result["ok"] is False
    assert result["failed_step"] == "reconstruction_gate"
    assert "registered fails or missing" in result["reason"]
    assert "reconstruction not accepted" in result["reason"]
    assert result["reconstruction_gate"]["pass"] is False
    assert result["reconstruction_gate"]["chosen_model"] == "0"
    evidence = tmp_path / "colmap" / "reconstruction_gate" / EVIDENCE_JSON_NAME
    assert json.loads(evidence.read_text(encoding="utf-8"))["verdict"] == "fail"


def test_run_poses_accepts_the_largest_model_not_sparse_zero(tmp_path, monkeypatch):
    _install_binaries(monkeypatch)
    outputs = {
        "0": analyzer_text(registered=2, points=48, observations="48.000000"),
        "1": analyzer_text(registered=15, points=100, observations="40.000000"),
    }
    result = _pose_run(monkeypatch, tmp_path, 15, ["0", "1"], outputs)
    assert result["ok"] is True
    assert result["reason"] == ""
    assert Path(result["model_dir"]).name == "1"
    assert result["reconstruction_gate"]["chosen_model"] == "1"
    assert result["reconstruction_gate"]["pass"] is True
    assert "failed_step" not in result


def test_run_video_does_not_accept_a_failed_gate(tmp_path, monkeypatch):
    _install_binaries(monkeypatch)

    def fake_run(command, timeout_s=600):
        if Path(str(command[0])).name == "ffmpeg":
            folder = Path(command[-1]).parent
            for index in range(1, 16):
                (folder / f"{index:04d}.jpg").write_bytes(b"frame")
        if len(command) > 1 and command[1] == "mapper":
            output = Path(command[command.index("--output_path") + 1])
            (output / "0").mkdir(parents=True)
        return {
            "command": command,
            "returncode": 0,
            "timed_out": False,
            "stdout_tail": "",
            "stderr_tail": "",
        }

    monkeypatch.setattr("sceneagent.colmap_stage._run", fake_run)

    def analyzer(model):
        return {
            "returncode": 0,
            "stdout": FIXTURE.read_text(encoding="utf-8"),
            "stderr": "",
        }

    monkeypatch.setattr("sceneagent.reconstruction_gate.run_model_analyzer", analyzer)
    summary = run_video(FRUITS, tmp_path / "out", color_calibrate=False)
    assert summary["colmap_decision"] == "reject"
    assert summary["reconstruction"] == "not_run"
    video = json.loads((tmp_path / "out" / "video.json").read_text(encoding="utf-8"))
    assert video["reconstruction"] == "not_run"
    ledger = json.loads((tmp_path / "out" / "ledger.json").read_text(encoding="utf-8"))
    pose = ledger["entries"][1]
    assert pose["item"] == "COLMAP camera poses"
    assert pose["decision"] == "reject"
    assert pose["failed_step"] == "reconstruction_gate"
    assert "registered fails or missing" in pose["reason"]
    poses = json.loads(
        (tmp_path / "out" / "colmap_poses.json").read_text(encoding="utf-8")
    )
    assert poses["ok"] is False


def test_run_video_accepts_when_the_gate_passes(tmp_path, monkeypatch):
    _install_binaries(monkeypatch)

    def fake_run(command, timeout_s=600):
        if Path(str(command[0])).name == "ffmpeg":
            folder = Path(command[-1]).parent
            for index in range(1, 16):
                (folder / f"{index:04d}.jpg").write_bytes(b"frame")
        if len(command) > 1 and command[1] == "mapper":
            output = Path(command[command.index("--output_path") + 1])
            (output / "0").mkdir(parents=True)
            (output / "1").mkdir(parents=True)
        return {
            "command": command,
            "returncode": 0,
            "timed_out": False,
            "stdout_tail": "",
            "stderr_tail": "",
        }

    monkeypatch.setattr("sceneagent.colmap_stage._run", fake_run)

    def analyzer(model):
        registered = 2 if model.name == "0" else 15
        points = 48 if model.name == "0" else 100
        return {
            "returncode": 0,
            "stdout": analyzer_text(
                registered=registered, points=points, observations="40.000000"
            ),
            "stderr": "",
        }

    monkeypatch.setattr("sceneagent.reconstruction_gate.run_model_analyzer", analyzer)
    summary = run_video(FRUITS, tmp_path / "out", color_calibrate=False)
    assert summary["colmap_decision"] == "accept"
    assert summary["reconstruction"] == "colmap"
    ledger = json.loads((tmp_path / "out" / "ledger.json").read_text(encoding="utf-8"))
    pose = ledger["entries"][1]
    assert pose["decision"] == "accept"
    assert Path(pose["model_dir"]).name == "1"
    poses = json.loads(
        (tmp_path / "out" / "colmap_poses.json").read_text(encoding="utf-8")
    )
    assert poses["ok"] is True
    assert poses["reconstruction_gate"]["chosen_model"] == "1"
