# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SF-22 inspector: CPU views, the settle summary, and the model-written report.

No test calls Gemini, Isaac Lab or the box. The harness and the Isaac runner
are fakes; meshes are two small boxes in a temporary folder.
"""

from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from sceneagent import inspector as ins

pytest.importorskip("trimesh")
pytest.importorskip("PIL")


class FakeHarness:
    model_id = "gemini-3.8-flash"

    def __init__(self, replies):
        self.replies = list(replies)
        self.call_count = 0
        self.calls = []

    def complete(self, prompt, images=(), response_schema=None):
        self.call_count += 1
        self.calls.append((prompt, list(images), response_schema))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _scene(tmp_path: Path) -> Path:
    import trimesh

    trimesh.creation.box(extents=(0.1, 0.1, 0.1)).export(tmp_path / "cube.obj")
    trimesh.creation.box(extents=(0.2, 0.1, 0.05)).export(tmp_path / "slab.obj")
    scene = {
        "objects": [
            {"name": "cube", "mesh": "cube.obj", "translation": [0.0, 0.0, 0.05]},
            {"name": "slab", "mesh": "slab.obj", "translation": [0.3, 0.0, 0.025]},
        ]
    }
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(scene), encoding="utf-8")
    return path


def _report(findings, summary="Two boxes on the ground."):
    return json.dumps({"findings": findings, "summary": summary})


def test_render_writes_views_sheet_zip_and_an_honest_settle_summary(tmp_path):
    out = tmp_path / "out"
    result = ins.render_inspection(_scene(tmp_path), out)
    with zipfile.ZipFile(result["views_zip"]) as archive:
        names = set(archive.namelist())
    assert ins.SHEET_NAME in names
    assert {f"view_{v}.png" for v in ins.VIEWS} <= names
    summary = json.loads((out / ins.SETTLE_NAME).read_text(encoding="utf-8"))
    # No Isaac Lab here: the summary says no simulation ran, never "settled".
    assert summary["ran"] is False and summary["settled"] is None
    assert "no exported stage" in summary["reason"]
    assert summary["objects"] == ["cube", "slab"]


def test_each_object_is_drawn_in_its_own_colour(tmp_path):
    import numpy as np

    clouds = {
        "a": np.array([[0.0, 0.0, 0.05]] * 50),
        "b": np.array([[0.5, 0.0, 0.05]] * 50),
    }
    top = ins.render_views(clouds)["top"]
    for i in range(2):
        assert (top == np.array(ins._colour(i), dtype=np.uint8)).all(axis=2).any()


def test_a_missing_isaac_is_recorded_not_faked(tmp_path):
    stage = tmp_path / "bin_physics.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    summary = ins.settle(stage, None, tmp_path)
    assert summary == {
        "simulator": None,
        "ran": False,
        "settled": None,
        "reason": "Isaac Lab is not available on this host; no settle run",
    }


def test_the_isaac_settle_is_judged_against_the_sf35_bar(tmp_path):
    stage = tmp_path / "bin_physics.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    isaac = tmp_path / "python.sh"
    isaac.write_text("", encoding="utf-8")
    seen = {}

    def fake_runner(argv, **kwargs):
        seen["argv"] = argv
        Path(argv[-1]).write_text(
            json.dumps(
                {
                    "seconds": 2.0,
                    "bodies": [
                        {"path": "/World/cube", "z": 0.05, "speed": 0.002},
                        {"path": "/World/slab", "z": -0.01, "speed": 0.0},
                    ],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0, "", "")

    summary = ins.settle(stage, str(isaac), tmp_path, fake_runner)
    assert seen["argv"][0] == str(isaac) and seen["argv"][2] == str(stage)
    assert summary["ran"] is True and summary["settled"] is False
    slab = next(b for b in summary["bodies"] if b["path"] == "/World/slab")
    assert slab["below_ground"] is True and slab["at_rest"] is True


def test_a_failed_isaac_run_is_reported_with_its_exit(tmp_path):
    stage = tmp_path / "s.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    isaac = tmp_path / "python.sh"
    isaac.write_text("", encoding="utf-8")

    def failing(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 1, "", "ImportError: isaacsim")

    summary = ins.settle(stage, str(isaac), tmp_path, failing)
    assert summary["ran"] is False and summary["settled"] is None
    assert "exit 1" in summary["reason"] and "isaacsim" in summary["reason"]


def test_findings_from_the_model_make_it_a_model_fix_report():
    finding = {"object": "slab", "issue": "below_ground", "fix": "Raise it 1 cm."}
    harness = FakeHarness([_report([finding])])
    doc = ins.inspect(b"png", {"ran": False}, harness, "42")
    assert doc["fix_report_from_model"] is True
    assert doc["findings"] == [finding] and doc["model"] == "gemini-3.8-flash"
    assert doc["vlm_calls"] == 1 and doc["prompt_sha256"] == ins.prompt_sha256()
    prompt, images, schema = harness.calls[0]
    assert images == [(b"png", "image/png")] and schema is ins.REPORT_SCHEMA
    assert '"ran": false' in prompt  # the settle summary is in the prompt


@pytest.mark.parametrize(
    "reply",
    [
        RuntimeError("HTTP 503"),  # no model output at all
        "not json",
        _report([]),  # the model replied, but listed no fix
        json.dumps(
            {
                "findings": [{"object": "x", "issue": "melted", "fix": "y"}],
                "summary": "s",
            }
        ),
    ],
)
def test_dc3_the_field_stays_false_without_model_findings(reply):
    doc = ins.inspect(b"png", {}, FakeHarness([reply]))
    assert doc["fix_report_from_model"] is False
    assert doc["vlm_calls"] == 1


def test_replies_are_parsed_strictly():
    ok = ins.parse_report("```json\n" + _report([]) + "\n```")
    assert ok == {"findings": [], "summary": "Two boxes on the ground."}
    for bad in (
        "[]",
        json.dumps({"findings": []}),  # no summary
        json.dumps(
            {"findings": [{"object": "", "issue": "other", "fix": "f"}], "summary": "s"}
        ),
        json.dumps(
            {"findings": [{"object": "o", "issue": "other", "fix": ""}], "summary": "s"}
        ),
    ):
        with pytest.raises(ValueError):
            ins.parse_report(bad)


def test_the_schema_enum_is_the_issue_list():
    item = ins.REPORT_SCHEMA["properties"]["findings"]["items"]
    assert item["properties"]["issue"]["enum"] == list(ins.ISSUES)
    assert set(item["required"]) == {"object", "issue", "fix"}


def test_report_cli_reads_the_zip_and_writes_the_document(tmp_path):
    out = tmp_path / "render"
    ins.render_inspection(_scene(tmp_path), out)
    target = tmp_path / "inspection" / "isaac_lab.json"
    finding = {"object": "cube", "issue": "floating", "fix": "Lower it 2 cm."}
    code = ins.main(
        [
            "report",
            "--views",
            str(out / ins.VIEWS_ZIP),
            "--settle",
            str(out / ins.SETTLE_NAME),
            "--out",
            str(target),
            "--source-run",
            "123",
        ],
        harness=FakeHarness([_report([finding])]),
    )
    doc = json.loads(target.read_text(encoding="utf-8"))
    assert code == 0 and doc["fix_report_from_model"] is True
    assert doc["source_run"] == "123" and doc["settle"]["ran"] is False
    with pytest.raises(SystemExit):
        ins.main(
            [
                "report",
                "--views",
                "v",
                "--settle",
                "s",
                "--out",
                "o",
                "--source-run",
                "x;y",
            ],
            harness=FakeHarness([]),
        )


def test_the_report_module_reads_no_key():
    source = Path(ins.__file__).read_text(encoding="utf-8")
    # The harness alone reads the key; this module never touches the environment.
    assert "environ" not in source and "getenv" not in source


def _report_args(out: Path, target: Path, *extra: str) -> list[str]:
    return [
        "report",
        "--views",
        str(out / ins.VIEWS_ZIP),
        "--settle",
        str(out / ins.SETTLE_NAME),
        "--out",
        str(target),
        *extra,
    ]


def test_kan23_the_isaac_frame_is_sent_second_and_recorded(tmp_path):
    out = tmp_path / "render"
    ins.render_inspection(_scene(tmp_path), out)
    frame = tmp_path / "settle_frame.png"
    frame.write_bytes(b"isaac-frame-png")
    target = tmp_path / "isaac_lab.json"
    finding = {"object": "cube", "issue": "floating", "fix": "Lower it 2 cm."}
    harness = FakeHarness([_report([finding])])
    code = ins.main(_report_args(out, target, "--frame", str(frame)), harness=harness)
    assert code == 0 and harness.call_count == 1
    prompt, images, _schema = harness.calls[0]
    sheet = zipfile.ZipFile(out / ins.VIEWS_ZIP).read(ins.SHEET_NAME)
    assert images == [(sheet, "image/png"), (b"isaac-frame-png", "image/png")]
    assert ins.FRAME_PROMPT in prompt and "Isaac Lab" in prompt
    doc = json.loads(target.read_text(encoding="utf-8"))
    assert doc["isaac_frame"] is True and doc["fix_report_from_model"] is True
    assert [item["role"] for item in doc["images_sent"]] == [
        "contact_sheet",
        "isaac_frame",
    ]
    assert doc["images_sent"][1]["path"] == str(frame)
    assert doc["prompt_sha256"] == ins.prompt_sha256(with_frame=True)
    assert doc["prompt_sha256"] != ins.prompt_sha256()


@pytest.mark.parametrize("given", [False, True])
def test_kan23_without_a_frame_only_the_sheet_is_sent(tmp_path, given):
    out = tmp_path / "render"
    ins.render_inspection(_scene(tmp_path), out)
    target = tmp_path / "isaac_lab.json"
    missing = tmp_path / "no_such_frame.png"
    extra = ("--frame", str(missing)) if given else ()
    harness = FakeHarness([_report([])])
    ins.main(_report_args(out, target, *extra), harness=harness)
    prompt, images, _schema = harness.calls[0]
    assert len(images) == 1 and ins.FRAME_PROMPT not in prompt
    assert prompt.startswith(ins.INSPECT_PROMPT)
    doc = json.loads(target.read_text(encoding="utf-8"))
    assert doc["isaac_frame"] is False
    assert [item["role"] for item in doc["images_sent"]] == ["contact_sheet"]
    assert doc["prompt_sha256"] == ins.prompt_sha256()
    assert ("isaac_frame_missing" in doc) is given


def test_kan23_both_report_workflows_pass_the_frame():
    root = Path(__file__).resolve().parents[2] / ".github" / "workflows"
    for name, folder in (
        ("sceneagent-inspector-report.yml", "inspect-input"),
        ("sceneagent-mvp-review.yml", "review-input"),
    ):
        text = (root / name).read_text(encoding="utf-8")
        line = next(
            ln for ln in text.splitlines() if "sceneagent.inspector report" in ln
        )
        assert f"--frame {folder}/settle_frame.png" in line
        assert text.count("secrets.GEMINI_API_KEY") == 1
