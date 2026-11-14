# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SF-18 step-runner critic. No test calls Gemini; the harness is a fake."""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest

from sceneagent import step_critic as sc

pytest.importorskip("PIL")
pytest.importorskip("scipy")


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


def _reply(verdict, reason="r", note=""):
    return json.dumps({"verdict": verdict, "reason": reason, "retry_note": note})


def test_every_stage_has_its_own_hashed_prompt():
    assert set(sc.PROMPTS) == set(sc.STAGES)
    hashes = {sc.prompt_sha256(s) for s in sc.STAGES}
    assert len(hashes) == len(sc.STAGES)
    assert sc.RESPONSE_SCHEMA["properties"]["verdict"]["enum"] == ["accept", "reject"]


def test_replies_must_say_accept_or_reject():
    parsed = sc.parse_step_verdict(
        "```json\n" + _reply("reject", "merged", "split it") + "\n```"
    )
    assert parsed == {"verdict": "reject", "reason": "merged", "retry_note": "split it"}
    # An accept carries no retry note, whatever the model wrote.
    assert sc.parse_step_verdict(_reply("accept", "ok", "x"))["retry_note"] == ""
    for bad in ('{"verdict": "keep"}', '"accept"', "accept", "[]"):
        with pytest.raises(ValueError):
            sc.parse_step_verdict(bad)


def test_images_are_read_by_stage_name_in_stage_order(tmp_path):
    zpath = tmp_path / "review_steps.zip"
    with zipfile.ZipFile(zpath, "w") as archive:
        for name in ("mesh.png", "masks.png", "notes.txt", "other.png"):
            archive.writestr(name, name.encode())
    assert list(sc.read_stage_images(zpath)) == ["masks", "mesh"]


def test_the_mock_vlm_judges_each_stage_and_records_failures(tmp_path):
    images = {"masks": b"m", "gaussians": b"g"}
    harness = FakeHarness(
        [_reply("reject", "bin outlined", "drop the bin"), RuntimeError("HTTP 503")]
    )
    doc = sc.judge_steps(images, harness, "42999999999")
    assert doc["model"] == "gemini-3.8-flash" and doc["vlm_calls"] == 2
    assert [s["stage"] for s in doc["stages"]] == ["masks"]
    assert doc["stages"][0]["verdict"] == "reject"
    assert doc["stages"][0]["prompt_sha256"] == sc.prompt_sha256("masks")
    assert [f["stage"] for f in doc["failed"]] == ["gaussians"]
    assert harness.calls[0][0] == sc.PROMPTS["masks"]
    assert harness.calls[0][1] == [(b"m", "image/png")]
    written = sc.write_stage_reviews(doc, tmp_path / "step-out")
    review = json.loads((tmp_path / "step-out" / "review" / "masks.json").read_text())
    # Jira SF-18 dc1: image path, verdict, reason and model.
    assert review["image"] == "review_steps/masks.png"
    assert review["verdict"] == "reject" and review["reason"] == "bin outlined"
    assert review["model"] == "gemini-3.8-flash"
    assert written[-1].name == "step_verdicts.json"


def test_a_reject_becomes_a_ledger_line_and_exactly_one_retry(tmp_path):
    doc = sc.judge_steps(
        {"masks": b"m", "mesh": b"x"},
        FakeHarness(
            [
                _reply("reject", "two fruits in one mask", "split the merged fruit"),
                _reply("accept"),
            ]
        ),
    )
    path = tmp_path / "step_verdicts.json"
    path.write_text(json.dumps(doc))
    verdicts = sc.load_step_verdicts(path)
    lines = sc.ledger_entries(verdicts)
    assert [line["item"] for line in lines] == [
        "step review (masks)",
        "step review (mesh)",
    ]
    assert (
        lines[0]["decision"] == "reject"
        and lines[0]["retry_note"] == "split the merged fruit"
    )
    assert lines[0]["vlm_model"] == "gemini-3.8-flash" and "retry_note" not in lines[1]
    assert sc.retry_requests(verdicts) == [
        {"stage": "masks", "note": "split the merged fruit"}
    ]
    # One retry only: a second reject of the same stage does not re-run it.
    assert sc.retry_requests(verdicts, retried={"masks": 1}) == []
    assert sc.ledger_entries(verdicts, attempt=1)[0]["attempt"] == 1


def test_step_verdicts_are_checked(tmp_path):
    path = tmp_path / "v.json"
    good = {
        "mode": "step",
        "model": "gemini-3.8-flash",
        "stages": [{"stage": "masks", "verdict": "accept"}],
    }
    path.write_text(json.dumps(good))
    assert sc.load_step_verdicts(path)["masks"]["verdict"] == "accept"
    for bad in (
        {**good, "mode": "accept"},
        {**good, "model": ""},
        {**good, "stages": [{"stage": "masks", "verdict": "maybe"}]},
        {**good, "stages": [{"stage": "../x", "verdict": "accept"}]},
    ):
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            sc.load_step_verdicts(path)


def _masks_run(tmp_path):
    from PIL import Image

    out = tmp_path / "bin"
    (out / "images").mkdir(parents=True)
    (out / "masks").mkdir()
    photo = np.full((60, 80, 3), 120, np.uint8)
    Image.fromarray(photo).save(out / "images" / "000001.png")
    masks = np.zeros((2, 60, 80), bool)
    masks[0, 10:30, 10:30] = True
    masks[1, 35:55, 40:70] = True
    np.savez_compressed(out / "masks" / "000001.npz", masks=masks)
    detections = {
        "000001.png": [
            {"phrase": "apple", "kept": True, "mask_index": 0},
            {"phrase": "box", "kept": True, "mask_index": 1},
            {"phrase": "bin", "kept": False},
        ]
    }
    (out / "masks" / "detections.json").write_text(json.dumps(detections))
    return out


def test_the_masks_review_image_outlines_each_kept_mask(tmp_path):
    from PIL import Image

    out = _masks_run(tmp_path)
    path = sc.render_masks(out)
    assert path == out / "review_steps" / "masks.png"
    image = np.asarray(Image.open(path))
    assert image.shape == (sc.TILE_PX, sc.TILE_PX, 3)
    colours = {tuple(c) for c in sc._palette(2)}
    seen = {tuple(px) for px in image.reshape(-1, 3)}
    assert colours <= seen  # both outlines drawn in their own colours
    archive = sc.pack_review_images(out)
    with zipfile.ZipFile(archive) as z:
        assert z.namelist() == ["masks.png"]
        assert Image.open(io.BytesIO(z.read("masks.png"))).size == (
            sc.TILE_PX,
            sc.TILE_PX,
        )


def test_the_masks_image_reads_the_scene_s_own_view_folder(tmp_path):
    """A video's masks are drawn on mask_views/, not images/ (scene.json view_dir)."""
    import dataclasses

    from sceneagent.bin_run import BIN_SCENE, VIDEO_MASK_DIR, write_scene

    out = _masks_run(tmp_path)
    (out / "images").rename(out / VIDEO_MASK_DIR)
    with pytest.raises(FileNotFoundError):
        sc.render_masks(out)  # no scene.json: the bin, whose views are images/
    write_scene(out, dataclasses.replace(BIN_SCENE, view_dir=VIDEO_MASK_DIR))
    assert sc.render_masks(out) == out / "review_steps" / "masks.png"


def test_the_cli_judges_a_zip_with_an_injected_harness(tmp_path):
    out = _masks_run(tmp_path)
    assert sc.main(["render-masks", "--out", str(out)]) == 0
    harness = FakeHarness([_reply("accept", "each object has one outline")])
    code = sc.main(
        [
            "judge",
            "--images",
            str(out / "review_steps.zip"),
            "--out-dir",
            str(tmp_path / "o"),
            "--source-run",
            "1",
        ],
        harness=harness,
    )
    assert code == 0
    assert (
        json.loads((tmp_path / "o" / "review" / "masks.json").read_text())["verdict"]
        == "accept"
    )
    with pytest.raises(SystemExit):
        sc.main(
            ["judge", "--images", "x.zip", "--out-dir", "o", "--source-run", "1; rm"],
            harness=harness,
        )


def test_the_critic_never_reads_a_key_itself():
    # The key stays inside harness.GeminiHarness; this module only gets a harness.
    import inspect

    source = inspect.getsource(sc)
    assert "os.environ" not in source and "load_gemini_api_key" not in source
    assert "import os" not in source


def _ply(path, xyz):
    xyz = np.asarray(xyz, dtype=np.float32)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + xyz.tobytes())


def _colours(path):
    from PIL import Image

    image = np.asarray(Image.open(path))
    return {tuple(int(c) for c in px) for px in image.reshape(-1, 3)}


def test_gaussians_mesh_and_parts_each_get_a_review_image(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    _ply(
        tmp_path / "splat" / "export" / "splat.ply",
        [[0, 0, 0], [0.2, 0, 0], [4, 0, 0], [4.2, 0, 0]],
    )
    (tmp_path / "association").mkdir()
    np.savez(
        tmp_path / "association" / "objects.npz",
        object_0=np.array([0, 1]),
        object_1=np.array([2, 3]),
    )
    (tmp_path / "sam3d").mkdir()
    trimesh.creation.box().export(tmp_path / "sam3d" / "object_0.glb")
    (tmp_path / "parts").mkdir()
    np.savez(
        tmp_path / "parts" / "object_0.npz",
        body=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        lid=np.array([[3.0, 0.0, 0.0], [3.0, 1.0, 0.0]]),
    )
    assert sc.main(["render", "--out", str(tmp_path)]) == 0
    drawn = ("gaussians.png", "mesh.png", "parts.png")
    with zipfile.ZipFile(tmp_path / "review_steps.zip") as archive:
        assert archive.namelist() == list(drawn)
    assert {tuple(c) for c in sc._palette(2)} <= _colours(
        tmp_path / "review_steps" / "gaussians.png"
    )
    mesh = _colours(tmp_path / "review_steps" / "mesh.png")
    assert (70, 70, 70) in mesh and (30, 90, 220) in mesh
    assert {tuple(c) for c in sc._palette(2)} <= _colours(
        tmp_path / "review_steps" / "parts.png"
    )


def test_render_with_nothing_to_draw_exits_nonzero(tmp_path):
    assert sc.main(["render", "--out", str(tmp_path)]) == 1
    assert not (tmp_path / "review_steps.zip").exists()
