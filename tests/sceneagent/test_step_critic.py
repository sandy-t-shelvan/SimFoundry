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


def _reject(note="split the cup from the marker", prompt="paper cup . marker ."):
    return json.dumps(
        {
            "verdict": "reject",
            "reason": "one outline covers the cup and the marker",
            "retry_note": note,
            "retry_prompt": prompt,
        }
    )


def test_a_retry_prompt_must_already_be_detector_phrases():
    assert sc.retry_prompt("paper cup . marker .") == "paper cup . marker ."
    assert sc.retry_prompt(" red  mug.blue scissors ") == "red mug . blue scissors ."
    for bad in ("", None, "Mug . <b>", "a . b . c . d . e . f . g .", 3, "mug; rm"):
        assert sc.retry_prompt(bad) == ""
    # An accept carries no retry prompt, whatever the model wrote.
    accept = json.dumps({"verdict": "accept", "reason": "ok", "retry_prompt": "a ."})
    assert "retry_prompt" not in sc.parse_step_verdict(accept)
    assert sc.parse_step_verdict(_reject())["retry_prompt"] == "paper cup . marker ."


def test_a_reject_reruns_the_stage_once_with_the_note_and_both_go_to_the_ledger(
    tmp_path,
):
    out = _masks_run(tmp_path)
    (out / "ledger.json").write_text(
        json.dumps({"entries": [{"item": "SAM masks", "decision": "accept"}]})
    )
    harness = FakeHarness([_reject(), _reply("accept", "one outline per object")])
    reruns = []

    def rerun(path, request):
        reruns.append(request)
        return 0

    record = sc.review_and_retry(out, harness, "masks", rerun=rerun)
    assert reruns == [
        {
            "stage": "masks",
            "note": "split the cup from the marker",
            "prompt": "paper cup . marker .",
        }
    ]
    assert record["outcome"] == "accept" and record["vlm_calls"] == 2
    assert [a["verdict"] for a in record["attempts"]] == ["reject", "accept"]
    lines = json.loads((out / "ledger.json").read_text())["entries"]
    assert [(e["item"], e["decision"], e.get("attempt")) for e in lines] == [
        ("SAM masks", "accept", None),
        ("step review (masks)", "reject", 0),
        ("step review (masks)", "accept", 1),
    ]
    assert lines[1]["retry_note"] == "split the cup from the marker"
    assert lines[1]["retry_prompt"] == "paper cup . marker ."
    for attempt, verdict in ((0, "reject"), (1, "accept")):
        folder = out / sc.CRITIC_DIR / f"attempt-{attempt}"
        review = json.loads((folder / "review" / "masks.json").read_text())
        assert review["verdict"] == verdict and review["model"] == "gemini-3.8-flash"
        assert (folder / "review_steps" / "masks.png").is_file()
    assert json.loads((out / sc.CRITIC_DIR / "loop.json").read_text()) == record


def test_a_second_reject_is_logged_and_never_re_run(tmp_path):
    out = _masks_run(tmp_path)
    harness = FakeHarness([_reject(), _reject("still merged")])
    reruns = []
    record = sc.review_and_retry(
        out, harness, "masks", rerun=lambda p, r: reruns.append(r) or 0
    )
    assert len(reruns) == 1 and record["outcome"] == "reject"
    assert harness.call_count == 2
    lines = json.loads((out / "ledger.json").read_text())["entries"]
    assert [(e["decision"], e["attempt"]) for e in lines] == [
        ("reject", 0),
        ("reject", 1),
    ]


def test_a_reject_without_a_prompt_logs_a_skipped_retry(tmp_path):
    out = _masks_run(tmp_path)
    harness = FakeHarness([_reject(prompt="")])
    record = sc.review_and_retry(
        out, harness, "masks", rerun=lambda p, r: pytest.fail("must not re-run")
    )
    assert record["outcome"] == "reject, retry skipped"
    lines = json.loads((out / "ledger.json").read_text())["entries"]
    assert [(e["item"], e["decision"]) for e in lines] == [
        ("step review (masks)", "reject"),
        ("step retry (masks)", "skipped"),
    ]


def test_an_accept_needs_no_retry_and_the_cli_reports_it(tmp_path, capsys):
    out = _masks_run(tmp_path)
    harness = FakeHarness([_reply("accept", "fine")])
    assert sc.main(["review", "--out", str(out)], harness=harness) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == {"outcome": "accept", "model": "gemini-3.8-flash", "vlm_calls": 1}


def test_the_masks_retry_runs_the_mask_stage_with_the_critics_prompt(tmp_path):
    from sceneagent import bin_masks

    out = tmp_path / "video"
    (out / "mask_views").mkdir(parents=True)
    (out / "mask_views" / "0001.jpg").write_bytes(b"x")
    (out / "scene.json").write_text(
        json.dumps(
            {
                "source": "simfoundry_example_video",
                "name": "V",
                "ledger_name": "V",
                "views": 1,
                "input_item": "video",
                "view_dir": "mask_views",
                "min_masked_views": 1,
                "ground_truth": False,
                "prompt": "block . mailbox .",
            }
        )
    )
    seen = []

    class Done:
        returncode = 0
        stdout = stderr = ""

    def runner(argv, **kwargs):
        seen.append(argv)
        if "sceneagent.mask_worker" in argv:
            (out / "masks").mkdir(exist_ok=True)
            (out / "masks" / "detections.json").write_text(
                json.dumps({"0001.jpg": [{"phrase": "green cube", "kept": True}]})
            )
        return Done()

    code = bin_masks.run_bin_masks(
        out,
        runner=runner,
        weights=dict,
        prompt="green cube . red mailbox .",
        critic_note="the black panel is outlined",
    )
    assert code == 0
    worker = seen[-1]
    assert worker[worker.index("--prompt") + 1] == "green cube . red mailbox ."
    evidence = json.loads((out / "masks_evidence.json").read_text())
    assert evidence["prompt_source"] == "step critic retry (gemini note)"
    assert evidence["critic_retry"] == {
        "note": "the black panel is outlined",
        "scene_prompt": "block . mailbox .",
    }
    line = json.loads((out / "ledger.json").read_text())["entries"][-1]
    assert line["attempt"] == 1 and line["critic_note"] == "the black panel is outlined"


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
