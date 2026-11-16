# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KAN-21/22 on a finished run: crops, vote, audit. A stub client; no network."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from sceneagent import physics_props as pp
from sceneagent.gemini_client import MissingGeminiKeyError
from sceneagent.material_audit import SIGNED_BY
from sceneagent.material_run import main, run, write_crops

VIEWS = ["0001.jpg", "0011.jpg", "0021.jpg", "0031.jpg"]


def _fake_run(root):
    """Two placed objects (cup, marker) and one unplaced, in four views."""
    (root / "mask_views").mkdir(parents=True)
    (root / "masks").mkdir()
    for k, view in enumerate(VIEWS):
        Image.fromarray(np.full((60, 80, 3), 40 * k, np.uint8)).save(
            root / "mask_views" / view
        )
        masks = np.zeros((3, 60, 80), dtype=bool)
        masks[0, 10:20, 10:30] = True
        if k != 2:  # the marker's mask is empty in one view
            masks[1, 30:40, 40:70] = True
        masks[2, 0:5, 0:5] = True
        np.savez(root / "masks" / f"{view[:4]}.npz", masks=masks)
    objects = [
        {
            "object": n,
            "masks": [{"view": v, "mask_index": n, "phrase": phrase} for v in VIEWS],
        }
        for n, phrase in enumerate(["cup", "marker", "table"])
    ]
    (root / "association_evidence.json").write_text(
        json.dumps({"object_list": objects})
    )
    placed = [{"object": 0, "placed": True}, {"object": 1, "placed": True}]
    placed.append({"object": 2, "placed": False})
    (root / "bin_score.json").write_text(json.dumps({"icp": {"objects": placed}}))
    return root


class Stub:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def generate_images(self, prompt, images=(), search=False):
        self.calls.append((prompt, list(images)))
        return {
            "text": self.replies.pop(0),
            "urls": [],
            "http_status": 200,
            "model_id": "stub-model",
        }


def _vote(material, confidence):
    return json.dumps({"material_class": material, "confidence": confidence})


def _audit(decision, material, reason):
    return json.dumps({"decision": decision, "audit_class": material, "reason": reason})


def test_write_crops_writes_png_per_placed_object_from_its_mask_views(tmp_path):
    source = _fake_run(tmp_path / "run")
    written = write_crops(source, tmp_path / "crops", views=4)
    assert [w["name"] for w in written] == ["cup", "marker"]
    assert written[0]["views"] == VIEWS
    # The empty marker mask in 0021.jpg is skipped, not written as a crop.
    assert written[1]["views"] == ["0001.jpg", "0011.jpg", "0031.jpg"]
    assert sorted(p.name for p in (tmp_path / "crops" / "cup").iterdir()) == [
        "00.png",
        "01.png",
        "02.png",
        "03.png",
    ]
    assert not (tmp_path / "crops" / "table").exists()
    with Image.open(tmp_path / "crops" / "cup" / "00.png") as crop:
        assert max(crop.size) >= 512  # crop_to_object enlarges small objects


def test_write_crops_spreads_views_over_the_orbit(tmp_path):
    source = _fake_run(tmp_path / "run")
    written = write_crops(source, tmp_path / "crops", views=2)
    assert written[0]["views"] == ["0001.jpg", "0031.jpg"]


def test_run_votes_audits_and_writes_a_signed_review_with_an_override(tmp_path):
    source = _fake_run(tmp_path / "run")
    out = tmp_path / "out"
    client = Stub(
        [
            _vote("plastic", 0.9),  # cup
            _vote("unknown", 0.5),  # marker
            _audit("replace", "metal", "brushed steel"),
            _audit("confirm", "plastic", "moulded barrel"),
        ]
    )
    summary = run(source, out, client, views=2)
    assert summary["vlm_calls"] == 4 and len(client.calls) == 4
    assert all(len(images) == 2 for _, images in client.calls)

    vote = json.loads((out / "material_vote.json").read_text())
    assert vote["unknown_scale"] == 0.4
    marker = vote["votes"][1]
    by_source = {s["source"]: s for s in marker["sources"]}
    assert by_source["physics_prior"]["null"] is True  # no packs in this run
    assert by_source["vlm"]["unknown_scale_applied"] is True
    assert by_source["vlm"]["weight"] == pytest.approx(0.5 * 0.4)
    assert json.loads((out / "manifest.json").read_text())["vlm_calls"] == 2

    review = json.loads((out / "material_review.json").read_text())
    assert review["vlm_calls"] == 2 and review["signed_by"] == SIGNED_BY
    cup = review["reviews"][0]
    # Keyword "cup" (1.0) outvotes the VLM's plastic (0.9); the audit overrides.
    assert cup["provisional_class"] == "glass and ceramic"
    assert cup["audit_class"] == "metal" and cup["decision"] == "replace"
    assert cup["final_class"] == "metal" and cup["reason"] == "brushed steel"
    assert summary["replaced"] == ["cup"]

    record = json.loads((out / "physics" / "object_0.json").read_text())
    assert record["material_class"] == "metal"
    assert record["sources"]["material_class"] == "VLM audit (SF-21)"
    assert not (source / "physics").exists()  # the run is read only


def test_run_refuses_an_output_inside_the_run(tmp_path):
    source = _fake_run(tmp_path / "run")
    with pytest.raises(ValueError, match="inside the run"):
        run(source, source / "material", Stub([]))


def test_main_without_a_key_fails_before_any_call(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    source = _fake_run(tmp_path / "run")
    with pytest.raises(MissingGeminiKeyError):
        main(["--run", str(source), "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_run_physics_applies_a_review_in_the_material_folder(tmp_path):
    source = _fake_run(tmp_path / "run")
    client = Stub(
        [
            _vote("glass and ceramic", 0.9),
            _vote("plastic", 0.9),
            _audit("replace", "metal", "steel"),
            _audit("confirm", "plastic", "barrel"),
        ]
    )
    run(source, tmp_path / "material", client, views=1)
    assert pp.run_physics(source, material=tmp_path / "material") == 0
    cup = json.loads((source / "physics" / "object_0.json").read_text())
    assert cup["material_class"] == "metal"
    assert cup["material_review"]["signed_by"] == SIGNED_BY
