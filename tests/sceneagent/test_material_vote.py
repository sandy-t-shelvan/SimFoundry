# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the section 3.4 equation (2) three-source vote.

No Gemini calls and no shipped prior npz. The with-prior case writes a
synthetic npz under tmp_path and loads it through vote_loaded_prior.
"""

from __future__ import annotations

import json
import logging
import traceback
import urllib.request
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from sceneagent.gemini_client import GeminiClient
from sceneagent.material_vote import (
    GEMINI_FLASH_MODEL_ID,
    GeminiFlashMaterialClient,
    MaterialVlmError,
    MissingGeminiKeyError,
    parse_vlm_vote,
    run_material_vote,
)
from sceneagent.mlp.apply import decode_mean, vote_loaded_prior
from sceneagent.mlp.prior import PhysicsPrior
from sceneagent.physics import UNKNOWN_WEIGHT_SCALE
from sceneagent.taxonomy import DEFAULT_KEYWORD_WEIGHT, MATERIAL_CLASSES

FAKE_KEY = "AQ.FAKE-TEST-KEY-123"


@pytest.fixture(autouse=True)
def _forbid_network(monkeypatch):
    def _blocked(*_args, **_kwargs):
        raise AssertionError("material vote tests must not open a network connection")

    monkeypatch.setattr(urllib.request, "urlopen", _blocked)


class _RecordingVlm:
    def __init__(self, material, confidence):
        self.material = material
        self.confidence = confidence
        self.calls = 0
        self.seen = []

    def classify_crops(self, crops):
        self.calls += 1
        self.seen.append(list(crops))
        return {"class": self.material, "confidence": self.confidence}


def _apple_unknown_vlm():
    return _RecordingVlm("unknown", 0.5)


def _run(out, client, objects=None, checkpoint=None):
    if objects is None:
        objects = [{"name": "apple", "crops": [b"crop-a", b"crop-b"]}]
    return run_material_vote(
        out,
        objects,
        client=client,
        checkpoint=checkpoint,
    )


def _source(vote, name):
    matches = [item for item in vote["sources"] if item["source"] == name]
    assert len(matches) == 1
    return matches[0]


def test_dc1_null_prior_when_npz_is_absent(tmp_path, monkeypatch):
    calls = []

    def _boom(*_args, **_kwargs):
        calls.append("vote_loaded_prior")
        raise AssertionError("vote_loaded_prior ran without an npz")

    monkeypatch.setattr("sceneagent.material_vote.vote_loaded_prior", _boom)
    missing = tmp_path / "no_such_physics_prior.npz"
    assert not missing.exists()
    features = np.zeros((2, 912))
    mask = np.array([True, True])
    client = _apple_unknown_vlm()
    body = _run(
        tmp_path / "explicit",
        client,
        objects=[
            {
                "name": "apple",
                "crops": [b"crop"],
                "features": features,
                "mask": mask,
            }
        ],
        checkpoint=missing,
    )
    vote = body["votes"][0]
    prior = _source(vote, "physics_prior")
    assert prior["null"] is True
    assert prior["class"] is None
    assert prior["distribution"] is None
    assert prior["raw_confidence"] is None
    assert prior["weight"] is None
    assert prior["unknown_scale"] is None
    assert prior["unknown_scale_applied"] is False
    assert vote["material_class"] == "food and organic matter"
    assert calls == []
    text = Path(body["vote_path"]).read_text(encoding="utf-8")
    assert '"class": null' in text
    assert '"distribution": null' in text
    assert text.count('"class": null') == 1

    configured = tmp_path / "configured" / "physics_prior.npz"
    monkeypatch.setattr("sceneagent.mlp.train_prior.CHECKPOINT", configured)
    assert not configured.exists()
    configured_run = run_material_vote(
        tmp_path / "configured-out",
        [{"name": "apple", "crops": [b"crop"]}],
        client=_apple_unknown_vlm(),
        checkpoint=None,
    )
    configured_prior = _source(configured_run["votes"][0], "physics_prior")
    assert configured_prior["null"] is True
    assert configured_prior["distribution"] is None
    assert configured_prior["class"] is None
    assert calls == []


def _synthetic_prior(path):
    prior = PhysicsPrior.initialize(1)
    prior.w1[:] = 0.0
    prior.b1[:] = 0.0
    prior.w2[:] = 0.0
    prior.b2[:] = 0.0
    prior.w_out[:] = 0.0
    prior.b_out[:] = 0.0
    prior.b_out[MATERIAL_CLASSES.index("metal")] = 8.0
    prior.b_out[24] = -0.7
    prior.b_out[25] = 6.2
    prior.b_out[26] = 2.0
    prior.b_out[27] = -2.0
    prior.save(path)
    return path


def test_dc1_with_prior_loads_synthetic_npz_through_vote_loaded_prior(
    tmp_path, monkeypatch
):
    path = _synthetic_prior(tmp_path / "physics_prior.npz")
    features = np.zeros((3, 912))
    mask = np.array([True, False, True])
    client = _RecordingVlm("plastic", 0.1)
    seen = {}
    real = vote_loaded_prior

    def _wrapped(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        seen["result"] = real(*args, **kwargs)
        return seen["result"]

    monkeypatch.setattr("sceneagent.material_vote.vote_loaded_prior", _wrapped)
    body = _run(
        tmp_path / "with-prior",
        client,
        objects=[
            {
                "name": "apple",
                "crops": [b"crop-a"],
                "features": features,
                "mask": mask,
            }
        ],
        checkpoint=path,
    )
    assert "result" in seen
    assert seen["kwargs"]["checkpoint"] == path
    loaded = seen["result"]
    vote = body["votes"][0]
    prior = _source(vote, "physics_prior")
    assert prior["null"] is False
    assert prior["class"] == loaded["proposal"]["class"] == "metal"
    assert prior["class"] is not None
    assert isinstance(prior["distribution"], dict)
    assert list(prior["distribution"]) == list(MATERIAL_CLASSES)
    assert sum(prior["distribution"].values()) == pytest.approx(1.0)
    assert prior["distribution"]["metal"] == pytest.approx(
        float(decode_mean(loaded["mean"])["material_probability"][1])
    )
    assert prior["distribution"]["metal"] > 0.9
    assert not all(value == 0.0 for value in prior["distribution"].values())
    assert prior["raw_confidence"] == pytest.approx(loaded["proposal"]["weight"])
    assert prior["unknown_scale_applied"] is False
    assert vote["material_class"] == loaded["winner"]
    assert vote["material_class"] == "food and organic matter"
    assert vote["scores"]["metal"] == pytest.approx(loaded["scores"]["metal"])
    assert vote["scores"]["plastic"] == pytest.approx(0.1)
    assert client.calls == 1


def test_dc2_vote_json_records_sources_weights_and_unknown_scale(tmp_path, monkeypatch):
    def _boom(*_args, **_kwargs):
        raise AssertionError("vote_loaded_prior ran without an npz")

    monkeypatch.setattr("sceneagent.material_vote.vote_loaded_prior", _boom)
    assert UNKNOWN_WEIGHT_SCALE == 0.4
    assert DEFAULT_KEYWORD_WEIGHT == 1.0
    client = _apple_unknown_vlm()
    body = _run(tmp_path / "dc2", client, checkpoint=tmp_path / "absent.npz")
    vote = json.loads(Path(body["vote_path"]).read_text(encoding="utf-8"))
    expected = {
        "equation": "2",
        "object": "apple",
        "material_class": "food and organic matter",
        "score": 1.0,
        "scores": {
            "unknown": 0.2,
            "food and organic matter": 1.0,
        },
        "unknown_scale": 0.4,
        "sources": [
            {
                "source": "physics_prior",
                "class": None,
                "distribution": None,
                "raw_confidence": None,
                "weight": None,
                "unknown_scale": None,
                "unknown_scale_applied": False,
                "null": True,
            },
            {
                "source": "vlm",
                "class": "unknown",
                "distribution": None,
                "raw_confidence": 0.5,
                "weight": 0.2,
                "unknown_scale": 0.4,
                "unknown_scale_applied": True,
                "null": False,
            },
            {
                "source": "keyword",
                "class": "food and organic matter",
                "distribution": None,
                "raw_confidence": None,
                "weight": 1.0,
                "unknown_scale": 0.4,
                "unknown_scale_applied": False,
                "null": False,
            },
        ],
    }
    assert vote == expected
    assert body["votes"][0] == expected
    vlm = _source(vote, "vlm")
    assert vlm["unknown_scale"] == UNKNOWN_WEIGHT_SCALE == 0.4
    assert vlm["unknown_scale_applied"] is True
    assert vlm["weight"] == pytest.approx(vlm["raw_confidence"] * 0.4)
    keyword = _source(vote, "keyword")
    assert keyword["weight"] == DEFAULT_KEYWORD_WEIGHT
    assert keyword["unknown_scale_applied"] is False


@pytest.mark.parametrize("stored", [None, "", "   ", "\n"])
def test_dc3_missing_or_blank_key_raises_before_client_construction(
    tmp_path, monkeypatch, stored
):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", FAKE_KEY)
    if stored is not None:
        monkeypatch.setenv("GEMINI_API_KEY", stored)
    factory = Mock(side_effect=AssertionError("client factory was called"))
    out = tmp_path / "missing-key"
    with pytest.raises(MissingGeminiKeyError) as caught:
        run_material_vote(
            out,
            [{"name": "apple", "crops": [b"crop"]}],
            client=None,
            client_factory=factory,
        )
    assert factory.call_count == 0
    assert factory.called is False
    message = str(caught.value)
    assert message == "GEMINI_API_KEY is missing or empty"
    assert FAKE_KEY not in message
    assert FAKE_KEY not in "".join(
        traceback.format_exception(type(caught.value), caught.value, caught.tb)
    )
    assert str(MissingGeminiKeyError(FAKE_KEY)) == "GEMINI_API_KEY is missing or empty"
    assert FAKE_KEY not in repr(MissingGeminiKeyError(FAKE_KEY))
    assert not (out / "manifest.json").exists()
    assert not (out / "material_vote.json").exists()


def test_dc3_fake_key_is_absent_from_logs_manifest_vote_and_errors(
    tmp_path, monkeypatch, caplog, capsys
):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="sceneagent.material_vote")
    client = _RecordingVlm("wood", 0.25)
    exc_text = ""
    body = None
    try:
        body = _run(tmp_path / "redact", client, checkpoint=tmp_path / "absent.npz")
    except Exception as exc:
        exc_text = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
    captured = capsys.readouterr()
    vote_text = ""
    manifest_text = ""
    if body is not None:
        vote_text = Path(body["vote_path"]).read_text(encoding="utf-8")
        manifest_text = Path(body["manifest_path"]).read_text(encoding="utf-8")
    blob = "\n".join(
        [caplog.text, captured.out, captured.err, exc_text, vote_text, manifest_text]
    )
    assert FAKE_KEY not in blob
    assert exc_text == ""
    assert body is not None
    manifest = json.loads(manifest_text)
    assert manifest["vlm_calls"] == client.calls == 1
    assert manifest["model_id"] == "gemini-3.8-flash"
    assert "api_key" not in manifest_text
    assert "GEMINI_API_KEY" not in manifest_text
    real = GeminiFlashMaterialClient(FAKE_KEY)
    assert FAKE_KEY not in repr(real)
    assert FAKE_KEY not in str(real)
    assert FAKE_KEY not in repr(real._http)
    assert FAKE_KEY not in str(real._http)


def test_dc3_client_errors_do_not_contain_the_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)

    def _explode(
        self, prompt, model_id, image_bytes=None, mime_type="image/jpeg", images=()
    ):
        raise RuntimeError(f"transport failed for {FAKE_KEY}")

    monkeypatch.setattr(GeminiClient, "generate", _explode)
    client = GeminiFlashMaterialClient(FAKE_KEY)
    with pytest.raises(MaterialVlmError) as caught:
        client.classify_crops([b"crop"])
    rendered = "".join(
        traceback.format_exception(type(caught.value), caught.value, caught.tb)
    )
    assert FAKE_KEY not in str(caught.value)
    assert FAKE_KEY not in rendered
    assert caught.value.__cause__ is None

    def _http_error(
        self, prompt, model_id, image_bytes=None, mime_type="image/jpeg", images=()
    ):
        return {
            "http_status": 400,
            "text": "",
            "error_body": f"body {FAKE_KEY}",
        }

    monkeypatch.setattr(GeminiClient, "generate", _http_error)
    with pytest.raises(MaterialVlmError) as http_caught:
        client.classify_crops([b"crop"])
    assert FAKE_KEY not in str(http_caught.value)


def test_dc3_manifest_vlm_calls_equals_mocked_invocations(
    tmp_path, monkeypatch, caplog, capsys
):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    caplog.set_level(logging.DEBUG)
    client = _RecordingVlm("plastic", 0.3)
    body = run_material_vote(
        tmp_path / "counts",
        [
            {"name": "apple", "crops": [b"a"]},
            {"name": "cup", "crops": [b"b", b"c"]},
        ],
        client=client,
        checkpoint=tmp_path / "absent.npz",
    )
    manifest_text = Path(body["manifest_path"]).read_text(encoding="utf-8")
    vote_text = Path(body["vote_path"]).read_text(encoding="utf-8")
    captured = capsys.readouterr()
    blob = "\n".join(
        [caplog.text, captured.out, captured.err, manifest_text, vote_text]
    )
    assert FAKE_KEY not in blob
    manifest = json.loads(manifest_text)
    assert manifest["vlm_calls"] == client.calls == 2
    assert client.seen == [[b"a"], [b"b", b"c"]]
    assert len(body["votes"]) == 2


def test_gemini_flash_client_sends_crops_to_gemini_3_8_flash(monkeypatch):
    sent = {}

    def _fake(
        self, prompt, model_id, image_bytes=None, mime_type="image/jpeg", images=()
    ):
        sent["model_id"] = model_id
        sent["images"] = images
        sent["prompt"] = prompt
        return {
            "http_status": 200,
            "text": '{"material_class": "wood", "confidence": 0.25}',
            "error_body": "",
        }

    monkeypatch.setattr(GeminiClient, "generate", _fake)
    client = GeminiFlashMaterialClient(FAKE_KEY)
    parsed = client.classify_crops([b"one", b"two"])
    assert sent["model_id"] == GEMINI_FLASH_MODEL_ID == "gemini-3.8-flash"
    assert sent["images"] == (b"one", b"two")
    assert "Section 3.4" in sent["prompt"]
    assert FAKE_KEY not in sent["prompt"]
    assert parsed == {"class": "wood", "confidence": 0.25}
    direct = parse_vlm_vote('prefix {"material_class": "wood", "confidence": 0.25}')
    assert direct == parsed


def test_parse_vlm_vote_rejects_a_non_class_without_echoing_the_body():
    with pytest.raises(MaterialVlmError) as caught:
        parse_vlm_vote('{"material_class": "unobtanium", "confidence": 0.2}')
    assert "unobtanium" not in str(caught.value)
    with pytest.raises(MaterialVlmError):
        parse_vlm_vote("not json")
