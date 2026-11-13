# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SF-21 material audit. The client is a stub, so CI makes no call."""

from __future__ import annotations

import inspect
import json

import pytest

from sceneagent import gemini_client
from sceneagent.material_audit import (
    SIGNED_BY,
    audit_votes,
    main,
    parse_audit,
    prompt_for,
)
from sceneagent.physics_props import property_record
from sceneagent.vlm_port import MissingGeminiKeyError, client_from_environ

VOTE = {
    "object": "mug",
    "material_class": "plastic",
    "sources": [{"source": "keyword", "class": "plastic", "weight": 0.4}],
}


class Fake:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
        self.images = []

    def generate_images(self, prompt, images=(), search=False):
        assert search is False
        self.prompts.append(prompt)
        self.images.append(list(images))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return {
            "text": reply,
            "urls": [],
            "http_status": 200,
            "model_id": "stub-model",
        }


def _reply(decision, audit_class, reason="seen"):
    return json.dumps(
        {"decision": decision, "audit_class": audit_class, "reason": reason}
    )


def test_dc2_the_audit_overrides_the_vote_and_signs_the_review():
    client = Fake([_reply("replace", "glass and ceramic", "glazed")])
    document = audit_votes([VOTE], {"mug": [b"crop"]}, client)
    review = document["reviews"][0]
    assert review["provisional_class"] == "plastic"
    assert review["audit_class"] == "glass and ceramic"
    assert review["final_class"] == "glass and ceramic"
    assert review["decision"] == "replace"
    assert review["reason"] == "glazed"
    assert review["signed_by"] == SIGNED_BY
    assert document["signed_by"] == SIGNED_BY
    assert document["replaced"] == ["mug"]
    assert document["vlm_calls"] == 1
    assert document["prompt_sha256"]
    assert client.images == [[b"crop"]]
    assert document["model"] == "stub-model"
    assert "plastic" in client.prompts[0] and "keyword" in client.prompts[0]
    record = property_record("mug", vote=VOTE, review=review)
    assert record["material_class"] == "glass and ceramic"
    assert record["sources"]["material_class"] == "VLM audit (SF-21)"


def test_a_confirm_keeps_the_provisional_class():
    document = audit_votes(
        [VOTE], {}, Fake([_reply("confirm", "plastic", "the class stands")])
    )
    review = document["reviews"][0]
    assert review["decision"] == "confirm"
    assert review["final_class"] == "plastic"
    assert document["replaced"] == []
    assert document["failed"] == []


def test_a_failed_call_keeps_the_provisional_class():
    document = audit_votes([VOTE], {}, Fake([RuntimeError("http 500")]))
    review = document["reviews"][0]
    assert review["decision"] == "failed"
    assert review["final_class"] == "plastic"
    assert review["audit_class"] is None
    assert document["replaced"] == []
    assert document["failed"][0]["name"] == "mug"
    assert "500" in document["failed"][0]["error"]
    record = property_record("mug", vote=VOTE, review=review)
    assert record["material_class"] == "plastic"


@pytest.mark.parametrize(
    "text",
    [
        _reply("confirm", "wood"),
        _reply("replace", "plastic"),
        "not json",
        json.dumps({"decision": "maybe", "audit_class": "wood", "reason": "x"}),
    ],
)
def test_a_reply_that_disagrees_with_itself_is_a_failure(text):
    document = audit_votes([VOTE], {}, Fake([text]))
    assert document["reviews"][0]["decision"] == "failed"
    assert document["reviews"][0]["final_class"] == "plastic"
    assert document["replaced"] == []


def test_parse_audit_rejects_a_confirm_of_another_class():
    with pytest.raises(ValueError, match="does not match"):
        parse_audit(_reply("confirm", "wood"), "plastic")


def test_the_prompt_names_the_provisional_class_and_the_sources():
    text = prompt_for("mug", "plastic", VOTE["sources"])
    assert "Provisional class: plastic" in text
    assert "keyword" in text


def test_the_cli_writes_the_review_with_an_injected_client(tmp_path):
    vote = tmp_path / "material_vote.json"
    vote.write_text(json.dumps({"votes": [VOTE]}), encoding="utf-8")
    crops = tmp_path / "crops" / "mug"
    crops.mkdir(parents=True)
    (crops / "0.png").write_bytes(b"png")
    out = tmp_path / "material_review.json"
    code = main(
        [
            "--vote",
            str(vote),
            "--crops-dir",
            str(tmp_path / "crops"),
            "--out",
            str(out),
        ],
        client=Fake([_reply("replace", "glass and ceramic")]),
    )
    assert code == 0
    document = json.loads(out.read_text(encoding="utf-8"))
    assert document["reviews"][0]["final_class"] == "glass and ceramic"
    assert document["vlm_calls"] == 1


def test_the_audit_module_reads_no_key():
    import sceneagent.material_audit as audit

    source = inspect.getsource(audit)
    assert "os.environ" not in source
    assert "GEMINI_API_KEY" not in source


def test_client_from_environ_prefers_the_shared_helper(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        gemini_client, "client_from_environ", lambda: sentinel, raising=False
    )
    assert client_from_environ() is sentinel


def test_until_the_helper_lands_generate_images_matches_its_dict(monkeypatch):
    monkeypatch.delattr(gemini_client, "client_from_environ", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-value")
    seen = {}

    def fake_generate(
        self, prompt, model_id, image_bytes=None, mime_type="image/jpeg", images=()
    ):
        seen["images"] = images
        seen["mime"] = mime_type
        seen["model"] = model_id
        return {"text": "ok", "http_status": 200, "model_id": model_id}

    monkeypatch.setattr(gemini_client.GeminiClient, "generate", fake_generate)
    client = client_from_environ()
    reply = client.generate_images("hello", images=[b"png"])
    assert reply == {
        "text": "ok",
        "urls": [],
        "http_status": 200,
        "model_id": seen["model"],
    }
    assert seen["images"] == (b"png",) and seen["mime"] == "image/png"
    assert "super-secret-value" not in repr(client)
    with pytest.raises(RuntimeError, match="search grounding"):
        client.generate_images("hello", search=True)


def test_a_missing_key_is_not_echoed(monkeypatch):
    monkeypatch.delattr(gemini_client, "client_from_environ", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(MissingGeminiKeyError, match="missing or empty"):
        client_from_environ()
