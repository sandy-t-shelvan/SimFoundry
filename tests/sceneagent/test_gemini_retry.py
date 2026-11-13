# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retries and the one model id. No live call."""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path

from sceneagent.fracture_proposal import propose_fracture
from sceneagent.gemini_client import (
    DEFAULT_MODEL_ID,
    GeminiClient,
    _request,
    active_model_id,
)
from sceneagent.harness import GeminiHarness
from sceneagent.vlm_check import advise


class _Body:
    def __init__(self, status, raw):
        self.status = status
        self._raw = raw

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def _ok_payload(text="ok"):
    return json.dumps(
        {"candidates": [{"content": {"parts": [{"text": text}]}}]}
    ).encode()


def test_request_retries_503_and_timeout_then_returns(monkeypatch):
    calls = {"n": 0}

    def urlopen(_request, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("timed out")
        if calls["n"] == 2:
            raise urllib.error.HTTPError(
                "https://example.invalid",
                503,
                "busy",
                {},
                io.BytesIO(b"busy"),
            )
        return _Body(200, _ok_payload())

    monkeypatch.setattr("sceneagent.gemini_client.urllib.request.urlopen", urlopen)
    monkeypatch.setattr("sceneagent.gemini_client._pause", lambda _seconds: None)
    status, raw = _request("https://example.invalid", "key", {"contents": []})
    assert calls["n"] == 3
    assert status == 200
    assert "ok" in raw


def test_request_gives_up_after_three_timeouts(monkeypatch):
    calls = {"n": 0}

    def urlopen(_request, timeout=None):
        calls["n"] += 1
        raise TimeoutError("timed out")

    pauses = []
    monkeypatch.setattr("sceneagent.gemini_client.urllib.request.urlopen", urlopen)
    monkeypatch.setattr(
        "sceneagent.gemini_client._pause", lambda seconds: pauses.append(seconds)
    )
    status, raw = _request("https://example.invalid", "key", {})
    assert calls["n"] == 3
    assert pauses == [1.0, 2.0]
    assert status == 0
    assert raw == "timeout"


def test_generate_flash_and_harness_use_the_active_id(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL_ID", raising=False)
    seen = []

    def fake_request(url, key, payload=None):
        seen.append(url)
        return 200, _ok_payload("wood").decode()

    monkeypatch.setattr("sceneagent.gemini_client._request", fake_request)
    result = GeminiClient("key").generate_flash("hello")
    assert result["model_id"] == DEFAULT_MODEL_ID
    assert result["http_status"] == 200
    assert seen[0].endswith(f"/models/{DEFAULT_MODEL_ID}:generateContent")

    monkeypatch.setenv("GEMINI_MODEL_ID", "override-id")
    urls = []

    def transport(url, key, payload):
        urls.append(url)
        return 200, _ok_payload("plan").decode()

    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    text = GeminiHarness(transport=transport).complete("plan")
    assert text == "plan"
    assert urls[0].endswith("/models/override-id:generateContent")
    assert active_model_id() == "override-id"


def test_a_404_tries_the_one_fallback_list(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL_ID", raising=False)
    calls = []

    def fake_request(url, key, payload=None):
        calls.append(url)
        if url.rstrip("/").endswith("/models") or "/models?" in url:
            listed = {
                "models": [
                    {"name": f"models/{DEFAULT_MODEL_ID}"},
                    {"name": "models/gemini-flash-latest"},
                ]
            }
            return 200, json.dumps(listed)
        if DEFAULT_MODEL_ID in url:
            body = (
                f"This model models/{DEFAULT_MODEL_ID} is no longer available "
                "to new users. Please update your code to use "
                "models/gemini-flash-latest."
            )
            return 404, body
        return 200, _ok_payload("wood").decode()

    monkeypatch.setattr("sceneagent.gemini_client._request", fake_request)
    result = GeminiClient("key").generate_flash("hello")
    assert result["model_id"] == "gemini-flash-latest"
    assert result["http_status"] == 200
    assert all("/models/gemini-2.5-flash:" not in url for url in calls)


def test_overload_is_unresolved_for_the_advisory_and_fracture_records():
    class Down:
        def generate_images(self, prompt, images=(), search=False):
            return {"text": "", "http_status": 429, "model_id": "stub-model"}

    report = advise([b"a", b"b"], "does it move?", Down())
    assert report["status"] == "unresolved"
    assert report["verdict"] == "unparsed"
    assert report["http_status"] == 429
    record = propose_fracture("lid", [b"a", b"b"], Down())
    assert record["status"] == "unresolved"
    assert record["error"] == "HTTP 429"
    assert record["render"] == "skipped"


def test_call_sites_do_not_keep_a_second_model_constant():
    root = Path(__file__).resolve().parents[2] / "sceneagent"
    names = (
        "gemini_client.py",
        "harness.py",
        "material_vote.py",
        "hinge_proposal.py",
        "vlm_check.py",
        "fracture_proposal.py",
        "fruits_agent.py",
        "orchestrator.py",
        "plan_runner.py",
    )
    for name in names:
        text = (root / name).read_text(encoding="utf-8")
        assert "PAPER_MODEL_ID" not in text
        assert "GEMINI_FLASH_MODEL_ID" not in text
        assert "GEMINI_MODEL_ID =" not in text
        assert '"gemini-2.5-flash"' not in text
        assert "'gemini-2.5-flash'" not in text
    history = (root / "section34.py").read_text(encoding="utf-8")
    assert "PAPER_MODEL_ID" not in history
