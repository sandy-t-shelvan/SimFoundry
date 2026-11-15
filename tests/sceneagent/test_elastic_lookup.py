# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Web lookup of E and ν. The Gemini client is a stub, so CI makes no call."""

from __future__ import annotations

import pytest

from sceneagent.gemini_client import (
    MissingGeminiKeyError,
    client_from_environ,
    grounding_urls,
)
from sceneagent.elastic_lookup import (
    SOURCE,
    UNRESOLVED,
    interpret,
    lookup_class,
    trace_soft_objects,
)
from sceneagent.soft_route import particle_params

URL = "https://example.com/foam"
TEXT = '{"youngs_modulus_pa": 2000000, "poisson_ratio": 0.3}'


class Stub:
    def __init__(self, text=TEXT, urls=(URL,)):
        self.text = text
        self.urls = list(urls)
        self.prompts = []

    def generate_images(self, prompt, images=(), search=False):
        self.prompts.append(prompt)
        assert search is True
        return {"text": self.text, "urls": list(self.urls), "model_id": "stub-model"}


def test_a_grounded_in_range_reply_is_a_web_lookup_and_is_cached():
    stub = Stub()
    cache = {}
    first = lookup_class("rubber and foam", stub, cache)
    second = lookup_class("rubber and foam", stub, cache)
    assert first is second
    assert len(stub.prompts) == 1
    assert first["source"] == SOURCE
    assert first["youngs_modulus_pa"] == pytest.approx(2e6)
    assert first["poisson_ratio"] == pytest.approx(0.3)
    assert first["urls"] == [URL]
    assert first["model_id"] == "stub-model"


def test_urls_come_from_grounding_metadata_not_the_reply_text():
    payload = {
        "candidates": [
            {
                "content": {"parts": [{"text": "see https://evil.example/fake"}]},
                "groundingMetadata": {"groundingChunks": [{"web": {"uri": URL}}]},
            }
        ]
    }
    assert grounding_urls(payload) == [URL]
    missed = interpret("foam", '{"youngs_modulus_pa": 1e6, "poisson_ratio": 0.2}', [])
    assert missed["source"] == UNRESOLVED
    assert missed["youngs_modulus_pa"] is None and missed["poisson_ratio"] is None


@pytest.mark.parametrize(
    ("text", "urls"),
    [
        ('{"youngs_modulus_pa": 0, "poisson_ratio": 0.3}', [URL]),
        ('{"youngs_modulus_pa": -5, "poisson_ratio": 0.3}', [URL]),
        ('{"youngs_modulus_pa": 1e6, "poisson_ratio": 0}', [URL]),
        ('{"youngs_modulus_pa": 1e6, "poisson_ratio": 0.5}', [URL]),
        ('{"youngs_modulus_pa": 1e6, "poisson_ratio": 0.9}', [URL]),
        ("not json", [URL]),
        (TEXT, []),
    ],
)
def test_out_of_range_or_ungrounded_stays_unresolved(text, urls):
    record = interpret("foam", text, urls)
    assert record["source"] == UNRESOLVED
    assert record["youngs_modulus_pa"] is None
    assert record["poisson_ratio"] is None
    assert record["urls"] == []


def test_particle_params_traces_the_lookup_and_not_a_default():
    elastic = interpret("rubber and foam", TEXT, [URL])
    traced = particle_params({"mass_kg": 0.5, "rigidity": 0.25}, 4, elastic)
    assert traced["youngs_modulus_pa"] == pytest.approx(2e6)
    assert traced["poisson_ratio"] == pytest.approx(0.3)
    assert traced["elastic_source"] == SOURCE
    text = "\n".join(traced["log"])
    assert "source: web_lookup" in text
    assert "material_class: rubber and foam" in text
    assert URL in text
    assert "measured" not in text
    blank = interpret("foam", TEXT, [])
    unresolved = particle_params({"mass_kg": 0.5, "rigidity": 0.25}, 4, blank)
    assert unresolved["youngs_modulus_pa"] is None
    assert "youngs_modulus_pa: unresolved" in unresolved["log"]
    assert "poisson_ratio: unresolved" in unresolved["log"]


def test_active_model_id_follows_one_constant(monkeypatch):
    from sceneagent.gemini_client import DEFAULT_MODEL_ID, active_model_id

    monkeypatch.delenv("GEMINI_MODEL_ID", raising=False)
    assert active_model_id() == DEFAULT_MODEL_ID
    monkeypatch.setenv("GEMINI_MODEL_ID", "override-id")
    assert active_model_id() == "override-id"


def test_only_soft_objects_are_looked_up_and_a_missing_key_is_not_echoed(monkeypatch):
    stub = Stub()
    objects = [
        {
            "object": "dough",
            "body_type": "soft deformable",
            "deformability": 0.2,
            "material_class": "food and organic matter",
            "mass_kg": 0.05,
            "rigidity": 0.2,
            "n_gaussians": 10,
        },
        {
            "object": "mug",
            "body_type": "dynamic rigid",
            "deformability": 0.1,
            "material_class": "glass and ceramic",
            "mass_kg": 0.4,
            "rigidity": 0.9,
            "n_gaussians": 8,
        },
    ]
    traced = trace_soft_objects(objects, stub)
    assert [row["object"] for row in traced] == ["dough"]  # not in the table
    assert stub.prompts and "glass and ceramic" not in stub.prompts[0]
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(MissingGeminiKeyError, match="missing or empty"):
        client_from_environ()
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-value")
    client = client_from_environ()
    assert "super-secret-value" not in repr(client)


def test_the_table_answers_by_keyword_then_class_default_and_carries_its_sources():
    from sceneagent.elastic_lookup import TABLE_SOURCE, table_lookup

    tyre = table_lookup("rubber and foam", "rubber duck")
    assert tyre["material"] == "rubber" and tyre["source"] == TABLE_SOURCE
    assert 0 < tyre["poisson_ratio"] < 0.5 and tyre["youngs_modulus_pa"] > 0
    assert tyre["urls"] and all(u.startswith("https://") for u in tyre["urls"])
    can = table_lookup("metal", "aluminium can")
    assert can["material"] == "aluminum"
    pot = table_lookup("metal", "pot")
    assert pot["material"].startswith("steel")  # class default
    sponge = table_lookup("rubber and foam", "kitchen sponge")
    assert sponge["material"].startswith("flexible open-cell polyurethane foam")
    assert 1e3 < sponge["youngs_modulus_pa"] < 1e5  # kPa, not rubber's MPa
    assert table_lookup("rubber and foam", "ball") is None  # no class default
    assert table_lookup("fabric", "towel") is None


def test_a_table_miss_falls_back_to_the_grounded_lookup_and_a_hit_does_not_call():
    from sceneagent.elastic_lookup import trace_soft_objects

    class Client:
        calls = 0

        def generate_images(self, prompt, images=(), search=False):
            Client.calls += 1
            return {"text": "{}", "urls": [], "http_status": 200}

    soft = {
        "body_type": "soft deformable",
        "deformability": 0.8,
        "mass_kg": 0.05,
        "rigidity": 0.2,
        "n_gaussians": 100,
    }
    hit = trace_soft_objects(
        [dict(soft, object="rubber ball", material_class="rubber and foam")], Client()
    )
    assert Client.calls == 0 and hit[0]["elastic_source"] == "material_table"
    miss = trace_soft_objects(
        [dict(soft, object="towel", material_class="fabric")], Client()
    )
    assert Client.calls == 1 and miss[0]["elastic_source"] == "unresolved"


def test_every_table_row_is_complete_and_in_range():
    import json

    from sceneagent.elastic_lookup import TABLE_PATH

    for row in json.loads(TABLE_PATH.read_text(encoding="utf-8"))["rows"]:
        e, nu = row["youngs_modulus_pa"], row["poisson_ratio"]
        assert 0 < e["low"] <= e["typical"] <= e["high"]
        assert 0 < nu["low"] <= nu["typical"] <= nu["high"] < 0.5
        assert len(row["sources"]) == 2 and all(s["quote"] for s in row["sources"])
