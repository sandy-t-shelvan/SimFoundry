# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Web lookup of E and ν. The Gemini client is a stub, so CI makes no call."""

from __future__ import annotations

import pytest

from sceneagent.elastic_lookup import (
    MissingGeminiKeyError,
    SOURCE,
    UNRESOLVED,
    client_from_environ,
    grounding_urls,
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

    def generate_grounded(self, prompt):
        self.prompts.append(prompt)
        return {"text": self.text, "urls": list(self.urls)}


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


def test_only_soft_objects_are_looked_up_and_a_missing_key_is_not_echoed(monkeypatch):
    stub = Stub()
    objects = [
        {
            "object": "sponge",
            "body_type": "soft deformable",
            "deformability": 0.2,
            "material_class": "rubber and foam",
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
    assert [row["object"] for row in traced] == ["sponge"]
    assert stub.prompts and "glass and ceramic" not in stub.prompts[0]
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(MissingGeminiKeyError, match="missing or empty"):
        client_from_environ()
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-value")
    client = client_from_environ()
    assert "super-secret-value" not in repr(client)
