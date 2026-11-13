# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Young's modulus and Poisson's ratio from one grounded Gemini call per class.

The note does not publish these numbers. A value is kept only when the reply
parses, the grounding metadata has a source URL, and the numbers sit in
range (E > 0, 0 < ν < 0.5). Otherwise the record is unresolved and no
default is written. The label is ``web_lookup``, never measured.

The API key is read from ``GEMINI_API_KEY`` at call time. It is not stored
on the record and is not placed in an error message.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from typing import Any

from sceneagent.gemini_client import API_ROOT, PAPER_MODEL_ID, redact
from sceneagent.gemini_client import _request as gemini_request
from sceneagent.soft_route import particle_params, route

SOURCE = "web_lookup"
UNRESOLVED = "unresolved"


class MissingGeminiKeyError(Exception):
    """``GEMINI_API_KEY`` is missing. The message does not carry a key."""

    def __init__(self) -> None:
        super().__init__("GEMINI_API_KEY is missing or empty")


def prompt_for_class(material_class: str) -> str:
    """One grounded question. The reply is JSON in pascals and a ratio."""
    name = " ".join(str(material_class).split())
    return (
        "Look up published values for the material class "
        f"{name!r}. Reply with one JSON object and no other text. "
        'Keys: "youngs_modulus_pa" (pascals, not GPa) and "poisson_ratio". '
        "Use a grounded source. Do not invent a number."
    )


def grounding_urls(payload: Mapping[str, Any]) -> list[str]:
    """Web URIs from Gemini grounding metadata. The reply text is ignored."""
    candidates = payload.get("candidates") or []
    if not candidates or not isinstance(candidates[0], Mapping):
        return []
    meta = candidates[0].get("groundingMetadata") or candidates[0].get(
        "grounding_metadata"
    )
    if not isinstance(meta, Mapping):
        return []
    chunks = meta.get("groundingChunks") or meta.get("grounding_chunks") or []
    found: list[str] = []
    for chunk in chunks:
        if not isinstance(chunk, Mapping):
            continue
        web = chunk.get("web") or {}
        uri = web.get("uri") if isinstance(web, Mapping) else None
        if isinstance(uri, str) and uri.strip():
            found.append(uri.strip())
    return found


def interpret(material_class: str, text: str, urls: Sequence[str]) -> dict[str, Any]:
    """Accept in-range numbers that have a grounding URL. Else unresolved."""
    record: dict[str, Any] = {
        "material_class": material_class,
        "youngs_modulus_pa": None,
        "poisson_ratio": None,
        "source": UNRESOLVED,
        "urls": [],
    }
    sources = [url for url in urls if isinstance(url, str) and url.strip()]
    parsed = _json_object(text)
    if not sources or parsed is None:
        return record
    youngs = _number(parsed, "youngs_modulus_pa", "E")
    poisson = _number(parsed, "poisson_ratio", "nu")
    if youngs is None or poisson is None or youngs <= 0.0 or not 0.0 < poisson < 0.5:
        return record
    record["youngs_modulus_pa"] = youngs
    record["poisson_ratio"] = poisson
    record["source"] = SOURCE
    record["urls"] = sources
    return record


def lookup_class(
    material_class: str,
    client: Any,
    cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One grounded call per class. A second ask for that class uses the cache."""
    stored = cache if cache is not None else {}
    key = " ".join(str(material_class or "").split()).casefold()
    if not key:
        return interpret("", "", [])
    if key in stored:
        return stored[key]
    reply = client.generate_grounded(prompt_for_class(material_class))
    record = interpret(
        str(material_class),
        str(reply.get("text") or ""),
        list(reply.get("urls") or []),
    )
    stored[key] = record
    return record


def trace_soft_objects(
    objects: Sequence[Mapping[str, Any]],
    client: Any,
    cache: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Look up each soft-routed object's class and trace it on ``particle_params``."""
    stored = cache if cache is not None else {}
    traced = []
    for obj in objects:
        decision, reason = route(obj)
        if decision != "physgaussian":
            continue
        material = obj.get("material_class") or obj.get("material") or ""
        elastic = lookup_class(str(material), client, stored)
        params = particle_params(obj, int(obj["n_gaussians"]), elastic)
        traced.append(
            {"object": obj.get("object"), "route": decision, "reason": reason, **params}
        )
    return traced


class GroundedGemini:
    """One ``generateContent`` call with Google Search grounding.

    ``__repr__`` hides the key. Error text is redacted before it is returned.
    """

    def __init__(self, api_key: str) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise MissingGeminiKeyError()
        self.api_key = api_key

    def __repr__(self) -> str:
        return "GroundedGemini(api_key='[redacted]')"

    def generate_grounded(self, prompt: str) -> dict[str, Any]:
        status, raw = gemini_request(
            f"{API_ROOT}/models/{PAPER_MODEL_ID}:generateContent",
            self.api_key,
            {
                "contents": [{"parts": [{"text": prompt}]}],
                "tools": [{"google_search": {}}],
            },
        )
        raw = redact(raw, self.api_key)
        if status != 200:
            return {
                "text": "",
                "urls": [],
                "http_status": status,
                "model_id": PAPER_MODEL_ID,
            }
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return {
                "text": "",
                "urls": [],
                "http_status": status,
                "model_id": PAPER_MODEL_ID,
            }
        text = ""
        candidates = payload.get("candidates") or []
        if candidates:
            parts = (candidates[0].get("content") or {}).get("parts") or []
            text = "".join(
                part.get("text", "") for part in parts if isinstance(part, dict)
            )
        return {
            "text": text.strip(),
            "urls": grounding_urls(payload),
            "http_status": status,
            "model_id": PAPER_MODEL_ID,
        }


def client_from_environ() -> GroundedGemini:
    """Build a client from ``GEMINI_API_KEY``. No other variable is read."""
    if "GEMINI_API_KEY" not in os.environ:
        raise MissingGeminiKeyError()
    return GroundedGemini(os.environ["GEMINI_API_KEY"])


def _json_object(text: str) -> dict[str, Any] | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _number(payload: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key not in payload or isinstance(payload[key], bool):
            continue
        try:
            value = float(payload[key])
        except (TypeError, ValueError):
            return None
        if math.isfinite(value):
            return value
        return None
    return None
