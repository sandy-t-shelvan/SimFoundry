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
from collections.abc import Mapping, Sequence
from typing import Any

from sceneagent.soft_route import particle_params, route

SOURCE = "web_lookup"
UNRESOLVED = "unresolved"


def prompt_for_class(material_class: str) -> str:
    """One grounded question. The reply is JSON in pascals and a ratio."""
    name = " ".join(str(material_class).split())
    return (
        "Look up published values for the material class "
        f"{name!r}. Reply with one JSON object and no other text. "
        'Keys: "youngs_modulus_pa" (pascals, not GPa) and "poisson_ratio". '
        "Use a grounded source. Do not invent a number."
    )


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
    reply = client.generate_images(prompt_for_class(material_class), search=True)
    record = interpret(
        str(material_class),
        str(reply.get("text") or ""),
        list(reply.get("urls") or []),
    )
    record["model_id"] = reply.get("model_id")
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
