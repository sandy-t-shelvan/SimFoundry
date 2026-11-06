# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Section 3.4 equation (2): three-source material vote.

The paper's vote is

    y* = argmax_y  sum_v  w_v * 1[y_v = y]

A vote whose class is ``unknown`` has its weight multiplied by
``physics.UNKNOWN_WEIGHT_SCALE`` (0.4, §3.4). ``vote_material`` is that sum.
This module does not reimplement it.

The three sources are the physics prior, the VLM, and keywords.

* The prior is included only when Sandeep's npz exists at the configured
  path (``mlp.train_prior.CHECKPOINT``, unless the caller passes another
  file). It is then loaded only through ``vote_loaded_prior``. A missing
  file is a null source: not a zero vector and not a default class. This
  module does not write or ship that npz. The catalog MLP row stays missing.
* The VLM weight is the crop confidence, and only when that confidence is
  inside ``[0, 1]``. ``unknown`` is scaled by 0.4 (§3.4 page 9).
* The keyword weight is ``taxonomy.DEFAULT_KEYWORD_WEIGHT`` (1.0). That
  number is a SimFoundry choice, not published: the paper says keyword
  priors use a fixed weight and does not give the number.

The material-vote VLM is ``GEMINI_FLASH_MODEL_ID`` (``gemini-3.8-flash``).
That model id is our configuration. The PDF's Gemini 2.5 Flash sentence is
the section 3.8 Meshy route, not this vote. The key is read only from
``os.environ['GEMINI_API_KEY']``.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sceneagent.gemini_client import GeminiClient, redact
from sceneagent.mlp.apply import decode_mean, vote_loaded_prior
from sceneagent.mlp.contract import PRIOR_DIM
from sceneagent.mlp.prior import PhysicsPrior
from sceneagent.physics import UNKNOWN_WEIGHT_SCALE, scaled_weight, vote_material
from sceneagent.taxonomy import (
    DEFAULT_KEYWORD_WEIGHT,
    MATERIAL_CLASSES,
    keyword_material,
    require_material,
)

logger = logging.getLogger(__name__)

# Our configuration for the material vote. The PDF's Gemini 2.5 Flash
# sentence is the section 3.8 Meshy route, not this vote.
GEMINI_FLASH_MODEL_ID = "gemini-3.8-flash"


class MissingGeminiKeyError(Exception):
    """``GEMINI_API_KEY`` is missing or empty. No client was constructed.

    The message is fixed. Arguments are ignored so a caller cannot append
    the key to the text.
    """

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        super().__init__("GEMINI_API_KEY is missing or empty")


class MaterialVlmError(Exception):
    """The VLM vote failed. The message must already be free of the API key."""


def gemini_api_key_from_environ() -> str:
    """Return ``os.environ['GEMINI_API_KEY']``.

    No other variable is consulted. A missing or blank value raises
    ``MissingGeminiKeyError`` and the value is not copied into the message.
    """

    if "GEMINI_API_KEY" not in os.environ:
        raise MissingGeminiKeyError()
    value = os.environ["GEMINI_API_KEY"]
    if not isinstance(value, str) or value.strip() == "":
        raise MissingGeminiKeyError()
    return value


def configured_prior_checkpoint(checkpoint: Path | str | None = None) -> Path:
    """Path of Sandeep's prior npz. Does not create the file."""

    if checkpoint is not None:
        return Path(checkpoint)
    from sceneagent.mlp import train_prior

    return Path(train_prior.CHECKPOINT)


def vlm_vote_prompt() -> str:
    """Ask for one class and a confidence. The prompt contains no API key."""

    classes = ", ".join(MATERIAL_CLASSES)
    return (
        "Section 3.4 material vote. The images are crops of one object. "
        "Reply with one JSON object and no other text. "
        'Keys: "material_class" and "confidence". '
        f"material_class is one of: {classes}. "
        "confidence is a number from 0 to 1."
    )


def parse_vlm_vote(text: str) -> dict[str, Any]:
    """Read ``class`` and ``confidence`` from a VLM JSON reply.

    The error text does not echo the reply, so a key that happened to be in
    the body is not copied into the exception.
    """

    raw = text or ""
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise MaterialVlmError("VLM reply did not contain a JSON object")
    try:
        payload = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        raise MaterialVlmError("VLM reply was not JSON") from None
    if not isinstance(payload, dict):
        raise MaterialVlmError("VLM reply was not a JSON object")
    material = payload.get("material_class", payload.get("class"))
    try:
        material = require_material(material)
    except ValueError:
        raise MaterialVlmError("VLM material class is not a published class") from None
    if "confidence" not in payload:
        raise MaterialVlmError("VLM reply did not include confidence")
    return {"class": material, "confidence": _confidence(payload["confidence"])}


class GeminiFlashMaterialClient:
    """Crop vote against ``gemini-3.8-flash``.

    Construct this only after ``gemini_api_key_from_environ`` has returned.
    ``repr`` and ``str`` name the model and do not include the key.
    """

    __slots__ = ("_http",)

    def __init__(self, api_key: str) -> None:
        if not isinstance(api_key, str) or api_key.strip() == "":
            raise MissingGeminiKeyError()
        self._http = GeminiClient(api_key)

    def __repr__(self) -> str:
        return f"GeminiFlashMaterialClient(model_id={GEMINI_FLASH_MODEL_ID!r})"

    def __str__(self) -> str:
        return repr(self)

    def classify_crops(self, crops: Sequence[bytes]) -> dict[str, Any]:
        """Return ``class`` and ``confidence`` for these crops. One HTTP call."""

        try:
            reply = self._http.generate(
                vlm_vote_prompt(),
                GEMINI_FLASH_MODEL_ID,
                images=tuple(crops),
            )
        except Exception as exc:
            raise self._scrubbed(exc) from None
        status = reply.get("http_status")
        if status != 200:
            detail = redact(str(reply.get("error_body") or ""), self._http.api_key)
            raise MaterialVlmError(
                f"{GEMINI_FLASH_MODEL_ID} returned HTTP {status}: {detail}"
            )
        return parse_vlm_vote(str(reply.get("text") or ""))

    def _scrubbed(self, exc: BaseException) -> MaterialVlmError:
        message = redact(f"{type(exc).__name__}: {exc}", self._http.api_key)
        return MaterialVlmError(message)


def open_vlm_client(client: Any = None, client_factory: Any = None) -> Any:
    """Return ``client``, or build one. The key check runs before the factory.

    A missing or blank ``GEMINI_API_KEY`` raises ``MissingGeminiKeyError``
    and does not call ``client_factory`` or construct ``GeminiFlashMaterialClient``.
    """

    if client is not None:
        return client
    key = gemini_api_key_from_environ()
    factory = GeminiFlashMaterialClient if client_factory is None else client_factory
    return factory(key)


def run_material_vote(
    out_dir: Path | str,
    objects: Sequence[Mapping[str, Any]],
    *,
    client: Any = None,
    client_factory: Any = None,
    checkpoint: Path | str | None = None,
) -> dict[str, Any]:
    """Vote each object and write the vote JSON plus a manifest.

    The manifest records ``vlm_calls``, the number of ``classify_crops``
    calls made in this run. It does not record the API key.

    An empty object list, and a present prior npz without features and a
    mask, both fail before a client is built and before any VLM call. A
    Gemini call is not spent on a vote that cannot be written.
    """

    if not objects:
        raise ValueError("material vote needs at least one object")
    prior_path = configured_prior_checkpoint(checkpoint)
    if prior_path.is_file():
        for obj in objects:
            _require_prior_inputs(obj.get("features"), obj.get("mask"))
            _require_prior_tensors(obj.get("features"), obj.get("mask"))
        # Load only after every object has features and a mask. A missing
        # row must keep the existing error, and a bad file or a bad tensor
        # must not call the VLM.
        PhysicsPrior.load(prior_path)
        for obj in objects:
            if not _mask_selects_a_gaussian(obj.get("mask")):
                raise ValueError("physics prior mask selects no Gaussian")
    resolved = open_vlm_client(client=client, client_factory=client_factory)
    counted = _CountedVlm(resolved)
    votes = [
        _vote_one(
            name=str(obj["name"]),
            crops=obj["crops"],
            features=obj.get("features"),
            mask=obj.get("mask"),
            checkpoint=checkpoint,
            client=counted,
        )
        for obj in objects
    ]
    out = Path(out_dir)
    # One object and many objects share this file. The vote objects
    # themselves are unchanged.
    vote_payload = {
        "equation": "2",
        "unknown_scale": UNKNOWN_WEIGHT_SCALE,
        "votes": votes,
    }
    vote_path = out / "material_vote.json"
    manifest = {
        "section": "3.4",
        "equation": "2",
        "step": "material_vote",
        "model_id": GEMINI_FLASH_MODEL_ID,
        "vlm_calls": counted.calls,
        "objects": [str(obj["name"]) for obj in objects],
    }
    _write_json(vote_path, vote_payload)
    manifest_path = out / "manifest.json"
    _write_json(manifest_path, manifest)
    logger.info(
        "equation (2) wrote %s vote(s); vlm_calls=%s",
        len(votes),
        counted.calls,
    )
    return {
        "votes": votes,
        "manifest": manifest,
        "vote_path": str(vote_path),
        "manifest_path": str(manifest_path),
    }


def _vote_one(name, crops, features, mask, checkpoint, client):
    # The prior file is checked before the VLM. A missing feature row must
    # not spend a Gemini call and then raise.
    prior_path = configured_prior_checkpoint(checkpoint)
    if prior_path.is_file():
        _require_prior_inputs(features, mask)
        _require_prior_tensors(features, mask)
        _require_usable_prior(prior_path, mask)
    reply = client.classify_crops(crops)
    if (
        not isinstance(reply, Mapping)
        or "class" not in reply
        or "confidence" not in reply
    ):
        raise MaterialVlmError("VLM vote did not include a class and a confidence")
    vlm_class = require_material(reply["class"])
    vlm_confidence = _confidence(reply["confidence"])
    keyword = keyword_material(name, weight=DEFAULT_KEYWORD_WEIGHT)
    vlm_proposal = {"source": "vlm", "class": vlm_class, "weight": vlm_confidence}
    keyword_proposal = {
        "source": "keyword",
        "class": keyword["class"],
        "weight": keyword["weight"],
    }
    if prior_path.is_file():
        loaded = vote_loaded_prior(
            features,
            mask,
            vlm=vlm_proposal,
            keyword=keyword_proposal,
            checkpoint=prior_path,
            name=name,
        )
        winner = loaded["winner"]
        scores = loaded["scores"]
        prior_source = _prior_source(loaded)
    else:
        # No npz: the prior is null. Do not pass a zero or default proposal.
        winner, scores = vote_material([vlm_proposal, keyword_proposal])
        prior_source = _null_source("physics_prior")
    sources = [
        prior_source,
        _class_source(
            "vlm",
            vlm_class,
            raw_confidence=vlm_confidence,
            weight_before_scale=vlm_confidence,
        ),
        _class_source(
            "keyword",
            keyword["class"],
            raw_confidence=None,
            weight_before_scale=keyword["weight"],
        ),
    ]
    score = None if winner is None else float(scores[winner])
    logger.info(
        "equation (2) object=%s class=%s prior=%s",
        name,
        winner,
        "null" if prior_source["null"] else "present",
    )
    return {
        "equation": "2",
        "object": name,
        "material_class": winner,
        "score": score,
        "scores": {key: float(value) for key, value in scores.items()},
        "unknown_scale": UNKNOWN_WEIGHT_SCALE,
        "sources": sources,
    }


def _prior_source(loaded: Mapping[str, Any]) -> dict[str, Any]:
    proposal = loaded["proposal"]
    decoded = decode_mean(loaded["mean"])
    probabilities = decoded["material_probability"]
    distribution = {
        material: float(probabilities[index])
        for index, material in enumerate(MATERIAL_CLASSES)
    }
    material = require_material(proposal["class"])
    raw = float(proposal["weight"])
    source = _class_source(
        "physics_prior",
        material,
        raw_confidence=raw,
        weight_before_scale=raw,
    )
    source["distribution"] = distribution
    return source


def _class_source(source, material, raw_confidence, weight_before_scale):
    """One present voter. ``weight`` is the value that enters equation (2)."""

    applied = material == "unknown"
    return {
        "source": source,
        "class": material,
        "distribution": None,
        "raw_confidence": None if raw_confidence is None else float(raw_confidence),
        "weight": float(scaled_weight(material, weight_before_scale)),
        "unknown_scale": UNKNOWN_WEIGHT_SCALE,
        "unknown_scale_applied": applied,
        "null": False,
    }


def _require_prior_tensors(features: Any, mask: Any) -> None:
    """Refuse a prior row whose shapes cannot be voted, before any VLM call.

    ``vote_loaded_prior`` needs ``(N, 912)`` features and a mask of length
    ``N``. Checking that here keeps a bad OpenCLIP or DINOv2 width from
    spending a Gemini call.
    """

    shape = getattr(features, "shape", None)
    if shape is None or len(tuple(shape)) != 2 or int(shape[1]) != PRIOR_DIM:
        raise ValueError(
            f"physics prior features must have shape (N, {PRIOR_DIM}), got {shape}"
        )
    count = int(shape[0])
    mask_shape = getattr(mask, "shape", None)
    if mask_shape is not None and len(tuple(mask_shape)) >= 1:
        length = int(mask_shape[0])
    else:
        try:
            length = len(mask)
        except TypeError:
            length = -1
    if length != count:
        raise ValueError("physics prior mask length must match the Gaussian count")
    values = np.asarray(features, dtype=np.float64)
    if values.shape != (count, PRIOR_DIM) or not np.isfinite(values).all():
        raise ValueError("physics prior features must be finite")


def _require_usable_prior(path: Path, mask: Any) -> None:
    """Load the npz and require a selected Gaussian before any VLM call."""

    PhysicsPrior.load(path)
    if not _mask_selects_a_gaussian(mask):
        raise ValueError("physics prior mask selects no Gaussian")


def _mask_selects_a_gaussian(mask: Any) -> bool:
    try:
        items = iter(mask)
    except TypeError:
        return bool(mask)
    for item in items:
        if item:
            return True
    return False


def _require_prior_inputs(features, mask) -> None:
    """Refuse a prior vote that has no Gaussian features or mask.

    Called before the VLM. The message is the only output; no client is built.
    """

    if features is None or mask is None:
        raise ValueError(
            "physics prior npz exists but features and mask were not provided"
        )


def _null_source(source: str) -> dict[str, Any]:
    """A source that did not vote. JSON null, not zeros and not a class."""

    return {
        "source": source,
        "class": None,
        "distribution": None,
        "raw_confidence": None,
        "weight": None,
        "unknown_scale": None,
        "unknown_scale_applied": False,
        "null": True,
    }


def _confidence(value: Any) -> float:
    """Accept a finite confidence in ``[0, 1]``. Do not clamp."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        raise MaterialVlmError("VLM confidence was not a number") from None
    # Reject values outside the closed interval. Clamping would turn 5 into 1.
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise MaterialVlmError("VLM confidence must be a finite number in [0, 1]")
    return number


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class _CountedVlm:
    """Counts ``classify_crops`` calls. ``repr`` does not delegate to the inner client."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls = 0

    def __repr__(self) -> str:
        return "CountedVlm(injected)"

    def __str__(self) -> str:
        return repr(self)

    def classify_crops(self, crops: Sequence[bytes]) -> Any:
        self.calls += 1
        return self._inner.classify_crops(crops)
