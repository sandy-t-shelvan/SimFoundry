# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One harness adapter behind ``agent_loop.harness_status``.

Figure 5 names six products. ``harness_status`` still reports every one of
them uncalled. Grok CLI is on that list and is not selected: the vendor pages
do not show it running headless in CI on the existing Cursor plan without a
new paid credential. See ``docs/sf24-harness-finding.md``.

The callable adapter is the Gemini fallback from :func:`sceneagent.gemini_client.active_model_id`.
The key is read only from ``GEMINI_API_KEY`` (the GitHub secret of that name). A
missing key raises ``MissingGeminiApiKeyError`` before any HTTP call.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from collections.abc import Sequence
from typing import Any, Protocol

from sceneagent.agent_loop import harness_status
from sceneagent.gemini_client import API_ROOT, active_model_id, redact

API_KEY_ENV = "GEMINI_API_KEY"
logger = logging.getLogger("sceneagent.harness")


class MissingGeminiApiKeyError(RuntimeError):
    """``GEMINI_API_KEY`` is unset. No harness call was made."""


class HarnessCallError(RuntimeError):
    """The Gemini call failed. The message has been redacted."""


class PlanHarness(Protocol):
    """The one method the orchestrator uses. Figure 5 names are not this."""

    name: str
    model_id: str
    call_count: int

    def complete(self, prompt: str) -> str:
        """Return the model text. Increment ``call_count`` only if a call starts."""


class _RedactFilter(logging.Filter):
    """Strip the key from any log record emitted during a call."""

    def __init__(self, secret: str) -> None:
        super().__init__()
        self._secret = secret

    def filter(self, record: logging.LogRecord) -> bool:
        secret = self._secret
        if not secret:
            return True
        if isinstance(record.msg, str):
            record.msg = record.msg.replace(secret, "[redacted]")
        if isinstance(record.args, tuple):
            record.args = tuple(
                arg.replace(secret, "[redacted]") if isinstance(arg, str) else arg
                for arg in record.args
            )
        return True


def load_gemini_api_key() -> str:
    """Read ``GEMINI_API_KEY`` from the environment and nowhere else.

    A file, ``GOOGLE_API_KEY``, or a function argument is not a substitute.
    The error text does not include a value, because a missing key has none.
    """

    raw = os.environ.get(API_KEY_ENV)
    if raw is None or not str(raw).strip().strip('"').strip("'"):
        raise MissingGeminiApiKeyError(
            "GEMINI_API_KEY is not set. The harness was not called."
        )
    return str(raw).strip().strip('"').strip("'")


def _candidate_text(raw: str) -> str:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    candidates = payload.get("candidates") or []
    if not candidates:
        return ""
    parts = (candidates[0].get("content") or {}).get("parts") or []
    return "".join(part.get("text", "") for part in parts).strip()


def _default_transport(url: str, key: str, payload: dict[str, Any]) -> tuple[int, str]:
    """POST generateContent through the shared request, which retries."""

    from sceneagent.gemini_client import _request

    try:
        status, body = _request(url, key, payload)
    except Exception as exc:
        raise HarnessCallError(
            redact(f"Gemini harness call failed: {exc}", key)
        ) from None
    return int(status), redact("" if body is None else str(body), key)


class GeminiHarness:
    """Gemini adapter. The id is :func:`active_model_id`. Tests pass a fake ``transport``."""

    def __init__(self, transport: Any = None) -> None:
        self._transport = transport or _default_transport
        self.call_count = 0
        self.model_id = active_model_id()
        self.name = self.model_id

    def __repr__(self) -> str:
        return (
            f"GeminiHarness(model_id={self.model_id!r}, call_count={self.call_count})"
        )

    def complete(
        self,
        prompt: str,
        images: Sequence[tuple[bytes, str]] = (),
        response_schema: dict[str, Any] | None = None,
    ) -> str:
        """Call the model. The key check runs before ``call_count`` or HTTP.

        ``images`` are ``(bytes, mime_type)`` pairs sent inline after the
        prompt. ``response_schema`` asks for JSON in that shape; the caller
        still validates the reply, since a schema is a request, not a promise.
        """

        key = load_gemini_api_key()
        url = f"{API_ROOT}/models/{self.model_id}:generateContent"
        parts: list[dict[str, Any]] = [{"text": prompt}]
        for blob, mime_type in images:
            parts.append(
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64.b64encode(blob).decode("ascii"),
                    }
                }
            )
        payload: dict[str, Any] = {"contents": [{"parts": parts}]}
        if response_schema is not None:
            payload["generationConfig"] = {
                "responseMimeType": "application/json",
                "responseSchema": response_schema,
            }
        redactor = _RedactFilter(key)
        logger.addFilter(redactor)
        failed: HarnessCallError | None = None
        status = 0
        raw = ""
        try:
            self.call_count += 1
            try:
                status, raw = self._transport(url, key, payload)
            except HarnessCallError as exc:
                failed = exc
            except Exception as exc:
                failed = HarnessCallError(
                    redact(f"Gemini harness call failed: {exc}", key)
                )
            if failed is None:
                raw = redact("" if raw is None else str(raw), key)
                logger.info(
                    "harness call model=%s http=%s count=%s",
                    self.model_id,
                    status,
                    self.call_count,
                )
        finally:
            logger.removeFilter(redactor)
        if failed is not None:
            raise failed
        if status != 200:
            raise HarnessCallError(f"Gemini harness call failed with HTTP {status}")
        return redact(_candidate_text(raw), key)


def selected_harness() -> GeminiHarness:
    """Build the one adapter the figure 5 status record allows.

    ``harness_status`` must still name Grok CLI and must still report that
    none of the six products was called. This function does not mark them
    called, and it does not construct a Grok CLI client.
    """

    status = harness_status()
    if "Grok CLI" not in status["harnesses"]:
        raise RuntimeError("Grok CLI is not in the figure 5 harness list")
    if status["called"]:
        raise RuntimeError("figure 5 harnesses must stay uncalled")
    if status["status"] != "still_open":
        raise RuntimeError("figure 5 harness status changed")
    return GeminiHarness()


def request_plan(prompt: str, harness: PlanHarness | None = None) -> str:
    """Ask the selected adapter for a plan. A figure 5 name is refused."""

    status = harness_status()
    if status["called"]:
        raise RuntimeError("figure 5 harnesses must stay uncalled")
    impl = harness if harness is not None else selected_harness()
    if impl.name in status["harnesses"]:
        raise RuntimeError(
            f"{impl.name} is named by the figure 5 harness API and is not called"
        )
    text = impl.complete(prompt)
    if not isinstance(text, str):
        raise HarnessCallError("harness reply is not text")
    return text
