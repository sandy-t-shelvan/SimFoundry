# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Call ``generate_images`` the way the shared helper will.

``GeminiClient.generate_images`` and ``client_from_environ`` land on
``sceneagent.gemini_client`` with the mailbox hinge branch. This module
uses that pair when it is present. Until then it calls ``GeminiClient.generate``
already on main and returns the same dict: ``text``, ``urls``,
``http_status``, ``model_id``. Images are raw bytes. Search grounding is
left to the shared helper. This file does not copy that helper.
"""

from __future__ import annotations

import os
from typing import Any, Protocol

from sceneagent.gemini_client import GeminiClient


class MissingGeminiKeyError(Exception):
    """The API key is missing. The message does not carry a key."""

    def __init__(self) -> None:
        super().__init__("GEMINI_API_KEY is missing or empty")


class ImageClient(Protocol):
    """The shared helper's method. A stub in tests has the same shape."""

    def generate_images(
        self, prompt: str, images: Any = (), search: bool = False
    ) -> dict[str, Any]: ...


class _UntilMerge:
    """``generate_images`` on top of the client that is already on main."""

    def __init__(self, inner: GeminiClient) -> None:
        self._inner = inner

    def __repr__(self) -> str:
        return repr(self._inner)

    def generate_images(self, prompt, images=(), search=False):
        if search:
            raise RuntimeError("search grounding arrives with the shared helper")
        blobs = tuple(images or ())
        # Model choice stays on the client. This module does not name one.
        reply = self._inner.generate_flash(
            prompt, image_bytes=blobs[0] if blobs else None
        )
        if len(blobs) > 1 and reply.get("http_status") == 200 and reply.get("model_id"):
            reply = self._inner.generate(
                prompt,
                reply["model_id"],
                images=blobs,
                mime_type="image/png",
            )
        return {
            "text": reply.get("text") or "",
            "urls": list(reply.get("urls") or []),
            "http_status": reply.get("http_status"),
            "model_id": reply.get("model_id"),
            "error_body": reply.get("error_body") or "",
        }


def client_from_environ() -> ImageClient:
    """The shared helper when it has landed, otherwise :class:`_UntilMerge`."""

    from sceneagent import gemini_client as gc

    native = getattr(gc, "client_from_environ", None)
    if native is not None:
        return native()
    if "GEMINI_API_KEY" not in os.environ:
        raise MissingGeminiKeyError()
    value = os.environ["GEMINI_API_KEY"]
    if not isinstance(value, str) or not value.strip():
        raise MissingGeminiKeyError()
    return _UntilMerge(GeminiClient(value))
