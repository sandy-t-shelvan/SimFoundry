# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runner half of soft_press: one grounded lookup, then the trace.

The box checks for PhysGaussian and does not install it. The key stays here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sceneagent.elastic_lookup import trace_soft_objects
from sceneagent.gemini_client import client_from_environ

PARAMS_NAME = "soft_press_params.json"
OBJECT_NAME = "soft_press_object.json"


def load_object(checkout: Path) -> dict[str, Any]:
    """The one soft object the press uses."""
    path = checkout / "sceneagent" / OBJECT_NAME
    return json.loads(path.read_text(encoding="utf-8"))


def write_params(obj: dict[str, Any], client: Any, path: Path) -> list[dict[str, Any]]:
    """Look up E and ν and trace them on ``particle_params``."""
    traced = trace_soft_objects([obj], client)
    path.write_text(json.dumps(traced, indent=2) + "\n", encoding="utf-8")
    return traced


def prepare_for_box(checkout: Path) -> Path | None:
    """Write the trace next to the code so the box can read it. No key, no file."""
    path = checkout / "sceneagent" / OBJECT_NAME
    if not path.is_file():
        return None
    try:
        client = client_from_environ()
    except Exception:
        return None
    out = checkout / "sceneagent" / PARAMS_NAME
    write_params(load_object(checkout), client, out)
    return out
