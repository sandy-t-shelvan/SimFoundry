# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned simfoundry interpreter for the run4 pipeline.

The GPU smoke launcher already refuses to start without this path. The
pipeline checks it again before it samples frames or runs COLMAP, and it
puts that interpreter's bin directory first on ``PATH`` for those
children. ``PATH`` is not searched for another Python, and there is no
fallback to ``python`` or ``python3``.
"""

from __future__ import annotations

import os
from pathlib import Path

# Instance 58301075 has no simfoundry env. Its mamba env ``sa`` imports numpy,
# scipy, trimesh and pillow (probe runs 42690238597 and 42908214934).
SIMFOUNDRY_PYTHON = "/root/.local/share/mamba/envs/sa/bin/python3"


class PinnedInterpreterMissing(RuntimeError):
    """The pinned simfoundry interpreter is missing or not executable."""


def require_simfoundry_python(path: str | None = None) -> str:
    """Return the pinned interpreter.

    ``path`` defaults to :data:`SIMFOUNDRY_PYTHON`. A missing file, or a
    file that is not executable, raises :class:`PinnedInterpreterMissing`
    with the path in the message. The function does not consult ``PATH``.
    """
    candidate = SIMFOUNDRY_PYTHON if path is None else path
    if not os.path.isfile(candidate) or not os.access(candidate, os.X_OK):
        raise PinnedInterpreterMissing(f"pinned interpreter missing: {candidate}")
    return candidate


def stage_environment(interpreter: str | None = None) -> dict[str, str]:
    """Environment for a native stage, with the pinned bin directory first.

    The caller must already have accepted ``interpreter``. A missing file
    still raises, so a stage cannot start on ``/usr/bin/python3``.
    """
    pinned = require_simfoundry_python(interpreter)
    env = os.environ.copy()
    bindir = str(Path(pinned).parent)
    current = env.get("PATH", "")
    env["PATH"] = bindir + (os.pathsep + current if current else "")
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    return env
