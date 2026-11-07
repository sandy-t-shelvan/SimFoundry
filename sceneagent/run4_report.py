# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report-only names of selected frames COLMAP did not register.

The matcher is unchanged. This module does not rerun matching and does
not turn a name list into an acceptance decision. The quality bar stays
in ``reconstruction_gate``.
"""

from __future__ import annotations

import struct
from typing import Any


def registered_image_names(blob: bytes) -> list[str]:
    """Read image names from a COLMAP ``images.bin``.

    The layout is COLMAP's little-endian reconstruction: a uint64 count,
    then per image an int32 id, four doubles of quaternion, three doubles
    of translation, an int32 camera id, a NUL-terminated name, a uint64
    point count, and that many (x, y, point3D id) records. A short buffer,
    a non-UTF-8 name, or trailing bytes fails. A missing file is not
    reported as an empty registration.
    """
    if len(blob) < 8:
        raise ValueError("images.bin missing registered count")
    (count,) = struct.unpack_from("<Q", blob, 0)
    if count > len(blob):
        raise ValueError("images.bin registered count does not fit")
    offset = 8
    names: list[str] = []
    header = struct.calcsize("<i4d3di")
    point = struct.calcsize("<ddQ")
    for _ in range(count):
        if offset + header > len(blob):
            raise ValueError("images.bin truncated")
        offset += header
        end = blob.find(b"\x00", offset)
        if end < 0:
            raise ValueError("images.bin name missing NUL")
        try:
            name = blob[offset:end].decode("utf-8")
        except UnicodeError as exc:
            raise ValueError("images.bin name is not utf-8") from exc
        if name == "":
            raise ValueError("images.bin name is empty")
        offset = end + 1
        if offset + 8 > len(blob):
            raise ValueError("images.bin truncated points")
        (npoints,) = struct.unpack_from("<Q", blob, offset)
        offset += 8
        span = npoints * point
        if offset + span > len(blob):
            raise ValueError("images.bin points truncated")
        offset += span
        names.append(name)
    if offset != len(blob):
        raise ValueError("images.bin trailing bytes")
    return names


def unregistered_frame_report(
    selected_names: list[str],
    registered_by_model: dict[str, list[str] | None],
    name_errors: dict[str, str] | None = None,
    largest_model: str | None = None,
) -> dict[str, Any]:
    """List selected inputs that are absent from each model's name set.

    ``registered_by_model`` values are the names read from that model, or
    ``None`` when the read failed. A missing read stays an error. It is
    not stored as an empty unregistered list. ``report_only`` is always
    true. ``largest_model`` is the gate's chosen model, not a new tie break.
    """
    errors = dict(name_errors or {})
    per: dict[str, list[str] | None] = {}
    selected = list(selected_names)
    for model, names in registered_by_model.items():
        if names is None:
            per[model] = None
            errors.setdefault(model, "registered names missing")
            continue
        per[model] = sorted(set(selected) - set(names))
    if largest_model is None or per.get(largest_model) is None:
        largest_unregistered = None
    else:
        largest_unregistered = per[largest_model]
    count = None if largest_unregistered is None else len(largest_unregistered)
    return {
        "input_count": len(selected),
        "per_model_unregistered": per,
        "per_model_errors": errors,
        "largest_model": largest_model,
        "largest_model_unregistered": largest_unregistered,
        "largest_model_unregistered_count": count,
        "report_only": True,
    }
