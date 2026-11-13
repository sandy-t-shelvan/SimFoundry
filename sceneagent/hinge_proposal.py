# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Propose joints for every object that has two or more parts.

One function, one Gemini call per object, with the part summary and the
images. A joint is revolute, prismatic, or fixed. The URDF checker accepts
a movable joint or the reply is retried once and then left unresolved.
The label is ``gemini_proposed``. A one-part object is rigid and is not sent.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from sceneagent.mailbox_hinge import hinge_from_reply, parts_manifest

SOURCE = "gemini_proposed"
MOVABLE = ("revolute", "prismatic")


def propose_articulation(
    objects: Sequence[Mapping[str, Any]],
    frames: Sequence[bytes],
    client: Any,
) -> dict[str, Any]:
    """Propose joints for each object. ``frames`` are image bytes, 2 to 4."""
    _require_frames(frames)
    records: list[dict[str, Any]] = []
    for obj in objects:
        parts = list(obj.get("parts") or [])
        if len(parts) < 2:
            records.append(
                {
                    "object": obj.get("name"),
                    "status": "rigid",
                    "source": SOURCE,
                    "joints": [],
                    "attempts": [],
                }
            )
            continue
        records.append(_one_object(obj, parts, frames, client))
    return {"source": SOURCE, "objects": records}


def prompt_for(
    name: str, parts: Sequence[Mapping[str, Any]], rejected: str | None = None
) -> str:
    """Ask for joints. The images are attached by the client, not described as boxes only."""
    lines = [
        f"Object {name!r}. The attached images are renders or video frames of it.",
        "Part boxes, when known:",
        json.dumps([_box(part) for part in parts]),
        "Reply with one JSON object and no other text.",
        'Keys: "joints", a list. Each joint has "type" (revolute, prismatic, or fixed),',
        '"parent", "child", "axis" (unit vector), "pivot", "lower", "upper", and "rationale".',
        "Limits are radians. Use fixed when a pair should not move.",
        "Do not describe any value as measured.",
    ]
    if rejected:
        lines.append(f"The previous reply was rejected: {rejected}")
        lines.append("Reply with a corrected JSON object.")
    return "\n".join(lines)


def _one_object(obj, parts, frames, client) -> dict[str, Any]:
    name = str(obj.get("name"))
    first = str(
        client.generate_images(prompt_for(name, parts), frames).get("text") or ""
    )
    joints, error = _checked(name, parts, first)
    attempts = [{"model_reply": first, "error": error}]
    if error is not None:
        second = str(
            client.generate_images(prompt_for(name, parts, rejected=error), frames).get(
                "text"
            )
            or ""
        )
        joints, error = _checked(name, parts, second)
        attempts.append({"model_reply": second, "error": error})
    status = "ok" if error is None else "unresolved"
    return {
        "object": name,
        "status": status,
        "source": SOURCE,
        "joints": joints if error is None else [],
        "attempts": attempts,
    }


def _checked(name: str, parts: Sequence[Mapping[str, Any]], text: str):
    parsed = _json_object(text)
    if parsed is None:
        return [], "reply is not a JSON object"
    raw_joints = parsed.get("joints")
    if not isinstance(raw_joints, list):
        return [], "joints is missing"
    known = {str(part["name"]) for part in parts}
    accepted = []
    errors = []
    for index, item in enumerate(raw_joints):
        if not isinstance(item, Mapping):
            errors.append(f"joint {index} is not an object")
            continue
        kind = str(item.get("type") or "").strip().lower()
        if kind in ("none", "fixed"):
            accepted.append(_fixed(item, kind))
            continue
        if kind not in MOVABLE:
            errors.append(
                f"joint {index} type {kind!r} is not revolute, prismatic, or fixed"
            )
            continue
        try:
            reply = _movable_reply(item, kind)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"joint {index}: {exc}")
            continue
        if reply["parent"] not in known or reply["child"] not in known:
            errors.append(f"joint {index} links are not parts")
            continue
        mini = parts_manifest(
            name,
            [
                {
                    "name": reply["parent"],
                    "face_count": 1,
                    "mesh_path": reply["parent"],
                },
                {"name": reply["child"], "face_count": 1, "mesh_path": reply["child"]},
            ],
        )
        try:
            built = hinge_from_reply(mini, reply)
        except ValueError as exc:
            errors.append(f"joint {index}: {exc}")
            continue
        joint = dict(built["joint"])
        joint["rationale"] = reply["rationale"]
        accepted.append(joint)
    if errors:
        return [], "; ".join(errors)
    return accepted, None


def _movable_reply(item: Mapping[str, Any], kind: str) -> dict[str, Any]:
    rationale = item.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("rationale is missing")
    return {
        "name": f"{item.get('parent')}_to_{item.get('child')}",
        "type": kind,
        "parent": str(item["parent"]),
        "child": str(item["child"]),
        "axis": [float(value) for value in item["axis"]],
        "origin": [float(value) for value in item["pivot"]],
        "lower": float(item["lower"]),
        "upper": float(item["upper"]),
        "rationale": rationale.strip(),
    }


def _fixed(item: Mapping[str, Any], kind: str) -> dict[str, Any]:
    return {
        "type": "fixed" if kind == "none" else kind,
        "parent": item.get("parent"),
        "child": item.get("child"),
        "rationale": item.get("rationale") or "",
        "source": SOURCE,
    }


def _box(part: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": str(part.get("name")),
        "bbox_min": part.get("bbox_min"),
        "bbox_max": part.get("bbox_max"),
    }


def _json_object(text: str) -> dict[str, Any] | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _require_frames(frames: Sequence[bytes]) -> None:
    if not 2 <= len(frames) <= 4 or any(not frame for frame in frames):
        raise ValueError("need 2 to 4 image frames")
