# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mailbox lid hinge, one object, from the export of run 43384974056.

The accepted mesh is ``object_1`` at
``/workspace/sf4-smoke/mailbox/sam3d/object_1.glb``. PartSAM, CubePart and
Articulate Anything are not started here. A caller passes a worker result
or a tool reply. The live Articulate Anything entry cannot take those parts.

Read from ``articulate.py`` (``ArticulationPipeline.run_pipeline``) and
``conf/config.yaml``: the first stage is PartNet ``mesh_retrieval``. The
config modalities are text, video and partnet. None of them is a list of
caller part meshes, so a live run stops. No joint is invented and no PartNet
asset is downloaded (``section34.articulate_download_allowed`` stays unused).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from xml.sax.saxutils import escape

from sceneagent.section34 import inspect_parts

MAILBOX_RUN = "43384974056"
MAILBOX_OBJECT = "object_1"
MAILBOX_MESH = "/workspace/sf4-smoke/mailbox/sam3d/object_1.glb"
# First stage of articulate.py. There is no argument for our part meshes.
ARTICULATE_TAKES_PART_MESHES = False
STOP_REASON = "tool cannot take part meshes"
EXTENT_SCALE = 1.5
SWEEP_SECONDS = 4.0
SWEEP_HZ = 60
SEPARATION_LIMIT_M = 0.01
SPEED_LIMIT_M_S = 5.0


def parts_manifest(parent: str, parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """One object's parts. ``graph_cut`` is the worker's flag, not a guess."""
    rows = []
    for part in parts:
        name = str(part["name"])
        if not name:
            raise ValueError("a part needs a name")
        rows.append(
            {
                "name": name,
                "face_count": int(part["face_count"]),
                "mesh_path": str(part["mesh_path"]),
                "parent_object": parent,
                "graph_cut": bool(part.get("graph_cut", False)),
            }
        )
    return {"object": parent, "source_run": MAILBOX_RUN, "parts": rows}


def review_parts(parts: Sequence[str], reply: str | None) -> dict[str, Any]:
    """Reuse ``inspect_parts``. A reply that is neither yes nor no is a failure."""
    inspected = inspect_parts(list(parts), reply)
    failed = bool(reply) and inspected["completion_needed"] is None
    return {
        **inspected,
        "status": "failure" if failed else "ok",
        "image_path": None,
    }


def apply_completion(
    review: Mapping[str, Any],
    manifest: Mapping[str, Any],
    worker: Callable[[], Mapping[str, Any]] | None,
    extent: Sequence[float],
) -> dict[str, Any]:
    """Append a CubePart mesh only when the review said completion is needed.

    A part larger than 1.5 times the object, or a part with no faces, is
    rejected and the manifest is left unchanged.
    """
    parts = [dict(part) for part in manifest["parts"]]
    if review.get("completion_needed") is not True:
        skipped = (
            "completion not needed"
            if review.get("completion_needed") is False
            else "completion reply failed"
        )
        return {"called": False, "skipped": skipped, "parts": parts}
    if worker is None:
        raise ValueError("completion was requested and no worker was given")
    made = dict(worker())
    size = [float(value) for value in made["size"]]
    limit = [EXTENT_SCALE * float(value) for value in extent]
    if any(side > cap for side, cap in zip(size, limit, strict=True)):
        return {
            "called": True,
            "rejected": "part outside 1.5x bounding box",
            "parts": parts,
        }
    if int(made.get("face_count", 0)) < 1:
        return {"called": True, "rejected": "part added no faces", "parts": parts}
    made["source"] = "cubepart"
    made["parent_object"] = manifest["object"]
    parts.append(made)
    return {"called": True, "rejected": None, "parts": parts}


def live_articulate() -> dict[str, str]:
    """Articulate Anything is skipped. It cannot take these part meshes.

    The Mailbox joint is a proposal in ``hinge_proposal``, not this call.
    """
    if ARTICULATE_TAKES_PART_MESHES:
        raise RuntimeError("part-mesh input appeared; this stop is out of date")
    return {"status": "stopped", "reason": STOP_REASON}


def hinge_from_reply(
    manifest: Mapping[str, Any], reply: Mapping[str, Any]
) -> dict[str, Any]:
    """One revolute joint whose links are parts. Raises on a bad reply."""
    if reply.get("status") == "stopped":
        return {"status": "stopped", "reason": str(reply.get("reason") or STOP_REASON)}
    joint = {
        "name": str(reply["name"]),
        "type": str(reply["type"]),
        "parent": str(reply["parent"]),
        "child": str(reply["child"]),
        "axis": [float(value) for value in reply["axis"]],
        "origin": [float(value) for value in reply.get("origin", (0.0, 0.0, 0.0))],
        "lower": float(reply["lower"]),
        "upper": float(reply["upper"]),
    }
    problems(manifest, [joint])
    return {"status": "ok", "joint": joint, "input": _input_record(manifest)}


def problems(manifest: Mapping[str, Any], joints: Sequence[Mapping[str, Any]]) -> None:
    """Fail when links and parts disagree, an axis is not a unit, or limits invert."""
    names = {str(part["name"]) for part in manifest["parts"]}
    linked: set[str] = set()
    for joint in joints:
        parent, child = str(joint["parent"]), str(joint["child"])
        for link in (parent, child):
            if link not in names:
                raise ValueError(f"link {link} has no part")
            linked.add(link)
        axis = [float(value) for value in joint["axis"]]
        length = math.sqrt(sum(value * value for value in axis))
        if not math.isclose(length, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"axis {axis} is not unit length")
        if str(joint["type"]) in ("revolute", "prismatic"):
            lower, upper = float(joint["lower"]), float(joint["upper"])
            if lower >= upper:
                raise ValueError(f"lower {lower} is not below upper {upper}")
    missing = names - linked
    if missing:
        raise ValueError(f"part {min(missing)} has no link")


def write_urdf(manifest: Mapping[str, Any], joint: Mapping[str, Any]) -> str:
    """One link per part, named as the part, and the one joint."""
    problems(manifest, [joint])
    lines = [f'<robot name="{escape(str(manifest["object"]))}">']
    for part in manifest["parts"]:
        lines.append(f'  <link name="{escape(str(part["name"]))}"/>')
    axis = " ".join(f"{float(value):g}" for value in joint["axis"])
    origin = " ".join(f"{float(value):g}" for value in joint["origin"])
    lines += [
        f'  <joint name="{escape(str(joint["name"]))}" type="{escape(str(joint["type"]))}">',
        f'    <parent link="{escape(str(joint["parent"]))}"/>',
        f'    <child link="{escape(str(joint["child"]))}"/>',
        f'    <axis xyz="{axis}"/>',
        f'    <origin xyz="{origin}" rpy="0 0 0"/>',
        f'    <limit lower="{float(joint["lower"]):g}" upper="{float(joint["upper"]):g}"/>',
        "  </joint>",
        "</robot>",
        "",
    ]
    return "\n".join(lines)


def hinge_scene(object_name: str, joint: Mapping[str, Any]) -> dict[str, Any]:
    """Saved-scene contract: ``joint_pos``, zero ``joint_vel``, limits by name.

    Limits stay in radians. USD degrees are a different file.
    """
    return {
        "state": {
            "registry": {
                "object_registry": {
                    object_name: {
                        "joint_pos": [0.0],
                        "joint_vel": [0.0],
                    }
                }
            }
        },
        "objects_info": {
            "init_info": {
                object_name: {
                    "args": {
                        "joint_limits": {
                            str(joint["name"]): {
                                "lower": float(joint["lower"]),
                                "upper": float(joint["upper"]),
                            }
                        }
                    }
                }
            }
        },
    }


def hinge_usda(object_name: str, joint: Mapping[str, Any]) -> str:
    """Text USDA. Revolute limits are degrees, matching the light editor."""
    lower = math.degrees(float(joint["lower"]))
    upper = math.degrees(float(joint["upper"]))
    parent, child = str(joint["parent"]), str(joint["child"])
    token = ("X", "Y", "Z")[_axis_index(joint)]
    joint_name = str(joint["name"])
    return "\n".join(
        [
            "#usda 1.0",
            "(",
            "    metersPerUnit = 1",
            '    upAxis = "Z"',
            ")",
            f'def Xform "{object_name}"',
            "{",
            f'    def Xform "{parent}" {{ }}',
            f'    def Xform "{child}" {{ }}',
            f'    def PhysicsRevoluteJoint "{joint_name}"',
            "    {",
            f'        uniform token physics:axis = "{token}"',
            *_local_pos(joint),
            f"        float physics:lowerLimit = {lower:g}",
            f"        float physics:upperLimit = {upper:g}",
            f"        rel physics:body0 = </{object_name}/{parent}>",
            f"        rel physics:body1 = </{object_name}/{child}>",
            "    }",
            "}",
            "",
        ]
    )


def proposal_documents(proposal: Mapping[str, Any]) -> dict[str, Any]:
    """USD text and scene-state joints for every movable proposal.

    Revolute limits in the USDA are degrees. The scene state stays in radians,
    with ``joint_vel`` zero and ``joint_limits`` keyed by joint name. Fixed
    joints are omitted. An object with no movable joint stays rigid.
    """
    blocks = []
    scenes = []
    for obj in proposal.get("objects") or []:
        name = str(obj.get("object") or obj.get("name"))
        movable = [
            joint
            for joint in obj.get("joints") or []
            if joint.get("type") in ("revolute", "prismatic")
        ]
        if not movable:
            continue
        for joint in movable:
            blocks.append(_joint_usda(name, joint))
        scenes.append(_scene_for_joints(name, movable))
    return {"usda": "\n".join(blocks), "scenes": scenes}


def _joint_usda(object_name: str, joint: Mapping[str, Any]) -> str:
    kind = str(joint["type"])
    if kind == "revolute":
        return hinge_usda(object_name, joint)
    parent, child = str(joint["parent"]), str(joint["child"])
    token = ("X", "Y", "Z")[_axis_index(joint)]
    return "\n".join(
        [
            "#usda 1.0",
            f'def Xform "{object_name}"',
            "{",
            f'    def Xform "{parent}" {{ }}',
            f'    def Xform "{child}" {{ }}',
            f'    def PhysicsPrismaticJoint "{joint["name"]}"',
            "    {",
            f'        uniform token physics:axis = "{token}"',
            *_local_pos(joint),
            f"        float physics:lowerLimit = {float(joint['lower']):g}",
            f"        float physics:upperLimit = {float(joint['upper']):g}",
            f"        rel physics:body0 = </{object_name}/{parent}>",
            f"        rel physics:body1 = </{object_name}/{child}>",
            "    }",
            "}",
            "",
        ]
    )


def _local_pos(joint: Mapping[str, Any]) -> list[str]:
    """Joint position in both body frames. Empty xforms share object space."""
    origin = joint.get("origin")
    if not isinstance(origin, (list, tuple)) or len(origin) != 3:
        return []
    point = ", ".join(f"{float(value):g}" for value in origin)
    return [
        f"        point3f physics:localPos0 = ({point})",
        f"        point3f physics:localPos1 = ({point})",
    ]


def _scene_for_joints(
    object_name: str, joints: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    limits = {
        str(joint["name"]): {
            "lower": float(joint["lower"]),
            "upper": float(joint["upper"]),
        }
        for joint in joints
    }
    count = len(joints)
    return {
        "object": object_name,
        "joint_pos": [0.0] * count,
        "joint_vel": [0.0] * count,
        "joint_limits": limits,
    }


def sweep_targets(
    lower: float, upper: float, seconds: float = SWEEP_SECONDS, hz: int = SWEEP_HZ
) -> list[dict[str, float]]:
    """Position targets from lower to upper and back, at ``hz`` for ``seconds``."""
    steps = int(seconds * hz)
    half = steps // 2
    rows = []
    for index in range(steps + 1):
        if index <= half:
            position = lower + (upper - lower) * (index / half)
        else:
            position = upper + (lower - upper) * ((index - half) / (steps - half))
        rows.append({"t": index / hz, "joint_pos": position})
    return rows


def motion_failed(separation_m: float, speed_m_s: float) -> bool:
    """Parts flew apart: more than 1 cm from the anchor, or faster than 5 m/s."""
    return separation_m > SEPARATION_LIMIT_M or speed_m_s > SPEED_LIMIT_M_S


def record_verdict(
    judge: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    limits: Mapping[str, float],
) -> dict[str, Any]:
    """One verdict. A reject is retried once with the corrected limits."""
    first = dict(judge(limits))
    attempts = [first]
    used = dict(limits)
    if first.get("verdict") == "reject" and isinstance(first.get("limits"), Mapping):
        used = {key: float(first["limits"][key]) for key in ("lower", "upper")}
        attempts.append(dict(judge(used)))
    return {"attempts": attempts, "limits": used}


def _axis_index(joint: Mapping[str, Any]) -> int:
    axis = [abs(float(value)) for value in joint["axis"]]
    return axis.index(max(axis))


def _input_record(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "part_meshes": [str(part["mesh_path"]) for part in manifest["parts"]],
        "views": 4,
        "text": f"a mailbox whose lid hinges on the {manifest['object']} body",
        "tool_takes_part_meshes": ARTICULATE_TAKES_PART_MESHES,
    }
