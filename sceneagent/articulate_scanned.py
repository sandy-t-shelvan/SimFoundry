# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KAN-42 to KAN-45 on a scanned object: PartSAM parts + a proposed joint -> two links.

    python -m sceneagent.articulate_scanned --parts PARTS_DIR \
        --proposal articulate_proposal.json --run VIDEO_RUN --object 1 --out LINKS

The CPU half of articulating a reconstructed object (the Isaac Lab half is
``articulate_scanned_drive.py``):

* the joint is the first movable one in a ``hinge_proposal`` record;
* the child link is the joint's child part plus every unnamed part whose
  bounding box lies inside the child's (``INSIDE_SHARE``); the rest stay on
  the parent. On the Mailbox video PartSAM split the lid into a rim and a top
  surface and the joint named only the rim; the top surface went with it;
* SAM 3D meshes are Y-up and unit-normalised, so the links are turned Z-up
  and scaled to metres by the placed mesh's extent from ``bin_score.json``;
* ``LINKS/body.obj``, ``LINKS/lid.obj`` and ``LINKS/joint.json`` (axis, pivot in
  metres, limits, and which part went where and why).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

# SAM 3D meshes are Y-up; the scene is Z-up: (x, y, z) -> (x, -z, y).
Y_UP_TO_Z_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
# An unnamed part moves with the child when this share of its bounding box
# volume lies inside the child's bounding box.
INSIDE_SHARE = 0.9
MOVABLE = ("revolute", "prismatic")


def box_inside_share(
    inner_lo: Sequence[float],
    inner_hi: Sequence[float],
    outer_lo: Sequence[float],
    outer_hi: Sequence[float],
) -> float:
    """The share of the inner box's volume that lies inside the outer box."""
    lo = np.maximum(np.asarray(inner_lo, float), np.asarray(outer_lo, float))
    hi = np.minimum(np.asarray(inner_hi, float), np.asarray(outer_hi, float))
    overlap = float(np.prod(np.clip(hi - lo, 0.0, None)))
    size = np.asarray(inner_hi, float) - np.asarray(inner_lo, float)
    volume = float(np.prod(np.clip(size, 1e-12, None)))
    return overlap / volume


def movable_joint(proposal: dict[str, Any]) -> dict[str, Any]:
    """The first revolute or prismatic joint of the first object, or raise."""
    for obj in proposal.get("objects") or []:
        for joint in obj.get("joints") or []:
            if joint.get("type") in MOVABLE:
                return dict(joint)
    raise ValueError("the proposal has no movable joint; the object stays rigid")


def group_parts(
    parts: Sequence[dict[str, Any]], joint: dict[str, Any]
) -> tuple[list[str], list[str], dict[str, str]]:
    """(parent link parts, child link parts, why each part went where)."""
    by_name = {str(p["name"]): p for p in parts}
    for end in ("parent", "child"):
        if joint.get(end) not in by_name:
            raise ValueError(f"joint {end} {joint.get(end)!r} is not a part")
    child = by_name[joint["child"]]
    lid, body = [str(joint["child"])], []
    why = {str(joint["child"]): "joint child"}
    for part in parts:
        name = str(part["name"])
        if name == joint["child"]:
            continue
        if name == joint["parent"]:
            body.append(name)
            why[name] = "joint parent"
            continue
        share = box_inside_share(
            part["bbox_min"], part["bbox_max"], child["bbox_min"], child["bbox_max"]
        )
        if share >= INSIDE_SHARE:
            lid.append(name)
            why[name] = f"inside the child's box ({share:.2f})"
        else:
            body.append(name)
            why[name] = f"stays with the parent ({share:.2f} inside the child's box)"
    return body, lid, why


def metric_frame(
    local_bounds: np.ndarray, placed_extent_m: Sequence[float]
) -> tuple[float, np.ndarray]:
    """Scale (m per unit) and the 4x4 transform: Z-up, metres, base on z=0, centred."""
    scale = max(placed_extent_m) / float(np.max(local_bounds[1] - local_bounds[0]))
    rotate = Y_UP_TO_Z_UP * scale
    corners = (
        np.array(
            [
                [x, y, z]
                for x in local_bounds[:, 0]
                for y in local_bounds[:, 1]
                for z in local_bounds[:, 2]
            ]
        )
        @ rotate.T
    )
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    transform = np.eye(4)
    transform[:3, :3] = rotate
    transform[:3, 3] = -np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    return scale, transform


def link_joints(
    body: Sequence[str], lid: Sequence[str], joint: dict[str, Any]
) -> list[dict[str, Any]]:
    """The proposed joint, plus a fixed joint tying every other part to its link's root.

    One URDF link per PartSAM part (KAN-44 dc2): the body's parts are fixed to
    the joint's parent part and the lid's to its child part.
    """
    joints = [
        dict(joint, name=joint.get("name") or f"{joint['parent']}_to_{joint['child']}")
    ]
    for root, members in ((joint["parent"], body), (joint["child"], lid)):
        for name in members:
            if name != root:
                joints.append(
                    {
                        "name": f"{root}_fixed_{name}",
                        "type": "fixed",
                        "parent": root,
                        "child": name,
                        "axis": [1.0, 0.0, 0.0],
                        "origin": [0.0, 0.0, 0.0],
                    }
                )
    return joints


def write_urdf(
    name: str,
    parts: Sequence[dict[str, Any]],
    joints: Sequence[dict[str, Any]],
    scale: float,
    mesh_dir: str = "parts",
) -> str:
    """URDF with one link per part (its mesh, metres) and every joint.

    Checked by ``mailbox_hinge.problems``: every part has a link, axes are
    unit length and movable limits are ordered. The meshes keep SAM 3D's
    Y-up frame; ``scale`` (metres per mesh unit) is applied to the meshes and
    the joint origins.
    """
    from xml.sax.saxutils import escape

    from sceneagent.mailbox_hinge import problems

    manifest = {"object": name, "parts": [{"name": str(p["name"])} for p in parts]}
    problems(manifest, joints)
    s = f"{scale:.9g}"
    lines = [
        f'<robot name="{escape(name)}">',
        "  <!-- meshes are SAM 3D's Y-up frame, in metres -->",
    ]
    for part in parts:
        mesh = f"{mesh_dir}/{Path(str(part['mesh'])).name}"
        geometry = f'<geometry><mesh filename="{escape(mesh)}" scale="{s} {s} {s}"/></geometry>'
        lines += [
            f'  <link name="{escape(str(part["name"]))}">',
            f"    <visual>{geometry}</visual>",
            f"    <collision>{geometry}</collision>",
            "  </link>",
        ]
    for joint in joints:
        origin = " ".join(f"{float(v) * scale:.6g}" for v in joint["origin"])
        axis = " ".join(f"{float(v):g}" for v in joint["axis"])
        lines += [
            f'  <joint name="{escape(str(joint["name"]))}" type="{escape(str(joint["type"]))}">',
            f'    <parent link="{escape(str(joint["parent"]))}"/>',
            f'    <child link="{escape(str(joint["child"]))}"/>',
            f'    <origin xyz="{origin}" rpy="0 0 0"/>',
        ]
        if joint["type"] in MOVABLE:
            lower, upper = float(joint["lower"]), float(joint["upper"])
            lines += [
                f'    <axis xyz="{axis}"/>',
                f'    <limit lower="{lower:g}" upper="{upper:g}" effort="1" velocity="1"/>',
            ]
        lines.append("  </joint>")
    lines += ["</robot>", ""]
    return "\n".join(lines)


def prepare(
    parts_dir: Path,
    proposal_path: Path,
    run: Path,
    object_number: int,
    out: Path,
    name: str | None = None,
) -> dict[str, Any]:
    import trimesh

    parts = json.loads((parts_dir / "parts.json").read_text(encoding="utf-8"))["parts"]
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    joint = movable_joint(proposal)
    body, lid, why = group_parts(parts, joint)
    meshes = {str(p["name"]): trimesh.load(p["mesh"], force="mesh") for p in parts}
    links = {
        "body": trimesh.util.concatenate([meshes[n] for n in body]),
        "lid": trimesh.util.concatenate([meshes[n] for n in lid]),
    }
    placed = json.loads((run / "bin_score.json").read_text(encoding="utf-8"))["icp"][
        "objects"
    ]
    record = next(o for o in placed if int(o["object"]) == object_number)
    local = trimesh.util.concatenate(list(links.values()))
    scale, transform = metric_frame(np.asarray(local.bounds), record["mesh_extent_m"])
    out.mkdir(parents=True, exist_ok=True)
    for link_name, mesh in links.items():
        mesh.apply_transform(transform)
        mesh.export(out / f"{link_name}.obj")
    pivot = transform[:3, :3] @ np.asarray(joint["origin"], float) + transform[:3, 3]
    axis = Y_UP_TO_Z_UP @ np.asarray(joint["axis"], float)
    axis = axis / np.linalg.norm(axis)
    result = {
        "type": joint["type"],
        "axis": [float(v) for v in axis],
        "pivot_m": [float(v) for v in pivot],
        "lower": float(joint["lower"]),
        "upper": float(joint["upper"]),
        "source": proposal.get("source"),
        "rationale": joint.get("rationale"),
        "scale_m_per_unit": scale,
        "placement": {
            "run": str(run),
            "object": object_number,
            "mesh_extent_m": record["mesh_extent_m"],
            "gate": record.get("gate"),
        },
        "body_parts": body,
        "lid_parts": lid,
        "grouping": why,
        "extents_m": {k: [float(v) for v in m.extents] for k, m in links.items()},
    }
    name = (name or f"object_{object_number}").replace(" ", "_")
    urdf_path = out / f"{name}.urdf"
    joints = link_joints(body, lid, joint)
    urdf_path.write_text(
        write_urdf(name, parts, joints, scale, mesh_dir=str(parts_dir / "parts")),
        encoding="utf-8",
    )
    result["urdf"] = str(urdf_path)
    (out / "joint.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--parts", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--object", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", default=None, help="the object's name, for the URDF")
    args = parser.parse_args(argv)
    result = prepare(
        args.parts, args.proposal, args.run, args.object, args.out, args.name
    )
    print(
        json.dumps(
            {k: result[k] for k in ("axis", "pivot_m", "lid_parts", "extents_m")}
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
