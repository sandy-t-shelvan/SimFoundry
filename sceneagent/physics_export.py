# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SF-35: the placed bin as one physics stage (USD for Isaac Lab, MJCF for MuJoCo).

Paper §3.4 and §3.7: the Isaac Lab export uses mass, friction, restitution,
body type and collider through the OpenUSD physics schema, and every other
property is USD metadata. Each placed object is its SAM 3D mesh at its ICP
pose (the similarity transform ``bin_place`` records), with an SF-34 property
record (``physics.build_physics_record`` keys).

Rules that are ours, not the paper's, are named here so a reader can find them:

- **Missing values stay unset.** A friction or restitution the record does not
  have is not written, so PhysX uses its own default; the metadata says
  ``physx default``. Mass and density are written only when known; with
  neither, PhysX computes mass from the collider at its default density.
- **Body types.** ``dynamic rigid`` is a rigid body; ``kinematic`` is a rigid
  body with ``physics:kinematicEnabled``; ``static rigid`` is a collider with
  no rigid body. ``soft deformable``, ``cloth shell`` and ``fluid or granular``
  are not faked as rigid: the mesh is written for display only and marked for
  PhysGaussian (SF-40 / KAN-46, Cut 4).
- **Colliders.** The collision primitive maps to ``physics:approximation``.
  PhysX cannot simulate a triangle mesh on a dynamic body, so a dynamic
  ``triangle mesh`` becomes ``convexDecomposition``, and the substitution is
  recorded. A ``cylinder`` or ``capsule`` becomes ``convexHull`` (no mesh
  approximation fits them exactly).
- **Units.** The stage is in metres and kilograms (``metersPerUnit = 1``,
  ``kilogramsPerUnit = 1``, Z up), so mass is kg and density kg/m^3. The
  caller gives ``metres_per_unit`` for the reconstruction frame and an
  optional ``world_from_scene`` 4x4 that makes it gravity aligned.

The USD is written as text, like ``usd_contract``, so writing needs no
``pxr``; the tests open it with ``pxr`` when ``usd-core`` is installed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.sax.saxutils import quoteattr

import numpy as np

from sceneagent.similarity_icp import apply_similarity

RIGID_BODY_TYPES = ("dynamic rigid", "static rigid", "kinematic")
PHYSGAUSSIAN_BODY_TYPES = ("soft deformable", "cloth shell", "fluid or granular")
DEFAULT_BODY_TYPE = "dynamic rigid"
DEFAULT_COLLISION = "convex hull"
# collision primitive -> UsdPhysics.MeshCollisionAPI physics:approximation
APPROXIMATION = {
    "box": "boundingCube",
    "sphere": "boundingSphere",
    "cylinder": "convexHull",
    "capsule": "convexHull",
    "convex hull": "convexHull",
    "triangle mesh": "none",
}
# Written as physics:* when present; everything else becomes sceneagent:*.
PHYSX_KEYS = (
    "body_type",
    "collision_primitive",
    "mass_kg",
    "density_kg_m3",
    "friction_static",
    "friction_dynamic",
    "restitution",
)
METADATA_SKIP = ("object", "material_scores", "material_sources")
GROUND_HALF_EXTENT_M = 5.0
GROUND_THICKNESS_M = 0.02


@dataclass
class PlacedObject:
    """One placed object: its mesh, its ICP pose and its SF-34 record.

    ``vertices`` are in the mesh's own frame; ``scale``, ``rotation`` and
    ``translation`` are ``bin_place``'s similarity into the reconstruction
    frame (``x' = s R x + t``).
    """

    name: str
    vertices: np.ndarray
    faces: np.ndarray
    scale: float = 1.0
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    properties: Mapping[str, Any] = field(default_factory=dict)


def token(name: str) -> str:
    """A valid USD prim / MJCF name from any object name."""
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(name))
    if not cleaned or cleaned[0].isdigit():
        cleaned = "obj_" + cleaned
    return cleaned


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def resolve_physics(properties: Mapping[str, Any]) -> dict[str, Any]:
    """What the stage authors for one record, and which values are defaults.

    Returns the body type, the simulation route (``physx`` or
    ``physgaussian``), the collider approximation, the numbers to write
    (``None`` = leave to PhysX), and ``defaults``: every field that was filled
    by a rule above rather than read from the record.
    """
    defaults: list[str] = []
    notes: list[str] = []
    body = properties.get("body_type")
    if body is None:
        body = DEFAULT_BODY_TYPE
        defaults.append("body_type")
    if body not in RIGID_BODY_TYPES + PHYSGAUSSIAN_BODY_TYPES:
        raise ValueError(f"unknown body type {body!r}")
    route = "physgaussian" if body in PHYSGAUSSIAN_BODY_TYPES else "physx"
    collision = properties.get("collision_primitive")
    if collision is None:
        collision = DEFAULT_COLLISION
        defaults.append("collision_primitive")
    approximation = None
    if route == "physx":
        if collision not in APPROXIMATION:
            raise ValueError(f"collider {collision!r} cannot be a PhysX rigid collider")
        approximation = APPROXIMATION[collision]
        if approximation == "none" and body == "dynamic rigid":
            approximation = "convexDecomposition"
            notes.append(
                "triangle mesh on a dynamic body is not simulated by PhysX; "
                "written as convexDecomposition"
            )
    values = {key: _number(properties.get(key)) for key in PHYSX_KEYS[2:]}
    for key in ("friction_static", "friction_dynamic", "restitution"):
        if values[key] is None:
            defaults.append(key)
    for key in ("mass_kg", "density_kg_m3"):
        amount = values[key]
        if amount is not None and amount <= 0:
            raise ValueError(f"{key} must be positive")
    if values["mass_kg"] is None and values["density_kg_m3"] is None:
        defaults.append("mass_kg")
    return {
        "body_type": body,
        "route": route,
        "collision_primitive": collision,
        "approximation": approximation,
        "values": values,
        "defaults": defaults,
        "notes": notes,
    }


def stage_points(
    obj: PlacedObject,
    metres_per_unit: float = 1.0,
    world_from_scene: np.ndarray | None = None,
) -> np.ndarray:
    """The object's vertices in the stage frame, in metres."""
    points = apply_similarity(obj.vertices, obj.scale, obj.rotation, obj.translation)
    if world_from_scene is not None:
        transform = np.asarray(world_from_scene, dtype=np.float64)
        points = points @ transform[:3, :3].T + transform[:3, 3]
    return np.asarray(points * float(metres_per_unit), dtype=np.float64)


def _usd_string(value: Any) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _metadata_lines(
    properties: Mapping[str, Any], resolved: dict[str, Any]
) -> list[str]:
    lines = [
        f"        custom string sceneagent:simulation = {_usd_string(resolved['route'])}",
        f"        custom string sceneagent:body_type = {_usd_string(resolved['body_type'])}",
        "        custom string sceneagent:collision_primitive = "
        + _usd_string(resolved["collision_primitive"]),
        "        custom string sceneagent:defaults = "
        + _usd_string(", ".join(resolved["defaults"]) or "none"),
    ]
    if resolved["notes"]:
        lines.append(
            "        custom string sceneagent:notes = "
            + _usd_string("; ".join(resolved["notes"]))
        )
    for key in sorted(properties):
        if key in PHYSX_KEYS or key in METADATA_SKIP:
            continue
        value = properties[key]
        if value is None:
            continue
        name = token(key)
        number = _number(value) if not isinstance(value, (list, dict, str)) else None
        if number is not None:
            lines.append(f"        custom double sceneagent:{name} = {number!r}")
        else:
            lines.append(
                f"        custom string sceneagent:{name} = {_usd_string(value)}"
            )
    return lines


def _mesh_lines(points: np.ndarray, faces: np.ndarray, indent: str) -> list[str]:
    pts = ", ".join(f"({x!r}, {y!r}, {z!r})" for x, y, z in points.tolist())
    counts = ", ".join("3" for _ in range(len(faces)))
    indices = ", ".join(str(int(i)) for face in faces for i in face)
    return [
        f"{indent}point3f[] points = [{pts}]",
        f"{indent}int[] faceVertexCounts = [{counts}]",
        f"{indent}int[] faceVertexIndices = [{indices}]",
        f'{indent}uniform token subdivisionScheme = "none"',
    ]


def _object_block(
    obj: PlacedObject, points: np.ndarray, resolved: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """The object's Xform and, when needed, its physics material."""
    name = token(obj.name)
    centre = points.mean(axis=0)
    local = points - centre
    values = resolved["values"]
    physx = resolved["route"] == "physx"
    rigid = physx and resolved["body_type"] in ("dynamic rigid", "kinematic")
    schemas = []
    if rigid:
        schemas += ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    elif physx and (
        values["mass_kg"] is not None or values["density_kg_m3"] is not None
    ):
        schemas.append("PhysicsMassAPI")
    head = f'    def Xform "{name}"'
    if schemas:
        head += (
            " (\n        prepend apiSchemas = ["
            + ", ".join(f'"{s}"' for s in schemas)
            + "]\n    )"
        )
    lines = [head, "    {"]
    cx, cy, cz = (float(v) for v in centre)
    lines.append(f"        double3 xformOp:translate = ({cx!r}, {cy!r}, {cz!r})")
    lines.append('        uniform token[] xformOpOrder = ["xformOp:translate"]')
    if rigid:
        lines.append("        bool physics:rigidBodyEnabled = true")
        if resolved["body_type"] == "kinematic":
            lines.append("        bool physics:kinematicEnabled = true")
    if physx and values["mass_kg"] is not None:
        lines.append(f"        float physics:mass = {values['mass_kg']!r}")
    if physx and values["density_kg_m3"] is not None:
        lines.append(f"        float physics:density = {values['density_kg_m3']!r}")
    lines += _metadata_lines(obj.properties, resolved)
    material_lines: list[str] = []
    friction = [
        values[k] for k in ("friction_static", "friction_dynamic", "restitution")
    ]
    has_material = physx and any(v is not None for v in friction)
    mesh_schemas = []
    if physx:
        mesh_schemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
        if has_material:
            mesh_schemas.append("MaterialBindingAPI")
    mesh_head = '        def Mesh "geometry"'
    if mesh_schemas:
        mesh_head += (
            " (\n            prepend apiSchemas = ["
            + ", ".join(f'"{s}"' for s in mesh_schemas)
            + "]\n        )"
        )
    lines += [mesh_head, "        {"]
    lines += _mesh_lines(local, obj.faces, "            ")
    if physx:
        lines.append("            bool physics:collisionEnabled = true")
        lines.append(
            f'            uniform token physics:approximation = "{resolved["approximation"]}"'
        )
        if has_material:
            lines.append(
                f"            rel material:binding:physics = </World/Looks/{name}_physics>"
            )
    else:
        # Display only: PhysGaussian (SF-40) simulates it, not this stage.
        lines.append('            uniform token purpose = "default"')
    lines += ["        }", "    }"]
    if has_material:
        material_lines = [
            f'        def Material "{name}_physics" (',
            '            prepend apiSchemas = ["PhysicsMaterialAPI"]',
            "        )",
            "        {",
        ]
        for key, attr in (
            ("friction_static", "physics:staticFriction"),
            ("friction_dynamic", "physics:dynamicFriction"),
            ("restitution", "physics:restitution"),
        ):
            if values[key] is not None:
                material_lines.append(f"            float {attr} = {values[key]!r}")
        material_lines.append("        }")
    return lines, material_lines


def write_physics_stage(
    objects: Sequence[PlacedObject],
    path: Path | str,
    *,
    metres_per_unit: float = 1.0,
    world_from_scene: np.ndarray | None = None,
    ground_z_m: float | None = 0.0,
) -> dict[str, Any]:
    """Write the placed bin as a USD physics stage; return what was written.

    ``ground_z_m`` places a static ground collider (a thin box whose top is at
    that height, in metres, stage frame); ``None`` leaves it out. The returned
    summary lists each object's route, approximation and defaults, for the
    ledger. The stage is a new file; no source asset is touched.
    """
    destination = Path(path)
    body: list[str] = []
    looks: list[str] = []
    summary: dict[str, Any] = {"stage": str(destination), "objects": []}
    seen: set[str] = set()
    for obj in objects:
        name = token(obj.name)
        if name in seen:
            raise ValueError(f"two objects map to the prim name {name!r}")
        seen.add(name)
        resolved = resolve_physics(obj.properties)
        points = stage_points(obj, metres_per_unit, world_from_scene)
        block, material = _object_block(obj, points, resolved)
        body += block
        looks += material
        summary["objects"].append(
            {
                "object": obj.name,
                "prim": f"/World/{name}",
                "route": resolved["route"],
                "body_type": resolved["body_type"],
                "approximation": resolved["approximation"],
                "defaults": resolved["defaults"],
                "notes": resolved["notes"],
            }
        )
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "World"',
        "    metersPerUnit = 1",
        "    kilogramsPerUnit = 1",
        '    upAxis = "Z"',
        '    doc = "SceneAgent SF-35: placed bin, paper section 3.4 physics export"',
        ")",
        "",
        'def Xform "World"',
        "{",
        '    def PhysicsScene "physicsScene"',
        "    {",
        "        vector3f physics:gravityDirection = (0, 0, -1)",
        "        float physics:gravityMagnitude = 9.81",
        "    }",
    ]
    if ground_z_m is not None:
        half = GROUND_HALF_EXTENT_M
        thick = GROUND_THICKNESS_M
        lines += [
            '    def Cube "ground" (',
            '        prepend apiSchemas = ["PhysicsCollisionAPI"]',
            "    )",
            "    {",
            "        double size = 1",
            f"        double3 xformOp:translate = (0, 0, {float(ground_z_m) - thick / 2!r})",
            f"        float3 xformOp:scale = ({2 * half!r}, {2 * half!r}, {thick!r})",
            '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]',
            "        bool physics:collisionEnabled = true",
            "    }",
        ]
        summary["ground_z_m"] = float(ground_z_m)
    if looks:
        lines += ['    def Scope "Looks"', "    {", *looks, "    }"]
    lines += body
    lines += ["}", ""]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    return summary


def write_physics_mjcf(
    objects: Sequence[PlacedObject],
    path: Path | str,
    *,
    metres_per_unit: float = 1.0,
    world_from_scene: np.ndarray | None = None,
    ground_z_m: float | None = 0.0,
) -> dict[str, Any]:
    """The same scene for MuJoCo (dc3); return what was written.

    MuJoCo collides meshes by their convex hull, has one sliding friction
    (the dynamic one is used, else the static one, else MuJoCo's default)
    and no restitution coefficient, so restitution is recorded only. A
    kinematic body becomes a static one here; a PhysGaussian object is left
    out and listed.
    """
    destination = Path(path)
    assets: list[str] = []
    world: list[str] = []
    summary: dict[str, Any] = {"model": str(destination), "objects": [], "left_out": []}
    if ground_z_m is not None:
        world.append(
            f'    <geom name="ground" type="plane" size="{GROUND_HALF_EXTENT_M} '
            f'{GROUND_HALF_EXTENT_M} 0.1" pos="0 0 {float(ground_z_m)!r}"/>'
        )
    for obj in objects:
        name = token(obj.name)
        resolved = resolve_physics(obj.properties)
        if resolved["route"] != "physx":
            summary["left_out"].append({"object": obj.name, "route": resolved["route"]})
            continue
        points = stage_points(obj, metres_per_unit, world_from_scene)
        centre = points.mean(axis=0)
        local = points - centre
        vertex = " ".join(repr(float(v)) for v in local.ravel())
        face = " ".join(str(int(i)) for i in np.asarray(obj.faces).ravel())
        assets.append(f'    <mesh name="{name}_mesh" vertex="{vertex}" face="{face}"/>')
        values = resolved["values"]
        attrs = [f'name="{name}_geom"', 'type="mesh"', f'mesh="{name}_mesh"']
        if values["mass_kg"] is not None:
            attrs.append(f'mass="{values["mass_kg"]!r}"')
        elif values["density_kg_m3"] is not None:
            attrs.append(f'density="{values["density_kg_m3"]!r}"')
        slide = values["friction_dynamic"]
        if slide is None:
            slide = values["friction_static"]
        if slide is not None:
            attrs.append(f'friction="{slide!r} 0.005 0.0001"')
        pos = " ".join(repr(float(v)) for v in centre)
        geom = "<geom " + " ".join(attrs) + "/>"
        comment = (
            "<!-- "
            + quoteattr(
                f"body_type={resolved['body_type']} defaults={','.join(resolved['defaults'])}"
            )[1:-1].replace("--", "- -")
            + " -->"
        )
        if resolved["body_type"] == "dynamic rigid":
            world += [
                f'    <body name="{name}" pos="{pos}">',
                "      <freejoint/>",
                f"      {geom}",
                f"      {comment}",
                "    </body>",
            ]
        else:
            world += [
                f'    <body name="{name}" pos="{pos}">',
                f"      {geom}",
                f"      {comment}",
                "    </body>",
            ]
        summary["objects"].append(
            {"object": obj.name, "body_type": resolved["body_type"]}
        )
    xml = [
        '<mujoco model="sceneagent_bin">',
        "  <!-- SceneAgent SF-35: placed bin, paper section 3.4 physics (rigid only) -->",
        '  <option gravity="0 0 -9.81" timestep="0.002"/>',
        "  <asset>",
        *assets,
        "  </asset>",
        "  <worldbody>",
        *world,
        "  </worldbody>",
        "</mujoco>",
        "",
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(xml), encoding="utf-8")
    return summary
