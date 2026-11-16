# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Open a SceneAgent physics export (SF-35) in the light editor (SF-45).

``sceneagent.physics_export.write_physics_stage`` writes the placed scene as
one USD stage: every object an ``Xform`` under ``/World`` with its mesh inline,
PhysX attributes (rigid body, mass, density, collider approximation, a bound
physics material for friction and restitution) and every other SF-34 property
as ``sceneagent:*`` metadata. The editor works on a scene-state JSON with one
USD per object, so this module turns the one into the other:

- each object becomes a **new** USD file under ``<scene>/assets/`` -- the
  export is read, never rewritten (AGENTS.md rule 1) -- whose root prim is its
  rigid body, so ``scene_io.read_usd_physics`` reads the link name a friction
  material must be keyed by, and its authored ``physics:mass``;
- friction and restitution go where the saved-scene contract puts them,
  ``init_info.args.link_physics_materials`` keyed by that link, which
  OmniGibson applies at load;
- body types map to the constructor's own flags: ``static rigid`` is
  ``fixed_base``, ``kinematic`` is ``kinematic_only``, and a body SF-34 routes
  to PhysGaussian is ``visual_only`` -- shown, never faked as rigid;
- everything the panels cannot edit (material, body type, simulator route,
  collider, which values are PhysX defaults, rigidity, ...) is kept per object
  under the top-level ``physics_records`` key, which the editor shows read-only
  and which survives a settle (``scene_io.merge_settled_scene`` keeps every
  top-level key the simulator does not own).

The simulator-owned blocks (``versions``, ``metadata``, the scene's own
``init_info``) cannot be made up here. With ``--template`` they come from a
scene OmniGibson already loads, the way ``compose.compose_scene`` starts a new
scene, and the template's editable props are dropped. Without one the document
opens in the editor but is marked not restorable until composed onto a
template.

Objects keep the export's stage-frame poses (metres, Z up). An object with
revolute or prismatic joints under it (SF-39) keeps them: its asset is a copy of
the object's own prim tree -- links, meshes and joints -- rather than one merged
mesh, because the joints panel reads joint names, axes and limits from each
object's own USD (``scene_io.read_usd_joints``) and drops a ``joint_limits``
entry whose joint the asset does not name. Such an asset's root is not a rigid
body (its links are), so no friction is keyed by the object's name for it.

Usage::

    python export_import.py --stage bin_physics.usda --out-dir scenes/bin_export
    python export_import.py --stage bin_physics.usda --out-dir scenes/bin_export \\
        --template scenes/kitchen/kitchen_scene_state_latest.json
"""

import argparse
import copy
import json
import math
import re
from pathlib import Path

from asset_library import md5, object_spec
from scene_io import (
    PHYSICS_RECORDS_KEY,
    SceneEditError,
    add_objects,
    atomic_write_text,
    editable_object_names,
    remove_objects,
)

#: What the exporter writes as PhysX attributes; everything else is metadata.
_PHYSX_FIELDS = ("mass_kg", "density_kg_m3", "friction_static",
                 "friction_dynamic", "restitution")

#: SF-34 body types (sceneagent.physics_export.RIGID_BODY_TYPES and
#: PHYSGAUSSIAN_BODY_TYPES).
_PHYSGAUSSIAN_BODY_TYPES = ("soft deformable", "cloth shell", "fluid or granular")

#: Recorded on a document written without a template.
NOT_RESTORABLE = (
    "written without a template: versions, metadata and the scene init_info are "
    "empty, so OmniGibson cannot restore this document until it is composed onto "
    "a working scene (export_import.py --template)"
)

_NAME = re.compile(r"[^A-Za-z0-9_]")


def prim_token(name):
    """A valid USD prim name (and editor object name) from any name."""
    cleaned = _NAME.sub("_", str(name))
    if not cleaned or cleaned[0].isdigit():
        cleaned = "obj_" + cleaned
    return cleaned


def _number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_export_stage(stage_path):
    """Read every object of an SF-35 physics stage. Needs ``pxr``.

    Returns:
        dict: ``objects`` (one record per object prim, in stage order; see
        `object_asset_text` for the keys) and ``ground_z_m`` (the top of the
        exporter's ground box, or None when it wrote none).
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise SceneEditError(f"could not open {stage_path}")
    world = stage.GetDefaultPrim() or stage.GetPrimAtPath("/World")
    if not world:
        raise SceneEditError(f"{stage_path} has no /World")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    bboxes = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

    objects, ground_z = [], None
    for prim in world.GetChildren():
        if prim.GetName() == "ground" and prim.IsA(UsdGeom.Gprim):
            ground_z = float(bboxes.ComputeWorldBound(prim).ComputeAlignedRange().GetMax()[2])
            continue
        if not prim.IsA(UsdGeom.Xform):
            continue
        meshes = [child for child in prim.GetChildren() if child.IsA(UsdGeom.Mesh)]
        if not meshes:
            continue
        mesh = UsdGeom.Mesh(meshes[0])
        origin = cache.GetLocalToWorldTransform(prim)
        to_object = cache.GetLocalToWorldTransform(meshes[0]) * origin.GetInverse()
        points = [list(to_object.Transform(Gf.Vec3d(*p))) for p in mesh.GetPointsAttr().Get() or []]

        def value(target, name):
            attribute = target.GetAttribute(name)
            return attribute.Get() if attribute and attribute.HasAuthoredValue() else None

        friction = {"friction_static": None, "friction_dynamic": None, "restitution": None}
        material, _ = UsdShade.MaterialBindingAPI(meshes[0]).ComputeBoundMaterial("physics")
        if material:
            for key, attr in (("friction_static", "physics:staticFriction"),
                              ("friction_dynamic", "physics:dynamicFriction"),
                              ("restitution", "physics:restitution")):
                friction[key] = _number(value(material.GetPrim(), attr))

        metadata = {}
        for attribute in prim.GetAttributes():
            name = attribute.GetName()
            if name.startswith("sceneagent:") and attribute.HasAuthoredValue():
                raw = attribute.Get()
                metadata[name.split(":", 1)[1]] = raw if isinstance(raw, str) else _number(raw)

        objects.append({
            "name": prim.GetName(),
            "position": [round(float(v), 9) for v in origin.ExtractTranslation()],
            "points": [[float(c) for c in p] for p in points],
            "face_counts": [int(c) for c in mesh.GetFaceVertexCountsAttr().Get() or []],
            "face_indices": [int(i) for i in mesh.GetFaceVertexIndicesAttr().Get() or []],
            "rigid": prim.HasAPI(UsdPhysics.RigidBodyAPI),
            "kinematic": bool(value(prim, "physics:kinematicEnabled")),
            "collider": meshes[0].HasAPI(UsdPhysics.CollisionAPI),
            "approximation": value(meshes[0], "physics:approximation"),
            "mass_kg": _number(value(prim, "physics:mass")),
            "density_kg_m3": _number(value(prim, "physics:density")),
            **friction,
            "metadata": metadata,
            "joints": _joint_records(prim),
        })
        if objects[-1]["joints"]:
            objects[-1]["articulated_usda"] = _articulated_asset_text(stage, prim)
    return {"objects": objects, "ground_z_m": ground_z}


def _joint_records(prim):
    """Named DOFs under one object, or [] when the export is rigid.

    Revolute limits are degrees in USD and radians in the scene state, the
    same conversion ``scene_io.read_usd_joints`` uses. Absent limits stay
    absent: a joint with no range is still a ``joint_pos`` entry.
    """
    found = []
    for child in prim.GetStage().Traverse():
        path = str(child.GetPath())
        if not path.startswith(str(prim.GetPath()) + "/"):
            continue
        kind = child.GetTypeName()
        if kind not in ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint"):
            continue
        def authored(name, target=child):
            attribute = target.GetAttribute(name)
            return attribute.Get() if attribute and attribute.HasAuthoredValue() else None
        lower, upper = _number(authored("physics:lowerLimit")), _number(authored("physics:upperLimit"))
        if kind == "PhysicsRevoluteJoint":
            lower = None if lower is None else math.radians(lower)
            upper = None if upper is None else math.radians(upper)
        found.append({
            "name": child.GetName(),
            "position": 0.0,
            "lower": lower,
            "upper": upper,
        })
    return found


def _articulated_asset_text(stage, prim):
    """The object's prim tree as its own USD: links, meshes and joints.

    The copy is made from the flattened stage so references and payloads come
    along. ``Sdf.CopySpec`` remaps joint body targets that point inside the
    copied tree. The root's own transform is dropped: the scene's
    ``root_link`` pose places the object, and keeping both would apply it twice.
    """
    from pxr import Sdf, Usd, UsdGeom

    name = prim_token(prim.GetName())
    flat = stage.Flatten()
    layer = Sdf.Layer.CreateAnonymous(".usda")
    root_path = Sdf.Path(f"/{name}")
    if not Sdf.CopySpec(flat, prim.GetPath(), layer, root_path):
        raise SceneEditError(f"could not copy {prim.GetPath()} out of the export")
    root = layer.GetPrimAtPath(root_path)
    for prop in list(root.properties):
        if prop.name.startswith("xformOp"):
            root.RemoveProperty(prop)
    layer.defaultPrim = name
    copy = Usd.Stage.Open(layer)
    UsdGeom.SetStageUpAxis(copy, UsdGeom.GetStageUpAxis(stage))
    UsdGeom.SetStageMetersPerUnit(copy, UsdGeom.GetStageMetersPerUnit(stage))
    return layer.ExportToString()


def body_type(record):
    """The SF-34 body type of a record: its metadata, else its PhysX schemas."""
    stated = (record.get("metadata") or {}).get("body_type")
    if isinstance(stated, str) and stated:
        return stated
    if record.get("rigid"):
        return "kinematic" if record.get("kinematic") else "dynamic rigid"
    return "static rigid" if record.get("collider") else "soft deformable"


def object_asset_text(record):
    """One object's own USD, as text (no ``pxr`` needed to write it).

    The root prim is the object's rigid body, named after the object, so the
    editor reads it as the link a friction material is keyed by. Mass and
    density are authored here, so they are the asset's own numbers and a mass
    edit in the panel is an override of them. Friction is not: it belongs in
    the scene's ``link_physics_materials``. A PhysGaussian body is written as
    display-only geometry with no collider.

    Args:
        record (dict): From `read_export_stage`: ``name``, ``points`` (object
            frame, metres), ``face_counts``, ``face_indices``, ``approximation``,
            ``mass_kg``, ``density_kg_m3`` and ``metadata``.

    Returns:
        str: The USDA text.
    """
    name = prim_token(record["name"])
    simulated = body_type(record) not in _PHYSGAUSSIAN_BODY_TYPES
    mass = _number(record.get("mass_kg")) if simulated else None
    density = _number(record.get("density_kg_m3")) if simulated else None
    schemas = ['"PhysicsRigidBodyAPI"']
    if mass is not None or density is not None:
        schemas.append('"PhysicsMassAPI"')
    lines = [
        "#usda 1.0",
        "(",
        f'    defaultPrim = "{name}"',
        "    metersPerUnit = 1",
        "    kilogramsPerUnit = 1",
        '    upAxis = "Z"',
        '    doc = "SceneAgent SF-45: one object of an SF-35 physics export"',
        ")",
        "",
        f'def Xform "{name}" (',
        f"    prepend apiSchemas = [{', '.join(schemas)}]",
        ")",
        "{",
        "    bool physics:rigidBodyEnabled = true",
    ]
    if mass is not None:
        lines.append(f"    float physics:mass = {mass!r}")
    if density is not None:
        lines.append(f"    float physics:density = {density!r}")
    mesh_head = '    def Mesh "geometry"'
    if simulated:
        mesh_head += ' (\n        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]\n    )'
    lines += [mesh_head, "    {"]
    points = ", ".join(f"({x!r}, {y!r}, {z!r})" for x, y, z in record["points"])
    lines += [
        f"        point3f[] points = [{points}]",
        f"        int[] faceVertexCounts = [{', '.join(str(int(c)) for c in record['face_counts'])}]",
        f"        int[] faceVertexIndices = [{', '.join(str(int(i)) for i in record['face_indices'])}]",
        '        uniform token subdivisionScheme = "none"',
    ]
    if simulated:
        approximation = record.get("approximation") or "convexHull"
        lines += ["        bool physics:collisionEnabled = true",
                  f'        uniform token physics:approximation = "{approximation}"']
    lines += ["    }", "}", ""]
    return "\n".join(lines)


def physics_record(record):
    """What the editor shows read-only about one object: the SF-34 record.

    Everything here either has no panel (material, body type, simulator route,
    collider) or is a PhysX default the export left unset (``defaults``).
    """
    metadata = dict(record.get("metadata") or {})
    defaults = metadata.pop("defaults", None)
    out = {
        "source": "sceneagent.physics_export (SF-35)",
        "body_type": body_type(record),
        "simulation": metadata.pop("simulation", None)
        or ("physgaussian" if body_type(record) in _PHYSGAUSSIAN_BODY_TYPES else "physx"),
        "collision_primitive": metadata.pop("collision_primitive", None),
        "approximation": record.get("approximation"),
        "defaults": [d.strip() for d in defaults.split(",") if d.strip() and d.strip() != "none"]
        if isinstance(defaults, str) else [],
        "restitution": _number(record.get("restitution")),
        "density_kg_m3": _number(record.get("density_kg_m3")),
    }
    metadata.pop("body_type", None)
    # The rest of the SF-34 record (material, rigidity, deformability, ...).
    out["properties"] = {k: v for k, v in sorted(metadata.items()) if v is not None}
    return out


def _apply_joints(spec, record):
    """Optional ``joint_pos`` and ``joint_limits``. A rigid object omits both.

    ``joint_vel`` is zeroed with the pose, or a saved velocity would move the
    joint when physics starts. Limits are keyed by joint name.
    """
    joints = record.get("joints") or []
    if not joints:
        return
    spec["registry"]["joint_pos"] = [
        0.0 if _number(joint.get("position")) is None else _number(joint.get("position"))
        for joint in joints
    ]
    spec["registry"]["joint_vel"] = [0.0] * len(joints)
    limits = {}
    for joint in joints:
        lower, upper = _number(joint.get("lower")), _number(joint.get("upper"))
        name = joint.get("name")
        if name and lower is not None and upper is not None:
            limits[str(name)] = {"lower": lower, "upper": upper}
    if limits:
        spec["init_info"]["args"]["joint_limits"] = limits


def _constructor_flags(record):
    """``init_info.args`` flags for the body type; OmniGibson constructor kwargs."""
    kind = body_type(record)
    if kind == "static rigid":
        return {"fixed_base": True}
    if kind == "kinematic":
        return {"kinematic_only": True}
    if kind in _PHYSGAUSSIAN_BODY_TYPES:
        # Not faked as rigid: PhysGaussian (SF-40) simulates it, not PhysX.
        return {"visual_only": True}
    return {}


def _materials(record, link):
    """``link_physics_materials`` for one object, or None when nothing is known."""
    entry = {}
    for field, key in (("friction_static", "static_friction"),
                       ("friction_dynamic", "dynamic_friction"),
                       ("restitution", "restitution")):
        number = _number(record.get(field))
        if number is not None:
            entry[key] = number
    if not entry or body_type(record) in _PHYSGAUSSIAN_BODY_TYPES:
        return None
    return {link: entry}


def object_specs(records, scene_json_path):
    """Write each object's USD next to the scene and build its add spec.

    Args:
        records (list[dict]): From `read_export_stage`.
        scene_json_path (Path): Where the scene JSON will be written; assets go
            to ``assets/`` beside it and are referenced relatively.

    Returns:
        tuple[list[dict], dict]: ``asset_library.object_spec`` specs, with the
        physics args added, and the read-only records keyed by object name.
    """
    scene_dir = Path(scene_json_path).resolve().parent
    assets = scene_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    specs, records_by_name, seen = [], {}, set()
    for record in records:
        name = prim_token(record["name"])
        if name in seen:
            raise SceneEditError(f"two exported objects map to the name {name!r}")
        seen.add(name)
        usd = assets / f"{name}.usda"
        articulated = record.get("articulated_usda")
        usd.write_text(articulated or object_asset_text(record), encoding="utf-8")
        category = str((record.get("metadata") or {}).get("material") or "object")
        spec = object_spec(
            name, usd.relative_to(scene_dir).as_posix(), usd, prim_token(category),
            position=record["position"], orientation=[0.0, 0.0, 0.0, 1.0],
            scale=[1.0, 1.0, 1.0],
        )
        args = spec["init_info"]["args"]
        args.update(_constructor_flags(record))
        materials = None if articulated else _materials(record, name)
        if materials:
            args["link_physics_materials"] = materials
        _apply_joints(spec, record)
        spec["init_info"]["args"]["expected_file_hash"] = md5(usd)
        specs.append(spec)
        records_by_name[name] = physics_record(record)
    return specs, records_by_name


def _empty_document():
    return {
        "versions": {},
        "metadata": {},
        "state": {"registry": {"object_registry": {}}},
        "init_info": {},
        "objects_info": {"init_info": {}},
    }


def build_document(records, scene_json_path, *, template=None, ground_z_m=None):
    """The scene-state document for an export, with its assets written.

    Args:
        records (list[dict]): From `read_export_stage`.
        scene_json_path (Path): Where the document will be written.
        template (dict or None): A scene document OmniGibson loads, already
            parsed, with its path under ``template["__path__"]`` removed by the
            caller. Its simulator-owned blocks, room and robot are kept; its
            editable props are dropped.
        template_path (Path or None): Where the template was read from.
        ground_z_m (float or None): The export's ground height; written as
            ``ground_plane_info`` when the template has none.

    Returns:
        dict: The document.
    """
    if template is None:
        scene = _empty_document()
        scene["sceneagent_export"] = {"restorable": False, "reason": NOT_RESTORABLE}
    else:
        scene = copy.deepcopy(template["document"])
        props = editable_object_names(scene, template["path"])
        remove_objects(scene, sorted(props), removable_names=props)
        args = scene.get("init_info", {}).get("args")
        if isinstance(args, dict) and args.get("scene_file") is not None:
            # The template's snapshot describes the template (compose.py does the same).
            args["scene_file"] = None
        scene["sceneagent_export"] = {"restorable": True,
                                      "template": str(template["path"])}
    specs, physics = object_specs(records, scene_json_path)
    add_objects(scene, specs)
    scene[PHYSICS_RECORDS_KEY] = {**scene.get(PHYSICS_RECORDS_KEY, {}), **physics}
    if ground_z_m is not None and not scene.get("ground_plane_info"):
        scene["ground_plane_info"] = {"position": [0.0, 0.0, float(ground_z_m)],
                                      "orientation": [0.0, 0.0, 0.0, 1.0],
                                      "visible": False}
    return scene


def import_export_stage(stage_path, out_dir, scene_name="export", *, template_json=None):
    """Write an editor scene directory for an SF-35 stage; return a summary."""
    stage = read_export_stage(stage_path)
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    scene_json = out / f"{scene_name}_scene_state_exported.json"
    template = None
    if template_json is not None:
        path = Path(template_json).resolve()
        template = {"document": json.loads(path.read_text(encoding="utf-8")), "path": path}
    scene = build_document(stage["objects"], scene_json, template=template,
                           ground_z_m=stage["ground_z_m"])
    atomic_write_text(scene_json, json.dumps(scene, indent=2) + "\n")
    return {"scene": str(scene_json), "objects": sorted(scene[PHYSICS_RECORDS_KEY]),
            "restorable": scene["sceneagent_export"]["restorable"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stage", required=True, type=Path,
                        help="SF-35 stage from sceneagent.physics_export")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--scene-name", default="export")
    parser.add_argument("--template", type=Path, default=None,
                        help="a scene OmniGibson loads; supplies versions, room and robot")
    args = parser.parse_args()
    summary = import_export_stage(args.stage, args.out_dir, args.scene_name,
                                  template_json=args.template)
    print(json.dumps(summary, indent=2))
    print(f"\nOpen it with:\n  python server.py --scene {summary['scene']}")
    if not summary["restorable"]:
        print("Note: " + NOT_RESTORABLE)


if __name__ == "__main__":
    main()
