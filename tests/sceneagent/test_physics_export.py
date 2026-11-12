# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SF-35: the placed bin as a USD physics stage and an MJCF model.

The property mapping is tested in pure Python. The USD checks open the stage
with ``pxr`` and the MuJoCo check loads and steps the model; each skips when
its package (usd-core, mujoco) is not installed, as on the CPU CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from sceneagent import physics_export as pe
from sceneagent.physics import build_physics_record

CUBE_V = np.array(
    [[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)]
)
CUBE_F = np.array(
    [
        [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
        [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
    ]
)  # fmt: skip


def _cube(name, z=0.05, **props):
    """A 10 cm cube resting on z = 0 (scale 0.1, centre at ``z``)."""
    return pe.PlacedObject(
        name=name,
        vertices=CUBE_V,
        faces=CUBE_F,
        scale=0.1,
        translation=np.array([0.0, 0.0, z]),
        properties=props,
    )


def test_missing_values_are_left_to_physx_and_named():
    resolved = pe.resolve_physics({})
    assert resolved["body_type"] == "dynamic rigid"
    assert resolved["approximation"] == "convexHull"
    assert set(resolved["defaults"]) == {
        "body_type",
        "collision_primitive",
        "friction_static",
        "friction_dynamic",
        "restitution",
        "mass_kg",
    }
    assert all(v is None for v in resolved["values"].values())


def test_a_dynamic_triangle_mesh_becomes_a_convex_decomposition():
    resolved = pe.resolve_physics(
        {"body_type": "dynamic rigid", "collision_primitive": "triangle mesh"}
    )
    assert resolved["approximation"] == "convexDecomposition"
    assert "convexDecomposition" in resolved["notes"][0]
    static = pe.resolve_physics(
        {"body_type": "static rigid", "collision_primitive": "triangle mesh"}
    )
    assert static["approximation"] == "none" and not static["notes"]


def test_soft_bodies_are_routed_to_physgaussian_not_faked_rigid():
    record = build_physics_record(
        "sponge",
        [{"class": "rubber and foam", "weight": 1.0}],
        numerical={"deformability": 0.8},
    )
    resolved = pe.resolve_physics(record)
    assert resolved["route"] == "physgaussian"
    assert resolved["approximation"] is None


def test_bad_records_are_refused():
    for bad in (
        {"body_type": "hovering"},
        {"collision_primitive": "tetrahedral mesh"},
        {"mass_kg": -1.0},
    ):
        with pytest.raises(ValueError):
            pe.resolve_physics(bad)


def test_points_follow_the_icp_pose_and_the_unit_scale():
    obj = _cube("a", z=2.0)
    points = pe.stage_points(obj, metres_per_unit=0.5)
    assert np.allclose(points.mean(axis=0), [0.0, 0.0, 1.0])
    assert np.isclose(np.ptp(points[:, 0]), 0.05)
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    assert np.allclose(pe.stage_points(obj, 0.5, flip).mean(axis=0), [0, 0, -1.0])


def test_the_summary_lists_routes_and_defaults(tmp_path):
    objects = [
        _cube(
            "mug", mass_kg=0.3, friction_dynamic=0.5, material_class="glass and ceramic"
        ),
        _cube("bin", body_type="static rigid", collision_primitive="triangle mesh"),
        _cube("sponge", body_type="soft deformable"),
    ]
    summary = pe.write_physics_stage(objects, tmp_path / "bin.usda")
    rows = {row["object"]: row for row in summary["objects"]}
    assert rows["mug"]["route"] == "physx" and "mass_kg" not in rows["mug"]["defaults"]
    assert rows["bin"]["approximation"] == "none"
    assert rows["sponge"]["route"] == "physgaussian"
    with pytest.raises(ValueError, match="prim name"):
        pe.write_physics_stage([_cube("a b"), _cube("a_b")], tmp_path / "x.usda")


def test_the_stage_carries_physx_attributes_and_metadata(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    objects = [
        _cube(
            "mug",
            mass_kg=0.3,
            friction_static=0.6,
            friction_dynamic=0.5,
            restitution=0.1,
            material_class="glass and ceramic",
            rigidity=0.9,
        ),
        _cube("bin", body_type="static rigid", collision_primitive="triangle mesh"),
        _cube(
            "tray", body_type="kinematic", collision_primitive="box", density_kg_m3=500
        ),
        _cube("sponge", body_type="soft deformable"),
    ]
    path = tmp_path / "bin.usda"
    pe.write_physics_stage(objects, path)
    stage = Usd.Stage.Open(str(path))
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert UsdPhysics.GetStageKilogramsPerUnit(stage) == 1.0

    mug = stage.GetPrimAtPath("/World/mug")
    assert mug.HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdPhysics.MassAPI(mug).GetMassAttr().Get() == pytest.approx(0.3)
    geom = stage.GetPrimAtPath("/World/mug/geometry")
    assert geom.HasAPI(UsdPhysics.CollisionAPI)
    assert (
        UsdPhysics.MeshCollisionAPI(geom).GetApproximationAttr().Get() == "convexHull"
    )
    material, _ = UsdShade.MaterialBindingAPI(geom).ComputeBoundMaterial("physics")
    phys = UsdPhysics.MaterialAPI(material.GetPrim())
    assert phys.GetStaticFrictionAttr().Get() == pytest.approx(0.6)
    assert phys.GetDynamicFrictionAttr().Get() == pytest.approx(0.5)
    assert phys.GetRestitutionAttr().Get() == pytest.approx(0.1)
    assert mug.GetAttribute("sceneagent:material_class").Get() == "glass and ceramic"
    assert mug.GetAttribute("sceneagent:rigidity").Get() == pytest.approx(0.9)
    assert (
        mug.GetAttribute("sceneagent:defaults").Get()
        == "body_type, collision_primitive"
    )

    bin_prim = stage.GetPrimAtPath("/World/bin")
    assert not bin_prim.HasAPI(UsdPhysics.RigidBodyAPI)
    bin_geom = stage.GetPrimAtPath("/World/bin/geometry")
    assert UsdPhysics.MeshCollisionAPI(bin_geom).GetApproximationAttr().Get() == "none"
    assert not UsdShade.MaterialBindingAPI(bin_geom).GetDirectBindingRel("physics")

    tray = stage.GetPrimAtPath("/World/tray")
    assert UsdPhysics.RigidBodyAPI(tray).GetKinematicEnabledAttr().Get() is True
    assert UsdPhysics.MassAPI(tray).GetDensityAttr().Get() == pytest.approx(500)

    sponge = stage.GetPrimAtPath("/World/sponge")
    assert not sponge.HasAPI(UsdPhysics.RigidBodyAPI)
    assert not stage.GetPrimAtPath("/World/sponge/geometry").HasAPI(
        UsdPhysics.CollisionAPI
    )
    assert sponge.GetAttribute("sceneagent:simulation").Get() == "physgaussian"

    ground = stage.GetPrimAtPath("/World/ground")
    assert ground.HasAPI(UsdPhysics.CollisionAPI)
    top = UsdGeom.Cube(ground).ComputeWorldBound(0, "default").ComputeAlignedRange()
    assert top.GetMax()[2] == pytest.approx(0.0, abs=1e-6)
    centre = UsdGeom.Xformable(mug).ComputeLocalToWorldTransform(0).ExtractTranslation()
    assert tuple(centre) == pytest.approx((0.0, 0.0, 0.05))


def test_the_mjcf_loads_and_a_resting_cube_stays_at_rest(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    objects = [
        _cube("mug", mass_kg=0.3, friction_dynamic=0.8),
        _cube("bin", z=0.05, body_type="static rigid"),
        _cube("sponge", body_type="soft deformable"),
    ]
    objects[1].translation = np.array([0.5, 0.0, 0.05])
    path = tmp_path / "bin.xml"
    summary = pe.write_physics_mjcf(objects, path)
    assert summary["left_out"] == [{"object": "sponge", "route": "physgaussian"}]
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    body = model.body("mug").id
    for _ in range(1000):  # 2 s at 2 ms
        mujoco.mj_step(model, data)
    assert np.linalg.norm(data.cvel[body][3:]) < 0.01
    assert data.xpos[body][2] == pytest.approx(0.05, abs=0.005)
