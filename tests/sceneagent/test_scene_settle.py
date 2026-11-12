# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KAN-41: the run export (support plane, gravity) and the Isaac Lab settle judge."""

from __future__ import annotations

import json
import types

import numpy as np

from sceneagent import physics_export, settle_run


def test_the_settle_judge_needs_every_object_at_rest_and_above_ground():
    rest = {"a": {"speed_m_s": 0.001, "centre_z_m": 0.05}}
    assert settle_run.judge(rest)[0] == "accept"
    moving = rest | {"b": {"speed_m_s": 0.2, "centre_z_m": 0.05}}
    decision, reason = settle_run.judge(moving)
    assert decision == "reject" and "b 0.200 m/s" in reason
    sunk = rest | {"c": {"speed_m_s": 0.0, "centre_z_m": -0.3}}
    assert settle_run.judge(sunk)[0] == "reject"
    assert settle_run.judge({})[0] == "reject"


def test_rotation_onto_z_sends_any_normal_up():
    rng = np.random.default_rng(1)
    for normal in [
        *rng.normal(size=(5, 3)),
        np.array([0, 0, -1.0]),
        np.array([0, 0, 1.0]),
    ]:
        n = normal / np.linalg.norm(normal)
        rotation = physics_export.rotation_onto_z(n)
        assert np.allclose(rotation @ n, [0, 0, 1], atol=1e-9)
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)


def _tilted_scene(rng):
    """A table tilted 20 degrees, a larger wall beside it, two objects on the table."""
    up = np.array([0.0, np.sin(0.35), np.cos(0.35)])
    side = np.cross(up, [1.0, 0, 0])
    side /= np.linalg.norm(side)
    ex = np.cross(up, side)
    table = rng.uniform(-1, 1, (3000, 2)) @ np.stack([ex, side]) + 0.4 * up
    wall = np.column_stack(
        [np.full(6000, -1.5), rng.uniform(-2, 2, 6000), rng.uniform(-2, 2, 6000)]
    )
    objects = [
        0.4 * up
        + 0.1 * ex
        + rng.uniform(0, 0.2, (300, 1)) * up
        + rng.normal(0, 0.03, (300, 3)) * 0.3,
        0.4 * up - 0.3 * side + rng.uniform(0, 0.1, (300, 1)) * up,
    ]
    return np.concatenate([table, wall]), objects, up


def test_the_support_is_the_plane_the_objects_rest_on_not_the_largest():
    rng = np.random.default_rng(0)
    background, objects, up = _tilted_scene(rng)
    support = physics_export.support_frame(background, objects, band=0.01)
    chosen = support["candidates"][support["chosen"]]
    assert abs(abs(np.dot(chosen["normal"], up)) - 1) < 1e-3
    frame = support["world_from_scene"]
    lows = [(o @ frame[:3, :3].T + frame[:3, 3])[:, 2].min() for o in objects]
    assert all(-0.02 < low < 0.02 for low in lows)


def test_export_run_writes_a_gravity_aligned_stage(tmp_path, monkeypatch):
    import trimesh

    rng = np.random.default_rng(2)
    background, objects, _ = _tilted_scene(rng)
    positions = np.concatenate([background, objects[0]])
    ids = np.arange(len(background), len(positions))
    (tmp_path / "association").mkdir()
    np.savez(tmp_path / "association" / "objects.npz", object_0=ids)
    (tmp_path / "sam3d").mkdir()
    glb = tmp_path / "sam3d" / "object_0.glb"
    # The mesh is the object's own points, so its similarity is the identity.
    trimesh.convex.convex_hull(objects[0]).export(glb)
    (tmp_path / "bin_score.json").write_text(
        json.dumps(
            {
                "icp": {
                    "objects": [
                        {
                            "object": 0,
                            "placed": True,
                            "glb": str(glb),
                            "similarity": {
                                "scale": 1.0,
                                "rotation": np.eye(3).tolist(),
                                "translation": [0, 0, 0],
                            },
                        }
                    ]
                }
            }
        )
    )
    (tmp_path / "association_evidence.json").write_text(
        json.dumps({"splat_unit_m": 0.5})
    )
    (tmp_path / "splat_evidence.json").write_text(json.dumps({"ply": "splat.ply"}))
    import sceneagent.bin_association as association

    monkeypatch.setattr(association, "read_ply_xyz", lambda path: positions)
    assert physics_export.export_run(tmp_path) == 0
    for name in ("scene.usda", "scene_settle.usda", "scene.xml"):
        assert (tmp_path / "export" / name).is_file()
    settle = (tmp_path / "export" / "scene_settle.usda").read_text()
    assert "PhysicsScene" not in settle and 'def Cube "ground"' not in settle
    evidence = json.loads((tmp_path / "export_evidence.json").read_text())
    # Resting on the table, which is now z = 0, in metres (0.5 m per unit).
    assert abs(evidence["lowest_point_z_m"]["object_0"]) < 0.01
    assert evidence["physics_records"] == 0
    entry = json.loads((tmp_path / "ledger.json").read_text())["entries"][-1]
    assert entry["item"] == physics_export.EXPORT_ITEM and entry["decision"] == "accept"


def test_run_settle_reads_the_isaac_result_into_the_ledger(tmp_path):
    (tmp_path / "export_evidence.json").write_text(
        json.dumps({"objects": [{"prim": "/World/object_0", "route": "physx"}]})
    )

    def runner(argv, **kwargs):
        assert argv[argv.index("--prims") + 1] == "object_0"
        assert "PYTHONPATH" not in kwargs["env"]
        (tmp_path / settle_run.RESULT).write_text(
            json.dumps(
                {"decision": "accept", "reason": "1 objects at rest", "frame": "f.png"}
            )
        )
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    assert settle_run.run_settle(tmp_path, runner=runner) == 0
    entry = json.loads((tmp_path / "ledger.json").read_text())["entries"][-1]
    assert entry["item"] == settle_run.ITEM and entry["decision"] == "accept"


def test_a_settle_that_writes_nothing_rejects_with_its_tail(tmp_path):
    (tmp_path / "export_evidence.json").write_text(json.dumps({"objects": []}))

    def runner(argv, **kwargs):
        return types.SimpleNamespace(
            returncode=139, stdout="", stderr="Segmentation fault"
        )

    assert settle_run.run_settle(tmp_path, runner=runner) == 1
    entry = json.loads((tmp_path / "ledger.json").read_text())["entries"][-1]
    assert "exited 139" in entry["reason"] and "Segmentation fault" in entry["reason"]
