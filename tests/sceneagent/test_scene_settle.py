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
                            "gate": "accept",
                            "glb": str(glb),
                            "similarity": {
                                "scale": 1.0,
                                "rotation": np.eye(3).tolist(),
                                "translation": [0, 0, 0],
                            },
                        },
                        {
                            "object": 1,
                            "placed": True,
                            "gate": "reject",
                            "gate_failed": ["coverage"],
                        },
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
    box = evidence["object_boxes"]["object_0"]
    assert abs(box["bottom_z_m"]) < 0.01 and min(box["size_m"]) > 0
    assert evidence["overlapping_boxes"] == []
    # A gate reject never reaches physics, but is listed with what failed.
    assert evidence["excluded_gate_rejects"] == {"object_1": ["coverage"]}
    assert "object_1" not in evidence["object_boxes"]
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


def test_overlapping_boxes_are_named_and_apart_ones_are_not():
    cube = np.array(
        [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=float
    )

    def body(name, shift):
        return physics_export.PlacedObject(
            name=name,
            vertices=cube + shift,
            faces=np.zeros((0, 3), dtype=int),
            scale=1.0,
            rotation=np.eye(3),
            translation=np.zeros(3),
            properties={},
        )

    placed = [body("a", 0.0), body("b", 0.5), body("c", 5.0)]
    found = physics_export.diagnose(placed, 0.1, np.eye(4), {"a": "accept"})
    assert found["overlapping_boxes"] == [["a", "b"]]
    assert found["object_boxes"]["a"] == {
        "size_m": [0.1, 0.1, 0.1],
        "bottom_z_m": 0.0,
        "gate": "accept",
    }


def test_a_settle_reject_carries_the_evidence_a_manifest_can_show(tmp_path):
    (tmp_path / "export_evidence.json").write_text(
        json.dumps(
            {
                "objects": [{"prim": "/World/object_1", "route": "physx"}],
                "association_objects": 3,
                "association_dropped": 4,
                "excluded_gate_rejects": {"object_2": ["p90"]},
                "object_boxes": {
                    "object_1": {
                        "size_m": [0.2, 0.1, 0.3],
                        "bottom_z_m": -0.004,
                        "gate": "reject",
                    }
                },
                "overlapping_boxes": [["object_1", "object_2"]],
            }
        )
    )
    (tmp_path / "association_evidence.json").write_text(
        json.dumps(
            {
                "object_list": [
                    {
                        "object": 1,
                        "views": 4,
                        "masks": [{"phrase": "mailbox"}] * 3 + [{"phrase": "cube"}],
                    }
                ]
            }
        )
    )
    (tmp_path / "bin_score.json").write_text(
        json.dumps({"icp": {"objects": [{"object": 1, "gate_failed": ["p90"]}]}})
    )
    moving = {
        "speed_m_s": 0.022,
        "centre_z_m": 0.1,
        "speed_1s_m_s": 0.05,
        "angular_speed_rad_s": None,
    }

    def runner(argv, **kwargs):
        decision, reason = settle_run.judge({"object_1": moving})
        (tmp_path / settle_run.RESULT).write_text(
            json.dumps(
                {
                    "decision": decision,
                    "reason": reason,
                    "objects": {"object_1": moving},
                }
            )
        )
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    assert settle_run.run_settle(tmp_path, runner=runner) == 1
    reason = json.loads((tmp_path / "ledger.json").read_text())["entries"][-1]["reason"]
    assert reason.startswith("still moving after 2 s: object_1 0.022 m/s | ")
    for piece in (
        "association 3 objects (4 dropped under 3 views)",
        "exported 1, 1 left out by the ICP gate",
        "object_1: 0.20x0.10x0.30 m, bottom -0.004 m, gate reject ['p90']",
        "found as cube x1, mailbox x3 in 4 views",
        "v 0.050->0.022 m/s, w ? rad/s",
        "overlapping boxes: object_1~object_2",
    ):
        assert piece in reason


def test_a_rejected_export_says_why_and_clears_the_stale_settle(tmp_path, monkeypatch):
    import sceneagent.bin_association as association

    monkeypatch.setattr(association, "read_ply_xyz", lambda path: np.zeros((4, 3)))
    rejected = {"object": 0, "placed": True, "gate": "reject"}
    (tmp_path / "bin_score.json").write_text(
        json.dumps({"icp": {"objects": [rejected]}})
    )
    (tmp_path / "association_evidence.json").write_text(
        json.dumps({"splat_unit_m": 0.5})
    )
    (tmp_path / "splat_evidence.json").write_text(json.dumps({"ply": "splat.ply"}))
    (tmp_path / "association").mkdir()
    np.savez(tmp_path / "association" / "objects.npz", object_0=np.arange(2))
    # The last run's settle: a new export makes it stale.
    old = {"item": settle_run.ITEM, "decision": "reject", "reason": "old"}
    (tmp_path / "ledger.json").write_text(json.dumps({"entries": [old]}))
    (tmp_path / settle_run.RESULT).write_text("{}")
    (tmp_path / settle_run.FRAME).write_bytes(b"png")
    assert physics_export.export_run(tmp_path) == 1
    entries = json.loads((tmp_path / "ledger.json").read_text())["entries"]
    assert [e["item"] for e in entries] == [physics_export.EXPORT_ITEM]
    assert "no placed object passed the ICP gate (1 rejected)" in entries[0]["reason"]
    assert not (tmp_path / settle_run.RESULT).exists()
    assert not (tmp_path / settle_run.FRAME).exists()


def test_bin_settle_runs_the_settle_only_after_an_accepted_export(tmp_path):
    from sceneagent.bin_settle import run_bin_settle

    calls = []

    def settle(out):
        calls.append(out)
        return 0

    assert run_bin_settle(tmp_path, export=lambda out: 1, settle=settle) == 1
    assert calls == []
    assert run_bin_settle(tmp_path, export=lambda out: 0, settle=settle) == 0
    assert calls == [tmp_path]


def test_the_bin_settle_pipeline_runs_in_the_bin_dir():
    from sceneagent.vast_smoke import (
        PIPELINES,
        REMOTE_BIN_OUT,
        REMOTE_PYTHON,
        build_remote_command,
        remote_out,
    )

    assert remote_out("bin_settle") == REMOTE_BIN_OUT
    argv = build_remote_command("/w/repo", REMOTE_BIN_OUT, pipeline="bin_settle")
    assert argv == [
        REMOTE_PYTHON,
        "-m",
        "sceneagent.bin_settle",
        "--out",
        REMOTE_BIN_OUT,
    ]
    assert PIPELINES["bin_settle"][0][-2:] == ("USD physics export", "Isaac Lab settle")
