# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SF-34: one physics.json per object, every field with its source.

No GPU and no network. The prior is a synthetic checkpoint whose output is
its bias, so each head's answer is set by the test.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from sceneagent import physics_props as pp
from sceneagent.mlp.contract import PRIOR_DIM
from sceneagent.mlp.prior import (
    HIDDEN,
    OUTPUT_DIM,
    SLOT_BODY,
    SLOT_COLLISION,
    SLOT_DEFORMABILITY,
    SLOT_LOG_DENSITY,
    SLOT_LOG_MASS,
    SLOT_MATERIAL,
    SLOT_RIGIDITY,
    PhysicsPrior,
)
from sceneagent.taxonomy import BODY_TYPES, COLLISION_PRIMITIVES, MATERIAL_CLASSES

# The keys physics_export (SF-35, PR #62) reads as PhysX attributes.
EXPORT_PHYSX_KEYS = (
    "body_type",
    "collision_primitive",
    "mass_kg",
    "density_kg_m3",
    "friction_static",
    "friction_dynamic",
    "restitution",
)
LOG_HALF_KG = math.log(0.5)
LOG_800_KG_M3 = math.log(800.0)


def _bias(
    material="metal",
    body="dynamic rigid",
    collision="convex hull",
    log_mass=LOG_HALF_KG,
    log_density=LOG_800_KG_M3,
    rigidity=2.0,
    deformability=-2.0,
):
    bias = np.zeros(OUTPUT_DIM)
    bias[SLOT_MATERIAL.start + MATERIAL_CLASSES.index(material)] = 3.0
    bias[SLOT_BODY.start + BODY_TYPES.index(body)] = 3.0
    bias[SLOT_COLLISION.start + COLLISION_PRIMITIVES.index(collision)] = 3.0
    bias[SLOT_LOG_MASS] = log_mass
    bias[SLOT_LOG_DENSITY] = log_density
    bias[SLOT_RIGIDITY] = rigidity
    bias[SLOT_DEFORMABILITY] = deformability
    return bias


def _constant_prior(bias):
    return PhysicsPrior(
        np.zeros((PRIOR_DIM, HIDDEN)),
        np.zeros(HIDDEN),
        np.zeros((HIDDEN, HIDDEN)),
        np.zeros(HIDDEN),
        np.zeros((HIDDEN, OUTPUT_DIM)),
        bias,
    )


def test_heads_are_read_from_the_mean_with_sigmoid_after_it():
    decoded = pp.decode_prior_mean(_bias(rigidity=0.0, deformability=math.log(3.0)))
    assert decoded["material"] == "metal" and decoded["body_type"] == "dynamic rigid"
    assert decoded["rigidity"] == pytest.approx(0.5)
    assert decoded["deformability"] == pytest.approx(0.75)
    assert decoded["log_mass"] == pytest.approx(math.log(0.5))


def test_the_prior_averages_outputs_then_applies_sigmoid():
    # Two Gaussians whose deformability logits are -4 and +4: the mean logit is
    # 0, so deformability is 0.5. Averaging sigmoids would also give 0.5, so
    # use asymmetric logits: -4 and 0 give sigmoid(-2), not the mean sigmoid.
    w_out = np.zeros((HIDDEN, OUTPUT_DIM))
    w_out[0, SLOT_DEFORMABILITY] = 4.0
    w1 = np.zeros((PRIOR_DIM, HIDDEN))
    w1[0, 0] = 10.0  # hidden[0] = tanh(10 x0) is about +1, 0 or -1
    prior = PhysicsPrior(
        w1, np.zeros(HIDDEN), np.eye(HIDDEN) * 10.0, np.zeros(HIDDEN), w_out, _bias()
    )
    features = np.zeros((2, PRIOR_DIM))
    features[0, 0] = -1.0
    decoded = pp.prior_for_pack(prior, features)
    hidden0 = np.tanh(10.0 * np.tanh(-10.0))
    mean_logit = (-2.0 + hidden0 * 4.0 + -2.0) / 2.0
    assert decoded["deformability"] == pytest.approx(
        1.0 / (1.0 + math.exp(-mean_logit))
    )
    assert decoded["gaussians"] == 2


def test_a_tie_on_any_head_gives_no_class():
    mean = _bias()
    mean[SLOT_MATERIAL.start + MATERIAL_CLASSES.index("wood")] = 3.0
    mean[SLOT_COLLISION.start + COLLISION_PRIMITIVES.index("box")] = 3.0
    decoded = pp.decode_prior_mean(mean)
    assert decoded["material"] is None and decoded["material_weight"] is None
    assert decoded["collision"] is None
    assert decoded["body_type"] == "dynamic rigid"


def test_every_field_is_present_with_its_source():
    record = pp.property_record("mug", prior=pp.decode_prior_mean(_bias()))
    for key in EXPORT_PHYSX_KEYS + (
        "material_class",
        "log_mass",
        "log_density",
        "rigidity",
        "deformability",
        "youngs_modulus",
        "poisson_ratio",
        "rendering_material",
        "simulator",
    ):
        assert key in record
    for key in EXPORT_PHYSX_KEYS + ("material_class", "log_mass", "rigidity"):
        assert key in record["sources"]
    assert record["mass_kg"] == pytest.approx(0.5)
    assert record["density_kg_m3"] == pytest.approx(800.0)
    assert record["friction_static"] is None
    assert record["sources"]["friction_static"] == "default"
    assert record["restitution"] is None
    assert record["sources"]["restitution"] == "default"
    assert record["youngs_modulus"] is None and record["poisson_ratio"] is None
    assert record["simulator"] == "physx"


def test_deformability_routes_to_physgaussian_whatever_the_body_head_says():
    prior = pp.decode_prior_mean(_bias(body="dynamic rigid", deformability=1.0))
    assert prior["deformability"] >= 0.65
    record = pp.property_record("sponge", prior=prior)
    assert record["body_type"] == "soft deformable"
    assert record["simulator"] == "physgaussian"
    assert record["sources"]["body_type"].startswith("deformability >= 0.65")
    soft_head = pp.decode_prior_mean(_bias(body="cloth shell", deformability=-3.0))
    assert pp.property_record("towel", prior=soft_head)["simulator"] == "physgaussian"


def test_a_soft_collider_is_not_handed_to_a_physx_rigid_body():
    prior = pp.decode_prior_mean(_bias(collision="tetrahedral mesh"))
    record = pp.property_record("can", prior=prior)
    assert record["collision_primitive"] is None
    assert record["prior_collision_primitive"] == "tetrahedral mesh"


def test_vlm_friction_is_used_when_present():
    record = pp.property_record(
        "can", prior=pp.decode_prior_mean(_bias()), vlm_physics={"friction": 0.4}
    )
    assert record["friction_static"] == record["friction_dynamic"] == 0.4
    assert record["sources"]["friction_dynamic"] == "VLM estimate"
    with pytest.raises(ValueError):
        pp.property_record("can", vlm_physics={"friction": -1})


def test_the_audit_overrides_the_vote():
    vote = {"object": "mug", "material_class": "plastic", "scores": {"plastic": 1.0}}
    review = {
        "name": "mug",
        "provisional_class": "plastic",
        "audit_class": "glass and ceramic",
        "decision": "replace",
        "final_class": "glass and ceramic",
        "reason": "glazed",
        "signed_by": "coding agent",
    }
    record = pp.property_record("mug", vote=vote, review=review)
    assert record["material_class"] == "glass and ceramic"
    assert record["material_vote"]["material_class"] == "plastic"
    assert record["sources"]["material_class"] == "VLM audit (SF-21)"


def test_without_a_prior_the_numbers_are_null_not_invented():
    record = pp.property_record("glass bottle")
    assert record["material_class"] == "glass and ceramic"  # keyword vote
    assert record["mass_kg"] is None and record["body_type"] is None
    assert record["simulator"] is None
    assert record["sources"]["mass_kg"] == "no prior"


def test_the_property_set_is_written_per_object(tmp_path):
    checkpoint = _constant_prior(_bias(material="plastic")).save(tmp_path / "prior.npz")
    packs = tmp_path / "packs"
    packs.mkdir()
    for n in (3, 7):
        np.save(packs / f"object_{n}.npy", np.zeros((5, PRIOR_DIM)))
    np.save(packs / "object_7_mask.npy", np.array([True, True, False, False, False]))
    objects = [{"object": 3, "name": "apple"}, {"object": 7, "name": "apple"}]
    vote = {
        "votes": [{"object": "apple", "material_class": "food and organic matter"}] * 2
    }
    result = pp.write_property_set(
        objects,
        packs,
        tmp_path / "out",
        checkpoint=checkpoint,
        vote=vote,
        vlm_physics={"objects": {"7": {"friction": 0.6}}},
    )
    first = json.loads((tmp_path / "out/physics/object_3.json").read_text())
    second = json.loads((tmp_path / "out/physics/object_7.json").read_text())
    assert first["prior_gaussians"] == 5 and second["prior_gaussians"] == 2
    assert first["material_class"] == "food and organic matter"
    assert second["friction_static"] == 0.6 and first["friction_static"] is None
    assert first["prior"]["weights"] == "ours" and len(first["prior"]["sha256"]) == 64
    assert result["summary"]["objects"] == 2
    with pytest.raises(ValueError, match="is not"):
        pp.write_property_set(
            objects,
            packs,
            tmp_path / "bad",
            vote={"votes": [{"object": "apple"}, {"object": "pear"}]},
        )


def test_catalogue_masses_are_a_check_not_an_input():
    record = pp.property_record("can", prior=pp.decode_prior_mean(_bias()))
    rows = pp.compare_catalogue_masses([record], {"can": 0.25})
    assert rows[0]["ratio"] == pytest.approx(2.0)
    assert record["mass_kg"] == pytest.approx(0.5)


def test_the_cli_writes_the_summary(tmp_path, capsys):
    objects = tmp_path / "objects.json"
    objects.write_text(json.dumps([{"object": 0, "name": "towel"}]))
    assert pp.main(["--objects", str(objects), "--out", str(tmp_path / "out")]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["objects"] == 1 and summary["prior"] is None
