# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SF-40 routing, the particle-parameter trace, and the frame order."""

from __future__ import annotations

import pytest

from sceneagent.soft_route import MASS_FORMULA, STIFFNESS_FORMULA, particle_params, route, step_frame


def test_the_threshold_from_both_sides():
    soft, soft_reason = route({"body_type": "dynamic rigid", "deformability": 0.65})
    assert soft == "physgaussian"
    assert soft_reason == "deformability 0.65 >= 0.65"
    below, below_reason = route({"body_type": "dynamic rigid", "deformability": 0.649})
    assert below == "physx_rigid"
    assert below_reason == "body type dynamic rigid"
    for body in ("static rigid", "kinematic"):
        decision, reason = route({"body_type": body, "deformability": 0.649})
        assert decision == "physx_rigid"
        assert reason == f"body type {body}"


def test_a_rigid_body_type_with_deformability_0_7_routes_soft():
    decision, reason = route({"body_type": "dynamic rigid", "deformability": 0.7})
    assert decision == "physgaussian"
    assert reason == "deformability 0.7 >= 0.65"


def test_soft_cloth_and_fluid_route_soft_below_the_threshold():
    for body in ("soft deformable", "cloth shell", "fluid or granular"):
        decision, reason = route({"body_type": body, "deformability": 0.1})
        assert decision == "physgaussian"
        assert reason == f"body type {body}"


def test_parameter_trace_records_the_formula_and_both_inputs():
    traced = particle_params({"mass_kg": 0.5, "rigidity": 0.25}, 4)
    assert traced["particle_mass"] == pytest.approx(0.125)
    assert traced["stiffness"] == pytest.approx(0.25)
    assert traced["log"] == [
        f"formula: {MASS_FORMULA}",
        "mass_kg: 0.5",
        "n_gaussians: 4",
        "particle_mass: 0.125",
        f"formula: {STIFFNESS_FORMULA}",
        "rigidity: 0.25",
        "stiffness: 0.25",
    ]
    assert "youngs" not in " ".join(traced["log"]).lower()
    assert "poisson" not in " ".join(traced["log"]).lower()
    with pytest.raises(ValueError):
        particle_params({"mass_kg": 0.5, "rigidity": 0.25}, 0)


def test_frame_order_from_a_stub_call_log():
    log: list[object] = []
    moved = [(0.0, 0.0, 0.0), (0.01, 0.0, -0.02)]
    forces = (0.0, 0.0, -1.2)

    class PhysX:
        def step(self):
            log.append("physx.step")

        def read_contact_forces(self, proxy):
            log.append(("physx.read_contact_forces", proxy))
            return forces

    class Particles:
        proxy = "sponge"

        def step(self, contact):
            log.append(("physgaussian.step", contact))

        def moved_gaussians(self):
            log.append("physgaussian.moved_gaussians")
            return moved

    class Renderer:
        def write_gaussians(self, gaussians):
            log.append(("renderer.write_gaussians", gaussians))

        def render(self):
            log.append("renderer.render")
            return "frame"

    assert step_frame(PhysX(), Particles(), Renderer()) == "frame"
    assert log == [
        "physx.step",
        ("physx.read_contact_forces", "sponge"),
        ("physgaussian.step", forces),
        "physgaussian.moved_gaussians",
        ("renderer.write_gaussians", moved),
        "renderer.render",
    ]
