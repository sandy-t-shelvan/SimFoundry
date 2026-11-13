# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SF-40: route one object to PhysGaussian or to a rigid PhysX actor.

Sandeep's note "Using the MLP prior with PhysX and PhysGaussian" (10 November
2026), sections 4 and 5:

- PhysGaussian when the body is soft deformable, cloth shell, or fluid or
  granular, or when deformability is at least 0.65. The deformability rule
  wins when it fires.
- The same Gaussians are the particles. MLP rigidity sets stiffness, so
  stiffness is that rigidity value. The note gives no scale.
- The MLP mass is the particle mass, split across the N Gaussians:
  ``particle_mass = mass_kg / n_gaussians``.
- Each frame steps PhysX, then PhysGaussian under that contact, then renders.

Young's modulus and Poisson's ratio are unpublished. This module does not
fill them in. A simulator that needs those two numbers cannot be given them
from the note.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from sceneagent.physics import SOFT_BODY_DEFORMABILITY
from sceneagent.physics_props import PHYSGAUSSIAN_BODY_TYPES, PHYSX_BODY_TYPES

MASS_FORMULA = "particle_mass = mass_kg / n_gaussians"
STIFFNESS_FORMULA = "stiffness = rigidity"


def route(props: Mapping[str, Any]) -> tuple[str, str]:
    """``("physgaussian" | "physx_rigid", reason)``.

    The reason names the value that decided. Deformability at or above
    0.65 wins over the body type.
    """
    body = props.get("body_type")
    raw = props.get("deformability")
    deformability = None if raw is None else float(raw)
    if deformability is not None and not math.isfinite(deformability):
        raise ValueError("deformability is not finite")
    if deformability is not None and deformability >= SOFT_BODY_DEFORMABILITY:
        return (
            "physgaussian",
            f"deformability {deformability:g} >= {SOFT_BODY_DEFORMABILITY:g}",
        )
    if body in PHYSGAUSSIAN_BODY_TYPES:
        return "physgaussian", f"body type {body}"
    if body in PHYSX_BODY_TYPES:
        return "physx_rigid", f"body type {body}"
    raise ValueError(f"no route for body type {body!r} and deformability {raw!r}")


def particle_params(props: Mapping[str, Any], n_gaussians: int) -> dict[str, Any]:
    """Per-particle mass and stiffness, with the formula and both inputs.

    Mass is the MLP mass divided by the number of Gaussians. Stiffness is
    the MLP rigidity (note section 4). ``log`` is the trace a run writes.
    """
    count = int(n_gaussians)
    if isinstance(n_gaussians, bool) or count < 1:
        raise ValueError("n_gaussians must be a positive count")
    mass = _required_float(props, "mass_kg")
    if mass < 0.0:
        raise ValueError("mass_kg must be non-negative")
    rigidity = _required_float(props, "rigidity")
    particle_mass = mass / count
    stiffness = rigidity
    log = [
        f"formula: {MASS_FORMULA}",
        f"mass_kg: {mass:g}",
        f"n_gaussians: {count}",
        f"particle_mass: {particle_mass:g}",
        f"formula: {STIFFNESS_FORMULA}",
        f"rigidity: {rigidity:g}",
        f"stiffness: {stiffness:g}",
    ]
    return {
        "particle_mass": particle_mass,
        "stiffness": stiffness,
        "log": log,
    }


def step_frame(physx: Any, physgaussian: Any, renderer: Any) -> Any:
    """One frame, in the note's order, using the simulators passed in.

    PhysX step, then the contact forces on ``physgaussian.proxy``, then
    the PhysGaussian step under those forces, then the moved Gaussians,
    then the render.
    """
    physx.step()
    forces = physx.read_contact_forces(physgaussian.proxy)
    physgaussian.step(forces)
    moved = physgaussian.moved_gaussians()
    renderer.write_gaussians(moved)
    return renderer.render()


def _required_float(props: Mapping[str, Any], key: str) -> float:
    if key not in props or props[key] is None or isinstance(props[key], bool):
        raise ValueError(f"{key} is required")
    value = float(props[key])
    if not math.isfinite(value):
        raise ValueError(f"{key} is not finite")
    return value
