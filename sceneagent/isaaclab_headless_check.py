# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Headless Isaac Lab check for the box setup stage.

Runs only on the GPU box, after ``s_isaaclab`` has installed Isaac Sim and
Isaac Lab. It builds a stage with a ground plane and one cube, steps 120
frames, and prints the cube's final height. The setup stage fails if that
line is missing.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("HEADLESS", "1")

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0))
ground = sim_utils.GroundPlaneCfg()
ground.func("/World/ground", ground)
cube = RigidObject(
    RigidObjectCfg(
        prim_path="/World/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    )
)
sim.reset()
for _ in range(120):
    sim.step()
    cube.update(sim.cfg.dt)
print(f"cube_height_m {float(cube.data.root_pos_w[0, 2])}")
simulation_app.close()
