# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Headless check for the Isaac Lab box stage.

Creates a stage, adds a ground plane and a rigid cube, steps 120 frames,
and prints the cube's final height. The box setup runs it through
``isaaclab.sh -p``. It is not imported by the CPU tests.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Lab ground and cube check.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

# AppLauncher has to run before these imports. Kit is not importable until then.
import isaaclab.sim as sim_utils
from pxr import Usd, UsdGeom

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0))
ground = sim_utils.GroundPlaneCfg()
ground.func("/World/defaultGroundPlane", ground)
cube = sim_utils.CuboidCfg(
    size=(0.2, 0.2, 0.2),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    collision_props=sim_utils.CollisionPropertiesCfg(),
)
cube.func("/World/Cube", cube, translation=(0.0, 0.0, 1.0))
sim.reset()
for _ in range(120):
    sim.step()
stage = sim.stage
prim = stage.GetPrimAtPath("/World/Cube")
height = (
    UsdGeom.Xformable(prim)
    .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    .ExtractTranslation()[2]
)
print(f"CUBE_HEIGHT {height}")
app.close()
