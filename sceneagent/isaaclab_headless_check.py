# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One headless Isaac Sim step for the box setup stage.

CPU tests read the constants. Importing ``isaacsim`` boots Kit, so this
module does not import it until ``main`` runs on the box.
"""

from __future__ import annotations

import os

EULA_ENV = "OMNI_KIT_ACCEPT_EULA"
FRAMES = 120
CUBE_PATH = "/World/Cube"
CUBE_Z_M = 0.5


def main() -> None:
    """Open a stage, drop a cube onto a ground plane, and print its height."""

    os.environ[EULA_ENV] = "YES"
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCuboid

        world = World(stage_units_in_meters=1.0)
        world.scene.add_default_ground_plane()
        cube = DynamicCuboid(
            prim_path=CUBE_PATH,
            name="cube",
            position=(0.0, 0.0, CUBE_Z_M),
            size=0.1,
        )
        world.scene.add(cube)
        world.reset()
        for _ in range(FRAMES):
            world.step(render=False)
        height = float(cube.get_world_pose()[0][2])
        print(f"cube_height_m={height:.6f}")
    finally:
        app.close()


if __name__ == "__main__":
    main()
