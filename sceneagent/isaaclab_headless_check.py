# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One headless Isaac Sim step for the box install.

Creates a stage, adds a ground plane and a dynamic cube, steps 120 frames,
and prints the cube's final height. CPU CI does not import Isaac Sim; the
box stage runs this file after the pinned install.
"""

from __future__ import annotations

import os

# The installer documents this for a non-interactive accept. A prompt would
# hang a detached setup stage.
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

FRAMES = 120


def main() -> None:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import numpy as np
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCuboid

        world = World(stage_units_in_meters=1.0)
        world.scene.add_default_ground_plane()
        cube = world.scene.add(
            DynamicCuboid(
                prim_path="/World/Cube",
                name="cube",
                position=np.array([0.0, 0.0, 0.5]),
                size=0.1,
            )
        )
        world.reset()
        for _ in range(FRAMES):
            world.step(render=False)
        pos, _quat = cube.get_world_pose()
        print(f"cube_final_height_m={float(pos[2])}")
    finally:
        app.close()


if __name__ == "__main__":
    main()
