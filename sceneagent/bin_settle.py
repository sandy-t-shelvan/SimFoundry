# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SF-35 (KAN-41) on the bin: export the placed GraspClutter6D scene, settle it, inspect it.

    python -m sceneagent.bin_settle --out /workspace/sf4-smoke/bin

The bin's place stage has already run in ``--out``. This exports the objects
that passed the ICP gate (:func:`sceneagent.physics_export.export_run`),
settles them in Isaac Lab (:func:`sceneagent.settle_run.run_settle`) and draws
the inspector's views of the settled scene (:func:`sceneagent.inspect_run.run_inspect`,
CPU), as ``video_run`` does. Each step runs only after the one before it is
accepted. SF-22 (KAN-23) asks for the model-written report on a bin object;
the report itself is written on the runner (``sceneagent-inspector-report``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sceneagent.inspect_run import run_inspect
from sceneagent.physics_export import export_run
from sceneagent.settle_run import run_settle


def run_bin_settle(
    out: Path | str, *, export=export_run, settle=run_settle, inspect=run_inspect
) -> int:
    """Export, settle, then inspect. Return 0 only when all three are accepted."""
    for step in (export, settle, inspect):
        code = step(Path(out))
        if code != 0:
            return code
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    return run_bin_settle(args.out)


if __name__ == "__main__":
    sys.exit(main())
