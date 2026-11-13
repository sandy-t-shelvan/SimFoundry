# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SF-35 (KAN-41) on the bin: export the placed GraspClutter6D scene and settle it.

    python -m sceneagent.bin_settle --out /workspace/sf4-smoke/bin

The bin's place stage has already run in ``--out``. This exports the objects
that passed the ICP gate (:func:`sceneagent.physics_export.export_run`) and
settles them in Isaac Lab (:func:`sceneagent.settle_run.run_settle`). Settle
runs only after export is accepted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sceneagent.physics_export import export_run
from sceneagent.settle_run import run_settle


def run_bin_settle(out: Path | str, *, export=export_run, settle=run_settle) -> int:
    """Export, then settle. Return 0 only when both are accepted."""
    code = export(Path(out))
    return code if code != 0 else settle(Path(out))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    return run_bin_settle(args.out)


if __name__ == "__main__":
    sys.exit(main())
