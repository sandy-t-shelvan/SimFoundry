# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Drop and press one soft object. PhysGaussian is not installed by this script."""

from __future__ import annotations

import json
import sys
from pathlib import Path

MISSING = "PhysGaussian is not installed on this box; soft_press does not install it"
ITEM = "soft press"
ROOT = Path("/workspace/src/PhysGaussian")


def deps_ready(importer=None, root: Path = ROOT) -> bool:
    """True only when warp, taichi, and the pinned checkout are already there."""
    load = importer or __import__
    try:
        load("warp")
        load("taichi")
    except ImportError:
        return False
    return Path(root).is_dir()


def main() -> None:
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    params = Path(sys.argv[sys.argv.index("--params") + 1])
    out.mkdir(parents=True, exist_ok=True)
    if params.is_file():
        (out / "soft_press_params.json").write_bytes(params.read_bytes())
    if not deps_ready():
        _finish(out, "reject", MISSING)
        print(MISSING, flush=True)
        sys.exit(1)
    _finish(
        out,
        "reject",
        "PhysGaussian is present; the drop and press scene is not run here",
    )
    print("PhysGaussian is present; drop and press was not started", flush=True)
    sys.exit(1)


def _finish(out: Path, decision: str, reason: str) -> None:
    (out / "soft_press_status.txt").write_text(reason + "\n", encoding="utf-8")
    (out / "ledger.json").write_text(
        json.dumps(
            {
                "scene_name": "soft press",
                "entries": [{"item": ITEM, "decision": decision, "reason": reason}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "plan.json").write_text(
        json.dumps({"pipeline": "soft_press", "item": ITEM}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
