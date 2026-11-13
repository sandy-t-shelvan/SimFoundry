# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KAN-52: the MVP acceptance checker, one run's artifacts in, pass or fail per criterion.

    python -m sceneagent.mvp_check --run DIR [--report DIR/isaac_lab.json]

Each criterion prints met, partly met or not met, and the file it read.
This first version checks dc4; the others are listed as not yet checked
(KAN-53 hand-off, task 3).

dc4 (Sandeep, 2026-11-13, option A): the scene settles in Isaac Lab, and the
inspector's report is model-written: the model replied and the reply parsed
(``model_replied`` true, no ``error``), whether or not it listed fixes. KAN-52's
literal ``fix_report_from_model: true`` would fail a clean scene, since the
inspector sets it only when the model lists at least one fix (#64).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

MET, PARTLY, NOT_MET = "met", "partly met", "not met"


def _read(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def check_dc4(run: Path, report: Path | None = None) -> dict[str, Any]:
    """Settle accepted, and an inspector report the model wrote (option A)."""
    settle_path = run / "settle.json"
    report_path = report or run / "isaac_lab.json"
    settle = _read(settle_path)
    inspection = _read(report_path)
    settled = bool(settle and settle.get("decision") == "accept")
    written = bool(
        inspection
        and inspection.get("model_replied") is True
        and not inspection.get("error")
    )
    status = MET if settled and written else PARTLY if settled or written else NOT_MET
    return {
        "criterion": "dc4",
        "status": status,
        "settled": settled,
        "settle_reason": None if settle is None else settle.get("reason"),
        "report_model_written": written,
        "findings": None
        if inspection is None
        else len(inspection.get("findings") or []),
        "fix_report_from_model": None
        if inspection is None
        else inspection.get("fix_report_from_model"),
        "files": [str(settle_path), str(report_path)],
        "rule": "option A: model replied and parsed, findings or not (Sandeep, 2026-11-13)",
    }


NOT_YET = ("dc1", "dc2", "dc6", "dc8")


def check_run(run: Path, report: Path | None = None) -> list[dict[str, Any]]:
    """Every MVP criterion for one run; those not yet checked say so."""
    rows = [check_dc4(run, report)]
    rows += [
        {"criterion": c, "status": "not yet checked", "files": []} for c in NOT_YET
    ]
    return sorted(rows, key=lambda r: r["criterion"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)
    rows = check_run(args.run, args.report)
    for row in rows:
        print(f"{row['criterion']}: {row['status']}  {', '.join(row['files'])}")
    print(json.dumps(rows, indent=2))
    return 0 if all(r["status"] == MET for r in rows if r["criterion"] == "dc4") else 1


if __name__ == "__main__":
    sys.exit(main())
