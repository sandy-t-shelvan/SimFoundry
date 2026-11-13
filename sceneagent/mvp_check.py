# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KAN-52: the MVP acceptance checker, one run's artifacts in, pass or fail per criterion.

    python -m sceneagent.mvp_check --run DIR [--report DIR/isaac_lab.json]
        [--review DIR]

Each criterion prints met, partly met or not met, and the file it read.
dc3, dc5 and dc7 are post-MVP and are not checked.

dc1: ``ledger.json`` accepts every ``video_run`` stage, and ``step_verdicts.json``
(from the runner review, beside the run or in ``--review``) accepts every
stage it judged.
dc2: ``physics/object_<n>.json`` for every exported object, with every §3.4
field and a ``sources`` entry. Met when at least one record is prior-backed.
``plan_source`` ``fallback`` keeps default physics and never counts as
prior-backed. Fields with no prior are option B and partly met.
dc4 (Sandeep, 2026-11-13, option A): the scene settles, and the inspector
report is model-written (``model_replied`` true, no ``error``), findings or not.
dc6: a ``*_scene_state_exported.json`` from the editor import, with physics
records and link materials. Joints are not required; articulation is post-MVP.
dc8: the 40-character commit SHA of the code that ran, from ``manifest.json``
or ``checked_out_commit.txt``. Whether that commit is on main is for the PM.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from sceneagent.physics import UNPUBLISHED_LOOKUPS
from sceneagent.video_run import STAGES, accepted

MET, PARTLY, NOT_MET = "met", "partly met", "not met"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_NUMBER = re.compile(r"(\d+)$")
# Every key ``physics_props.property_record`` writes for one object.
SECTION_34_FIELDS = (
    "material_class",
    "body_type",
    "simulator",
    "collision_primitive",
    "log_mass",
    "log_density",
    "rigidity",
    "deformability",
    "mass_kg",
    "density_kg_m3",
    "friction_static",
    "friction_dynamic",
    "restitution",
    *UNPUBLISHED_LOOKUPS,
)
MVP = ("dc1", "dc2", "dc4", "dc6", "dc8")


def _read(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _verdicts_path(run: Path, review: Path | None) -> Path:
    return (review or run) / "step_verdicts.json"


def check_dc1(run: Path, review: Path | None = None) -> dict[str, Any]:
    """Ledger accepts every video stage, and the critic accepted what it judged."""
    ledger_path = run / "ledger.json"
    verdicts_path = _verdicts_path(run, review)
    items = tuple(item for _, stage_items in STAGES for item in stage_items)
    ledger_ok = accepted(run, items)
    verdicts = _read(verdicts_path)
    stages: Any = None
    if verdicts is not None:
        stages = verdicts.get("stages")
    critic_ok = (
        verdicts is not None
        and verdicts.get("mode") == "step"
        and isinstance(stages, list)
        and len(stages) > 0
        and all(
            isinstance(item, dict) and item.get("verdict") == "accept"
            for item in stages
        )
    )
    if ledger_ok and critic_ok:
        status = MET
    elif ledger_ok or critic_ok:
        status = PARTLY
    else:
        status = NOT_MET
    return {
        "criterion": "dc1",
        "status": status,
        "ledger_accepted": ledger_ok,
        "critic_accepted": critic_ok,
        "critic_stages": len(stages) if isinstance(stages, list) else 0,
        "files": [str(ledger_path), str(verdicts_path)],
    }


def _exported_numbers(evidence: dict[str, Any]) -> list[int] | None:
    objects = evidence.get("objects")
    if not isinstance(objects, list):
        return None
    numbers: list[int] = []
    for item in objects:
        name = item.get("object") if isinstance(item, dict) else item
        if isinstance(name, bool):
            return None
        if isinstance(name, int):
            numbers.append(name)
            continue
        match = _NUMBER.search(str(name))
        if match is None:
            return None
        numbers.append(int(match.group(1)))
    return numbers


def _record_complete(record: dict[str, Any] | None) -> tuple[bool, bool]:
    """(fields and sources present, prior sha256 present)."""
    if not record or record.get("section") != "3.4":
        return False, False
    sources = record.get("sources")
    if not isinstance(sources, dict):
        return False, False
    fields = all(key in record and key in sources for key in SECTION_34_FIELDS)
    # A fallback keeps default physics. It is never prior-backed.
    if record.get("plan_source") == "fallback":
        return fields, False
    prior = record.get("prior")
    sha = prior.get("sha256") if isinstance(prior, dict) else None
    return fields, isinstance(sha, str) and _SHA.fullmatch(sha) is not None


def check_dc2(run: Path) -> dict[str, Any]:
    """Every exported object has a §3.4 record. Prior sha256 is option A."""
    evidence_path = run / "export_evidence.json"
    evidence = _read(evidence_path)
    numbers = None if evidence is None else _exported_numbers(evidence)
    files = [str(evidence_path)]
    if not numbers:
        return {
            "criterion": "dc2",
            "status": NOT_MET,
            "objects": 0,
            "with_fields": 0,
            "with_prior": 0,
            "files": files,
        }
    with_fields = 0
    with_prior = 0
    for number in numbers:
        path = run / "physics" / f"object_{number}.json"
        files.append(str(path))
        fields, prior = _record_complete(_read(path))
        with_fields += int(fields)
        with_prior += int(fields and prior)
    if with_fields == len(numbers) and with_prior >= 1:
        status = MET
    elif with_fields == len(numbers):
        status = PARTLY
    else:
        status = NOT_MET
    return {
        "criterion": "dc2",
        "status": status,
        "objects": len(numbers),
        "with_fields": with_fields,
        "with_prior": with_prior,
        "files": files,
    }


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


def _editor_scene(run: Path) -> Path | None:
    matches = sorted(run.glob("*_scene_state_exported.json"))
    matches += sorted(
        p for p in run.glob("*/*_scene_state_exported.json") if p.is_file()
    )
    return matches[0] if matches else None


def _editor_ready(document: dict[str, Any]) -> bool:
    info = document.get("objects_info")
    init = info.get("init_info") if isinstance(info, dict) else None
    records = document.get("physics_records")
    if (
        not isinstance(init, dict)
        or not init
        or not isinstance(document.get("state"), dict)
    ):
        return False
    if not isinstance(records, dict) or not records:
        return False
    for spec in init.values():
        args = spec.get("args") if isinstance(spec, dict) else None
        materials = (
            args.get("link_physics_materials") if isinstance(args, dict) else None
        )
        if not isinstance(materials, dict):
            return False
    return True


def check_dc6(run: Path) -> dict[str, Any]:
    """The editor import wrote a scene state the physics panel can read."""
    path = _editor_scene(run)
    document = None if path is None else _read(path)
    ready = bool(document and _editor_ready(document))
    status = MET if ready else PARTLY if document else NOT_MET
    return {
        "criterion": "dc6",
        "status": status,
        "files": [] if path is None else [str(path)],
        "note": "joints are not required; articulation is post-MVP",
    }


def check_dc8(run: Path) -> dict[str, Any]:
    """The commit SHA recorded for the run. Merged-to-main is not decided here."""
    manifest_path = run / "manifest.json"
    text_path = run / "checked_out_commit.txt"
    manifest = _read(manifest_path)
    sha = None if manifest is None else manifest.get("commit_sha")
    used = manifest_path
    if not isinstance(sha, str) or _SHA.fullmatch(sha) is None:
        try:
            sha = text_path.read_text(encoding="utf-8").strip()
            used = text_path
        except OSError:
            sha = ""
    ok = isinstance(sha, str) and _SHA.fullmatch(sha) is not None
    return {
        "criterion": "dc8",
        "status": MET if ok else NOT_MET,
        "commit_sha": sha if ok else None,
        "files": [str(used)],
    }


def check_run(
    run: Path, report: Path | None = None, review: Path | None = None
) -> list[dict[str, Any]]:
    """The MVP criteria for one run, in criterion order."""
    rows = [
        check_dc1(run, review),
        check_dc2(run),
        check_dc4(run, report),
        check_dc6(run),
        check_dc8(run),
    ]
    return sorted(rows, key=lambda row: row["criterion"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--review",
        type=Path,
        default=None,
        help="Runner review directory holding step_verdicts.json",
    )
    args = parser.parse_args(argv)
    rows = check_run(args.run, args.report, args.review)
    for row in rows:
        print(f"{row['criterion']}: {row['status']}  {', '.join(row['files'])}")
    print(json.dumps(rows, indent=2))
    return 0 if all(row["status"] == MET for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
