# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Section 3.1 orchestrator: a harness plan, then ingest, then the runner.

The paper's order is the plan, then ingest, then the runner. Procedures are
written as markdown files a later conversion can read. ``run_agent_loop`` is
not used here: an empty or unknown plan there is replaced with
``default_plan()``. This path raises ``InvalidPlanError`` instead, and it
writes the harness order as given.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from sceneagent.agent_loop import (
    ALLOWED_STEPS,
    STEP_SECTION,
    STEP_SKILL,
    append_step,
    default_plan,
)
from sceneagent.gemini_client import redact
from sceneagent.harness import GEMINI_MODEL_ID, PlanHarness, request_plan

# Our inference, not a list the paper prints. The edges are the predecessors
# this tree already refuses to skip: PartSAM rejects a run with no SAM 3D
# mesh (``fruits_agent``), ICP aligns that mesh, the aligned-mesh inspector
# follows ICP, the hull rollout follows that inspector, and CubePart and
# Articulate Anything consume the PartSAM labels. The other stages have no
# predecessor here. See ``docs/sf24-harness-finding.md``.
STAGE_REQUIRES: dict[str, tuple[str, ...]] = {
    "material_vote": (),
    "mujoco_load": (),
    "sam3d": (),
    "partsam": ("sam3d",),
    "isaac_inspector": (),
    "icp": ("sam3d",),
    "isaac_mesh": ("icp",),
    "mujoco_mesh": ("isaac_mesh",),
    "cubepart": ("partsam",),
    "articulate": ("partsam",),
}

HARNESS_PLAN_PROMPT = (
    "Section 3.1 of SceneAgent. Plan the remaining conversion. "
    "Reply with one JSON object and no other text, of the form "
    '{"steps": ["<stage>"], "procedures": {"<stage>": "<markdown>"}}. '
    "Use only these stage ids: " + ", ".join(ALLOWED_STEPS) + ". "
    "Do not repeat a stage. A stage may appear only after the stages it "
    "requires: partsam after sam3d, icp after sam3d, isaac_mesh after icp, "
    "mujoco_mesh after isaac_mesh, cubepart after partsam, articulate after "
    "partsam. Put a short procedure for each chosen stage under procedures. "
    "Those files are what a later conversion reads."
)


class InvalidPlanError(ValueError):
    """The harness plan is not a usable stage order.

    There is no fallback to ``default_plan()``. Callers see this error
    instead of a silently forced order.
    """


# ``harness`` means the steps came from a model reply. ``default_fallback``
# means ``order_plan`` substituted ``default_plan()`` for an empty parse.
# The second is allowed to exist. It is not allowed to count as dc2.
PLAN_SOURCE_HARNESS = "harness"
PLAN_SOURCE_DEFAULT_FALLBACK = "default_fallback"


def plan_source_name(used_fallback: bool) -> str:
    """Name the source. A fallback is never labeled ``harness``."""

    if used_fallback:
        return PLAN_SOURCE_DEFAULT_FALLBACK
    return PLAN_SOURCE_HARNESS


def counts_toward_dc2(plan: dict[str, Any]) -> dict[str, Any]:
    """Reject a recorded fallback as dc2 evidence.

    dc2 needs an order the model chose. ``default_fallback`` is recorded so
    the substitution is visible, and ``counts`` stays false. A ``harness``
    source is the right kind of order. This function does not claim the live
    dc2 artifact exists.
    """

    source = plan.get("plan_source") if isinstance(plan, dict) else None
    if source == PLAN_SOURCE_DEFAULT_FALLBACK:
        return {
            "counts": False,
            "plan_source": source,
            "reason": (
                "default_plan() was substituted. dc2 needs an order the model "
                "chose, so a fallback plan cannot count."
            ),
        }
    if source != PLAN_SOURCE_HARNESS:
        return {
            "counts": False,
            "plan_source": source,
            "reason": "plan_source is not harness, so the plan cannot count toward dc2.",
        }
    return {
        "counts": True,
        "plan_source": source,
        "reason": ("plan_source is harness. This record is not the live dc2 artifact."),
    }


def stamp_plan_source(out_dir: Path | str, plan_source: str) -> dict[str, Any]:
    """Record ``plan_source`` on an existing ``plan.json`` without replacing steps."""

    path = Path(out_dir) / "plan.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise InvalidPlanError("plan.json is not an object")
    payload["plan_source"] = plan_source
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def validate_plan(steps: list[Any]) -> list[str]:
    """Accept a harness order only when every check below holds.

    Known stages, no duplicates, and each dependency earlier in the list.
    The returned list is the harness order, not ``default_plan()``.
    """

    if not steps:
        raise InvalidPlanError("plan has no steps")
    allowed = set(ALLOWED_STEPS)
    chosen: list[str] = []
    for item in steps:
        if not isinstance(item, str) or item not in allowed:
            raise InvalidPlanError(f"unknown stage: {item!r}")
        if item in chosen:
            raise InvalidPlanError(f"duplicate stage: {item}")
        for dep in STAGE_REQUIRES[item]:
            if dep not in chosen:
                raise InvalidPlanError(f"{item} requires {dep} earlier in the plan")
        chosen.append(item)
    return chosen


def parse_harness_reply(text: str) -> tuple[list[str], dict[str, str]]:
    """Parse a harness reply. Unknown ids are an error, not a drop."""

    raw = text or ""
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise InvalidPlanError("harness reply has no JSON object")
    try:
        payload = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        raise InvalidPlanError("harness reply is not JSON") from exc
    if not isinstance(payload, dict):
        raise InvalidPlanError("harness reply JSON is not an object")
    steps = payload.get("steps")
    if not isinstance(steps, list):
        raise InvalidPlanError("harness reply has no steps list")
    procedures = payload.get("procedures") or {}
    if not isinstance(procedures, dict):
        raise InvalidPlanError("procedures must be an object")
    bodies: dict[str, str] = {}
    for key, value in procedures.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise InvalidPlanError("procedure entries must be strings")
        bodies[key] = value
    return validate_plan(list(steps)), bodies


def procedure_markdown(step_id: str, body: str) -> str:
    """One procedure file. The body is the harness text when it sent one."""

    section = STEP_SECTION[step_id]
    skill = STEP_SKILL[step_id]
    text = body.strip()
    if not text:
        text = (
            f"Run {step_id} ({skill}) for section {section}. "
            "Later conversions read this file instead of inventing the step."
        )
    return f"# {step_id}\n\nsection: {section}\nskill: {skill}\n\n{text}\n"


def write_procedures(
    out_dir: Path, steps: list[str], bodies: dict[str, str], secret: str
) -> dict[str, str]:
    """Write ``procedures/<stage>.md`` for the planned stages."""

    folder = out_dir / "procedures"
    folder.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for step_id in steps:
        content = redact(procedure_markdown(step_id, bodies.get(step_id, "")), secret)
        (folder / f"{step_id}.md").write_text(content, encoding="utf-8")
        written[step_id] = content
    return written


def read_procedures(out_dir: Path | str) -> dict[str, str]:
    """Read procedure files a previous conversion saved. Does not call a model."""

    folder = Path(out_dir) / "procedures"
    if not folder.is_dir():
        raise FileNotFoundError(f"no procedures in {out_dir}")
    found = {
        path.stem: path.read_text(encoding="utf-8")
        for path in sorted(folder.glob("*.md"))
    }
    if not found:
        raise FileNotFoundError(f"no procedures in {out_dir}")
    return found


def later_conversion(out_dir: Path | str) -> dict[str, Any]:
    """A later conversion reads the saved plan order and procedure files."""

    out = Path(out_dir)
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    procedures = read_procedures(out)
    order = [step["id"] for step in plan["steps"]]
    missing = [step_id for step_id in order if step_id not in procedures]
    if missing:
        raise FileNotFoundError(f"procedures missing for {missing}")
    return {"order": order, "procedures": procedures}


def _secret() -> str:
    """The same env var, used only to redact bytes about to be written."""

    raw = os.environ.get("GEMINI_API_KEY", "")
    return str(raw).strip().strip('"').strip("'")


def _write_json(path: Path, payload: Any, secret: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2) + "\n"
    path.write_text(redact(text, secret), encoding="utf-8")


def _run_steps(
    out: Path,
    steps: list[str],
    tools: dict[str, Any] | None,
    context: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Runner: one tool, one manifest, one ledger line. No plan rewrite."""

    tools = tools or {}
    ctx = dict(context or {})
    entries = []
    for step_id in steps:
        tool = tools.get(step_id)
        if tool is None:
            result: dict[str, Any] = {
                "ok": False,
                "artifact": None,
                "reason": "no tool registered",
                "detail": {},
            }
        else:
            result = tool(ctx)
        entries.append(append_step(out, step_id, result, repair=False))
    return entries


def run_harness_plan(
    out_dir: Path | str,
    harness: PlanHarness | None = None,
    tools: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Plan with the harness, validate, then ingest, then run the stages.

    An invalid reply raises ``InvalidPlanError`` before ``plan.json`` is
    written. ``default_plan()`` is not a substitute.
    """

    impl = harness
    before = int(getattr(impl, "call_count", 0)) if impl is not None else 0
    reply = request_plan(HARNESS_PLAN_PROMPT, harness=impl)
    after = int(getattr(impl, "call_count", before + 1)) if impl is not None else 1
    call_count = after - before
    if call_count < 1:
        raise RuntimeError("harness was not called")
    steps, bodies = parse_harness_reply(reply)
    secret = _secret()
    reply = redact(reply, secret)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan_body = {
        "section": "3.1",
        "order_source": PLAN_SOURCE_HARNESS,
        "plan_source": PLAN_SOURCE_HARNESS,
        "harness": GEMINI_MODEL_ID,
        "model_id": getattr(impl, "model_id", GEMINI_MODEL_ID)
        if impl is not None
        else GEMINI_MODEL_ID,
        "forced": False,
        "steps": [
            {"id": step_id, "section": STEP_SECTION[step_id]} for step_id in steps
        ],
    }
    _write_json(out / "plan.json", plan_body, secret)
    _write_json(
        out / "transcript.json",
        {
            "section": "3.1",
            "harness": GEMINI_MODEL_ID,
            "model_id": plan_body["model_id"],
            "call_count": call_count,
            "reply": reply,
        },
        secret,
    )
    written = write_procedures(out, steps, bodies, secret)
    _write_json(
        out / "manifest.json",
        {
            "section": "3.1",
            "harness": GEMINI_MODEL_ID,
            "model_id": plan_body["model_id"],
            "plan_source": PLAN_SOURCE_HARNESS,
            "call_count": call_count,
            "steps": steps,
        },
        secret,
    )
    _write_json(
        out / "steps" / "SCENEAGENT-INGEST-DATA" / "manifest.json",
        {
            "skill": "SCENEAGENT-INGEST-DATA",
            "section": "3.1",
            "n_steps": len(steps),
            "limitation": (
                "Ingest runs after the harness plan and before the runner. "
                "This CPU record does not run COLMAP, splatfacto, or SAM."
            ),
        },
        secret,
    )
    entries: list[dict[str, Any]] = [
        {
            "item": "plan",
            "section": "3.1",
            "decision": "accept",
            "path": "plan.json",
            "order_source": PLAN_SOURCE_HARNESS,
            "plan_source": PLAN_SOURCE_HARNESS,
            "forced": False,
        },
        {
            "item": "transcript",
            "section": "3.1",
            "decision": "accept",
            "path": "transcript.json",
        },
    ]
    for step_id in steps:
        entries.append(
            {
                "item": f"procedure:{step_id}",
                "section": STEP_SECTION[step_id],
                "decision": "accept",
                "skill": STEP_SKILL[step_id],
                "path": f"procedures/{step_id}.md",
            }
        )
    entries.append(
        {
            "item": "ingest",
            "section": "3.1",
            "decision": "accept",
            "skill": "SCENEAGENT-INGEST-DATA",
        }
    )
    _write_json(out / "ledger.json", {"section": "3.1", "entries": entries}, secret)
    runner_entries = _run_steps(out, steps, tools, context)
    return {
        "out_dir": str(out),
        "steps": steps,
        "call_count": call_count,
        "procedures": written,
        "entries": entries + runner_entries,
        "forced_order": list(default_plan()),
    }
