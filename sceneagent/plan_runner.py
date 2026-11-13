# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SF-24: the orchestrator's plan from Gemini on the runner, run on the box.

The paper's order is the orchestrator's plan, then ingest, then the runner,
with procedures saved to files later conversions read (section 3.1). Here
the plan comes from ``gemini-3.8-flash`` (Sandeep, 2026-11-11: one Gemini API
for the VLM and the agents). The paper's 60.75% bin number is with Codex as
the harness, so anything this plan produces is Gemini's result, not that one.

    python -m sceneagent.plan_runner plan --out-dir vlm-out [--source-run RUN]
    python -m sceneagent.plan_runner run --plan sceneagent/plans/harness_plan.json --out OUT

``plan`` runs on the GitHub runner, where ``GEMINI_API_KEY`` is: one call
with a response schema, the reply checked by ``orchestrator.validate_plan``
(known stages, no repeats, every dependency earlier), and ``plan.json`` plus
``transcript.json`` written. An invalid reply raises; it is never reordered
or replaced with ``default_plan()``. The plan is committed and reaches the
box in the shipped ``sceneagent/`` tree, as the VLM verdicts do, so the key
never does.

``run`` runs on the box: it checks the committed plan again (same stages,
same rules, the prompt it was asked with), runs the stages in the model's
order, and after each stage writes ``procedures/<stage>.md``: the model's
procedure plus what the run did. ``orchestrator.later_conversion`` reads
those files back.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sceneagent.agent_loop import (
    ALLOWED_STEPS,
    STEP_SECTION,
    STEP_SKILL,
    append_step,
    consider_repair,
    harness_status,
)
from sceneagent.harness import GEMINI_MODEL_ID, HarnessCallError
from sceneagent.orchestrator import (
    PLAN_SOURCE_HARNESS,
    STAGE_REQUIRES,
    InvalidPlanError,
    procedure_markdown,
    validate_plan,
)

PLAN_MODE = "plan"
PLAN_RELATIVE = "sceneagent/plans/harness_plan.json"
PROCEDURE_CHARS = 4000


def _dependency_sentence() -> str:
    """The hard dependencies, written out from ``STAGE_REQUIRES`` itself."""
    rules = [
        f"{stage} after {', '.join(deps)}"
        for stage, deps in STAGE_REQUIRES.items()
        if deps
    ]
    return "A stage may appear only after the stages it requires: " + "; ".join(rules)


# Ours. The model chooses which stages to run and in what order; the schema
# fixes only the shape and the vocabulary, so validation still decides.
PLAN_PROMPT = (
    "Section 3.1 of SceneAgent. You are the orchestrator. Plan the remaining "
    "conversion of a captured scene into a simulation-ready one: choose the "
    "stages to run and their order. Use only these stage ids: "
    + ", ".join(ALLOWED_STEPS)
    + ". Do not repeat a stage. "
    + _dependency_sentence()
    + ". For every stage you choose, write a short procedure in markdown: what "
    "the stage takes in, what it must produce, and how to tell it worked. "
    "Later conversions read those procedures instead of planning again. "
    "Answer with steps (your order) and procedures (one entry per chosen stage)."
)

PLAN_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "steps": {
            "type": "ARRAY",
            "items": {"type": "STRING", "enum": list(ALLOWED_STEPS)},
        },
        # A list, not a map: the response schema cannot express free keys.
        "procedures": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "stage": {"type": "STRING", "enum": list(ALLOWED_STEPS)},
                    "markdown": {"type": "STRING"},
                },
                "required": ["stage", "markdown"],
            },
        },
    },
    "required": ["steps", "procedures"],
}

Tool = Callable[[dict[str, Any]], dict[str, Any]]


def prompt_sha256() -> str:
    """Hash of the prompt and schema, recorded in the plan and checked on the box."""
    blob = PLAN_PROMPT + json.dumps(PLAN_SCHEMA, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def parse_plan_reply(text: str) -> tuple[list[str], dict[str, str]]:
    """The model's reply as ``(steps, procedures)``; raise on anything off.

    Steps go through ``validate_plan`` unchanged. A procedure for a stage
    the plan does not run, or two for one stage, is an error rather than
    something to drop quietly.
    """
    body = (text or "").strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", body, flags=re.DOTALL)
    if fenced:
        body = fenced.group(1)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise InvalidPlanError("plan reply is not JSON") from exc
    if not isinstance(payload, dict):
        raise InvalidPlanError("plan reply is not an object")
    steps = payload.get("steps")
    if not isinstance(steps, list):
        raise InvalidPlanError("plan reply has no steps list")
    chosen = validate_plan(list(steps))
    entries = payload.get("procedures")
    if not isinstance(entries, list):
        raise InvalidPlanError("plan reply has no procedures list")
    procedures: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise InvalidPlanError("a procedure entry is not an object")
        stage, markdown = entry.get("stage"), entry.get("markdown")
        if not isinstance(stage, str) or not isinstance(markdown, str):
            raise InvalidPlanError("a procedure entry needs stage and markdown")
        if stage not in chosen:
            raise InvalidPlanError(
                f"procedure for {stage!r}, which the plan does not run"
            )
        if stage in procedures:
            raise InvalidPlanError(f"two procedures for {stage}")
        procedures[stage] = markdown.strip()[:PROCEDURE_CHARS]
    return chosen, procedures


def ask_for_plan(harness: Any) -> str:
    """One schema-constrained call. A figure 5 harness name is refused, as in ``request_plan``."""
    status = harness_status()
    if status["called"]:
        raise RuntimeError("figure 5 harnesses must stay uncalled")
    if getattr(harness, "name", "") in status["harnesses"]:
        raise RuntimeError(f"{harness.name} is a figure 5 harness and is not called")
    text = harness.complete(PLAN_PROMPT, response_schema=PLAN_SCHEMA)
    if not isinstance(text, str):
        raise HarnessCallError("plan reply is not text")
    return text


def make_plan(
    harness: Any, source_run: str = ""
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Ask once, validate, and return ``(plan, transcript)``. Nothing is retried."""
    before = int(getattr(harness, "call_count", 0))
    reply = ask_for_plan(harness)
    calls = int(getattr(harness, "call_count", before)) - before
    if calls < 1:
        raise RuntimeError("the harness was not called")
    steps, procedures = parse_plan_reply(reply)
    model = getattr(harness, "model_id", GEMINI_MODEL_ID)
    plan = {
        "mode": PLAN_MODE,
        "section": "3.1",
        "plan_source": PLAN_SOURCE_HARNESS,
        "forced": False,
        "model": model,
        "prompt_sha256": prompt_sha256(),
        "prompt_source": "ours",
        "vlm_calls": calls,
        "source_run": source_run,
        "harness_note": (
            "Gemini's plan; the paper's 60.75% bin number used Codex as the harness"
        ),
        "steps": [{"id": step, "section": STEP_SECTION[step]} for step in steps],
        "procedures": {step: procedures.get(step, "") for step in steps},
        "procedure_source": {
            step: "harness" if procedures.get(step) else "missing" for step in steps
        },
    }
    transcript = {
        "section": "3.1",
        "model": model,
        "prompt_sha256": plan["prompt_sha256"],
        "vlm_calls": calls,
        "prompt": PLAN_PROMPT,
        "reply": reply,
    }
    return plan, transcript


def load_plan(path: Path | str) -> dict[str, Any]:
    """A committed plan, checked again before the box acts on it.

    The same rules as on the runner, plus the prompt hash: a plan made with
    another prompt or schema is refused rather than run.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("mode") != PLAN_MODE:
        raise InvalidPlanError("not a harness plan file")
    if payload.get("plan_source") != PLAN_SOURCE_HARNESS or payload.get("forced"):
        raise InvalidPlanError("the plan was not chosen by the harness")
    if payload.get("prompt_sha256") != prompt_sha256():
        raise InvalidPlanError("the plan was made with a different prompt or schema")
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list) or not all(
        isinstance(item, dict) for item in raw_steps
    ):
        raise InvalidPlanError("steps must be a list of objects")
    payload["steps"] = [
        {"id": step, "section": STEP_SECTION[step]}
        for step in validate_plan([item.get("id") for item in raw_steps])
    ]
    procedures = payload.get("procedures")
    if not isinstance(procedures, dict) or not all(
        isinstance(value, str) for value in procedures.values()
    ):
        raise InvalidPlanError("procedures must map stages to text")
    return payload


def _procedure_with_run(step: str, body: str, entries: list[dict[str, Any]]) -> str:
    """The model's procedure, then what this run did, for a later conversion."""
    lines = [procedure_markdown(step, body).rstrip(), "", "## Run record", ""]
    for entry in entries:
        lines.append(
            f"- attempt {entry.get('attempt')}: {entry.get('decision')}"
            + (" (repair)" if entry.get("repair") else "")
            + (f", reason: {entry['reason']}" if entry.get("reason") else "")
        )
    return "\n".join(lines) + "\n"


def run_plan(
    out_dir: Path | str,
    plan: dict[str, Any],
    tools: dict[str, Tool] | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a checked plan in its own order; write each procedure after its stage.

    A stage with no registered tool is recorded as a reject ("no tool
    registered"), not skipped, so the ledger shows what the box can run.
    A ``<stage>_repair`` tool runs after a reject, as in ``run_agent_loop``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tools = tools or {}
    ctx = dict(context or {})
    steps = [item["id"] for item in plan["steps"]]
    record = {key: value for key, value in plan.items() if key != "procedures"}
    (out / "plan.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    folder = out / "procedures"
    folder.mkdir(parents=True, exist_ok=True)
    calls: list[str] = []
    entries: list[dict[str, Any]] = []
    for step in steps:
        tool = tools.get(step)
        if tool is None:
            result: dict[str, Any] = {
                "ok": False,
                "artifact": None,
                "reason": "no tool registered",
                "detail": {},
            }
        else:
            calls.append(step)
            result = tool(ctx)
        step_entries = [
            append_step(out, step, item, repair=is_repair)
            for item, is_repair in consider_repair(step, result, tools, ctx, calls)
        ]
        (folder / f"{step}.md").write_text(
            _procedure_with_run(step, plan["procedures"].get(step, ""), step_entries),
            encoding="utf-8",
        )
        entries.extend(step_entries)
    return {
        "out_dir": str(out),
        "steps": steps,
        "entries": entries,
        "skills": {step: STEP_SKILL[step] for step in steps},
    }


def main(argv: list[str] | None = None, harness: Any = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    plan_cmd = sub.add_parser("plan", help="runner: ask Gemini for the plan")
    plan_cmd.add_argument("--out-dir", type=Path, required=True)
    plan_cmd.add_argument("--source-run", default="")
    run_cmd = sub.add_parser("run", help="box: run a committed plan")
    run_cmd.add_argument("--plan", type=Path, required=True)
    run_cmd.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.stage == "plan":
        if args.source_run and not re.fullmatch(r"[0-9]{1,20}", args.source_run):
            parser.error("--source-run must be a run id (digits only)")
        if harness is None:
            from sceneagent.harness import selected_harness

            harness = selected_harness()
        plan, transcript = make_plan(harness, args.source_run)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for name, payload in (("plan.json", plan), ("transcript.json", transcript)):
            (args.out_dir / name).write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
        print(
            json.dumps(
                {
                    "steps": [s["id"] for s in plan["steps"]],
                    "vlm_calls": plan["vlm_calls"],
                }
            )
        )
        return 0
    summary = run_plan(args.out, load_plan(args.plan))
    print(json.dumps({"steps": summary["steps"], "entries": len(summary["entries"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
