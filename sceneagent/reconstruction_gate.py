# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accept or reject a COLMAP sparse reconstruction.

The bar is PM KAN-4 comments 10236 and 10237 as amended by 10239. Sandeep
may overrule 10239; the four names below are the only thresholds, so a
later ruling changes them here and nowhere else. 10236(b) (points at least
100 times the registered-image count) was withdrawn. Points are reported
and are not a rule.

A model is accepted only when all three hold on the largest sparse model:

* registered images >= max(10, ceil(0.8 * input images))
* mean observations per image >= 30
* mean reprojection error <= 4.0 px

The largest model is the one with the most registered images. A tie is
broken by the points count only when that picks a single model. Anything
that cannot be ranked (no model, a failed or unparsable analyzer, a
registered count that will not parse, or a points value missing on a
registered tie) fails closed. A missing count is not treated as zero, and
a zero is not filled in so that some other model can win.

Everywhere else the selection and the three comparisons match collector
v9. The one exception is PM KAN-5 comment 10275: a double tie, the same
registered-image count and the same points count on more than one model.
v9 keeps the first directory name. This gate fails closed, reason
``ambiguous_largest_model``, in either directory order. A pass does not
depend on which name sorts first.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

# PM KAN-4 10236 / 10237 / 10239. Do not retune these to make a run pass.
MIN_REG_FLOOR = 10
REG_FRACTION = 0.8
MIN_OBS_PER_IMAGE = 30.0
MAX_REPROJ_PX = 4.0
# Exact COLMAP 3.11.1 model_analyzer label. The observations regex uses it.
OBS_LABEL = "Mean observations per image:"

_METRIC_KEYS = (
    "registered",
    "points",
    "mean_obs_per_image",
    "mean_reproj_err_px",
)
_FLOAT_KEYS = frozenset({"mean_obs_per_image", "mean_reproj_err_px"})
_RE = {
    "registered": re.compile(r"Registered images:\s*(\d+)\s*$", re.MULTILINE),
    "points": re.compile(r"\]\s*Points:\s*(\d+)\s*$", re.MULTILINE),
    "mean_obs_per_image": re.compile(
        rf"{re.escape(OBS_LABEL)}\s*([0-9.eE+-]+)\s*$", re.MULTILINE
    ),
    # No end anchor: COLMAP prints ``0.609572px`` with nothing after ``px``.
    "mean_reproj_err_px": re.compile(r"Mean reprojection error:\s*([0-9.eE+-]+)\s*px"),
}

AnalyzerRunner = Callable[[Path], dict[str, Any]]
EVIDENCE_DIRNAME = "reconstruction_gate"
EVIDENCE_JSON_NAME = "reconstruction_gate.json"
_SUMS_NAME = "SHA256SUMS"


def parse_analyzer(text: str) -> dict[str, Any]:
    """Parse model_analyzer text. A value stays None unless one match parses.

    Two copies of a label, a non-numeric token, or a non-finite float are
    unparsable. They are not coerced to zero.
    """
    parsed: dict[str, Any] = {}
    for key, pattern in _RE.items():
        found = pattern.findall(text or "")
        value: Any = None
        if len(found) == 1:
            try:
                value = float(found[0]) if key in _FLOAT_KEYS else int(found[0])
                if isinstance(value, float) and not math.isfinite(value):
                    value = None
            except ValueError:
                value = None
        parsed[key] = value
    return parsed


def run_model_analyzer(model: Path) -> dict[str, Any]:
    """Run ``colmap model_analyzer`` on one sparse model directory.

    A missing binary, a non-zero exit, a timeout, or an OS error is a
    result dict, not a successful parse. Callers fail closed on that.
    """
    binary = shutil.which("colmap")
    if not binary:
        return {
            "returncode": None,
            "stdout": "",
            "stderr": "colmap is not on PATH",
        }
    try:
        completed = subprocess.run(
            [binary, "model_analyzer", "--path", str(model)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
            env=_analyzer_env(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "returncode": None,
            "stdout": "",
            "stderr": f"analyzer error: {exc!r}",
        }
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout or "",
        "stderr": completed.stderr or "",
    }


def _analyzer_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    return env


def _empty_metrics() -> dict[str, Any]:
    return {key: None for key in _METRIC_KEYS}


def analyze_models(
    sparse_dir: Path, evidence_dir: Path, runner: AnalyzerRunner
) -> list[dict[str, Any]]:
    """Run ``runner`` on every subdirectory of ``sparse_dir``.

    Subdirectory order is sorted so a later tie sees a stable list. A
    runner that raises is recorded as a failed analyzer, same as a
    non-zero exit: the text is kept and none of its numbers are ranked.
    """
    models: list[dict[str, Any]] = []
    if not sparse_dir.is_dir():
        return models
    children = sorted(path for path in sparse_dir.iterdir() if path.is_dir())
    for model in children:
        try:
            result = runner(model)
        except (OSError, subprocess.SubprocessError) as exc:
            result = {
                "returncode": None,
                "stdout": "",
                "stderr": f"analyzer error: {exc!r}",
            }
        stdout = str(result.get("stdout") or "")
        stderr = str(result.get("stderr") or "")
        returncode = result.get("returncode")
        transcript = evidence_dir / f"model_analyzer_{model.name}.txt"
        transcript.write_text(
            f"# command: model_analyzer --path {model}\n"
            f"# returncode: {returncode}\n"
            f"# ---- stdout ----\n{stdout}\n"
            f"# ---- stderr ----\n{stderr}\n",
            encoding="utf-8",
        )
        if returncode == 0:
            values = parse_analyzer(stdout + "\n" + stderr)
        else:
            values = _empty_metrics()
        models.append(
            {
                "model": model.name,
                "path": str(model),
                "analyzer_rc": returncode,
                "analyzer_file": transcript.name,
                **values,
            }
        )
    return models


def pick_largest(
    models: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    """Return ``(largest, error)``. Nothing is ranked with a sentinel.

    Registered images decide. Points break a tie only when one tied model
    has strictly more points than the others. A double tie, the same
    registered count and the same points count, has no winner. v9's
    ``max`` keeps the first directory name. PM KAN-5 10275 rejects that
    fallback: the reason is ``ambiguous_largest_model`` whichever name
    sorts first. That is the only departure from v9.
    """
    if not models:
        return None, "no sparse model"
    failed = [model["model"] for model in models if model.get("analyzer_rc") != 0]
    if failed:
        return None, f"model_analyzer failed on sub-model(s) {failed}"
    unparsed = [model["model"] for model in models if model["registered"] is None]
    if unparsed:
        return None, f"registered missing/unparsable on sub-model(s) {unparsed}"
    top = max(model["registered"] for model in models)
    tied = [model for model in models if model["registered"] == top]
    if len(tied) == 1:
        return tied[0], None
    missing_points = [model["model"] for model in tied if model["points"] is None]
    if missing_points:
        return None, (
            f"registered tie at {top}; points missing/unparsable on {missing_points}"
        )
    best_points = max(model["points"] for model in tied)
    winners = [model for model in tied if model["points"] == best_points]
    if len(winners) != 1:
        # Directory order is not a rule. v9 would return the first name.
        return None, (
            "ambiguous_largest_model: "
            f"registered tie at {top}; points tied at {best_points} with no rule"
        )
    return winners[0], None


def _counted_inputs(inputs: Any) -> int | None:
    """Positive int image count. Anything else is missing, including bool."""
    if isinstance(inputs, bool) or not isinstance(inputs, int) or inputs <= 0:
        return None
    return inputs


def quality_bar(
    largest: dict[str, Any] | None,
    inputs: Any,
    selection_error: str | None = None,
) -> dict[str, Any]:
    """Score ``largest`` against the three rules. Points are informational."""
    counted = _counted_inputs(inputs)
    reasons: list[str] = []
    if largest is None:
        reasons.append(selection_error or "no sparse model")
    if counted is None:
        reasons.append("input frame count missing/ambiguous")
    registered = None if largest is None else largest.get("registered")
    points = None if largest is None else largest.get("points")
    reprojection = None if largest is None else largest.get("mean_reproj_err_px")
    observations = None if largest is None else largest.get("mean_obs_per_image")
    if counted is None:
        registered_threshold = None
    else:
        scaled = math.ceil(REG_FRACTION * counted)
        registered_threshold = max(MIN_REG_FLOOR, scaled)
    registered_ok = (
        registered is not None
        and registered_threshold is not None
        and registered >= registered_threshold
    )
    observations_ok = observations is not None and observations >= MIN_OBS_PER_IMAGE
    reprojection_ok = reprojection is not None and reprojection <= MAX_REPROJ_PX
    for name, ok in (
        ("registered", registered_ok),
        ("mean_obs_per_image", observations_ok),
        ("mean_reproj_err_px", reprojection_ok),
    ):
        if not ok:
            reasons.append(f"{name} fails or missing")
    passed = bool(
        largest is not None
        and counted is not None
        and registered_ok
        and observations_ok
        and reprojection_ok
    )
    return {
        "model": None if largest is None else largest["model"],
        "inputs": counted,
        "registered": {
            "value": registered,
            "threshold": registered_threshold,
            "rule": ">= max(10, ceil(0.8*inputs))",
            "pass": registered_ok,
        },
        "mean_obs_per_image": {
            "value": observations,
            "threshold": MIN_OBS_PER_IMAGE,
            "rule": ">= 30",
            "label": OBS_LABEL,
            "pass": observations_ok,
        },
        "points_info_only": points,
        "mean_reproj_err_px": {
            "value": reprojection,
            "threshold": MAX_REPROJ_PX,
            "rule": "<= 4.0",
            "pass": reprojection_ok,
        },
        "pass": passed,
        "fail_reasons": reasons,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sha256sums(evidence_dir: Path) -> Path:
    """Write ``SHA256SUMS`` for every other file under ``evidence_dir``.

    Two spaces separate the digest and the relative path, matching the
    collector's evidence sums. The sums file itself is not an input.
    """
    destination = evidence_dir / _SUMS_NAME
    lines = [
        f"{_sha256(path)}  {path.relative_to(evidence_dir).as_posix()}"
        for path in sorted(evidence_dir.rglob("*"))
        if path.is_file() and path != destination
    ]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def _reset_evidence(evidence_dir: Path) -> None:
    if evidence_dir.exists():
        shutil.rmtree(evidence_dir)
    evidence_dir.mkdir(parents=True)


def measured_summary(bar: dict[str, Any], models: int) -> str:
    """The measured values beside their limits, so a failed run's manifest shows them."""
    registered = bar.get("registered") or {}
    observations = bar.get("mean_obs_per_image") or {}
    reprojection = bar.get("mean_reproj_err_px") or {}
    return (
        f" (largest of {models} models: registered {registered.get('value')} of "
        f"{bar.get('inputs')} inputs, needs >= {registered.get('threshold')}; "
        f"observations per image {observations.get('value')}, needs >= "
        f"{observations.get('threshold')}; reprojection {reprojection.get('value')} px, "
        f"needs <= {reprojection.get('threshold')})"
    )


def accept_reconstruction(
    sparse_dir: Path,
    input_images: Any,
    evidence_dir: Path,
    runner: AnalyzerRunner | None = None,
) -> dict[str, Any]:
    """Evaluate every model under ``sparse_dir`` and write the evidence.

    ``runner`` defaults to :func:`run_model_analyzer`. Tests pass a runner
    so this does not need a COLMAP binary. ``pass`` is true only when the
    largest model clears every rule. The evidence directory is replaced on
    each call.
    """
    if runner is None:
        runner = run_model_analyzer
    _reset_evidence(evidence_dir)
    models = analyze_models(sparse_dir, evidence_dir, runner)
    largest, selection_error = pick_largest(models)
    bar = quality_bar(largest, input_images, selection_error)
    document: dict[str, Any] = {
        "inputs": bar["inputs"],
        "sparse_models": models,
        "largest_model": None if largest is None else largest["model"],
        "chosen_model": None if largest is None else largest["model"],
        "quality_bar": bar,
        "verdict": "pass" if bar["pass"] else "fail",
        "pass": bar["pass"],
        "fail_reasons": list(bar["fail_reasons"]),
        "selection_error": selection_error,
    }
    text = json.dumps(document, indent=2) + "\n"
    (evidence_dir / EVIDENCE_JSON_NAME).write_text(text, encoding="utf-8")
    write_sha256sums(evidence_dir)
    if document["pass"]:
        reason = ""
    else:
        reason = (
            "reconstruction not accepted: "
            + "; ".join(document["fail_reasons"])
            + measured_summary(bar, len(models))
        )
    return {
        "pass": document["pass"],
        "model": document["chosen_model"],
        "reason": reason,
        "document": document,
        "evidence_dir": str(evidence_dir),
    }
