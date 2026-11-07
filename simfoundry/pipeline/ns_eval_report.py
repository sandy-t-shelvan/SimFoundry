# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report-only PSNR, SSIM, and LPIPS from nerfstudio's held-out eval split.

nerfstudio 1.1.5 ``ns-eval`` (``nerfstudio/scripts/eval.py``) accepts
``--load-config``, ``--output-path``, and an optional ``--render-output-path``.
It loads that config and averages ``get_average_eval_image_metrics`` over the
eval dataloader (``fixed_indices_eval_dataloader``). That dataloader is the
dataparser's held-out split. The CLI has no train-split switch, and this
module does not add one. ``--render-output-path`` is left unset so the stage
does not also write a render directory.

``eval_mode: all`` puts every image, including the train images, on that
dataloader. Those numbers are not a held-out score, so the command is not
run and the report records why.

The report never decides whether training or export succeeded. Callers keep
their existing verdict. Expected failures (missing binary, missing config,
non-held-out config, non-zero exit, timeout, malformed metrics) are written
into the JSON. A previous report is not read back and is not left in place
as a success.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import signal
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Bounds a stuck ``ns-eval`` so the stage can still finish. Not a quality bar.
NS_EVAL_TIMEOUT_S = 1800

_METRIC_KEYS = ("psnr", "ssim", "lpips")
# A dumped nerfstudio config.yml is a Python-object YAML document. Loading it
# would execute those tags. Match only the field line.
_EVAL_MODE_ALL = re.compile(
    r"""^\s*eval_mode\s*:\s*['"]?all['"]?\s*(?:#.*)?$""",
    re.IGNORECASE,
)
_STDERR_LIMIT = 500


def discard_stale_ns_eval_report(report_path: Path) -> None:
    """Remove a previous report before a new train starts.

    The file is the score of whatever checkpoint the last train left behind.
    Once this run starts training, that checkpoint is no longer the one this
    file would be claiming. Deleting it up front means a train that never
    reaches eval cannot be read as the previous success.
    """
    report_path.unlink(missing_ok=True)


def write_heldout_ns_eval_report(
    *,
    load_config: Path | None,
    report_path: Path,
    argv_prefix: Sequence[str],
    cwd: Path | None,
    env: Mapping[str, str] | None,
    timeout_s: float = NS_EVAL_TIMEOUT_S,
    run: Any = None,
) -> dict[str, Any]:
    """Write ``report_path`` from one ``ns-eval`` of ``load_config``.

    Does not raise. ``ok`` is true only when this invocation exits 0 and its
    JSON has finite ``results.psnr``, ``results.ssim``, and ``results.lpips``.
    Any other outcome stores null metrics and an ``error`` string, overwriting
    whatever ``report_path`` used to say.
    """
    if run is None:
        run = _run_bounded
    try:
        payload = _evaluate(
            load_config=load_config,
            report_path=report_path,
            argv_prefix=argv_prefix,
            cwd=cwd,
            env=env,
            timeout_s=timeout_s,
            run=run,
        )
    except Exception as exc:
        payload = _empty_report(load_config)
        payload["error"] = f"ns-eval report failed: {type(exc).__name__}: {exc}"
    _write_report(report_path, payload)
    if payload["ok"]:
        logger.info(
            "Held-out ns-eval PSNR=%s SSIM=%s LPIPS=%s written to %s",
            payload["psnr"],
            payload["ssim"],
            payload["lpips"],
            report_path,
        )
    else:
        logger.warning(
            "Held-out ns-eval report only; stage verdict unchanged: %s",
            payload["error"],
        )
    return payload


def _evaluate(
    *,
    load_config: Path | None,
    report_path: Path,
    argv_prefix: Sequence[str],
    cwd: Path | None,
    env: Mapping[str, str] | None,
    timeout_s: float,
    run: Any,
) -> dict[str, Any]:
    payload = _empty_report(load_config)
    if load_config is None or not load_config.is_file():
        payload["error"] = f"training config not found: {load_config}"
        return payload
    try:
        config_text = load_config.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        payload["error"] = f"could not read training config: {exc}"
        return payload
    if any(_EVAL_MODE_ALL.match(line) for line in config_text.splitlines()):
        payload["error"] = (
            "config eval_mode is all, so ns-eval would score every image "
            "rather than the held-out eval split"
        )
        return payload

    report_path.parent.mkdir(parents=True, exist_ok=True)
    raw_fd, raw_name = tempfile.mkstemp(
        prefix=".ns_eval_",
        suffix=".json",
        dir=report_path.parent,
    )
    os.close(raw_fd)
    raw_path = Path(raw_name)
    argv = [
        *argv_prefix,
        "ns-eval",
        "--load-config",
        str(load_config),
        "--output-path",
        str(raw_path),
    ]
    payload["argv"] = argv
    try:
        try:
            completed = run(
                argv,
                cwd=None if cwd is None else str(cwd),
                env=None if env is None else dict(env),
                timeout=timeout_s,
                check=False,
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired:
            payload["error"] = f"ns-eval timed out after {timeout_s:g}s"
            return payload
        except FileNotFoundError as exc:
            payload["error"] = f"ns-eval failed to start: {exc}"
            return payload
        returncode = getattr(completed, "returncode", None)
        payload["returncode"] = returncode
        if returncode != 0:
            detail = _tail(getattr(completed, "stderr", None)) or _tail(
                getattr(completed, "stdout", None)
            )
            payload["error"] = f"ns-eval exited {returncode}"
            if detail:
                payload["error"] = f"{payload['error']}: {detail}"
            return payload
        return _metrics_from_raw(payload, raw_path)
    finally:
        raw_path.unlink(missing_ok=True)


def _metrics_from_raw(payload: dict[str, Any], raw_path: Path) -> dict[str, Any]:
    """Copy the three metrics out of this invocation's JSON, or record why not.

    A non-zero exit never gets here, so a file left by an older run cannot
    satisfy this one. ``raw_path`` is a temp file created for this call.
    """
    if not raw_path.is_file():
        payload["error"] = "ns-eval exited 0 but wrote no JSON"
        return payload
    try:
        parsed = json.loads(raw_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        payload["error"] = f"ns-eval JSON is malformed: {exc}"
        return payload
    if not isinstance(parsed, dict):
        payload["error"] = "ns-eval JSON is malformed: top level is not an object"
        return payload
    results = parsed.get("results")
    if not isinstance(results, dict):
        payload["error"] = "ns-eval JSON has no results object"
        return payload
    metrics = {key: _as_finite(results.get(key)) for key in _METRIC_KEYS}
    missing = [key for key, value in metrics.items() if value is None]
    if missing:
        payload["error"] = (
            "ns-eval results missing finite psnr/ssim/lpips: " + ", ".join(missing)
        )
        return payload
    checkpoint = parsed.get("checkpoint")
    payload["ok"] = True
    payload["error"] = None
    payload["psnr"] = metrics["psnr"]
    payload["ssim"] = metrics["ssim"]
    payload["lpips"] = metrics["lpips"]
    payload["checkpoint"] = checkpoint if isinstance(checkpoint, str) else None
    return payload


def _empty_report(load_config: Path | None) -> dict[str, Any]:
    return {
        "report_only": True,
        "split": "eval",
        "ok": False,
        "psnr": None,
        "ssim": None,
        "lpips": None,
        "error": None,
        "load_config": None if load_config is None else str(load_config),
        "checkpoint": None,
        "returncode": None,
        "argv": None,
    }


def _as_finite(value: Any) -> float | None:
    """Return a finite float copied from ns-eval, or None if it is not one.

    Booleans are rejected because they are ints and are not metrics.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _tail(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    stripped = text.strip()
    if len(stripped) <= _STDERR_LIMIT:
        return stripped
    return stripped[-_STDERR_LIMIT:]


def _write_report(report_path: Path, payload: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=".ns_eval_report_",
        suffix=".json",
        dir=report_path.parent,
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(tmp_name, report_path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _run_bounded(
    argv: Sequence[str],
    *,
    cwd: str | None,
    env: dict[str, str] | None,
    timeout: float,
    check: bool,
    capture_output: bool,
    text: bool,
) -> subprocess.CompletedProcess[str]:
    """Run ``argv`` and kill its process group when ``timeout`` expires.

    ``subprocess.run`` kills only the direct child. Stage 5 starts ``ns-eval``
    under ``mamba run``, so a timeout has to signal the whole group or the
    eval keeps the GPU after the stage has moved on.
    """
    del check, text
    if not capture_output:
        raise ValueError("capture_output is required")
    proc = subprocess.Popen(
        list(argv),
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        stdout, stderr = proc.communicate()
        raise subprocess.TimeoutExpired(
            proc.args,
            timeout,
            output=stdout,
            stderr=stderr,
        )
    return subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)


def _kill_process_group(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except PermissionError:
        proc.kill()
