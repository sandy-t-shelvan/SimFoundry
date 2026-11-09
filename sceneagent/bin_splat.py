# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cut 2 phase 2: a full-scene splatfacto splat of GraspClutter6D 000091.

Reuses the bin poses when ``bin_evidence.json`` in ``--out`` already passes and
its model directory exists; otherwise runs :func:`sceneagent.bin_run.run_bin_poses`
first. The splat follows pipeline A's ``2c_train_bg_splat.py`` (train, held-out
``ns-eval``, ``ns-export gaussian-splat``) but starts from those COLMAP poses
with ``ns-process-data images --skip-colmap`` instead of re-posing a video.

The ``ns-*`` tools come from :data:`NS_BIN`, not from the pinned interpreter's
env, and run without this process's ``PYTHONPATH`` so their own env imports
cleanly. ``ns-eval`` is report-only, as in pipeline A. The ledger accepts the
splat when ``ns-export`` leaves a PLY with at least one Gaussian.

    python -m sceneagent.bin_splat --out /workspace/sf4-smoke/run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from simfoundry.pipeline.ns_eval_report import write_heldout_ns_eval_report

from sceneagent.bin_run import DATA_DIR, SCENE, SOURCE, run_bin_poses

# nerfstudio 1.1.4 and gsplat 1.0.0 on instance 58301075 (probe run 42693162920).
NS_BIN = "/root/.local/share/mamba/envs/nerf/bin"
# That env's torch is cu121; gsplat compiles its kernels against this toolkit.
CUDA_HOME = "/usr/local/cuda-12.1"
METHOD = "splatfacto"
MAX_NUM_ITERATIONS = 30000
STEP_TIMEOUT_S = {"process-data": 900, "train": 3600, "export": 900}
SPLAT_DIRNAME = "splat"
ITEM = "full-scene splat"

Runner = Callable[..., Any]


def ns_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for the ``ns-*`` tools: their bin and CUDA first, no PYTHONPATH."""
    env = dict(os.environ if base is None else base)
    env.pop("PYTHONPATH", None)
    env["PATH"] = os.pathsep.join(
        [NS_BIN, f"{CUDA_HOME}/bin", env.get("PATH", "")]
    ).rstrip(os.pathsep)
    env["CUDA_HOME"] = CUDA_HOME
    env["TORCH_DYNAMO_DISABLE"] = "1"
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    return env


def poses_ready(out: Path) -> Path | None:
    """The passing model directory from an earlier bin run in ``out``, if any.

    The ledger must still be the bin's: another pipeline sharing ``out`` (run4)
    rewrites it, and its entries are not this scene's.
    """
    try:
        evidence = json.loads((out / "bin_evidence.json").read_text(encoding="utf-8"))
        ledger = json.loads((out / "ledger.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if ledger.get("scene_name") != f"{SOURCE} {SCENE}":
        return None
    bar = evidence.get("quality_bar") or {}
    if not bar.get("pass") or bar.get("model") is None:
        return None
    model = out / "sparse" / str(bar["model"])
    return model if (model / "cameras.bin").is_file() else None


def splat_commands(out: Path, model: Path) -> list[tuple[str, list[str]]]:
    """Process the posed images, then train. Export follows once a config exists."""
    splat = out / SPLAT_DIRNAME
    processed = splat / "processed"
    return [
        (
            "process-data",
            [
                "ns-process-data",
                "images",
                "--data",
                str((out / "images").resolve()),
                "--output-dir",
                str(processed.resolve()),
                "--skip-colmap",
                "--colmap-model-path",
                str(model.resolve()),
            ],
        ),
        (
            "train",
            [
                "ns-train",
                METHOD,
                "--max-num-iterations",
                str(MAX_NUM_ITERATIONS),
                "--data",
                str(processed.resolve()),
                "--output-dir",
                str((splat / "outputs").resolve()),
                "--viewer.quit-on-train-completion",
                "True",
            ],
        ),
    ]


def ply_vertex_count(path: Path) -> int:
    """``element vertex N`` from a PLY header; 0 when absent or unreadable."""
    try:
        with path.open("rb") as handle:
            for _ in range(64):
                line = handle.readline()
                if not line or line.strip() == b"end_header":
                    break
                parts = line.split()
                if parts[:2] == [b"element", b"vertex"] and len(parts) == 3:
                    return int(parts[2])
    except (OSError, ValueError):
        pass
    return 0


def _run_step(
    name: str, argv: Sequence[str], cwd: Path, env: dict[str, str], runner: Runner
) -> dict[str, Any]:
    started = time.monotonic()
    record: dict[str, Any] = {"step": name, "argv": list(argv)}
    try:
        done = runner(
            list(argv),
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=STEP_TIMEOUT_S[name],
            check=False,
        )
        record["returncode"] = done.returncode
        tail = (done.stderr or "") or (done.stdout or "")
        record["output_tail"] = tail[-2000:]
    except subprocess.TimeoutExpired:
        record["returncode"] = 124
        record["output_tail"] = f"timed out after {STEP_TIMEOUT_S[name]} s"
    except OSError as exc:
        record["returncode"] = 127
        record["output_tail"] = f"{type(exc).__name__}: {exc}"
    record["seconds"] = round(time.monotonic() - started, 1)
    return record


def _latest_config(outputs: Path) -> Path | None:
    configs = sorted(
        outputs.glob("**/config.yml"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return configs[0] if configs else None


def train_splat(
    out: Path, model: Path, *, runner: Runner = subprocess.run
) -> dict[str, Any]:
    """Run the ns steps in order and stop at the first failure."""
    splat = out / SPLAT_DIRNAME
    splat.mkdir(parents=True, exist_ok=True)
    env = ns_environment()
    evidence: dict[str, Any] = {
        "method": METHOD,
        "max_num_iterations": MAX_NUM_ITERATIONS,
        "ns_bin": NS_BIN,
        "colmap_model": str(model),
        "steps": [],
    }
    for name, argv in splat_commands(out, model):
        record = _run_step(name, argv, splat, env, runner)
        evidence["steps"].append(record)
        if record["returncode"] != 0:
            evidence["failed_step"] = name
            return evidence
    config = _latest_config(splat / "outputs")
    evidence["config"] = None if config is None else str(config)
    evidence["ns_eval"] = write_heldout_ns_eval_report(
        load_config=config,
        report_path=out / "ns_eval.json",
        argv_prefix=[],
        cwd=splat,
        env=env,
        run=runner,
    )
    if config is None:
        evidence["failed_step"] = "train"
        return evidence
    export = splat / "export"
    record = _run_step(
        "export",
        [
            "ns-export",
            "gaussian-splat",
            "--load-config",
            str(config.resolve()),
            "--output-dir",
            str(export.resolve()),
        ],
        splat,
        env,
        runner,
    )
    evidence["steps"].append(record)
    plys = sorted(export.glob("*.ply"))
    evidence["ply"] = str(plys[0]) if plys else None
    evidence["gaussians"] = ply_vertex_count(plys[0]) if plys else 0
    if record["returncode"] != 0 or not evidence["gaussians"]:
        evidence["failed_step"] = "export"
    return evidence


def run_bin_splat(
    out: Path | str,
    data: Path | str = DATA_DIR,
    *,
    poses: Callable[[Path, Path], int] = run_bin_poses,
    runner: Runner = subprocess.run,
) -> int:
    """Poses (reused or rerun), then the splat. Return 0 only when both pass."""
    out, data = Path(out), Path(data)
    model = poses_ready(out)
    reused = model is not None
    if model is None:
        poses(out, data)
        model = poses_ready(out)
    ledger_path = out / "ledger.json"
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        ledger = {"entries": []}
    entries = [e for e in ledger.get("entries", []) if e.get("item") != ITEM]
    evidence: dict[str, Any] = {"scene": SCENE, "poses_reused": reused}
    if model is None:
        entry: dict[str, Any] = {
            "item": ITEM,
            "decision": "reject",
            "reason": "bin poses did not pass",
        }
    else:
        evidence.update(train_splat(out, model, runner=runner))
        failed = evidence.get("failed_step")
        entry = {"item": ITEM, "method": METHOD}
        if failed:
            entry.update(decision="reject", failed_step=failed)
        else:
            entry.update(
                decision="accept",
                use_gpu=True,
                gaussians=evidence["gaussians"],
                psnr=evidence["ns_eval"].get("psnr"),
            )
    entries.append(entry)
    ledger.update(scene_name=f"{SOURCE} {SCENE}", source=SOURCE, entries=entries)
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    (out / "splat_evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    passed = all(e.get("decision") == "accept" for e in entries) and len(entries) == 3
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path(DATA_DIR))
    args = parser.parse_args(argv)
    return run_bin_splat(args.out, args.data)


if __name__ == "__main__":
    sys.exit(main())
