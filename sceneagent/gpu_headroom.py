# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed free-memory check for one COLMAP or splatfacto stage.

6144 MiB is the COLMAP floor and 16384 MiB is the splatfacto floor. Both
are provisional budgets, not a measured peak and not a reservation. The
query reads aggregate GPU 0 ``index``, ``memory.total``, and
``memory.free`` once. A non-zero status, an OS error, a sample that is
not one finite GPU 0 row, or a free value under the floor raises
:class:`GpuHeadroomError` carrying that return code and stderr. The query
is not repeated, and nothing else on the machine is signalled.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from typing import Any

COLMAP_MIN_FREE_MIB = 6144
SPLATFACTO_MIN_FREE_MIB = 16384
_MIN_FREE_MIB = {
    "colmap": COLMAP_MIN_FREE_MIB,
    "splatfacto": SPLATFACTO_MIN_FREE_MIB,
}
# No utilization column and no per-process query.
NVIDIA_SMI_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,memory.total,memory.free",
    "--format=csv,noheader,nounits",
]


class GpuCommandResult:
    """Stdout and stderr from one ``nvidia-smi`` invocation."""

    def __init__(self, returncode: int | None, stdout: str, stderr: str) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class GpuHeadroomError(RuntimeError):
    """The one allowed memory query failed or the floor was not met."""

    def __init__(
        self,
        message: str,
        *,
        returncode: int | None,
        stdout: str,
        stderr: str,
        stage: str,
        sample: dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.stage = stage
        self.sample = sample
        self.retries = 0


def query_nvidia_smi(argv: list[str] | None = None) -> GpuCommandResult:
    """Run the aggregate query once. An OS error is a result, not a retry."""
    command = list(NVIDIA_SMI_QUERY if argv is None else argv)
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return GpuCommandResult(None, "", f"{type(exc).__name__}: {exc}")
    return GpuCommandResult(
        completed.returncode, completed.stdout or "", completed.stderr or ""
    )


def parse_gpu_sample(text: str) -> dict[str, int]:
    """Parse one ``index, total, free`` row. Anything else is refused.

    ``inf`` and ``nan`` do not parse as integers. Two GPUs, a missing
    column, or a free value outside ``[0, total]`` is not treated as zero.
    """
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("requires exactly one GPU")
    parts = [part.strip() for part in lines[0].split(",")]
    if len(parts) != 3:
        raise ValueError("invalid aggregate memory sample")
    try:
        index = int(parts[0])
        total = int(parts[1])
        free = int(parts[2])
    except ValueError as exc:
        raise ValueError("invalid aggregate memory sample") from exc
    if index != 0 or total <= 0 or free < 0 or free > total:
        raise ValueError("invalid aggregate memory sample")
    return {"gpu_index": index, "total_mib": total, "free_mib": free}


def require_gpu_headroom(
    stage: str,
    *,
    query: Callable[[list[str]], GpuCommandResult] | None = None,
    retries: int = 0,
) -> dict[str, Any]:
    """Require free memory for ``stage``. ``retries`` other than 0 is refused.

    ``query`` defaults to :func:`query_nvidia_smi` and is called once.
    ``stage`` is ``colmap`` (6144 MiB) or ``splatfacto`` (16384 MiB).
    """
    if retries != 0:
        raise GpuHeadroomError(
            "retries are not allowed",
            returncode=None,
            stdout="",
            stderr="",
            stage=stage,
            sample={"retries": retries},
        )
    if stage not in _MIN_FREE_MIB:
        raise GpuHeadroomError(
            f"unknown GPU stage {stage}",
            returncode=None,
            stdout="",
            stderr="",
            stage=stage,
            sample={"retries": 0},
        )
    minimum = _MIN_FREE_MIB[stage]
    runner = query_nvidia_smi if query is None else query
    result = runner(list(NVIDIA_SMI_QUERY))
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    returncode = result.returncode
    if returncode != 0:
        detail = stderr.strip() or stdout.strip() or "nvidia-smi failed"
        sample = {
            "stage": stage,
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "minimum_required_free_mib": minimum,
            "retries": 0,
        }
        raise GpuHeadroomError(
            f"nvidia-smi failed for {stage}: rc={returncode}: {detail}",
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            stage=stage,
            sample=sample,
        )
    try:
        parsed = parse_gpu_sample(stdout)
    except ValueError as exc:
        sample = {
            "stage": stage,
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "minimum_required_free_mib": minimum,
            "retries": 0,
        }
        raise GpuHeadroomError(
            f"{exc}; rc={returncode}; stderr={stderr.strip()}",
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            stage=stage,
            sample=sample,
        ) from exc
    sample = {
        **parsed,
        "stage": stage,
        "minimum_required_free_mib": minimum,
        "returncode": returncode,
        "stderr": stderr,
        "retries": 0,
    }
    if parsed["free_mib"] < minimum:
        raise GpuHeadroomError(
            (
                f"insufficient free GPU memory for {stage}: "
                f"{parsed['free_mib']} MiB < {minimum} MiB; no retry"
            ),
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            stage=stage,
            sample=sample,
        )
    sample["passed"] = True
    return sample
