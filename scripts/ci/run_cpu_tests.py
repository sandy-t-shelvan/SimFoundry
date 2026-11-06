#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the lightweight CPU tests outside tests/sceneagent.

The skip list is ci/cpu-test-skips.txt. Network entries are marked
``network`` and dropped by ``-m "not gpu and not network"``. They are not
marked ``gpu``. Other entries are deselected or omitted. A skip result
fails the process: a skip is not a pass.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SKIP_FILE = ROOT / "ci" / "cpu-test-skips.txt"
TESTS_DIR = ROOT / "tests"


def _matches(nodeid: str, spec: str) -> bool:
    """True when nodeid is spec, a parametrized case of spec, or a child of spec."""
    return nodeid == spec or nodeid.startswith(spec + "[") or nodeid.startswith(spec + "::")


_KINDS = {"network", "torch", "skip"}


def load_skip_list(path: Path) -> list[tuple[str, str, str]]:
    """Return (kind, node, reason) rows.

    Kinds are ``network``, ``torch``, and ``skip``. ``network`` is a marker.
    It is not ``gpu``.
    """
    entries: list[tuple[str, str, str]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 3 or parts[0] not in _KINDS or not parts[1] or not parts[2]:
            raise SystemExit(
                f"{path}:{lineno}: expected 'network|torch|open3d|skip | node | reason'"
            )
        entries.append((parts[0], parts[1], parts[2]))
    if not entries:
        raise SystemExit(f"{path} has no entries")
    nodes = [node for _, node, _ in entries]
    if len(nodes) != len(set(nodes)):
        raise SystemExit(f"{path} lists a node more than once")
    return entries


class _CpuGate:
    """Apply the skip list during collection and fail if anything skips."""

    def __init__(self, entries: list[tuple[str, str, str]], omitted_files: int) -> None:
        self._node_entries = [entry for entry in entries if "::" in entry[1]]
        self._matched = {entry[1]: 0 for entry in self._node_entries}
        self._omitted_files = omitted_files
        self._deselected = 0

    def pytest_deselected(self, items: list[pytest.Item]) -> None:
        self._deselected += len(items)

    @pytest.hookimpl(tryfirst=True)
    def pytest_collection_modifyitems(self, config: pytest.Config, items: list[pytest.Item]) -> None:
        kept: list[pytest.Item] = []
        dropped: list[pytest.Item] = []
        for item in items:
            hit = False
            for kind, spec, _reason in self._node_entries:
                if not _matches(item.nodeid, spec):
                    continue
                self._matched[spec] += 1
                hit = True
                if kind == "network":
                    # `-m "not gpu and not network"` drops these. Not `gpu`.
                    item.add_marker(pytest.mark.network)
                    kept.append(item)
                else:
                    dropped.append(item)
                break
            if not hit:
                kept.append(item)
        missing = [spec for spec, count in self._matched.items() if count == 0]
        if missing:
            raise pytest.UsageError(
                "cpu skip list matched no collected tests:\n" + "\n".join(missing)
            )
        if dropped:
            config.hook.pytest_deselected(items=dropped)
            items[:] = kept

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        del session
        print(f"Deselected tests: {self._deselected}", flush=True)
        print(f"Omitted files: {self._omitted_files}", flush=True)
        print("See ci/cpu-test-skips.txt", flush=True)

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        skipped = []
        if reporter is not None:
            skipped = list(reporter.stats.get("skipped", []))
        if skipped:
            names = [report.nodeid for report in skipped]
            sys.stderr.write(
                "cpu gate: a skipped test is not a pass:\n" + "\n".join(names) + "\n"
            )
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


def _whole_file(node: str) -> bool:
    return "::" not in node


def main() -> int:
    """Print the file list and the skip list, then run pytest."""
    os.chdir(ROOT)
    existing = os.environ.get("PYTHONPATH", "")
    prefix = str(ROOT)
    if existing:
        prefix = prefix + os.pathsep + existing
    os.environ["PYTHONPATH"] = prefix

    entries = load_skip_list(SKIP_FILE)
    omitted = {node for _kind, node, _reason in entries if _whole_file(node)}
    discovered = sorted(path.relative_to(ROOT).as_posix() for path in TESTS_DIR.glob("test_*.py"))
    unknown = sorted(set(omitted) - set(discovered))
    if unknown:
        raise SystemExit("skip list names files that are not under tests/:\n" + "\n".join(unknown))
    files = [path for path in discovered if path not in omitted]

    print("CPU test files outside tests/sceneagent:")
    for path in files:
        print(f"  {path}")
    print(f"{len(files)} files")
    print()
    kind_counts: dict[str, int] = {}
    for kind, _node, _reason in entries:
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
    detail = ", ".join(f"{kind}={kind_counts[kind]}" for kind in sorted(kind_counts))
    print(f"Excluded entries by kind: {detail}")
    print("See ci/cpu-test-skips.txt")
    print()
    for kind in ("network", "torch", "skip"):
        rows = [entry for entry in entries if entry[0] == kind]
        if not rows:
            continue
        print(f"Not run ({kind}):")
        for _kind, node, reason in rows:
            print(f"  {node}")
            print(f"    {reason}")
        print()
    print("tests/sceneagent stays on the pytest-unit job.")
    print()

    args = [
        *files,
        "-v",
        "--tb=short",
        "-m",
        "not gpu and not network",
        "--strict-markers",
        "--override-ini",
        "addopts=--strict-markers",
    ]
    return int(pytest.main(args, plugins=[_CpuGate(entries, len(omitted))]))


if __name__ == "__main__":
    raise SystemExit(main())
