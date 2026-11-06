"""CPU gate for sceneagent: registered markers, and the workflow that uses them.

A test that needs a GPU is marked ``gpu``. The CI pytest job collects
``-m "not gpu"`` instead of skipping that test inside the job.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_gpu_marker_is_registered_and_strict(pytestconfig):
    marker_lines = pytestconfig.getini("markers")
    names = [line.split(":", 1)[0].strip() for line in marker_lines]
    assert "gpu" in names
    assert "unit" in names
    assert "integration" in names
    assert "slow" in names
    assert pytestconfig.option.strict_markers is True


def test_sceneagent_ci_jobs_are_separate_and_pinned():
    workflow = (ROOT / ".github" / "workflows" / "sceneagent-ci.yml").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements-sceneagent-ci.txt").read_text(encoding="utf-8")
    for name in ("ruff:", "mypy:", "pytest-unit:"):
        assert f"\n  {name}" in workflow
    assert "requirements-sceneagent-ci.txt" in workflow
    assert '-m "not gpu"' in workflow
    assert "compileall" in workflow
    pinned = [
        line.strip()
        for line in requirements.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert pinned
    for line in pinned:
        assert "==" in line
        assert ">=" not in line
    for package in ("ruff==0.17.4", "mypy==2.4.1", "numpy==2.4.6", "pytest==9.2.0"):
        assert package in requirements
