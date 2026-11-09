# Session handoff (Cut 2)

Living status for whoever picks up Cut 2 next, whether a cloud session or a local one with SSH to the GPU
box. The plan is [`cut2-plan.md`](cut2-plan.md). Updated 2026-11-09.

## Where Cut 2 stands

| Phase | State |
|---|---|
| 0. Finish Cut 1 | Done. Cut 1 is in `main` (PR #28). Run4 passed on 59601312: 300/300 registered, 1.10 px. Gate code removed (PR #34, PR #35). |
| 1. Benchmark and scorer | Done in code (PR #33, KAN-31): `sceneagent/bin_benchmark.py` and `sceneagent/gc6d_fetch.py`. Scene 000091 checked in a dev VM: GT vs itself scores F1 1.0. Getting the scene onto the box happens on `bin_run`'s first run. |
| 2. Poses, splat, masks | Poses: PR #36 (`sceneagent/bin_run.py`, `--pipeline bin`). Code and tests done; the live GPU run is pending a box with the software. Splat and SF-15 masks not started. |
| 3. Association and grouping | Not started |
| 4. Meshes and first score | Not started (SF-29 / KAN-32 SAM 3D install) |

## The GPU box

| Instance | State | Notes |
|---|---|---|
| 58301075 | running | RTX 4090 24 GB, 500 GB disk, image `pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel` (Ubuntu 20.04, rsync 3.1.3). SSH `47.186.21.5:51808`, host key `SHA256:SaDE0hIpp4+P7JzCYt1STtqXe8HHxLMEEQgEaxQJvOA`. The CI key `simfoundry-gha-59601312` is authorized. **Missing `/root/miniforge3/envs/simfoundry`** (runs exit 127). |
| 59601312 | stopped ("Receiving copy…") | Has the full software stack. |

Code pins one instance id in about 10 files. PR #37 switches them to 58301075 and also fixes the payload
rsync for rsync versions older than 3.2.3.

## Open PRs

- #31: this plan and handoff (docs).
- #36: bin poses. Merge once a live run registers at least 11 of 13 views.
- #37: switch to 58301075, plus the rsync fix.

## Decisions already made

Bin score is the Cut 2 target. If COLMAP can't register the 13 photos, use the dataset's camera poses and
label the result. Masks use SAM vit_h. Physics comes from the VLM. `fitted_splat` and the
`mlp-physics-prior` branch are not touched. One PR per cohesive change; drafts until ready; GPU-touching
PRs get a gpu-smoke run on their own branch before merge.

## How to check locally (same as CI)

Python 3.11 with `requirements-cpu-ci.txt` and `requirements-sceneagent-ci.txt`:

    pytest tests/sceneagent -m "not gpu" --strict-markers
    python scripts/ci/run_cpu_tests.py
    ruff check sceneagent && ruff format --check sceneagent && mypy sceneagent

On the box, with SSH: `python -m sceneagent.bin_run --out <dir>` runs the bin poses directly.

## For Sandeep

- Delete the merged branches listed in the session, plus `claude/hostkey-scan` (the session can't
  delete branches).
