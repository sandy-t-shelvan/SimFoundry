# Session handoff (Cut 2)

Living status for whoever picks up Cut 2 next, whether a cloud session or a local one with SSH to the GPU
box. The plan is [`cut2-plan.md`](cut2-plan.md). Updated 2026-11-09.

## Where Cut 2 stands

| Phase | State |
|---|---|
| 0. Finish Cut 1 | Done. Cut 1 is in `main` (PR #28). Run4 passed on 59601312: 300/300 registered, 1.10 px. Gate code removed (PR #34, PR #35). |
| 1. Benchmark and scorer | Done in code (PR #33, KAN-31): `sceneagent/bin_benchmark.py` and `sceneagent/gc6d_fetch.py`. Scene 000091 is now on the box at `/workspace/data/gc6d`. |
| 2. Poses, splat, masks | **Poses pass live** on 58301075: [run 42690954531](https://github.com/sandy-t-shelvan/SimFoundry/actions/runs/42690954531), 13/13 views registered, 0.55 px, CUDA COLMAP 3.11.1 (code PR #36 merged; box pins PR #37). **Splat** (PR #38, `--pipeline splat`) and **SF-15 masks** (PR #39, `--pipeline masks`): code and tests done, first live runs in progress. |
| 3. Association and grouping | Not started. Needs the splat PLY and the masks from phase 2. |
| 4. Meshes and first score | Not started. SAM 3D is already installed on the box (`/root/sam-3d-objects` with checkpoints, envs `sam3d`/`sam3d10`). |

## The GPU box: 58301075

RTX 4090 24 GB, 128 cores, 503 GB RAM, 500 GB disk (about 170 GB free), Ubuntu 20.04, rsync 3.1.3.
SSH `47.186.21.5:51808`, host key `SHA256:SaDE0hIpp4+P7JzCYt1STtqXe8HHxLMEEQgEaxQJvOA`; the CI key
`simfoundry-gha-59601312` is authorized. 59601312 is stopped.

There is no `/root/miniforge3/envs/simfoundry` on this box. What the pipelines use instead (found by
`--pipeline probe`, PR #37):

| Need | Where |
|---|---|
| Pinned interpreter (numpy, scipy, trimesh, pillow) | mamba env `sa`: `/root/.local/share/mamba/envs/sa/bin/python3` (Python 3.10) |
| COLMAP 3.11.1 with CUDA | `/opt/colmap-cuda/bin/colmap` (`/usr/bin/colmap` is 3.6 without CUDA) |
| nerfstudio 1.1.4, gsplat 1.0.0 | mamba env `nerf` (`ns-*` in its bin), torch 2.1.2 cu121, CUDA 12.1 toolkit |
| GroundingDINO SwinT-OGC, SAM 2.1 | mamba env `gsam2`, code and weights in `/root/SceneagentX/third_party/Grounded-SAM-2` |
| SAM vit_h | downloaded by `bin_masks` to `/workspace/models/sam/`; `segment-anything` in `/workspace/sf4-smoke/pydeps-gsam2` |
| py7zr | `/workspace/sf4-smoke/pydeps` (`--target`, not in `sa`) |
| Also present | envs `2dgs`, `trellis`, `partfield`, `physgauss`, `sam3d`, `sam3d10`; COLMAP 4.1.1 CUDA in `nerf` |

Shared folders on the box that belong to other work and are not touched: `/workspace/sceneagent-simfoundry`
(MLP physics-prior data) and `/root/SceneagentX`. Our runs write only under `/workspace/sf4-smoke` and
`/workspace/models/sam`.

## How a GPU run happens from a cloud session

The session has no SSH. It dispatches `.github/workflows/sceneagent-gpu-smoke.yml` with one fixed
`pipeline` choice (`run4`, `bin`, `splat`, `masks`, `probe`). The runner SSHes in with the fixed key,
rsyncs `sceneagent/` (only), runs the pipeline, fetches the JSON evidence and uploads it as an artifact.
All pipelines share `/workspace/sf4-smoke/run`, so `splat` and `masks` reuse the bin's images and poses.

## Open PRs

- #31: this plan and handoff (docs).
- #37: switch to 58301075, probe, box stack. Ready; Cursor's router asked for a human review because it
  changes SSH host trust, so it was not self-merged.
- #38 (on #37): bin splat. #39 (on #38): SF-15 masks. Drafts until their live runs pass.

## Decisions already made

Bin score is the Cut 2 target. If COLMAP can't register the 13 photos, use the dataset's camera poses and
label the result (not needed: 13/13 registered). Masks use SAM vit_h. Physics comes from the VLM.
`fitted_splat` and the `mlp-physics-prior` branch are not touched. One PR per cohesive change; drafts
until ready; GPU-touching PRs get a gpu-smoke run on their own branch before merge.

## How to check locally (same as CI)

Python 3.11 with `requirements-cpu-ci.txt` and `requirements-sceneagent-ci.txt`:

    pytest tests/sceneagent -m "not gpu" --strict-markers
    python scripts/ci/run_cpu_tests.py
    ruff check sceneagent && ruff format --check sceneagent && mypy sceneagent

On the box, with SSH, from the synced repo with `PYTHONPATH` set to it:
`python -m sceneagent.bin_run --out <dir>` (poses), `sceneagent.bin_splat`, `sceneagent.bin_masks`.

## For Sandeep

- Review and merge #37 (host key and instance pins), then #38 and #39 once their runs pass.
- Delete the merged branches listed in the session, plus `claude/hostkey-scan` (the session can't
  delete branches).
