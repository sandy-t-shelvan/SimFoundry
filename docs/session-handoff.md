# Session handoff (Cut 2)

Living status for whoever picks up Cut 2 next, whether a cloud session or a local one with SSH to the GPU
box. The plan is [`cut2-plan.md`](cut2-plan.md). Updated 2026-11-10 (after PRs #37, #38 and #39 merged).

## Where Cut 2 stands

| Phase | State |
|---|---|
| 0. Finish Cut 1 | Done. Cut 1 is in `main` (PR #28). Run4 passed on 59601312: 300/300 registered, 1.10 px. |
| 1. Benchmark and scorer | Done (PR #33, KAN-31): `bin_benchmark.py`, `gc6d_fetch.py`, `metrics.py`. Scene 000091 is on the box at `/workspace/data/gc6d`. |
| 2. Poses, splat, masks | **Done** (rerun in the bin's own dir, PR #40). Poses: 13/13 views, 0.55 px ([run 42751476849](https://github.com/sandy-t-shelvan/SimFoundry/actions/runs/42751476849)). Splat: 214,976 Gaussians, held-out PSNR 23.07, SSIM 0.825, LPIPS 0.178 (same run). SF-15 masks: 13/13 views, 94 masks (5–11 per view) with a YCB category prompt ([run 42753282904](https://github.com/sandy-t-shelvan/SimFoundry/actions/runs/42753282904)). The splat merged with #38 (943,199 Gaussians, PSNR 30.04) was trained on 313 images, 300 of them run4's Fruits frames, and is void. |
| 3. Association and grouping | **Done** (PR #41, [run 42759126731](https://github.com/sandy-t-shelvan/SimFoundry/actions/runs/42759126731)). 94 masks → 25 objects (GT 20), 15 seen in ≥ 2 views. Lengths are in the paper's 0.915 m unit via a Depth Anything 3 scale (0.201 m per splat unit; GT cameras say 0.206, check only). Per-object Gaussian indices: `/workspace/sf4-smoke/bin/association/objects.npz`. |
| 4. Meshes and first score | **First score written** (PR #42). SAM 3D meshed all 25 objects ([run 42760087159](https://github.com/sandy-t-shelvan/SimFoundry/actions/runs/42760087159)). ICP plus one camera-based scene similarity: **F1@1cm 0.058, matched Chamfer 0.13 m** ([run 42762718646](https://github.com/sandy-t-shelvan/SimFoundry/actions/runs/42762718646), `docs/results/`). Open: the SAM 3D pose starts ~3.7 splat units off, so seeds win 20 of 25 fits. |

## Living docs: update these as the work goes

Sandeep asked (2026-11-10) that these follow every result, not only the cut plans:

1. [`docs/expansion-and-harvard-results.md`](expansion-and-harvard-results.md), the plan of record: the "Cut 2 board" and "Roadmap to the MVP" under Status.
2. [`docs/canvases/sceneagent-paper-gaps.canvas.tsx`](canvases/sceneagent-paper-gaps.canvas.tsx): the dated callout and each row's `ticket` field. A status word moves only when its Jira ticket is Accepted (comment 10222).
3. The paper-map artifact, [SceneAgent Paper Map](https://claude.ai/artifact/4mKRLY3RvCZ2LYuntQfiBx): the "Our code" and "To build" lines per block and the status line under the title.
4. Jira (project KAN): edit the tickets themselves as work lands, not only comments. That covers status (In Progress, In Review; Done stays with QA and PM), done criteria marked met or not met with proof links, and descriptions corrected when scope or box changes. The road to the MVP is epics KAN-33 (Cut 3), KAN-34 (Cut 4) and KAN-35 (Cut 5, MVP), with SF-30 to SF-46 as KAN-36 to KAN-52.

## The GPU box: 58301075

RTX 4090 24 GB, 128 cores, 503 GB RAM, Ubuntu 20.04, driver 595.99, about 170 GB free.
59601312 was deleted on 2026-11-10; 58301075 is the box again, with the layout it had on 2026-11-08.

| Need | Where on 58301075 (probes 42908214934, 42908974589) |
|---|---|
| Pinned interpreter | mamba env `sa`: `/root/.local/share/mamba/envs/sa/bin/python3` (numpy, scipy, trimesh, pillow; torch 2.14.1+cu130) |
| COLMAP 3.11.1 with CUDA | `/opt/colmap-cuda/bin/colmap` (`REMOTE_COLMAP_BIN`); `/usr/bin/colmap` is 3.6 without CUDA |
| nerfstudio 1.1.4, gsplat 1.0.0 | `/root/.local/share/mamba/envs/nerf/bin`, torch 2.1.2+cu121, `CUDA_HOME=/usr/local/cuda-12.1` |
| GroundingDINO | mamba env `gsam2` (torch 2.14.1+cu130), code and weights in `/root/SceneagentX/third_party/Grounded-SAM-2`, whose root must be on `PYTHONPATH` |
| SAM vit_h | `/workspace/models/sam/sam_vit_h_4b8939.pth` (SHA-256 checked); `segment-anything` installed into `/workspace/sf4-smoke/pydeps-masks` when missing |
| py7zr | `/workspace/sf4-smoke/pydeps` (`--target`; on `PYTHONPATH` only for `BIN_PIPELINES`) |
| Depth Anything 3 (metric) | the `sa` env, code from `/workspace/src/depth-anything-3/src` (read only), `DA3METRIC-LARGE` in the root HF cache; run offline by `bin_scale` |
| SAM 3D Objects | `/opt/conda/envs/sam3d10` (pytorch3d 0.7.8, torch 2.5.1+cu121), code and `checkpoints/hf/pipeline.yaml` in `/root/sam-3d-objects` (read only); needs `CONDA_PREFIX` set. The `sam3d` env lacks pytorch3d |
| Also present | COLMAP 4.1.1 with CUDA at `/root/.local/share/mamba/envs/nerf/bin/colmap` (has `global_mapper`) |

`/root/SceneagentX` and `/root/sam-3d-objects` belong to someone else's setup on the box: read from them,
never write. Our runs write only under `/workspace/sf4-smoke` and `/workspace/data/gc6d`.

## How GPU and SSH work (PR #37, merged 2026-11-10)

A cloud session has no SSH and no GPU. Every GPU step goes through GitHub Actions:

1. The session dispatches `.github/workflows/sceneagent-gpu-smoke.yml` with one fixed `pipeline` choice
   (`run4`, `bin`, `splat`, `masks`, `probe`). The workflow checker allows no other input.
2. The runner reads the instance with the Vast API (show only). It never starts, stops or resizes a box;
   a stopped box is a failed run.
3. It SSHes to the direct endpoint `public_ipaddr` plus `ports['22/tcp'].HostPort` with the fixed key
   `VAST_SSH_PRIVATE_KEY` and the pinned host key in `ci/vast-ssh-known-host` (no keyscan, no
   trust-on-first-use). For 59601312 that is `[47.186.21.5]:51823`.
4. It rsyncs `sceneagent/` only (plus Fruits.mp4), creating the remote folder with
   `--rsync-path="mkdir -p <repo> && rsync"` because older rsync lacks `--mkpath`.
5. The remote shell puts the pinned interpreter's bin and `REMOTE_COLMAP_BIN` first on `PATH`, deletes a
   stale `inventory.txt`, runs the pipeline, and the runner fetches the JSON evidence as an artifact.
6. `bin`, `splat`, `masks`, `association`, `sam3d` and `place` share `/workspace/sf4-smoke/bin` and its ledger
   (`vast_smoke.remote_out`); run4 keeps `/workspace/sf4-smoke/run`. Each bin pipeline is judged only on
   its own items (`BIN_PIPELINES`). Sharing run4's dir once let COLMAP pose run4's 300 Fruits frames
   beside the 13 bin views, so a bin poses run now clears its images, database and models and rejects a
   model whose input count is not 13. A failed run publishes only `manifest.json`, so failure reasons are
   written into the ledger entry.

**Switching to another box** (done twice so far, 59601312 → 58301075 → 59601312):

1. Sandeep starts it; sessions never start, stop or rent instances.
2. Attach the CI public key (`simfoundry-gha-59601312`, `SHA256:QhtGPtd0v6cjom2mHhkWpfibQit+QVIPmLB/u6XUPms`)
   with the Vast API `POST /api/v0/instances/<id>/ssh/`, with Sandeep's yes.
3. Pin its host key in `ci/vast-ssh-known-host` (read twice with ssh-keyscan from a runner) and swap the
   instance id in `vast_smoke.ALLOWED_INSTANCE_ID`, the workflow literal and concurrency group,
   `gpu_smoke_trigger`, `sf5_colmap_evidence.INSTANCE_ID` and `ci/gpu-run-request.schema.json`.
4. Run `pipeline=probe` and set `SIMFOUNDRY_PYTHON`, `REMOTE_COLMAP_BIN`, `bin_splat.NS_BIN`/`CUDA_HOME`
   and `bin_masks.MASK_PYTHON`/checkpoints from what it reports.

The probe's inventory is a public artifact: it lists tools, envs and model files only, never folder
trees, file contents or the environment. Earlier probe artifacts are kept (Sandeep, 2026-11-10).

## PRs

All merged: #37 (GPU route), #38 (splat), #39 (masks), #31 (this plan and handoff). Next work starts
from `main`.

## Decisions already made

Bin score is the Cut 2 target. Masks use SAM vit_h. Physics comes from the VLM. `fitted_splat` and the
`mlp-physics-prior` branch are not touched. The splat export uses a numpy-only safe-globals launcher for
torch ≥ 2.6 (option B), not a global weights_only switch. Masks are prompted with a YCB category
vocabulary (ours), because GraspClutter6D publishes object ids only. One PR per cohesive change; drafts until ready;
GPU-touching PRs get a gpu-smoke run on their own branch before merge.

## How to check locally (same as CI)

Python 3.11 with `requirements-cpu-ci.txt` and `requirements-sceneagent-ci.txt`:

    pytest tests/sceneagent -m "not gpu" --strict-markers
    python scripts/ci/run_cpu_tests.py
    ruff check sceneagent && ruff format --check sceneagent && mypy sceneagent

## For Sandeep

- Delete merged branches (the session can't): `claude/instance-58301075`, `claude/cut2-bin-splat`,
  `claude/cut2-bin-masks`, `claude/hostkey-scan`.

## Cut 2: what is left

SF-19 is built (PR #44): every object's eleven reviewer views are drawn, a retry is chosen from them and logged, and
the bin is scored again (F1@1cm 0.067, Chamfer 0.125 m). Open: separate the centred-start and retry gains (the
reviewer now scores both), and test camera consensus at kappa 2 (PR #45) against the floaters the views show. Both
need 59601312 running. Cut 3 planning can start from here.
