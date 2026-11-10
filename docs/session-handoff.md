# Session handoff (Cut 2)

Living status for whoever picks up Cut 2 next, whether a cloud session or a local one with SSH to the GPU
box. The plan is [`cut2-plan.md`](cut2-plan.md). Updated 2026-11-10.

## Where Cut 2 stands

| Phase | State |
|---|---|
| 0. Finish Cut 1 | Done. Cut 1 is in `main` (PR #28). Run4 passed on 59601312: 300/300 registered, 1.10 px. |
| 1. Benchmark and scorer | Done (PR #33, KAN-31): `bin_benchmark.py`, `gc6d_fetch.py`, `metrics.py`. Scene 000091 is on the box at `/workspace/data/gc6d`. |
| 2. Poses, splat, masks | **Poses** pass (PR #36 merged): 13/13 views, 0.55 px. **Splat** passes on 59601312 (PR #38): 943,199 Gaussians, held-out PSNR 30.04, SSIM 0.900, LPIPS 0.257. **SF-15 masks** pass 13/13 views (PR #39, draft) but keep only 2–3 masks per view out of 20 objects; coverage fix is the open decision. |
| 3. Association and grouping | Not started. `association.py` has every constant from §3.3; the runner over a whole splat is next. |
| 4. Meshes and first score | Not started. The `sam3d-objects` env is on 59601312. |

## The GPU box: 59601312

RTX 4090 24 GB, 128 cores, 503 GB RAM, 600 GB disk (about 314 GB free), Ubuntu 24.04, image
`nvidia/cuda:13.2.0-devel-ubuntu24.04`. 58301075 is stopped, its disk kept, and not used.

| Need | Where on 59601312 (probe 42713577064, 42714221807) |
|---|---|
| Pinned interpreter | `/root/miniforge3/envs/simfoundry/bin/python3` (Python 3.11, numpy, scipy, trimesh, pillow) |
| COLMAP 3.11.1 with CUDA | `/usr/local/bin/colmap` (`REMOTE_COLMAP_BIN`) |
| nerfstudio 1.1.5, gsplat 1.5.3 | `/root/miniforge3/envs/nerfstudio_simfoundry/bin`, torch 2.7.1+cu128, `CUDA_HOME=/usr/local/cuda-12.8` |
| GroundingDINO, segment_anything | `/root/miniforge3/envs/groundingdino` (torch 2.5.1+cu124), code in `/root/SimFoundry/deps/GroundingDINO` |
| SAM vit_h, GroundingDINO weights | `/root/SimFoundry/checkpoints/` (vit_h SHA-256 checked, never deleted) |
| py7zr | `/workspace/sf4-smoke/pydeps` (`--target`; on `PYTHONPATH` only for `BIN_PIPELINES`) |
| Also present | envs `sam3d-objects`, `da3`; COLMAP 4.1.1 at `/opt/colmap-4.1.1` |

`/root/SimFoundry` is someone else's checkout on the box: read from it, never write. Our runs write only
under `/workspace/sf4-smoke` and `/workspace/data/gc6d`.

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
6. `bin`, `splat` and `masks` share `/workspace/sf4-smoke/run` and its ledger; each is judged only on its
   own items (`BIN_PIPELINES`). A failed run publishes only `manifest.json`, so failure reasons are
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

## Open PRs

- #31: this plan and handoff (docs).
- #38: bin splat, base `main`, Cursor approved, CI and GPU run green. Ready to merge.
- #39 (on #38): SF-15 masks, draft. Needs a choice for coverage: YCB object-name prompts (recommended),
  lower thresholds, or SAM automatic masks.

## Decisions already made

Bin score is the Cut 2 target. Masks use SAM vit_h. Physics comes from the VLM. `fitted_splat` and the
`mlp-physics-prior` branch are not touched. The splat export uses a numpy-only safe-globals launcher for
torch ≥ 2.6 (option B), not a global weights_only switch. One PR per cohesive change; drafts until ready;
GPU-touching PRs get a gpu-smoke run on their own branch before merge.

## How to check locally (same as CI)

Python 3.11 with `requirements-cpu-ci.txt` and `requirements-sceneagent-ci.txt`:

    pytest tests/sceneagent -m "not gpu" --strict-markers
    python scripts/ci/run_cpu_tests.py
    ruff check sceneagent && ruff format --check sceneagent && mypy sceneagent

## For Sandeep

- Merge #38 when ready; pick the #39 coverage option.
- Delete merged branches (the session can't), including `claude/instance-58301075` and `claude/hostkey-scan`.
