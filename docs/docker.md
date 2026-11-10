# Basic Docker image

A CPU-only image of the `sceneagent` package with the same pinned packages as the CPU CI jobs
(`requirements-sceneagent-ci.txt` and `requirements-cpu-ci.txt`, Python 3.11). It runs the test suite, the
phase-1 contract and the GraspClutter6D scorer on any machine with Docker, no GPU needed.

```bash
docker build -t simfoundry-basic .
```

Run the CPU test suite (the default command; about 45 s):

```bash
docker run --rm simfoundry-basic
```

Run the phase-1 contract, writing to `./runs` on the host:

```bash
mkdir -p runs && docker run --rm -v "$PWD/runs:/opt/simfoundry/runs" simfoundry-basic python -m sceneagent examples/phase1_place_scene.json --out runs/phase1_place
```

The container runs as a non-root user (uid 1000), so on Linux the host `runs/` folder must be writable by
that uid.

## What is and is not inside

| Inside | Not inside |
|---|---|
| `sceneagent`, `simfoundry`, tests, docs, examples | `.git`, `deps/`, data, checkpoints, credential files |
| numpy, scipy, trimesh, Pillow, open3d, OpenCV (headless), py7zr, pytest, ruff, mypy | torch and the GPU stacks: OmniGibson/Isaac Sim, nerfstudio, COLMAP with CUDA, GroundingDINO/SAM, SAM 3D, Depth Anything 3 |
| ffmpeg, git | COLMAP (the video path records a missing binary rather than a reconstruction) |

The GPU stages run on the Vast box through the gpu-smoke workflow, and `scripts/installation/` installs
the full environments.

One test reads `git rev-parse HEAD` from the checkout. The image leaves `.git` out, so the default command
deselects that test (`test_offline_dry_run_registers_every_native_frame`). It runs in CI and in a clone.

## Building behind a TLS-intercepting proxy

The Dockerfile uses the public Debian and PyPI mirrors. Behind a proxy that re-signs TLS, pass the proxy
with `--build-arg https_proxy=...` and trust its CA in a local copy of the Dockerfile; do not commit the
CA or proxy settings.
