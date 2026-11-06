# SDLC

How changes land in this tree. The product phases are in `docs/DESIGN.md`.

## Stages

1. **Requirements.** A change names the SceneAgent section, figure, or SimFoundry stage it serves. Claims about either paper stay tied to a section. Unpublished numbers stay unpublished.
2. **Design.** New behavior goes in `sceneagent/` or in a patch under `patches/` if OmniGibson itself must change. `scripts/pipeline/A_reconstruction`, `B_augmentation`, and `C_application` keep their current contracts. Scene JSON from SimFoundry stays compatible with `AGENTS.md`: additive keys only, no rewritten USD to move an object.
3. **Implement.** Phase-1 code is CPU-only. GPU reconstruction, Isaac, and learned policies wait for a later phase and a machine that can hold them.
4. **Test.** `python -m pytest tests/sceneagent -q` passes locally before a change is called done. Tests call the formulas and the pipeline. A test that only asserts `True` is not a test.
5. **Release.** `main` is the integration branch. Feature work can sit on a branch and merge by pull request. CI on push and pull request runs `.github/workflows/sceneagent-ci.yml` on `ubuntu-latest` with Python 3.11. Three jobs: `ruff` (`ruff check sceneagent` and `ruff format --check sceneagent`), `mypy` (`mypy sceneagent`), and `pytest unit` (`compileall` on `sceneagent`, then `pytest tests/sceneagent -m "not gpu"`). Each job installs `requirements-sceneagent-ci.txt`: `ruff==0.17.4`, `mypy==2.4.1`, `numpy==2.4.6`, `pytest==9.2.0`, `trimesh==5.2.0`, and `pillow==13.0.0`. The jobs are CPU-only. They do not install OmniGibson, CUDA, or checkpoints. A test that needs a GPU is marked `gpu` and is not collected by the pytest job. Unknown markers fail collection (`--strict-markers`).

## Out of scope

Harvard SceneAgent is the specification. The only code base being upgraded is this SimFoundry fork. Do not import, copy, or call any other application checkout from this tree. When a block changes status, regenerate `docs/GAP_MATRIX.md` from `sceneagent.catalog.to_markdown()` and keep `tests/sceneagent/test_catalog_and_video.py` green.

## Definition of done

- The behavior is covered by `tests/sceneagent` or, for a later GPU phase, by a command the design doc names and a log that shows it ran.
- The ledger or the design doc states what is still deferred.
- No benchmark figure from either paper is reported as a result of this repo unless this repo ran that protocol and the command is recorded.
- Secrets stay out of git: `.env`, `.env.local`, `api_keys.txt`, `Data/`, `checkpoints/`, `pulled/`, and Vast dumps that contain tokens. `tools/vast_texas_instance.json` may hold id, GPU, location, SSH host, SSH port, and price. It must not hold an API key.
- The existing Vast instance `58357498` (`simfoundry-r580`) is not stopped, resized, or overwritten by work on this tree.

## How a change lands

1. Branch from `main` for anything beyond a local phase-1 fix, or commit on `main` only when the change is the phase-1 tree itself and CI is green locally.
2. Open a pull request against `main` for review. CI must pass. Do not force-push `main`.
3. Do not push to `origin` (`sandy-t-shelvan/SimFoundry`) or `upstream` (`NVlabs/SimFoundry`) unless someone asked for that push. This checkout's remotes point at those repositories so the fork can be updated later. A push is not part of phase 1.
4. Generated runs live in `runs/` and are not committed.
5. A GPU install, if a later phase needs one, runs detached with a log on the machine named for that job. It does not use `/workspace/SimFoundry` on instance `58357498`.

## CI

Workflow: `.github/workflows/sceneagent-ci.yml`.

Three jobs, each on `ubuntu-latest` and Python 3.11. Each job installs the pins in `requirements-sceneagent-ci.txt`: `ruff==0.17.4`, `mypy==2.4.1`, `numpy==2.4.6`, `pytest==9.2.0`, `trimesh==5.2.0`, and `pillow==13.0.0`.

- **ruff** runs `ruff check sceneagent`, then `ruff format --check sceneagent`.
- **mypy** runs `mypy sceneagent`. numpy, trimesh, and Pillow are installed so those boundaries are typed. Optional stacks (torch, scipy, MuJoCo, SAM) are not installed.
- **pytest unit** compiles `sceneagent`, then runs `pytest tests/sceneagent -m "not gpu"`.

`pytest.ini` registers `gpu` and sets `--strict-markers`, so an unknown marker is an error. A GPU test is marked `gpu` and left out of this job. It is not skipped inside the job. SimFoundry's broader `tests/` tree needs optional packages (torch, Open3D, faiss) and is outside these jobs on purpose. `pytest.ini` still points `testpaths` at `tests` for people who have those environments. CI passes an explicit path so collection stays on the phase-1 suite.

## GPU smoke

`.github/workflows/sceneagent-gpu-smoke.yml` is manual (`workflow_dispatch` on the Actions tab, input `instance_id`). It does not run on push or pull request. The only accepted id is `58301075`, the existing Texas Vast instance. The job reads `VAST_API_KEY` from the Actions secret, starts that instance when it is stopped, runs `python -m sceneagent` on `docs/assets/example_videos/Fruits.mp4`, and uploads `plan.json`, `ledger.json`, and `manifest.json` as the artifact `sceneagent-fruits-smoke`. If this job started the instance, it stops it again. A ledger entry of `reject` or `deferred`, or a COLMAP accept that did not set `use_gpu`, fails the job. The workflow does not create, rent, resize, or destroy an instance.
