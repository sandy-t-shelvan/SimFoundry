# SDLC

How changes land in this tree. The product phases are in `docs/DESIGN.md`.

## Stages

1. **Requirements.** A change names the SceneAgent section, figure, or SimFoundry stage it serves. Claims about either paper stay tied to a section. Unpublished numbers stay unpublished.
2. **Design.** New behavior goes in `sceneagent/` or in a patch under `patches/` if OmniGibson itself must change. `scripts/pipeline/A_reconstruction`, `B_augmentation`, and `C_application` keep their current contracts. Scene JSON from SimFoundry stays compatible with `AGENTS.md`: additive keys only, no rewritten USD to move an object.
3. **Implement.** Phase-1 code is CPU-only. GPU reconstruction, Isaac, and learned policies wait for a later phase and a machine that can hold them.
4. **Test.** `python -m pytest tests/sceneagent -q` passes locally before a change is called done. Tests call the formulas and the pipeline. A test that only asserts `True` is not a test.
5. **Release.** `main` is the integration branch. Feature work can sit on a branch and merge by pull request. CI on push and pull request runs `.github/workflows/sceneagent-ci.yml` on `ubuntu-latest` with Python 3.11. Jobs: `ruff` (`ruff check sceneagent` and `ruff format --check sceneagent`), `mypy` (`mypy sceneagent`), `pytest unit` (`compileall` on `sceneagent`, then `pytest tests/sceneagent -m "not gpu"`), and `pytest cpu` (lightweight tests under `tests/` outside `tests/sceneagent`, also `-m "not gpu"`). The sceneagent jobs install `requirements-sceneagent-ci.txt`: `ruff==0.17.4`, `mypy==2.4.1`, `numpy==2.4.6`, `pytest==9.2.0`, `trimesh==5.2.0`, and `pillow==13.0.0`. The CPU job also installs `requirements-cpu-ci.txt`. The jobs are CPU-only. They do not install OmniGibson, CUDA, or checkpoints. A test that needs a GPU or a network call is marked `gpu` and is not collected. Unknown markers fail collection (`--strict-markers`). Tests the CPU job cannot run are listed with a reason in `ci/cpu-test-skips.txt`. A skip inside that job is a failure.

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

Four jobs, each on `ubuntu-latest` and Python 3.11. The sceneagent jobs install the pins in `requirements-sceneagent-ci.txt`: `ruff==0.17.4`, `mypy==2.4.1`, `numpy==2.4.6`, `pytest==9.2.0`, `trimesh==5.2.0`, and `pillow==13.0.0`. The broader CPU job also installs `requirements-cpu-ci.txt`.

- **ruff** runs `ruff check sceneagent`, then `ruff format --check sceneagent`.
- **mypy** runs `mypy sceneagent`. numpy, trimesh, and Pillow are installed so those boundaries are typed. Optional stacks (torch, scipy, MuJoCo, SAM) are not installed for mypy.
- **pytest unit** compiles `sceneagent`, then runs `pytest tests/sceneagent -m "not gpu"`.
- **pytest cpu** runs the lightweight tests under `tests/` outside `tests/sceneagent`, with `-m "not gpu"`. scipy, matplotlib, OmegaConf, and embreex are the extra pins. Torch, Open3D, faiss, OmniGibson, CUDA, and checkpoints are not installed.

`pytest.ini` registers `gpu` and sets `--strict-markers`, so an unknown marker is an error. A test that needs a GPU or a network call is marked `gpu` and left out of both pytest jobs. It is not skipped inside the job. `ci/cpu-test-skips.txt` names every other test or file the CPU job does not run, with a reason. A skip inside that job fails the run. `pytest.ini` still points `testpaths` at `tests` for people who have the heavier environments. The sceneagent job passes an explicit path so its collection stays on that suite.
