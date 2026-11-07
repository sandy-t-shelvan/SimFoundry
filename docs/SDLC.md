# SDLC

How changes land in this tree. The product phases are in `docs/DESIGN.md`.

## Stages

1. **Requirements.** A change names the SceneAgent section, figure, or SimFoundry stage it serves. Claims about either paper stay tied to a section. Unpublished numbers stay unpublished.
2. **Design.** New behavior goes in `sceneagent/` or in a patch under `patches/` if OmniGibson itself must change. `scripts/pipeline/A_reconstruction`, `B_augmentation`, and `C_application` keep their current contracts. Scene JSON from SimFoundry stays compatible with `AGENTS.md`: additive keys only, no rewritten USD to move an object.
3. **Implement.** Phase-1 code is CPU-only. GPU reconstruction, Isaac, and learned policies wait for a later phase and a machine that can hold them.
4. **Test.** `python -m pytest tests/sceneagent -q` passes locally before a change is called done. Tests call the formulas and the pipeline. A test that only asserts `True` is not a test.
5. **Release.** `main` is the integration branch. Feature work can sit on a branch and merge by pull request. CI on push and pull request runs `.github/workflows/sceneagent-ci.yml` on `ubuntu-latest` with Python 3.11. Jobs: `ruff` (`ruff check sceneagent` and `ruff format --check sceneagent`), `mypy` (`mypy sceneagent`), `pytest unit` (`compileall` on `sceneagent`, then `pytest tests/sceneagent -m "not gpu"`), `pytest cpu` (lightweight tests under `tests/` outside `tests/sceneagent`, with `-m "not gpu and not network"`), and `actionlint` (actionlint 1.7.12, sha256 `8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8`, plus shellcheck). The sceneagent jobs install `requirements-sceneagent-ci.txt`: `ruff==0.17.4`, `mypy==2.4.1`, `numpy==2.4.6`, `pytest==9.2.0`, `trimesh==5.2.0`, and `pillow==13.0.0`. The CPU job also installs `requirements-cpu-ci.txt`. The jobs are CPU-only. They do not install OmniGibson, CUDA, or checkpoints. A test that needs a GPU is marked `gpu`. A network-client test is marked `network`, not `gpu`. The CPU job excludes both with `-m "not gpu and not network"`. Unknown markers fail collection (`--strict-markers`). Tests the CPU job cannot run are listed with a reason in `ci/cpu-test-skips.txt`. A skip inside that job is a failure.

## Out of scope

Harvard SceneAgent is the specification. The only code base being upgraded is this SimFoundry fork. Do not import, copy, or call any other application checkout from this tree. When a block changes status, regenerate `docs/GAP_MATRIX.md` from `sceneagent.catalog.to_markdown()` and keep `tests/sceneagent/test_catalog_and_video.py` green.

## Definition of done

- The behavior is covered by `tests/sceneagent` or, for a later GPU phase, by a command the design doc names and a log that shows it ran.
- The ledger or the design doc states what is still deferred.
- No benchmark figure from either paper is reported as a result of this repo unless this repo ran that protocol and the command is recorded.
- Secrets stay out of git: `.env`, `.env.local`, `api_keys.txt`, `Data/`, `checkpoints/`, `pulled/`, and Vast dumps that contain tokens. `tools/vast_texas_instance.json` may hold id, GPU, location, SSH host, SSH port, and price. It must not hold an API key.
- The only Vast instance this tree may use is `59601312`.

## How a change lands

1. Branch from `main` for anything beyond a local phase-1 fix, or commit on `main` only when the change is the phase-1 tree itself and CI is green locally.
2. Open a pull request against `main` for review. CI must pass. Do not force-push `main`.
3. Do not push to `origin` (`sandy-t-shelvan/SimFoundry`) or `upstream` (`NVlabs/SimFoundry`) unless someone asked for that push. This checkout's remotes point at those repositories so the fork can be updated later. A push is not part of phase 1.
4. Generated runs live in `runs/` and are not committed.
5. A GPU install, if a later phase needs one, runs detached with a log on the machine named for that job. The only Vast instance that job may use is `59601312`.

## CI

Workflow: `.github/workflows/sceneagent-ci.yml`.

Five jobs, each on `ubuntu-latest`. The sceneagent jobs install the pins in `requirements-sceneagent-ci.txt`: `ruff==0.17.4`, `mypy==2.4.1`, `numpy==2.4.6`, `pytest==9.2.0`, `trimesh==5.2.0`, and `pillow==13.0.0`. The broader CPU job also installs `requirements-cpu-ci.txt`. `actionlint` downloads actionlint 1.7.12 and checks that sha256 `8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8` before it lints `.github/workflows`. Shellcheck is installed so the `run` scripts are linted too.

- **ruff** runs `ruff check sceneagent`, then `ruff format --check sceneagent`.
- **mypy** runs `mypy sceneagent`. numpy, trimesh, and Pillow are installed so those boundaries are typed. Optional stacks (torch, scipy, MuJoCo, SAM) are not installed for mypy.
- **pytest unit** compiles `sceneagent`, then runs `pytest tests/sceneagent -m "not gpu"`.
- **pytest cpu** runs the lightweight tests under `tests/` outside `tests/sceneagent`, with `-m "not gpu and not network"`. Extra pins include scipy, matplotlib, OmegaConf, embreex, open3d 0.19.0, and opencv-python-headless 4.13.0.90. Torch, faiss, OmniGibson, CUDA, and checkpoints are not installed.
- **actionlint** runs actionlint 1.7.12 on `.github/workflows`, after the sha256 check above. Shellcheck lints the `run` scripts.

`pytest.ini` registers `gpu` and `network` and sets `--strict-markers`, so an unknown marker is an error. A test that needs a GPU is marked `gpu`. A test that needs a network service is marked `network`. The CPU job's command leaves both out. `ci/cpu-test-skips.txt` names every other test or file that job does not run, with a reason. A skip inside that job fails the run. `pytest.ini` still points `testpaths` at `tests` for people who have the heavier environments. The sceneagent job passes an explicit path so its collection stays on that suite.

## GPU smoke

`.github/workflows/sceneagent-gpu-smoke.yml` keeps `workflow_dispatch` with no inputs. It also starts from a same-repo `run-gpu-smoke` label (`pull_request_target`, types `labeled`) and from a push to `Main_duplicate` that touches only `ci/gpu-run-request.json`. How a bot starts each one is `docs/GPU_SMOKE_TRIGGERS.md`. These runs use Sandeep's standing GPU approval for instance 59601312 only. The job passes `--instance-id 59601312` as a literal, and the CLI rejects any other id before a network call. No label, input, or request-file field can change it. The job reads `VAST_API_KEY` from the Actions secret. No trigger starts or stops instance 59601312, including `workflow_dispatch`. A stopped, starting, or unknown instance is refused after show, and no state PUT is sent. The job then runs `python -m sceneagent` on `docs/assets/example_videos/Fruits.mp4`, and uploads `plan.json`, `ledger.json`, and `manifest.json` as the artifact `sceneagent-fruits-smoke`. `plan.json` is the file the pipeline wrote. The collector does not synthesize one, and a missing plan fails the job. This job does not start or stop the instance. A ledger entry of `reject` or `deferred`, or a COLMAP accept that did not set `use_gpu`, fails the job. The workflow does not create, rent, resize, or destroy an instance. One run at a time: concurrency group `gpu-smoke-59601312`, `cancel-in-progress: false`.

`pull_request_target` loads the workflow file from the default branch (`main`), not from the pull request's base. A label does not start this workflow until `main` contains it. The request-file push is a plain `push`, so it uses the workflow on `Main_duplicate` and does not wait for that merge. On a label run the keyed step executes `sceneagent.vast_smoke` from the pull request base SHA. The pull request head is checked out only into `payload/` and is what the GPU runs. The runner does not import, install, or execute that tree in a step that holds `VAST_API_KEY` or a write-scoped token. `LABEL_SMOKE_GATE` is the smoke job's `if:`. The `!false` token leaves the QA PASS and PM PASS requirement off. A push whose `before..after` range changes any path besides `ci/gpu-run-request.json` logs the reason and does not call the GPU. The checker that decides that is the commit before the push. The manifest records the trigger (`dispatch`, `label`, or `request-file`), the actor, the commit SHA, and `secrets_forwarded: false`. SSH does not forward the runner environment, and rsync does not copy `.git`.

After attach, the job probes SSH at most 6 times inside a 45s budget and raises if the key is still refused. An attach response with no key id is logged. The key is still detached when the listing has exactly one match. Once show has succeeded, a failed run writes only a redacted `manifest.json`. The failure upload step sends that file as `sceneagent-fruits-failure-manifest`. The same key scan applies: if the key survives redaction, the artifact directory stays empty and nothing is uploaded. The allowlist adds one read, `GET /api/v0/instances/59601312/ssh/`, pinned to that instance. Before attach, that call supplies the manifest's key count and locally computed `SHA256:` fingerprints. One more GET of the same URL after attach matches this run's public key with trailing whitespace stripped and nothing else. Exactly one match is removed with `DELETE /api/v0/instances/59601312/ssh/{id}` on success and on failure. Zero or several matches delete nothing. The request guard refuses every other id. Earlier runs' keys are not removed.
