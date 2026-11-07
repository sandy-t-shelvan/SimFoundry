# Starting the GPU smoke

These runs use Sandeep's standing GPU approval for Vast instance **58301075**
only. A label, a dispatch form, and `ci/gpu-run-request.json` cannot change
that id. The workflow does not create, rent, resize, or destroy an instance.
One run at a time: concurrency group `gpu-smoke-58301075`, and an in-progress
run is not cancelled.

The manifest records the trigger (`dispatch`, `label`, or `request-file`),
the actor, the commit SHA under test, and `secrets_forwarded: false`.
`VAST_API_KEY` stays in the Actions secret. It is not forwarded over SSH, not
written into the payload, and not copied from `.git`.

## workflow_dispatch

Anyone who can dispatch Actions runs `.github/workflows/sceneagent-gpu-smoke.yml`
from the Actions tab. The workflow has no inputs. The ref that was dispatched
is the tree the GPU runs, and it is also the orchestrator on the runner.
This is the only trigger that may start instance 58301075 when it is stopped.
That click is Sandeep's explicit yes. A label or a request-file push never
sends the start PUT.

## Label `run-gpu-smoke`

An allowlisted actor adds the label `run-gpu-smoke` to a pull request whose
head is in this repository. Forks do not run. The allowlist is in the workflow
file: `sandy-t-shelvan` and `cursor[bot]`. `cursor[bot]` is the login on pull
request reviews from the Cursor GitHub App in this repository (for example
<https://github.com/sandy-t-shelvan/SimFoundry/pull/13#pullrequestreview-5796167292>).

The pull request base ref must be `main` or `Main_duplicate`. The keyed step
checks out that base SHA, so any other base is refused before the secret is
used.

The event is `pull_request_target`. GitHub loads that workflow from the
default branch, `main`, not from the pull request's base branch. A label does
not start a run until `main` contains this file. The request-file push below
does not have that limit.

When the label path does run, the runner executes `python -m sceneagent.vast_smoke`
from a checkout of the pull request **base** SHA (`orchestrator/`). The pull
request head is checked out only into `payload/` with `persist-credentials:
false`. That directory is the tree copied to the GPU. The runner does not
import it, pip-install it, or run it in any step that has `VAST_API_KEY` or a
`GITHUB_TOKEN` with write scope. The orchestrator also refuses to start when
`payload` is on `PYTHONPATH`, `PYTHONSTARTUP`, `sys.path`, a `.pth` entry, or
`sitecustomize`. The cleanup job, which has
`pull-requests: write`, does not check the head out. It removes
`run-gpu-smoke` and comments the run URL and the conclusion, including when
the smoke fails.

`LABEL_SMOKE_GATE` is the smoke job's `if:`. The token `!false` means the
`qa-pass` and `pm-pass` labels are not required. Replacing it with `!true`
turns that requirement on without moving the secret or the checkout.

The job logs `gpu-smoke actor=<login>`.

If the instance is stopped, not running, or in an unknown state, the label
run refuses before any start PUT. The failure manifest says
`instance stopped; start needs Sandeep` when the show call reports stopped.

## Request file

Push a commit to `Main_duplicate` that changes only `ci/gpu-run-request.json`.
This is a plain `push`, so GitHub uses the workflow file on `Main_duplicate`.
Once that file is on `Main_duplicate`, the push starts a run by itself. It does
not need `workflow_dispatch`, and it does not need Sandeep to click anything.
It does not wait for `main` to be updated. It does not start a stopped
instance. A stopped or unknown machine is refused before any PUT, with the
same rule as a label run. Only `workflow_dispatch` may start 58301075.

The file is only a run trigger. It is not a way to land code without the QA
and PM merge rule. The changed-path set must be exactly
`ci/gpu-run-request.json`, with no stripping or case folding. A near-miss
such as `.ci/gpu-run-request.json` or `./ci/gpu-run-request.json` is a
different path. A rename or a delete of the request file is not that set
either. If the push range changes any other path, the guard logs why and the
GPU job does not run. The guard checks out the commit **before** the push and
runs the checker from that tree, so the commit being pushed cannot replace
the checker and approve itself. Replacing the workflow file is still a commit
on `Main_duplicate`, which is the merge rule.

Do not commit `ci/gpu-run-request.json` on `Main_duplicate` except when a run
is intended. The schema is `ci/gpu-run-request.schema.json`. An example that
is not on the trigger path is `ci/gpu-run-request.example.json`. The only
fields are `ticket_id` and `reason`. An instance id in the file is rejected.
