# Starting the GPU smoke

These runs use Sandeep's standing GPU approval for Vast instance **59601312**
only. A label, a dispatch form, and `ci/gpu-run-request.json` cannot change
that id. The workflow does not create, rent, resize, destroy, start, or stop
an instance. One run at a time: concurrency group `gpu-smoke-59601312`, and
an in-progress run is not cancelled.

SSH uses one fixed private key, the Actions secret `VAST_SSH_PRIVATE_KEY`.
This repository does not contain that secret, and this change does not create
it. The runner writes it to a mode-0600 temp file and deletes that file on
exit, including when SIGTERM raises `SystemExit` so `finally` runs. The smoke
does not generate, attach, or delete a remote authorized key. Keys already on
the instance stay until someone removes them there.

`StrictHostKeyChecking=yes`. SSH connects to `public_ipaddr` and
`ports['22/tcp'][0].HostPort` from the show response. `ssh_host` and
`ssh_port` are not a fallback. The pin is the public known_hosts line in
`ci/vast-ssh-known-host` for `[47.186.21.5]:51823`, fingerprint
`SHA256:NDJOu/2d7YbJz6Wj/HgqNSraxzsNkp8Plj/QGtAShO8`. A missing or invalid
direct endpoint, a missing secret, or a pin that does not match that
endpoint fails before SSH. There is no `ssh-keyscan` and no
trust-on-first-use. The failure text does not include the secret or the pin
line.

The fixed key remains authorized across runs. Revocation and rotation are
manual. That is less isolation than a per-run key, and it is what removes the
dependency on Vast's attach and detach API. Keep the Actions secret restricted,
keep the host pin reviewed, and rotate the key when it should stop working.
This workflow does not add an approval loop.

The manifest records the trigger (`dispatch`, `label`, or `request-file`),
the actor, the commit SHA under test, and `secrets_forwarded: false`.
`VAST_API_KEY` stays in the Actions secret. It is not forwarded over SSH, not
written into the payload, and not copied from `.git`.

## workflow_dispatch

Anyone who can dispatch Actions runs `.github/workflows/sceneagent-gpu-smoke.yml`
from the Actions tab. The workflow has no inputs. The ref that was dispatched
is the tree the GPU runs, and it is also the orchestrator on the runner.
It does not start or stop instance 59601312. A stopped instance is refused.
Starting or stopping it takes two separate user yeses outside this workflow.

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

The job logs `gpu-smoke actor=<login>`.

If the instance is stopped, not running, or in an unknown state, the label
run refuses and sends no state PUT. The failure manifest says
`instance stopped; this workflow does not start or stop instances` when the
show call reports stopped.

## Request file

Push a commit to `Main_duplicate` that changes only `ci/gpu-run-request.json`.
This is a plain `push`, so GitHub uses the workflow file on `Main_duplicate`.
Once that file is on `Main_duplicate`, the push starts a run by itself. It does
not need `workflow_dispatch`, and it does not need Sandeep to click anything.
It does not wait for `main` to be updated. It does not start or stop the
instance. A stopped, starting, or unknown machine is refused with no state
PUT, including when the trigger is `workflow_dispatch`.

The file is only a run trigger, not a way to land code. The changed-path set must be exactly
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
