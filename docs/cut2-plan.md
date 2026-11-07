# Cut 2 plan: a first scored number on the GraspClutter6D bin

Drafted 2026-11-07, after Cut 1 merged into `Main_duplicate`. The plan of record stays
[`expansion-and-harvard-results.md`](expansion-and-harvard-results.md). This file is the order and
sizing for Cut 2 (Jira epic KAN-15).

## Goal

Cut 2 ends with **one score file for GraspClutter6D `hard/000091`**: F1 at 1 cm and Chamfer under the
paper's bin protocol (section 9.1 of the plan of record). It is produced by our pipeline, and the rows
it depends on are honestly labeled.

Why this goal: every section 4 result needs a benchmark input, and the repo has none. The open KAN-15
tickets (SF-15 to SF-27) improve the pipeline on Fruits, but a Fruits number does not move section 4
(plan of record, 9.1). The bin is the cheapest benchmark: one scene, 13 RGB views, 20 YCB objects with
published meshes, and the paper prints the comparison (SceneAgent best 60.75% F1; SimFoundry 6.9% F1 and
4.29 cm Chamfer).

The number is ours. It is not a reproduction of Harvard's figure, and the paper's numbers stay the
paper's.

## Rules carried over

- Physics (mass, friction) comes from the VLM, as in the original repo. Harvard's per-Gaussian prior
  with voting is in scope as **ours**: the catalog MLP row stays `missing`, and the report says
  `weights: "ours"`.
- `fitted_splat` is Sandeep's work and is not touched.
- No stand-ins (KAN-13 comment 10318): no PartNet, no upstream SAM 3, and no already-fitted splat in place
  of a step's real output.
- Each GPU-touching PR runs gpu-smoke on its own branch before merge, then merges on green CI.
- The box (59601312) bills on demand, so GPU runs are not the constraint. Agent usage is.

## Sizing

One **point** is roughly one PR with its tests and, where marked, one GPU run. Sizes are relative. I
can't see the account's usage balance, so I report points spent at each phase boundary, and the stop
line can move.

| Phase | Points | Cumulative | Ends with |
|---|---|---|---|
| 0. Finish Cut 1 | 3 | 3 | Cut 1 in `main`, dead gate code gone |
| 1. Benchmark and scorer | 3 | 6 | Bin inputs on the box, scorer proven on ground truth |
| 2. Perception | 6 | 12 | Per-object masks on all 13 bin views |
| 3. Geometry and first score | 8 | 20 | **First bin score file** |
| 4. Agent repair loop | 8 | 28 | Score file rerun with reviewer-driven retries |
| 5. Physics | 6 | 34 | VLM mass and friction plus our prior on each bin object |

**The 70% line falls after phase 4** (28 of 34 points is about 82% of the planned work). Phases 0–4
are the core. Phase 5 is the first to move to Cut 3 if usage runs short. Each phase leaves something
usable on its own.

## Phase 0: finish Cut 1 (3 points)

| Item | Done when |
|---|---|
| Run4 result and Cut 1 merge (PR #28) | Run4 meets the 10239 bar (240 or more of 300 registered); `main` has Cut 1; `main`'s gpu-smoke names 59601312 |
| Remove the old QA/PM gate code and slim the GPU workflow | `gpu_smoke_trigger.py` gate lines, workflow lines 72–90, `docs/GPU_SMOKE_TRIGGERS.md` and their tests are gone; gpu-smoke runs on push with a path filter |
| KAN-29: GPU query timeout is a resource exit (3), not an identity exit (2) | Test for the exit code |
| Close KAN-30 | Superseded: PR #26 made SSH direct-only |

After this phase, PRs target `main` and `Main_duplicate` retires (agreed 2026-11-07).

## Phase 1: benchmark and scorer (3 points, new ticket SF-28)

| Item | Done when | GPU |
|---|---|---|
| Fetch GraspClutter6D `hard/000091` to the box: 13 views at 1920×1080, intrinsics, the 20 YCB meshes and their poses | A manifest with file SHA-256s; license noted | no (download) |
| `score_bin`: 10,000 surface points per GT object, Hungarian on Chamfer cost, F1 at 1 cm over all 20 objects (unmatched objects score 0), Chamfer over matched objects only, one similarity for the whole run, no per-object refit | Scorer on the GT meshes scores 100% F1 and 0 Chamfer; known perturbations give known drops; CPU tests in CI | no |

This reuses `metrics.f1_at_threshold`, `symmetric_chamfer` and `_hungarian`, which already match 20
objects. Only the loader and the single-similarity step are new.

## Phase 2: perception (6 points)

| Ticket | Done when | GPU |
|---|---|---|
| SF-15 (KAN-16): text prompt → GroundingDINO boxes → SAM vit_h masks | Masks for all 13 bin views (and Fruits at paper stride), boxes and prompts recorded | yes |
| SF-16 (KAN-17): multi-view mask grouping into objects | One object id per physical object across views; count reported against the 20 GT objects | yes |

The box has Grounded-SAM2, which uses SAM 2. The plan of record says SAM 2 does not close the paper's
SAM row, so SF-15 installs SAM vit_h next to it. That's a small install, unless you rule SAM 2
acceptable (decision 3).

## Phase 3: geometry and first score (8 points)

| Ticket | Done when | GPU |
|---|---|---|
| Poses and splat on the bin: COLMAP on the 13 views (reusing run4's code), then splatfacto | Registered views and a checkpoint; if COLMAP can't register 13 sparse views, decision 2 applies | yes |
| SF-29 (new): SAM 3D on the new box, one mesh per grouped object | Install pinned and recorded; one `.glb` per object; Hunyuan/TRELLIS not substituted | yes |
| Association + ICP + one similarity + scoring | Existing `association.gaussian_belongs` and `similarity_icp` on each object; `score_bin` writes the score file | yes |

Point at which Cut 2 has its headline result.

## Phase 4: agent repair loop (8 points)

This is what "with SceneAgent" means in the paper: the agent looks at renders and retries.

| Ticket | Done when |
|---|---|
| SF-19 (KAN-20): per-object reviewer, with registration overlays plus 8 isolated and 2 in-scene views, then retry | Views are drawn (today they are only named); a retry is chosen from them and logged |
| SF-18 (KAN-19): step-runner visual QA with `gemini-3.8-flash` | Each step's output gets a recorded VLM verdict |
| SF-24 (KAN-25, in progress): orchestrator plan from a real harness | Finish part 2; the plan comes from the harness, not `default_plan()` |
| SF-27 (KAN-28): held-out thresholds for `accept_alignment` | Thresholds set on held-out data, labeled ours |
| Rerun the bin with the loop on | Second score file, next to the first, with the loop's ledger |

## Phase 5: physics (6 points; first to move to Cut 3)

| Ticket | Done when |
|---|---|
| SF-20 (KAN-21, in progress): equation (2) three-source vote, plus the per-Gaussian prior as ours | Vote of prior + VLM + keywords on each bin object; prior trained here and labeled `weights: "ours"` |
| SF-21 (KAN-22): VLM material audit | Audit recorded per object |
| SF-22 (KAN-23): Isaac Lab inspector report written by the model | `fix_report_from_model: true` on a bin object |

## Cut 3 (not in Cut 2)

MessyKitchens score; SF-17 LangSplat-style lift; SF-23 background orientation; SF-25 sisters;
SF-26 scripted expert; pi0.5 LoRA; 4D (section 3.6); fracture; Meshy; Unreal; hybrid scenes; SF-8,
SF-9 and SF-10 live GPU packages (unblocked by SF-15 and SF-29, so they can be pulled in if a phase
finishes early).

## Risks

- **13 sparse views.** COLMAP and splatfacto may do poorly. Mitigation: exhaustive matching, plus
  decision 2.
- **SAM 3D on CUDA 13.2.** It ran on the old box, but it isn't installed on 59601312. The install
  could take more than its 4 points.
- **Dataset size.** Only one scene is needed. Download that scene, not the archive.
- **Shared GPU.** Interactive work on the box can make a gpu-smoke run refuse to start. The memory check
  refuses instead of crashing.

## Decisions needed

1. **Retarget Cut 2** from "Fruits pipeline stages" to "first bin score", using the existing KAN-15
   tickets plus SF-28 and SF-29. Recommended.
2. **If COLMAP can't register the 13 views:** use the dataset's camera poses and record that as a
   deviation in the score file, or stop and report? Recommended: use them, and label the result.
3. **Masks:** install SAM vit_h (matches the plan of record), or accept SAM 2 from Grounded-SAM2?
   Recommended: SAM vit_h.
4. **Jira:** may I create SF-28 and SF-29 under KAN-15 and close KAN-30?
