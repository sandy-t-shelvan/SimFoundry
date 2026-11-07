# Cut 2 plan: a first scored number on the GraspClutter6D bin

Drafted 2026-11-07, after Cut 1 merged into `Main_duplicate`, and checked against
[`expansion-and-harvard-results.md`](expansion-and-harvard-results.md) (plan of record), the gap canvas
[`canvases/sceneagent-paper-gaps.canvas.tsx`](canvases/sceneagent-paper-gaps.canvas.tsx) and the
`sceneagent` package as merged. Jira epic: KAN-15.

## Goal

Cut 2 ends with **one score file for GraspClutter6D `hard/000091`**: F1 at 1 cm and Chamfer under the
paper's bin protocol (plan of record, 9.1). It is produced by our pipeline, and the rows it depends on are
honestly labeled.

Why this goal: every section 4 result needs a benchmark input, and the repo has none. The open KAN-15
tickets (SF-15 to SF-27) improve the pipeline on Fruits, but a Fruits number does not move section 4
(plan of record, 9.1). The bin is the cheapest benchmark: one scene, 13 RGB views, 20 YCB objects with
published meshes, and the paper prints the comparison (SceneAgent best 60.75% F1; SimFoundry 6.9% F1 and
4.29 cm Chamfer).

The number is ours. It is not a reproduction of Harvard's figure, and the paper's numbers stay the
paper's. Canvas and catalog status words change only with an Accepted ticket (KAN-13 comment 10222).

**Frame counts.** The bin protocol is the scene's 13 photos. That's all the scene has, and using a
different count would make the score incomparable. Video captures (Fruits) keep the 300-frame sharpest-
per-bin rule from run4 (KAN-4 comment 10311).

## What the code already has, and what's new

| Need | Already in `sceneagent` | New in Cut 2 |
|---|---|---|
| Image set in | `ingest_assets.read_still_image_sequence`, per-image color calibration | — |
| Camera poses | `run4.run_colmap_on_images` takes an image folder; the reconstruction gate | An exhaustive matcher option for 13 unordered photos (rule 10311 allows it) |
| Full-scene splat | Nothing in `sceneagent`. Pipeline A's `2c_train_bg_splat.py` trains splatfacto and writes `ns_eval.json` (PR #25) | A thin call that reuses it, not a second trainer |
| Boxes | `website_step02.run_groundingdino` (one image) | Loop over views |
| Masks | Only status helpers; `sam2_user` is SAM 2 and does not count | SAM vit_h from boxes |
| Gaussian → object | `association.py`: the paper's per-point rules (front-surface band, centroid and overlap tests) | Run them over a whole splat: project, depth-bin per cell, keep, then group masks across views |
| Object mesh | Nothing; the Fruits SAM 3D mesh was made by hand on the old box | SAM 3D install and a per-object wrapper |
| Pose to scene | `similarity_icp` (Umeyama, trimmed ICP, residuals, silhouette); `alignment.accept_alignment` | — |
| Score | `metrics.score_reconstruction`: Hungarian on Chamfer, F1 over all references (unmatched = 0), Chamfer over matched; `aggregate_messy_kitchens` for later | Load GT meshes, sample 10,000 points each, apply one similarity |
| Repair loop | `agent_loop`, `harness.request_plan` (SF-24 part 1), `reviewer_views.write_view_manifest` (names 11 views, draws none) | Draw the views, retry from them |

## Rules carried over

- Physics (mass, friction) comes from the VLM, as in the original repo. Harvard's per-Gaussian prior
  with voting is in scope as **ours**: the catalog MLP row stays `missing`, and the report says
  `weights: "ours"`.
- `fitted_splat` is Sandeep's work and is not touched.
- No stand-ins (KAN-13 comment 10318): no PartNet, no upstream SAM 3, no Hunyuan/TRELLIS mesh labeled
  SAM 3D, and no already-fitted splat in place of a step's real output.
- Each GPU-touching PR runs gpu-smoke on its own branch before merge, then merges on green CI.
- The box (59601312) bills on demand, so GPU runs are not the constraint. Agent usage is.

## Sizing

One **point** is roughly one PR with its tests and, where marked, one GPU run. Sizes are relative. I
can't see the account's usage balance, so I report points spent at each phase boundary, and the line
can move.

| Phase | Points | Cumulative | Ends with |
|---|---|---|---|
| 0. Finish Cut 1 | 3 | 3 | Cut 1 in `main`, dead gate code gone |
| 1. Benchmark and scorer | 2 | 5 | Bin inputs on the box; scorer proven on ground truth |
| 2. Poses, splat, masks | 6 | 11 | Bin splat, plus per-view masks from boxes |
| 3. Association and grouping | 5 | 16 | Gaussians per object, grouped across views |
| 4. Meshes and first score | 6 | 22 | **First bin score file** |
| 5. Agent repair loop | 8 | 30 | Score rerun with reviewer-driven retries |
| 6. Physics | 6 | 36 | VLM mass and friction plus our prior on each object |

**Cut 2 at 70% of the remaining usage = 25 points: phases 0–4 plus SF-19 (the reviewer that draws
views and retries).** The rest of phase 5 and all of phase 6 are stretch, then Cut 3. Each phase leaves
something usable on its own.

## Phase 0: finish Cut 1 (3 points)

| Item | Done when |
|---|---|
| Run4 (PR #32 gives the mapper room) and the Cut 1 merge (PR #28) | Run4 meets the 10239 bar (240 or more of 300 registered); `main` has Cut 1; `main`'s gpu-smoke names 59601312 |
| Remove the old QA/PM gate code and slim the GPU workflow | `gpu_smoke_trigger.py` gate lines, workflow lines 72–90, `docs/GPU_SMOKE_TRIGGERS.md` and their tests are gone; gpu-smoke runs on push with a path filter |
| KAN-29: GPU query timeout is a resource exit (3), not an identity exit (2) | Test for the exit code |
| Refresh stale facts in the plan of record (no status word changes; the canvas was refreshed with this plan) | Section 1's `caf16b7` branch and Windows paths; "CI installs only numpy and pytest" (now ruff, mypy, two pytest jobs, actionlint); LaMa row's instance 58301075 (now 59601312); "color calibration is still undone" (code exists in `ingest_assets`, the live run is what's pending) |
| Close KAN-30 | Done 2026-11-07: superseded, PR #26 made SSH direct-only |

After this phase, PRs target `main` and `Main_duplicate` retires (agreed 2026-11-07).

## Phase 1: benchmark and scorer (2 points, SF-28 / KAN-31)

| Item | Done when | GPU |
|---|---|---|
| Fetch GraspClutter6D `hard/000091` to the box: 13 views at 1920×1080, intrinsics, the 20 YCB meshes and their poses | A manifest with file SHA-256s; license noted; that scene only, not the archive | no |
| Bin loader: GT meshes placed by their poses, 10,000 surface points each (trimesh is already a dependency), one similarity for the whole delivered set, then `score_reconstruction` | GT against itself scores 100% F1 and 0 Chamfer; known perturbations give known drops; CPU tests | no |

Canvas rows touched: "GraspClutter6D hard/000091" (input exists, row stays missing), "F1 at 1 cm and
Chamfer" (score path exists, row stays partial until a score file).

## Phase 2: poses, splat, masks (6 points)

| Ticket | Done when | GPU |
|---|---|---|
| Poses on the bin: `run_colmap_on_images` with an exhaustive matcher option | 13 registered views and the gate document; if COLMAP can't register them, decision 2 applies | yes |
| Full-scene splat: reuse pipeline A's `2c_train_bg_splat.py` and its `ns_eval.json` | A splatfacto checkpoint from the bin poses, plus held-out PSNR/SSIM/LPIPS | yes |
| SF-15 (KAN-16): text prompt → `run_groundingdino` boxes → SAM vit_h masks, then prompt-to-object filtering (a SimFoundry addition, comment 10304) | Masks for all 13 bin views (every view: the paper's every-10th stride is for video and would leave 2 of 13), plus Fruits at paper stride, with boxes and prompts recorded | yes |

Canvas rows touched: "GroundingDINO" (missing → can be Accepted), "SAM masks" (coverage), "COLMAP camera
poses" on a second input.

## Phase 3: association and grouping (5 points)

| Ticket | Done when | GPU |
|---|---|---|
| Association runner: project every Gaussian into each mask view, take the minimum depth per cell (`mask_cell_side_px`), keep those that pass `gaussian_belongs` | One Gaussian set per mask; tests on a synthetic splat with known answers | CPU-testable, then GPU for the bin |
| SF-16 (KAN-17): group masks across views with `masks_same_object` (centroid distance plus containment or Jaccard of Gaussian sets) | One object id per physical object; count reported against the 20 GT objects | yes |

Grouping compares Gaussian sets, so it needs the splat from phase 2. This is the paper's order (KAN-4
comment 10304).

Canvas rows touched: "Gaussian association cuts" (one Fruits mask → every bin object; multi-view
grouping, which the canvas lists as not exercised).

## Phase 4: meshes and first score (6 points)

| Ticket | Done when | GPU |
|---|---|---|
| SF-29 (KAN-32): SAM 3D on 59601312 and a per-object wrapper | Install pinned and recorded; one `.glb` per grouped object; nothing else substituted | yes |
| SF-8 part 2 (KAN-8): `similarity_icp` per object from its stored SAM 3D pose, with the silhouette filled | Residuals per object; this is the GPU package comment 10318 held until SF-15 and the SAM 3D ticket exist | yes |
| Place and score: one similarity to the GT frame and `score_reconstruction` | The score file, named with point counts, alignment rule and matcher | yes |

Point at which Cut 2 has its headline result. The alignment gate stays `undecided` unless thresholds are
passed (phase 5's SF-27). The score file does not need the gate to accept.

## Phase 5: agent repair loop (8 points; SF-19 is inside the 70% line)

This is what "with SceneAgent" means in the paper: the agent looks at renders and retries.

| Ticket | Done when |
|---|---|
| SF-19 (KAN-20): per-object reviewer, with registration overlays plus 8 isolated and 2 in-scene views, then retry | `reviewer_views` draws the views instead of only naming them; a retry is chosen from them and logged |
| SF-18 (KAN-19): step-runner visual QA with `gemini-3.8-flash` | Each step's output gets a recorded VLM verdict |
| SF-24 (KAN-25, in progress): orchestrator plan from a real harness | Part 2 finished; the plan comes from `harness.request_plan`, not `default_plan()` |
| SF-27 (KAN-28): held-out thresholds for `accept_alignment` | Thresholds set on held-out data, labeled ours; no constant edited to fit a file |
| Rerun the bin with the loop on | Second score file, next to the first, with the loop's ledger |

## Phase 6: physics (6 points; stretch)

| Ticket | Done when |
|---|---|
| SF-20 (KAN-21, in progress): equation (2) three-source vote, plus the per-Gaussian prior as ours | Vote of prior + VLM + keywords on each bin object; prior trained with `mlp/train_prior.py` and labeled `weights: "ours"` |
| SF-21 (KAN-22): VLM material audit | Audit recorded per object (`section34.request_audit`, live) |
| SF-22 (KAN-23): Isaac Lab inspector report written by the model | `fix_report_from_model: true` on a bin object |

## Cut 3 (not in Cut 2)

MessyKitchens score (aggregator already exists); SF-17 LangSplat-style lift; SF-23 background
orientation; SF-25 sisters; SF-26 scripted expert; pi0.5 LoRA; 4D (section 3.6); fracture; Meshy;
Unreal; hybrid scenes; CubePart and Articulate Anything (canvas phase 3, SF-10); the SF-9 background-plane
GPU package, which can be pulled in if a phase finishes early. (SF-8 part 2 is in phase 4.)

## Risks

- **13 photos.** COLMAP and splatfacto may do poorly on sparse views. Mitigation: exhaustive matching,
  plus decision 2.
- **SAM 3D on CUDA 13.2.** It ran on the old box, but it isn't installed on 59601312. The install could
  take more than its share of phase 4.
- **Association at scale.** The Fruits run applied it to one mask, on the host, outside this package.
  Twenty objects across 13 views is the first real test, which is why phase 3 has its own points.
- **Shared GPU.** Interactive work on the box can make a gpu-smoke run refuse to start. The memory check
  refuses instead of crashing.

## Decisions (2026-11-07)

1. **Target: the bin score.** Decided (Sandeep, 2026-11-07). Phase 1 is PR #33.
2. **If COLMAP can't register the 13 photos:** use the dataset's camera poses, and record that as a
   deviation in the score file. Decided.
3. **Masks:** SAM vit_h (matches the plan of record; SAM 2 would leave the paper's mask row open).
   Decided.
4. **Jira:** done. SF-28 is KAN-31, SF-29 is KAN-32, and KAN-30 is closed as superseded by PR #26.
