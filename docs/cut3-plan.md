# Cut 3 plan: from a working bin pipeline to a SceneAgent-quality score

Drafted 2026-11-10, after Cut 2's phases 0–4 merged (#40–#42) and SF-19 went up for review (#44). Checked
against [`expansion-and-harvard-results.md`](expansion-and-harvard-results.md) (plan of record, §9.5 order),
the gap canvas [`canvases/sceneagent-paper-gaps.canvas.tsx`](canvases/sceneagent-paper-gaps.canvas.tsx) (25 rows
still `missing` or `partial`), the paper-map artifact (pipeline blocks, agent lanes, VLM roles), and
[`cut2-plan.md`](cut2-plan.md)'s "Cut 3" list.

## Where Cut 2 left the bin

| Step | Result on GraspClutter6D `hard/000091` |
|---|---|
| Poses | 13/13 views, 0.55 px; camera centres fit the dataset's to 3.5 mm |
| Splat | 214,976 Gaussians, held-out PSNR 23.07, SSIM 0.825, LPIPS 0.178 |
| Masks | 94 (5–11 per view) |
| Association | 25 objects against 20; DA3 scale 0.201 m per unit (GT cameras say 0.206) |
| SAM 3D | 25 of 25 meshed on the 24 GB 4090 |
| Score | F1@1cm **0.058** (merged); **0.067** with the SF-19 reviewer loop; Chamfer 0.125 m |
| Paper | SimFoundry 6.9% / 4.29 cm; SceneAgent best **60.75%** |

What the reviewer's drawn views show is the order of the problems:

1. **Floater Gaussians.** Every object's Gaussian set is a cluster plus a fan of points only one view admits,
   so ICP pulls meshes onto neighbours (object 3 lands on the SPAM box). PR #45 tries the paper's camera
   consensus at κ=2.
2. **Grouping.** 25 objects for 20; one group mixes "bottle jar", "clamp", "jar"; the two largest span 12–13
   views and are probably the bin itself.
3. **SAM 3D pose.** Its translation starts a median 0.75 m off, because it comes from SAM 3D's own camera
   (MoGe depth and intrinsics); 20 of 25 fits come from pose-free seeds.
4. **The reviewer is rule-based.** The paper's reviewer is a VLM looking at the images (SF-18).

## Goal

**Cut 3 ends with three milestones met:**
- **M1:** a bin score above the paper's SimFoundry baseline (6.9% F1).
- **M2:** a VLM in the review loop.
- **M3:** the paper's full §3.4 physics on every bin object, in a USD scene that settles in Isaac Lab.

The number stays ours, and the paper's numbers stay the paper's. A score file is written whatever the number is.

Cut 3 is the first of three cuts to the MVP. Cut 4 adds full articulation, soft bodies and fracture; Cut 5 is the MVP itself. Sandeep asked (2026-11-10) that physics and articulation be in the MVP in full. The roadmap and the MVP definition are in [`expansion-and-harvard-results.md`](expansion-and-harvard-results.md) under "Roadmap to the MVP", and in Jira as epics KAN-33 (Cut 3), KAN-34 (Cut 4) and KAN-35 (Cut 5).

## Rules carried over

- Paper numbers stay the paper's; ours are labelled ours in the evidence.
- No stand-ins (KAN-13 comment 10318). No constant tuned to fit a file: the ours_iteration_1 gate, the
  association cuts and the trim stay as they are.
- `fitted_splat` and the MLP physics prior files are not touched.
- Each GPU-touching PR runs gpu-smoke on its own branch before merge; one PR per feature.
- The box is 58301075 (59601312 was deleted on 2026-11-10); sessions never start, stop or rent instances.
- **New:** dispatch gpu-smoke runs one at a time. A newer pending run cancels an older pending one (that
  is how place run 4 was lost on 2026-11-10).

## Sizing

Same unit as Cut 2: one point is roughly one PR with its tests and, where marked, a GPU run.

| Phase | Milestone | Points | Cumulative | Ends with | Jira |
|---|---|---|---|---|---|
| A. Bin: data quality | M1 | 6 | 6 | Floaters filtered, grouping fixed, pose chain fixed; bin above 6.9% | KAN-36, KAN-37, KAN-38 |
| B. VLM in the loop | M2 | 6 | 12 | Reviewer and step-runner verdicts from a VLM; bin rescored | KAN-19, KAN-20, KAN-25 |
| C. Full §3.4 physics on the bin | M3 | 9 | 21 | Every property the paper lists, per object; USD stage that settles in Isaac Lab and MuJoCo; inspector report by the model | KAN-39, KAN-40, KAN-41, KAN-21, KAN-22, KAN-23 |

**Cut 3 line: phases A–C (21 points).** The kitchen and the videos move to Cut 5, and articulation is Cut 4 (see "After Cut 3").

## Phase A: bin data quality (6 points)

**Result so far (2026-11-11, box 58301075).** These are F1@1cm and matched Chamfer, delivered against GT:

| Change | Runs | F1@1cm | Chamfer |
|---|---|---|---|
| Baseline, κ=1 (main) | place 42916527232 | 0.079 | 0.070 m |
| **κ=2 (PR #45, SF-30)** | association 42924495643, place 42925055261 | **0.125** | **0.046 m** |
| κ=2 + container split (PR #47, SF-31): **not kept** | association 42927798966, place 42929004562 | 0.109 | 0.032 m |
| **κ=2 + SAM 3D pose rescale (PR #48, SF-32)** | association 42929907431, sam3d 42930467798, place 42932270322 | **0.327** | **0.021 m** |
| … then the rule-based reviewer (SF-19) | review 42934419274 | 0.323 | 0.024 m |
| … then Gemini's verdicts (SF-19 VLM, PR #54) | verdict 42941121692, review_vlm 42943461864 | 0.327 | 0.021 m |
| … Gemini picks and accepts retries (round 2, PR #55) | verdicts 42947401149 and 42964570514, candidates 42963190714, apply 42965157633 | 0.327 | 0.021 m |

κ=2 keeps 65,972 of 78,121 Gaussians, and only κ differs from the baseline. **M1 (bin above 6.9%) is met by a change of ours.** κ=2 is ours; the paper uses κ=1. The container split (#47) set aside four groups that held real objects (GT 8, 24, 29 and 37). Their matches were lost, so F1 fell, and the split is not kept; the 0.35 m limit is not tuned to this scene.

**Pose rescale (#48), kept.** SAM 3D now records MoGe's median depth inside the object's mask (1.14–1.66 m on the 14 objects), and placement divides our depth of the object's Gaussians in the same view by it. The ratio is 0.31–0.46 (median 0.334): MoGe puts every object about three times farther away than the splat does, so SAM 3D's translation and size were both about three times too large. Multiplying both by the ratio gives one more ICP start, `sam3d_rescaled`, and the fit's size bound now follows the rescaled size. Against the same 14 groups:

- the best SAM 3D start lands a median 0.090 splat units from its Gaussians, against 5.79 for the raw pose and 0.464 for the centred start;
- 5 of 14 fits now win from a rescaled start (seeds 6, raw pose 2, centred 1; before: seeds 8, centred 4, raw pose 2);
- every object's fit distance falls (14 of 14);
- against GT, 12 matches improve (GT 9: 0.18 → 0.94, GT 17: 0.27 → 0.94), GT 3 falls (0.32 → 0.25), GT 37 loses its match and GT 40 gains one (0.49).

The size bound and the new start change in the same run, so their shares are not separated; seeds that won both times also fit better, which points at the bound. No ground truth is used, and no constant is fitted to this scene. The `ours_iteration_1` gate still rejects all 14 (its cutoffs are unchanged). The paper's bin numbers (6.9% for original SimFoundry, 60.75% for SceneAgent) stay the paper's. Score file: `docs/results/…_58301075_kappa2_rescale.json`.

**The rule-based reviewer no longer helps.** On top of the rescale, SF-19 called `icp_flip` on 12 of 14 objects and `keep` on 2, and kept 5 flips because each lowered the fit distance a little (by 0.00003–0.017). Against GT the bin got slightly worse: F1 0.327 → 0.323, Chamfer 0.021 → 0.024 m (`…_kappa2_rescale_reviewed.json`). On κ=2 alone it had helped (0.125 → 0.128). With better starts, a smaller fit distance no longer means a better placement. Its isolated-view agreement is below 0.5 on 12 objects even where the overlay agrees (0.78–0.90 on objects 1, 2 and 11), which points at the rule, not the fit. This is the case Phase B's VLM verdict is for: it is measured against both 0.327 and the rule's 0.323.

| Item | Done when | GPU |
|---|---|---|
| Camera consensus κ=2 (PR #45) | Association + review rerun; score against κ=1 recorded; kept only if it helps | yes |
| Reviewer before/after split (PR #44, `0186e5c`) | One run gives the score before and after retries, separating the centred-start gain from the retries | yes |
| Grouping: drop container and support regions | Masks whose Gaussians span most of the scene (the bin, the table) are recorded and left out of objects (rule labelled ours); object count reported against 20 | yes |
| SAM 3D pose chain | SAM 3D's MoGe pointmap compared with our DA3 depth on the same view; the pose start uses the measured scale and intrinsics; median start distance reported before and after | yes |

Canvas rows touched: "Trimmed similarity ICP", "Alignment acceptance gate" (residuals improve; the gate's
cutoffs stay), "F1 at 1 cm and Chamfer".

## Phase B: VLM in the loop (6 points)

| Item | Done when | GPU |
|---|---|---|
| SF-18 (KAN-19): step-runner visual QA | Each pipeline step's output gets a recorded VLM verdict (`gemini-3.8-flash`, the model the ledger already names) | no |
| SF-19 VLM verdict | The reviewer sends each object's contact sheet to the VLM, which picks keep / icp_flip / sam3d_next_view; the rule-based choice is logged beside it; both scores recorded | yes |
| SF-24 part 2 (KAN-25): orchestrator plan from a real harness | The plan comes from `harness.request_plan`, not `default_plan()` | no |

The VLM is the Gemini API (`gemini-3.8-flash`), with the key from the GitHub secret `GEMINI_API_KEY` (decision 2). The GPU workflow forwards no secrets to the box by design, so the calls run on the GitHub runner, on the contact sheets and renders the box publishes.

**SF-18 step-runner critic, first part (2026-11-11, `sceneagent.step_critic`).** Per-stage prompts (masks, Gaussians, mesh, parts) and one schema: accept or reject, a reason, and a retry note for a reject. `judge` runs on the runner and writes `review/<stage>.json`, which is dc1's shape. The box side covers three pieces: it draws the masks review image from the masks stage's own files, turns committed verdicts into ledger lines (dc2's shape), and allows at most one retry per stage. The masks stage is hooked first. Still to wire:
- a `step` mode in the verdict workflow (the one place that holds `GEMINI_API_KEY`);
- a box pipeline that publishes `review_steps.zip` and re-runs a rejected stage with the note.

**Runner plumbing (built once, reused by SF-18 and SF-24).** Each secret lives in exactly one step of one workflow, and the stages hand off through files:

1. **Box draws.** gpu-smoke `review` draws each object's eleven views into `review_sheets.zip` (Vast and SSH keys only).
2. **Runner judges.** The new runner-only workflow `sceneagent-vlm-verdict` takes that run's id, downloads the sheets, and runs `sceneagent.vlm_review`: one Gemini call per sheet, a fixed prompt (ours, hashed into the output), JSON constrained to the three retries. It uploads `vlm_verdicts.json`. It holds `GEMINI_API_KEY` in the Gemini step only, has read-only permissions, and never connects to the box. A failed call is recorded per object, never guessed.
3. **The verdicts are committed** as `sceneagent/verdicts/vlm_verdicts.json` (the box receives only `sceneagent/` and the Fruits video), so the model's output is visible in review before anything acts on it.
4. **Box applies.** gpu-smoke `review_vlm` copies the checkout as usual and runs `bin_review --verdicts`: the box first checks that the scene is still placed as in the judged run (same objects, same fit distances, recorded in the verdicts as `judged_placement`) and refuses otherwise, then validates every field, runs the VLM's retry per object (an object with no verdict is not retried, and the rule is not substituted), logs the rule's choice beside it, and writes `bin_review_vlm.json` and the `_vlm_reviewed` score.

**Legible sheets first.** Run 42934419274's sheets were checked by eye before any Gemini call.

- **Isolated views:** each view was framed around every Gaussian, so a few floaters left small objects a dot in five of eight views (object 7). The picture is now framed on the mesh. Gaussians outside that frame are not drawn, but they still count in the measured IoU, which is identical to before over 1,600 random renders.
- **Overlay:** the whole photo shrunk into one tile left small objects about 15 px wide. It is now cut to the object and enlarged.

The rule's choices cannot change, so the first round needs a fresh `review` run for new sheets, and that run should reproduce 0.3268 before review and 0.3235 after.

**How the first round is read (set before the number).**

- **What it measures:** a retry is still kept only when the fit distance improves, and that test no longer tracks GT (phase A). So this round measures only which objects Gemini sends to a retry, not whether it judges fits better.
- **Where the signal is:** where Gemini and the rule agree, the outcome is identical, so the signal is the per-object GT change on the objects where they disagree, reported with `agrees_with_rule`. Gemini choosing `keep` everywhere ties 0.327 and says nothing. Beating 0.323 by retrying less is not evidence that Gemini sees better.
- **Next round:** Gemini judges each retry's before and after views, replacing the fit-distance acceptance.
- **Check before reading the score:** `score_before_review` must be 0.3268, and each object's `fit_total` must match the judged run's. Otherwise the verdicts were applied to a different scene (the bin dir is shared across branches).

**Round 1 result (2026-11-11).** Gemini judged the sheets of review run 42939458339 (verdict run 42941121692: 14 calls, none failed) and `review_vlm` applied them (run 42943461864). The placement matched the judged one: 0.3268 before review, and every object's `fit_total` identical.

| Review | F1@1cm | Matched Chamfer |
|---|---|---|
| None | 0.3268 | 0.0211 m |
| Rule (12 `icp_flip`, 2 `keep`) | 0.3235 | 0.0237 m |
| Gemini (chose 7 `keep`, 2 `icp_flip`, 5 `sam3d_next_view`; object 13 has no other view, so 8 keep, 2 flip and 4 regenerations ran) | 0.3267 | 0.0211 m |

- **They agree on 3 of 14.**
- **Gemini avoids the rule's harm.** The rule's flips cost GT 24 (crate, 0.083 → 0.005) and GT 38 (0.395 → 0.372), against small gains on three others. Under Gemini all five stayed as placed: four keeps, and one rejected regeneration (object 5, GT 38: silhouette 0.569 → 0.528).
- **Gemini's regenerations, one at a time:**
  - object 8, the two fused fruits the rule kept: GT 6 0.525 → 0.570;
  - object 0, the bottle: GT 29 0.333 → 0.290;
  - object 4: GT 19 0.173 → 0.170;
  - object 5: regenerated, rejected (silhouette fell), unchanged.
  - The three kept were kept because their silhouette improved, and object 0's did while its GT fell: the same acceptance problem phase A found for the fit distance.
- **Run-to-run spread: one retry outcome flipped.** The first `review_vlm` (run 42942342408) kept 2 retries and scored 0.3270 and 0.0208 m; the rerun kept 3 and scored 0.3267 and 0.0211 m. SAM 3D regeneration is not bit-identical, so one regeneration crossed the silhouette bar in one run and not the other. The first run's per-object file was not published, so which one is unknown, and **the per-object regeneration effects above are not stable from run to run.** The next round needs deterministic regenerations or more than one run per verdict set.
- **Reading, as set in advance:** Gemini ties no review and beats the rule by retrying less. That is not yet evidence that it judges fits better. It does pick the right retry on two errors the rule cannot see (merged objects, wrong shape), and its retries are then accepted by a test that does not track GT.
- **Next:** Gemini judges each retry's before and after views, so acceptance is the VLM's, not the silhouette's or the fit distance's.

**Round 2: the VLM accepts retries (set before any number).** Round 1 left acceptance with the old tests, which do not track GT. In round 2 Gemini also accepts:

1. **Retry verdicts:** a fresh Gemini run on review run 42939458339's sheets, with `judged_placement`. The round 1 file is refused by the guard.
2. **Box builds candidates** (gpu-smoke `review_candidates`, `bin_review_pairs build`):
   - runs each chosen retry once and saves the result under `reviewer/pairs/<hash>/`, where the hash covers every saved cloud;
   - draws each changed object's before and after sheets as a blind pair, A and B in an order from a recorded seed (`PAIR_SEED`);
   - writes `candidates.json`, which maps A and B back and stays out of the model's input.
   - A flip that moved nothing is not drawn or judged. A flip candidate is still the best of three half-turns by fit distance, so Gemini sees only the fit-distance winner.
3. **Gemini judges pairs** (`sceneagent-vlm-verdict`, mode `accept`): one call per pair, A then B, with its own prompt (hashed into the output). The prompt never says which side is the retry. The output names the candidate hash.
4. **Box applies** (gpu-smoke `review_accept`, `bin_review_pairs apply`): loads the saved candidates by hash, refuses on any hash, cloud or placement mismatch, never regenerates, and keeps the side Gemini judged better.

**Metric, set in advance.** With about five retried objects, scene F1 barely moves, so the measure is per decision:

- **Oracle (evaluation only, ours):** keeping a retry is better when keeping it alone raises scene F1@1cm. It is computed in the apply step, after Gemini has judged, and nowhere earlier.
- **Report:** how often Gemini's acceptance agrees with the oracle, and how often the old test does, on the same candidates (ties excluded). Also scene F1 for no review, test-accepted, Gemini-accepted and oracle (an upper bound).
- **Caveat:** n is about five, so the result is indicative only.

**Round 2 result (2026-11-11).**

- **Retry verdicts** (run 42947401149, judged on review run 42939458339's placement): keep 9, `icp_flip` 1, `sam3d_next_view` 4. Object 13 has no other view, so 3 regenerations and 1 flip ran (candidates run 42963190714, hash `e03b08b8…`). The flip moved nothing and was not judged, leaving 3 blind pairs.
- **Pair verdicts** (run 42964570514): Gemini picked A once and B twice, which maps to the retry each time. Its reasons name visible flaws: an extra handle with no points on the bottle, a silhouette shifted left on object 4, a double-lobed mesh on the fused fruits. The silhouette test also keeps all three, so the two accepted scenes are the same.
- **Applied and scored** (run 42965157633):

| Acceptance | F1@1cm | Matched Chamfer |
|---|---|---|
| None | 0.3268 | 0.0211 m |
| Silhouette/fit test | 0.3272 | 0.0211 m |
| Gemini | 0.3272 | 0.0211 m |
| Oracle (upper bound, evaluation only) | 0.3291 | 0.0208 m |

| Object | Retry | Gemini | Test | Oracle | F1 change if kept alone |
|---|---|---|---|---|---|
| 8 (two fused fruits) | regenerate | keep | keep | keep | +0.0023 |
| 0 (bottle) | regenerate | keep | keep | reject | −0.0016 |
| 4 | regenerate | keep | keep | reject | −0.0003 |

**Reading, against the measure set in advance.**

- **Agreement with the oracle:** Gemini 1 of 3, the test 1 of 3. With n = 3 this says nothing about which judge is better. It does show Gemini does not yet beat the test.
- **The pairs were not matched (a flaw in this round's design).** For a regeneration, the after sheet's registration overlay is drawn on the view the new mesh came from, so A and B show different photos and masks. The pair prompt told Gemini the two "differ only in the object's mesh and its placement". That was false for all three judged pairs, and each mesh appears on its own source view, where it tends to look like a good fit.
- **The bottle is one data point with two explanations, neither established.** Gemini preferred the regenerated bottle (one clean bottle, where the old mesh had a stray handle). Against GT it is worse (−0.0016).
  - Explanation 1: the cleaner mesh is placed or sized worse. The check is inconclusive: the regenerated bottle is 27% smaller, but the fused fruits shrank 35% and improved.
  - Explanation 2: the unmatched overlay above made the retry look better than it is.
  - No general claim about what the sheets show is made from this.
- **What the oracle bound says:** the most any judge could add on these candidates is 0.3268 → 0.3291. Better acceptance alone cannot move the bin much. The retries themselves, especially regeneration from a weaker view, are now the limit.
- **Next (round 3), in order:**
  1. Matched pairs: draw both sheets' overlays on the same views (the original view, and the regeneration's view, on both sheets), and correct the pair prompt.
  2. More decisions per run: candidates for all 14 objects, with Gemini choosing among before, flip and regenerate.
  3. A placement-focused view in the sheets (mesh against points at true scale, side by side).

**Round 3, step 1: matched pairs (2026-11-11).** `review_object` takes the views the registration overlay is drawn on. The pair builder passes the same ones to both sheets: the original view and, for a regeneration, its own view, which appears as `registration_overlay_2` in the sheet's empty slot. Both sheets now show the same photos and masks, so the pair prompt's "differ only in the mesh and its placement" is true, and the prompt names the second overlay. Normal review is unchanged (one overlay on the record's own view, same IoU). The two sheets share views and masks; their crops still differ, because each is cut around its own mesh.

**Round 3, step 1 result (2026-11-11).**

- **Candidates** (run 42976670426, hash `e6f60899…`): same retry verdicts as round 2 (`726278c4…`), regenerations re-run. 3 regenerations (objects 0, 4, 8) and 1 flip that moved nothing (object 9, not judged). Each regeneration's pair shows two overlays: the original view and the regeneration's view.
- **Pair verdicts** (run 42977750829): Gemini picked B on all three. The A/B order was the same as round 2. That means taking the regeneration for objects 0 and 8 and keeping the original for object 4.
- **Applied and scored** (run 42977960610). Compare within this round only: the regenerated meshes differ from round 2's (SAM 3D is not bit-identical). The always-accept score moved 0.3272 → 0.3284 between rounds with the same decisions.

| Policy | F1@1cm | Matched Chamfer |
|---|---|---|
| Never accept | 0.3268 | 0.0211 m |
| Always accept (what the silhouette/fit test did here) | 0.3284 | 0.0206 m |
| Gemini | 0.3273 | 0.0207 m |
| Oracle (best F1 pick; evaluation only)\* | 0.3303 | 0.0206 m |

\* The oracle maximises F1, not Chamfer: always accept has a slightly lower Chamfer (0.02057 m against 0.02065 m).

| Object | Retry | Gemini | Test | Oracle | F1 change if kept alone | Round 2 change |
|---|---|---|---|---|---|---|
| 8 (two fused fruits) | regenerate | keep | keep | keep | +0.0025 | +0.0023 |
| 4 | regenerate | reject | keep | keep | +0.0011 | −0.0003 |
| 0 (bottle) | regenerate | keep | keep | reject | −0.0019 | −0.0016 |

**Reading.**

- **The test was "always accept".** It accepted every changed candidate in both rounds, so its 2 of 3 here only counts how many regenerations happened to be good. Read it as: on these candidates, always accepting beat Gemini (0.3284 against 0.3273). Three pairs cannot rank the judges.
- **The bottle repeated with matched overlays.** Gemini again preferred the regenerated bottle and named the mesh shape and silhouette. Against GT it is again worse (−0.0019). Round 2's unmatched-overlay explanation is no longer needed for the bottle. The placement/scale explanation is still not established.
- **Object 4's oracle label is unstable.** It changed sign between rounds on a different regenerated mesh, so it is within regeneration noise. Objects 0 and 8 kept their sign.
- **Object 4's pick is model variance.** A repeat verdict on the same pairs and prompt (run 42979528622, not applied) picked A, B, B → B, A, B: object 4 flipped to the regeneration, which the oracle prefers, and objects 0 and 8 held with the same reasons. With that repeat, Gemini's picks equal always accept (0.3284) and agree with the oracle on 2 of 3. So on this pair Gemini's choice moves between runs, and one run per pair is not a stable verdict.
- **Position habit:** the repeat answered B, A, B on the same order, so "B on 3 of 3" was not a fixed habit. Two runs cannot rule out a lean.
- **What it says overall:** Gemini's reasons describe mesh plausibility (a clean bottle, a single fruit, a cap on), while the oracle scores geometry against GT. Three decisions per run cannot separate judges, and one call per pair is not stable. These support round 3's next two items, more decisions per run and a placement-focused view, in either order.

**Redaction check (dc4).** The literal key is not available to us, so the check is a search of the job log, the verdict artifact and the uploaded `sf4-artifacts` for the Gemini key shape (`AIza` followed by 35 key characters), and for the string `GEMINI_API_KEY=`.

Why this shape: the GPU workflow pins `secrets.VAST_API_KEY` to one reference and allows only the pipeline choice as input, so a second box step or a run-id input there would loosen two security pins. The commit step costs one hop and makes the model's output auditable. SF-19 is measured against both 0.327 (no review) and 0.323 (the rule).

Canvas rows: "Visual review" (agent lane), "Orchestrator", and the section 3.1 "Agent repair loop".

## Phase C: full §3.4 physics on the bin (9 points)

The paper's §3.4 in full, on every bin object. Nothing in this phase changes `fitted_splat` or the mlp-physics-prior files.

**The prior is Sandeep's, used as it is.** It lives in [`sandy-t-shelvan/physics-prior-mlp`](https://github.com/sandy-t-shelvan/physics-prior-mlp) and is pinned by commit (`b052009` on 2026-11-10). Its two notes are `paper/physics_prior_mlp.pdf` and `paper/mlp_physx_physgaussian.pdf` ("Using the MLP prior with PhysX and PhysGaussian").
- **Checkpoint (the model):** [`data/mlp/physics_prior_splat.npz`](https://github.com/sandy-t-shelvan/physics-prior-mlp/blob/b052009284d7794fc38fb8d158d5eb5c9fb1a7cb/data/mlp/physics_prior_splat.npz), epoch 10 of a 14-epoch run.
  - 260,290 bytes, SHA-256 `7d737d52e82e1363efa7dbdb0dd1986b6f9d5d3461829bac9420802cfc09c125`.
  - Layers: `w1` 912×32, `w2` 32×32, `w_out` 32×33 (float64), with the 10 material, 6 body and 8 collision names stored inside.
  - Runs download it from that commit and refuse it if the hash differs, the same check the SAM vit_h weights get. It is not copied into this repo.
  - `data/mlp/train_report_splat.json` is the run's scores, not the model: `epochs_ran` 14, `checkpoint_epoch` 10, `n_fitted` 3736.
- **Training data:** 3736 fitted GSO and ABO objects (YCB 0), labelled by `gemini-3.8-flash` as the paper does.
- **Test scores (n = 323):** material accuracy 0.889, body 0.991, collision 0.768.

So it is not retrained here. Its rules, from the PhysX/PhysGaussian note:

1. **Input.** The input is the object's fitted splat, not the scene splat: the mesh is rendered as 16 orbit views (8 azimuths at 18° and 50°), registered by COLMAP and fitted by splatfacto for 600 steps. Each Gaussian is then packed into one row of 912 columns:
   - 0–383: DINOv2 ViT-S/14;
   - 384–895: OpenCLIP ViT-B-16 with the OpenAI weights;
   - 896–911: 16 geometry values (xyz, 3 exp scales, sigmoid opacity, quaternion, f_dc_0..2, log volume, anisotropy).

   Opacity must vary: its standard deviation must be at least 10⁻³, and it must not be the constant 0.99.
2. **Inference.** The MLP (32 tanh, 32 tanh, 33) runs on every in-mask Gaussian, and the 33 outputs are averaged. Averaging the inputs first is a different function and is not used. Sigmoid is applied to rigidity and deformability after the mean.
3. **Outputs.** The prior gives material (10), body (6), collision (8), log mass, log density, rigidity and deformability. Slots 28–32 are masked. **Friction and restitution are null.** Young's modulus, Poisson's ratio and the render material are unpublished, so they are recorded as null, not invented. A class tie returns no class. An unknown class has its weight × 0.4, and the keyword proposal is passed in by the caller.
4. **Simulators.**
   - Rigid, static or kinematic bodies become one **PhysX** actor with the collision primitive, mass and density. Friction and restitution are PhysX defaults unless the VLM sets friction.
   - Soft, cloth or fluid bodies, or any object with deformability ≥ 0.65 (that rule wins), become **PhysGaussian** particles. The particles are the same Gaussians, rigidity sets stiffness, and the MLP mass is the particle mass.
   - PhysX owns contact.
   - Each frame steps PhysX, then PhysGaussian under that contact, then renders.

| Item | Done when |
|---|---|
| SF-33 (KAN-39): fitted splat per object | Each SAM 3D mesh goes through the prior repo's own fitted-splat path (orbit, COLMAP, splatfacto 600 steps, pack), called as it is; one `(N, 912)` `prior_input` per object with varying opacity |
| SF-20 (KAN-21): equation (2) three-source vote | Prior (per-Gaussian mean, `weights: "ours"`) + VLM + keywords per object, unknown × 0.4, tie → no class |
| SF-21 (KAN-22): VLM material audit | Audit recorded per object; it may replace the class |
| SF-34 (KAN-40): full property set | Material, body, collider, log mass, log density, rigidity and deformability from the prior and vote. Friction from the VLM, else the PhysX default (recorded as default). Restitution at the PhysX default. E, ν and the render material null. Body routed to PhysX or PhysGaussian by the rule above. Each field's source recorded |
| SF-35 (KAN-41): USD physics export and settle test | One OpenUSD stage: PhysX actors with collider, mass and density. It settles in Isaac Lab (maximum speed below 1 cm/s after 2 s, nothing below the ground) and loads and settles in MuJoCo. PhysGaussian objects, if any, are marked for Cut 4 (SF-40) |
| SF-22 (KAN-23): inspector report written by the model | `fix_report_from_model: true` on the exported bin |

**Watch:** the prior's fitted set has no YCB objects, and the bin is all YCB. Its test scores are on GSO and ABO. The YCB catalogue masses are a check on the bin only, never an input.

## After Cut 3

| Cut | Jira | Milestones |
|---|---|---|
| 4 | KAN-34 | **M4. Full articulation (paper Figure 8).** On Mailbox and a capped marker: PartSAM parts; a VLM missing-geometry check; CubePart completion appended; Articulate Anything joints and ranges from those parts; a joint sweep in Isaac Lab; joints in the USD and the scene state. **M5. Soft bodies and fracture.** A deformable object simulates as PhysGaussian particles under PhysX contact, per the prior's PhysX/PhysGaussian note; the Warp thin-sheet tear and the Blast ceramic shatter run as authored scenes. |
| 5 | KAN-35 | **M6.** Video to sim on PutCupInBowl, PutMarkerInCup, ThrowAwayTrash and Mailbox, with the background completed and oriented. **M7.** The MessyKitchens score file (all-pairs matching + global mapper; PR #43: same registration, about 4× faster on Fruits). **M8.** MVP acceptance: one command on an unseen video, no hand edits. |

After the MVP: 4D and 4DCodeBench (§3.6), sisters and hybrid scenes (§3.8–3.9), Meshy, Unreal, the LangSplat-style codebook (SF-17), and policy (SF-25, SF-26, pi0.5 LoRA).

## Decisions for Sandeep

1. **Order inside Cut 3: settled (2026-11-10), A → B → C.** Sandeep left it to software and system-design principles:
   - **Fix the input first.** B and C both consume the grouped objects and their meshes, so A goes first.
   - **Build shared plumbing once.** B's Gemini runner job (secret on the runner only, artifact hand-off, `vlm_calls`, redaction) is reused by C's vote and audit, so B goes before C.
   - **One change per experiment.** Within A, κ=2, then the container split, then the pose rescale, each scored on its own against the 0.079 baseline.
2. **VLM: settled (2026-11-10).** The Gemini API, `gemini-3.8-flash`, with the key only from the GitHub secret `GEMINI_API_KEY` (KAN-15 approval). The GPU workflow forwards no secrets to the box, so the VLM calls run on the GitHub runner, on the review sheets and renders the box publishes. Every run records `vlm_calls`, and no key string appears in logs or artifacts.

   **Agents: same API (Sandeep, 2026-11-11).** The agents use the same Gemini API as the VLM: the same client (`harness.GeminiHarness`), the same model (`gemini-3.8-flash`), the same key, and the same runner-only workflow pattern. That covers the orchestrator plan (SF-24 part 2, `harness.request_plan`), the step-runner visual QA (SF-18) and the reviewer (SF-19). One adapter means one place for key handling, redaction and `vlm_calls` counting.

   Figure 5's six harness products stay recorded and uncalled (`harness_status`). The paper's best bin number, 60.75%, is with Codex as the harness, so our agent results are labelled as Gemini's and not presented as the same agent's.
3. **MLP prior: settled (2026-11-10).** Sandeep's prior is already trained the paper's way (GSO and ABO, `gemini-3.8-flash` labels), so it is used as it is from `physics-prior-mlp` at a pinned commit, with no retraining.
4. **MessyKitchens data (Cut 5).** Is the shared capture and its GT available to us?
5. **Box time.** Phase A alone is about six GPU runs of 5–15 minutes each, and 58301075 has to be running.

## Risks

- **Grouping may need masks we don't have.** If GroundingDINO's boxes split or merge objects, fixing the
  grouping rule only goes so far; a better prompt or SAM's automatic masks may be needed.
- **VLM latency and cost** per object per retry; a sheet per object keeps it to 25 calls a pass.
- **The kitchen data** may not be public.
- **24 GB GPU.** SAM 3D peaks at 19.3 GB; anything else on the box at the same time can push it over.
