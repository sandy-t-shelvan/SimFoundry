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

**Result so far (2026-11-10, box 58301075).** These are F1@1cm and matched Chamfer, delivered against GT:

| Change | Runs | F1@1cm | Chamfer |
|---|---|---|---|
| Baseline, κ=1 (main) | place 42916527232 | 0.079 | 0.070 m |
| **κ=2 (PR #45, SF-30)** | association 42924495643, place 42925055261 | **0.125** | **0.046 m** |
| κ=2 + container split (PR #47, SF-31): **not kept** | association 42927798966, place 42929004562 | 0.109 | 0.032 m |

κ=2 keeps 65,972 of 78,121 Gaussians, and only κ differs from the baseline. **M1 (bin above 6.9%) is met by a change of ours.** κ=2 is ours; the paper uses κ=1. The container split (#47) set aside four groups that held real objects (GT 8, 24, 29 and 37). Their matches were lost, so F1 fell, and the split is not kept; the 0.35 m limit is not tuned to this scene. The pose rescale (#48) is measured next against 0.125.

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
3. **MLP prior: settled (2026-11-10).** Sandeep's prior is already trained the paper's way (GSO and ABO, `gemini-3.8-flash` labels), so it is used as it is from `physics-prior-mlp` at a pinned commit, with no retraining.
4. **MessyKitchens data (Cut 5).** Is the shared capture and its GT available to us?
5. **Box time.** Phase A alone is about six GPU runs of 5–15 minutes each, and 58301075 has to be running.

## Risks

- **Grouping may need masks we don't have.** If GroundingDINO's boxes split or merge objects, fixing the
  grouping rule only goes so far; a better prompt or SAM's automatic masks may be needed.
- **VLM latency and cost** per object per retry; a sheet per object keeps it to 25 calls a pass.
- **The kitchen data** may not be public.
- **24 GB GPU.** SAM 3D peaks at 19.3 GB; anything else on the box at the same time can push it over.
