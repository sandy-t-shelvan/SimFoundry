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

The VLM call needs a key. The GPU workflow forwards no secrets by design, so either the VLM runs in the
cloud session on the published contact sheets (decision 2), or a key is added to the box.

Canvas rows: "Visual review" (agent lane), "Orchestrator", and the section 3.1 "Agent repair loop".

## Phase C: full §3.4 physics on the bin (9 points)

The paper's §3.4 in full, on every bin object. Nothing in this phase changes `fitted_splat` or the mlp-physics-prior files.

| Item | Done when |
|---|---|
| SF-33 (KAN-39): crop features | OpenCLIP ViT-B-16 (512-d) and DINOv2 ViT-S/14 per object from its masks, plus the 16 geometric values: the `(N, 912)` prior input |
| SF-20 (KAN-21): equation (2) three-source vote | Prior + VLM + keywords per object, with unknown × 0.4; the prior is the existing one, labelled `weights: "ours"` |
| SF-21 (KAN-22): VLM material audit | Audit recorded per object; it may replace the class |
| SF-34 (KAN-40): full property set | 10 materials, 6 body types, 8 colliders; log mass and log density, friction, restitution, rigidity; Young's modulus and Poisson's ratio from the class; soft-body flag at deformability ≥ 0.65; each field's source recorded |
| SF-35 (KAN-41): USD physics export and settle test | One OpenUSD stage with the PhysX schema. It settles in Isaac Lab (maximum speed below 1 cm/s after 2 s, nothing below the ground) and loads and settles in MuJoCo |
| SF-22 (KAN-23): inspector report written by the model | `fix_report_from_model: true` on the exported bin |

## After Cut 3

| Cut | Jira | Milestones |
|---|---|---|
| 4 | KAN-34 | **M4. Full articulation (paper Figure 8).** On Mailbox and a capped marker: PartSAM parts; a VLM missing-geometry check; CubePart completion appended; Articulate Anything joints and ranges from those parts; a joint sweep in Isaac Lab; joints in the USD and the scene state. **M5. Soft bodies and fracture.** A deformable object simulates as a soft body; the Warp thin-sheet tear and the Blast ceramic shatter run as authored scenes. |
| 5 | KAN-35 | **M6.** Video to sim on PutCupInBowl, PutMarkerInCup, ThrowAwayTrash and Mailbox, with the background completed and oriented. **M7.** The MessyKitchens score file (all-pairs matching + global mapper; PR #43: same registration, about 4× faster on Fruits). **M8.** MVP acceptance: one command on an unseen video, no hand edits. |

After the MVP: 4D and 4DCodeBench (§3.6), sisters and hybrid scenes (§3.8–3.9), Meshy, Unreal, the LangSplat-style codebook (SF-17), and policy (SF-25, SF-26, pi0.5 LoRA).

## Decisions for Sandeep

1. **Order inside Cut 3.** A → B → C as proposed, or physics (C) before the VLM loop (B)?
2. **VLM key.** Run the VLM in the cloud session on the published contact sheets (no secret on the box), or add a Gemini key to the box's environment?
3. **MLP prior.** The paper trains its prior on GSO, YCB and ABO with labels written by a VLM. Phase C applies the existing prior as it is. Matching the paper means retraining it, which changes the prior files. Yes or no?
4. **MessyKitchens data (Cut 5).** Is the shared capture and its GT available to us?
5. **Box time.** Phase A alone is about six GPU runs of 5–15 minutes each, and 58301075 has to be running.

## Risks

- **Grouping may need masks we don't have.** If GroundingDINO's boxes split or merge objects, fixing the
  grouping rule only goes so far; a better prompt or SAM's automatic masks may be needed.
- **VLM latency and cost** per object per retry; a sheet per object keeps it to 25 calls a pass.
- **The kitchen data** may not be public.
- **24 GB GPU.** SAM 3D peaks at 19.3 GB; anything else on the box at the same time can push it over.
