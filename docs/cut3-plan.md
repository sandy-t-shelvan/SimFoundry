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

**Cut 3 ends with a bin score above the paper's SimFoundry baseline (6.9% F1), produced with a VLM in the
review loop, plus one more section 4 protocol started.** The number stays ours; the paper's numbers stay
the paper's. A score file is written whatever the number is.

Why this goal: the bin is the one protocol we can already score end to end, and the gap to 6.9% and to
60.75% is now measured problem by problem. The plan of record's §9.5 puts the kitchen next; policy and 4D
come after an inspector accept, which needs better placed objects first.

## Rules carried over

- Paper numbers stay the paper's; ours are labelled ours in the evidence.
- No stand-ins (KAN-13 comment 10318). No constant tuned to fit a file: the ours_iteration_1 gate, the
  association cuts and the trim stay as they are.
- `fitted_splat` and the MLP physics prior files are not touched.
- Each GPU-touching PR runs gpu-smoke on its own branch before merge; one PR per feature.
- The box is 59601312; sessions never start, stop or rent instances.
- **New:** dispatch gpu-smoke runs one at a time. A newer pending run cancels an older pending one (that
  is how place run 4 was lost on 2026-11-10).

## Sizing

Same unit as Cut 2: one point is roughly one PR with its tests and, where marked, a GPU run.

| Phase | Points | Cumulative | Ends with |
|---|---|---|---|
| A. Bin: data quality | 6 | 6 | Floaters filtered, grouping fixed, pose chain fixed; bin rescored |
| B. VLM in the loop (SF-18, SF-19 VLM verdict) | 6 | 12 | Reviewer decisions made by a VLM from the drawn views; bin rescored |
| C. Physics on the bin (Cut 2 phase 6) | 5 | 17 | Mass and friction per object; USD export with physics |
| D. Second protocol: MessyKitchens | 6 | 23 | Kitchen capture through the chain; first kitchen score file |
| E. Videos to sim (cup, marker, trash) | 5 | 28 | Video path with parallel mapping; placed scene per video |
| F. Policy (stretch) | 8 | 36 | Sisters, scripted expert, first episodes |

**Proposed Cut 3 line: phases A–D (23 points).** E and F are stretch, then Cut 4.

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

## Phase C: physics on the bin (5 points)

| Item | Done when |
|---|---|
| SF-20 (KAN-21): equation (2) three-source vote | Prior + VLM + keywords on each bin object; prior labelled `weights: "ours"` |
| SF-21 (KAN-22): VLM material audit | Audit recorded per object |
| USD export of the placed bin | Meshes, poses, mass and friction in one stage; loads in MuJoCo, objects settle |
| SF-22 (KAN-23): inspector report written by the model | `fix_report_from_model: true` on a bin object |

## Phase D: MessyKitchens (6 points)

| Item | Done when | GPU |
|---|---|---|
| Fetch the shared capture and GT scans | Manifest with SHA-256s and licence; if the data is not public, this phase becomes a decision (decision 3) | no |
| Run the chain on the capture | Video path: all-pairs matching + global mapper (PR #43: same registration, ~4× faster on Fruits) | yes |
| Kitchen score file | `metrics.aggregate_messy_kitchens`: one rigid transform, 100,000 points per object, three seeds; any dropped object named | yes |

## Phase E (stretch): videos to sim

PutCupInBowl, PutMarkerInCup and ThrowAwayTrash through the bin chain with the 300-frame rule and the
global mapper; one placed, physics-ready scene each. Mailbox waits for articulation (SF-10).

## Phase F (stretch): policy

SF-25 sisters, SF-26 scripted expert on the phase E scenes, then the pi0.5 LoRA (rank 16/32 as printed).
Plan of record §9.5 item 6: only after an inspector line with `fix_report_from_model: true`.

## Not in Cut 3

4D and 4DCodeBench (§3.6), fracture (§3.5), Meshy sisters, Unreal, hybrid scenes, CubePart, the
LangSplat-style codebook (SF-17), and the official-release row.

## Decisions for Sandeep

1. **Headline.** Bin quality first (A–C), as proposed, or move the kitchen (D) ahead of physics (C)?
2. **VLM key.** Run the VLM reviewer in the cloud session on the published contact sheets (no secret on
   the box), or add a Gemini key to the box's environment?
3. **MessyKitchens data.** Is the shared capture and its GT available to us? If not, D swaps for E.
4. **Box time.** Phase A alone is about six GPU runs of 5–15 minutes each; 59601312 has to be running.

## Risks

- **Grouping may need masks we don't have.** If GroundingDINO's boxes split or merge objects, fixing the
  grouping rule only goes so far; a better prompt or SAM's automatic masks may be needed.
- **VLM latency and cost** per object per retry; a sheet per object keeps it to 25 calls a pass.
- **The kitchen data** may not be public.
- **24 GB GPU.** SAM 3D peaks at 19.3 GB; anything else on the box at the same time can push it over.
