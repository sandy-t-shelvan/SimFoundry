import {
  BarChart,
  Callout,
  CollapsibleSection,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  computeDAGLayout,
  useHostTheme,
  useState,
} from "cursor/canvas";

type Status = "done" | "partial" | "missing";

type FlowNode = {
  id: string;
  title: string;
  detail: string;
  status?: Status;
};

type FlowEdge = { from: string; to: string };

type GapRow = {
  sub: string;
  status: Status;
  paper: string;
  plan: string;
  changelog: string;
  ran: string;
  miss: string;
  addressed?: "fixed" | "still open";
};

type Block = {
  id: string;
  title: string;
  rollup: Status;
  open?: boolean;
  rows: GapRow[];
};

const TONE: Record<Status, "success" | "warning" | "danger"> = {
  done: "success",
  partial: "warning",
  missing: "danger",
};

const NODE_W = 158;
const NODE_H = 48;

function rollup(rows: GapRow[]): Status {
  if (rows.every((row) => row.status === "done")) return "done";
  if (rows.every((row) => row.status === "missing")) return "missing";
  return "partial";
}

const BLOCKS: Block[] = [
  {
    id: "3.1",
    title: "3.1 Agent loop",
    rollup: "partial",
    open: true,
    rows: [
      {
        sub: "Plan, manifest, ledger",
        status: "done",
        addressed: "fixed",
        paper: "Section 3.1: agents pass a plan, a manifest per step, and a ledger of accepted or rejected files. Figure 7 skill SCENEAGENT-RUN-PIPELINE.",
        plan: "EXPANSION_PLAN adds sceneagent.pipeline writing plan.json, one manifest per step, and ledger.json. Done when those files exist.",
        changelog: "Phase 1 file contract shipped. The repair loop is a separate row.",
        ran: "On Quebec 55073623 the loop wrote /workspace/fruits-agent/plan.json, one manifest per attempt, and ledger.json.",
        miss: "The file contract is not the paper's full repair loop. That loop is the next row.",
      },
      {
        sub: "Agent repair loop",
        status: "done",
        addressed: "still open",
        paper: "Section 3.1: write a plan, run a tool, write a manifest, append accept or reject, and on rejection repair and rerun. Agents render the scene and correct scale, orientation, and position.",
        plan: "DESIGN phases the inspector into phase 4 and the repair behavior into sceneagent.agent_loop. The opening 'phase 1 does not' list still says no Isaac and no MuJoCo.",
        changelog: "Latest entries: USDA rejected, MJCF accepted after MjModel loaded; SAM 3D attempt 2, PartSAM attempt 2, inspector attempt 2, ICP attempt 1, isaac_mesh attempt 1, and mujoco_mesh attempt 1 accepted. cubepart and articulate attempt 1 rejected.",
        ran: "Same ledger. gemini-2.5-flash returned HTTP 404. gemini-3.8-flash returned HTTP 200 and the class wood. The MuJoCo hull trial was accepted and did not place the object.",
        miss: "Catalog done means the files exist. Section 3.1's agents are six coding harnesses on model APIs (figure 5). This loop is local Python. The plan prompt already contains the JSON to emit, and order_plan forces material_vote then mujoco_load. The only repair is write_physics_mjcf, not a model looking at renders. The reviewer never sees registration overlays, eight isolated views, or two assembled views.",
      },
      {
        sub: "Alignment acceptance gate",
        status: "partial",
        addressed: "fixed",
        paper: "Section 3.1: accept a pose only when normalized median, p90, coverage, and silhouette all pass fixed thresholds. The numbers are not published.",
        plan: "The gate ran on 55073623 and catalog marks it partial. DESIGN phase 3 says choose thresholds on a held-out set and do not present them as the paper's. Missing thresholds stay undecided.",
        changelog: "ICP residual written. accept_alignment stayed undecided. TRIM_KEEP 0.7 is this implementation's overlap, not a published threshold. Silhouette raster union was empty.",
        ran: "icp_residual.json: normalized median 0.033103985722272695, p90 0.21563629180764, coverage 0.14616666666666667, silhouette None. Median nearest distance 0.04999262711481225. Full-rate residual icp_residual_fullrate.json: median 0.04373191475939739, p90 0.13867399840532532, normalized median 0.034355469766853275, coverage 0.09366666666666666, silhouette 0.17505342248569655.",
        miss: "The gate cannot decide with the paper's unpublished thresholds. ours_iteration_1 rejects the residual on coverage and silhouette. The cutoffs were not lowered.",
      },
    ],
  },
  {
    id: "3.2",
    title: "3.2 COLMAP and splatfacto",
    rollup: "partial",
    rows: [
      {
        sub: "COLMAP camera poses",
        status: "partial",
        addressed: "still open",
        paper: "Section 3.2: RGB sequences use COLMAP for camera poses. Agents may correct per-image color calibration.",
        plan: "Done only when model_analyzer prints a full-rate sequence and those cameras feed splatfacto. The 1 fps sample and 16-frame cap are the runner's, not the paper's.",
        changelog: "CUDA COLMAP 4.2.1 on Fruits.mp4, exit 0. Earlier Texas run was COLMAP 3.7 without CUDA. Row stays partial.",
        ran: "15/15 images, 603 points, mean reprojection 0.867273 px. Artifact /workspace/fruits-cuda-colmap/colmap/sparse/0. Texas record: 674 points, 0.892696 px. On 2 Nov 2026 the same clip at 30 fps: 452 frames, 452/452 registered, 37004 points, mean reprojection 0.800435 px, /workspace/fruits-fullrate-colmap/sparse/0. Those cameras trained splatfacto-big.",
        miss: "The 15-frame sample remains on disk. Color calibration from section 3.2 was not run.",
      },
      {
        sub: "splatfacto big",
        status: "done",
        addressed: "still open",
        paper: "Section 3.2: Nerfstudio splatfacto big trains the 3DGS. Scene S = (G_bg, {o_k}) with o_k = (P_k, M_k, T_k, theta_k, J_k).",
        plan: "Done when a checkpoint and one rendered view exist and the ledger accepts them. The plan's COLMAP rule still requires those cameras to feed this train.",
        changelog: "ns-train splatfacto-big, 2000 iterations, exit 0. Checkpoint step-000001999.ckpt. The iteration cap is this run's, not nerfstudio's default 30000. A later full-rate train used max-num-iterations 30000 and exited 0. Checkpoint step-000029999.ckpt, internal step 29999, 199655698 bytes.",
        ran: "Preset cull_alpha_thresh 0.005, densify_grad_thresh 0.0005. Catalog cites renders/train/rgb/0002.jpg. The run record counts 13 train renders. The 2000-step train used /workspace/fruits-cuda-colmap/colmap. The 2 Nov train used the 452-frame model and saved /workspace/fruits-splat-fullrate/unnamed/splatfacto/2026-11-02_063644/nerfstudio_models/step-000029999.ckpt, 268015 Gaussian means.",
        miss: "The 2000-step checkpoint remains. The 30000-iteration cap on the full-rate cameras finished, and the saved file is step-000029999.ckpt. That is still a background splat, not the paper's object with parts and joints.",
      },
      {
        sub: "Photogrammetry, LiDAR, generated 3DGS",
        status: "missing",
        addressed: "fixed",
        paper: "Section 3.2 also accepts existing 3DGS, photogrammetry, LiDAR, and generated scenes (World Labs Marble, NVIDIA Lyra, Tencent HY-World). Dynamic video is section 3.6.",
        plan: "EXPANSION_PLAN adds a reader. Polycam is a capture source, not a repository. Done when one non-video asset passes the ledger.",
        changelog: "Not shipped. The unreleased list does not claim these importers.",
        ran: "Video and ZED only. No photogrammetry, LiDAR, or generated-splat file was accepted.",
        miss: "Named in the plan, not implemented. No importer is specified beyond 'the caller's format.'",
      },
      {
        sub: "Example video metadata",
        status: "partial",
        addressed: "still open",
        paper: "Section 3.2 starts from an RGB sequence.",
        plan: "sceneagent.video records byte size and mvhd duration. Partial until the splat view exists.",
        changelog: "Fruits.mp4 is 1,733,022 bytes and 15.067 s. Poses exist for the 1 fps sample.",
        ran: "Header duration is known. The splat view is the separate splatfacto row.",
        miss: "Duration is not a reconstructed scene.",
      },
    ],
  },
  {
    id: "3.3",
    title: "3.3 Association and meshes",
    rollup: "partial",
    open: true,
    rows: [
      {
        sub: "SAM masks",
        status: "done",
        addressed: "still open",
        paper: "Section 3.3: SAM masks every tenth view by default. Across their scenes the stride ranges from 1 to 10.",
        plan: "The section 3.3 evaluation table records the SAM vit_h masks. Catalog row is done.",
        changelog: "SAM vit_h on every 10th frame of the 15-frame sample. Exit 0.",
        ran: "0001.jpg has 33 masks. 0011.jpg has 19. Files 0001_mask.png and 0011_mask.png.",
        miss: "Two frames of a 1 fps sample, not a full-rate scene.",
      },
      {
        sub: "GroundingDINO",
        status: "missing",
        addressed: "still open",
        paper: "Not in section 3.3. The website step 02 says GroundingDINO + SAM assign semantic features to each Gaussian.",
        plan: "Named beside SAM. Masks and the two feature vectors exist. GroundingDINO did not run, and the feature-on-Gaussian file is still missing.",
        changelog: "Not shipped.",
        ran: "Not started.",
        miss: "Not in the paper's section 3.3. Required only by the website. Still absent from the code.",
      },
      {
        sub: "OpenCLIP ViT-B-16",
        status: "done",
        addressed: "still open",
        paper: "Section 3.3: each masked crop is a 512-d vector. It is a physics-prior input, not the Gaussian assignment.",
        plan: "EXPANSION_PLAN forbids substituting another CLIP. The evaluation table records the OpenCLIP vectors in features.json. Catalog row is done.",
        changelog: "512-d vectors for 0001.jpg and 0011.jpg. No new checkpoint download.",
        ran: "openai weights, stored in /workspace/fruits-features/features.json.",
        miss: "Two crop vectors, not a feature on every Gaussian.",
      },
      {
        sub: "DINOv2 ViT-S/14",
        status: "done",
        addressed: "still open",
        paper: "Section 3.3: DINOv2 ViT-S/14 is the other physics-prior feature. It does not assign Gaussians.",
        plan: "Explicit rule: do not substitute DINOv3. The evaluation table records the DINOv2 ViT-S/14 vectors. Catalog row is done.",
        changelog: "dinov2_vits14, 384-d, checkpoint already on disk. Not DINOv3.",
        ran: "384-d vectors for 0001.jpg and 0011.jpg in features.json.",
        miss: "The forbidden DINOv3 swap was not made. The vectors are still one crop each, not the per-Gaussian MLP input in section 3.4.",
      },
      {
        sub: "LangSplat-style codebook",
        status: "missing",
        addressed: "still open",
        paper: "Section 3.3 follows LangSplat's approach of attaching features to Gaussians. It does not say that repository is executed as-is.",
        plan: "Read the method. Do not vendor the research repo until the feature lift is specified.",
        changelog: "Not shipped.",
        ran: "Not started. No language-Gaussian codebook.",
        miss: "The approach is named. The lift onto Gaussians is not specified and did not run.",
      },
      {
        sub: "Gaussian association cuts",
        status: "done",
        addressed: "still open",
        paper: "Section 3.3 equation (1): project inside the mask and within max(0.08, 0.03 z-hat) of the front of a cell of side min(16, round(s/192)). Grouping uses min(0.10, 0.6 e), containment 0.18, or Jaccard 0.08. Experiments use kappa = 1.",
        plan: "sceneagent.association implements the published cuts. Done when gaussians.json lists centers that passed equation (1).",
        changelog: "An early bullet moved this row to partial. It was not rewritten. Later text records the cut. Catalog status, which the matrix copies, is done.",
        ran: "27548 of 84993 Gaussians kept for the largest mask on 0001.jpg. 76629 projected, 48534 landed in the mask. Cell side 4 px. Artifact gaussians.json.",
        miss: "One mask on one frame. Mask grouping across views and the optional camera-consensus mode were not exercised. The early changelog line disagrees with catalog.",
      },
      {
        sub: "SAM 3D meshes",
        status: "done",
        addressed: "still open",
        paper: "Section 3.3: SAM3D generates object geometry. Trimmed similarity ICP starts from the SAM3D pose.",
        plan: "Hunyuan, TRELLIS.2, and Pixal3D are not SAM 3D. The evaluation table records 0001.glb. Catalog row is done.",
        changelog: "An early bullet says the token was rejected and the row stays missing. The later bullet records the mesh. Catalog is done.",
        ran: "0001.glb is 3,238,692 bytes, max side 720, plus 0001_gaussian.ply (10,153,632 bytes), render, and overlay. Ledger sam3d attempt 2 accepted. Not Hunyuan.",
        miss: "One frame. The stored SAM 3D rotation vector was absent, so ICP could not start from that pose.",
      },
      {
        sub: "Trimmed similarity ICP",
        status: "partial",
        addressed: "still open",
        paper: "Section 3.3: trimmed similarity ICP, one uniform scale, initialized from the SAM3D pose, aligns the mesh to P_k.",
        plan: "A mesh file and an ICP residual file are the done condition, shared with LaMa. DESIGN says the pose stays undecided without the paper's thresholds.",
        changelog: "Residual file 2,197 bytes. Ledger icp attempt 1 accepted the file. The section 3.1 gate stays undecided.",
        ran: "Scale 3.0328311154811303. Translation [-0.7755219189905452, 2.157819477783848, -0.28589919065664093]. Rotation refined from the export frame because the SAM 3D rotation vector was absent.",
        miss: "GAP_MATRIX matches catalog: partial, not done. Initialization was not the SAM 3D pose. The gate is undecided and the silhouette union was empty.",
      },
      {
        sub: "LaMa background infill",
        status: "missing",
        addressed: "still open",
        paper: "Section 3.3: after foreground Gaussians are removed, detected planes complete the background, with optional LaMa texture inpainting.",
        plan: "Optional. Stays missing until a plane-completion render shows an inpainted patch.",
        changelog: "Not shipped.",
        ran: "Not started. No plane-completion image.",
        miss: "Not started. Plane completion itself, which LaMa is optional on top of, also has no artifact.",
      },
    ],
  },
  {
    id: "3.4",
    title: "3.4 Materials, parts, articulation",
    rollup: "partial",
    open: true,
    rows: [
      {
        sub: "Material vote",
        status: "done",
        addressed: "still open",
        paper: "Section 3.4 equation (2): weighted vote of the physics prior, the VLM, and keyword priors. Unknown-class weights scale by 0.4. Ten classes, six body types, eight primitives. Soft body at deformability >= 0.65. Mass and density are log targets.",
        plan: "Phase 1 implements the 0.4 scale. The keyword dictionary is ours; the paper does not print it. Modulus lookups stay null.",
        changelog: "The vote code is in the phase-1 contract. The live class is recorded on the VLM row, not as a three-source vote.",
        ran: "No measured proposals were added. The only live Fruits class is the VLM row's wood.",
        miss: "Catalog marks the formula done. Equation (2) was not run as a vote of prior + VLM + keywords on Fruits Gaussians. The per-Gaussian prior does not exist.",
      },
      {
        sub: "Per-Gaussian MLP",
        status: "missing",
        addressed: "still open",
        paper: "Section 3.4: an MLP on DINOv2, OpenCLIP, and 16 geometric features, trained on VLM labels from GSO, YCB, and ABO. Nine numerical properties. Weights are not released. Targets are VLM labels, not measurements.",
        plan: "Do not train a stand-in and call it their prior. DESIGN repeats the 16 geometric features and leaves the weights unreleased.",
        changelog: "Listed under 'Not in this tree.'",
        ran: "Not started. Crop vectors exist. There is no per-Gaussian input and no network.",
        miss: "Correctly refused as a fake checkpoint. The 16 geometric features and nine heads are not an interface in the code.",
      },
      {
        sub: "VLM material proposal",
        status: "partial",
        addressed: "fixed",
        paper: "Section 3.4: a VLM is one vote source and may audit the class. Section 3.8 names Gemini 2.5 Flash for sister edits. The recorded review is signed by a coding agent.",
        plan: "Optional Gemini proposal through the existing path. Sister edits name Gemini 2.5 Flash.",
        changelog: "An early bullet says no class was saved. The later bullet records gemini-3.8-flash, class wood, and keeps the row partial because it is not the paper model.",
        ran: "gemini-2.5-flash HTTP 404. gemini-3.8-flash HTTP 200, class wood. Artifact /workspace/fruits-agent/material_vote.json.",
        miss: "One class from gemini-3.8-flash, not Gemini 2.5 Flash. Equation (2) needs a physics-prior vote, a VLM vote, and a keyword vote, then a second VLM audit. The client sends one image and one prompt. Friction, mass, density, and the coding-agent signature are absent.",
      },
      {
        sub: "PartSAM",
        status: "done",
        addressed: "fixed",
        paper: "Section 3.4: PartSAM decomposes each reconstructed object. A VLM inspects parts for missing geometry.",
        plan: "Phase 7. The PDF names PartSAM and prints no URL.",
        changelog: "0001.ply, 3,723,721 bytes. 1,168 masks at 0.65, 15 after NMS. Ledger partsam attempt 2 accepted.",
        ran: "0001.glb has 161,892 faces, so use_graph_cut was false. Batch size 4. NVIDIA apex was not built; FusedLayerNorm fell back to torch.nn.LayerNorm and the checkpoint loaded.",
        miss: "One mesh. No VLM pass over the 15 parts asking whether hidden geometry is missing, which is what sends the object to CubePart.",
      },
      {
        sub: "CubePart",
        status: "missing",
        addressed: "still open",
        paper: "Section 3.4: CubePart completes hidden geometry, such as a marker tip under the cap, and appends it to the mesh.",
        plan: "Phase 7, reviewed by the ledger. No scene file is specified beyond the marker example in the paper.",
        changelog: "catalog.py still says attempt 1 was rejected on disk and nothing was downloaded. That line is behind the 2 Nov log.",
        ran: "After the disk cleanup, examples/run_inference.py started and exited 1 at step 0 of 50. TypeError: QwenEmbedRope.forward() got multiple values for argument 'device' in hijack_qwenimage_multi.py. No part mesh. Ledger cubepart attempt 2 rejected. Free space after the failure was 115,921,776,640 bytes.",
        miss: "Installed, then the Qwen rope call crashed. The paper's missing-geometry VLM never looked at the 15 PartSAM parts, so CubePart was not asked for a hidden tip. catalog.py does not record attempt 2.",
      },
      {
        sub: "Articulate Anything",
        status: "partial",
        addressed: "still open",
        paper: "Section 3.4: Articulate Anything proposes URDF joints from image or text. Agents review the motion in simulation.",
        plan: "Phase 7. Do not treat SimFoundry stage 9 as this system.",
        changelog: "Attempt 1 rejected the extract. Attempt 2 accepted a PartNet table URDF. On 2 Nov 2026 attempts 3 and 4 rejected that table and a retry table for the fruit. The files were kept.",
        ran: "Both URDFs are 13408 bytes, robot name table, object 25959, links base, furniture_body, wheel, wheel_2, wheel_3, wheel_4, five fixed joints. conda env articulate-anything is present. CoTracker checkpoint is 204396415 bytes and wrote aug_Fruits.mp4, 13307277 bytes. sapien_simulate.py exited 139. Joint actor IndexError. feedback_score -1.",
        miss: "These are not joints of the 15 PartSAM fruit parts. The tool retrieves a PartNet asset and has no input for the SAM 3D mesh or those parts. Motion review did not finish because the renderer segfaulted.",
      },
    ],
  },
  {
    id: "3.5",
    title: "3.5 Fracture",
    rollup: "missing",
    rows: [
      {
        sub: "Warp, Blast, Newton",
        status: "missing",
        paper: "Section 3.5: parchment uses NVIDIA Warp cohesive fracture. Ceramics use NVIDIA Blast. Section 3.6 and the site use Newton for jelly and watermelon. These examples are authored scenes, not automatic USD export. Parameters are engineering estimates.",
        plan: "Phase 8: explicit example scenes only. 'Do not put fracture in the USD contract.' No input scene, no output file, and no split between Warp, Blast, and Newton.",
        changelog: "Not shipped.",
        ran: "Not started. Rigid and articulated bodies only.",
        miss: "Named, not specified. The paper's illustrated scenes are not in the plan as files to build, and their numbers are not ours to copy.",
      },
    ],
  },
  {
    id: "3.6",
    title: "3.6 4D and 4DCodeBench",
    rollup: "missing",
    rows: [
      {
        sub: "Dynamic scene and build script",
        status: "missing",
        paper: "Section 3.6 extends S to (K, E, G, {d_m}) with equation (3). Solvers are MPM, FLIP, and fracture models. A build script regenerates frames, meshes, and trajectories without reading the video. Section 4.2 scores B_09, Z_06, and Z_12. Figure 4 token stages are S0–S6.",
        plan: "One missing row tied to section 4.2. 'Do not report the paper's 99.09% dynamic IoU as our score.' The S0–S6 stage list and the regenerable build script are not artifacts in the plan.",
        changelog: "Not shipped. The 99.09% figure is called out as not this repo's score.",
        ran: "Not started. No 4D video, no MPM or FLIP run, no build script.",
        miss: "Not in the plan as a representation or a build. Figure 6's S3 SplatGraph and the 4DCodeBench checker stage are also absent from both the plan and the code.",
      },
    ],
  },
  {
    id: "3.7",
    title: "3.7 OpenUSD, simulators, viewer",
    rollup: "partial",
    open: true,
    rows: [
      {
        sub: "USD physics contract",
        status: "done",
        paper: "Section 3.7: OpenUSD rigid body, mass, collision, friction, restitution, density, and joints. Deformables, fracture, and fluids do not ride along in the file.",
        plan: "Phase 1 writes JSON with usd_written false. Phase 4 emits USD and leaves deformables out.",
        changelog: "phase1_place.usda opened in Isaac. Null mass or density stays a comment.",
        ran: "Isaac opened phase1_place.usda: opened=True, prims /World, /World/box, /World/bowl (3 prims).",
        miss: "The contract scene is a box primitive and a bowl primitive, not the reconstructed Fruits object. That mesh is the aligned-USDA row.",
      },
      {
        sub: "Isaac Lab empty stage",
        status: "done",
        paper: "Sections 3.1 and 3.7: experiments use Isaac Lab. The inspector reviews the exported scene.",
        plan: "EXPANSION_PLAN's dependency list still says install Isaac and smoke an empty stage, then 'Splatfacto waits.' Both already happened.",
        changelog: "isaaclab 0.54.2, Isaac Sim 5.1.0-rc.19, Isaac Lab v2.3.2. EMPTY_STAGE_OK. Host glibc 2.31; kit python uses the jammy 2.35 loader. python.sh on the host glibc segfaults.",
        ran: "/workspace/empty-stage-result.txt, exit 0.",
        miss: "This row is the empty stage only. It is not the Fruits inspector.",
      },
      {
        sub: "Isaac inspector",
        status: "done",
        paper: "Sections 3.1 and 3.7 and figure 7 skill SCENEAGENT-INSPECT-ISAAC: open the exported scene and accept or reject it.",
        plan: "EXPANSION_PLAN records the headless open of phase1_place.usda and the accepted ledger line. It still names inspection/isaac_lab.json as the output path.",
        changelog: "Kit python opened phase1_place.usda. Ledger isaac_inspector attempt 2 accepted.",
        ran: "opened=True, prims /World, /World/box, /World/bowl. Artifact isaac_inspector.txt.",
        miss: "The opened stage is primitives, not the reconstructed mesh.",
      },
      {
        sub: "Aligned mesh USDA",
        status: "done",
        paper: "Section 3.7: the exported scene carries the reconstructed object, and the inspector opens that stage.",
        plan: "The phase-1 USDA is described as box and bowl primitives. The aligned-mesh stage is a later catalog row. GAP_MATRIX matches catalog: done.",
        changelog: "phase1_aligned_mesh.usda is 5,339,577 bytes. Ledger isaac_mesh attempt 1 accepted.",
        ran: "opened=True, nprims=4: /World, /World/aligned_mesh, /World/aligned_mesh/geometry, /World/bowl. The run record counts 8,095 faces. The box primitive was replaced; the bowl was kept.",
        miss: "Isaac opened it. The alignment gate that should have accepted the pose is still undecided, so the export is ahead of section 3.1.",
      },
      {
        sub: "MuJoCo",
        status: "done",
        paper: "Section 3.7: exported scenes were also loaded in MuJoCo. Deformable behavior does not carry through USD.",
        plan: "Phase 4 is a USD load-check. The plan does not specify an MJCF repair when MuJoCo cannot parse USDA.",
        changelog: "An early bullet says the row stays missing after the USDA parse error. Later bullets and catalog say done: MJCF loaded, then one hull trial.",
        ran: "MuJoCo 3.17.0 rejected phase1_place.usda. phase1_place.xml loaded (nbody 3, nq 7, 30 steps). phase1_place_mesh.xml is the convex hull, 327 vertices, 650 faces, fit to the contract box's longest side of 0.096 m. Body center ended at z 0.18175340000000004 m, not inside the bowl. Catalog records the hull file and does not mention a trimesh.convex_hull crash.",
        miss: "Loaded is not a place success. The bowl's null mass uses the loader density. One trial is not a success rate.",
      },
      {
        sub: "Unreal Engine",
        status: "missing",
        paper: "Section 3.7 names Unreal as a third target. The paper says the scenes were tested there.",
        plan: "Confirm a USD load only after a person does it. No loader and no scene list.",
        changelog: "Not shipped.",
        ran: "Not started.",
        miss: "Named only. Not in the code.",
      },
      {
        sub: "Three.js viewer",
        status: "partial",
        paper: "Section 3.7 and figure 14: robot placement, background cropping, Omniverse insertion, pose and physics edits, sisters, parts, and an operator.",
        plan: "Extend scripts/interactive/light_editor. Do not add a second viewer. Phase 5 adds the experiment grid to that editor.",
        changelog: "The light editor is acknowledged. No entry says it opened the Fruits USDA.",
        ran: "The SceneAgent figure 14 viewer is not in the sceneagent package. The SimFoundry light editor was not pointed at phase1_place.usda or the aligned mesh.",
        miss: "Partial only because a different Three.js editor already exists. The paper's viewer on this capture was not started.",
      },
      {
        sub: "Experiment grid page",
        status: "partial",
        paper: "Figure 14: object sister by lighting, place success over ten trials. The website adds a 95% Wilson interval and nine lights.",
        plan: "Phase 1 HTML uses the nine light names and marks lighting as not simulated. Done for the shape, not for the website's pi0.5 cells.",
        changelog: "place_grid.json is one MuJoCo rollout. Wilson on 0 of 1 is 0 to 0.7934506856227626.",
        ran: "Lighting columns were not run. The single trial is not a place success.",
        miss: "A one-cell file is not figure 14. The ten-trial by nine-light grid was not run.",
      },
    ],
  },
  {
    id: "3.8",
    title: "3.8 Digital sisters",
    rollup: "partial",
    rows: [
      {
        sub: "Model-free mesh and texture",
        status: "done",
        paper: "Section 3.8 first method: bounded mesh deformation plus noise, hue rotation, mottling, and speckle, checked for extent, volume, and grasp. The grasp cutoff is not published.",
        plan: "sceneagent.sisters. The site's ±5% axis is included. The paper's numeric grasp gate stays unpublished.",
        changelog: "box.json and bowl.json. The box length +5% sister does not fit the bowl interior. No SAM 3D or Meshy call.",
        ran: "Contract primitives only, not the aligned SAM 3D mesh. Grasp band is not published.",
        miss: "Two primitive sisters. Texture baking on the reconstructed mesh was not run. The +5% box fails the fit check.",
      },
      {
        sub: "Meshy generative sisters",
        status: "missing",
        paper: "Section 3.8 second method: Gemini 2.5 Flash splits functional components, Gemini 3 Pro Image edits one component, Meshy image-to-3D lifts it. Meshy-6 retexture keeps geometry and UVs.",
        plan: "Do not label Hunyuan as Meshy. No open repository matched Meshy-6.",
        changelog: "Not shipped. Pipeline B cousins stay the SimFoundry path.",
        ran: "No Meshy call. No Gemini 3 Pro Image edit.",
        miss: "The forbidden Hunyuan substitution was not made, and Meshy was not run. Pipeline B is a different system.",
      },
      {
        sub: "Scene and task sisters",
        status: "partial",
        paper: "Figure 7 skills SCENEAGENT-SCENE-SISTERS and SCENEAGENT-TASK-SISTERS. Section 3.10 also randomizes placement. Scene sisters sample layouts around a task anchor.",
        plan: "Phase 1 writes template offsets and three task instructions. Pipeline B remains the generative layout path.",
        changelog: "Templates and horizontal offsets only.",
        ran: "No failure-driven sister from a real-robot recording. No layout sampled around a task anchor in Isaac.",
        miss: "Templates are not the figure 7 skills. Placement randomization in section 3.10 was not run.",
      },
    ],
  },
  {
    id: "3.9",
    title: "3.9 Hybrid scenes",
    rollup: "missing",
    rows: [
      {
        sub: "Captured object in an artist scene",
        status: "missing",
        paper: "Section 3.9 places one converted object into an artist-authored environment. The site shows kitchen, warehouse, chemistry lab, data center, and studio hybrids, often with Omniverse assets.",
        plan: "One short subsection: use the light editor after a captured mesh exists. No artist scene, no asset list, and no output file.",
        changelog: "Not shipped. Do not claim the hybrid videos were rebuilt.",
        ran: "Not started. The aligned mesh exists and was not inserted into an artist scene.",
        miss: "Named only. The paper's environments are not in the plan as scenes to assemble.",
      },
    ],
  },
  {
    id: "3.10",
    title: "3.10 Expert episodes and place grid",
    rollup: "partial",
    open: true,
    rows: [
      {
        sub: "Scripted place check",
        status: "partial",
        paper: "Section 3.10: the expert uses simulator state for approach, descend, grasp, lift, carry, lower, release, and retreat. Keep episodes whose object ends inside the target. Observations are depth-composited over the splat. Later experts are controllers written by a coding model.",
        plan: "The section text names those eight phases. The done condition that follows is a geometric inside-target check, and the Quebec run is a 30-step drop. 'Partial today: not an Isaac rollout.'",
        changelog: "Box at (0, 0, 0.1817534) m. Bowl geom centered at (0.25, 0, 0.2) m with half-extent 0.05 m. One trial.",
        ran: "place_grid.json. The box center is not inside the bowl. No robot, no eight-phase controller, no composited camera observations. The full-rate hull drop ended at z 0.18175340000000004 m with x about 0. Bowl center (0.25, 0, 0.2). inside_bowl false.",
        miss: "The plan names the expert and then specifies a different, smaller check. That smaller check did not succeed. The newer mesh missed the bowl the same way.",
      },
      {
        sub: "Wilson interval",
        status: "done",
        paper: "The website reports a 95% Wilson interval on each ten-trial cell. Figure 14 reports the count. Ten paired trials per cell.",
        plan: "Implement the interval on counts this repo produces. That does not make the cells a reproduction of figure 14.",
        changelog: "0 of 1, interval 0 to 0.7934506856227626. Not a ten-trial cell.",
        ran: "The function ran on the single MuJoCo trial.",
        miss: "The formula is done. The experiment it is for was not run.",
      },
      {
        sub: "pi0 / pi0.5 fine-tune",
        status: "missing",
        paper: "Section 4.4 fine-tunes pi0.5 with LoRA on simulated demonstrations. The eggplant table, including 40 episodes at 57/200 sim and 24/50 real, is the paper's result.",
        plan: "Phase 9. Report only runs we execute. Do not copy the eggplant table. Tool named as the public pi0 code.",
        changelog: "Not shipped.",
        ran: "Not started. No episode archive and no LoRA checkpoint.",
        miss: "Named, with a prohibition on copying the table, and no runnable data layout. Section 4.4 is not started.",
      },
    ],
  },
  {
    id: "Sec 4",
    title: "Section 4 results",
    rollup: "partial",
    rows: [
      {
        sub: "F1 at 1 cm and Chamfer",
        status: "partial",
        paper: "Section 4.1 equations (4) and (5), tau = 1 cm, Hungarian match, unmatched F1 = 0. The kitchen is MessyKitchens. The bin is GraspClutter6D hard/000091, 13 views, 20 YCB objects.",
        plan: "Functions are tested on synthetic clouds. DESIGN phase 1 says no GraspClutter6D and no MessyKitchens scores.",
        changelog: "An early bullet groups this with association as partial. The functions stay partial. No score file.",
        ran: "Synthetic clouds only. No kitchen score and no bin score.",
        miss: "The kitchen half of section 4.1 is excluded by DESIGN rather than designed. The formulas are not a reconstruction result.",
      },
      {
        sub: "GraspClutter6D hard/000091",
        status: "missing",
        paper: "Section 4.1 compares harnesses and SimFoundry on that bin. Their best F1 is 60.75%. SimFoundry in the paper is 6.9% F1 and 4.29 cm Chamfer. Figure 6 token stages are S0–S7.",
        plan: "Score only after meshes exist. Their 60.75% and 6.9% stay theirs. Figure 6's S0–S7 list is not a plan.",
        changelog: "Not scored.",
        ran: "Not started. The Fruits mesh is not this bin.",
        miss: "Not run. S3 SplatGraph clustering, from figure 6, is not in the plan or the code.",
      },
      {
        sub: "Official release",
        status: "missing",
        paper: "The PDF and the project site say the code, processed scenes, and example objects will be released. The PDF prints no git URL. Section 4.3 qualitative conversion has no separate catalog row. Section 4.2 is the section 3.6 row, not a second score here.",
        plan: "This tree is the SimFoundry fork plus the phase-1 package. It is not the Harvard release. 4.3 has no reproduction phase.",
        changelog: "Harvard's code is unreleased. Do not substitute another repository.",
        ran: "No upstream SceneAgent checkout and no qualitative scene set from section 4.3.",
        miss: "The release is missing because the paper's code is not this tree. Section 4.3 is not in the plan as work to do.",
      },
    ],
  },
];

const PAPER_NODES: FlowNode[] = [
  { id: "s1", title: "Stage 1", detail: "Repair 3.1–3.7", status: "partial" },
  { id: "s2", title: "Stage 2", detail: "Augment 3.8–3.9", status: "partial" },
  { id: "s3", title: "Stage 3", detail: "Demos 3.10", status: "partial" },
  { id: "b31", title: "3.1 Agent loop", detail: "partial", status: "partial" },
  { id: "b32", title: "3.2 COLMAP+splat", detail: "partial", status: "partial" },
  { id: "b33", title: "3.3 Association", detail: "partial", status: "partial" },
  { id: "b34", title: "3.4 Materials", detail: "partial", status: "partial" },
  { id: "b35", title: "3.5 Fracture", detail: "missing", status: "missing" },
  { id: "b36", title: "3.6 4D", detail: "missing", status: "missing" },
  { id: "b37", title: "3.7 USD export", detail: "partial", status: "partial" },
  { id: "b38", title: "3.8 Sisters", detail: "partial", status: "partial" },
  { id: "b39", title: "3.9 Hybrid", detail: "missing", status: "missing" },
  { id: "b310", title: "3.10 Expert demos", detail: "partial", status: "partial" },
  { id: "b4", title: "Section 4", detail: "results, partial", status: "partial" },
];

const PAPER_EDGES: FlowEdge[] = [
  { from: "s1", to: "b31" },
  { from: "s1", to: "b32" },
  { from: "s1", to: "b33" },
  { from: "s1", to: "b34" },
  { from: "s1", to: "b35" },
  { from: "s1", to: "b36" },
  { from: "s1", to: "b37" },
  { from: "s2", to: "b38" },
  { from: "s2", to: "b39" },
  { from: "s3", to: "b310" },
  { from: "b310", to: "b4" },
];

const SITE_NODES: FlowNode[] = [
  { id: "w1", title: "01 3DGS", detail: "partial", status: "partial" },
  { id: "w2", title: "02 Features", detail: "partial", status: "partial" },
  { id: "w3", title: "03 Physics", detail: "partial", status: "partial" },
  { id: "w4", title: "04 Parts", detail: "partial", status: "partial" },
  { id: "w5", title: "05 Joints", detail: "missing", status: "missing" },
  { id: "w6", title: "06 Sisters", detail: "partial", status: "partial" },
  { id: "w7", title: "07 USD pack", detail: "partial", status: "partial" },
  { id: "w8", title: "08 Three.js", detail: "partial", status: "partial" },
];

const SITE_EDGES: FlowEdge[] = [
  { from: "w1", to: "w2" },
  { from: "w2", to: "w3" },
  { from: "w3", to: "w4" },
  { from: "w4", to: "w5" },
  { from: "w5", to: "w6" },
  { from: "w6", to: "w7" },
  { from: "w7", to: "w8" },
];

const POLICY_NODES: FlowNode[] = [
  { id: "p1", title: "01 Real world", detail: "missing", status: "missing" },
  { id: "p2", title: "02 Twin+sisters", detail: "missing", status: "missing" },
  { id: "p3", title: "03 Policy eval", detail: "missing", status: "missing" },
  { id: "p4", title: "04 Expert revise", detail: "missing", status: "missing" },
  { id: "p5", title: "05 Fine-tune", detail: "missing", status: "missing" },
  { id: "p6", title: "06 Re-evaluate", detail: "missing", status: "missing" },
];

const POLICY_EDGES: FlowEdge[] = [
  { from: "p1", to: "p2" },
  { from: "p2", to: "p3" },
  { from: "p3", to: "p4" },
  { from: "p4", to: "p5" },
  { from: "p5", to: "p6" },
  { from: "p6", to: "p1" },
];

const FIG6_NODES: FlowNode[] = [
  { id: "f0", title: "S0 Setup", detail: "missing", status: "missing" },
  { id: "f1", title: "S1 Cameras", detail: "partial", status: "partial" },
  { id: "f2", title: "S2 2D masks", detail: "partial", status: "partial" },
  { id: "f3", title: "S3 SplatGraph", detail: "not in plan", status: "missing" },
  { id: "f4", title: "S4 Physics", detail: "partial", status: "partial" },
  { id: "f5", title: "S5 SAM 3D", detail: "one frame", status: "partial" },
  { id: "f6", title: "S6 Pose+USD", detail: "partial", status: "partial" },
  { id: "f7", title: "S7 Rendering", detail: "missing", status: "missing" },
];

const FIG6_EDGES: FlowEdge[] = [
  { from: "f0", to: "f1" },
  { from: "f1", to: "f2" },
  { from: "f2", to: "f3" },
  { from: "f3", to: "f4" },
  { from: "f4", to: "f5" },
  { from: "f5", to: "f6" },
  { from: "f6", to: "f7" },
];

const BENCH_NODES: FlowNode[] = [
  { id: "d0", title: "S0 Setup", detail: "missing", status: "missing" },
  { id: "d1", title: "S1 Video cam", detail: "missing", status: "missing" },
  { id: "d2", title: "S2 Static", detail: "missing", status: "missing" },
  { id: "d3", title: "S3 MPM/FLIP", detail: "missing", status: "missing" },
  { id: "d4", title: "S4 Meshing", detail: "missing", status: "missing" },
  { id: "d5", title: "S5 Rendering", detail: "missing", status: "missing" },
  { id: "d6", title: "S6 Checker", detail: "missing", status: "missing" },
];

const BENCH_EDGES: FlowEdge[] = [
  { from: "d0", to: "d1" },
  { from: "d1", to: "d2" },
  { from: "d2", to: "d3" },
  { from: "d3", to: "d4" },
  { from: "d4", to: "d5" },
  { from: "d5", to: "d6" },
];

const AGENT_NODES: FlowNode[] = [
  { id: "a1", title: "Harness API", detail: "6 named, 0 called", status: "missing" },
  { id: "a2", title: "Orchestrator", detail: "text prompt only", status: "partial" },
  { id: "a3", title: "Visual review", detail: "no step images", status: "missing" },
  { id: "a4", title: "VLM class vote", detail: "fallback plastic + 3 numbers", status: "done" },
  { id: "a5", title: "VLM audit", detail: "second pass absent", status: "missing" },
  { id: "a6", title: "Part VLM", detail: "no hidden geometry", status: "done" },
  { id: "a7", title: "Sister VLM", detail: "no image edit", status: "missing" },
  { id: "a8", title: "Policy VLM", detail: "no corrections", status: "missing" },
];

const AGENT_EDGES: FlowEdge[] = [
  { from: "a1", to: "a2" },
  { from: "a2", to: "a3" },
  { from: "a3", to: "a4" },
  { from: "a4", to: "a5" },
  { from: "a4", to: "a6" },
  { from: "a6", to: "a7" },
  { from: "a7", to: "a8" },
];

function FlowDiagram({
  nodes,
  edges,
  direction,
}: {
  nodes: FlowNode[];
  edges: FlowEdge[];
  direction: "horizontal" | "vertical";
}) {
  const theme = useHostTheme();
  const layout = computeDAGLayout({
    nodes: nodes.map((node) => ({ id: node.id })),
    edges,
    direction,
    nodeWidth: NODE_W,
    nodeHeight: NODE_H,
    rankGap: direction === "horizontal" ? 36 : 28,
    nodeGap: 12,
    padding: 8,
  });
  const pos = new Map(layout.nodes.map((node) => [node.id, node]));
  return (
    <div style={{ overflowX: "auto" }}>
      <svg width={layout.width} height={layout.height} role="img">
        {layout.edges.map((edge) => (
          <line
            key={`${edge.from}-${edge.to}`}
            x1={edge.sourceX}
            y1={edge.sourceY}
            x2={edge.targetX}
            y2={edge.targetY}
            stroke={edge.isBackEdge ? theme.stroke.primary : theme.stroke.secondary}
            strokeWidth={1}
            strokeDasharray={edge.isBackEdge ? "4 3" : undefined}
          />
        ))}
        {nodes.map((node) => {
          const at = pos.get(node.id);
          if (!at) return null;
          const fill = !node.status
            ? theme.fill.tertiary
            : node.status === "done"
              ? theme.diff.insertedLine
              : node.status === "missing"
                ? theme.diff.removedLine
                : theme.fill.secondary;
          const stroke = node.status === "partial" ? theme.accent.primary : theme.stroke.secondary;
          return (
            <g key={node.id}>
              <rect
                x={at.x}
                y={at.y}
                width={NODE_W}
                height={NODE_H}
                rx={6}
                fill={fill}
                stroke={stroke}
              />
              <text x={at.x + 8} y={at.y + 18} fill={theme.text.primary} fontSize={11} fontWeight={590}>
                {node.title}
              </text>
              <text x={at.x + 8} y={at.y + 34} fill={theme.text.secondary} fontSize={10}>
                {node.detail}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function BlockDetails() {
  const [filter, setFilter] = useState<Status | "all">("all");
  const done = BLOCKS.reduce((sum, block) => sum + block.rows.filter((row) => row.status === "done").length, 0);
  const partial = BLOCKS.reduce((sum, block) => sum + block.rows.filter((row) => row.status === "partial").length, 0);
  const missing = BLOCKS.reduce((sum, block) => sum + block.rows.filter((row) => row.status === "missing").length, 0);
  const labels: Array<{ id: Status | "all"; label: string }> = [
    { id: "all", label: `All ${done + partial + missing} rows` },
    { id: "done", label: `Done ${done}` },
    { id: "partial", label: `Partial ${partial}` },
    { id: "missing", label: `Missing ${missing}` },
  ];
  return (
    <Stack gap={8}>
      <Row gap={8} wrap>
        {labels.map((item) => (
          <Pill key={item.id} active={filter === item.id} onClick={() => setFilter(item.id)}>
            {item.label}
          </Pill>
        ))}
      </Row>
      {BLOCKS.map((block) => {
        const rows = block.rows.filter((row) => filter === "all" || row.status === filter);
        if (rows.length === 0) return null;
        return (
          <CollapsibleSection
            key={block.id}
            title={block.title}
            count={rows.length}
            defaultOpen={block.open}
            trailing={<Text size="small">{block.rollup}</Text>}
          >
            {block.rows.some((row) => row.addressed) ? (
              <Text size="small" tone="secondary">
                This pass marks the paper requirement fixed or still open. The paper, plan, changelog, ran, and miss sentences in the other columns are unchanged. Sections 3.5 and later have no This pass column.
              </Text>
            ) : null}
            <Table
              headers={
                block.rows.some((row) => row.addressed)
                  ? ["Sub-block", "Paper requires", "Plan and design", "Changelog", "What ran", "Specific miss", "This pass"]
                  : ["Sub-block", "Paper requires", "Plan and design", "Changelog", "What ran", "Specific miss"]
              }
              columnAlign={
                block.rows.some((row) => row.addressed)
                  ? ["left", "left", "left", "left", "left", "left", "left"]
                  : ["left", "left", "left", "left", "left", "left"]
              }
              rows={rows.map((row) => {
                const cells = [row.sub, row.paper, row.plan, row.changelog, row.ran, row.miss];
                if (block.rows.some((item) => item.addressed)) cells.push(row.addressed ?? "");
                return cells;
              })}
              rowTone={rows.map((row) => TONE[row.status])}
              striped
              stickyHeader
            />
          </CollapsibleSection>
        );
      })}
    </Stack>
  );
}

const MISSED_PASS = "Added 2 Nov 2026 · missed on the Fruits run";

function MissedMark() {
  return (
    <Pill size="sm" active>
      {MISSED_PASS}
    </Pill>
  );
}

function MissedFruitsPass() {
  return (
    <Stack gap={12}>
      <H2>Missed on the Fruits run — added 2 Nov 2026</H2>
      <Row gap={10} align="center" wrap>
        <MissedMark />
        <Text size="small" tone="secondary">
          Legend. This pill is only for the side-chat missed-tool pass on 2 Nov 2026. It sits on every row below. A one-line reconfirm means the gap is already written above. A longer note is new, or it records a later catalog line. Nothing in this section deletes an earlier sentence.
        </Text>
      </Row>
      <Grid columns={4} gap={16}>
        <Stat value="17 / 11 / 14" label="catalog.py on this read" tone="warning" />
        <Stat value="partial" label="Articulate Anything now" tone="warning" />
        <Stat value="missing" label="CubePart, no part mesh" tone="danger" />
        <Stat value="0 / 6" label="Figure 5 harnesses called" tone="danger" />
      </Grid>
      <Callout tone="info" title="Counts match catalog.py">
        catalog.py, the GAP_MATRIX.md header, and this canvas count 17 done, 11 partial, and 14 missing. Articulate Anything is partial: a PartNet URDF was accepted and the motion review failed.
      </Callout>
      <Text size="small" tone="secondary">
        Live page checked in the browser on 2 Nov 2026. Paper text is arXiv:2610.34745v1. Run status is sceneagent/catalog.py. A local file, runs/section34_pass/vlm_pass.json, is cited only where it is newer than that catalog row.
      </Text>

      <H3>1. Same kind of miss as GroundingDINO</H3>
      <Text size="small" tone="secondary">
        The catalog names the tool. The Fruits capture did not execute that tool. GroundingDINO is the pattern: website step 02, weights never loaded, catalog missing.
      </Text>
      <Table
        headers={["Item", "This pass", "Marker"]}
        rows={[
          [
            "Mesh step, beside GroundingDINO",
            "Live Figure 1 steps 01–06 do not name a mesh: 01 3DGS, 02 GroundingDINO + SAM assign semantic features to each Gaussian, 03 predictive per-Gaussian physics, 04 object decomposition, 05 articulation, 06 object sisters, then 07 USDZ and 08 Three.js. The mesh is elsewhere. The marker walkthrough is 01 Masks, 02 Gaussians, 03 Mesh, 04 Parts, 05 Under the cap, 06 Cap friction, 07 Writing. Its caption is cap decomposition, tip infill, and the friction that holds the cap on the tip. Token chart S5 is “Mesh generation / SAM-3D meshes”. The 4D chart S4 is “Surface meshing / persistent dynamics”. Skill 09 is PartSAM, then CubePart for hidden geometry. Paper figure 8 uses the same marker steps. The paper and the live page do not name SAM 2. The Fruits masks are SAM vit_h. catalog.py does not store a weight filename. sceneagent/sam2_user.py is a separate user-requested SAM 2 path, recorded not_run_gpu_busy, and it does not close the vit_h row. The recorded GroundingDINO pass is not_run_gpu_busy with weights_loaded false; the function returns not_run when weights are missing and the GPU-busy flag is off. Catalog status missing. The only object mesh the catalog records is /workspace/fruits-features/sam3d/0001.glb. Putting GroundingDINO+SAM in place of that file would match website step 02 and would leave the mesh step undone.",
            <MissedMark key="mesh" />,
          ],
          [
            "Per-Gaussian physics",
            "Reconfirmed. Site step 03 and the physics paragraph ask for material, mass, friction, and a 0–1 rigidity on every Gaussian. Paper section 3.4 trains an unpublished per-Gaussian MLP on DINOv2, OpenCLIP, and 16 geometric features, with nine numeric heads, on VLM labels from GSO, YCB, and ABO. Equation (2) is the material vote, and unknown is scaled by 0.4. The Quebec ledger stored one image class, wood, from gemini-3.8-flash. New: the live diagram copies one DINOv2+OpenCLIP vector onto that object’s Gaussians, gives the rest the scene mean, and adds position, size, opacity, and color. That list is the site’s. The paper’s list is 16 geometric features. A later equation (2) record keeps the physics prior absent. catalog.py keeps the historical class wood and a later one-image reply on 0001_crop.jpg: plastic, friction 0.4, mass 0.05 kg, density 950 kg/m3, from gemini-3.8-flash after gemini-2.5-flash returned HTTP 404. None of those are per-Gaussian values. The unpublished training recipe stays unpublished.",
            <MissedMark key="phys" />,
          ],
          [
            "LangSplat-style codebook",
            "Reconfirmed. OpenCLIP ViT-B-16 512-d and DINOv2 ViT-S/14 384-d exist. Association kept 27548/84993 Gaussians under equation (1), epsilon_a 0.08, epsilon_r 0.03, cell side round(s/192) and at most 16. The codebook was not built. Paper section 3.3 says the CLIP and DINO embeddings do not decide Gaussian membership. New: the live intro says those features are saved in a codebook inspired by LangSplat, while Figure 1 step 02 says GroundingDINO+SAM assigns them. The paper uses the mask-and-depth test in equation (1).",
            <MissedMark key="codebook" />,
          ],
          [
            "LaMa",
            "Reconfirmed. Paper section 3.3 makes LaMa optional after plane infill. The local interface writes no image when no plane render exists. The recorded pass is weights_absent, emitted false. The plane sample on gaussians.json is a hole-fraction placeholder, spread about 4.08 reconstruction units, and the note calls it not a fitted plane. No inpainted background. The live skill 06 fills the background from detected planes and does not name LaMa.",
            <MissedMark key="lama" />,
          ],
          [
            "CubePart",
            "The crash sentence above stays. Catalog, read this pass, still marks CubePart missing and records the weights: /workspace/cube/cubepart/weights/multi_part_dit.safetensors and vae.safetensors. Attempt 2 remains the diffusers 0.39 QwenEmbedRope mismatch, forward(video_fhw, device, max_txt_seq_len) against the repo call pos_embed(img_shapes, txt_seq_lens, device). No part mesh. Newer than the block above: a part-VLM reply from gemini-3.8-flash set completion_needed false, described the geometry as a fully captured table, and ledger cubepart attempt 3 was rejected without another inference. PartSAM’s /workspace/PartSAM/evaluation/results/0001.ply still has 15 parts after NMS. The catalog says the ply has 19 face colors and the VLM saw the 15 largest. The recorded part list is a face color and a face count, with no part-name field.",
            <MissedMark key="cube" />,
          ],
          [
            "Articulate Anything",
            "The “no URDF” sentence above stays. catalog.py and the matrix body now say partial. The PartNet zip was extracted to /workspace/articulate-dataset/extracted/partnet-mobility-v0/dataset. gemini-1.5-flash-latest returned HTTP 404. gemini-3.8-flash labeled Fruits.mp4 as a table and retrieved PartNet Table 25959. Link placement wrote /workspace/fruits-agent/articulate/link_placement/iter_0/seed_0/mobility.urdf, 13408 bytes, five fixed joints, base to body and four wheels. Ledger articulate attempt 2 accepted that file. The joint actor failed because conda env articulate-anything is missing and the CoTracker checkpoint is missing. Motion was not reviewed in Isaac. Those joints are not the 15 PartSAM parts.",
            <MissedMark key="art" />,
          ],
          [
            "USDZ package",
            "Reconfirmed, with the format split stated in one place. Live step 07 and the Figure 1 caption say USDZ for Isaac Lab, MuJoCo, and Unreal. Paper section 3.7 says OpenUSD and does not say USDZ. This run wrote /workspace/phase1_aligned_mesh.usda and a MuJoCo convex hull. The Three.js viewer row stays partial. catalog.py has a USD physics row and no USDZ row. DESIGN.md names “USD / USDZ” once in a comparison cell.",
            <MissedMark key="usdz" />,
          ],
          [
            "Coding harnesses and the signed review",
            "Reconfirmed. Paper section 3.1 and Figure 5 name six harnesses: Codex/GPT, Claude Code, Antigravity/Gemini, Muse, GLM via Stirrup, and Grok. This loop is local Python plus Gemini generateContent. None of the six was called. The material review was not signed by one of them. Paper model id gemini-2.5-flash returns HTTP 404 for new users. The live call is gemini-3.8-flash and has to be labeled as that. The ledger skill for a new line is the figure-7 id for that step. New label split: the live no-skills strip says ZCode/GLM 5.3, Grok Build/Grok 4.7, Muse/Muse Spark 1.3, and Antigravity/Gemini 3.8 Flash. Figure 5’s bin table says Stirrup/GLM-5.3, Grok CLI/Grok 4.7, Muse 1.0 Wonder, and Antigravity/Gemini 4 Argon. Paper section 4.1’s kitchen paragraph uses the names ZCode and Grok Build. Those kitchen scores stay the paper’s.",
            <MissedMark key="harness" />,
          ],
        ]}
        rowTone={["danger", "warning", "warning", "warning", "danger", "danger", "warning", "danger"]}
        striped
        stickyHeader
      />

      <H3>2. Named and not started</H3>
      <Text size="small" tone="secondary">
        Fruits is one capture. It is not the published GraspClutter6D, 4DCodeBench, or pi0.5 protocol.
      </Text>
      <Table
        headers={["Item", "This pass", "Marker"]}
        rows={[
          [
            "Meshy generative sisters",
            "Reconfirmed. Box and bowl sisters are a size check. Hunyuan stays off the Meshy label, DINOv3 stays off the DINOv2 label, and Hunyuan stays off the SAM 3D label. New: the live page text does not say Meshy. The sisters carousel is the Rudbeckia flower, labeled 9 sisters, with no generator name. Paper section 3.8 names Gemini 2.5 Flash, Gemini 3 Pro Image, and Meshy-6.",
            <MissedMark key="meshy" />,
          ],
          [
            "Alignment accept or reject gate",
            "Reconfirmed. icp_residual.json still has median about 0.050, normalized median about 0.033, p90 about 0.216, coverage about 0.146, silhouette empty. TRIM_KEEP 0.7 is this tree’s overlap. catalog.py now records ours_iteration_1 as this tree’s gate, not the paper’s, and that gate rejects the residual on coverage and silhouette. The paper’s four thresholds stay unpublished. The four ours numbers are already written in the changes section, so this row does not repeat them.",
            <MissedMark key="align" />,
          ],
          [
            "GraspClutter6D hard/000091",
            "Reconfirmed. Section 4.1 F1 at 1 cm and Chamfer need that bin, 13 views, 20 YCB objects. Fruits cannot produce that score. The paper’s 60.75% and 6.9% stay the paper’s.",
            <MissedMark key="grasp" />,
          ],
          [
            "4DCodeBench B_09, Z_06, Z_12",
            "Reconfirmed. Paper section 4.2 reports dynamic IoU only, equation (6). The live page also scores appearance, optical flow, point tracks, depth, and 3D trajectories, inside a sandboxed harness, on 200 tasks. The 99.09% figure and the chart’s other dynamic IoU labels stay the site’s and the paper’s. S4 on that chart is surface meshing for persistent dynamics, and it was not run.",
            <MissedMark key="bench" />,
          ],
          [
            "Site demos absent from the paper body",
            "Reconfirmed against the live page. Position-based fluids on the milk, with MPM. Cohesive volumetric network on the salmon cut. Isaac Lab RTX path tracing. Agent swarm. RoboCasa-style Isaac kitchen. GR00T 1.6 and a 10,000-episode finetune. Paper section 4.4 is pi0.5 only. Jetson Orin or Thor is a future on-device idea, and the posted figures do not require it. The live skill 11 label names Warp only, while the milk, jelly, and salmon section also names Newton, MPM, position-based fluids, and the cohesive network.",
            <MissedMark key="siteonly" />,
          ],
        ]}
        rowTone={["warning", "warning", "danger", "danger", "warning"]}
        striped
        stickyHeader
      />

      <H3>3. Nuances the plan still understates</H3>
      <Text size="small" tone="secondary">
        Checked against sections 3.1–3.10 and 4.1–4.4, the live page, and docs/EXPANSION_PLAN.md. Rows already written above are one line.
      </Text>
      <Table
        headers={["Item", "This pass", "Marker"]}
        rows={[
          [
            "Input methods and the lab robot",
            "Reconfirmed. The live page still offers RGB video, still images, an existing 3DGS .ply, photogrammetry/Polycam, LiDAR/Matterport, World Labs generated 3DGS, a Sketchfab mesh, skill SCENEAGENT-VIDEO-TO-4D, and the Hugging Face credit CollectDataIO/tidy-up-kitchens. The five scene buttons are still the lab clamp, the CollectData kitchen, the Sketchfab flower, the World Labs kitchen, and the Matterport telescope. New disagreement: paper figure 20 labels the Computational Robotics Lab rollout “TeleAvatar + two Sharpa Wave hands”. The live lab viewer aria-label is “HCRG lab robot + two Sharpa Wave hands”. The live DOM does not contain SO101. The older sentence that names an SO101 policy-canvas label stays above. three.js r184 is on the pipeline canvas. Paper robots remain Franka with UMI, Sharpa, Dexmate, and TeleAvatar.",
            <MissedMark key="input" />,
          ],
          [
            "World-model sentence",
            "The live sentence says alternative methods may use generative diffusion- or autogressive-based world models, spelled autogressive on the page, and that this pipeline can be lighter and more precise. The page does not use the words “do not replace”. The paper does not name that alternative. The paraphrase in the input section stays.",
            <MissedMark key="world" />,
          ],
          [
            "COLMAP and splatfacto caps",
            "Reconfirmed. COLMAP on Fruits was the 1 fps, 16-frame cap: 15/15 images, 603 points. That row stays partial. splatfacto-big stopped at 2000 steps. The catalog’s cited default is 30000.",
            <MissedMark key="colmap" />,
          ],
          [
            "Camera consensus and reviewer views",
            "Reconfirmed. Paper section 3.3’s optional camera consensus uses kappa = 1, so the kept set is the union. That stricter filter was not exercised. The reviewer manifest still names eleven files: one registration overlay, six axis views, two oblique, two assembled. They stay missing while the png files are absent.",
            <MissedMark key="views" />,
          ],
          [
            "Place check, Wilson, Isaac open",
            "Reconfirmed. The MuJoCo hull rollout left the box at z 0.1817534 m, outside the bowl. Wilson on 0 of 1 is not a place success. Isaac opened the aligned USDA with four prims. That open is not a coding-agent repair report. The live policy cells carry a 95% Wilson interval on the aria-label of each ten-trial cell. Those intervals belong to the site’s grid.",
            <MissedMark key="place" />,
          ],
          [
            "SceneAgent code",
            "Reconfirmed. The live page says CODE, coming soon, and that the code, processed scenes, and example objects will be released with the paper. Catalog row “SceneAgent code” is missing. This tree does not stand in for that release, and another application checkout stays out.",
            <MissedMark key="code" />,
          ],
          [
            "Figure 9 hand-set stiffness",
            "New. Paper figure 9 labels the corduroy cap’s regional stiffness as hand-authored, 350–6,000 N/m, “not measured”, and “Hand-set physics; no learned policy.” DESIGN.md says the fracture examples are hand-authored. It does not say this illustrated rigidity is hand-set rather than the per-Gaussian MLP. Those paper numbers stay the paper’s.",
            <MissedMark key="fig9" />,
          ],
          [
            "CPU CI has not seen this tree",
            "New, as a process gap only. SDLC says .github/workflows/sceneagent-ci.yml runs compileall and pytest on ubuntu-latest with numpy, pytest, trimesh, and pillow, which is what main installs. SF-2 is changing that workflow on another branch. That job cannot prove COLMAP, splatfacto, Isaac, or CubePart. On this read, main is 8 commits ahead of origin/main, and the 3.1–3.4 files, including agent_loop.py, website_step02.py, and section34.py, are uncommitted. GitHub CI has not seen this working tree.",
            <MissedMark key="ci" />,
          ],
        ]}
        rowTone={["warning", "warning", "warning", "warning", "warning", "danger", "warning", "warning"]}
        striped
        stickyHeader
      />
    </Stack>
  );
}

export default function SceneAgentPaperGaps() {
  const catalogDone = BLOCKS.reduce((sum, block) => sum + block.rows.filter((row) => row.status === "done").length, 0);
  const catalogPartial = BLOCKS.reduce((sum, block) => sum + block.rows.filter((row) => row.status === "partial").length, 0);
  const catalogMissing = BLOCKS.reduce((sum, block) => sum + block.rows.filter((row) => row.status === "missing").length, 0);
  const chartCategories = BLOCKS.map((block) => block.id);
  return (
    <Stack gap={20}>
      <Stack gap={8}>
        <H1>SceneAgent paper versus the SimFoundry upgrade</H1>
        <Text>
          The expansion plan covers sections 3.1–3.10 and section 4, but the paper’s agent is a coding harness on a model API, and this tree’s agent is two Gemini generateContent calls plus local Python. Fracture, 4D, hybrid scenes, Unreal, Meshy, and the policy loop are only named. catalog.py and GAP_MATRIX.md agree: 17 done, 11 partial, 14 missing. That “done” on the agent loop means the ledger files exist.
        </Text>
        <Text size="small" tone="secondary">
          Catalog status is the catalog.py field. A done row is an artifact this tree produced, not a paper metric. The This pass column on sections 3.1–3.4 is separate: fixed means this code or a real run does that paper requirement. Quebec evidence is instance 55073623, RTX 4090. Paper: Hollis, Fan, and Yang, arXiv:2610.34745v1. Site text checked against the saved extract and the live page: section 3.3 does not name GroundingDINO or SAM 2; website step 02 says GroundingDINO + SAM. Quebec SSH on 2 Nov 2026 connected. The RTX 4090 was not idle: articulate.py held 834 MiB, so GroundingDINO, SAM 2, full-rate COLMAP, and a 30000-step splat were not started.
        </Text>
      </Stack>

      <Grid columns={4} gap={16}>
        <Stat value={String(catalogDone)} label="Catalog rows done" tone="success" />
        <Stat value={String(catalogPartial)} label="Catalog rows partial" tone="warning" />
        <Stat value={String(catalogMissing)} label="Catalog rows missing" tone="danger" />
        <Stat value="42" label="Catalog rows total" />
      </Grid>

      <Callout tone="warning" title="The gap matrix does not lag the catalog">
        GAP_MATRIX.md matches catalog.py on the ICP row (partial), the aligned USDA (done), and the mesh MJCF inside the MuJoCo row (done, 327 vertices, 650 faces). DESIGN.md’s opening counts match catalog.py: 17 done, 11 partial, 14 missing, 42 rows. Its block-map table is still an earlier snapshot and is not regenerated. EXPANSION_PLAN records that the alignment gate ran and that the Isaac inspector opened the Fruits USDA, and its section 3.3 table records the SAM masks, OpenCLIP and DINOv2 features, and the SAM 3D mesh. skills.py lists SCENEAGENT-INSPECT-ISAAC as stage_opened. No coding model wrote a fix report.
      </Callout>

      <MissedFruitsPass />

      <H2>Changes made for 3.1–3.4</H2>
      <Text>
        Five of the 22 rows in sections 3.1–3.4 are marked fixed. The other seventeen stay open. pytest tests/sceneagent passed 72 tests. The table under the earlier-pass heading is the previous file list. This pass adds the alignment gate, the properties call, and the part answer. Quebec SSH connected and the GPU was busy with Articulate Anything, so full-rate COLMAP, the 30000-step splat, CubePart, and GroundingDINO or SAM 2 were not started. Estonia and Texas were not touched. Figure 5 harnesses were not called.
      </Text>
      <Grid columns={4} gap={16}>
        <Stat value="72" label="pytest tests/sceneagent passed" tone="success" />
        <Stat value="5" label="3.1–3.4 rows marked fixed" />
        <Stat value="17" label="3.1–3.4 rows still open" tone="warning" />
        <Stat value="busy" label="Quebec GPU, articulate.py" tone="warning" />
      </Grid>
      <Callout tone="warning" title="Full-rate fruit run, 2 Nov 2026">
        The table URDF stays on disk and is rejected for the fruit. The retry is also PartNet table 25959, not the 15 PartSAM parts, and motion review failed when SAPIEN exited 139. COLMAP on 452 frames registered 452/452, 37004 points, mean reprojection 0.800435 px. splatfacto-big finished the 30000-iteration cap; the file is step-000029999.ckpt. The new ICP gate fails: coverage 0.09366666666666666, silhouette 0.17505342248569655. The 30-step drop ended at z 0.18175340000000004 m and missed the bowl. The GPU was idle afterward. CubePart was not rerun. Nothing in this note is marked fixed.
      </Callout>
      <Callout tone="warning" title="Changes made on this pass">
        Alignment uses our iteration-1 gate, not a paper number: normalized median at most 0.05, coverage at least 0.5, p90 at most 0.25, and a silhouette score present and at least 0.5. The Fruits residual is rejected on coverage 0.146 and silhouette None. GroundingDINO is website step 02 and SAM 2 is labeled user-requested SAM 2, not the paper's SAM vit_h. Both are code paths with status not_run_gpu_busy. They do not close section 3.3. LaMa weights are absent, so a synthetic hole produced no inpaint file. A fallback gemini-3.8-flash call, after gemini-2.5-flash HTTP 404, returned class plastic, friction 0.4, mass 0.05 kg, and density 950 kg/m3. The same fallback, shown 1168 masks and 15 part colors, answered no hidden geometry. CubePart was not started.
      </Callout>
      <Callout tone="warning" title="Paper and site disagree on GroundingDINO">
        Website step 02 says GroundingDINO plus SAM assign semantic features to each Gaussian. Section 3.3 uses SAM masks, OpenCLIP, DINOv2, and a LangSplat-style association, and it says those embeddings do not decide Gaussian membership. The paper does not name GroundingDINO or SAM 2. The GroundingDINO hook is labeled website step 02. SAM 2 is labeled user-requested SAM 2, not the paper's SAM vit_h. Both returned not_run_gpu_busy because articulate.py held the Quebec GPU. Neither closes section 3.3. Still images are not a disagreement: the site says sequences come from video or as images, and section 3.2 plus figure 7 already name RGB image sequences.
      </Callout>
      <Text size="small" tone="secondary">
        Source: sceneagent package after this pass, pytest on 2 Nov 2026, one generate_flash audit the same day. The audit model id is the id the API returned. No paper F1, Chamfer, or dynamic IoU is reported.
      </Text>
      <Table
        headers={["What changed", "File", "Check that passed"]}
        rows={[
          ["Ledger skill is the figure-7 id for that step. Isaac is INSPECT-ISAAC, alignment is ALIGN-SCENE, articulation is ARTICULATE-NATIVE-ASSETS. Physics and the step runner stay RUN-STEP, which is what figure 7 prints.", "sceneagent/agent_loop.py", "test_ledger_skill_follows_figure_7_and_harnesses_stay_uncalled"],
          ["SCENEAGENT-INSPECT-ISAAC status is stage_opened. fix_report_from_model is false.", "sceneagent/skills.py", "test_inspect_isaac_skill_matches_the_opened_stage"],
          ["plan_with_client asks for a plan from the input state. Unknown ids are dropped. An empty reply uses default_plan. No figure-5 harness is called.", "sceneagent/fruits_agent.py", "test_plan_asks_from_input_state_and_falls_back_when_empty"],
          ["record_and_repair stores a rejected step and reruns only a registered repair tool. A missing tool leaves pose unset.", "sceneagent/agent_loop.py", "test_repair_hook_reruns_only_a_registered_tool_and_invents_no_pose"],
          ["Reviewer manifest writes 11 names: one registration overlay, six axis views, two oblique views, and two assembled views. Status is missing when the png files are absent. No images were drawn.", "sceneagent/reviewer_views.py", "test_reviewer_manifest_names_the_paper_views_and_does_not_draw_them"],
          ["accept_alignment stays undecided when thresholds are absent, including a null threshold map.", "sceneagent/alignment.py", "test_alignment_stays_undecided_without_thresholds"],
          ["A directory of still images is recorded as a sequence with count and paths. colmap_ran is false.", "sceneagent/ingest_assets.py", "test_still_images_are_a_sequence_and_existing_assets_need_a_file"],
          ["Existing 3DGS, photogrammetry, LiDAR, and generated 3DGS ingest records use skill SCENEAGENT-INGEST-DATA and accept only an existing file. Marble, Lyra, and HY-World are not downloaded.", "sceneagent/ingest_assets.py", "same test: local ply, obj, and las accepted; a Marble URL rejected"],
          ["Per-image mean RGB is recorded. A frame whose L1 distance from the set median exceeds 30 is flagged. The label is ours_mean_rgb_l1_above_30, not the paper's unpublished correction.", "sceneagent/ingest_assets.py", "test_color_heuristic_is_labeled_ours"],
          ["GroundingDINO hook is website step 02, status not_run, weights_loaded false.", "sceneagent/website_step02.py", "test_groundingdino_is_website_step_02_and_not_run"],
          ["plane_infill writes no image when the plane render is missing, and it does not run LaMa when a render is present.", "sceneagent/infill.py", "test_plane_infill_refuses_to_emit_without_a_plane_render"],
          ["equation2_vote applies the 0.4 unknown scale. A missing physics prior is absent and claimed_mlp is false. Keyword weight is taxonomy.DEFAULT_KEYWORD_WEIGHT, labeled as ours because the paper does not publish the number.", "sceneagent/section34.py", "test_equation2_keeps_a_missing_prior_out_of_the_mlp_claim"],
          ["request_audit returns the second-pass prompt and does not call unless live is set. One live call answered model id gemini-3.8-flash, HTTP 200, used_paper_model false, text 'The provisional class stands.' Historical gemini-2.5-flash 404 and gemini-3.8-flash wood are unchanged.", "sceneagent/section34.py", "unit test keeps live false; the live call was outside pytest"],
          ["Part inspection writes a prompt that asks whether hidden geometry is missing. With no model reply the decision is pending and CubePart is not called.", "sceneagent/section34.py", "test_part_prompt_stays_pending_and_does_not_call_cubepart"],
          ["choose_qwen_embed_rope_call selects forward(video_fhw, device) for diffusers 0.39 and pos_embed(img_shapes, txt_seq_lens, device) for the repo call. CubePart stays open without a part mesh. Articulate Anything is not downloaded while the GPU is down.", "sceneagent/section34.py", "test_cubepart_rope_call_shape_and_articulate_stay_open"],
        ]}
        rowTone={["success", "success", "success", "success", "warning", "warning", "success", "success", "warning", "warning", "warning", "warning", "warning", "warning", "warning"]}
        striped
        stickyHeader
      />
      <Text size="small" tone="secondary">
        Earlier pass, kept above. This pass adds the rows in the next table. Source: sceneagent package and one properties call plus one part call on 2 Nov 2026. The model id is the id the API returned. gemini-3.8-flash is the fallback, not Gemini 2.5 Flash.
      </Text>
      <Table
        headers={["What changed", "File", "Check that passed"]}
        rows={[
          ["Ours iteration-1 gate. Comment in alignment.py: SceneAgent did not publish these cutoffs. These are our iteration-1 gates. Fruits residual rejected. A synthetic residual at the four bounds is accepted.", "sceneagent/alignment.py", "test_ours_alignment_gate_rejects_fruits_and_accepts_a_passing_residual"],
          ["GroundingDINO predict path is website step 02. SAM 2 predict path is user-requested SAM 2, not the paper's SAM vit_h. gpu_busy returns not_run_gpu_busy. Neither closes section 3.3.", "sceneagent/website_step02.py, sceneagent/sam2_user.py", "test_groundingdino_and_sam2_labels_do_not_close_section_3_3"],
          ["complete_background writes a plane-fill placeholder from association counts, then calls LaMa only if an image exists. No image in writes nothing. Weights absent writes no PNG.", "sceneagent/infill.py", "test_lama_refuses_without_forging_an_image and the older refusal test"],
          ["request_material_properties asks for class, friction, mass_kg, and density_kg_m3. Live result: fallback gemini-3.8-flash, HTTP 200, paper HTTP 404, plastic, friction 0.4, mass 0.05 kg, density 950. equation2_vote still has no physics prior.", "sceneagent/section34.py", "test_material_properties_parse_without_calling; live call outside pytest"],
          ["Part prompt carried 1168 masks and 15 face-color counts. Live fallback gemini-3.8-flash answered completion_needed false. call_cubepart was not executed.", "sceneagent/section34.py", "same unit test; live call outside pytest"],
        ]}
        rowTone={["success", "warning", "warning", "success", "success"]}
        striped
        stickyHeader
      />

      <H2>Agent API and VLM, read against section 3.1</H2>
      <Text>
        Harvard’s conversion agent is not a local script. Section 3.1 and figures 5–7 name coding harnesses that call a frontier model, look at renders, and repair the asset. This repo calls the Gemini HTTP API twice and then runs Python.
      </Text>
      <Grid columns={4} gap={16}>
        <Stat value="0 / 6" label="Figure 5 harnesses called" tone="danger" />
        <Stat value="2" label="generateContent calls in the loop" />
        <Stat value="404" label="gemini-2.5-flash, the paper id" tone="warning" />
        <Stat value="plastic" label="fallback class plus friction, mass, density" tone="success" />
      </Grid>
      <Callout tone="danger" title="Catalog “agent loop done” is the file contract">
        Before this pass, sceneagent/agent_loop.py wrote plan.json, one manifest per step, and ledger.json, and every ledger line hardcoded skill SCENEAGENT-RUN-STEP. fruits_agent.plan_with_client sent a text prompt that already contained the only legal JSON, and order_plan then forced material_vote and mujoco_load to the front. The only registered repair was mujoco_load_repair, which writes an MJCF locally. make_tools still returns failure for partsam and isaac_inspector; later accepts were appended by other scripts, not chosen by a model looking at a render. This pass maps each new ledger line to its figure-7 skill, asks the model to plan from the input state, and reruns a repair only when one is registered. It still does not call a figure-5 harness or render the reviewer views.
      </Callout>
      <Grid columns={2} gap={20}>
        <Stack gap={8}>
          <H3>Where a model API is required</H3>
          <Text size="small" tone="secondary">
            Left to right is the paper’s order: a harness, then an orchestrator, then visual review of each tool, then the VLM uses in sections 3.4, 3.8, and figure 7 skill 21. Green is a call that returned a class. Accent is a call that exists and does less than the sentence. Red was not called. Source: arXiv:2610.34745v1 section 3.1 and figure 7, compared with sceneagent/gemini_client.py and sceneagent/fruits_agent.py on 2 Nov 2026.
          </Text>
          <FlowDiagram nodes={AGENT_NODES} edges={AGENT_EDGES} direction="horizontal" />
        </Stack>
        <Stack gap={8}>
          <H3>What the client actually sends</H3>
          <Text size="small" tone="secondary">
            Endpoint generativelanguage.googleapis.com/v1beta, method models/id:generateContent, header x-goog-api-key. The key is redacted from stored bodies. Source: sceneagent/gemini_client.py.
          </Text>
          <Table
            headers={["Call", "Model", "Image", "Result"]}
            rows={[
              ["Plan", "gemini-2.5-flash, then a listed Flash id", "None. The prompt now describes the input state and does not paste a steps array.", "An empty reply uses default_plan. A parsed reply keeps the model order after unknown ids are dropped."],
              ["Material vote", "Same fallback", "Fruits frame 0001.jpg as inline base64", "HTTP 200, class wood. Ledger model id is gemini-3.8-flash, not 2.5. Historical."],
              ["Properties", "gemini-2.5-flash, then gemini-3.8-flash as the fallback", "Fruits 0001_crop.jpg", "HTTP 200. Class plastic, friction 0.4, mass 0.05 kg, density 950 kg/m3. used_paper_model false."],
              ["Part inspection", "Same fallback", "Text only: 1168 masks, 15 color and face counts", "HTTP 200. completion_needed false. CubePart was not started."],
              ["Audit, sisters, policy, image edit", "Not sent", "Not sent", "No second audit. Gemini 3 Pro Image and Meshy were not called."],
            ]}
            rowTone={["warning", "success", "success", "success", "danger"]}
            striped
          />
        </Stack>
      </Grid>
      <H3>Figure 5 harnesses</H3>
      <Text size="small" tone="secondary">
        These are the agents in the paper. This tree does not invoke them, and their GraspClutter6D scores are not results of this repo. SimFoundry’s own Gemini path in pipelines A and B is a different client.
      </Text>
      <Table
        headers={["Harness", "Model API named in figure 5", "Called here"]}
        rows={[
          ["Codex", "GPT-6 Astra", "No"],
          ["Claude Code", "Opus 5.5", "No"],
          ["Antigravity", "Gemini 4 Argon", "No"],
          ["Muse", "Muse 1.0 Wonder", "No"],
          ["GLM-5.3 via Stirrup", "GLM-5.3", "No"],
          ["Grok CLI", "Grok 4.7", "No"],
        ]}
        rowTone={["danger", "danger", "danger", "danger", "danger", "danger"]}
        striped
      />
      <H3>Every VLM sentence in the paper</H3>
      <Table
        headers={["Paper line", "What the VLM or agent must do", "Code that ran", "Missing"]}
        rows={[
          ["3.1, page 5", "Orchestrator plans the conversion from the input state.", "plan_with_client, text only. The prompt describes the capture and does not dictate the JSON.", "No capture image is attached. An empty reply falls back to the local step list. That fallback is not a harness."],
          ["3.1, page 6", "Runner does actor-critic visual review of each tool output.", "Local tool return values. record_and_repair reruns a registered repair tool for any step.", "No render is sent back to the model. A reject does not ask the API for a new pose, and the hook does not invent one."],
          ["3.1, page 6", "Reviewer checks registration overlays, six axis views, two oblique views, and two assembled views, then retries.", "write_view_manifest names one overlay, six axis views, two oblique views, and two assembled views.", "The image files are absent, so the manifest status is missing. No model could reject a render."],
          ["3.1, page 6", "Agents research a method, benchmark it, and open a draft PR if the user approves.", "Not in sceneagent.", "No benchmark harness and no skill-edit loop."],
          ["Figure 6 S2 and S4", "Token spend is the harness doing visual QA and material semantics.", "SAM, OpenCLIP, and DINOv2 ran locally. The historical class word was wood. This pass also stored plastic, friction 0.4, mass 0.05 kg, and density 950 from the fallback.", "Grounded visual QA is not a generateContent call. The properties call is one crop, not a per-Gaussian label."],
          ["3.4 equation (2)", "Weighted vote of the physics prior, the VLM, and keyword priors. Unknown votes are scaled by 0.4. A later VLM audit can replace the class. A coding agent signs the review.", "equation2_vote scales unknown by 0.4. A missing prior is absent and is not called the MLP. A live properties call on the Fruits crop used fallback gemini-3.8-flash after gemini-2.5-flash HTTP 404: class plastic, friction 0.4, mass 0.05 kg, density 950 kg/m3. The historical wood class is unchanged.", "The Fruits per-Gaussian prior is still absent. There is no second audit and no coding-agent signature. Rigidity was not in the request. The model id is the fallback, not Gemini 2.5 Flash."],
          ["3.4 training", "A VLM labels GSO, YCB, and ABO. Those labels train the MLP. Targets are not measurements.", "Not started.", "No label set and no network. The weights are unpublished."],
          ["3.4 parts", "A VLM inspects PartSAM parts and asks CubePart only for missing geometry.", "15 face-color counts, 1168 masks. Live fallback gemini-3.8-flash HTTP 200 answered completion_needed false. CubePart was not started.", "No part mesh exists. The Qwen RoPE crash is unchanged. The answer no means this pass did not ask CubePart to run."],
          ["3.8", "Gemini 2.5 Flash splits functional parts. Gemini 3 Pro Image edits one. Meshy lifts it.", "box.json and bowl.json, model-free.", "No image-edit call. Pipeline B’s Gemini is not this skill."],
          ["Figure 7 skill 21", "SCENEAGENT-ADAPT-VLA-POLICIES finetunes with VLM-controlled corrections.", "skills.py marks it not_implemented.", "No policy and no correction images."],
        ]}
        rowTone={["warning", "danger", "danger", "danger", "warning", "warning", "danger", "danger", "danger", "danger"]}
        striped
        stickyHeader
      />
      <H3>Figure 7 skills versus a model call</H3>
      <Text size="small" tone="secondary">
        skills.py marks nine ids as phase-1 file contracts. That status means a JSON writer exists. It does not mean a harness called an API. SCENEAGENT-INSPECT-ISAAC is no longer in LATER_SKILLS. Its status is stage_opened because Isaac opened the stage, and fix_report_from_model is false.
      </Text>
      <Table
        headers={["Skill", "Paper role", "Model API in this tree"]}
        rows={[
          ["01 SCENEAGENT-RUN-PIPELINE", "Orchestrator plans conversion and sisters.", "Text generateContent. The prompt no longer contains the step ids. Unknown ids are dropped."],
          ["02 SCENEAGENT-INGEST-DATA", "Ingest builds the 3DGS.", "Local COLMAP and splatfacto. No agent."],
          ["04 SCENEAGENT-RUN-STEP", "Reviews masks, fields, objects, and parts.", "Ledger lines for physics, parts, and the step runner use this id. Isaac, alignment, and articulation use their own figure-7 ids."],
          ["05 SCENEAGENT-ALIGN-SCENE", "Fits scale, position, and rotation.", "Trimmed ICP. Ours iteration-1 gate rejects the Fruits residual. No model proposes the pose."],
          ["08 Physics, same RUN-STEP", "Fuses the per-Gaussian prior with a VLM.", "Fallback gemini-3.8-flash returned plastic, friction 0.4, mass 0.05 kg, density 950. The per-Gaussian prior is still absent."],
          ["09 Parts", "PartSAM, then CubePart if the VLM sees a hole.", "PartSAM local. Fallback gemini-3.8-flash said no hidden geometry. CubePart was not started."],
          ["14 SCENEAGENT-INSPECT-ISAAC", "Reviews the export and reports fixes.", "Isaac opened the USDA. skills.py status is stage_opened. No fix report from a model."],
          ["16–18 Sisters", "A model proposes shape, layout, and task variants.", "Model-free box and bowl JSON. Templates for scene and task."],
          ["03, 06, 07, 10–13, 15, 19–21", "4D, infill, orientation, joints, fracture, packager, researcher, evaluator, trainer.", "Named in skills.py as not_implemented, or a file writer with no API."],
        ]}
        rowTone={["warning", "warning", "warning", "warning", "warning", "danger", "warning", "warning", "danger"]}
        striped
        stickyHeader
      />

      <H2>On the Harvard site, absent from the arXiv paper, and absent from the plan</H2>
      <Text>
        These are the tools and systems the project site uses to produce the posted scenes, and that a search of arXiv:2610.34745v1 plus EXPANSION_PLAN.md, DESIGN.md, and GAP_MATRIX.md does not name. The live page is a client app; this list is from the site text captured with the preprint. GroundingDINO, a USDZ package, and a LangSplat-style codebook are on the site and not in the paper, and they are already in the plan, so they are not in this table.
      </Text>
      <Grid columns={3} gap={16}>
        <Stat value="7" label="Site-only systems missing from the plan" tone="danger" />
        <Stat value="1" label="Dynamic IoU, the only 4D score in the paper" />
        <Stat value="6" label="Scorers the site adds for 4DCodeBench" tone="warning" />
      </Grid>
      <Table
        headers={["Site item", "Where the site uses it", "Why it is not the paper", "Plan"]}
        rows={[
          [
            "Position-based fluids",
            "Section 2.4. Milk is simulated with MPM and position-based fluids.",
            "Section 3.6 names MPM and FLIP for liquids. It does not name position-based fluids.",
            "No milk scene and no PBF solver.",
          ],
          [
            "Cohesive volumetric network",
            "Section 2.4. The salmon cut uses this network on the knife region. Jelly and watermelon stay on Newton.",
            "Section 3.5 and 3.6 name Warp, Blast, and Newton. They do not name this network.",
            "Fracture is a named phase with no scene file.",
          ],
          [
            "Isaac Lab RTX path tracing",
            "Section 2.5. Recorded contact states are re-rendered with RTX path tracing so a cut keeps the scanned color.",
            "The paper says Isaac Lab renders the scenes. It does not name path tracing.",
            "The plan smokes a headless stage. It does not record a path-traced pass.",
          ],
          [
            "4DCodeBench sandbox and the extra scorers",
            "Section 1.8. A sandboxed harness scores appearance, dynamic IoU, optical flow, point tracks, depth, and 3D trajectories. The set is 200 tasks, 100 real and 100 synthetic.",
            "Section 4.2 reports only dynamic IoU, equation (6), on B_09, Z_06, and Z_12. The other five scorers and the sandbox are not in that section.",
            "The plan forbids copying 99.09% and does not name the sandbox or those five scorers.",
          ],
          [
            "Agent swarm",
            "Section 3. The site says the easiest conversion is an agent swarm that runs each step and reviews it.",
            "Section 3.1 is one orchestrator, ingest agents, one runner, and one reviewer. It does not say swarm.",
            "The loop is one Python process and two Gemini calls.",
          ],
          [
            "RoboCasa-style Isaac kitchen",
            "Hybrid section. The fruit-picking kitchen is built from Isaac Lab assets and described as similar to RoboCasa.",
            "Section 3.9 says artist environments and, on the site’s later paragraph, Omniverse. RoboCasa is not in the paper.",
            "The plan says to use the light editor after a mesh exists. It names no kitchen asset pack.",
          ],
          [
            "GR00T 1.6 and a 10,000-episode finetune",
            "Section 7.3. The site finetunes pi0.5 and NVIDIA GR00T 1.6 for 10,000 simulated episodes in the lab scene.",
            "Section 4.4 finetunes pi0.5 only. The bibliography cites GR00T N1. It does not train GR00T 1.6 or state 10,000 episodes.",
            "The plan’s policy tool is openpi / pi0.5. DESIGN.md says GR00T is absent. It does not add GR00T 1.6 as work.",
          ],
        ]}
        rowTone={["danger", "danger", "danger", "danger", "warning", "danger", "danger"]}
        striped
        stickyHeader
      />
      <Callout tone="warning" title="Site-only, and not a requirement for the posted figures">
        Section 3.4 of the site imagines a later on-device run on an NVIDIA Jetson Orin or Thor, with the capture and the finetune staying on the robot. That hardware is not in the paper and not in the plan. The posted GraspClutter6D, 4DCodeBench, and robot figures do not depend on it.
      </Callout>

      <H2>Input methods on the live site</H2>
      <Text>
        Checked in the browser on the SceneAgent page, 2 Nov 2026. The page is a client app. Figure 1 and the five scene buttons are the input contract. This repo has run one path: Fruits.mp4 through a 1 fps, 16-frame COLMAP sample.
      </Text>
      <Grid columns={3} gap={16}>
        <Stat value="9" label="Input paths on the live page" />
        <Stat value="1" label="Paths this repo has run" tone="warning" />
        <Stat value="r184" label="Three.js build on the policy canvas" />
      </Grid>
      <Table
        headers={["Input", "What the page does with it", "Paper", "Plan and our run"]}
        rows={[
          [
            "RGB video",
            "A short CollectData messy-kitchen video becomes an Isaac, MuJoCo, and Unreal scene. The page hosts that clip’s harness renders.",
            "Section 3.2: COLMAP, then splatfacto big.",
            "Fruits.mp4 only, capped at 1 fps and 16 frames. The CUDA cameras are not in the 2000-step splat.",
          ],
          [
            "Still images",
            "Figure 1 says RGB sequences come from video or as images.",
            "Section 3.2 says RGB image sequences. Figure 7 lists RGB images or video. The site's 'from video or as images' matches that sequence. They do not disagree.",
            "read_still_image_sequence records count and paths and sets colmap_ran false. No still folder was reconstructed.",
          ],
          [
            "Existing 3DGS .ply",
            "The same pipeline accepts a public .ply and skips capture.",
            "Section 3.2 accepts an existing 3DGS.",
            "ingest_asset accepts a local .ply when the file exists. A unit-test file was accepted. No public splat was downloaded.",
          ],
          [
            "Photogrammetry",
            "Polycam objects, including orbit passes so cloth folds are visible from every side.",
            "Section 3.2 and figure 10.",
            "A local .obj is accepted when the file exists. No Polycam capture is in the run.",
          ],
          [
            "LiDAR / Matterport",
            "Observatory button. The prompt target is the telescope.",
            "Figure 20.",
            "A local .las is accepted when the file exists. No Matterport scan is in the run.",
          ],
          [
            "Generated 3DGS",
            "World Labs kitchen button. The object in that scene is a Hawaiian pizza.",
            "Section 3.2 names Marble, Lyra, and HY-World.",
            "A remote Marble, Lyra, or HY-World URL is rejected and not downloaded. No generated splat is in the run.",
          ],
          [
            "Sketchfab mesh",
            "Greenhouse button. The prompt is: convert the flower 3D capture into a usable Isaac Lab asset. Credits also name a statue, a motherboard, and a GPU.",
            "The paper’s capture credits include those Sketchfab models.",
            "The plan never names Sketchfab or that prompt.",
          ],
          [
            "4D video",
            "Skill SCENEAGENT-VIDEO-TO-4D. The page hosts source and target mp4 pairs for B_09, Z_06, and Z_12. The 4D camera stage is labeled frames and camera fit.",
            "Section 3.6 estimates the camera from vanishing points and silhouettes, then a build script that does not read the video.",
            "Not started. The plan has no reader for those source videos.",
          ],
          [
            "Hugging Face kitchen set",
            "Citation link: huggingface.co/datasets/CollectDataIO/tidy-up-kitchens.",
            "Bibliography entry for CollectDataIO.",
            "The plan says CollectData in a figure caption and does not name this dataset as an input.",
          ],
        ]}
        rowTone={["warning", "danger", "warning", "danger", "danger", "danger", "danger", "danger", "danger"]}
        striped
        stickyHeader
      />
      <Callout tone="warning" title="Two rebuild facts that are only on the page">
        The policy canvas is Three.js r184, and its aria label is an SO101 arm in the captured workspace. The paper’s robots are Franka with a UMI gripper, Sharpa hands, Dexmate, and TeleAvatar. SO101 is not in the paper or the plan. The no-skills Muse clip is labeled Muse Spark 1.3. Figure 5 of the paper says Muse 1.0 Wonder. The page also says not to replace this pipeline with a diffusion or autoregressive world model that generates multi-view video. That sentence is the site’s warning. The paper does not name an autoregressive world model.
      </Callout>
      <Text size="small" tone="secondary">
        The five live prompts are the acceptance tests the page shows: articulate the lab clamp’s jaw and screw, convert the CollectData kitchen, convert the Sketchfab flower for Isaac Lab, convert the World Labs kitchen, and convert the Matterport telescope. A VLM pass on an unlabeled capture is supposed to emit material, mass, friction, and a 0–1 rigidity for every Gaussian. This pass’s fallback call returned one class plus friction, mass, and density for the crop, not a value on every Gaussian, and it did not return rigidity.
      </Text>

      <H2>Catalog rows by paper block</H2>
      <Text size="small" tone="secondary">
        Vertical axis: paper block. Horizontal axis: number of catalog.py rows. Stacked series are done, partial, and missing. Source: sceneagent/catalog.py status field, same 17 / 11 / 14 split as GAP_MATRIX.md. Quebec run 1 Nov 2026. The Sec 4 bar is the two section 4.1 rows plus the official-release row. Section 4.2 is the 3.6 bar, not a second count.
      </Text>
      <BarChart
        horizontal
        stacked
        height={420}
        categories={chartCategories}
        series={[
          { name: "Done", tone: "success", data: BLOCKS.map((block) => block.rows.filter((row) => row.status === "done").length) },
          { name: "Partial", tone: "warning", data: BLOCKS.map((block) => block.rows.filter((row) => row.status === "partial").length) },
          { name: "Missing", tone: "danger", data: BLOCKS.map((block) => block.rows.filter((row) => row.status === "missing").length) },
        ]}
      />

      <H2>What the Quebec run produced</H2>
      <Text size="small" tone="secondary">
        Instance 55073623, Quebec, RTX 4090, driver 580.82.09. Paths are on that machine under /workspace. Source: catalog.py next fields and CHANGELOG.md, read from the later bullets when an early bullet disagrees.
      </Text>
      <Table
        headers={["Artifact", "Recorded result", "Catalog status"]}
        rows={[
          ["Agent plan / manifest / ledger", "/workspace/fruits-agent. Repair accepted MJCF, SAM 3D, PartSAM, inspector, ICP file, aligned USDA, and one failed hull rollout.", "3.1 loop done"],
          ["CUDA COLMAP 4.2.1", "15/15 images, 603 points, mean reprojection 0.867273 px. Runner still 1 fps and a 16-frame cap.", "3.2 partial"],
          ["splatfacto-big", "2000 steps, not 30000. step-000001999.ckpt. 13 train renders. Cameras from this COLMAP are not in the train.", "3.2 done"],
          ["SAM, OpenCLIP, DINOv2", "vit_h every 10th frame: 0001 has 33 masks, 0011 has 19. OpenCLIP 512-d. DINOv2 ViT-S/14 384-d. Not DINOv3.", "3.3 done"],
          ["Association", "Equation (1) kept 27548/84993 Gaussians.", "3.3 done"],
          ["SAM 3D + ICP", "0001.glb, overlay, gaussian ply. Trimmed ICP median 0.04999262711481225. Ours iteration-1 gate rejects: coverage 0.146 and silhouette None.", "mesh done, ICP partial"],
          ["Aligned USDA", "phase1_aligned_mesh.usda, 5,339,577 bytes, 8095 faces. Isaac opened=True, 4 prims.", "3.7 done"],
          ["MuJoCo 3.17", "30 steps. Hull MJCF 327 verts, 650 faces. Box ended (0, 0, 0.1817534) m. Not a place.", "3.7 done"],
          ["PartSAM", "1168 masks, 15 after NMS, ply written.", "3.4 done"],
          ["CubePart", "2 Nov: inference exited 1 at step 0/50. QwenEmbedRope.forward() got multiple values for argument 'device'. No mesh. catalog.py still says the weights were not downloaded.", "catalog missing; log is a failed run"],
          ["Articulate Anything", "PartNet table 25959 URDF accepted (13408 bytes, five fixed joints). Motion review failed: sapien_simulate.py exited 139, feedback_score -1. Ledger attempts 3 and 4 rejected the table URDFs.", "partial"],
          ["Material vote", "gemini-2.5-flash HTTP 404. gemini-3.8-flash class wood. Unpublished MLP not started.", "vote done, VLM partial, MLP missing"],
          ["Sisters and place grid", "box.json and bowl.json only. +5% box does not fit. Wilson 0/1 is 0–0.7934506856227626.", "sisters done, grid partial"],
        ]}
        rowTone={["success", "warning", "success", "success", "success", "warning", "success", "warning", "success", "danger", "warning", "warning", "warning"]}
        striped
        stickyHeader
      />

      <H2>Three paper stages, not one list of six</H2>
      <Text>
        Section 3 of the paper is three stages: agent-guided reconstruction and repair (3.1–3.7), scene augmentation (3.8–3.9), and demonstration generation (3.10). Section 4 is results, drawn after 3.10 so it is not a fourth stage. Node color is the catalog rollup of that block. Green fill is done, accent stroke is partial, red fill is missing.
      </Text>
      <FlowDiagram nodes={PAPER_NODES} edges={PAPER_EDGES} direction="vertical" />
      <Text size="small" tone="secondary">
        Source: paper section 3 opening paragraph, pages 5–16 of the 26 Oct 2026 preprint, compared with catalog.py. A dashed edge, if the layout marks one, is a cycle and is not a paper dependency.
      </Text>

      <Grid columns={2} gap={20}>
        <Stack gap={8}>
          <H3>Website conversion steps 01–08</H3>
          <Text size="small" tone="secondary">
            A different list. Steps 01–06 are the site’s conversion sequence inside its first figure, then 07 USD and 08 the Three.js viewer. Step 02 names GroundingDINO, which section 3.3 does not. Step 06 is sisters, which the paper puts in stage 2.
          </Text>
          <FlowDiagram nodes={SITE_NODES} edges={SITE_EDGES} direction="horizontal" />
        </Stack>
        <Stack gap={8}>
          <H3>Policy loop, the other six</H3>
          <Text size="small" tone="secondary">
            DESIGN.md and figure 19, also the site’s “Agentic Policy Evaluation” steps. This is not the conversion list above. The dashed return is the site’s “restart with new failure modes.” Model-free box and bowl sisters are section 3.8, not step 02 of this loop.
          </Text>
          <FlowDiagram nodes={POLICY_NODES} edges={POLICY_EDGES} direction="horizontal" />
        </Stack>
      </Grid>

      <Grid columns={2} gap={20}>
        <Stack gap={8}>
          <H3>Figure 6 token stages S0–S7</H3>
          <Text size="small" tone="secondary">
            GraspClutter6D hard/000091 token accounting. S3 is SplatGraph clustering. These colors mean whether this tree ran that kind of stage. They are not the paper’s token counts, and they are not a fourth copy of sections 3.1–3.7.
          </Text>
          <FlowDiagram nodes={FIG6_NODES} edges={FIG6_EDGES} direction="horizontal" />
        </Stack>
        <Stack gap={8}>
          <H3>4DCodeBench stages S0–S6</H3>
          <Text size="small" tone="secondary">
            Figure 4, cases B_09, Z_06, and Z_12. S3 is continuum, MPM, and fluid. S6 is the clean-room rebuild. All missing here. The paper’s dynamic IoU is not a result of this repo.
          </Text>
          <FlowDiagram nodes={BENCH_NODES} edges={BENCH_EDGES} direction="horizontal" />
        </Stack>
      </Grid>

      <Divider />

      <H2>Every paper block against the plan, the changelog, and the run</H2>
      <Text>
        Each row is one catalog.py entry. The dot is that entry’s status. Open a block for the five-way comparison. Filter pills hide rows; they do not change the 17 / 11 / 14 counts.
      </Text>
      <BlockDetails />

      <H2>Plan and design misses</H2>
      <Stack gap={10}>
        <Callout tone="warning" title="1. Opening counts match the catalog; two snapshots remain">
          EXPANSION_PLAN section 3.3 records the splat, SAM masks, OpenCLIP, DINOv2, and the SAM 3D mesh. Section 3.1 records that the alignment gate ran and that the Isaac inspector opened the Fruits USDA. The dependency order still says splatfacto is waiting on disk. DESIGN.md’s opening counts match catalog.py (17 done, 11 partial, 14 missing, 42 rows). Its block-map table is still an earlier snapshot and is not regenerated. The early CHANGELOG bullets that leave association partial, MuJoCo missing, and SAM 3D missing are superseded by later bullets; catalog.py is the status to use.
        </Callout>
        <Callout tone="warning" title="2. Two done-rules disagree on the splat">
          The splatfacto row is done at 2000 steps because a checkpoint and one render exist. The COLMAP row is done only when a full-rate sequence feeds splatfacto. Catalog says those cameras are not in the train, the runner is still 1 fps with a 16-frame cap, and nerfstudio’s default 30000 was not used. Marking one done and the other partial hides that the scene in section 3.2 was not built.
        </Callout>
        <Callout tone="warning" title="3. Substitutions the docs forbid, and one the website adds">
          DINOv2 ViT-S/14 ran. DINOv3 was not used. SAM 3D ran. Hunyuan was not labeled SAM 3D. Meshy did not run, and pipeline B’s Hunyuan cousins must not be labeled Meshy. GroundingDINO is website step 02, not section 3.3, and the plan bundled it into the stale “all missing” SAM line, so it has no separate done condition.
        </Callout>
        <Callout tone="danger" title="4. Figure 6 S3, the 4D stages, fracture, and hybrid scenes are only named">
          SplatGraph clustering is not in EXPANSION_PLAN or DESIGN.md and not in the code. 4DCodeBench S0–S6, equation (3), and the build script that does not read the video are not artifacts. Section 3.5 names Warp, Blast, and Newton and specifies no scene. Section 3.9 names artist environments and specifies no scene. Website agent-graph step “07 Orientation” (Isaac sweeps that orient the background) is not in the plan. Section 4.3 qualitative conversion is not in the plan as work to do.
        </Callout>
        <Callout tone="danger" title="5. Section 3.10 is specified as a smaller check than the paper">
          The plan quotes approach, descend, grasp, lift, carry, lower, release, and retreat, then accepts a geometric inside-target test. The run is one 30-step MuJoCo drop that did not place. The figure 19 loop, composited observations, the ten-trial by nine-light grid, and pi0.5 LoRA are not a data layout that can be executed. Wilson on 0/1 is the formula, not that grid.
        </Callout>
      </Stack>

      <H3>Also recorded, and not a status change</H3>
      <Text>
        The material-vote row is done as equation (2)’s formula. The Fruits class wood is the partial VLM row, from gemini-3.8-flash after gemini-2.5-flash returned HTTP 404. The unpublished per-Gaussian MLP is missing. Three.js is partial only because the SimFoundry light editor exists; it was not opened on these USDA files. Articulate Anything is partial: a PartNet table 25959 URDF was accepted (13408 bytes, five fixed joints) and the motion review failed (`sapien_simulate.py` exited 139, `feedback_score` -1). catalog.py and GAP_MATRIX.md record that URDF. The convex-hull MJCF is recorded as written (327 vertices, 650 faces). Those files do not describe a trimesh crash.
      </Text>

      <H2>Next phases from the run</H2>
      <Text>
        These are the remaining phases that already have an input or a measured blocker. They are not DESIGN.md phases 1–9, several of which describe work that has since run. No unpublished threshold and no paper score is filled in here.
      </Text>
      <Table
        headers={["Phase", "Input already on disk", "Output", "Blocker"]}
        rows={[
          [
            "1. Alignment gate",
            "icp_residual.json: normalized median 0.033103985722272695, p90 0.21563629180764, coverage 0.14616666666666667, silhouette None.",
            "Rejected by ours iteration-1 gate: median <= 0.05, p90 <= 0.25, coverage >= 0.5, silhouette present and >= 0.5. Those numbers are ours, to be iterated, not the paper's.",
            "The paper's four thresholds are still unpublished. Coverage and a missing silhouette fail. The cutoffs were not lowered to accept Fruits.",
          ],
          [
            "2. Full-rate COLMAP and full splat",
            "Fruits.mp4. CUDA sparse model, 15/15, 603 points, 0.867273 px. Checkpoint step-000001999.ckpt and 13 train renders.",
            "Full-rate cameras inside splatfacto-big, trained past this 2000-step cap. The tool default cited by the catalog is 30000.",
            "The runner’s 1 fps and 16-frame cap. Catalog says the CUDA cameras are not in the current train.",
          ],
          [
            "3. CubePart, then Articulate",
            "0001.glb. PartSAM 0001.ply, 1168 masks, 15 after NMS. About 108 GB free after the failed inference.",
            "A part mesh from CubePart, then a URDF only if Articulate Anything emits one, then a motion review.",
            "CubePart crashed earlier in QwenEmbedRope.forward and this pass did not start it. The part VLM answered no hidden geometry. Articulate Anything wrote a PartNet table 25959 URDF; the ledger rejected it for this fruit and the motion review failed.",
          ],
          [
            "4. LaMa",
            "Association kept 27548/84993 Gaussians. A 16 by 16 plane-fill placeholder exists. A 4000-center sample has least-spread axis z and spread 4.08, so it is not a tight plane.",
            "An inpainted background patch after a real plane render exists.",
            "LaMa weights are not installed. The placeholder is not an inpaint, and no forged PNG was written. The row stays open.",
          ],
          [
            "5. Fracture examples",
            "No parchment, ceramic, jelly, or watermelon scene in this tree.",
            "A hand-authored Warp, Blast, or Newton example, kept out of the USD contract.",
            "Not started. The plan names the solvers and specifies no scene. Paper parameters stay the paper’s.",
          ],
          [
            "6. 4D",
            "No dynamic-video build.",
            "A build script that regenerates frames without reading the video. No 4DCodeBench score in that step.",
            "Not started. Equation (3), MPM, FLIP, and the S0–S6 checker are not specified as files.",
          ],
          [
            "7. Unreal and the Three.js viewer",
            "phase1_place.usda (3 prims) and phase1_aligned_mesh.usda (opened, 4 prims, 8095 faces).",
            "A person-confirmed Unreal load, and the existing light editor showing this mesh with figure 14’s edits.",
            "Unreal is not started. The figure 14 viewer is not in the sceneagent package.",
          ],
          [
            "8. Ten-trial grid",
            "place_grid.json is 0/1. sisters/box.json and bowl.json. The +5% box does not fit the bowl.",
            "Ten trials and the site’s lighting columns, with a Wilson interval on those counts.",
            "Lighting was not simulated. The one MuJoCo trial is not a place. The box ended at z 0.1817534 m.",
          ],
          [
            "9. Policy",
            "No expert-episode archive. No pi0 checkpoint.",
            "Scripted episodes that end inside the target, and a pi0 or pi0.5 run only if it is actually trained.",
            "Not started. Section 4.4’s eggplant counts stay the paper’s. The figure 19 loop has not run.",
          ],
        ]}
        striped
        stickyHeader
      />

      <Divider />
      <Text size="small" tone="tertiary">
        Sources: sceneagent/catalog.py, sceneagent/agent_loop.py, sceneagent/fruits_agent.py, sceneagent/gemini_client.py, sceneagent/skills.py, docs/GAP_MATRIX.md, docs/EXPANSION_PLAN.md, docs/DESIGN.md, and CHANGELOG.md. Paper: arXiv:2610.34745v1. Site text: computationalrobotics.seas.harvard.edu/SceneAgent, captured with the preprint and checked 2 Nov 2026. Counts above follow catalog.py. The CubePart log is newer than that catalog row.
      </Text>
    </Stack>
  );
}
