# Physics-prior MLP

This file is the GitHub guide for branch `mlp-physics-prior`. A person opening this branch can see what was trained, where the rows live, and how to finish. The shorter build record is [mlp_physics_prior_plan.md](mlp_physics_prior_plan.md). Follow this file for the data locations and the remaining steps.

The weights in `data/mlp/physics_prior_splat.npz` are the weights trained in this repository. They are our run of the section 3.4 prior.

## 1. What this branch is

This branch is the physics-prior MLP only. It carries the trainer, the Gemini labeling path, the tests for that path, this documentation, and small checkpoints. Meshes and per-Gaussian `prior_input.npy` dumps are not in git. Together those dumps are about 150 GB. `data/` is gitignored. The five artifact paths below are the tracked exceptions.

Committed label and checkpoint files:

| Path | What it is |
| --- | --- |
| `data/mlp/stratified_labels.4090.jsonl` | The 4427-row Gemini label set. One JSON object per ok label, about 5 MB. This is the labeled output. |
| `data/mlp/physics_prior_splat.npz` | Finished holdout-only fitted-splat checkpoint, about 260 KB. |
| `data/mlp/train_report_splat.json` | Report for that holdout-only run. |
| `data/mlp/physics_prior_splat_holdout462.npz` | Backup of that same checkpoint. Keep it before the validation retrain. |
| `data/mlp/train_report_splat_holdout462.json` | Backup of that same report. |

`data/mlp/stratified_labels.4090.jsonl` is in this branch. It is the 4427-row Gemini label set (YCB 70, GSO 244, ABO 4113, every row `status=ok`). The meshes and the fitted `prior_input.npy` files are not in git.

The Gemini labeling code is on this branch. These modules download or check meshes, render views, call Gemini, and write `stratified_labels` rows. Their content is the copy already committed here (the same bytes as `main` for every file except the stratified test, which this branch updates):

| Path | Role |
| --- | --- |
| `sceneagent/mlp/sources.py` | Download catalog meshes and accept a sample only after `check_sample`. |
| `sceneagent/mlp/check.py` | Mesh usability check used by `sources.py`. The branch keeps the copy already on `main`. |
| `sceneagent/mlp/represent.py` | Render labeling views and run the frozen DINOv2 and OpenCLIP encoders. |
| `sceneagent/mlp/splat_from_mesh.py` | Load a textured mesh. The triangle stand-in also lives here. A triangle `prior_input.npy` is not a training row. |
| `sceneagent/mlp/stratum.py` | Material strata and the stop rule for the label set. |
| `sceneagent/mlp/labels.py` | Label schema. `parse_label` logs mass and density once. |
| `sceneagent/mlp/label_three.py` | `gemini-3.8-flash` request: `thinkingLevel` `low`, temperature 0, and the RPM / TPM / RPD caps. The prompt and response schema are in this file. |
| `sceneagent/mlp/stratified.py` | Queue, acquire, and the writer for `data/mlp/stratified_labels.jsonl`. |
| `sceneagent/taxonomy.py` | Material, body, and collision names. There is no `sceneagent/mlp/taxonomy.py`. |
| `sceneagent/gemini_client.py` | HTTP client imported by `label_three.py`. It redacts credentials. Do not print API keys. |

Tests on this branch: `tests/sceneagent/test_mlp_stratified.py`, `tests/sceneagent/test_mlp_contract.py`, `tests/sceneagent/test_mlp_views.py`.

The fitted-splat packer is `sceneagent/mlp/fitted_splat.py`. The trainer is `sceneagent/mlp/train_prior.py`. Voting goes through `sceneagent/mlp/apply.py` `vote_loaded_prior`. The fitted vote path loads `data/mlp/physics_prior_splat.npz`.

## 2. What happened

Work ran in this order.

### Contract

Input is one row per Gaussian, shape `(N, 912)`. Output width is 33.

| Columns | Contents |
| --- | --- |
| 0–383 | DINOv2 ViT-S/14 (384) |
| 384–895 | OpenCLIP ViT-B-16, OpenAI weights (512) |
| 896–911 | Geometry: `x`, `y`, `z`, three exp scales, sigmoid opacity, quaternion `rot_0`..`rot_3`, `f_dc_0`..`f_dc_2`, `log_volume`, anisotropy |

Missing ply columns are dropped. They are not filled. A non-finite source value drops that Gaussian.

| Slots | Contents |
| --- | --- |
| 0–9 | Material logits |
| 10–15 | Body logits |
| 16–23 | Collision logits |
| 24 | `log_mass` |
| 25 | `log_density` |
| 26 | Rigidity, raw. Sigmoid is outside `forward`. |
| 27 | Deformability, raw. Sigmoid is outside `forward`. |
| 28–32 | `numeric_unpublished`, masked |

Friction and restitution stay null. Young's modulus, Poisson's ratio, and the rendering material are unpublished. Leave them unpublished.

Inference averages the 33 outputs across in-mask Gaussians. It does not average the 912 inputs and then run the MLP once. A tie returns no class. The keyword proposal is whatever the caller passes. An unknown class weight is multiplied by 0.4 (`UNKNOWN_WEIGHT_SCALE` in `sceneagent/physics.py`). Soft body is deformability at least 0.65. `resolve_body_type` applies that threshold when a record is built.

### Network and loss

Two hidden layers, width 32, tanh. Shapes: `w1` (912, 32), `b1` (32,), `w2` (32, 32), `b2` (32,), `w_out` (32, 33), `b_out` (33,). Sigmoid is not inside `forward`. It is applied to slots 26 and 27 in the loss and when decoding. Weights are Glorot uniform. Biases are zero. There is no RNN and no CNN. The encoders and splatfacto stay frozen.

Adam is hand-written in `train_prior.py`: learning rate `1e-3`, `beta1` 0.9, `beta2` 0.999, `eps` `1e-8`, weight decay `1e-6`, 30 epochs, seed 0, global L2 clip 5.0, one object per step.

Loss, on the averaged 33-vector:

- Cross-entropy on slots 0–9, 10–15, and 16–23.
- Squared error on slots 24 and 25.
- Squared error after sigmoid on slots 26 and 27.
- Slots 28–32 are masked and do not enter the loss.

### Gemini targets

Labels come from `gemini-3.8-flash` only, `thinkingLevel` `low`, temperature 0. Caps in `label_three.py`: 500 RPM, 1,500,000 TPM, 8,000 RPD. Do not relabel an id whose `status` is already `ok`.

### Labeling on Vast 58301075

Labeling on Vast instance 58301075 reached 4427 ok rows (YCB 70, GSO 244, ABO 4113) and then stopped because the disk filled. The job was not finished. The queue is about 7538 ids (YCB 71, GSO 1024, ABO 6443). Permanent early rejects were about 28.

### Triangle priors were the wrong training input

Triangle `prior_input.npy` files, with opacity fixed at 0.99, are the wrong training input. Those files were deleted for already-ok objects during cleanup. The right input is a fitted splat:

1. 16 CPU orbit views: 8 azimuths at elevations 18° and 50° (`ORBIT_AZIMUTHS`, `ORBIT_ELEVATIONS` in `fitted_splat.py`).
2. COLMAP.
3. splatfacto for 600 steps (`SPLAT_ITERS`).
4. Pack `(N, 912)`. Opacity is the fitted sigmoid opacity. It must vary.

A fitted row has `fitted_splat.json` with `kind` `fitted` beside `prior_input.npy`.

### Fitting on that GPU

Fitting on the same GPU packed 3736 objects (YCB 0, GSO 162, ABO 3574). About 191 failed, mostly because COLMAP wrote no model. The original 500 labeled ids have no mesh on that GPU (YCB 70, GSO 80, ABO 350), so no YCB object was fitted.

### The training run that finished

The run that finished used the fitted rows and a 12% id holdout. It did not use the later validation split.

| Quantity | Value |
| --- | --- |
| `n_train` | 3274 |
| `n_holdout` | 462 |
| Epoch loss | 3.460 → 0.924 |
| Scored train loss | 0.838 |
| Train material accuracy | 0.981 |
| Holdout material | 0.894 |
| Holdout body | 0.978 |
| Holdout collision | 0.777 |
| Holdout MAE log mass | 0.365 |
| Holdout MAE log density | 0.557 |
| Holdout MAE rigidity | 0.039 |
| Holdout MAE deformability | 0.038 |
| Holdout loss | 2.399 |

The three known ids (`002_master_chef_can`, `Weisshai_Great_White_Shark`, `B079TXJCR5`) were absent. The checkpoint is `data/mlp/physics_prior_splat.npz`. The report is `data/mlp/train_report_splat.json`. Epochs ran to 30. That report has no validation slice and no test slice.

`data/mlp/physics_prior.npz` is a different file: the 4 November epoch-2 triangle-Gaussian run. Leave it untouched. Do not describe it as this model, and do not overwrite it.

### Validation split, not yet trained

`train_prior.py` now splits that holdout into validation and test. The retrain has not run.

- Sort the holdout by `sha256(id)` bytes, then by the id string.
- Validation is the first 30%, half-up: `divmod(3 * n, 10)`. A remainder of 5 or more rounds up.
- For 462 holdout ids that is 139 validation and 323 test. `divmod(3 * 462, 10)` is quotient 138 and remainder 6, so the count rounds to 139.
- Train stays 3274.
- Early stopping uses validation loss. From epoch 12, stop when the best of the last 4 is not within the 0.995 factor of the best before that window. The comparison is strict, so a drop of exactly 0.5% does not stop.
- Save the earliest epoch with the lowest validation loss.
- Score the test ids once after that checkpoint is chosen. Test loss does not affect stopping.
- Seed 0 does not shuffle the validation slice. The seed still permutes object order inside each training epoch.

The fitted vote path loads `physics_prior_splat.npz` (`SPLAT_CHECKPOINT` in `train_prior.py`, passed into `vote_loaded_prior`).

### Known label issues

Leave these as they are. Do not relabel them in this task.

- Some ABO masses are extreme.
- 26 names are the placeholder `consolidatedScene`.
- Cloth-shell rows with deformability greater than or equal to 0.65 would be rewritten to soft deformable by `resolve_body_type`.

## 3. Where the data is

A copy onto Vast instance 59601312 was requested on 7 November 2026. That copy may still be running. This file does not record the copy as finished. Use 59601312 as the working machine only after its counts match the source. Compare `fitted_splat.json` count, `prior_input.npy` count, and the label line count. Do not wait out a multi-hour rsync before reading this; check the destination when you are ready to train.

### Source volume

Vast instance 58301075 (RTX 4090). This instance was stopped with `resources_unavailable` on the night of 6 November 2026.

Path: `/workspace/sceneagent-simfoundry/data/mlp/`.

- Labels on that machine: `stratified_labels.jsonl` (about 4427 lines).
- Fitted rows: `prior_input.npy` plus `fitted_splat.json` next to each packed object.
- Meshes and textures are under that same tree (`abo`, `gso`, `ycb`).
- Checkpoints: `physics_prior_splat.npz`, `train_report_splat.json`.
- Logs: `/workspace/sceneagent-simfoundry/logs/` (`stratified_acquire.log`, `fitted_convert.log`).

Known source figures: 4427 ok label lines, and 3736 fitted objects (YCB 0, GSO 162, ABO 3574). Each fitted object has a varying-opacity `(N, 912)` `prior_input.npy` and a `fitted_splat.json`.

### Requested destination

Vast instance 59601312, same paths under `/workspace/sceneagent-simfoundry/`. Treat it as the working machine only after the three counts match the source. If the copy is not there, or the counts do not match, start 58301075 when the RTX 4090 is free. Do not rent a third machine.

### Local Windows checkout

`C:\Users\sande\cursor\sceneagent-simfoundry\data\mlp\`:

- `stratified_labels.4090.jsonl` is the full 4427-row label backup, and it is the file committed on this branch. Use it when the GPU is stopped.
- `stratified_labels.jsonl` is an older 500-row snapshot. It is not in git. Do not use it as the full label set, and do not overwrite it with the 4427-row file.
- `physics_prior_splat.npz` and `train_report_splat.json` are the finished holdout-only run. Both are on this branch.
- `physics_prior_splat_holdout462.npz` and `train_report_splat_holdout462.json` are the backups to keep before the validation retrain overwrites the main checkpoint. Both are on this branch.
- `physics_prior.npz` is the 4 November triangle run. It is not on this branch. Leave it untouched.

### Not part of this project

These paths and machines are not on this branch and are not part of this prior: `/root/SceneagentX`, `/root/sam-3d-objects`, Isaac Sim, mamba caches, Quebec 55073623, Texas 58591955, Estonia 58357498.

## 4. Label record

Each ok row in `stratified_labels.4090.jsonl` is JSON from `gemini-3.8-flash`, temperature 0, thinking level low. The file on this branch has 4427 lines. Every line has `status` `ok`. Sources: YCB 70, GSO 244, ABO 4113.

Fields include `id`, `name`, `source`, `status`, `material`, `body_type`, `collision`, `mass_kg`, `density_kg_m3`, `log_mass`, `log_density`, `rigidity`, `deformability`, `friction_static` null, `friction_dynamic` null, `restitution` null. Rows also carry bookkeeping such as `model_id`, `temperature`, `thinking_level`, `views`, and `prior_input`. Do not relabel an id whose status is already ok.

Material names, in order: plastic, metal, wood, fabric, paper and cardboard, rubber and foam, glass and ceramic, food and organic matter, liquid and granular matter, unknown.

Body: dynamic rigid, static rigid, kinematic, soft deformable, cloth shell, fluid or granular.

Collision: box, sphere, cylinder, capsule, convex hull, triangle mesh, tetrahedral mesh, cloth mesh.

`sceneagent/mlp/labels.py` requires `material`, `body_type`, `collision`, `mass_kg`, `density_kg_m3`, `rigidity`, and `deformability`. `parse_label` stores natural logs of mass and density. Friction and restitution are parsed beside those keys and stay null.

## 5. How to complete the project

Do these in order. The retrain uses only the 3736 fitted rows that already exist. New Gemini labels are for a later, larger train.

1. Confirm the fitted rows and `stratified_labels.jsonl` are on the machine you will train on. The preferred machine is 59601312 after the copy matches the source counts in section 3. If that copy is not there, start 58301075 when the RTX 4090 is free. Do not rent a third machine. Do not refit the 3736 objects that already have a varying-opacity `(N, 912)` prior.

2. Put this branch's code on that machine. The trainer that matters is `sceneagent/mlp/train_prior.py`. The labeling modules in section 1 are already on the branch.

3. Retrain from the existing fitted rows only:

   ```bash
   python -m sceneagent.mlp.train_prior fitted
   ```

   No new splats. No Gemini. Keep `physics_prior_splat_holdout462.npz` and `train_report_splat_holdout462.json`. Let the new run write `physics_prior_splat.npz` and `train_report_splat.json`. Early stop on validation loss. Score the 323 test ids once. Copy the new checkpoint and report back to the local `data/mlp/` folder. Do not overwrite `physics_prior.npz`.

   `python -m sceneagent.mlp.train_prior` without `fitted` calls `train_and_report()` and writes `physics_prior.npz`. Use the `fitted` argument.

4. Then label the remaining queue ids with Gemini, for later training, not for this retrain. Skip every id that is already `status=ok`. About 4 workers:

   ```bash
   python -m sceneagent.mlp.stratified acquire 4
   ```

   Model and caps are in section 2. `acquire` is required. The default stage `all` (`python -m sceneagent.mlp.stratified` with no arguments) calls `train_and_report()` after labeling and writes `physics_prior.npz`.

   The acquire writer rewrites `data/mlp/stratified_labels.jsonl`. On the GPU that path is the 4427-row set. Do not let the new run replace it. Append the new ok rows to a new file, `data/mlp/stratified_labels.remaining.jsonl`. Do not overwrite `stratified_labels.jsonl` or `stratified_labels.4090.jsonl`. Load the 4427 ok ids as the skip set. `run_acquire` refuses to relabel an id already in that set.

   After each new label is saved, delete that new object's triangle `prior_input.npy` and its views. Do not delete meshes. Do not delete fitted priors for ids that are already ok. `_delete_unprotected` removes a whole object directory, including the mesh. Do not use it on an object you still need. Stop if free disk would fall under 40 GB.

5. Later, fit splats only for newly labeled ids, then train a larger model. Leave the roughly 191 COLMAP failures out until they pack.

## 6. What not to do

- Do not commit meshes, `prior_input.npy`, views, ply files, or `_fitted_tmp`.
- Do not print API keys or token values.
- Do not describe the 4 November `physics_prior.npz` as the trained fitted model, and do not overwrite that file.
- Do not relabel an id whose `status` is already `ok`.
- Do not train this retrain on labels that are not already in the 4427-row set.
- Do not refit the 3736 objects that already have a varying-opacity `(N, 912)` prior.
- Do not rent a third Vast machine for this prior.
- Do not put this dataset on Quebec 55073623, Texas 58591955, or Estonia 58357498.
