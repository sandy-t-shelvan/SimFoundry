# Physics-prior MLP

This branch holds the section 3.4 physics prior trained in this repository, the fitted-splat data path, and the validation split the next retrain will use. The weights are ours. They correspond to the project's section 3.4 prior. They are not a released Harvard checkpoint. The gap-matrix row for the per-Gaussian MLP stays missing for that reason.

The completed run below is the holdout-only fit. The 30 percent validation split is in the trainer and the tests. That retrain has not run.

## Purpose

Pack one fitted Gaussian splat per labeled object, run a small MLP on each in-mask Gaussian, average the 33 outputs, and supervise that mean with Gemini labels. Frozen DINOv2 ViT-S/14 and OpenCLIP ViT-B-16 supply the appearance features. splatfacto fits the Gaussians and is not updated by Adam. Only the six MLP arrays are trained.

The keyword dictionary in `sceneagent/taxonomy.py` is ours. The paper does not publish that map.

## Contract

Input is one row per Gaussian, shape `(N, 912)`.

| Columns | Contents |
| --- | --- |
| 0–383 | DINOv2 ViT-S/14 (384) |
| 384–895 | OpenCLIP ViT-B-16, OpenAI weights (512) |
| 896–911 | Geometry: `x`, `y`, `z`, three exp scales, sigmoid opacity, quaternion `rot_0`..`rot_3`, `f_dc_0`..`f_dc_2`, `log_volume`, anisotropy |

Missing ply columns are dropped, not filled. A non-finite source value drops that Gaussian. Zeros are not written in their place.

Output width is 33.

| Slots | Contents |
| --- | --- |
| 0–9 | Material logits |
| 10–15 | Body logits |
| 16–23 | Collision logits |
| 24 | `log_mass` |
| 25 | `log_density` |
| 26 | Rigidity, raw; sigmoid is outside `forward` |
| 27 | Deformability, raw; sigmoid is outside `forward` |
| 28–32 | `numeric_unpublished`, masked |

Friction and restitution stay null. Young's modulus, Poisson's ratio, and the rendering material are unpublished and are not invented.

Materials, in order: plastic, metal, wood, fabric, paper and cardboard, rubber and foam, glass and ceramic, food and organic matter, liquid and granular matter, unknown.

Body types, in order: dynamic rigid, static rigid, kinematic, soft deformable, cloth shell, fluid or granular.

Collision primitives, in order: box, sphere, cylinder, capsule, convex hull, triangle mesh, tetrahedral mesh, cloth mesh.

## Inference

The MLP runs on each in-mask Gaussian. The 33 outputs are averaged. The 912 inputs are not averaged first. A tie in the material vote returns no class. The keyword proposal is whatever the caller passes; this path does not look the name up itself. An unknown class weight is multiplied by 0.4. Soft body is deformability at least 0.65. `resolve_body_type` applies that threshold when a record is built.

`vote_loaded_prior` loads the checkpoint it is given. The fitted-splat vote loads `data/mlp/physics_prior_splat.npz`. It does not load the 4 Nov triangle-Gaussian file `physics_prior.npz`.

## Network

```
hidden = tanh(x @ w1 + b1)
hidden = tanh(hidden @ w2 + b2)
outputs = hidden @ w_out + b_out
```

| Array | Shape |
| --- | --- |
| `w1` | (912, 32) |
| `b1` | (32,) |
| `w2` | (32, 32) |
| `b2` | (32,) |
| `w_out` | (32, 33) |
| `b_out` | (33,) |

`HIDDEN` is 32, `OUTPUT_DIM` is 33, `PRIOR_DIM` is 912. Weights are Glorot uniform. Biases are zero. Sigmoid is not inside `forward`. It is applied to slots 26 and 27 in the loss and in decode. There is no RNN and no CNN. The encoders stay frozen. Adam does not step splatfacto. Only `w1`, `b1`, `w2`, `b2`, `w_out`, and `b_out` are trained.

## Optimizer and loss

`train_prior.py` uses a hand-written Adam. Learning rate `1e-3`, `beta1` 0.9, `beta2` 0.999, `eps` `1e-8`. Weight decay `1e-6` is added to the gradient before the moments. The schedule is 30 epochs from seed 0. Gradients are clipped to global L2 5.0. One object is one Adam step. `_CHUNK` 4096 is a forward and gradient allocation, not a Gaussian minibatch.

Loss, on the averaged 33-vector:

- Cross-entropy on material 0–9, body 10–15, and collision 16–23.
- Squared error on raw `log_mass` (24) and `log_density` (25).
- Squared error after sigmoid on rigidity (26) and deformability (27).
- Slots 28–32 are masked and do not enter the loss.

## Data flow

The right training input is a fitted splat, not a triangle stand-in.

1. Orbit render on CPU with Numba. 16 views: 8 azimuths at elevations 18° and 50°.
2. COLMAP.
3. splatfacto on the RTX 4090. `SPLAT_ITERS` is 600. `sh_degree` is 0.
4. Pack `(N, 912)`. Opacity is the fitted sigmoid opacity. It must not be the constant 0.99 used by the triangle prior. A `prior_input.npy` from that triangle prior is not a training row. A fitted row carries `fitted_splat.json` with `kind` `fitted`.

Gemini labels are the targets. The model is `gemini-3.8-flash` only, `thinkingLevel` `low`, temperature 0. Caps are `GEMINI_MAX_RPM` 500, `GEMINI_MAX_TPM` 1500000, and `GEMINI_MAX_RPD` 8000. Ids already marked `status=ok` are not relabeled.

## What this branch contains

The branch is cut from current `main`. This commit adds the fitted-splat prior, the label record, and this plan. It does not add the rest of the dirty working tree.

Code on the branch:

- `sceneagent/mlp/`: `prior`, `apply`, `contract`, `represent`, `splat_from_mesh`, `fitted_splat`, `stratified`, `stratum`, `labels`, `label_three`, `sources`, `check`, `pilot`, `train_prior`.
- `sceneagent/taxonomy.py` supplies the class lists and the keyword dictionary. That dictionary is ours.
- Tests: `tests/sceneagent/test_mlp_stratified.py`, `test_mlp_contract.py`, `test_mlp_views.py`, and `test_mlp_check.py` as already on `main`.
- `docs/GAP_MATRIX.md` still marks the per-Gaussian MLP missing. The note says `physics_prior_splat.npz` is our run.
- `CHANGELOG.md` records the same distinction.
- This file, `docs/mlp_physics_prior_plan.md`.

Artifacts in the commit:

- `data/mlp/stratified_labels.4090.jsonl` — local backup of the 4427 ok labels (about 5 MB).
- `data/mlp/physics_prior_splat.npz` and `data/mlp/train_report_splat.json` — the holdout-only fitted-splat run (about 260 KB and about 400 KB).
- `data/mlp/physics_prior_splat_holdout462.npz` and `data/mlp/train_report_splat_holdout462.json` — local copies of that same run.

Not in this commit:

- `data/mlp/physics_prior.npz`. That file is the 4 Nov epoch-2 triangle-Gaussian checkpoint. It is a different weight hash from `physics_prior_splat.npz`. Leave it untracked. Do not describe it as this model, and do not overwrite it.
- `data/mlp/stratified_labels.jsonl`. That local file is only the first 500 labels. It is not the full set.
- `prior_input.npy`, meshes, views, images, ply dumps, `_fitted_tmp`, and zip dumps.
- `tools/_tmp_*` and one-off probe scripts.
- The uncommitted edit in `sceneagent/mlp/check.py` and `tests/sceneagent/test_mlp_check.py`. That diff reverts the RGB / RGBA / palette download check. It is not part of the physics prior. The branch keeps the copy already on `main`.

`data/` is gitignored. The five artifact paths above are the exception recorded in this commit. `.gitignore` itself is unchanged.

## What was trained

Labels with `status=ok`: 4427. By source: YCB 70, GSO 244, ABO 4113. The remote copy of that set was `/workspace/sceneagent-simfoundry/data/mlp/stratified_labels.jsonl` on Vast instance 58301075. The local full backup is `stratified_labels.4090.jsonl`.

Fitted rows on that GPU volume: 3736. By source: YCB 0, GSO 162, ABO 3574. About 191 failed, mostly because the COLMAP mapper wrote no model. The original 500 labeled ids have no mesh on that GPU (YCB 70, GSO 80, ABO 350), so the fitted YCB count is 0.

Completed holdout-only train, from those fitted rows:

| Quantity | Value |
| --- | --- |
| `n_train` | 3274 |
| `n_holdout` | 462 |
| `n_fitted` | 3736 |
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

Known ids `002_master_chef_can`, `Weisshai_Great_White_Shark`, and `B079TXJCR5` were absent from that fit. The report has no validation slice and no test slice. Epochs ran to 30. That is the holdout-only trainer, not the validation-split retrain.

Checkpoint: `data/mlp/physics_prior_splat.npz`. Report: `data/mlp/train_report_splat.json`.

## Validation split

The split code is updated on this branch and the tests passed. The retrain has not run.

Rules for the next fit:

- Train ids are the complement of `holdout_ids`. The holdout rate is 0.12, plus the three known ids when those ids are present.
- The holdout is sorted by `sha256(id)` bytes, then by the id string.
- Validation is the first 30 percent, half-up via `divmod(3 * n, 10)`, with a remainder of 5 or more rounding up.
- For 462 holdout ids that is 139 validation and 323 test. `divmod(3 * 462, 10)` is quotient 138 and remainder 6, so the count rounds to 139.
- Early stopping uses validation loss and the existing patience. From epoch 12, stop when the best of the last 4 is not within 0.5 percent of the best before that window. The threshold is 0.995. A drop of exactly 0.5 percent does not stop, because the comparison is strict.
- Saved weights are the earliest epoch with the lowest validation loss.
- Test is scored once after the checkpoint is chosen. Test loss does not affect stopping.
- Seed 0 does not shuffle the validation slice. The seed still permutes object order inside each training epoch.

Keep the current `physics_prior_splat.npz` as the holdout462 backup when that retrain starts. Write a new `train_report_splat.json`. Do not overwrite `physics_prior.npz`.

## Known label problems

These are data problems, not code bugs. This commit does not edit labels.

- Some ABO masses are extreme. That includes furniture, and a few clear name/mass conflicts such as a charger at hundreds of kilograms.
- 26 names are the placeholder `consolidatedScene`.
- Cloth-shell rows can sit at deformability greater than or equal to 0.65. `resolve_body_type` would then rewrite them to soft deformable.

## Next steps

Do these in order. Do not rent a second machine for this prior.

1. Restart the same Vast instance 58301075 when the RTX 4090 is free. The fitted `(N, 912)` rows and the meshes are on that volume. A new GPU starts empty and would have to refit about 3736 objects, about 13 seconds each. The instance is stopped (`resources_unavailable` as of the night of 2026-11-06). The overlay was 500 GB. Do not put this dataset on the other named machines: Quebec 55073623, Texas 58591955, or Estonia 58357498.

2. Copy the updated split code up and retrain from the existing fitted rows only. No new splats. No Gemini calls. Keep the current checkpoint as the holdout462 backup. Write a new `train_report_splat.json`. Copy the results back. Do not overwrite the 4 Nov `physics_prior.npz`.

3. Gemini-label queue ids that are not `status=ok`, for later training. Skip the 4427 ok ids. Do not train on the new labels yet. Use about 4 workers, not 8. After each new label, delete that object's triangle `prior_input.npy` and its views. Do not delete meshes or fitted priors that are already ok. Stop if free disk would drop under 40 GB. Keep the caps above. Backup the new labels without overwriting `stratified_labels.jsonl` or `stratified_labels.4090.jsonl`.

4. Later, fit splats only for the newly labeled ids, then train a larger model. The roughly 191 COLMAP failures stay out until they pack.
