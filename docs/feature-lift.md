# Per-Gaussian semantic features

Each Gaussian receives the semantic vector of the object it already belongs to. The vector is the object's existing DINOv2 ViT-S/14 crop embedding (384) concatenated with its existing OpenCLIP ViT-B-16 crop embedding (512). The file is `features.npz`, array name `features`, shape `N_gaussians × 896`. Column order matches the physics-prior row without the 16 geometry columns: DINOv2, then OpenCLIP.

The assignment array is an input from the association step. This lift does not decide membership, does not run an encoder, and does not train a codebook. A Gaussian with no object (`-1`) is left NaN. Missing numbers are not filled with zeros.

A run log line is `features shape N x 896`.

## Where the vectors come from in a run

The `prior_input` stage (`sceneagent/fitted_prior.py`) already runs DINOv2 and OpenCLIP: each accepted object's pack `prior_input/object_<n>.npy` holds one row per Gaussian of the object's fitted splat, with the patch tokens of the views that see it in columns 0–895. The object's vector is the mean of those columns over its rows, ignoring non-finite entries. That is a pooling of existing vectors; no encoder runs for the lift.

The assignment is the association's `association/objects.npz` (`object_<n>` index sets into the scene splat). A Gaussian in no set, in an object without a pack, or in two sets stays NaN.

The prior stage calls `feature_lift.run_feature_lift` once its packs are written and records the shape line in its ledger entry (`features`) and in `prior_input_ledger.json` (`feature_lift`). A failed lift is recorded there and does not undo the packs. A finished run can be lifted without writing into it:

    python -m sceneagent.feature_lift --out RUN --dest DIR
