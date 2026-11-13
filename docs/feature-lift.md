# Per-Gaussian semantic features

Each Gaussian receives the semantic vector of the object it already belongs to. The vector is the object's existing DINOv2 ViT-S/14 crop embedding (384) concatenated with its existing OpenCLIP ViT-B-16 crop embedding (512). The file is `features.npz`, array name `features`, shape `N_gaussians × 896`. Column order matches the physics-prior row without the 16 geometry columns: DINOv2, then OpenCLIP.

The assignment array is an input from the association step. This lift does not decide membership, does not run an encoder, and does not train a codebook. A Gaussian with no object (`-1`) is left NaN. Missing numbers are not filled with zeros.

A run log line is `features shape N x 896`.
