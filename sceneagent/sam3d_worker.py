# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU half of SF-29: runs in the box's ``sam3d-objects`` env, started by ``bin_sam3d``.

Reads ``--jobs`` (a JSON list of ``{"object", "image", "masks", "mask_index"}``)
and runs SAM 3D Objects once per job, with the upstream notebook's
``Inference(config, compile=False)`` and ``seed=42``. For each object it
writes ``object_<n>.glb`` (the pipeline's ``glb``: mesh decoder output, no
post-processing or texture baking, which is ``Inference.__call__``'s
default) and ``object_<n>_pose.json`` (``rotation`` wxyz, ``translation``,
``scale``, local to camera in PyTorch3D's camera frame, as the pipeline's
pose decoder names them). One object failing is recorded and the rest
still run. ``results.json`` lists every job's outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def _floats(value) -> list:
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=float).reshape(-1).tolist()


def mask_depth(pointmap, mask) -> float | None:
    """Median depth of SAM 3D's own pointmap (MoGe, metres) inside the object mask.

    The pointmap can come back smaller than the image, so the mask is sampled
    down to it by nearest neighbour. ``bin_place`` divides its own depth of the
    object by this to rescale SAM 3D's pose into the splat's metres.
    """
    import numpy as np

    if pointmap is None:
        return None
    if hasattr(pointmap, "detach"):
        pointmap = pointmap.detach().float().cpu().numpy()
    points = np.asarray(pointmap, dtype=float)
    height, width = points.shape[:2]
    rows = np.arange(height) * mask.shape[0] // height
    cols = np.arange(width) * mask.shape[1] // width
    inside = mask[np.ix_(rows, cols)]
    z = np.abs(points[..., 2][inside])
    z = z[np.isfinite(z) & (z > 0)]
    return float(np.median(z)) if z.size else None


def without_dino_download(hub_load):
    """``torch.hub.load`` that builds DINOv2 without fetching its pretrained weights.

    SAM 3D's Dino embedder calls ``torch.hub.load("facebookresearch/dinov2",
    ...)``, which downloads the weights from dl.fbaipublicfiles.com. That
    host's certificate had expired on 2026-11-12 (runs 43179492561 to
    43181696152, and curl elsewhere). The download only seeds weights that
    SAM 3D then replaces: the condition embedder, DINO included, is loaded
    from ss_generator.ckpt and slat_generator.ckpt with strict=True, so a
    DINO weight missing from those raises instead of staying random.
    """

    def load(repo_or_dir, model, *args, **kwargs):
        if "dinov2" in str(repo_or_dir):
            kwargs["pretrained"] = False
        return hub_load(repo_or_dir, model, *args, **kwargs)

    return load


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--notebook", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    sys.path.insert(0, args.notebook)
    import numpy as np
    import torch

    torch.hub.load = without_dino_download(torch.hub.load)

    # The upstream notebook's module, found through --notebook on sys.path;
    # imported by name because "inference" is too generic to list for mypy.
    Inference = importlib.import_module("inference").Inference
    from PIL import Image

    jobs = json.loads(args.jobs.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    model = Inference(args.config, compile=False)
    results: dict = {
        "load_seconds": round(time.monotonic() - started, 1),
        "objects": [],
    }
    for job in jobs:
        name = f"object_{job['object']}"
        record: dict = {"object": job["object"], "image": job["image"]}
        torch.cuda.reset_peak_memory_stats()
        tic = time.monotonic()
        try:
            with Image.open(job["image"]) as image:
                rgb = np.array(image.convert("RGB"), dtype=np.uint8)
            mask = np.load(job["masks"])["masks"][job["mask_index"]].astype(bool)
            output = model(rgb, mask, seed=args.seed)
            pose: dict[str, Any] = {
                k: _floats(output[k]) for k in ("rotation", "translation", "scale")
            }
            pose.update(convention="local to camera, PyTorch3D camera frame, wxyz")
            depth = mask_depth(output.get("pointmap"), mask)
            if depth is not None:
                pose["mask_depth_m"] = depth
            glb = output.get("glb")
            if glb is None:
                raise RuntimeError(f"no glb in the output (keys {sorted(output)})")
            glb_path = args.out / f"{name}.glb"
            glb.export(str(glb_path))
            digest = hashlib.sha256(glb_path.read_bytes()).hexdigest()
            pose["glb_sha256"] = digest
            (args.out / f"{name}_pose.json").write_text(
                json.dumps(pose, indent=2) + "\n"
            )
            record.update(
                ok=True,
                glb=str(glb_path),
                glb_bytes=glb_path.stat().st_size,
                glb_sha256=digest,
                vertices=len(getattr(glb, "vertices", [])),
                pose=pose,
            )
        except Exception as exc:  # one object's failure must not stop the rest
            record.update(
                ok=False,
                error=f"{type(exc).__name__}: {exc}"[-600:],
                trace=traceback.format_exc()[-1500:],
            )
        record["seconds"] = round(time.monotonic() - tic, 1)
        record["peak_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
        results["objects"].append(record)
        torch.cuda.empty_cache()
        (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
