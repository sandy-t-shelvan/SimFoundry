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
