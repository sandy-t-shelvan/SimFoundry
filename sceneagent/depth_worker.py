# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU half of the bin's metric scale: runs in the box's ``da3`` env, started by ``bin_scale``.

For each PNG in ``--images``, one monocular Depth Anything 3 inference with
``--model``. Writes ``<stem>.npz`` holding ``depth`` (the network output,
H' x W' float32 at the processed resolution) and ``processed_hw``, plus
``meta.json`` with the model, the package path and the image sizes. The
network output is stored as is: turning it into metres (DA3METRIC-LARGE's
``focal * output / 300``) needs the camera's focal length, which
``bin_scale`` holds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args(argv)

    import depth_anything_3
    import numpy as np
    import torch
    from depth_anything_3.api import DepthAnything3
    from PIL import Image

    model = DepthAnything3.from_pretrained(args.model).to(torch.device("cuda"))
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {"model": args.model, "package": str(Path(depth_anything_3.__file__).parent)}
    views = {}
    for image_path in sorted(args.images.glob("*.png")):
        with Image.open(image_path) as image:
            width, height = image.size
        prediction = model.inference([str(image_path)])
        depth = np.asarray(prediction.depth, dtype=np.float32)[0]
        np.savez_compressed(
            args.out / f"{image_path.stem}.npz",
            depth=depth,
            processed_hw=np.array(depth.shape, dtype=np.int64),
        )
        views[image_path.name] = {
            "image_wh": [width, height],
            "processed_hw": list(depth.shape),
            "output_median": float(np.median(depth)),
        }
    meta["views"] = views
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
