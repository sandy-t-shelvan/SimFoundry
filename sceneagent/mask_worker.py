# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU half of SF-15: runs inside the box's gsam2 env, started by ``bin_masks``.

For each PNG in ``--images``: GroundingDINO boxes for ``--prompt``, the
filter in :func:`sceneagent.mask_filter.filter_detections`, then one SAM vit_h
mask per kept box. Writes ``<stem>.npz`` (``masks``: kept x H x W booleans,
in kept order) and ``detections.json`` (every detection, kept or with its
drop reason, and the mask index of each kept one).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sceneagent.mask_filter import filter_detections


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--box-threshold", type=float, required=True)
    parser.add_argument("--text-threshold", type=float, required=True)
    parser.add_argument("--gdino-config", required=True)
    parser.add_argument("--gdino-weights", required=True)
    parser.add_argument("--sam-weights", required=True)
    args = parser.parse_args(argv)

    import numpy as np
    import torch
    from groundingdino.util.inference import load_image, load_model, predict
    from segment_anything import SamPredictor, sam_model_registry

    device = "cuda"
    detector = load_model(args.gdino_config, args.gdino_weights, device=device)
    sam = sam_model_registry["vit_h"](checkpoint=args.sam_weights).to(device)
    predictor = SamPredictor(sam)
    args.out.mkdir(parents=True, exist_ok=True)
    report: dict[str, list[dict]] = {}
    for image_path in sorted(args.images.glob("*.png")):
        rgb, tensor = load_image(str(image_path))
        height, width = rgb.shape[:2]
        boxes, logits, phrases = predict(
            model=detector,
            image=tensor,
            caption=args.prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            device=device,
        )
        detections = []
        for (cx, cy, w, h), logit, phrase in zip(
            boxes.tolist(), logits.tolist(), phrases, strict=True
        ):
            box = [(cx - w / 2) * width, (cy - h / 2) * height]
            box += [(cx + w / 2) * width, (cy + h / 2) * height]
            detections.append(
                {"phrase": str(phrase), "logit": float(logit), "box_xyxy": box}
            )
        detections = filter_detections(detections, width, height, args.prompt)
        kept = [d for d in detections if d["kept"]]
        masks = np.zeros((len(kept), height, width), dtype=bool)
        if kept:
            predictor.set_image(rgb)
            box_tensor = torch.tensor([d["box_xyxy"] for d in kept], device=device)
            transformed = predictor.transform.apply_boxes_torch(
                box_tensor, (height, width)
            )
            with torch.no_grad():
                out, _scores, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed,
                    multimask_output=False,
                )
            masks = out[:, 0].cpu().numpy().astype(bool)
        for index, det in enumerate(kept):
            det["mask_index"] = index
            det["mask_area_px"] = int(masks[index].sum())
        np.savez_compressed(args.out / f"{image_path.stem}.npz", masks=masks)
        report[image_path.name] = detections
    (args.out / "detections.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
