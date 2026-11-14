# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KAN-21 and KAN-22 on a finished video run: crops, the vote, the audit.

    python -m sceneagent.material_run --run RUN/ --out OUT/ [--views 3]

Reads a finished ``video_run`` directory and never writes inside it. Into
``OUT`` it writes:

- ``crops/<object name>/<k>.png``: each placed object cut from its mask views
  with :func:`sceneagent.bin_review.crop_to_object`, the layout
  :func:`sceneagent.material_audit.load_crops` reads.
- ``material_vote.json`` and ``manifest.json``: equation (2) through
  :func:`sceneagent.material_vote.run_material_vote` (sources, weights, the
  0.4 unknown scale, ``vlm_calls``).
- ``material_review.json``: :func:`sceneagent.material_audit.audit_votes` on
  that vote and the crops (provisional class, audit class, reason,
  ``signed_by``, ``vlm_calls``).
- ``physics/object_<n>.json``: :func:`sceneagent.physics_props.write_property_set`
  with the vote and the review, so the reviewed class reaches the property set.

Both VLM passes use the shared client (:func:`sceneagent.gemini_client.client_from_environ`),
which reads ``GEMINI_API_KEY`` from the environment and nothing else; one vote
and one audit call per object. The prior joins the vote when the run's packs
and the pinned checkpoint are present (``physics_props._prior_inputs``), for
every object or for none.

The vote adapter follows ``HarnessVlm`` in draft PR #61
(``sceneagent/material_runner.py``), but calls the shared client's
``generate_images`` so the crops go up as PNG.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sceneagent.bin_review import crop_to_object
from sceneagent.gemini_client import call_failed
from sceneagent.material_audit import audit_votes, load_crops, write_review
from sceneagent.material_vote import (
    MaterialVlmError,
    parse_vlm_vote,
    run_material_vote,
    vlm_vote_prompt,
)
from sceneagent.physics_props import (
    _fit_fallbacks,
    _prior_inputs,
    object_names,
    write_property_set,
)
from sceneagent.vlm_port import ImageClient

DEFAULT_VIEWS = 3


def placed_objects(run: Path) -> list[dict[str, Any]]:
    """Placed objects in ``bin_score.json`` order, named as ``run_physics`` names them."""
    association = json.loads(
        (run / "association_evidence.json").read_text(encoding="utf-8")
    )
    placement = json.loads((run / "bin_score.json").read_text(encoding="utf-8"))
    names = object_names(association)
    masks = {
        int(o["object"]): list(o.get("masks") or [])
        for o in association.get("object_list", [])
    }
    objects = [
        {
            "object": int(r["object"]),
            "name": names.get(int(r["object"]), "object"),
            "masks": masks.get(int(r["object"]), []),
        }
        for r in placement["icp"]["objects"]
        if r.get("placed")
    ]
    if not objects:
        raise ValueError("no placed object")
    seen = [o["name"] for o in objects]
    if len(set(seen)) != len(seen):
        # load_crops keys crops by folder name; two "apple"s would collide.
        raise ValueError(f"placed objects share a name: {seen}")
    return objects


def _spread(items: Sequence[Any], count: int) -> list[Any]:
    """``count`` items evenly spaced over ``items``, so crops span the orbit."""
    if len(items) <= count:
        return list(items)
    picks = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[i] for i in sorted(set(picks.tolist()))]


def write_crops(
    run: Path | str, crops_dir: Path | str, views: int = DEFAULT_VIEWS
) -> list[dict[str, Any]]:
    """``crops_dir/<name>/<k>.png`` for each placed object, from its mask views.

    A view whose mask is empty or missing is skipped. An object left with no
    crop is an error, so the vote never spends a call on nothing.
    """
    from PIL import Image

    run, crops_dir = Path(run), Path(crops_dir)
    written = []
    for obj in placed_objects(run):
        usable = []
        for entry in obj["masks"]:
            view = str(entry.get("view") or "")
            mask_file = run / "masks" / f"{Path(view).stem}.npz"
            image_file = run / "mask_views" / view
            if not view or not mask_file.is_file() or not image_file.is_file():
                continue
            usable.append((view, image_file, mask_file, int(entry["mask_index"])))
        folder = crops_dir / obj["name"]
        folder.mkdir(parents=True, exist_ok=True)
        used: list[str] = []
        for view, image_file, mask_file, index in _spread(usable, views):
            with np.load(mask_file) as stack:
                all_masks = stack["masks"]
            if index >= len(all_masks) or not all_masks[index].any():
                continue
            image = np.asarray(Image.open(image_file).convert("RGB"))
            crop = crop_to_object(image, all_masks[index])
            Image.fromarray(crop).save(folder / f"{len(used):02d}.png")
            used.append(view)
        if not used:
            raise ValueError(f"object {obj['object']} ({obj['name']}) has no crop")
        written.append({"object": obj["object"], "name": obj["name"], "views": used})
    return written


class SharedClientVlm:
    """``classify_crops`` for ``run_material_vote`` over ``generate_images``."""

    def __init__(self, client: ImageClient) -> None:
        self._client = client

    def __repr__(self) -> str:
        return "SharedClientVlm()"

    def classify_crops(self, crops: Sequence[bytes]) -> dict[str, Any]:
        reply = self._client.generate_images(vlm_vote_prompt(), images=list(crops))
        failure = call_failed(reply)
        if failure is not None:
            raise MaterialVlmError(f"vote call failed: {failure}")
        return parse_vlm_vote(str(reply.get("text") or ""))


def run(
    run_dir: Path | str,
    out_dir: Path | str,
    client: ImageClient,
    *,
    views: int = DEFAULT_VIEWS,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Crops, vote, audit and property set for one finished run."""
    run_dir, out_dir = Path(run_dir).resolve(), Path(out_dir).resolve()
    if out_dir == run_dir or run_dir in out_dir.parents:
        raise ValueError("--out must not be inside the run directory")
    crops_dir = out_dir / "crops"
    written = write_crops(run_dir, crops_dir, views)
    crops = load_crops(crops_dir)
    objects = [{"object": o["object"], "name": o["name"]} for o in written]

    packs, used = _prior_inputs(run_dir, checkpoint)
    fallbacks = _fit_fallbacks(run_dir)
    have_packs = packs is not None and all(
        (packs / f"object_{o['object']}.npy").is_file() for o in objects
    )
    vote_objects = []
    for obj in objects:
        entry: dict[str, Any] = {"name": obj["name"], "crops": crops[obj["name"]]}
        if have_packs and packs is not None:
            features = np.load(packs / f"object_{obj['object']}.npy")
            entry["features"] = features
            entry["mask"] = np.ones(features.shape[0], dtype=bool)
        vote_objects.append(entry)
    # run_material_vote uses a prior whenever its checkpoint path is a file,
    # and then needs features for every object. Without packs for all of them,
    # point it at a path that is not a file so the prior is a null source.
    vote_checkpoint = used if have_packs else out_dir / "no_prior_for_this_vote.npz"
    vote = run_material_vote(
        out_dir,
        vote_objects,
        client=SharedClientVlm(client),
        checkpoint=vote_checkpoint,
    )
    vote_doc = json.loads(Path(vote["vote_path"]).read_text(encoding="utf-8"))
    review = audit_votes(vote_doc["votes"], crops, client)
    review["source_run"] = str(run_dir)
    review_path = write_review(review, out_dir / "material_review.json")

    properties = write_property_set(
        objects,
        packs if have_packs else None,
        out_dir,
        checkpoint=used if have_packs else None,
        vote=vote_doc,
        review=review,
        fallbacks=fallbacks,
    )
    summary = {
        "source_run": str(run_dir),
        "objects": written,
        "prior_in_vote": bool(have_packs),
        "vote_vlm_calls": vote["manifest"]["vlm_calls"],
        "audit_vlm_calls": review["vlm_calls"],
        "vlm_calls": vote["manifest"]["vlm_calls"] + review["vlm_calls"],
        "provisional": {v["object"]: v["material_class"] for v in vote_doc["votes"]},
        "final": {r["name"]: r["final_class"] for r in review["reviews"]},
        "replaced": review["replaced"],
        "failed": review["failed"],
        "vote_path": vote["vote_path"],
        "review_path": str(review_path),
        "physics": [
            f"physics/object_{r['object_number']}.json" for r in properties["records"]
        ],
    }
    (out_dir / "material_run.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None, client: ImageClient | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True, help="finished run dir")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--views", type=int, default=DEFAULT_VIEWS)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args(argv)
    if client is None:
        from sceneagent.gemini_client import client_from_environ

        client = client_from_environ()
    summary = run(
        args.run, args.out, client, views=args.views, checkpoint=args.checkpoint
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in ("vlm_calls", "provisional", "final", "replaced")
            }
        )
    )
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
