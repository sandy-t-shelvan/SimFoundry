# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KAN-42: PartSAM parts for one mesh, in the shape ``hinge_proposal`` reads.

Runs in PartSAM's own env (``/root/.local/share/mamba/envs/partsam``), not
in the pipeline's, so it imports nothing from ``sceneagent``:

    /root/.local/share/mamba/envs/partsam/bin/python sceneagent/partsam_worker.py \
        --mesh RUN/sam3d/object_1.glb --out RUN/parts/object_1

PartSAM (github.com/czvvd/PartSAM, b16d3e8) is run unmodified: its
``evaluation/eval_everypart.py`` with the stock ``configs/partsam.yaml``,
pointed at one mesh by Hydra overrides, writing into ``--out``. Its result
keeps each part only as a face colour drawn from a shuffled palette, so
colours can repeat once a mesh has more parts than the palette. This wrapper
replaces ``post_processing`` with the same steps (PartSAM's own helpers, in
the same order) that also keep the per-face label array, then splits the
mesh into one file per part and writes ``parts.json``:
``[{name, faces, bbox_min, bbox_max, centroid, mesh}]``. No PartSAM file is
edited.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import shutil
import sys
from pathlib import Path

PARTSAM = Path("/workspace/PartSAM")
# Parts below this share of the mesh's faces are graph-cut crumbs, not parts
# a joint could move; they are reported under "dropped", not silently lost.
MIN_FACE_SHARE = 0.01


def _patch_post_processing(store: dict) -> None:
    import numpy as np
    import torch
    import utils.infer_utils as iu  # type: ignore[import-not-found]

    original = iu.post_processing

    def post_processing(mesh_group, mesh, cfg):
        # The first half of PartSAM's post_processing (utils/infer_utils.py,
        # b16d3e8), in its order, so the labels can be kept.
        face_labels = torch.from_numpy(mesh_group)
        label_dict = dict(zip(range(len(face_labels)), face_labels.tolist()))
        mesh_graph = iu.construct_mesh_graph(mesh)
        components = iu.label_components(label_dict, mesh, mesh_graph)
        label_smoothed = iu.smooth(label_dict, mesh, mesh_graph, components, cfg)
        for face in range(len(mesh.faces)):
            if face not in label_smoothed:
                label_smoothed[face] = 0
        components = iu.label_components(label_smoothed, mesh, mesh_graph)
        consistent = iu.split(label_smoothed, components)
        if cfg.use_graph_cut:
            consistent = iu.smooth_repartition_faces(consistent, None, mesh)
        labels = np.zeros(len(mesh.faces), dtype=np.int64)
        for face, label in consistent.items():
            labels[int(face)] = int(label)
        store["labels"] = labels
        store["vertices"] = np.asarray(mesh.vertices)
        store["faces"] = np.asarray(mesh.faces)
        # PartSAM's own colouring of the result, unchanged.
        return original(mesh_group, mesh, cfg)

    iu.post_processing = post_processing


def run_partsam(
    mesh: Path, out: Path, partsam: Path = PARTSAM, batch_size: int = 8
) -> dict:
    """PartSAM on one mesh; the per-face labels and the mesh it labelled."""
    work = out / "partsam_work"
    shutil.rmtree(work, ignore_errors=True)
    (work / "input").mkdir(parents=True)
    (work / "results").mkdir()
    shutil.copyfile(mesh, work / "input" / mesh.name)
    sys.path.insert(0, str(partsam))
    store: dict = {}
    _patch_post_processing(store)
    argv, cwd = sys.argv, os.getcwd()
    try:
        os.chdir(work)  # PartSAM writes results/<id>.ply under the cwd
        sys.argv = [
            "eval_everypart.py",
            f"dataset.root_dir={work / 'input'}",
            f"eval_params.ckpt_path={partsam / 'pretrained' / 'model.safetensors'}",
            # PartSAM's config: "Reduce the batch_size when OOM". The stock 32
            # ran out of the 4090's 24 GB on a SAM 3D mesh.
            f"eval_params.batch_size={batch_size}",
        ]
        runpy.run_path(
            str(partsam / "evaluation" / "eval_everypart.py"), run_name="__main__"
        )
    finally:
        sys.argv = argv
        os.chdir(cwd)
    if "labels" not in store:
        raise RuntimeError("PartSAM produced no labels for the mesh")
    return store


def split_parts(store: dict, out: Path, stem: str) -> dict:
    """One mesh per part and parts.json. Crumbs under MIN_FACE_SHARE are listed, not kept."""
    import numpy as np
    import trimesh

    labels, vertices, faces = store["labels"], store["vertices"], store["faces"]
    total = len(faces)
    parts: list[dict] = []
    dropped: list[dict] = []
    (out / "parts").mkdir(parents=True, exist_ok=True)
    order = sorted(set(labels.tolist()), key=lambda lab: -(labels == lab).sum())
    for label in order:
        chosen = np.flatnonzero(labels == label)
        if len(chosen) < MIN_FACE_SHARE * total:
            dropped.append({"label": int(label), "faces": len(chosen)})
            continue
        part = trimesh.Trimesh(vertices=vertices, faces=faces[chosen], process=False)
        part.remove_unreferenced_vertices()
        name = f"part_{len(parts)}"
        path = out / "parts" / f"{name}.glb"
        part.export(path)
        low, high = part.bounds
        parts.append(
            {
                "name": name,
                "partsam_label": int(label),
                "faces": len(chosen),
                "face_share": round(len(chosen) / total, 4),
                "bbox_min": [float(v) for v in low],
                "bbox_max": [float(v) for v in high],
                "centroid": [float(v) for v in part.centroid],
                "mesh": str(path),
            }
        )
    record = {
        "source_mesh": stem,
        "tool": "PartSAM (github.com/czvvd/PartSAM) b16d3e8, stock configs/partsam.yaml",
        "faces": int(total),
        "min_face_share": MIN_FACE_SHARE,
        "parts": parts,
        "dropped": dropped,
    }
    (out / "parts.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--partsam", type=Path, default=PARTSAM)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    mesh, out = args.mesh.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    store = run_partsam(mesh, out, args.partsam.resolve(), args.batch_size)
    record = split_parts(store, out, mesh.name)
    print(
        f"{len(record['parts'])} parts, {len(record['dropped'])} crumbs dropped -> {out / 'parts.json'}"
    )
    return 0 if record["parts"] else 1


if __name__ == "__main__":
    sys.exit(main())
