# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SF-33: a fitted splat per bin object, packed per Gaussian as the prior input.

    python -m sceneagent.fitted_prior --out /workspace/sf4-smoke/bin

Sandeep's physics prior (github.com/sandy-t-shelvan/physics-prior-mlp) reads a
splat fitted to the object's own mesh, not the scene splat:

1. 16 orbit views (8 azimuths at elevations 18 and 50 degrees);
2. COLMAP;
3. splatfacto for 600 steps;
4. each Gaussian packed into one row of width 912.

The row is DINOv2 ViT-S/14 (384), then OpenCLIP ViT-B-16 with the OpenAI
weights (512), then 16 geometry columns.

This module runs that repository's own per-object path as it is, at the pinned
commit :data:`PRIOR_COMMIT`. It changes nothing in that repository. It:

- prepares each bin object's SAM 3D mesh;
- calls ``sceneagent.mlp.fitted_splat._convert_one`` in a separate interpreter;
- checks every ``prior_input.npy`` it writes with :func:`validate_pack`;
- records the commit, the encoder ids and weight hashes, N and the opacity
  spread per object.

Why a separate process: the prior repository's top-level package is also
named ``sceneagent``, so the two cannot share one interpreter. The driver runs
with only the prior repository on ``PYTHONPATH``.

Why the mesh is prepared: the prior samples colour from a UV texture
(``load_textured_mesh`` refuses a mesh without one). Our SAM 3D meshes are the
decoder's output with no texture baking, so they carry vertex colours.
:func:`ensure_uv_texture` gives such a mesh a per-face flat texture atlas
(ours, labelled in the evidence) and leaves a UV-textured mesh untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from sceneagent.bin_run import SCENE, SOURCE

ITEM = "fitted splat prior input"
REQUIRED = ("SAM 3D meshes", ITEM)

# The prior repository and the interpreter that runs it on the box.
PRIOR_REPO = "/workspace/src/physics-prior-mlp"
PRIOR_COMMIT = "b052009284d7794fc38fb8d158d5eb5c9fb1a7cb"
PRIOR_URL = "https://github.com/sandy-t-shelvan/physics-prior-mlp"
# nerfstudio 1.1.4 and gsplat 1.0.0 live here; the prior's fitted path also
# needs numba, timm and open_clip_torch in this env (see the PR for SF-33).
FIT_PYTHON = "/root/.local/share/mamba/envs/nerf/bin/python3"
COLMAP_BIN = "/opt/colmap-cuda/bin"
CUDA_HOME = "/usr/local/cuda-12.1"
# Encoder weights download once into our cache, never /root's.
HF_CACHE = "/workspace/sf4-smoke/hf-cache"
TORCH_CACHE = "/workspace/sf4-smoke/torch-cache"
WORKER_TIMEOUT_S = 6 * 3600

# The prior's input layout (README of the prior repository).
DINO_DIM = 384
CLIP_DIM = 512
GEOM_DIM = 16
PRIOR_WIDTH = DINO_DIM + CLIP_DIM + GEOM_DIM
# Geometry is x, y, z, three exp scales, sigmoid opacity, ...: opacity is the
# seventh geometry column.
OPACITY_COLUMN = DINO_DIM + CLIP_DIM + 6
MIN_OPACITY_STD = 1e-3
CONSTANT_OPACITY = 0.99
# The prior's own floor on kept Gaussians (fitted_splat.MIN_KEPT).
MIN_ROWS = 32
DINO_ID = "vit_small_patch14_dinov2.lvd142m"
CLIP_ID = "ViT-B-16 openai"
TEXTURE_SIDE_MAX = 1024  # the prior thumbnails a larger texture, which would blur the atlas
ATLAS_NOTE = "ours: per-face flat texture atlas from SAM 3D vertex colours"

Runner = Callable[..., Any]

# Runs in the prior repository's process, with that repository alone on
# PYTHONPATH. It loads the encoders once and calls the prior's own per-object
# conversion; one object failing is recorded and the rest still run.
DRIVER = """\
import hashlib, json, sys
from pathlib import Path

jobs = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
from sceneagent.mlp import fitted_splat as fs
from sceneagent.mlp.represent import CLIP_NAME, CLIP_PRETRAINED, DINO_NAME, FeatureEncoders

fs._require_tools()
encoders = FeatureEncoders().load()


def state_sha256(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode("utf-8"))
        digest.update(value.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


results = {
    "encoders": {
        "dino": {"id": DINO_NAME, "state_sha256": state_sha256(encoders._dino)},
        "clip": {"id": f"{CLIP_NAME} {CLIP_PRETRAINED}", "state_sha256": state_sha256(encoders._clip)},
    },
    "objects": [],
}
for job in jobs["objects"]:
    row = {
        "id": job["name"],
        "source": "bin",
        "mesh": Path(job["mesh"]),
        "directory": Path(job["directory"]),
        "work": Path(job["work"]),
    }
    try:
        summary = fs._convert_one(row, encoders)
        results["objects"].append({"object": job["object"], "ok": True, "summary": summary})
    except Exception as exc:
        fs._delete_tree(row["work"])
        results["objects"].append(
            {"object": job["object"], "ok": False, "error": f"{type(exc).__name__}: {exc}"[-600:]}
        )
    Path(jobs["results"]).write_text(json.dumps(results, indent=2) + "\\n", encoding="utf-8")
"""


class PackError(ValueError):
    """A prior input that is not the prior's (N, 912) contract."""


def validate_pack(array: Any) -> dict[str, float | int]:
    """Check one ``prior_input`` against the prior's contract; return N and the opacity spread.

    The rule is the prior's own:
    - rank 2, width 384 + 512 + 16, all finite, at least 32 rows;
    - opacity in [0, 1] whose standard deviation is at least 1e-3 and which
      is not the constant 0.99 a mesh-sampled splat gives.
    """
    packed = np.asarray(array, dtype=np.float64)
    if packed.ndim != 2:
        raise PackError(f"prior input must be rank 2, got shape {packed.shape}")
    if packed.shape[1] != PRIOR_WIDTH:
        raise PackError(
            f"prior input width must be {PRIOR_WIDTH} "
            f"({DINO_DIM} + {CLIP_DIM} + {GEOM_DIM}), got {packed.shape[1]}"
        )
    if packed.shape[0] < MIN_ROWS:
        raise PackError(f"only {packed.shape[0]} Gaussians; the prior needs {MIN_ROWS}")
    if not np.isfinite(packed).all():
        raise PackError("prior input has non-finite values")
    opacity = packed[:, OPACITY_COLUMN]
    if opacity.min() < 0.0 or opacity.max() > 1.0:
        raise PackError("opacity column is outside [0, 1]: not a sigmoid opacity")
    std = float(opacity.std())
    if std < MIN_OPACITY_STD:
        raise PackError(f"opacity is constant (std {std:.3g} < {MIN_OPACITY_STD})")
    if float(np.max(np.abs(opacity - CONSTANT_OPACITY))) < MIN_OPACITY_STD:
        raise PackError("opacity is the constant 0.99 of a mesh-sampled splat")
    return {
        "n": int(packed.shape[0]),
        "opacity_std": std,
        "opacity_mean": float(opacity.mean()),
    }


def meshes_from_sam3d(out: Path) -> list[tuple[int, Path]]:
    """``(object number, glb)`` for every object SAM 3D meshed, from its ``results.json``."""
    results = json.loads((out / "sam3d" / "results.json").read_text(encoding="utf-8"))
    meshes = []
    for record in results.get("objects", []):
        if record.get("ok") and record.get("glb"):
            meshes.append((int(record["object"]), Path(record["glb"])))
    return sorted(meshes)


def ensure_uv_texture(glb: Path, dest: Path) -> tuple[Path, str]:
    """A single-mesh glb the prior can sample colour from; return it and how it was made.

    A mesh with a UV texture is passed through unchanged. A mesh with vertex
    colours gets a texture atlas with one pixel per face, holding the mean of
    the face's vertex colours, and UVs at that pixel's centre. The prior's
    bilinear lookup then returns exactly that colour (ours: flat per face,
    finer than the 518-pixel views the encoders see).
    """
    import trimesh
    from PIL import Image

    loaded = trimesh.load(glb, force="scene", process=False)
    meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
    if len(meshes) != 1:
        raise ValueError(f"expected one mesh in {glb}, found {len(meshes)}")
    mesh = meshes[0]
    visual = mesh.visual
    uv = getattr(visual, "uv", None)
    material = getattr(visual, "material", None)
    image = None
    if material is not None:
        image = getattr(material, "image", None) or getattr(
            material, "baseColorTexture", None
        )
    if uv is not None and len(uv) == len(mesh.vertices) and image is not None:
        return glb, "SAM 3D UV texture, unchanged"
    try:
        colours = visual if getattr(visual, "kind", None) else visual.to_color()
        vertex_rgb = np.asarray(colours.vertex_colors)[:, :3]
    except Exception as exc:  # no colour of any kind
        raise ValueError(f"{glb} has neither a UV texture nor vertex colours") from exc
    if len(vertex_rgb) != len(mesh.vertices):
        raise ValueError(f"{glb} has neither a UV texture nor vertex colours")
    faces = np.asarray(mesh.faces)
    count = len(faces)
    side = max(2, math.ceil(math.sqrt(count)))
    if side > TEXTURE_SIDE_MAX:
        raise ValueError(
            f"{count} faces need a {side}-pixel atlas; the prior would resize "
            f"anything above {TEXTURE_SIDE_MAX} and blur it"
        )
    face_rgb = vertex_rgb[faces].astype(np.float64).mean(axis=1)
    atlas = np.zeros((side, side, 3), dtype=np.uint8)
    rows, cols = np.divmod(np.arange(count), side)
    atlas[rows, cols] = np.clip(np.rint(face_rgb), 0, 255).astype(np.uint8)
    # The prior samples x = u * (w - 1) and y = (1 - v) * (h - 1); these UVs
    # land on the pixel centres exactly, so no neighbour bleeds in.
    face_uv = np.stack(
        [cols / (side - 1), 1.0 - rows / (side - 1)], axis=1
    ).astype(np.float64)
    vertices = np.asarray(mesh.vertices)[faces].reshape(-1, 3)
    new_faces = np.arange(3 * count).reshape(count, 3)
    new_uv = np.repeat(face_uv, 3, axis=0)
    textured = trimesh.Trimesh(
        vertices=vertices,
        faces=new_faces,
        visual=trimesh.visual.TextureVisuals(
            uv=new_uv, image=Image.fromarray(atlas, mode="RGB")
        ),
        process=False,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    textured.export(dest)
    return dest, ATLAS_NOTE


def _prior_commit(repo: Path, runner: Runner) -> str:
    done = runner(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return (done.stdout or "").strip() if done.returncode == 0 else ""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_ledger(out: Path) -> dict[str, Any]:
    try:
        return json.loads((out / "ledger.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"entries": []}


def driver_env(base: dict[str, str] | None, prior_repo: Path) -> dict[str, str]:
    """The prior repository alone on PYTHONPATH; COLMAP and CUDA on PATH; our caches."""
    env = dict(os.environ if base is None else base)
    env["PYTHONPATH"] = str(prior_repo)
    env["PATH"] = os.pathsep.join(
        [str(Path(FIT_PYTHON).parent), COLMAP_BIN, f"{CUDA_HOME}/bin", env.get("PATH", "")]
    )
    env["CUDA_HOME"] = CUDA_HOME
    env["HF_HUB_CACHE"] = HF_CACHE
    env["TORCH_HOME"] = TORCH_CACHE
    env.pop("HF_HUB_OFFLINE", None)
    return env


def run_fitted_prior(
    out: Path | str,
    *,
    runner: Runner = subprocess.run,
    prior_repo: Path | str = PRIOR_REPO,
    python: str = FIT_PYTHON,
) -> int:
    """Fit and pack every meshed bin object; write the evidence and one ledger item."""
    out, prior_repo = Path(out), Path(prior_repo)
    ledger = _load_ledger(out)
    entries = [e for e in ledger.get("entries", []) if e.get("item") != ITEM]
    entry: dict[str, Any] = {"item": ITEM, "use_gpu": True}
    evidence: dict[str, Any] = {
        "scene": SCENE,
        "prior": {"url": PRIOR_URL, "pinned": PRIOR_COMMIT, "entry": "fitted_splat._convert_one"},
        "objects": [],
    }
    try:
        commit = _prior_commit(prior_repo, runner)
        evidence["prior"]["commit"] = commit
        if commit != PRIOR_COMMIT:
            raise RuntimeError(
                f"prior repository at {prior_repo} is at {commit or 'no commit'}, "
                f"not the pinned {PRIOR_COMMIT}"
            )
        root = out / "fitted"
        jobs: list[dict[str, Any]] = []
        prepared: dict[int, dict[str, Any]] = {}
        for number, glb in meshes_from_sam3d(out):
            name = f"object_{number}"
            directory = root / name
            mesh, note = ensure_uv_texture(glb, directory / "mesh_textured.glb")
            prepared[number] = {
                "object": number,
                "sam3d_glb": str(glb),
                "sam3d_glb_sha256": _sha256(glb),
                "mesh": str(mesh),
                "texture": note,
            }
            jobs.append(
                {
                    "object": number,
                    "name": name,
                    "mesh": str(mesh),
                    "directory": str(directory),
                    "work": str(out / "fitted_tmp" / name),
                }
            )
        if not jobs:
            raise RuntimeError("SAM 3D meshed no object")
        results_path = root / "results.json"
        jobs_path = root / "jobs.json"
        root.mkdir(parents=True, exist_ok=True)
        jobs_path.write_text(
            json.dumps({"objects": jobs, "results": str(results_path)}, indent=2) + "\n",
            encoding="utf-8",
        )
        results_path.unlink(missing_ok=True)
        argv = [python, "-c", DRIVER, str(jobs_path)]
        done = runner(
            argv,
            check=False,
            capture_output=True,
            text=True,
            # The prior's mesh loader appends debug lines to a hard-coded
            # Windows path, which on Linux is a file in the working directory.
            # Running from our own folder keeps that out of the prior checkout.
            cwd=str(root),
            env=driver_env(None, prior_repo),
            timeout=WORKER_TIMEOUT_S,
        )
        evidence["worker"] = {
            "returncode": done.returncode,
            "tail": ((done.stdout or "") + (done.stderr or ""))[-1500:],
        }
        if not results_path.is_file():
            raise RuntimeError("the prior's driver wrote no results.json")
        results = json.loads(results_path.read_text(encoding="utf-8"))
        evidence["encoders"] = results.get("encoders")
        for record in results.get("objects", []):
            number = int(record["object"])
            row = dict(prepared.get(number, {"object": number}))
            if not record.get("ok"):
                row.update(ok=False, error=record.get("error", "unknown"))
                evidence["objects"].append(row)
                continue
            summary = record.get("summary", {})
            pack_path = Path(summary.get("prior_input", root / f"object_{number}" / "prior_input.npy"))
            try:
                checked = validate_pack(np.load(pack_path))
            except (OSError, ValueError) as exc:
                row.update(ok=False, error=f"pack check: {exc}")
                evidence["objects"].append(row)
                continue
            row.update(
                ok=True,
                prior_input=str(pack_path),
                prior_input_sha256=_sha256(pack_path),
                ply=summary.get("ply"),
                gaussians_in_ply=summary.get("gaussians_in_ply"),
                orbit_views=summary.get("orbit_views"),
                splat_iterations=summary.get("splat_iterations"),
                **checked,
            )
            evidence["objects"].append(row)
        packed = [o for o in evidence["objects"] if o.get("ok")]
        failed = [o for o in evidence["objects"] if not o.get("ok")]
        entry.update(
            decision="accept" if packed and not failed else "reject",
            packed=len(packed),
            failed=len(failed),
            meshed=len(jobs),
            prior_commit=commit,
            encoders={k: v.get("id") for k, v in (results.get("encoders") or {}).items()},
        )
        if failed:
            entry["reason"] = "; ".join(
                f"object_{o['object']}: {o.get('error', '')}"[:200] for o in failed
            )[-800:]
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as exc:
        entry.update(decision="reject", reason=f"{type(exc).__name__}: {exc}"[-800:])
    entries.append(entry)
    ledger.update(scene_name=f"{SOURCE} {SCENE}", source=SOURCE, entries=entries)
    (out / "ledger.json").write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    (out / "fitted_prior_evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if entry["decision"] == "accept" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="the bin run directory")
    parser.add_argument("--prior-repo", type=Path, default=Path(PRIOR_REPO))
    args = parser.parse_args(argv)
    return run_fitted_prior(args.out, prior_repo=args.prior_repo)


if __name__ == "__main__":
    sys.exit(main())
