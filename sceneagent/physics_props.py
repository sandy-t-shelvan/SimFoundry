# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SF-34: the full §3.4 property set per bin object, one ``physics.json`` each.

    python -m sceneagent.physics_props --objects OBJECTS.json --packs PACKS/ \
        --out OUT/ [--checkpoint PRIOR.npz] [--vote material_vote.json] \
        [--review material_review.json] [--vlm-physics vlm_physics.json]

Every field records where it came from (``sources``):

- **material:** equation (2) over the prior mean, the VLM and the keyword
  proposal (SF-20's ``material_vote.json``), then the VLM audit (SF-21's
  ``material_review.json``) when present. Without a vote file the vote is cast
  here over the prior and the keyword proposal only, and says so.
- **body type, collider:** the prior's heads on the in-mask mean. A tie
  between the top logits gives no class, as a material tie does.
- **log mass, log density, rigidity, deformability:** the prior's mean.
  Sigmoid is applied to rigidity and deformability after the mean, never
  before; the 912 inputs are never averaged (Sandeep's note, §2).
- **mass, density in kg:** ``exp`` of the log slots (the prior is trained on
  natural logs of kilograms, ``mlp.labels``), so the USD export can write them.
- **friction:** the VLM estimate (paper §3.4), else left to PhysX and recorded
  as ``default``. **Restitution:** PhysX default, recorded as ``default``.
- **Young's modulus, Poisson's ratio, render material:** null. The paper looks
  them up from a table it does not publish; nothing is invented.
- **route:** PhysGaussian when deformability is >= 0.65 (that rule wins, paper
  §3.4) or the body is soft deformable, cloth shell or fluid/granular; PhysX
  for dynamic rigid, static rigid or kinematic bodies.

Ours, not the paper's (named so they can be found):

- A collider a PhysX rigid body cannot use (``tetrahedral mesh``, ``cloth
  mesh``) on a PhysX-routed object is not passed to the export; it is kept as
  ``prior_collision_primitive`` and the export's default applies.
- One VLM friction coefficient is written as both static and dynamic friction.

The prior is Sandeep's checkpoint used as it is (``weights: "ours"``, not
Harvard's), loaded through ``sceneagent.mlp.prior.PhysicsPrior``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sceneagent.mlp.prior import (
    OUTPUT_DIM,
    SLOT_BODY,
    SLOT_COLLISION,
    SLOT_DEFORMABILITY,
    SLOT_LOG_DENSITY,
    SLOT_LOG_MASS,
    SLOT_MATERIAL,
    SLOT_RIGIDITY,
    PhysicsPrior,
)
from sceneagent.physics import (
    SOFT_BODY_DEFORMABILITY,
    UNPUBLISHED_LOOKUPS,
    resolve_body_type,
    vote_material,
)
from sceneagent.taxonomy import (
    BODY_TYPES,
    COLLISION_PRIMITIVES,
    MATERIAL_CLASSES,
    keyword_material,
    require_material,
)

PHYSX_BODY_TYPES = ("dynamic rigid", "static rigid", "kinematic")
PHYSGAUSSIAN_BODY_TYPES = ("soft deformable", "cloth shell", "fluid or granular")
# Colliders a PhysX rigid body can take (physics_export.APPROXIMATION's keys).
PHYSX_RIGID_COLLIDERS = (
    "box",
    "sphere",
    "cylinder",
    "capsule",
    "convex hull",
    "triangle mesh",
)
DEFAULT = "default"
TIE_TOL = 1e-9


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, read in blocks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.asarray(logits, dtype=np.float64) - float(np.max(logits))
    exponential = np.exp(shifted)
    return exponential / float(exponential.sum())


def head_class(logits: np.ndarray, classes: Sequence[str]) -> str | None:
    """The top class of one head, or ``None`` when two logits share the top."""
    values = np.asarray(logits, dtype=np.float64)
    top = float(values.max())
    winners = np.flatnonzero(np.isclose(values, top, rtol=0.0, atol=TIE_TOL))
    return None if winners.size != 1 else classes[int(winners[0])]


def decode_prior_mean(mean: np.ndarray) -> dict[str, Any]:
    """Read the prior's heads from one in-mask mean of the 33 outputs.

    Slots 28-32 are masked (not decoded). Sigmoid is applied after the mean.
    """
    mean = np.asarray(mean, dtype=np.float64)
    if mean.shape != (OUTPUT_DIM,):
        raise ValueError(f"in-mask mean must have shape ({OUTPUT_DIM},)")
    if not np.isfinite(mean[:28]).all():
        raise ValueError("named prior outputs are not finite")
    material = head_class(mean[SLOT_MATERIAL], MATERIAL_CLASSES)
    probabilities = _softmax(mean[SLOT_MATERIAL])
    weight = (
        float(probabilities[MATERIAL_CLASSES.index(material)])
        if material is not None
        else None
    )
    return {
        "material": material,
        "material_weight": weight,
        "body_type": head_class(mean[SLOT_BODY], BODY_TYPES),
        "collision": head_class(mean[SLOT_COLLISION], COLLISION_PRIMITIVES),
        "log_mass": float(mean[SLOT_LOG_MASS]),
        "log_density": float(mean[SLOT_LOG_DENSITY]),
        "rigidity": _sigmoid(float(mean[SLOT_RIGIDITY])),
        "deformability": _sigmoid(float(mean[SLOT_DEFORMABILITY])),
    }


def prior_for_pack(
    prior: PhysicsPrior, features: Any, mask: Any | None = None
) -> dict[str, Any]:
    """Run the prior on every in-mask Gaussian of one ``(N, 912)`` pack."""
    n = int(features.shape[0])
    selected = np.ones(n, dtype=bool) if mask is None else np.asarray(mask, bool)
    mean = prior.in_mask_mean(features, selected)
    decoded = decode_prior_mean(mean)
    decoded["gaussians"] = int(np.count_nonzero(selected))
    return decoded


def simulator_route(body_type: str | None, deformability: float | None) -> str | None:
    """``physgaussian`` or ``physx``; the deformability rule wins (paper §3.4)."""
    if deformability is not None and deformability >= SOFT_BODY_DEFORMABILITY:
        return "physgaussian"
    if body_type in PHYSGAUSSIAN_BODY_TYPES:
        return "physgaussian"
    if body_type in PHYSX_BODY_TYPES:
        return "physx"
    return None


def _vlm_friction(vlm_physics: Mapping[str, Any] | None) -> float | None:
    if not vlm_physics:
        return None
    value = vlm_physics.get("friction")
    if value is None or isinstance(value, bool):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"VLM friction {value!r} is not a non-negative number")
    return number


def property_record(
    name: str,
    *,
    prior: Mapping[str, Any] | None = None,
    vote: Mapping[str, Any] | None = None,
    review: Mapping[str, Any] | None = None,
    vlm_physics: Mapping[str, Any] | None = None,
    prior_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One object's ``physics.json``: every §3.4 field and its source.

    ``prior`` is :func:`prior_for_pack`'s output (``None`` without an npz),
    ``vote`` one entry of ``material_vote.json`` (SF-20), ``review`` one entry
    of ``material_review.json`` (SF-21), ``vlm_physics`` a VLM estimate with a
    ``friction`` key (``section34`` format).
    """
    sources: dict[str, str] = {}
    record: dict[str, Any] = {"object": name, "section": "3.4"}

    # Material: equation (2), then the audit.
    if vote is not None:
        provisional = vote.get("material_class")
        record["material_vote"] = {
            "material_class": provisional,
            "scores": dict(vote.get("scores") or {}),
            "sources": list(vote.get("sources") or []),
        }
        sources["material_class"] = "equation (2) vote (SF-20)"
    else:
        proposals = []
        if prior is not None and prior.get("material") is not None:
            proposals.append(
                {
                    "source": "physics_prior",
                    "class": prior["material"],
                    "weight": prior["material_weight"],
                }
            )
        keyword = keyword_material(name)
        proposals.append(
            {
                "source": "keyword",
                "class": keyword["class"],
                "weight": keyword["weight"],
            }
        )
        provisional, scores = vote_material(proposals)
        record["material_vote"] = {
            "material_class": provisional,
            "scores": {k: float(v) for k, v in scores.items()},
            "sources": proposals,
            "note": "no VLM vote file; prior and keyword only",
        }
        sources["material_class"] = "equation (2) vote, prior and keyword only"
    final = provisional
    if review is not None:
        final = review.get("final_class", provisional)
        record["material_review"] = {
            key: review.get(key)
            for key in (
                "provisional_class",
                "audit_class",
                "decision",
                "reason",
                "signed_by",
            )
        }
        if review.get("decision") in ("confirm", "replace"):
            sources["material_class"] = "VLM audit (SF-21)"
    record["material_class"] = None if final is None else require_material(final)
    if final is None:
        sources["material_class"] += "; tie, no class"

    # Body, collider and numbers from the prior.
    deformability = None if prior is None else prior.get("deformability")
    explicit_body = None if prior is None else prior.get("body_type")
    body, body_rule = resolve_body_type(deformability, explicit_body)
    record["body_type"] = body
    sources["body_type"] = {
        "deformability_at_least_0.65": "deformability >= 0.65 (paper §3.4)",
        "explicit_proposal": "prior body head",
        "unresolved": "no prior" if prior is None else "prior body head tie",
    }[body_rule]
    route = simulator_route(body, deformability)
    record["simulator"] = route
    collision = None if prior is None else prior.get("collision")
    record["collision_primitive"] = collision
    sources["collision_primitive"] = (
        "no prior"
        if prior is None
        else ("prior collision head" if collision else "prior collision head tie")
    )
    if (
        route == "physx"
        and collision is not None
        and collision not in PHYSX_RIGID_COLLIDERS
    ):
        # Ours: the export cannot give a PhysX rigid body this collider.
        record["prior_collision_primitive"] = collision
        record["collision_primitive"] = None
        sources["collision_primitive"] = (
            f"prior proposed {collision!r}, not a PhysX rigid collider; export default"
        )
    for key in ("log_mass", "log_density", "rigidity", "deformability"):
        record[key] = None if prior is None else float(prior[key])
        sources[key] = "no prior" if prior is None else "prior mean"
    record["mass_kg"] = None if prior is None else math.exp(prior["log_mass"])
    record["density_kg_m3"] = None if prior is None else math.exp(prior["log_density"])
    for key in ("mass_kg", "density_kg_m3"):
        sources[key] = "no prior" if prior is None else "exp of the prior's log slot"

    # Contact coefficients.
    friction = _vlm_friction(vlm_physics)
    record["friction_static"] = friction
    record["friction_dynamic"] = friction
    for key in ("friction_static", "friction_dynamic"):
        sources[key] = "VLM estimate" if friction is not None else DEFAULT
    record["restitution"] = None
    sources["restitution"] = DEFAULT

    # Unpublished class lookups stay null.
    for key in UNPUBLISHED_LOOKUPS:
        record[key] = None
        sources[key] = "unpublished class lookup; not invented"

    record["soft_body_threshold"] = SOFT_BODY_DEFORMABILITY
    record["sources"] = sources
    if prior is not None:
        record["prior_gaussians"] = prior.get("gaussians")
    if prior_provenance:
        record["prior"] = dict(prior_provenance)
    return record


def compare_catalogue_masses(
    records: Sequence[Mapping[str, Any]], catalogue_kg: Mapping[str, float]
) -> list[dict[str, Any]]:
    """Predicted mass against a catalogue (YCB) mass, as a check only.

    Never an input: the record is not changed. The prior's fitted set has no
    YCB objects, so the gap is reported, not corrected.
    """
    rows = []
    for record in records:
        name = str(record["object"])
        known = catalogue_kg.get(name)
        predicted = record.get("mass_kg")
        rows.append(
            {
                "object": name,
                "predicted_kg": predicted,
                "catalogue_kg": known,
                "ratio": (
                    None
                    if known in (None, 0) or predicted is None
                    else float(predicted) / float(known)
                ),
            }
        )
    return rows


def _by_position(
    entries: Sequence[Mapping[str, Any]] | None,
    objects: Sequence[Mapping[str, Any]],
    key: str,
    label: str,
) -> list[Mapping[str, Any] | None]:
    """Match a vote or review file to the objects by position, checking names.

    Two objects can share a name (two apples), so position is the key, as in
    ``material_runner``.
    """
    if entries is None:
        return [None] * len(objects)
    if len(entries) != len(objects):
        raise ValueError(
            f"{label} has {len(entries)} entries for {len(objects)} objects"
        )
    for entry, obj in zip(entries, objects, strict=True):
        if str(entry.get(key)) != str(obj["name"]):
            raise ValueError(f"{label} entry {entry.get(key)!r} is not {obj['name']!r}")
    return list(entries)


def write_property_set(
    objects: Sequence[Mapping[str, Any]],
    packs: Path | None,
    out: Path,
    *,
    checkpoint: Path | None = None,
    vote: Mapping[str, Any] | None = None,
    review: Mapping[str, Any] | None = None,
    vlm_physics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``out/physics/<n>.json`` for every object and a summary.

    ``objects`` is ``[{"object": n, "name": "..."}]`` (``crops.zip``'s
    ``objects.json``). With a checkpoint, each object needs
    ``packs/object_<n>.npy`` of shape ``(N, 912)`` and may have
    ``object_<n>_mask.npy``.
    """
    model = None
    provenance = None
    if checkpoint is not None and Path(checkpoint).is_file():
        model = PhysicsPrior.load(checkpoint)
        provenance = {
            "checkpoint": str(checkpoint),
            "sha256": file_sha256(Path(checkpoint)),
            "weights": "ours",
        }
    votes = _by_position(
        None if vote is None else vote.get("votes"), objects, "object", "material_vote"
    )
    reviews = _by_position(
        None if review is None else review.get("reviews"),
        objects,
        "name",
        "material_review",
    )
    estimates = (vlm_physics or {}).get("objects", {})
    folder = Path(out) / "physics"
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for obj, vote_entry, review_entry in zip(objects, votes, reviews, strict=True):
        number = int(obj["object"])
        prior = None
        if model is not None:
            if packs is None:
                raise ValueError("a prior checkpoint needs --packs")
            pack_path = Path(packs) / f"object_{number}.npy"
            features = np.load(pack_path, mmap_mode="r")
            mask_path = Path(packs) / f"object_{number}_mask.npy"
            mask = np.load(mask_path) if mask_path.is_file() else None
            prior = prior_for_pack(model, features, mask)
        record = property_record(
            str(obj["name"]),
            prior=prior,
            vote=vote_entry,
            review=review_entry,
            vlm_physics=estimates.get(str(number)),
            prior_provenance=provenance,
        )
        record["object_number"] = number
        (folder / f"object_{number}.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        records.append(record)
    summary = {
        "section": "3.4",
        "step": "property_set",
        "objects": len(records),
        "prior": provenance,
        "physgaussian": [
            r["object"] for r in records if r["simulator"] == "physgaussian"
        ],
        "unresolved_material": [
            r["object"] for r in records if r["material_class"] is None
        ],
    }
    (Path(out) / "physics_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return {"records": records, "summary": summary}


PROPERTIES_ITEM = "§3.4 properties"


def object_names(association: Mapping[str, Any]) -> dict[int, str]:
    """Each association object's most frequent GroundingDINO phrase, its name here."""
    names = {}
    for obj in association.get("object_list", []):
        counts: dict[str, int] = {}
        for mask in obj.get("masks", []):
            phrase = str(mask.get("phrase") or "object")
            counts[phrase] = counts.get(phrase, 0) + 1
        names[int(obj["object"])] = (
            max(sorted(counts), key=lambda p: counts[p]) if counts else "object"
        )
    return names


def _prior_inputs(
    out: Path, checkpoint: Path | None
) -> tuple[Path | None, Path | None]:
    """Packs and the pinned checkpoint when both are present, else option B.

    A checkpoint file whose SHA-256 is not the pinned prior is a reject, not
    a silent fallback.
    """
    from sceneagent.fitted_prior import (
        CHECKPOINT_RELATIVE,
        PRIOR_REPO,
        checked_checkpoint,
    )

    packs = out / "prior_input"
    if not any(packs.glob("object_*.npy")):
        return None, None
    path = Path(checkpoint) if checkpoint else Path(PRIOR_REPO) / CHECKPOINT_RELATIVE
    if not path.is_file():
        return None, None
    return packs, checked_checkpoint(path)


def run_physics(out: Path | str, *, checkpoint: Path | None = None) -> int:
    """Write ``physics/object_<n>.json`` for every placed object; one ledger item.

    The MVP path (Sandeep, 2026-11-13, option B): no prior checkpoint, so the
    material is equation (2) over the object's own phrase (``keyword_material``)
    and the other §3.4 fields are null with their source, which the export
    leaves to its defaults. The prior joins once SAM 3D meshes are textured and
    KAN-39 packs them (``--packs``, ``--checkpoint``).
    """
    out = Path(out)
    ledger_path = out / "ledger.json"
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        ledger = {"entries": []}
    entries = [e for e in ledger.get("entries", []) if e.get("item") != PROPERTIES_ITEM]
    entry: dict[str, Any] = {"item": PROPERTIES_ITEM}
    try:
        association = json.loads(
            (out / "association_evidence.json").read_text(encoding="utf-8")
        )
        placement = json.loads((out / "bin_score.json").read_text(encoding="utf-8"))
        names = object_names(association)
        objects = [
            {"object": int(r["object"]), "name": names.get(int(r["object"]), "object")}
            for r in placement["icp"]["objects"]
            if r.get("placed")
        ]
        if not objects:
            raise ValueError("no placed object")
        packs, used = _prior_inputs(out, checkpoint)
        if used is not None and packs is not None:
            objects = [
                obj
                for obj in objects
                if (packs / f"object_{obj['object']}.npy").is_file()
            ]
            if not objects:
                raise ValueError("prior packs match no placed object")
        result = write_property_set(objects, packs, out, checkpoint=used)
        entry.update(
            decision="accept",
            use_gpu=False,
            objects=len(objects),
            prior=result["summary"]["prior"] is not None,
            materials={
                f"object_{r['object_number']}": r["material_class"]
                for r in result["records"]
            },
        )
    except (OSError, ValueError, KeyError) as exc:
        entry.update(decision="reject", reason=f"{type(exc).__name__}: {exc}"[:800])
    entries.append(entry)
    ledger["entries"] = entries
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    return 0 if entry["decision"] == "accept" else 1


def _read_json(path: Path | None) -> Any:
    if path is None:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--objects", type=Path, required=True)
    parser.add_argument("--packs", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--vote", type=Path, default=None)
    parser.add_argument("--review", type=Path, default=None)
    parser.add_argument(
        "--vlm-physics",
        type=Path,
        default=None,
        help='{"objects": {"<n>": {"friction": ...}}}',
    )
    args = parser.parse_args(argv)
    objects = _read_json(args.objects)
    if not isinstance(objects, list) or not objects:
        parser.error("--objects must be a non-empty list")
    result = write_property_set(
        objects,
        args.packs,
        args.out,
        checkpoint=args.checkpoint,
        vote=_read_json(args.vote),
        review=_read_json(args.review),
        vlm_physics=_read_json(args.vlm_physics),
    )
    print(json.dumps(result["summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
