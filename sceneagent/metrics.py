"""Reconstruction and evaluation statistics named by SceneAgent.

Chamfer distance and F1 at 1 cm follow equations (4) and (5) in section 4.1.
Points are in meters. The paper's threshold tau is 1 cm, so the default tau
is 0.01 m. F1 is 0 when precision + recall is 0, and a missing object scores 0
when a scene is summarized the way GraspClutter6D is summarized in that section.

MessyKitchens uses a different summary. ``aggregate_messy_kitchens`` averages
each object's F1 and Chamfer over that object's sampling seeds, then averages
those object means with equal weight per object. An object the run omits is
named in the result and left out of the mean.

The Wilson interval is the 95% interval the project website prints on each
place-success cell (for example 1 of 10 trials is reported as 1.8% to 40.4%).
The paper's figure 14 reports the same cells as successes over ten trials.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

TAU_1CM_M = 0.01
WILSON_Z_95 = 1.959963984540054


def wilson_interval(successes, trials, z=WILSON_Z_95):
    successes = int(successes)
    trials = int(trials)
    if trials <= 0:
        raise ValueError("trials must be positive")
    if successes < 0 or successes > trials:
        raise ValueError("successes must lie in [0, trials]")
    phat = successes / trials
    z2 = z * z
    denom = 1.0 + z2 / trials
    center = (phat + z2 / (2.0 * trials)) / denom
    half = z * math.sqrt(phat * (1.0 - phat) / trials + z2 / (4.0 * trials * trials)) / denom
    return center - half, center + half


def _as_points(points):
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    return arr


def _min_distances(source, target):
    delta = source[:, None, :] - target[None, :, :]
    return np.linalg.norm(delta, axis=2).min(axis=1)


def symmetric_chamfer(points_x, points_y):
    """One-half the sum of the two directed mean nearest-point distances."""
    x = _as_points(points_x)
    y = _as_points(points_y)
    if len(x) == 0 or len(y) == 0:
        raise ValueError("Chamfer distance needs points on both clouds")
    return 0.5 * (float(_min_distances(x, y).mean()) + float(_min_distances(y, x).mean()))


def f1_at_threshold(points_x, points_y, tau=TAU_1CM_M):
    x = _as_points(points_x)
    y = _as_points(points_y)
    if len(x) == 0 or len(y) == 0:
        return 0.0
    precision = float((_min_distances(x, y) <= tau).mean())
    recall = float((_min_distances(y, x) <= tau).mean())
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _hungarian(cost):
    """Minimum-cost assignment. Exact enumeration; scenes here are small."""
    cost = np.asarray(cost, dtype=float)
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return []
    if n_rows > 8 or n_cols > 8:
        raise ValueError("phase-1 matcher supports at most 8 objects on a side")
    best = None
    best_cost = None
    columns = range(n_cols)
    for choice in itertools.permutations(columns, n_rows):
        total = float(sum(cost[i, choice[i]] for i in range(n_rows)))
        if best_cost is None or total < best_cost:
            best_cost = total
            best = [(i, choice[i]) for i in range(n_rows)]
    return best


def score_reconstruction(delivered, reference, tau=TAU_1CM_M):
    """Match delivered clouds to reference clouds by Chamfer cost.

    ``delivered`` and ``reference`` map a name to an (n, 3) cloud in meters.
    F1 is averaged over every reference object, with unmatched objects at 0.
    Chamfer is averaged over matched objects only, matching section 4.1's
    GraspClutter6D summary. This function does not load that dataset.
    """
    ref_names = list(reference)
    del_names = list(delivered)
    if not ref_names:
        raise ValueError("reference is empty")
    f1_by_ref = {name: 0.0 for name in ref_names}
    if not del_names:
        return {
            "f1_mean": 0.0,
            "chamfer_mean_matched_m": None,
            "matches": [],
            "f1_by_reference": f1_by_ref,
            "tau_m": tau,
        }
    cost = np.zeros((len(ref_names), len(del_names)), dtype=float)
    for i, ref_name in enumerate(ref_names):
        for j, del_name in enumerate(del_names):
            cost[i, j] = symmetric_chamfer(reference[ref_name], delivered[del_name])
    pairs = _hungarian(cost)
    matches = []
    chamfers = []
    for i, j in pairs:
        ref_name = ref_names[i]
        del_name = del_names[j]
        chamfer = float(cost[i, j])
        f1 = f1_at_threshold(delivered[del_name], reference[ref_name], tau=tau)
        f1_by_ref[ref_name] = f1
        chamfers.append(chamfer)
        matches.append({
            "reference": ref_name,
            "delivered": del_name,
            "chamfer_m": chamfer,
            "f1": f1,
        })
    return {
        "f1_mean": float(np.mean([f1_by_ref[name] for name in ref_names])),
        "chamfer_mean_matched_m": float(np.mean(chamfers)) if chamfers else None,
        "matches": matches,
        "f1_by_reference": f1_by_ref,
        "tau_m": tau,
    }


# Section 4.1 samples each MessyKitchens object with three seeds. A rectangular
# table makes "seeds then objects" identical to the other reductions, so the
# order is observable only when objects list different numbers of seeds.
MESSY_KITCHENS_SEEDS = (0, 1, 2)


class MissingSeedError(ValueError):
    """An object's seed measurements are incomplete, so no scene mean is returned."""

    def __init__(self, object_name, missing_seeds, detail=None):
        self.object_name = object_name
        self.missing_seeds = tuple(missing_seeds)
        if self.missing_seeds:
            listed = ", ".join(str(seed) for seed in self.missing_seeds)
            message = f"object {object_name!r} is missing seed {listed}"
        else:
            message = f"object {object_name!r} has no seed measurements"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(f"{message}; the remaining seeds were not averaged")


def _object_names(names):
    found = []
    seen = set()
    for name in names:
        if not isinstance(name, str) or name == "":
            raise ValueError("object names must be non-empty strings")
        if name in seen:
            raise ValueError(f"duplicate object name {name!r}")
        seen.add(name)
        found.append(name)
    return found


def _seed_sort_key(seed):
    if isinstance(seed, bool) or not isinstance(seed, (int, float)):
        return (1, str(seed))
    return (0, seed)


def _required_seeds(expected_seeds):
    """Return the seed ids every object must contain, or None to take each list as complete."""
    if expected_seeds is None:
        return None
    found = []
    seen = set()
    for seed in expected_seeds:
        if isinstance(seed, bool) or not isinstance(seed, (int, float, str)):
            raise ValueError(f"seed id {seed!r} is not an int or str")
        if seed in seen:
            raise ValueError(f"duplicate expected seed {seed!r}")
        seen.add(seed)
        found.append(seed)
    return tuple(found)


def _finite_metric(object_name, seed, record):
    """Return ``(f1, chamfer_m)`` or raise when this seed cannot be averaged."""
    if not isinstance(record, dict) or "f1" not in record or "chamfer_m" not in record:
        raise MissingSeedError(object_name, (seed,), detail="f1 or chamfer_m is absent")
    f1 = record["f1"]
    chamfer = record["chamfer_m"]
    if f1 is None or chamfer is None:
        raise MissingSeedError(object_name, (seed,), detail="f1 or chamfer_m is absent")
    try:
        f1_value = float(f1)
        chamfer_value = float(chamfer)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"object {object_name!r} seed {seed} has non-numeric f1 or chamfer_m"
        ) from exc
    if not np.isfinite(f1_value) or not np.isfinite(chamfer_value):
        raise MissingSeedError(
            object_name,
            (seed,),
            detail="f1 or chamfer_m is not finite",
        )
    return f1_value, chamfer_value


def _scene_summary(scored, omitted, full_scene, expected_given, per_object):
    scored_text = ", ".join(scored)
    if full_scene:
        return (
            f"Full MessyKitchens scene score over {scored_text}. "
            "Each object was averaged over its 3 seeds, then objects were "
            "averaged with equal weight."
        )
    sentences = [f"Subset score over {scored_text}."]
    if not expected_given:
        sentences.append("No expected object list was provided.")
    elif omitted:
        sentences.append("Omitted: " + ", ".join(omitted) + ".")
    uneven = []
    for name in scored:
        stats = per_object[name]
        if stats["n_seeds"] != len(MESSY_KITCHENS_SEEDS):
            seed_text = ", ".join(str(seed) for seed in stats["seeds"])
            uneven.append(f"{name} seeds {seed_text}")
    if uneven:
        sentences.append(
            "Seed measurements are not the three-seed protocol: " + "; ".join(uneven) + "."
        )
    sentences.append(
        "This is not the full-scene MessyKitchens score and not the GraspClutter6D bin score."
    )
    return " ".join(sentences)


def aggregate_messy_kitchens(
    per_object,
    expected_objects=None,
    expected_seeds=MESSY_KITCHENS_SEEDS,
):
    """Average MessyKitchens F1 and Chamfer over seeds, then over objects.

    ``per_object`` maps an object name to a mapping of seed id to a record
    with ``f1`` and ``chamfer_m`` (meters). Each object's seeds are averaged
    first. Those object means are then averaged with equal weight per object.
    ``score_reconstruction`` is the GraspClutter6D summary (unmatched F1 at 0,
    Chamfer over matched objects only) and is not called here.

    ``expected_objects`` is the full scene. ``full_scene`` is true only when
    every expected name was scored and every scored object has exactly three
    finite seed measurements. Otherwise the result is a subset score:
    ``summary`` names ``scored_objects`` and, when the expected list was
    provided, names ``omitted_objects``. Omitted objects are left out of the
    mean.

    ``expected_seeds`` defaults to ``(0, 1, 2)``. A scored object that lacks
    one of those ids, or whose record lacks a finite ``f1`` and ``chamfer_m``,
    raises ``MissingSeedError``. The seeds that are present are not averaged
    in its place, and the other objects are not returned as a scene score.
    Pass ``expected_seeds=None`` only when the ids each object lists are its
    complete set. Counts may then differ, which is when this reduction
    disagrees with a mean of every seed measurement and with a mean taken
    across objects inside each seed and then across seeds. An explicit hole
    (a seed whose metrics are missing or not finite) still raises.

    An empty ``per_object`` raises ``ValueError``. This function does not
    load MessyKitchens.
    """
    if not isinstance(per_object, dict):
        raise ValueError("per_object must map object names to seed records")
    if len(per_object) == 0:
        if expected_objects is None:
            raise ValueError("empty input: no objects were scored")
        omitted = _object_names(expected_objects)
        message = "empty input: no objects were scored"
        if omitted:
            message += ". Omitted: " + ", ".join(omitted)
        raise ValueError(message)

    expected = None
    if expected_objects is not None:
        expected = _object_names(expected_objects)
        unexpected = sorted(set(per_object) - set(expected))
        if unexpected:
            raise ValueError(
                "scored objects not in the expected scene: " + ", ".join(unexpected)
            )

    required = _required_seeds(expected_seeds)
    per_object_stats = {}
    object_f1 = []
    object_chamfer = []
    for name in sorted(per_object):
        if not isinstance(name, str) or name == "":
            raise ValueError("object names must be non-empty strings")
        seed_map = per_object[name]
        if not isinstance(seed_map, dict):
            raise ValueError(f"object {name!r} must map seed ids to metric records")
        present = set(seed_map)
        if required is None:
            if not present:
                raise MissingSeedError(name, ())
            seed_ids = sorted(present, key=_seed_sort_key)
        else:
            missing = tuple(seed for seed in required if seed not in present)
            if missing:
                raise MissingSeedError(name, missing)
            extra = sorted(present - set(required), key=_seed_sort_key)
            if extra:
                listed = ", ".join(str(seed) for seed in extra)
                raise ValueError(
                    f"object {name!r} has unexpected seeds {listed}; "
                    "those seeds were not dropped and were not averaged"
                )
            seed_ids = list(required)
        f1s = []
        chamfers = []
        for seed in seed_ids:
            f1_value, chamfer_value = _finite_metric(name, seed, seed_map[seed])
            f1s.append(f1_value)
            chamfers.append(chamfer_value)
        f1_mean = float(np.mean(f1s))
        chamfer_mean = float(np.mean(chamfers))
        per_object_stats[name] = {
            "f1_mean": f1_mean,
            "chamfer_mean_m": chamfer_mean,
            "seeds": list(seed_ids),
            "n_seeds": len(seed_ids),
        }
        object_f1.append(f1_mean)
        object_chamfer.append(chamfer_mean)

    scored = list(per_object_stats)
    if expected is None:
        omitted = None
        objects_complete = False
    else:
        omitted = sorted(name for name in expected if name not in per_object_stats)
        objects_complete = len(omitted) == 0
    n_protocol_seeds = len(MESSY_KITCHENS_SEEDS)
    protocol_seeds = all(
        stats["n_seeds"] == n_protocol_seeds for stats in per_object_stats.values()
    )
    full_scene = bool(objects_complete and protocol_seeds)
    summary = _scene_summary(scored, omitted, full_scene, expected is not None, per_object_stats)
    return {
        "protocol": "messy_kitchens",
        "aggregation": "seeds_then_objects",
        "full_scene": full_scene,
        "protocol_seeds_complete": protocol_seeds,
        "objects_complete": objects_complete,
        "reported_as": "messy_kitchens_full_scene" if full_scene else "messy_kitchens_subset",
        "summary": summary,
        "scored_objects": scored,
        "omitted_objects": omitted,
        "expected_objects": None if expected is None else list(expected),
        "f1_mean": float(np.mean(object_f1)),
        "chamfer_mean_m": float(np.mean(object_chamfer)),
        "per_object": per_object_stats,
        "expected_seeds": None if required is None else list(required),
    }
