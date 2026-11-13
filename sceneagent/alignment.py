"""Alignment acceptance interface from SceneAgent section 3.1.

A candidate is accepted only when all four residuals pass fixed thresholds:
normalized median distance, 90th-percentile distance, coverage, and silhouette
agreement. The paper does not publish the four numbers. Calling
``accept_alignment`` without thresholds leaves the candidate undecided.

``ours_iteration1_thresholds`` is a separate rule this tree can accept or
reject with. It is not the paper's unpublished cutoff.
"""

# SceneAgent did not publish these cutoffs. These are our iteration-1 gates.
OURS_NORMALIZED_MEDIAN_MAX = 0.05
OURS_COVERAGE_MIN = 0.5
OURS_P90_MAX = 0.25
OURS_SILHOUETTE_MIN = 0.5
OURS_GATE_LABEL = "ours_iteration_1"
OURS_GATE_NOTE = (
    "SceneAgent did not publish these cutoffs. These are our iteration-1 gates."
)

# Held-out profile. Chosen on the derivation half of the checked-in bin
# scores (scripts/derive_heldout_thresholds.py). Not the iteration-1 numbers,
# and not a published paper cutoff.
HELDOUT_GATE_LABEL = "ours_heldout_v1"
HELDOUT_NORMALIZED_MEDIAN_MAX = 0.0784567069559569
HELDOUT_P90_MAX = 0.15229201952511362
HELDOUT_COVERAGE_MIN = 0.4732394366197183
HELDOUT_SILHOUETTE_MIN = 0.08971836478763888
HELDOUT_GATE_NOTE = "ours, not published in the paper."

# Ours, iteration 2 (2026-11-12, Sandeep's yes): the placement gate the bin
# stages use. On the bin's 14 placed objects (run 43315660864) silhouette
# tracked ground-truth F1 with Spearman +0.85 and p90 with -0.79, while
# normalized median (-0.19) and coverage (-0.13, the wrong sign) carried no
# signal: coverage counts Gaussians a mesh sample lands next to, so it
# follows how dense an object's Gaussian set is, and 5,000 samples cap it
# below 0.5 for any set over 7,000. Iteration 1 accepted 1 object (9th of 14
# by F1) and rejected the three best (Chamfer 5 mm). The silhouette minimum
# is iteration 1's; the p90 maximum is iteration 1's 0.25 splat units at the
# bin's 0.20 m per unit, in metres so it holds for a video's own unit. Both
# residuals the gate drops are still measured and recorded.
PLACEMENT_GATE_LABEL = "ours_iteration_2"
PLACEMENT_P90_MAX_M = 0.05
PLACEMENT_SILHOUETTE_MIN = OURS_SILHOUETTE_MIN
PLACEMENT_GATE_NOTE = (
    "Ours, not the paper's: silhouette >= 0.5 and p90 <= 0.05 m. Normalized "
    "median and coverage are recorded only; on the bin they did not track "
    "ground truth."
)

RESIDUAL_NAMES = ("normalized_median", "p90", "coverage", "silhouette")
DISTANCE_RESIDUALS = ("normalized_median", "p90")
AGREEMENT_RESIDUALS = ("coverage", "silhouette")

# Fruits ICP residual recorded on Quebec. Coverage and silhouette fail the
# iteration-1 gate. Do not lower the cutoffs to accept this residual.
FRUITS_ICP_RESIDUAL = {
    "normalized_median": 0.033103985722272695,
    "p90": 0.21563629180764,
    "coverage": 0.14616666666666667,
    "silhouette": None,
    "source": "/workspace/fruits-features/sam3d/icp_residual.json",
}


def ours_iteration1_thresholds():
    """Thresholds this tree will iterate. Not a published SceneAgent number."""
    return {
        "label": OURS_GATE_LABEL,
        "paper_published": False,
        "note": OURS_GATE_NOTE,
        "normalized_median": {"max": OURS_NORMALIZED_MEDIAN_MAX},
        "p90": {"max": OURS_P90_MAX},
        "coverage": {"min": OURS_COVERAGE_MIN},
        "silhouette": {"min": OURS_SILHOUETTE_MIN},
    }


def ours_heldout_v1_thresholds():
    """Cutoffs fit on a held-out split. Not a published SceneAgent number."""
    return {
        "label": HELDOUT_GATE_LABEL,
        "paper_published": False,
        "note": HELDOUT_GATE_NOTE,
        "normalized_median": {"max": HELDOUT_NORMALIZED_MEDIAN_MAX},
        "p90": {"max": HELDOUT_P90_MAX},
        "coverage": {"min": HELDOUT_COVERAGE_MIN},
        "silhouette": {"min": HELDOUT_SILHOUETTE_MIN},
    }


def _profile_thresholds(profile):
    if profile == OURS_GATE_LABEL:
        return ours_iteration1_thresholds()
    if profile == HELDOUT_GATE_LABEL:
        return ours_heldout_v1_thresholds()
    raise ValueError(f"unknown alignment profile {profile!r}")


def _using_ours(thresholds):
    return isinstance(thresholds, dict) and thresholds.get("label") == OURS_GATE_LABEL


def _using_heldout(thresholds):
    return (
        isinstance(thresholds, dict) and thresholds.get("label") == HELDOUT_GATE_LABEL
    )


def accept_alignment(residuals, thresholds=None, profile=None):
    """Accept or reject a pose.

    ``thresholds=None`` stays undecided: the paper's four numbers are absent.
    Pass ``ours_iteration1_thresholds()`` or the label ``ours_iteration_1``
    to decide with this tree's iteration-1 gate. A missing silhouette fails
    that gate. It is not treated as a pass.

    ``profile="ours_heldout_v1"`` selects the held-out cutoffs. The default
    does not. An unknown profile name raises.
    """
    named = profile
    if profile is not None:
        thresholds = _profile_thresholds(profile)
    elif thresholds == OURS_GATE_LABEL:
        thresholds = ours_iteration1_thresholds()
    elif thresholds == HELDOUT_GATE_LABEL:
        thresholds = ours_heldout_v1_thresholds()
        named = HELDOUT_GATE_LABEL
    residuals = dict(residuals or {})
    thresholds = dict(thresholds or {})
    ours = _using_ours(thresholds)
    heldout = _using_heldout(thresholds)
    if not ours and not heldout:
        missing = [
            name
            for name in RESIDUAL_NAMES
            if name not in residuals
            or name not in thresholds
            or residuals.get(name) is None
        ]
        if missing or not thresholds:
            return {
                "decision": "undecided",
                "accepted": False,
                "missing": missing or list(RESIDUAL_NAMES),
                "gate": None,
                "paper_published": None,
                "reason": "SceneAgent states that fixed thresholds exist and does not publish them.",
            }
    failed = []
    measured = {}
    for name in DISTANCE_RESIDUALS:
        limit = thresholds.get(name) or {}
        if "max" not in limit and not ours and not heldout:
            return {
                "decision": "undecided",
                "accepted": False,
                "missing": [name],
                "reason": f"{name} needs a max threshold",
            }
        raw = residuals.get(name)
        if raw is None or "max" not in limit:
            failed.append(name)
            continue
        value = float(raw)
        measured[name] = value
        if value > float(limit["max"]):
            failed.append(name)
    for name in AGREEMENT_RESIDUALS:
        limit = thresholds.get(name) or {}
        if "min" not in limit and not ours and not heldout:
            return {
                "decision": "undecided",
                "accepted": False,
                "missing": [name],
                "reason": f"{name} needs a min threshold",
            }
        raw = residuals.get(name)
        if raw is None or "min" not in limit:
            failed.append(name)
            continue
        value = float(raw)
        measured[name] = value
        if value < float(limit["min"]):
            failed.append(name)
    body = {
        "decision": "reject" if failed else "accept",
        "accepted": not failed,
        "failed": failed,
        "residuals": measured,
        "gate": (
            OURS_GATE_LABEL if ours else HELDOUT_GATE_LABEL if heldout else "caller"
        ),
        "paper_published": False if ours or heldout else None,
    }
    if ours:
        body["note"] = OURS_GATE_NOTE
        body["constants"] = {
            "normalized_median_max": OURS_NORMALIZED_MEDIAN_MAX,
            "p90_max": OURS_P90_MAX,
            "coverage_min": OURS_COVERAGE_MIN,
            "silhouette_min": OURS_SILHOUETTE_MIN,
        }
    elif heldout:
        body["note"] = HELDOUT_GATE_NOTE
        body["constants"] = {
            "normalized_median_max": HELDOUT_NORMALIZED_MEDIAN_MAX,
            "p90_max": HELDOUT_P90_MAX,
            "coverage_min": HELDOUT_COVERAGE_MIN,
            "silhouette_min": HELDOUT_SILHOUETTE_MIN,
        }
    if named is not None:
        body["profile"] = named
        body["thresholds_published"] = False
    return body


def accept_placement(residuals):
    """Iteration 2: accept a placed mesh on silhouette and p90 in metres.

    ``residuals`` needs ``silhouette`` and ``p90_m``; a missing one fails,
    as in iteration 1. The other residuals are not read.
    """
    residuals = dict(residuals or {})
    failed = []
    silhouette = residuals.get("silhouette")
    if silhouette is None or float(silhouette) < PLACEMENT_SILHOUETTE_MIN:
        failed.append("silhouette")
    p90_m = residuals.get("p90_m")
    if p90_m is None or float(p90_m) > PLACEMENT_P90_MAX_M:
        failed.append("p90_m")
    return {
        "decision": "reject" if failed else "accept",
        "accepted": not failed,
        "failed": failed,
        "gate": PLACEMENT_GATE_LABEL,
        "paper_published": False,
        "note": PLACEMENT_GATE_NOTE,
        "constants": {
            "silhouette_min": PLACEMENT_SILHOUETTE_MIN,
            "p90_max_m": PLACEMENT_P90_MAX_M,
        },
    }
