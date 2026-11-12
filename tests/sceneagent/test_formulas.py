"""Unit tests for the published SceneAgent formulas used in phase 1."""

import math

import numpy as np
import pytest

from sceneagent.alignment import accept_alignment
from sceneagent.association import (
    centroid_threshold,
    gaussian_belongs,
    mask_cell_side_px,
    masks_same_object,
)
from sceneagent.evaluation import score_trials
from sceneagent.metrics import (
    f1_at_threshold,
    score_reconstruction,
    symmetric_chamfer,
    wilson_interval,
)
from sceneagent.physics import build_physics_record, resolve_body_type, vote_material
from sceneagent.sisters import box_mesh, geometry_sisters, mesh_volume


def test_unknown_class_weight_is_scaled_by_0_4():
    winner, scores = vote_material(
        [
            {"class": "unknown", "weight": 1.0},
            {"class": "plastic", "weight": 0.3},
        ]
    )
    assert scores["unknown"] == pytest.approx(0.4)
    assert winner == "unknown"
    winner, _ = vote_material(
        [
            {"class": "unknown", "weight": 1.0},
            {"class": "plastic", "weight": 0.5},
        ]
    )
    assert winner == "plastic"


def test_soft_body_threshold_is_inclusive():
    assert resolve_body_type(0.65, explicit="dynamic rigid") == (
        "soft deformable",
        "deformability_at_least_0.65",
    )
    assert resolve_body_type(0.649, explicit="dynamic rigid") == (
        "dynamic rigid",
        "explicit_proposal",
    )
    assert resolve_body_type(None, explicit=None) == (None, "unresolved")


def test_mass_is_stored_as_a_logarithm():
    record = build_physics_record(
        "marker",
        [{"source": "keyword", "class": "plastic", "weight": 1.0}],
        numerical={"mass_kg": 0.12, "density_kg_m3": 700.0, "rigidity": 0.4},
    )
    assert record["material_class"] == "plastic"
    assert record["log_mass"] == pytest.approx(math.log(0.12))
    assert record["log_density"] == pytest.approx(math.log(700.0))
    assert record["unpublished_class_lookups"] == [
        "youngs_modulus",
        "poisson_ratio",
        "rendering_material",
    ]


def test_mask_grouping_uses_published_cutoffs():
    extent = 0.2
    assert centroid_threshold(extent) == pytest.approx(0.10)
    assert centroid_threshold(0.1) == pytest.approx(0.06)
    assert masks_same_object(0.06, 0.1, 0.18, 0.0)
    assert not masks_same_object(0.0601, 0.1, 0.18, 0.0)
    assert masks_same_object(0.05, 0.2, 0.0, 0.08)
    assert not masks_same_object(0.05, 0.2, 0.179, 0.079)


def test_gaussian_front_surface_band():
    # eps_a = 0.08 dominates when 0.03 * depth is smaller.
    assert gaussian_belongs(True, 1.08, 1.0)
    assert not gaussian_belongs(True, 1.0801, 1.0)
    assert not gaussian_belongs(False, 1.0, 1.0)
    # At depth 10, eps_r * depth = 0.3 dominates.
    assert gaussian_belongs(True, 10.3, 10.0)
    assert not gaussian_belongs(True, 10.31, 10.0)


def test_mask_cell_side_is_capped_at_16():
    assert mask_cell_side_px(1920) == 10
    assert mask_cell_side_px(4000) == 16


def test_f1_and_chamfer_on_known_clouds():
    cloud = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.02, 0.0]])
    assert symmetric_chamfer(cloud, cloud) == pytest.approx(0.0)
    assert f1_at_threshold(cloud, cloud, tau=0.01) == pytest.approx(1.0)
    shifted = cloud + np.array([0.05, 0.0, 0.0])
    assert f1_at_threshold(cloud, shifted, tau=0.01) == pytest.approx(0.0)
    scored = score_reconstruction(
        {"a": cloud, "b": shifted},
        {"a": cloud, "missing": shifted + np.array([1.0, 0.0, 0.0])},
    )
    assert scored["f1_by_reference"]["missing"] == 0.0
    assert 0.0 <= scored["f1_mean"] <= 1.0


def test_wilson_matches_the_website_interval_for_one_of_ten():
    low, high = wilson_interval(1, 10)
    assert low == pytest.approx(0.018, abs=0.002)
    assert high == pytest.approx(0.404, abs=0.002)
    low0, high0 = wilson_interval(0, 10)
    assert low0 == pytest.approx(0.0, abs=1e-9)
    assert high0 == pytest.approx(0.278, abs=0.005)


def test_alignment_requires_caller_supplied_thresholds():
    residuals = {
        "normalized_median": 0.01,
        "p90": 0.02,
        "coverage": 0.9,
        "silhouette": 0.8,
    }
    undecided = accept_alignment(residuals, {})
    assert undecided["decision"] == "undecided"
    assert undecided["accepted"] is False
    thresholds = {
        "normalized_median": {"max": 0.05},
        "p90": {"max": 0.08},
        "coverage": {"min": 0.5},
        "silhouette": {"min": 0.5},
    }
    assert accept_alignment(residuals, thresholds)["accepted"] is True
    residuals["p90"] = 0.2
    rejected = accept_alignment(residuals, thresholds)
    assert rejected["decision"] == "reject"
    assert "p90" in rejected["failed"]


def test_box_volume_scales_with_one_axis():
    verts, faces = box_mesh((1.0, 1.0, 1.0))
    assert mesh_volume(verts, faces) == pytest.approx(1.0)
    sisters = geometry_sisters(
        (0.096, 0.04, 0.04), target_inner=(0.10, 0.10, 0.08), length_scales=(1.4,)
    )
    by_name = {row["name"]: row for row in sisters}
    assert by_name["original"]["accepted"] is True
    assert by_name["length +40%"]["accepted"] is False
    assert "does_not_fit_target" in by_name["length +40%"]["reasons"]
    assert by_name["length +40%"]["volume_ratio"] == pytest.approx(1.4)


def test_scripted_trials_fail_outside_the_reach_disc():
    starts = [(0.0, 0.0), (0.04, 0.0), (0.02, 0.02)]
    successes, rows = score_trials(starts, reach_radius_m=0.03, sister_accepted=True)
    assert successes == 2
    assert rows[1]["success"] is False
    successes, _ = score_trials(starts, reach_radius_m=0.03, sister_accepted=False)
    assert successes == 0


def test_placement_gate_iteration_2_reads_silhouette_and_p90_in_metres():
    from sceneagent.alignment import PLACEMENT_GATE_LABEL, accept_placement

    good = accept_placement({"silhouette": 0.7, "p90_m": 0.02, "coverage": 0.1})
    assert good["decision"] == "accept" and good["gate"] == PLACEMENT_GATE_LABEL
    # Coverage and normalized median are not read.
    assert accept_placement({"silhouette": 0.7, "p90_m": 0.02})["accepted"]
    assert accept_placement({"silhouette": 0.4, "p90_m": 0.02})["failed"] == [
        "silhouette"
    ]
    assert accept_placement({"silhouette": 0.7, "p90_m": 0.06})["failed"] == ["p90_m"]
    assert accept_placement({})["failed"] == ["silhouette", "p90_m"]


# The bin's 14 placed objects in run 43315660864: (object, normalized median,
# p90 in splat units, coverage, silhouette, ground-truth F1). 0.20116 m per unit.
BIN_RUN_43315660864 = (
    (0, 0.076, 0.111, 0.300, 0.323, 0.338),
    (1, 0.044, 0.095, 0.118, 0.707, 0.654),
    (2, 0.042, 0.054, 0.164, 0.879, 0.926),
    (3, 0.091, 0.156, 0.363, 0.489, 0.253),
    (4, 0.079, 0.234, 0.390, 0.410, 0.189),
    (5, 0.092, 0.106, 0.341, 0.542, 0.393),
    (6, 0.068, 0.126, 0.327, 0.505, 0.442),
    (7, 0.108, 0.053, 0.504, 0.722, 0.458),
    (8, 0.106, 0.046, 0.525, 0.724, 0.547),
    (9, 0.160, 0.144, 0.451, 0.390, 0.496),
    (10, 0.102, 0.052, 0.250, 0.735, 0.888),
    (11, 0.077, 0.043, 0.303, 0.858, 0.842),
    (12, 0.116, 0.446, 0.116, 0.210, 0.085),
    (13, 0.040, 0.067, 0.518, 0.701, 0.402),
)


def test_iteration_2_keeps_the_bins_best_placements_that_iteration_1_rejected():
    from sceneagent.alignment import accept_placement

    unit = 0.20116
    old, new = {}, {}
    for number, median, p90, coverage, silhouette, f1 in BIN_RUN_43315660864:
        residuals = {
            "normalized_median": median,
            "p90": p90,
            "p90_m": p90 * unit,
            "coverage": coverage,
            "silhouette": silhouette,
        }
        old[number] = accept_alignment(residuals, "ours_iteration_1")["accepted"]
        new[number] = accept_placement(residuals)["accepted"]
    assert [n for n, ok in old.items() if ok] == [13]
    assert [n for n, ok in new.items() if ok] == [1, 2, 5, 6, 7, 8, 10, 11, 13]
    f1 = {row[0]: row[5] for row in BIN_RUN_43315660864}
    kept = [f1[n] for n, ok in new.items() if ok]
    dropped = [f1[n] for n, ok in new.items() if not ok]
    assert sum(kept) / len(kept) > 0.6 > 0.3 > sum(dropped) / len(dropped)
