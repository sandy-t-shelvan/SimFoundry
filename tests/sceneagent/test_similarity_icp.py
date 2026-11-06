"""Section 3.3 trimmed similarity ICP, and the section 3.7 mesh USDA."""

import json
import math

import numpy as np
import pytest

from sceneagent.alignment import (
    FRUITS_ICP_RESIDUAL,
    OURS_COVERAGE_MIN,
    OURS_NORMALIZED_MEDIAN_MAX,
    OURS_P90_MAX,
    OURS_SILHOUETTE_MIN,
    accept_alignment,
)
from sceneagent.similarity_icp import (
    TRIM_KEEP,
    apply_similarity,
    load_stored_sam3d_pose,
    measure_residuals,
    quaternion_xyzw_to_matrix,
    rasterize_orthographic_silhouette,
    rotation_matrix_from_vector,
    rotation_seeds,
    rotation_vector_from_matrix,
    silhouette_agreement,
    silhouette_iou,
    trimmed_similarity_icp,
    write_alignment_residual,
)
from sceneagent.usd_contract import write_aligned_mesh_usda


def test_icp_recovers_a_known_similarity_and_does_not_invent_a_gate():
    rng = np.random.default_rng(0)
    source = rng.normal(size=(48, 3))
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    target = apply_similarity(source, 2.5, rotation, np.array([4.0, -1.0, 0.25]))
    extras = rng.normal(size=(12, 3)) * 30.0
    result = trimmed_similarity_icp(source, np.vstack([target, extras]), iterations=25)
    aligned = apply_similarity(source, result["scale"], result["rotation"], result["translation"])
    gap = np.median(np.linalg.norm(aligned - target, axis=1))
    assert gap < 1e-3
    residuals = measure_residuals(result, np.vstack([target, extras]))
    assert residuals["silhouette"] is None
    assert residuals["thresholds_published"] is False
    decision = accept_alignment(
        {
            "normalized_median": residuals["normalized_median"],
            "p90": residuals["p90"],
            "coverage": residuals["coverage"],
            "silhouette": 0.0,
        },
        {},
    )
    assert decision["decision"] == "undecided"
    assert decision["accepted"] is False


def test_aligned_mesh_usda_keeps_the_bowl_and_replaces_the_box(tmp_path):
    contract = {
        "objects": [
            {
                "name": "box",
                "rigid_body": True,
                "mass_kg": 0.12,
                "density_kg_m3": 700.0,
                "friction_static": 0.45,
                "friction_dynamic": 0.35,
                "restitution": 0.05,
                "body_type": "dynamic rigid",
                "collision_approximation": "box",
                "material_class": "paper and cardboard",
            },
            {
                "name": "bowl",
                "rigid_body": True,
                "mass_kg": None,
                "density_kg_m3": None,
                "friction_static": None,
                "friction_dynamic": None,
                "restitution": None,
                "body_type": "static rigid",
                "collision_approximation": "convex hull",
                "material_class": "glass and ceramic",
            },
        ]
    }
    written = write_aligned_mesh_usda(
        contract,
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        [(0, 1, 2)],
        tmp_path / "aligned.usda",
    )
    text = written.read_text(encoding="utf-8")
    assert 'def Xform "aligned_mesh"' in text
    assert 'def Mesh "geometry"' in text
    assert 'def Xform "bowl"' in text
    assert 'def Xform "box"' not in text
    assert "physics:mass = 0.12" in text


def test_silhouette_iou_matches_known_mask_overlaps():
    rendered = np.array(
        [
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ],
        dtype=bool,
    )
    observed = np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 0],
        ],
        dtype=bool,
    )
    # Intersection is two pixels. Union is four. Jaccard is 1/2.
    assert silhouette_iou(rendered, observed) == 0.5
    identical = np.array([[1, 0], [0, 0]], dtype=bool)
    assert silhouette_iou(identical, identical) == 1.0
    assert silhouette_iou(np.array([[1, 0]], dtype=bool), np.array([[0, 1]], dtype=bool)) == 0.0
    assert silhouette_iou(np.zeros((2, 2), dtype=bool), np.zeros((2, 2), dtype=bool)) is None
    agreed = silhouette_agreement([identical, rendered], [identical, observed])
    assert agreed["silhouette"] == 0.75
    assert agreed["views"] == 2
    assert agreed["empty_views"] == 0
    empty = np.zeros((2, 2), dtype=bool)
    partial = silhouette_agreement([identical, empty], [identical, empty])
    assert partial["silhouette"] == 1.0
    assert partial["empty_views"] == 1
    blank = silhouette_agreement([empty], [empty])
    assert blank["silhouette"] is None
    assert blank["views"] == 0
    assert blank["empty_views"] == 1


def test_rasterized_square_silhouette_is_one_and_one_half():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    mask = rasterize_orthographic_silhouette(vertices, faces, width=4, height=4)
    assert mask.tolist() == [
        [True, True, False, False],
        [True, True, False, False],
        [False, False, False, False],
        [False, False, False, False],
    ]
    assert silhouette_iou(mask, mask) == 1.0
    left_half = np.array(
        [
            [True, False, False, False],
            [True, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
        ],
        dtype=bool,
    )
    # The square covers four pixel centers. The observed mask keeps two of them.
    assert silhouette_iou(mask, left_half) == 0.5
    agreed = silhouette_agreement([mask], [left_half])
    assert agreed["silhouette"] == 0.5


def test_icp_starts_from_stored_sam3d_rotation_not_the_seed_sweep(tmp_path):
    rng = np.random.default_rng(1)
    source = rng.normal(size=(40, 3))
    rotation = rotation_matrix_from_vector([0.0, 0.0, math.pi / 2.0])
    translation = np.array([0.2, -0.4, 0.5])
    target = apply_similarity(source, 1.7, rotation, translation)
    pose_path = tmp_path / "sam3d_pose.json"
    pose_path.write_text(
        json.dumps(
            {
                "rotation": [0.0, 0.0, math.pi / 2.0],
                "translation": translation.tolist(),
                "scale": 1.7,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    before = pose_path.read_text(encoding="utf-8")
    loaded = load_stored_sam3d_pose(pose_path)
    assert pose_path.read_text(encoding="utf-8") == before
    assert loaded["rotation_stored"] is False
    assert loaded["rotation_was_absent"] is False
    assert loaded["rotation_source"] == "rotation"
    result = trimmed_similarity_icp(
        source,
        target,
        initial_rotation=loaded["rotation"],
        initial_scale=loaded["scale"],
        initial_translation=loaded["translation"],
    )
    sweep = trimmed_similarity_icp(source, target)
    assert result["seeds_searched"] == 1
    assert result["pose_source"] == "stored_sam3d"
    assert result["rotation_seed_index"] == 0
    assert sweep["seeds_searched"] == len(rotation_seeds())
    assert sweep["seeds_searched"] > 1
    assert sweep["pose_source"] == "rotation_seeds"
    np.testing.assert_allclose(result["initial_rotation"], rotation, atol=1e-8)
    aligned = apply_similarity(source, result["scale"], result["rotation"], result["translation"])
    gap = np.median(np.linalg.norm(aligned - target, axis=1))
    assert gap < 1e-3


def test_missing_rotation_vector_is_stored_and_icp_starts_from_it(tmp_path):
    pose_path = tmp_path / "sam3d_pose.json"
    pose_path.write_text(
        json.dumps({"scale": 1.0, "translation": [0.0, 0.0, 0.0]}) + "\n",
        encoding="utf-8",
    )
    full = np.ones((2, 2), dtype=bool)
    written = write_alignment_residual(
        _cube_points(),
        _cube_points(),
        pose_path,
        tmp_path / "icp_residual.json",
        rendered_masks=[full],
        observed_masks=[full],
        ledger_dir=tmp_path / "ledger",
    )
    stored = json.loads(pose_path.read_text(encoding="utf-8"))
    assert stored["rotation"] == [0.0, 0.0, 0.0]
    assert stored["rotation_source"] == "export_frame_identity"
    pose_info = written["pose_initialization"]
    assert pose_info["source"] == "stored_sam3d"
    assert pose_info["rotation"] == [0.0, 0.0, 0.0]
    assert pose_info["rotation_source"] == "export_frame_identity"
    assert pose_info["rotation_was_absent"] is True
    assert pose_info["rotation_stored"] is True
    assert pose_info["seeds_searched"] == 1
    ledger = json.loads((tmp_path / "ledger" / "ledger.json").read_text(encoding="utf-8"))
    entry = ledger["entries"][0]
    assert entry["item"] == "icp"
    assert entry["gate"] == "ours_iteration_1"
    assert entry["pose_initialization"] == pose_info
    assert entry["decision"] in {"accept", "reject"}
    again = load_stored_sam3d_pose(pose_path)
    assert again["rotation_stored"] is False
    assert again["rotation_source"] == "export_frame_identity"
    with pytest.raises(FileNotFoundError):
        load_stored_sam3d_pose(tmp_path / "absent.json")


def test_quaternion_and_matrix_are_stored_as_rotation_vectors(tmp_path):
    half = math.sin(math.pi / 4.0)
    pose_path = tmp_path / "quat.json"
    pose_path.write_text(
        json.dumps({"quaternion_xyzw": [0.0, 0.0, half, math.cos(math.pi / 4.0)]}) + "\n",
        encoding="utf-8",
    )
    loaded = load_stored_sam3d_pose(pose_path)
    stored = json.loads(pose_path.read_text(encoding="utf-8"))
    assert stored["rotation_source"] == "quaternion_xyzw"
    assert loaded["rotation_was_absent"] is True
    np.testing.assert_allclose(stored["rotation"], [0.0, 0.0, math.pi / 2.0], atol=1e-8)
    np.testing.assert_allclose(loaded["rotation_vector"], [0.0, 0.0, math.pi / 2.0], atol=1e-8)
    expected = quaternion_xyzw_to_matrix([0.0, 0.0, half, math.cos(math.pi / 4.0)])
    np.testing.assert_allclose(loaded["rotation"], expected, atol=1e-8)
    started = trimmed_similarity_icp(
        _cube_points(),
        _cube_points(),
        initial_rotation=loaded["rotation"],
    )
    assert started["seeds_searched"] == 1
    assert started["pose_source"] == "stored_sam3d"
    np.testing.assert_allclose(started["initial_rotation"], expected, atol=1e-8)
    matrix = rotation_matrix_from_vector([math.pi, 0.0, 0.0])
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(
        json.dumps({"rotation_matrix": matrix.tolist()}) + "\n",
        encoding="utf-8",
    )
    from_matrix = load_stored_sam3d_pose(matrix_path)
    assert json.loads(matrix_path.read_text(encoding="utf-8"))["rotation_source"] == "rotation_matrix"
    round_trip = rotation_matrix_from_vector(from_matrix["rotation_vector"])
    np.testing.assert_allclose(round_trip, matrix, atol=1e-8)
    vector = rotation_vector_from_matrix(matrix)
    np.testing.assert_allclose(np.linalg.norm(vector), math.pi, atol=1e-8)


def _cube_points():
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
            [0.2, 0.8, 0.1],
        ]
    )


def _write_pose(path, rotation):
    path.write_text(
        json.dumps(
            {
                "rotation": rotation,
                "translation": [0.0, 0.0, 0.0],
                "scale": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _unchanged_constants():
    assert TRIM_KEEP == 0.7
    assert OURS_NORMALIZED_MEDIAN_MAX == 0.05
    assert OURS_P90_MAX == 0.25
    assert OURS_COVERAGE_MIN == 0.5
    assert OURS_SILHOUETTE_MIN == 0.5
    return {
        "normalized_median_max": 0.05,
        "p90_max": 0.25,
        "coverage_min": 0.5,
        "silhouette_min": 0.5,
    }


def test_residual_json_has_four_measures_and_ledger_accepts(tmp_path):
    constants = _unchanged_constants()
    source = _cube_points()
    pose_path = tmp_path / "pose.json"
    _write_pose(pose_path, [0.0, 0.0, 0.0])
    full = np.ones((2, 2), dtype=bool)
    written = write_alignment_residual(
        source,
        source,
        pose_path,
        tmp_path / "icp_residual.json",
        rendered_masks=[full],
        observed_masks=[full],
        ledger_dir=tmp_path / "ledger",
    )
    body = json.loads((tmp_path / "icp_residual.json").read_text(encoding="utf-8"))
    assert body["normalized_median"] == pytest.approx(0.0, abs=1e-9)
    assert body["p90"] == pytest.approx(0.0, abs=1e-9)
    assert body["coverage"] == pytest.approx(0.7)
    assert body["silhouette"] == 1.0
    assert body["thresholds_published"] is False
    assert body["trim_keep"] == 0.7
    assert body["pose_initialization"]["source"] == "stored_sam3d"
    assert body["pose_initialization"]["seeds_searched"] == 1
    assert body["pose_initialization"]["rotation"] == [0.0, 0.0, 0.0]
    for name in ("normalized_median", "p90", "coverage", "silhouette"):
        assert isinstance(body[name], float)
    assert written["decision"]["decision"] == "accept"
    ledger = json.loads((tmp_path / "ledger" / "ledger.json").read_text(encoding="utf-8"))
    entry = ledger["entries"][0]
    assert entry["item"] == "icp"
    assert entry["decision"] == "accept"
    assert entry["skill"] == "SCENEAGENT-ALIGN-SCENE"
    assert entry["gate"] == "ours_iteration_1"
    assert entry["reason"] == "ours_iteration_1 accept"
    assert entry["alignment_constants"] == constants
    assert entry["pose_initialization"]["source"] == "stored_sam3d"
    assert entry["pose_initialization"]["seeds_searched"] == 1
    assert entry["residuals"]["silhouette"] == 1.0
    assert entry["residuals"]["normalized_median"] == pytest.approx(0.0, abs=1e-9)
    assert entry["residuals"]["p90"] == pytest.approx(0.0, abs=1e-9)
    assert entry["residuals"]["coverage"] == pytest.approx(0.7)


def test_low_silhouette_ledger_rejects_without_changing_cutoffs(tmp_path):
    constants = _unchanged_constants()
    source = _cube_points()
    pose_path = tmp_path / "pose.json"
    _write_pose(pose_path, [0.0, 0.0, 0.0])
    rendered = np.array([[1, 1], [0, 0]], dtype=bool)
    observed = np.array([[0, 0], [1, 1]], dtype=bool)
    assert silhouette_iou(rendered, observed) == 0.0
    written = write_alignment_residual(
        source,
        source,
        pose_path,
        tmp_path / "icp_residual.json",
        rendered_masks=[rendered],
        observed_masks=[observed],
        ledger_dir=tmp_path / "ledger",
    )
    body = json.loads((tmp_path / "icp_residual.json").read_text(encoding="utf-8"))
    assert body["silhouette"] == 0.0
    assert isinstance(body["normalized_median"], float)
    assert isinstance(body["p90"], float)
    assert isinstance(body["coverage"], float)
    assert written["decision"]["decision"] == "reject"
    assert written["decision"]["failed"] == ["silhouette"]
    assert written["decision"]["constants"] == constants
    ledger = json.loads((tmp_path / "ledger" / "ledger.json").read_text(encoding="utf-8"))
    entry = ledger["entries"][0]
    assert entry["decision"] == "reject"
    assert entry["reason"] == "ours_iteration_1 reject: silhouette"
    assert entry["gate"] == "ours_iteration_1"
    assert entry["alignment_constants"] == constants
    assert entry["pose_initialization"]["seeds_searched"] == 1
    assert entry["pose_initialization"]["source"] == "stored_sam3d"


def test_mesh_silhouette_is_written_and_empty_union_rejects(tmp_path):
    constants = _unchanged_constants()
    source = _cube_points()
    pose_path = tmp_path / "pose.json"
    _write_pose(pose_path, [0.0, 0.0, 0.0])
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    observed = np.array(
        [
            [True, True, False, False],
            [True, True, False, False],
            [False, False, False, False],
            [False, False, False, False],
        ]
    )
    written = write_alignment_residual(
        source,
        source,
        pose_path,
        tmp_path / "mesh_residual.json",
        observed_masks=[observed],
        mesh={"vertices": vertices, "faces": faces, "width": 4, "height": 4},
        ledger_dir=tmp_path / "accept",
    )
    body = json.loads((tmp_path / "mesh_residual.json").read_text(encoding="utf-8"))
    assert body["silhouette"] == 1.0
    assert written["decision"]["decision"] == "accept"
    assert written["decision"]["constants"] == constants
    empty = np.zeros((4, 4), dtype=bool)
    rejected = write_alignment_residual(
        source,
        source,
        pose_path,
        tmp_path / "empty_residual.json",
        rendered_masks=[empty],
        observed_masks=[empty],
        ledger_dir=tmp_path / "reject",
    )
    empty_body = json.loads((tmp_path / "empty_residual.json").read_text(encoding="utf-8"))
    assert empty_body["silhouette"] is None
    assert empty_body["silhouette_empty_views"] == 1
    assert empty_body["normalized_median"] == pytest.approx(0.0, abs=1e-9)
    assert rejected["decision"]["decision"] == "reject"
    assert "silhouette" in rejected["decision"]["failed"]
    assert rejected["decision"]["constants"] == constants
    ledger = json.loads((tmp_path / "reject" / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["entries"][0]["decision"] == "reject"
    assert ledger["entries"][0]["residuals"]["silhouette"] is None
    assert ledger["entries"][0]["alignment_constants"] == constants


def test_gate_constants_and_fruits_residual_are_unchanged():
    constants = _unchanged_constants()
    fruits = accept_alignment(FRUITS_ICP_RESIDUAL, "ours_iteration_1")
    assert fruits["decision"] == "reject"
    assert fruits["accepted"] is False
    assert fruits["gate"] == "ours_iteration_1"
    assert set(fruits["failed"]) == {"coverage", "silhouette"}
    assert fruits["constants"] == constants
    assert FRUITS_ICP_RESIDUAL["normalized_median"] == 0.033103985722272695
    assert FRUITS_ICP_RESIDUAL["p90"] == 0.21563629180764
    assert FRUITS_ICP_RESIDUAL["coverage"] == 0.14616666666666667
    assert FRUITS_ICP_RESIDUAL["silhouette"] is None
