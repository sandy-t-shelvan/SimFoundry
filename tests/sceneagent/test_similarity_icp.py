"""Section 3.3 trimmed similarity ICP, and the section 3.7 mesh USDA."""

import hashlib
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
    SILHOUETTE_UNDEFINED_REASON,
    TRIM_KEEP,
    PoseSidecarHashMismatchError,
    PoseSidecarHashMissingError,
    PoseSidecarMissingError,
    PoseSidecarRotationMissingError,
    _score_silhouette,
    _warp_mask,
    apply_similarity,
    icp_step_exit_code,
    load_stored_sam3d_pose,
    main,
    measure_residuals,
    quaternion_xyzw_to_matrix,
    rasterize_orthographic_silhouette,
    rotation_matrix_from_vector,
    rotation_seeds,
    rotation_vector_from_matrix,
    run_icp_step,
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
    pose_path, glb = _sidecar(
        tmp_path,
        "sam3d_pose.json",
        rotation=[0.0, 0.0, math.pi / 2.0],
        translation=translation.tolist(),
        scale=1.7,
    )
    before = pose_path.read_text(encoding="utf-8")
    loaded = load_stored_sam3d_pose(pose_path, glb)
    assert pose_path.read_text(encoding="utf-8") == before
    assert loaded["rotation_stored"] is False
    assert loaded["rotation_source"] == "rotation"
    assert loaded["glb_sha256"] == hashlib.sha256(glb.read_bytes()).hexdigest()
    assert loaded["path"] == str(pose_path)
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
    """Ruling 10152(1): a matching hash with no rotation is an error, not identity."""
    pose_path, glb = _sidecar(tmp_path, scale=1.0, translation=[0.0, 0.0, 0.0])
    before = pose_path.read_text(encoding="utf-8")
    full = np.ones((2, 2), dtype=bool)
    with pytest.raises(PoseSidecarRotationMissingError):
        write_alignment_residual(
            _cube_points(),
            _cube_points(),
            pose_path,
            tmp_path / "icp_residual.json",
            rendered_masks=[full],
            observed_masks=[full],
            ledger_dir=tmp_path / "ledger",
            glb_path=glb,
        )
    assert pose_path.read_text(encoding="utf-8") == before
    stored = json.loads(before)
    assert "rotation" not in stored
    assert "rotation_matrix" not in stored
    assert "quaternion_xyzw" not in stored
    assert stored.get("rotation_source") != "export_frame_identity"
    assert not (tmp_path / "icp_residual.json").exists()
    assert not (tmp_path / "ledger" / "ledger.json").exists()
    with pytest.raises(PoseSidecarRotationMissingError):
        load_stored_sam3d_pose(pose_path, glb)


def test_missing_sidecar_raises_pose_sidecar_missing_error(tmp_path):
    glb = tmp_path / "object.glb"
    glb.write_bytes(b"fake-glb")
    with pytest.raises(PoseSidecarMissingError):
        load_stored_sam3d_pose(tmp_path / "absent.json", glb)
    assert not (tmp_path / "absent.json").exists()


def test_sidecar_hash_mismatch_raises(tmp_path):
    pose_path, glb = _sidecar(tmp_path, rotation=[0.0, 0.0, 0.0])
    before = pose_path.read_text(encoding="utf-8")
    glb.write_bytes(b"different-glb-bytes")
    with pytest.raises(PoseSidecarHashMismatchError):
        load_stored_sam3d_pose(pose_path, glb)
    assert pose_path.read_text(encoding="utf-8") == before
    assert "export_frame_identity" not in before


def test_sidecar_missing_hash_raises(tmp_path):
    glb = tmp_path / "object.glb"
    glb.write_bytes(b"fake-glb")
    pose_path = tmp_path / "pose.json"
    pose_path.write_text(
        json.dumps({"rotation": [0.0, 0.0, 0.0], "scale": 1.0}) + "\n",
        encoding="utf-8",
    )
    before = pose_path.read_text(encoding="utf-8")
    with pytest.raises(PoseSidecarHashMissingError):
        load_stored_sam3d_pose(pose_path, glb)
    assert pose_path.read_text(encoding="utf-8") == before
    assert "export_frame_identity" not in before


def test_quaternion_and_matrix_are_stored_as_rotation_vectors(tmp_path):
    half = math.sin(math.pi / 4.0)
    pose_path, glb = _sidecar(
        tmp_path,
        "quat.json",
        quaternion_xyzw=[0.0, 0.0, half, math.cos(math.pi / 4.0)],
    )
    loaded = load_stored_sam3d_pose(pose_path, glb)
    stored = json.loads(pose_path.read_text(encoding="utf-8"))
    assert stored["rotation_source"] == "quaternion_xyzw"
    assert loaded["rotation_stored"] is True
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
    matrix_path, matrix_glb = _sidecar(
        tmp_path,
        "matrix.json",
        "matrix.glb",
        rotation_matrix=matrix.tolist(),
    )
    from_matrix = load_stored_sam3d_pose(matrix_path, matrix_glb)
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


def _sidecar(tmp_path, name="pose.json", glb_name="object.glb", glb_bytes=b"fake-glb", **fields):
    glb = tmp_path / glb_name
    glb.write_bytes(glb_bytes)
    payload = {"glb_sha256": hashlib.sha256(glb_bytes).hexdigest()}
    payload.update(fields)
    pose = tmp_path / name
    pose.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return pose, glb


def _write_pose(path, rotation):
    _pose, glb = _sidecar(
        path.parent,
        path.name,
        rotation=rotation,
        translation=[0.0, 0.0, 0.0],
        scale=1.0,
    )
    return glb


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
    glb = _write_pose(pose_path, [0.0, 0.0, 0.0])
    full = np.ones((2, 2), dtype=bool)
    written = write_alignment_residual(
        source,
        source,
        pose_path,
        tmp_path / "icp_residual.json",
        rendered_masks=[full],
        observed_masks=[full],
        ledger_dir=tmp_path / "ledger",
        glb_path=glb,
    )
    body = json.loads((tmp_path / "icp_residual.json").read_text(encoding="utf-8"))
    assert body["normalized_median"] == pytest.approx(0.0, abs=1e-9)
    assert body["p90"] == pytest.approx(0.0, abs=1e-9)
    assert body["coverage"] == pytest.approx(0.7)
    assert body["silhouette"] == 1.0
    assert body["thresholds_published"] is False
    assert body["trim_keep"] == 0.7
    assert body["pose_initialization"]["source"] == "sidecar"
    assert body["pose_initialization"]["sidecar_path"] == str(pose_path)
    assert body["pose_initialization"]["glb_sha256"] == hashlib.sha256(glb.read_bytes()).hexdigest()
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
    assert entry["pose_initialization"]["source"] == "sidecar"
    assert entry["pose_initialization"]["sidecar_path"] == str(pose_path)
    assert entry["pose_initialization"]["glb_sha256"] == hashlib.sha256(glb.read_bytes()).hexdigest()
    assert entry["pose_initialization"]["seeds_searched"] == 1
    assert entry["residuals"]["silhouette"] == 1.0
    assert entry["residuals"]["normalized_median"] == pytest.approx(0.0, abs=1e-9)
    assert entry["residuals"]["p90"] == pytest.approx(0.0, abs=1e-9)
    assert entry["residuals"]["coverage"] == pytest.approx(0.7)


def test_low_silhouette_ledger_rejects_without_changing_cutoffs(tmp_path):
    constants = _unchanged_constants()
    source = _cube_points()
    pose_path = tmp_path / "pose.json"
    glb = _write_pose(pose_path, [0.0, 0.0, 0.0])
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
        glb_path=glb,
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
    assert entry["reason"] != SILHOUETTE_UNDEFINED_REASON
    assert written["reason"] == "ours_iteration_1 reject: silhouette"
    assert entry["gate"] == "ours_iteration_1"
    assert entry["alignment_constants"] == constants
    assert entry["pose_initialization"]["seeds_searched"] == 1
    assert entry["pose_initialization"]["source"] == "sidecar"
    assert entry["pose_initialization"]["glb_sha256"] == hashlib.sha256(glb.read_bytes()).hexdigest()


def test_mesh_silhouette_is_written_and_empty_union_rejects(tmp_path):
    constants = _unchanged_constants()
    source = _cube_points()
    pose_path = tmp_path / "pose.json"
    glb = _write_pose(pose_path, [0.0, 0.0, 0.0])
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
        glb_path=glb,
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
        glb_path=glb,
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
    assert ledger["entries"][0]["reason"] == SILHOUETTE_UNDEFINED_REASON
    assert rejected["reason"] == SILHOUETTE_UNDEFINED_REASON
    assert ledger["entries"][0]["alignment_constants"] == constants
    assert run_icp_step(
        source,
        source,
        pose_path,
        glb,
        tmp_path / "step_empty.json",
        rendered_masks=[empty],
        observed_masks=[empty],
        ledger_dir=tmp_path / "step_empty",
    ) == 0
    step_ledger = json.loads((tmp_path / "step_empty" / "ledger.json").read_text(encoding="utf-8"))
    assert step_ledger["entries"][0]["decision"] == "reject"
    assert step_ledger["entries"][0]["reason"] == SILHOUETTE_UNDEFINED_REASON


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


def test_icp_step_gate_reject_exits_zero_and_bad_measures_do_not(tmp_path):
    source = _cube_points()
    pose_path = tmp_path / "pose.json"
    glb = _write_pose(pose_path, [0.0, 0.0, 0.0])
    rendered = np.array([[1, 1], [0, 0]], dtype=bool)
    observed = np.array([[0, 0], [1, 1]], dtype=bool)
    code = run_icp_step(
        source,
        source,
        pose_path,
        glb,
        tmp_path / "residual.json",
        rendered_masks=[rendered],
        observed_masks=[observed],
        ledger_dir=tmp_path / "ledger",
    )
    assert code == 0
    entry = json.loads((tmp_path / "ledger" / "ledger.json").read_text(encoding="utf-8"))["entries"][0]
    assert entry["decision"] == "reject"
    assert entry["reason"] == "ours_iteration_1 reject: silhouette"
    below = {
        "normalized_median": 0.0,
        "p90": 0.0,
        "coverage": 0.7,
        "silhouette": 0.0,
    }
    assert icp_step_exit_code(below, {"decision": "reject", "reason": entry["reason"]}) == 0
    missing_p90 = dict(below)
    missing_p90["p90"] = None
    assert icp_step_exit_code(missing_p90, {"decision": "reject", "reason": entry["reason"]}) == 1
    text_p90 = dict(below)
    text_p90["p90"] = "0.0"
    assert icp_step_exit_code(text_p90, {"decision": "reject", "reason": entry["reason"]}) == 1
    nan_coverage = dict(below)
    nan_coverage["coverage"] = float("nan")
    assert icp_step_exit_code(nan_coverage, {"decision": "reject", "reason": entry["reason"]}) == 1
    undefined = dict(below)
    undefined["silhouette"] = None
    assert icp_step_exit_code(undefined, {"decision": "reject", "reason": "ours_iteration_1 reject: silhouette"}) == 1
    assert icp_step_exit_code(undefined, {"decision": "reject", "reason": SILHOUETTE_UNDEFINED_REASON}) == 0
    assert run_icp_step(
        source,
        source,
        tmp_path / "missing.json",
        glb,
        tmp_path / "crash.json",
        rendered_masks=[rendered],
        observed_masks=[observed],
        ledger_dir=tmp_path / "crash",
    ) == 1
    coincident = np.zeros((5, 3))
    assert run_icp_step(
        coincident,
        coincident,
        pose_path,
        glb,
        tmp_path / "coincident.json",
        rendered_masks=[rendered],
        observed_masks=[observed],
        ledger_dir=tmp_path / "coincident",
    ) == 1
    np.save(tmp_path / "source.npy", source)
    np.save(tmp_path / "target.npy", source)
    np.save(tmp_path / "rendered.npy", rendered)
    np.save(tmp_path / "observed.npy", observed)
    assert main(
        [
            "--pose",
            str(pose_path),
            "--glb",
            str(glb),
            "--out",
            str(tmp_path / "cli.json"),
            "--ledger",
            str(tmp_path / "cli"),
            "--source",
            str(tmp_path / "source.npy"),
            "--target",
            str(tmp_path / "target.npy"),
            "--rendered-mask",
            str(tmp_path / "rendered.npy"),
            "--observed-mask",
            str(tmp_path / "observed.npy"),
        ]
    ) == 0
    cli_entry = json.loads((tmp_path / "cli" / "ledger.json").read_text(encoding="utf-8"))["entries"][0]
    assert cli_entry["decision"] == "reject"
    assert main(["--pose", str(pose_path)]) == 2


def _shifted_square(tmp_path):
    """A square ICP must slide by 0.6 m. The initial raster and the refined one differ."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    shift = np.array([0.6, 0.0, 0.0])
    observed = rasterize_orthographic_silhouette(vertices + shift, faces, width=4, height=4)
    initial = rasterize_orthographic_silhouette(vertices, faces, width=4, height=4)
    pose_path, glb = _sidecar(
        tmp_path,
        rotation=[0.0, 0.0, 0.0],
        translation=[0.0, 0.0, 0.0],
        scale=1.0,
    )
    mesh = {"vertices": vertices, "faces": faces, "width": 4, "height": 4}
    return vertices, faces, observed, initial, pose_path, glb, mesh


def test_silhouette_is_the_refined_pose_not_the_pre_icp_pose(tmp_path):
    vertices, _faces, observed, initial, pose_path, glb, mesh = _shifted_square(tmp_path)
    initial_iou = silhouette_iou(initial, observed)
    assert initial_iou == pytest.approx(1.0 / 3.0)
    written = write_alignment_residual(
        vertices,
        vertices + np.array([0.6, 0.0, 0.0]),
        pose_path,
        tmp_path / "residual.json",
        observed_masks=[observed],
        mesh=mesh,
        ledger_dir=tmp_path / "ledger",
        glb_path=glb,
    )
    assert written["residuals"]["silhouette"] == pytest.approx(1.0)
    assert written["residuals"]["silhouette"] != pytest.approx(initial_iou)
    transform = written["pose_initialization"]["silhouette_transform"]
    assert written["pose_initialization"]["scored_pose"] == "post_icp"
    assert transform["scale"] == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(transform["translation"], [0.6, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(transform["rotation"], np.eye(3), atol=1e-6)
    refined = apply_similarity(
        vertices,
        transform["scale"],
        transform["rotation"],
        transform["translation"],
    )
    refined_iou = silhouette_iou(
        rasterize_orthographic_silhouette(refined, mesh["faces"], width=4, height=4),
        observed,
    )
    assert written["residuals"]["silhouette"] == pytest.approx(refined_iou)
    entry = json.loads((tmp_path / "ledger" / "ledger.json").read_text(encoding="utf-8"))["entries"][0]
    assert entry["pose_initialization"]["scored_pose"] == "post_icp"
    assert entry["pose_initialization"]["silhouette_transform"]["translation"] == pytest.approx(
        [0.6, 0.0, 0.0], abs=1e-6
    )
    assert entry["residuals"]["silhouette"] == pytest.approx(1.0)


def test_mesh_fallback_uses_the_refined_transform_not_a_stale_mask(tmp_path):
    vertices, _faces, observed, initial, pose_path, glb, mesh = _shifted_square(tmp_path)
    stale_iou = silhouette_iou(initial, observed)
    assert stale_iou == pytest.approx(1.0 / 3.0)
    written = write_alignment_residual(
        vertices,
        vertices + np.array([0.6, 0.0, 0.0]),
        pose_path,
        tmp_path / "residual.json",
        rendered_masks=[initial],
        observed_masks=[observed],
        mesh=mesh,
        ledger_dir=tmp_path / "ledger",
        glb_path=glb,
    )
    assert written["residuals"]["silhouette"] == pytest.approx(1.0)
    assert written["residuals"]["silhouette"] != pytest.approx(stale_iou)
    assert written["pose_initialization"]["scored_pose"] == "post_icp"
    entry = json.loads((tmp_path / "ledger" / "ledger.json").read_text(encoding="utf-8"))["entries"][0]
    assert entry["pose_initialization"]["scored_pose"] == "post_icp"
    assert entry["residuals"]["silhouette"] == pytest.approx(1.0)


def test_rendered_mask_is_warped_by_the_refined_similarity(tmp_path):
    vertices, _faces, observed, initial, pose_path, glb, _mesh = _shifted_square(tmp_path)
    stale_iou = silhouette_iou(initial, observed)
    assert stale_iou == pytest.approx(1.0 / 3.0)
    written = write_alignment_residual(
        vertices,
        vertices + np.array([0.6, 0.0, 0.0]),
        pose_path,
        tmp_path / "residual.json",
        rendered_masks=[initial],
        observed_masks=[observed],
        ledger_dir=tmp_path / "ledger",
        glb_path=glb,
    )
    assert written["residuals"]["silhouette"] == pytest.approx(1.0)
    assert written["residuals"]["silhouette"] != pytest.approx(stale_iou)
    assert written["pose_initialization"]["scored_pose"] == "post_icp"
    transform = written["pose_initialization"]["silhouette_transform"]
    np.testing.assert_allclose(transform["translation"], [0.6, 0.0, 0.0], atol=1e-6)


def _square_pair(size, x0, y0, side):
    vertices = np.array(
        [
            [x0, y0, 0.0],
            [x0 + side, y0, 0.0],
            [x0 + side, y0 + side, 0.0],
            [x0, y0 + side, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    source = rasterize_orthographic_silhouette(vertices, faces, width=size, height=size)
    return vertices, faces, source


def _interior_hole_count(mask):
    """False pixels whose four orthogonal neighbors are True."""
    if mask.shape[0] < 3 or mask.shape[1] < 3:
        return 0
    enclosed = (
        ~mask[1:-1, 1:-1]
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
    )
    return int(enclosed.sum())


def _expected_raster(vertices, faces, size, scale, rotation, translation):
    moved = apply_similarity(vertices, scale, rotation, translation)
    return rasterize_orthographic_silhouette(moved, faces, width=size, height=size)


def test_inverse_warp_scale_two_fills_four_times_the_area_without_holes():
    size = 160
    vertices, faces, source = _square_pair(size, 24.0, 24.0, 20.0)
    scale = 2.0
    rotation = np.eye(3)
    translation = np.zeros(3)
    warped = _warp_mask(source, scale, rotation, translation)
    expected = _expected_raster(vertices, faces, size, scale, rotation, translation)
    assert int(source.sum()) > 0
    assert int(warped.sum()) == pytest.approx(4 * int(source.sum()), rel=0.08)
    assert _interior_hole_count(warped) == 0
    assert silhouette_iou(warped, expected) >= 0.95


def test_inverse_warp_forty_five_degrees_has_no_holes():
    size = 180
    vertices, faces, source = _square_pair(size, 70.0, 70.0, 24.0)
    rotation = rotation_matrix_from_vector([0.0, 0.0, math.pi / 4.0])
    translation = np.zeros(3)
    warped = _warp_mask(source, 1.0, rotation, translation)
    expected = _expected_raster(vertices, faces, size, 1.0, rotation, translation)
    assert _interior_hole_count(warped) == 0
    assert silhouette_iou(warped, expected) >= 0.95


def _orthographic_det(rotation):
    """Determinant of the x,y rows ``_warp_mask`` inverts."""
    rotation = np.asarray(rotation, dtype=np.float64)
    return float(rotation[0, 0] * rotation[1, 1] - rotation[0, 1] * rotation[1, 0])


def test_inverse_warp_ninety_degree_x_rotation_refuses_degenerate_projection():
    # A quarter turn about x lays the source plane on edge. The orthographic
    # 2x2 then has |det| under 1e-8, and the warp returns an empty mask
    # instead of inverting it. Translation y is row 0's pixel center: without
    # that refusal the singular inverse is finite and copies source row 0.
    size = 32
    source = np.zeros((size, size), dtype=bool)
    source[0, 8:16] = True
    source[10:18, 10:18] = True
    rotation = rotation_matrix_from_vector([math.pi / 2.0, 0.0, 0.0])
    translation = np.array([0.0, 0.5, 0.0])
    assert abs(_orthographic_det(rotation)) < 1e-8
    warped = _warp_mask(source, 1.0, rotation, translation)
    assert int(source.sum()) > 0
    assert int(warped.sum()) == 0
    assert not np.array_equal(warped, source)


def test_inverse_warp_forty_five_degrees_stays_on_canvas():
    size = 180
    x0, y0, side = 90.0, 90.0, 24.0
    vertices, faces, source = _square_pair(size, x0, y0, side)
    rotation = rotation_matrix_from_vector([0.0, 0.0, math.pi / 4.0])
    center = np.array([x0 + side / 2.0, y0 + side / 2.0, 0.0])
    translation = center - rotation @ center
    warped = _warp_mask(source, 1.0, rotation, translation)
    expected = _expected_raster(vertices, faces, size, 1.0, rotation, translation)
    assert int(source.sum()) > 0
    assert int(warped.sum()) == pytest.approx(int(source.sum()), rel=0.08)
    assert _interior_hole_count(warped) == 0
    assert silhouette_iou(warped, expected) >= 0.95
    for mask in (warped, expected):
        assert not mask[0].any()
        assert not mask[-1].any()
        assert not mask[:, 0].any()
        assert not mask[:, -1].any()


def test_inverse_warp_scale_half_thirty_degrees_and_translation():
    size = 180
    vertices, faces, source = _square_pair(size, 40.0, 50.0, 28.0)
    scale = 0.5
    rotation = rotation_matrix_from_vector([0.0, 0.0, math.pi / 6.0])
    translation = np.array([12.0, -8.0, 0.0])
    warped = _warp_mask(source, scale, rotation, translation)
    expected = _expected_raster(vertices, faces, size, scale, rotation, translation)
    assert _interior_hole_count(warped) == 0
    assert silhouette_iou(warped, expected) >= 0.95


def test_score_silhouette_warps_on_the_mesh_grid():
    size = 48
    origin = (1.0, 2.0)
    pixel_size = 0.05
    x0, y0, side = 1.25, 2.25, 0.40
    vertices = np.array(
        [
            [x0, y0, 0.0],
            [x0 + side, y0, 0.0],
            [x0 + side, y0 + side, 0.0],
            [x0, y0 + side, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    source = rasterize_orthographic_silhouette(
        vertices, faces, size, size, origin=origin, pixel_size=pixel_size
    )
    result = {
        "scale": 1.0,
        "rotation": np.eye(3),
        "translation": np.array([0.50, -0.15, 0.0]),
    }
    moved = apply_similarity(
        vertices, result["scale"], result["rotation"], result["translation"]
    )
    expected = rasterize_orthographic_silhouette(
        moved, faces, size, size, origin=origin, pixel_size=pixel_size
    )
    mask_score = _score_silhouette(
        result,
        [source],
        [expected],
        None,
        origin=origin,
        pixel_size=pixel_size,
    )
    mesh = {
        "vertices": vertices,
        "faces": faces,
        "width": size,
        "height": size,
        "origin": origin,
        "pixel_size": pixel_size,
    }
    mesh_score = _score_silhouette(result, None, [expected], mesh)
    assert int(source.sum()) > 0
    assert mask_score["silhouette"] >= 0.95
    assert mesh_score["silhouette"] >= 0.95
    assert abs(mask_score["silhouette"] - mesh_score["silhouette"]) <= 0.05


def test_warped_mask_iou_matches_mesh_reraster_within_0_05():
    size = 160
    vertices, faces, source = _square_pair(size, 36.0, 44.0, 22.0)
    scale = 1.4
    rotation = rotation_matrix_from_vector([0.35, 0.2, 0.4])
    translation = np.array([6.0, 3.0, 0.0])
    warped = _warp_mask(source, scale, rotation, translation)
    reraster = _expected_raster(vertices, faces, size, scale, rotation, translation)
    # Both are compared with the same observed region, the re-raster itself,
    # and with a shifted copy so neither IoU is forced to 1.
    observed = np.roll(reraster, 2, axis=1)
    warp_iou = silhouette_iou(warped, observed)
    mesh_iou = silhouette_iou(reraster, observed)
    assert warp_iou is not None and mesh_iou is not None
    assert abs(warp_iou - mesh_iou) <= 0.05
