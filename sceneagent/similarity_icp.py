"""Section 3.3 trimmed similarity ICP.

Umeyama (1991) solves the scale, rotation, and translation between paired
points. Each iteration keeps the closest fraction of those pairs. That trim
is the overlap used by trimmed ICP. ``TRIM_KEEP`` is this implementation's
overlap, not a number published by SceneAgent.

The section 3.1 gate still needs the unpublished acceptance thresholds.
A residual file records the measured distances. It does not accept the pose.
``write_alignment_residual`` asks ``accept_alignment`` with the unchanged
``ours_iteration_1`` cutoffs and records that decision on the ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
import sys
from pathlib import Path

import numpy as np

from sceneagent.agent_loop import append_step
from sceneagent.alignment import RESIDUAL_NAMES, accept_alignment

# Closest-pair fraction retained each iteration. Not a SceneAgent threshold.
TRIM_KEEP = 0.7

# Not a published SceneAgent reduction and not an acceptance cutoff.
# An empty union on every view stays null so the gate can reject it.
SILHOUETTE_DEFINITION = (
    "mean Jaccard index of the rendered silhouette and the observed mask "
    "over views whose raster union is non-empty; null when every union is empty"
)

# Ledger reason for a null silhouette. A measured 0 keeps the below-minimum
# reason from the iteration-1 gate. KAN-8 comment 10152, ruling 2.
SILHOUETTE_UNDEFINED_REASON = "silhouette_undefined"


class PoseSidecarError(Exception):
    """The stored SAM 3D pose cannot start ICP."""


class PoseSidecarMissingError(PoseSidecarError):
    """The pose sidecar file is not there. Identity is not a substitute."""


class PoseSidecarHashMissingError(PoseSidecarError):
    """The sidecar does not record the GLB SHA-256."""


class PoseSidecarHashMismatchError(PoseSidecarError):
    """The sidecar SHA-256 is not the hash of the GLB on disk."""


class PoseSidecarRotationMissingError(PoseSidecarError):
    """The sidecar matches the GLB and stores no rotation."""


def umeyama(source, target):
    """Similarity that maps source points onto paired target points.

    Returns scale, a 3x3 rotation, and a translation. The transform is
    ``scale * rotation @ point + translation``.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must be matching Nx3 arrays")
    if len(source) < 3:
        raise ValueError("Umeyama needs at least 3 pairs")
    count = float(len(source))
    mean_source = source.mean(axis=0)
    mean_target = target.mean(axis=0)
    centered_source = source - mean_source
    centered_target = target - mean_target
    covariance = centered_target.T @ centered_source / count
    left, singular, right = np.linalg.svd(covariance)
    sign = np.eye(3)
    if np.linalg.det(left) * np.linalg.det(right) < 0:
        sign[2, 2] = -1.0
    rotation = left @ sign @ right
    variance = float((centered_source**2).sum() / count)
    if variance <= 0:
        raise ValueError("source points are coincident")
    scale = float(np.trace(np.diag(singular) @ sign) / variance)
    translation = mean_target - scale * (rotation @ mean_source)
    return scale, rotation, translation


def apply_similarity(points, scale, rotation, translation):
    points = np.asarray(points, dtype=np.float64)
    transformed = scale * (np.asarray(rotation, dtype=np.float64) @ points.T)
    return transformed.T + np.asarray(translation, dtype=np.float64)


def _rodrigues(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return cosine * np.eye(3) + sine * cross + (1.0 - cosine) * np.outer(axis, axis)


def rotation_seeds():
    """Export-frame identity plus quarter-turns about the coordinate axes.

    These seeds are the search only when the caller did not pass a stored
    SAM 3D pose. They are not that pose, and they are not acceptance
    thresholds. ``write_alignment_residual`` does not use them, and it does
    not fill a missing rotation with identity.
    """
    seeds = [np.eye(3)]
    for axis in np.eye(3):
        for turn in (1, 2, 3):
            seeds.append(_rodrigues(axis, turn * np.pi / 2.0))
    return seeds


def rotation_matrix_from_vector(vector):
    """Rodrigues rotation vector to a 3x3 matrix. A zero vector is identity."""
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    if not np.isfinite(vector).all():
        raise ValueError("rotation vector must be finite")
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    return _rodrigues(vector / angle, angle)


def quaternion_xyzw_to_matrix(quaternion):
    """XYZW quaternion to a rotation matrix. The scene frame uses XYZW."""
    quat = np.asarray(quaternion, dtype=np.float64).reshape(4)
    if not np.isfinite(quat).all():
        raise ValueError("quaternion must be finite")
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        raise ValueError("quaternion has zero length")
    x, y, z, w = quat / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_vector_from_matrix(rotation):
    """3x3 rotation to a Rodrigues vector. Identity is ``[0, 0, 0]``."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation matrix must be a finite 3x3")
    if float(np.linalg.det(matrix)) < 0.0:
        raise ValueError("rotation matrix must have a positive determinant")
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(math.acos(cosine))
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float64)
    if math.pi - angle < 1e-6:
        axis = _axis_from_pi_rotation(matrix)
        return axis * angle
    axis = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=np.float64,
    )
    axis = axis / np.linalg.norm(axis)
    return axis * angle


def _axis_from_pi_rotation(matrix):
    """Unit axis of a rotation near 180 degrees.

    The skew-symmetric part vanishes at 180 degrees, so the axis has to come
    from the diagonal. The largest diagonal entry is the stable one. The other
    components are read from that row, which carries their signs.
    """
    axis_index = int(np.argmax(np.diag(matrix)))
    axis = np.zeros(3, dtype=np.float64)
    diagonal = float(matrix[axis_index, axis_index])
    axis[axis_index] = math.sqrt(max((diagonal + 1.0) * 0.5, 0.0))
    if axis[axis_index] < 1e-8:
        raise ValueError("rotation near 180 degrees has no axis")
    for other in range(3):
        if other == axis_index:
            continue
        axis[other] = float(matrix[axis_index, other]) / (2.0 * axis[axis_index])
    length = float(np.linalg.norm(axis))
    if length < 1e-8:
        raise ValueError("rotation near 180 degrees has no axis")
    return axis / length


def _nearest(query, target, tree):
    """Nearest neighbors. A KD-tree is used when scipy is installed."""
    query = np.asarray(query, dtype=np.float64)
    if tree is not None:
        distance, index = tree.query(query, k=1)
        return np.asarray(distance, dtype=np.float64), np.asarray(index, dtype=np.int64)
    target = np.asarray(target, dtype=np.float64)
    nearest_distance = np.empty(len(query), dtype=np.float64)
    nearest_index = np.empty(len(query), dtype=np.int64)
    chunk = 256
    for start in range(0, len(query), chunk):
        block = query[start : start + chunk]
        squared = ((block[:, None, :] - target[None, :, :]) ** 2).sum(axis=2)
        index = squared.argmin(axis=1)
        nearest_index[start : start + len(block)] = index
        nearest_distance[start : start + len(block)] = np.sqrt(
            squared[np.arange(len(block)), index]
        )
    return nearest_distance, nearest_index


def initial_similarity(source, target):
    """Place the exported mesh on the target centroid. Rotation starts as identity.

    The SAM 3D export frame is the source. Scale matches the two clouds' RMS
    sizes so the first correspondence is not a unit mesh against a scene-sized
    splat. That scale is a measured ratio, not an acceptance threshold.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_rms = float(
        np.sqrt(((source - source.mean(axis=0)) ** 2).sum(axis=1).mean())
    )
    target_rms = float(
        np.sqrt(((target - target.mean(axis=0)) ** 2).sum(axis=1).mean())
    )
    if source_rms <= 0:
        raise ValueError("source points are coincident")
    scale = target_rms / source_rms
    rotation = np.eye(3)
    translation = target.mean(axis=0) - scale * source.mean(axis=0)
    return scale, rotation, translation


def trimmed_similarity_icp(
    source,
    target,
    trim_keep=TRIM_KEEP,
    iterations=30,
    initial_rotation=None,
    initial_scale=None,
    initial_translation=None,
):
    """Refine a similarity from the source frame onto the target points.

    ``trim_keep`` is the fraction of nearest pairs retained. The paper does
    not publish this overlap. Pass ``initial_rotation`` to start from a stored
    SAM 3D pose. That path searches that one rotation. Omitting it falls back
    to ``rotation_seeds``, which is not the paper's initial pose.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if not 0 < float(trim_keep) <= 1:
        raise ValueError("trim_keep must be in (0, 1]")
    if initial_rotation is None:
        seeds = rotation_seeds()
        pose_source = "rotation_seeds"
    else:
        seed = np.asarray(initial_rotation, dtype=np.float64)
        if seed.shape != (3, 3) or not np.isfinite(seed).all():
            raise ValueError("initial_rotation must be a finite 3x3 matrix")
        seeds = [seed]
        pose_source = "stored_sam3d"
    if initial_translation is not None:
        initial_translation = np.asarray(initial_translation, dtype=np.float64).reshape(
            3
        )
        if not np.isfinite(initial_translation).all():
            raise ValueError("initial_translation must be finite")
    if initial_scale is not None:
        initial_scale = float(initial_scale)
        if not math.isfinite(initial_scale) or initial_scale <= 0.0:
            raise ValueError("initial_scale must be a positive finite number")
    tree = None
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(target)
    except Exception:
        tree = None
    base_scale, _, _ = initial_similarity(source, target)
    keep_count = max(3, round(float(trim_keep) * len(source)))
    best = None
    for seed_index, seed in enumerate(seeds):
        scale = base_scale if initial_scale is None else initial_scale
        rotation = np.asarray(seed, dtype=np.float64)
        if initial_translation is None:
            translation = target.mean(axis=0) - scale * (rotation @ source.mean(axis=0))
        else:
            translation = np.asarray(initial_translation, dtype=np.float64)
        started = np.asarray(rotation, dtype=np.float64).copy()
        for _ in range(int(iterations)):
            transformed = apply_similarity(source, scale, rotation, translation)
            distance, index = _nearest(transformed, target, tree)
            chosen = np.argsort(distance)[:keep_count]
            scale, rotation, translation = umeyama(
                source[chosen], target[index[chosen]]
            )
        transformed = apply_similarity(source, scale, rotation, translation)
        distance, index = _nearest(transformed, target, tree)
        chosen = np.argsort(distance)[:keep_count]
        score = float(np.median(distance[chosen]))
        if best is None or score < best["score"]:
            best = {
                "score": score,
                "seed_index": seed_index,
                "scale": float(scale),
                "rotation": np.asarray(rotation, dtype=np.float64),
                "translation": np.asarray(translation, dtype=np.float64),
                "initial_rotation": started,
                "distance": distance,
                "inlier_index": chosen,
                "target_index": index,
                "aligned": transformed,
            }
    if best is None:
        raise RuntimeError("trimmed similarity ICP produced no seed")
    inlier_distance = best["distance"][best["inlier_index"]]
    return {
        "scale": best["scale"],
        "rotation": best["rotation"],
        "translation": best["translation"],
        "rotation_seed_index": best["seed_index"],
        "trim_keep": float(trim_keep),
        "iterations": int(iterations),
        "inlier_count": int(keep_count),
        "source_count": len(source),
        "target_count": len(target),
        "distance": best["distance"],
        "inlier_index": best["inlier_index"],
        "target_index": best["target_index"],
        "inlier_distance": inlier_distance,
        "aligned": best["aligned"],
        "initial_rotation": best["initial_rotation"],
        "seeds_searched": len(seeds),
        "pose_source": pose_source,
    }


def measure_residuals(result, target, silhouette=None):
    """Measured residuals. No pass/fail cutoff is applied.

    ``normalized_median`` divides the median nearest distance by the target
    cloud's RMS size. ``coverage`` is the fraction of target points that are
    the neighbor of at least one trimmed inlier. ``silhouette`` stays null
    unless the caller passes a score from ``silhouette_agreement``. A null
    score is what an empty raster union produces. It is not a pass.
    """
    target = np.asarray(target, dtype=np.float64)
    distance = np.asarray(result["distance"], dtype=np.float64)
    inlier_distance = np.asarray(result["inlier_distance"], dtype=np.float64)
    target_rms = float(
        np.sqrt(((target - target.mean(axis=0)) ** 2).sum(axis=1).mean())
    )
    if target_rms <= 0:
        raise ValueError("target points are coincident")
    median = float(np.median(distance))
    p90 = float(np.percentile(distance, 90))
    covered = len(
        {int(index) for index in result["target_index"][result["inlier_index"]]}
    )
    return {
        "normalized_median": median / target_rms,
        "p90": p90,
        "median": median,
        "p90_inlier": float(np.percentile(inlier_distance, 90)),
        "median_inlier": float(np.median(inlier_distance)),
        "normalizer": target_rms,
        "normalizer_definition": "RMS distance of the target points from their centroid",
        "coverage": covered / float(len(target)),
        "coverage_definition": "fraction of target points chosen by at least one trimmed inlier",
        "silhouette": _recorded_silhouette(silhouette),
        "silhouette_definition": SILHOUETTE_DEFINITION,
        "thresholds_published": False,
    }


def _recorded_silhouette(silhouette):
    if silhouette is None:
        return None
    value = float(silhouette)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("silhouette must be a finite number in [0, 1]")
    return value


def _edge(ax, ay, bx, by, px, py):
    return (px - ax) * (by - ay) - (py - ay) * (bx - ax)


def rasterize_orthographic_silhouette(
    vertices, faces, width, height, origin=(0.0, 0.0), pixel_size=1.0
):
    """Binary mask of a mesh under an orthographic camera looking along +Z.

    Pixel ``(row, col)`` is covered when its center falls inside a triangle.
    The center is ``origin + (col + 0.5, row + 0.5) * pixel_size`` in world
    x, y. This is a CPU raster for a silhouette score. It is not a GPU render
    of a SAM 3D GLB.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must be Nx3")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be Mx3")
    if faces.size and (int(faces.min()) < 0 or int(faces.max()) >= len(vertices)):
        raise ValueError("face index out of range")
    width = int(width)
    height = int(height)
    if width < 1 or height < 1:
        raise ValueError("silhouette raster must have a positive size")
    pixel = float(pixel_size)
    if not math.isfinite(pixel) or pixel <= 0.0:
        raise ValueError("pixel_size must be positive")
    origin = np.asarray(origin, dtype=np.float64).reshape(2)
    cols = origin[0] + (np.arange(width, dtype=np.float64) + 0.5) * pixel
    rows = origin[1] + (np.arange(height, dtype=np.float64) + 0.5) * pixel
    grid_x, grid_y = np.meshgrid(cols, rows)
    mask = np.zeros((height, width), dtype=bool)
    for face in faces:
        xs = vertices[face, 0]
        ys = vertices[face, 1]
        area = float(_edge(xs[0], ys[0], xs[1], ys[1], xs[2], ys[2]))
        if not math.isfinite(area) or abs(area) < 1e-15:
            continue
        w0 = _edge(xs[1], ys[1], xs[2], ys[2], grid_x, grid_y)
        w1 = _edge(xs[2], ys[2], xs[0], ys[0], grid_x, grid_y)
        w2 = _edge(xs[0], ys[0], xs[1], ys[1], grid_x, grid_y)
        if area < 0.0:
            w0, w1, w2 = -w0, -w1, -w2
        mask |= (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
    return mask


def silhouette_iou(rendered, observed):
    """Jaccard index of two masks. An empty union returns None.

    None is the first Fruits residual's outcome: the raster union was empty,
    so there was no score to record. Zero is a real score for disjoint masks.
    """
    rendered = np.asarray(rendered, dtype=bool)
    observed = np.asarray(observed, dtype=bool)
    if rendered.ndim != 2 or rendered.shape != observed.shape:
        raise ValueError("silhouette masks must be matching 2D arrays")
    intersection = int(np.logical_and(rendered, observed).sum())
    union = int(np.logical_or(rendered, observed).sum())
    if union == 0:
        return None
    return intersection / float(union)


def silhouette_agreement(rendered_masks, observed_masks):
    """Mean per-view Jaccard index. See ``SILHOUETTE_DEFINITION``.

    Views with an empty union are counted and left out of the mean. When
    every view is empty the score is None.
    """
    rendered_masks = list(rendered_masks)
    observed_masks = list(observed_masks)
    if not rendered_masks or len(rendered_masks) != len(observed_masks):
        raise ValueError("silhouette needs one observed mask per rendered mask")
    scores = []
    empty = 0
    for rendered, observed in zip(rendered_masks, observed_masks, strict=True):
        score = silhouette_iou(rendered, observed)
        if score is None:
            empty += 1
        else:
            scores.append(score)
    return {
        "silhouette": None if not scores else float(np.mean(scores)),
        "views": len(scores),
        "empty_views": empty,
        "definition": SILHOUETTE_DEFINITION,
    }


def _vector3(value, label):
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be 3 finite numbers")
    return array


def resolve_sam3d_rotation(payload):
    """Rotation ICP should start from.

    ``rotation`` is a Rodrigues vector. ``rotation_matrix`` and
    ``quaternion_xyzw`` are converted into that vector. The scene frame stores
    quaternions XYZW. None of those fields is filled with identity: a sidecar
    that matches its GLB and still has no rotation raises
    ``PoseSidecarRotationMissingError``.
    """
    if not isinstance(payload, dict):
        raise ValueError("SAM 3D pose must be a JSON object")
    if payload.get("rotation") is not None:
        vector = _vector3(payload["rotation"], "SAM 3D rotation vector")
        return vector, rotation_matrix_from_vector(vector), "rotation"
    if payload.get("rotation_matrix") is not None:
        matrix = np.asarray(payload["rotation_matrix"], dtype=np.float64)
        vector = rotation_vector_from_matrix(matrix)
        return vector, rotation_matrix_from_vector(vector), "rotation_matrix"
    if payload.get("quaternion_xyzw") is not None:
        matrix = quaternion_xyzw_to_matrix(payload["quaternion_xyzw"])
        vector = rotation_vector_from_matrix(matrix)
        return vector, rotation_matrix_from_vector(vector), "quaternion_xyzw"
    raise PoseSidecarRotationMissingError(
        "SAM 3D pose sidecar has no rotation, rotation_matrix, or quaternion_xyzw"
    )


def file_sha256(path):
    """SHA-256 hex digest of a file. Used to bind a pose sidecar to its GLB."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _recorded_glb_sha256(payload):
    if "glb_sha256" not in payload or payload.get("glb_sha256") is None:
        raise PoseSidecarHashMissingError("pose sidecar is missing glb_sha256")
    recorded = payload["glb_sha256"]
    if not isinstance(recorded, str) or not recorded.strip():
        raise PoseSidecarHashMissingError("pose sidecar glb_sha256 is empty")
    return recorded.strip().lower()


def _require_matching_glb(payload, glb_path):
    recorded = _recorded_glb_sha256(payload)
    glb = Path(glb_path)
    if not glb.is_file():
        raise PoseSidecarHashMismatchError(f"GLB is not at {glb}")
    actual = file_sha256(glb)
    if actual != recorded:
        raise PoseSidecarHashMismatchError(
            f"pose sidecar glb_sha256 {recorded} does not match {actual}"
        )
    return actual


def _optional_scale(payload):
    if payload.get("scale") is None:
        return None, "centroid_rms"
    scale = float(payload["scale"])
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("SAM 3D scale must be a positive finite number")
    return scale, "stored_sam3d"


def _optional_translation(payload):
    if payload.get("translation") is None:
        return None, "centroid"
    return _vector3(payload["translation"], "SAM 3D translation"), "stored_sam3d"


def load_stored_sam3d_pose(path, glb_path):
    """Read a SAM 3D pose sidecar bound to ``glb_path`` by SHA-256.

    A missing sidecar, a missing ``glb_sha256``, or a hash that is not the
    GLB's digest raises a named error. A matching sidecar with no rotation,
    matrix, or quaternion raises ``PoseSidecarRotationMissingError``. None of
    those paths writes an identity rotation. A matrix or XYZW quaternion is
    converted to a Rodrigues vector and stored, because that vector was
    derived from a rotation the sidecar already held.
    """
    pose_path = Path(path)
    if not pose_path.is_file():
        raise PoseSidecarMissingError(
            f"SAM 3D pose sidecar is not stored at {pose_path}"
        )
    payload = json.loads(pose_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("SAM 3D pose must be a JSON object")
    digest = _require_matching_glb(payload, glb_path)
    vector, matrix, source = resolve_sam3d_rotation(payload)
    stored = False
    if "rotation" not in payload or payload.get("rotation") is None:
        payload["rotation"] = [float(item) for item in vector]
        payload["rotation_source"] = source
        pose_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        stored = True
    elif payload.get("rotation_source"):
        source = str(payload["rotation_source"])
    scale, scale_source = _optional_scale(payload)
    translation, translation_source = _optional_translation(payload)
    return {
        "rotation_vector": vector,
        "rotation": matrix,
        "scale": scale,
        "translation": translation,
        "scale_source": scale_source,
        "translation_source": translation_source,
        "rotation_source": source,
        "rotation_stored": stored,
        "path": str(pose_path),
        "glb_path": str(Path(glb_path)),
        "glb_sha256": digest,
    }


def _score_silhouette(result, rendered_masks, observed_masks, mesh):
    """Silhouette of the refined similarity, not the pre-ICP vertices.

    The gate judges the pose ICP finished on. A mesh is rasterized only after
    ``apply_similarity`` with that scale, rotation, and translation. Supplied
    masks cannot stand in for the mesh: they may be a render of the export
    frame, and using them would skip the refined transform.
    """
    if mesh is not None:
        scored = dict(mesh)
        scored["vertices"] = apply_similarity(
            mesh["vertices"],
            result["scale"],
            result["rotation"],
            result["translation"],
        )
        rendered = _rendered_masks(None, observed_masks, scored)
    else:
        rendered = _rendered_masks(rendered_masks, observed_masks, None)
    return silhouette_agreement(rendered, observed_masks)


def _rendered_masks(rendered_masks, observed_masks, mesh):
    if rendered_masks is not None:
        return list(rendered_masks)
    if mesh is None:
        raise ValueError("silhouette needs rendered masks or a mesh")
    observed = list(observed_masks or [])
    if len(observed) != 1:
        raise ValueError(
            "an orthographic mesh silhouette is one view; pass rendered masks for more"
        )
    sample = np.asarray(observed[0])
    if sample.ndim != 2:
        raise ValueError("observed mask must be 2D")
    height = int(mesh["height"]) if mesh.get("height") is not None else sample.shape[0]
    width = int(mesh["width"]) if mesh.get("width") is not None else sample.shape[1]
    rendered = rasterize_orthographic_silhouette(
        mesh["vertices"],
        mesh["faces"],
        width,
        height,
        origin=mesh.get("origin", (0.0, 0.0)),
        pixel_size=mesh.get("pixel_size", 1.0),
    )
    return [rendered]


def _pose_initialization(pose, result):
    translation = pose["translation"]
    return {
        "source": "sidecar",
        "sidecar_path": pose["path"],
        "glb_path": pose["glb_path"],
        "glb_sha256": pose["glb_sha256"],
        "rotation": [float(item) for item in pose["rotation_vector"]],
        "rotation_source": pose["rotation_source"],
        "rotation_stored": bool(pose["rotation_stored"]),
        "seeds_searched": int(result["seeds_searched"]),
        "scale": None if pose["scale"] is None else float(pose["scale"]),
        "scale_source": pose["scale_source"],
        "translation": None
        if translation is None
        else [float(item) for item in translation],
        "translation_source": pose["translation_source"],
        "scored_pose": "post_icp",
        "silhouette_transform": {
            "scale": float(result["scale"]),
            "rotation": np.asarray(result["rotation"], dtype=np.float64).tolist(),
            "translation": np.asarray(result["translation"], dtype=np.float64).tolist(),
        },
    }


def _gate_reason(decision, residuals):
    """Accept, ``silhouette_undefined``, or the below-minimum residual names.

    A null silhouette is the empty-raster case. A numeric silhouette that
    misses the minimum stays on the ordinary reject line.
    """
    if decision["decision"] == "accept":
        return "ours_iteration_1 accept"
    if residuals.get("silhouette") is None:
        return SILHOUETTE_UNDEFINED_REASON
    failed = ", ".join(decision.get("failed") or [])
    return f"ours_iteration_1 reject: {failed}"


def _is_finite_number(value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return False
    return math.isfinite(float(value))


def icp_step_exit_code(residuals, ledger_entry):
    """Exit code for the separate ICP step.

    0 when the ledger accepted or rejected and normalized median, p90, and
    coverage are finite numbers. Silhouette must be a finite number, except
    the ruled null whose ledger reason is ``silhouette_undefined``. Anything
    else, including a missing ledger line, is 1.
    """
    if not isinstance(ledger_entry, dict):
        return 1
    if ledger_entry.get("decision") not in {"accept", "reject"}:
        return 1
    if not isinstance(residuals, dict):
        return 1
    for name in ("normalized_median", "p90", "coverage"):
        if not _is_finite_number(residuals.get(name)):
            return 1
    silhouette = residuals.get("silhouette")
    if silhouette is None:
        if ledger_entry.get("reason") != SILHOUETTE_UNDEFINED_REASON:
            return 1
        return 0
    if not _is_finite_number(silhouette):
        return 1
    return 0


def write_alignment_residual(
    source,
    target,
    pose_path,
    out_path,
    rendered_masks=None,
    observed_masks=None,
    mesh=None,
    ledger_dir=None,
    glb_path=None,
):
    """Align from the stored SAM 3D pose and write the residual JSON.

    The pose sidecar has to exist and its ``glb_sha256`` has to be the SHA-256
    of ``glb_path`` before ICP runs. Silhouette is measured after ICP, on the
    refined scale, rotation, and translation. An empty union leaves silhouette
    null and the ledger reason is ``silhouette_undefined``. A measured 0
    rejects on the ordinary below-minimum line. This function does not change
    the gate's cutoffs.
    """
    if glb_path is None:
        raise PoseSidecarHashMissingError(
            "ICP needs the GLB whose hash the sidecar records"
        )
    pose = load_stored_sam3d_pose(pose_path, glb_path)
    if observed_masks is None:
        raise ValueError("silhouette needs observed masks")
    result = trimmed_similarity_icp(
        source,
        target,
        trim_keep=TRIM_KEEP,
        initial_rotation=pose["rotation"],
        initial_scale=pose["scale"],
        initial_translation=pose["translation"],
    )
    if result["seeds_searched"] != 1 or result["pose_source"] != "stored_sam3d":
        raise RuntimeError("ICP did not start from the stored SAM 3D pose")
    agreement = _score_silhouette(result, rendered_masks, observed_masks, mesh)
    residuals = measure_residuals(result, target, silhouette=agreement["silhouette"])
    pose_info = _pose_initialization(pose, result)
    body = dict(residuals)
    body["silhouette_views"] = agreement["views"]
    body["silhouette_empty_views"] = agreement["empty_views"]
    body["pose_initialization"] = pose_info
    body["scale"] = result["scale"]
    body["rotation"] = result["rotation"].tolist()
    body["translation"] = result["translation"].tolist()
    body["trim_keep"] = result["trim_keep"]
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    decision = accept_alignment(
        {name: residuals[name] for name in RESIDUAL_NAMES},
        "ours_iteration_1",
    )
    constants = decision.get("constants")
    if decision.get("gate") != "ours_iteration_1" or not isinstance(constants, dict):
        raise RuntimeError("ours_iteration_1 did not report its constants")
    reason = _gate_reason(decision, residuals)
    entry = None
    if ledger_dir is not None:
        entry = append_step(
            ledger_dir,
            "icp",
            {
                "ok": decision["accepted"],
                "artifact": str(destination),
                "reason": reason,
                "detail": {
                    "gate": decision["gate"],
                    "pose_initialization": pose_info,
                    "alignment_constants": constants,
                    "residuals": {name: residuals[name] for name in RESIDUAL_NAMES},
                },
            },
        )
    return {
        "path": str(destination),
        "residuals": residuals,
        "pose_initialization": pose_info,
        "decision": decision,
        "reason": reason,
        "ledger_entry": entry,
    }


def run_icp_step(
    source,
    target,
    pose_path,
    glb_path,
    out_path,
    rendered_masks=None,
    observed_masks=None,
    mesh=None,
    ledger_dir=None,
):
    """Entry point for the separate ICP step. Returns a process exit code.

    A gate accept or reject, including ``silhouette_undefined``, is 0 and the
    ledger records that decision. A crash, a pose-sidecar error, or a missing
    or non-numeric measure is 1. This step is not the Fruits smoke.
    """
    try:
        written = write_alignment_residual(
            source,
            target,
            pose_path,
            out_path,
            rendered_masks=rendered_masks,
            observed_masks=observed_masks,
            mesh=mesh,
            ledger_dir=ledger_dir,
            glb_path=glb_path,
        )
    except Exception:
        return 1
    return icp_step_exit_code(written["residuals"], written["ledger_entry"])


def main(argv=None):
    """``python -m sceneagent.similarity_icp``. Same exit codes as ``run_icp_step``."""
    parser = argparse.ArgumentParser(prog="python -m sceneagent.similarity_icp")
    parser.add_argument("--pose", required=True)
    parser.add_argument("--glb", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--rendered-mask", required=True)
    parser.add_argument("--observed-mask", required=True)
    try:
        args = parser.parse_args(argv)
        source = np.load(args.source)
        target = np.load(args.target)
        rendered = np.load(args.rendered_mask)
        observed = np.load(args.observed_mask)
    except SystemExit as exc:
        code = exc.code
        if code in (None, 0):
            return 1
        return code if isinstance(code, int) else 1
    except Exception:
        return 1
    return run_icp_step(
        source,
        target,
        args.pose,
        args.glb,
        args.out,
        rendered_masks=[rendered],
        observed_masks=[observed],
        ledger_dir=args.ledger,
    )


if __name__ == "__main__":
    sys.exit(main())
