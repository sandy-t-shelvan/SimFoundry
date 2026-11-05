"""Reconstruction and evaluation statistics named by SceneAgent.

Chamfer distance and F1 at 1 cm follow equations (4) and (5) in section 4.1.
Points are in meters. The paper's threshold tau is 1 cm, so the default tau
is 0.01 m. F1 is 0 when precision + recall is 0, and a missing object scores 0
when a scene is summarized the way GraspClutter6D is summarized in that section.

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


# 8! column orders is what the phase-1 matcher enumerated. One more object makes
# that impractical, and the GraspClutter6D bin has 20, so anything larger uses
# the Hungarian algorithm below. Matrices this size still take the enumerator
# when every row can be matched: it is the lexicographically first minimum, and
# a later solver would be free to pick a different optimal assignment on a tie.
_ENUMERATION_MAX = 8


def _enumerate_assignment(cost):
    """Lexicographically first minimum-cost assignment of every row.

    ``itertools.permutations`` yields column tuples in lexicographic order and
    the first minimum is kept. That is the phase-1 result for at most 8 objects
    on a side when there is a column for every row.
    """
    n_rows, n_cols = cost.shape
    best = None
    best_cost = None
    columns = range(n_cols)
    for choice in itertools.permutations(columns, n_rows):
        total = float(sum(cost[i, choice[i]] for i in range(n_rows)))
        if best_cost is None or total < best_cost:
            best_cost = total
            best = [(i, choice[i]) for i in range(n_rows)]
    return best


def _assign_every_row(cost):
    """Column chosen for each row. ``cost`` has shape (n, m) with n <= m.

    Kuhn-Munkres, in the successive-shortest-path form. SciPy is not imported:
    the phase-1 CI job does not install it, and a reconstruction score still
    has to run there. The result is a minimum-cost assignment.
    """
    n_rows, n_cols = cost.shape
    if n_rows > n_cols:
        raise ValueError("assignment expects at least as many columns as rows")
    if not np.isfinite(cost).all():
        # A NaN reduced cost never updates the Dijkstra labels, and the
        # augmenting-path loop would not terminate.
        raise ValueError("assignment costs must be finite")
    # Column 0 is a dummy. u[1..n] and v[1..m] are the potentials, and p[j] is
    # the row matched to column j. Costs stay 0-based; this layout is the usual
    # 1-based presentation of the algorithm.
    u = [0.0] * (n_rows + 1)
    v = [0.0] * (n_cols + 1)
    p = [0] * (n_cols + 1)
    way = [0] * (n_cols + 1)
    inf = math.inf
    for i in range(1, n_rows + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (n_cols + 1)
        used = [False] * (n_cols + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0
            for j in range(1, n_cols + 1):
                if used[j]:
                    continue
                cur = float(cost[i0 - 1, j - 1]) - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n_cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assigned = [-1] * n_rows
    for j in range(1, n_cols + 1):
        if p[j] != 0:
            assigned[p[j] - 1] = j - 1
    return assigned


def _hungarian(cost):
    """Minimum-cost assignment of rows to distinct columns.

    The shorter side is fully matched: every row when there are at least as
    many columns as rows, otherwise every column. Pairs are sorted by row.
    An unmatched row is omitted, which is how ``score_reconstruction`` leaves
    that reference's F1 at 0.

    At most 8 rows and 8 columns, with a column for every row, still returns
    the phase-1 enumeration, including its tie break. Every other shape uses
    ``_assign_every_row``, which is optimal for the same sum of costs.
    """
    cost = np.asarray(cost, dtype=float)
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return []
    if (
        n_rows <= n_cols
        and n_rows <= _ENUMERATION_MAX
        and n_cols <= _ENUMERATION_MAX
    ):
        return _enumerate_assignment(cost)
    if n_rows <= n_cols:
        columns = _assign_every_row(cost)
        return [(i, columns[i]) for i in range(n_rows)]
    # More rows than columns. Match every column; the enumerator cannot, because
    # a permutation cannot be longer than the column list, and it used to return
    # None even under the 8-object cap. In a reconstruction score those extra
    # rows are the unmatched reference objects.
    rows = _assign_every_row(np.ascontiguousarray(cost.T))
    return sorted((rows[j], j) for j in range(n_cols))


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
