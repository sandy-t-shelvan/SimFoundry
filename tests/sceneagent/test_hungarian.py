"""Optimal object matching for reconstruction scores.

The phase-1 matcher enumerated column permutations and refused a side longer
than 8. These tests lock three facts: a 20-object scene has a known optimal
matching, a rectangular score matches the shorter side, and every matrix that
enumerator accepted still produces the same pairs. The oracle below is that
enumerator, copied so this file does not call the function it is checking.
"""

import itertools

import numpy as np
import pytest

from sceneagent.metrics import _hungarian, score_reconstruction

# Fixed so the expected pairs can be read off the test, not drawn again.
_TWENTY = (
    3, 0, 11, 7, 19, 2, 15, 8, 4, 13,
    1, 17, 9, 12, 5, 18, 6, 14, 16, 10,
)


def _phase1_enumeration(cost):
    """Matcher body from before this change, for shapes it accepted.

    A side longer than 8 raised. More rows than columns returned None, because
    ``permutations`` cannot draw more columns than exist. Neither case belongs
    in the "unchanged pairs" check.
    """
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


def _pair_cost(cost, pairs):
    rows = [i for i, _ in pairs]
    cols = [j for _, j in pairs]
    assert rows == sorted(rows)
    assert len(rows) == len(set(rows))
    assert len(cols) == len(set(cols))
    assert len(pairs) == min(cost.shape)
    assert all(0 <= i < cost.shape[0] and 0 <= j < cost.shape[1] for i, j in pairs)
    return float(sum(cost[i, j] for i, j in pairs))


def _brute_minimum(cost):
    """Minimum assignment cost. Feasible only for the small shapes below."""
    cost = np.asarray(cost, dtype=float)
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return 0.0
    best = None
    if n_rows <= n_cols:
        for choice in itertools.permutations(range(n_cols), n_rows):
            total = float(sum(cost[i, choice[i]] for i in range(n_rows)))
            if best is None or total < best:
                best = total
    else:
        for chosen in itertools.combinations(range(n_rows), n_cols):
            for perm in itertools.permutations(chosen):
                total = float(sum(cost[perm[j], j] for j in range(n_cols)))
                if best is None or total < best:
                    best = total
    return best


def _cloud(x):
    return np.array([[float(x), 0.0, 0.0], [float(x), 0.001, 0.0]])


def test_twenty_objects_use_the_planted_matching():
    perm = _TWENTY
    assert sorted(perm) == list(range(20))
    cost = np.full((20, 20), 1.0)
    for row, col in enumerate(perm):
        cost[row, col] = 0.0
    assert _hungarian(cost) == [(row, col) for row, col in enumerate(perm)]


def test_twenty_objects_prefer_the_zero_cycle_over_the_diagonal():
    # The diagonal costs 4 each. The cycle of zeros is the unique minimum.
    # A matcher that keeps the identity, or that stops at 8, fails this.
    n = 20
    cost = np.full((n, n), 9.0)
    for i in range(n):
        cost[i, i] = 4.0
        cost[i, (i + 1) % n] = 0.0
    assert _hungarian(cost) == [(i, (i + 1) % n) for i in range(n)]


def test_nine_objects_are_matched_instead_of_raising():
    cost = np.full((9, 9), 1.0)
    np.fill_diagonal(cost, 0.0)
    assert _hungarian(cost) == [(i, i) for i in range(9)]


def test_non_finite_cost_past_the_enumerator_raises():
    cost = np.zeros((9, 9))
    cost[3, 4] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _hungarian(cost)


def test_more_predictions_than_ground_truth():
    # 6 references, 12 predictions. One zero per row, in distinct columns.
    planted = (0, 2, 5, 7, 9, 11)
    cost = np.ones((6, 12))
    for row, col in enumerate(planted):
        cost[row, col] = 0.0
    assert _hungarian(cost) == [(row, col) for row, col in enumerate(planted)]

    # Both sides are under the old cap, so this one still takes the enumerator.
    small = np.array([
        [4.0, 0.0, 3.0, 3.0],
        [5.0, 5.0, 0.0, 2.0],
        [0.0, 4.0, 4.0, 1.0],
    ])
    assert _hungarian(small) == [(0, 1), (1, 2), (2, 0)]
    assert _hungarian(small) == _phase1_enumeration(small)


def test_more_ground_truth_than_predictions():
    planted_rows = (1, 3, 4, 7, 8, 11)
    cost = np.ones((12, 6))
    for col, row in enumerate(planted_rows):
        cost[row, col] = 0.0
    assert _hungarian(cost) == [(row, col) for col, row in enumerate(planted_rows)]

    # 5 references and 3 predictions, under the old cap. The enumerator returned
    # None: it asked for a permutation longer than the column list.
    small = np.array([
        [9.0, 9.0, 9.0],
        [0.0, 4.0, 4.0],
        [4.0, 0.0, 4.0],
        [4.0, 4.0, 0.0],
        [9.0, 9.0, 9.0],
    ])
    assert _phase1_enumeration(small) is None
    assert _hungarian(small) == [(1, 0), (2, 1), (3, 2)]


def test_at_most_eight_matches_phase1_enumeration():
    rng = np.random.default_rng(8)
    # Both permutations cost 0, so the first one in permutation order is kept:
    # columns (0, 1), which is the identity pairing.
    tied = np.zeros((2, 2))
    assert _hungarian(tied) == [(0, 0), (1, 1)]
    assert _hungarian(tied) == _phase1_enumeration(tied)
    # The lexicographically first permutation costs 10. The later one costs 0.
    off = np.array([[5.0, 0.0], [0.0, 5.0]])
    assert _hungarian(off) == [(0, 1), (1, 0)]
    assert _hungarian(off) == _phase1_enumeration(off)

    for n_rows in range(0, 9):
        for n_cols in range(0, 9):
            if n_rows > n_cols > 0:
                continue
            samples = [np.zeros((n_rows, n_cols))]
            if n_rows and n_cols:
                samples.append(rng.integers(0, 6, size=(n_rows, n_cols)))
                samples.append(rng.normal(size=(n_rows, n_cols)))
            for sample in samples:
                cost = np.asarray(sample, dtype=float)
                assert _hungarian(cost) == _phase1_enumeration(cost)


def test_new_shapes_match_the_brute_force_minimum():
    rng = np.random.default_rng(20)
    shapes = (
        (1, 9),
        (9, 1),
        (3, 10),
        (10, 3),
        (5, 9),
        (9, 5),
        (8, 9),
        (9, 8),
        (9, 9),
        (2, 15),
        (15, 2),
    )
    matrices = [rng.integers(0, 11, size=shape).astype(float) for shape in shapes]
    matrices.append(rng.integers(-5, 6, size=(4, 9)).astype(float))
    for cost in matrices:
        pairs = _hungarian(cost)
        assert _pair_cost(cost, pairs) == _brute_minimum(cost)


def test_score_reconstruction_on_twenty_objects():
    perm = _TWENTY
    reference = {f"g{i}": _cloud(i) for i in range(20)}
    delivered = {f"p{j}": _cloud(perm[j]) for j in range(20)}
    scored = score_reconstruction(delivered, reference)
    inverse = {perm[j]: j for j in range(20)}
    by_ref = {match["reference"]: match for match in scored["matches"]}
    assert len(scored["matches"]) == 20
    for i in range(20):
        match = by_ref[f"g{i}"]
        assert match["delivered"] == f"p{inverse[i]}"
        assert match["chamfer_m"] == pytest.approx(0.0)
        assert match["f1"] == pytest.approx(1.0)
    assert scored["f1_mean"] == pytest.approx(1.0)
    assert scored["chamfer_mean_matched_m"] == pytest.approx(0.0)


def test_score_reconstruction_leaves_extra_objects_unmatched():
    reference = {f"g{i}": _cloud(i) for i in range(10)}
    delivered = {f"p{k}": _cloud(x) for k, x in enumerate((1, 4, 6, 8))}
    scored = score_reconstruction(delivered, reference)
    matched = {match["reference"] for match in scored["matches"]}
    assert matched == {"g1", "g4", "g6", "g8"}
    assert scored["f1_mean"] == pytest.approx(0.4)
    assert scored["chamfer_mean_matched_m"] == pytest.approx(0.0)
    for name, f1 in scored["f1_by_reference"].items():
        assert f1 == pytest.approx(1.0 if name in matched else 0.0)

    reference = {f"g{i}": _cloud(i) for i in range(4)}
    delivered = {f"p{i}": _cloud(i) for i in range(4)}
    delivered["far"] = _cloud(100)
    for i in range(7):
        delivered[f"extra{i}"] = _cloud(50 + i)
    scored = score_reconstruction(delivered, reference)
    matched = {match["reference"] for match in scored["matches"]}
    assert matched == {"g0", "g1", "g2", "g3"}
    assert len(scored["matches"]) == 4
    assert scored["f1_mean"] == pytest.approx(1.0)
    assert scored["chamfer_mean_matched_m"] == pytest.approx(0.0)
    assert all(row["f1"] == pytest.approx(1.0) for row in scored["matches"])
