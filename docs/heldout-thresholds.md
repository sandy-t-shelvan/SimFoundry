# Held-out alignment thresholds

ours, not published in the paper.

The paper states four tests and fixed thresholds, and it prints no numbers. The values below are a separate profile, `ours_heldout_v1`. They do not replace `ours_iteration_1`, and they are not the placement gate.

## Set

The labelled candidates are the placed objects in the checked-in GraspClutter6D bin score files (`docs/results/bin_000091_score_*.json`). A row needs all four residuals and a matched F1. That is 42 candidates and 14 object ids, listed in `docs/results/heldout_candidates.csv`. The set is one bin. Fourteen objects is a small set.

A candidate is good when its F1 at 1 cm is above the median F1 of the matched objects in that same run.

## Split

Seed 0, split by object id so an object is never in both halves. Derivation ids: 0, 2, 3, 4, 5, 7, 13 (21 candidates). Held-out ids: 1, 6, 8, 9, 10, 11, 12 (21 candidates).

## Method

On the derivation half only, each residual gets the cutoff with the highest balanced accuracy on the unique values in that half. A tie takes the stricter cutoff (a lower maximum, or a higher minimum). The gate is the AND of the four tests. The held-out half is scored after that choice and does not move the cutoffs.

| Residual | Cutoff |
|---|---|
| normalized median maximum | 0.0784567069559569 |
| 90th-percentile maximum | 0.15229201952511362 |
| coverage minimum | 0.4732394366197183 |
| silhouette minimum | 0.08971836478763888 |

On the derivation half the AND gate accepts 1 of 21 candidates (precision 1, recall 0.111). On the held-out half it accepts 0 of 21 (12 of those 21 are good). The zero is reported as the result. The cutoffs were not loosened to raise it.

`scripts/derive_heldout_thresholds.py` rebuilds the CSV and these four numbers from the score files.

## Fruits residuals

`FRUITS_ICP_RESIDUAL` under both profiles. Both record `thresholds_published: false` when the profile is named. Silhouette on that residual is missing, so it fails either gate.

| Profile | Decision | Failed |
|---|---|---|
| `ours_iteration_1` | reject | coverage, silhouette |
| `ours_heldout_v1` | reject | p90, coverage, silhouette |
