# Scores kept out of the held-out gate

`sceneagent/heldout_thresholds.py` derives `ours_heldout_v1` from every
`docs/results/bin_000091_score_*.json`, and the profile test checks the published
limits against a fresh derivation. A score committed there moves the gate. Scores
kept as evidence only, recorded after the profile was published, live here; the
glob is not recursive.

- `..._60254439_kappa2_rescale.json`: bin 000091 on box 60254439 (run 43298215329),
  the KAN-15 export, settle and inspect chain.
