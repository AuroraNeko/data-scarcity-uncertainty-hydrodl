# Scoring fields

The scientific source/results archive supplies machine-readable results under
`results/tables/`. All log-flow interval and distribution scores use the
de-standardized log1p(mm/day) coordinate.

- `distribution_scores.csv` and `scoring_summary.json` use `pooled` for
  all observed basin-days. This is the value in the manuscript IQS table.
- `low` and `high` use observed test Q0.33/Q0.67 separately within each
  basin. Ties stay in the lower group. These are retrospective groups.
  `low_n` and `high_n` give their actual counts.
- `absolute_pooled_low` and `absolute_pooled_high` retain the older
  absolute all-basin threshold diagnostic. Its thresholds and the grouping
  distinction are recorded in `scoring_summary.json.group_definitions`.
- `submission_diagnostics.json` records original USGS quality flags,
  test-only A-flag filtering of fixed predictors, calibrated absolute high
  predicted-flow coverage intervals, direct MC Winkler contrasts and an
  upstream area-list exclusion. Seeds, draw counts and conditioning are
  explicit. `experiments/analyze/submission_diagnostics.py` regenerates it
  and the bilingual S25 fragments using raw flag files and frozen arrays.

Use `python experiments/prepare/verify_submission_snapshot.py` for the
small offline verification that needs only the delivered arrays and metadata.
Training reuse additionally requires verified current processed CSV inputs.
