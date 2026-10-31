# Reproducibility Guide

`run_pipeline.py` is the single entry point; `python run_pipeline.py --list`
shows the nine stages (prepare → train primary → train benchmark → correct →
train sensitivity → evaluate → audit → analyze → figures). Dependencies flow
one way: `run_pipeline.py → experiments/* → src/*`; scripts never import each
other and shared logic lives in `src/`.

## Environment

Python 3.11, PyTorch 2.11.0+cu128 (reference GPU: RTX 5060 Ti, 8 GB).
`pip install -r requirements.txt`; the pinned reference environment is
`requirements-reference-cu128.txt`. Stages 2, 3 and 5 retrain models and
require CUDA. Stage 6 reloads trained checkpoints for inference and also
finishes on CPU. Stages 1, 4, 7, 8 and 9 are table-driven and run on CPU.

## Data

`python download_camels.py` fetches the CAMELS-US inputs (3.4 GB zip, about
14 GB extracted); `python src/data/data_preprocessing.py` builds the
processed per-basin files under `data/processed/`. The processed-data audit
(stage 7, `experiments/prepare/audit_processed_data.py`) re-checks all 671
processed files for date-range and continuity violations, duplicate dates,
sentinel values, mask/target consistency, and expected static-attribute
missingness. Missing flow is always masked and excluded from normalization,
losses, calibration and evaluation.

## Determinism and verification

- Training uses fixed seeds (basin-selection seed 42; model seeds 42/123/456)
  and fixed final checkpoints; the sensitivity queue verifies protocol
  identities and artifact hashes before reusing anything.
- The evaluation protocol (`src/evaluation/protocol.py`) uses exact
  non-interpolated split-conformal ranks, within-basin 0.33/0.67 group cuts
  with ties in the lower group, and empty-set midpoint projection.
- Finite-grid distributional scores are integrated quantile scores, not full
  CRPS; the Gaussian analytic score is full CRPS. Legacy artifact ids `umal`
  and `encqr` denote a Gaussian location-scale quantile head and static
  memberwise split-CQR averaging, not the cited UMAL/EnCQR algorithms.
- Resampling intervals condition on the fitted models and the historical
  periods; model-seed standard deviations and basin/cluster uncertainty answer
  different questions and are reported separately.

## Archived-field caveats

A few field names in archived JSON tables predate the final protocol. The
bytes are kept unchanged so archived hashes stay valid; readers should know:

- `coverage_low_flow` / `coverage_normal_flow` / `coverage_high_flow` in the
  per-run `raw_interval_metrics` / `calibrated_interval_metrics` blocks (and
  in `corrected_full_671/method_comparison.json`) pool all test samples of a
  regime across basins. The paper's conditional-coverage numbers use the
  per-basin regime paths recomputed by
  `experiments/analyze/conditional_robustness.py` and
  `experiments/evaluate/coverage_groups.py`.
- `cluster_bootstrap_ci95` fields hold a basin-level bootstrap, not a
  HUC2-cluster bootstrap.
- Timing entries in `results/tables/benchmark_671/s18_reliability.json`
  reflect the archived workstation; reruns will differ and this is expected.

## Outputs

Everything regenerates under `results/` (gitignored): `results/tables/`
(compact summaries), `results/figures/`, `results/predictions/`,
`results/checkpoints/`, `results/cache/`, `results/logs/`. Nothing under
`results/` is tracked by git, so a git tag pins code only. The complete
frozen artifact set (compact tables, per-sample prediction archives,
checkpoints, figures) is packaged as three release archives with a SHA-256
manifest; they are attached to the repository's GitHub release when one is
published, and no public DOI is issued by this repository automatically
(publishing is an author action).
