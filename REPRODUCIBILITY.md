# Reproducibility Guide

`reproduce.py` is the single entry point; `python reproduce.py --list` shows
the nine stages (prepare → train primary → train benchmark → correct → train
sensitivity → evaluate → analyze → audit → figures). Dependencies flow one
way: `reproduce.py → experiments/* → src/*`; scripts never import each other
and shared logic lives in `src/`. The frozen submission-round snapshot is git
tag `JOH-revision-20261030`.

## Environment

Python 3.11, PyTorch 2.11.0+cu128 (reference GPU: RTX 5060 Ti, 8 GB).
`pip install -r requirements.txt`; the pinned reference environment is
`requirements-reference-cu128.txt`. Stages 2, 3 and 5 require CUDA; the
remaining stages run on CPU against archived prediction arrays.

## Data

`python download_camels.py` fetches CAMELS-US (~24 GB); `python
src/data/data_preprocessing.py` builds the processed per-basin files under
`data/processed/`. The raw-source forcing audit (stage 8) found no missing or
sentinel values and no altered values in the common 1980–2008 range; missing
flow is always masked and excluded from normalization, losses, calibration
and evaluation.

## Determinism and verification

- Training uses fixed seeds (basin-selection seed 42; model seeds 42/123/456)
  and fixed final checkpoints; the sensitivity queue verifies protocol
  identities and artifact hashes before reusing anything.
- The evaluation protocol (`src/evaluation/protocol.py`) uses exact
  non-interpolated split-conformal ranks, within-basin 0.33/0.67 group cuts
  with ties in the lower group, and empty-set midpoint projection. It is
  covered by `tests/` (protocol tests need no dataset).
- Finite-grid distributional scores are integrated quantile scores, not full
  CRPS; the Gaussian analytic score is full CRPS. Legacy artifact ids `umal`
  and `encqr` denote a Gaussian location-scale quantile head and static
  memberwise split-CQR averaging, not the cited UMAL/EnCQR algorithms.
- Resampling intervals condition on the fitted models and the historical
  periods; model-seed standard deviations and basin/cluster uncertainty answer
  different questions and are reported separately.

## Outputs

Everything regenerates under `results/` (gitignored): `results/tables/`
(compact summaries), `results/figures/`, `results/predictions/`,
`results/checkpoints/`, `results/cache/`, `results/logs/`. Compact result
tables and per-sample prediction archives for the submission round live in
the frozen release packages identified by `JOH-revision-20261030` and its
SHA-256 manifest; no new public DOI is implied.

No new public release or journal submission is performed from this repository
automatically; those are author actions.
