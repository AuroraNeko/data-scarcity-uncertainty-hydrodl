[English](README.md) | [简体中文](README.zh-CN.md)

# How supervised training-record length affects point skill and interval reliability in streamflow forecasting

This repository holds the full pipeline for a study of training-record
scarcity in deep-learning streamflow forecasting: how much does the model
gain from each additional year of gauge record, and what does it take for
its 90% uncertainty bands to reach their nominal coverage? Point skill grows
steadily with the record. Interval coverage does not; it is fixed by
recalibrating on a short held-out window.

## Approach

![Experiment design](docs/figures/design_timeline.png)

*Each model is trained on the last 1, 3, 5, or 15 years of labels before the
September 1995 cutoff. The calibration window (1995-2000) and the test
period (2000-2008) are identical for every training length.*

**The model.** All experiments share one LSTM forecaster. Each day it reads
the previous 180 days of weather forcings plus static catchment attributes
and outputs a median streamflow forecast with a 0.05/0.95 band around it. A
quantile (pinball) loss keeps the band monotone, so the upper bound never
falls below the median.

**The question.** Gauge records are expensive, and many basins have only a
few years of measurements, so it matters whether a forecaster needs the full
record or works well with a short one. The same model is trained four times
per basin and seed, on the last 1, 3, 5, or 15 years of labels before the
September 1995 cutoff. Nothing after the cutoff is used for training.

**Calibration.** The raw quantile bands of a neural network usually miss
their nominal coverage. Every model is therefore recalibrated with split
conformalized quantile regression (CQR): the held-out 1995-2000 window
measures how far the bands are off, and a single constant widens or narrows
all intervals by that amount. Three variants are compared: a global
correction, per-basin corrections, and corrections stratified by predicted
flow level.

**Scale.** The primary set covers 50 CAMELS-US basins with three training
seeds each. Four disjoint 50-basin subsets repeat the endpoints on 250 unique
basins, a 671-basin benchmark compares eight uncertainty constructions, and
48 further fits cover architecture and protocol sensitivities (TCN, a
standard LSTM, a 13-quantile head, alternative training years, 365-day
memory). Every reported effect carries a bootstrap interval that resamples
basins and HUC2 regions jointly.

## Key results

Two figures summarize the findings; the table lists the exact numbers.

**A longer record improves accuracy but not coverage.** Median basin NSE
rises from 0.511 with one supervised year to 0.666 with fifteen, and
interval quality improves along with it. Raw 90% coverage, however, stays
below the nominal level at every training length: 0.776 at one year, 0.821
at fifteen. Longer records alone therefore do not make uncalibrated
quantile predictions reliable.

![Effect of supervised training-record length](docs/figures/fig1_scarcity_effects.png)

*Supervised training-record length against point skill and interval behavior
on the 50 primary basins. Skill and coverage grow with the record; raw
coverage stays below the nominal 0.90 until recalibration.*

**Calibration is what restores coverage.** Recalibrating on the held-out
1995-2000 window with conformalized quantile regression brings marginal
coverage back to the nominal level, and stratifying the correction on
predicted flow restores coverage within each flow regime. The high
predicted-flow group, where raw intervals miss most, gains the most.

![Conditional coverage before and after CQR calibration](docs/figures/fig2_conditional_coverage.png)

*Coverage by predicted-flow regime before and after CQR recalibration on the
held-out calibration window.*

| Claim | Value |
|---|---|
| Median basin NSE, 1yr → 15yr | 0.511 → 0.666; paired median effect 0.140 [0.104, 0.173] |
| Raw pooled 90% coverage, 1yr → 15yr | 0.776 → 0.821; equal-basin effect +4.6 pp [1.5, 7.7], 72% of basins positive |
| One-year high **predicted**-flow coverage, global → stratified CQR | 0.832 → 0.882; gain +5.0 pp, positive under basin, HUC2 and synchronous-year resampling |
| Dense distribution scores (13-point IQS, log1p scale) | MDN 0.077511, Gaussian head 0.078314, noise-MC 0.081902; analytic Gaussian CRPS 0.080759 |
| Multi-level counterexample | 50% raw coverage 0.420 (1yr) vs 0.370 (15yr); TCN over-covers 0.930/0.922 |
| Matched 365-vs-180-day memory NSE effect | +0.0002 [-0.0022, 0.0070] short; +0.0072 [0.0002, 0.0125] long |

Across the resampling schemes, score and point-skill directions are more
stable than raw coverage, and no universal raw-coverage gradient is claimed.

## Repository layout

```
src/                    library (imported, never run directly)
├─ evaluation/          scoring protocol: CQR radius, interval scores, IQS,
│                       analytic CRPS, group rules, empty-set projection
├─ models/              LSTM (LPU-Stream), standard LSTM, TCN
├─ heads/               dense distribution heads (MDN, Gaussian, noise)
├─ benchmark/           671-basin benchmark loaders
├─ corrections/         shared loaders for the corrected scarcity runs
├─ data/                CAMELS reading, preprocessing, normalization
└─ losses/              pinball loss, CQR calibrator

experiments/            runnable pipeline, one job per script
├─ prepare/             cache build/verify, data-chain and checkpoint audits
├─ train_primary/       main 12 fits, replication subsets, endpoint reruns
├─ train_benchmark/     the eight 671-basin constructions
├─ bias_correction/     median-bias-fix diagnostics
├─ train_sensitivity/   the 48 endpoint fits (queue-driven)
├─ evaluate/            coverage, scores, method comparison, calibration variants
├─ analyze/             summaries, paired effects, representativeness, curves
└─ figures/             all paper figures from the frozen tables

run_pipeline.py         single entry point; runs the stages in order
download_camels.py      fetches the CAMELS-US archive
tests/                  protocol unit tests (no dataset required; the three
                        recorded-table checks skip until results/ exists)
configs/                data configuration
```

Dependencies flow one way: `run_pipeline.py` calls the scripts in
`experiments/`, and those scripts import shared logic from `src/`. Scripts
do not import each other.

## Reproduce

```bash
pip install -r requirements.txt
python download_camels.py
python src/data/data_preprocessing.py
python run_pipeline.py --list      # inspect the stages
python run_pipeline.py --dry-run   # resolve every script path, run nothing
python run_pipeline.py             # full rerun (stages 2/3/5 need a CUDA GPU)
python -m pytest tests -q          # protocol tests
```

Reference environment: Python 3.11, PyTorch 2.11.0+cu128, RTX 5060 Ti (8 GB);
see `requirements-reference-cu128.txt`. Data, trained weights and per-sample
prediction arrays are not tracked; each stage regenerates its outputs under
`results/`. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the frozen
submission-round snapshot and its SHA-256 manifest.

## Citation and license

See [CITATION.cff](CITATION.cff) and [LICENSE](LICENSE). The earlier release
series has Zenodo concept DOI
[10.5281/zenodo.21256955](https://doi.org/10.5281/zenodo.21256955); that
concept record alone does not identify the revision round.
