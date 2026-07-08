[English](README.md) | [简体中文](README.zh-CN.md)

# Data Scarcity and Uncertainty in Hydrological Deep Learning

Companion code for the manuscript:

**How Data Scarcity Compromises Uncertainty Estimates in Hydrological Deep
Learning and How Conformal Calibration Mitigates It**

This repository studies how limited training data affects uncertainty estimates
in deep-learning streamflow prediction on CAMELS-US. The experiments use 671
basins, three meteorological forcing products, and a lightweight LPU-Stream
network. The central finding is that uncertainty coverage degrades much faster
than point-prediction skill under data scarcity, while split conformalized
quantile regression (CQR) recovers near-target marginal coverage without
retraining the neural network.

The repository includes analysis scripts, result JSON/CSV files, and generated
figures. Raw CAMELS-US data and trained checkpoints are not tracked because of
their size.

## Main Results

### Matched Data-Scarcity Gradient

All rows use the same seed-42 subset of 50 basins.

| Training data | Pooled median-quantile NSE | Raw PICP | CQR PICP | `q_cal` | CQR MPIW |
|---|---:|---:|---:|---:|---:|
| 1 year | 0.719 | 0.671 | 0.883 | 0.341 | 1.298 |
| 3 years | 0.813 | 0.741 | 0.869 | 0.155 | 0.968 |
| 5 years | 0.843 | 0.774 | 0.866 | 0.096 | 0.833 |
| 15 years | 0.884 | 0.828 | 0.876 | 0.042 | 0.662 |

Moving from 15 years to 1 year of training data reduces uncalibrated interval
coverage by about 16 percentage points, compared with a 19% relative decline in
NSE. The result motivates treating uncertainty reliability as a first-order
evaluation target, not as a side effect of point accuracy.

### Marginal and Conditional Calibration

For the 1-year scarcity model:

| Flow regime | Raw QR | Global CQR | Predicted-regime CQR | Observed-regime CQR |
|---|---:|---:|---:|---:|
| Low | 0.910 | 0.996 | 0.991 | 0.885 |
| Normal | 0.560 | 0.938 | 0.925 | 0.876 |
| High | 0.504 | 0.736 | 0.791 | 0.888 |

Global CQR fixes marginal coverage but still under-covers high-flow events.
Predicted-regime CQR is the deployable conditional variant; observed-regime CQR
is a diagnostic upper bound that uses the true test-time flow regime.

<p align="center">
  <img src="results/figures/fig1_degradation.png" width="90%"><br>
  <em>Data scarcity produces a sharper loss in uncertainty coverage than in point-prediction skill.</em>
</p>

<p align="center">
  <img src="results/figures/fig2_method_comparison.png" width="90%"><br>
  <em>Fair 671-basin comparison of MC Dropout, Deep Ensembles, Deep Ensembles + CQR, and single-model CQR.</em>
</p>

## Installation

```bash
git clone https://github.com/AuroraNeko/data-scarcity-uncertainty-hydrodl.git
cd data-scarcity-uncertainty-hydrodl
pip install -r requirements.txt
```

A CUDA GPU is recommended for training. Figure generation and manuscript-number
verification can run on CPU once the result files are present. The experiments
were run with Python 3.11 and PyTorch 2.x.

## Data Setup

Download CAMELS-US and preprocess the raw files:

```bash
python download_camels.py
python src/data/data_preprocessing.py
```

The preprocessing step creates:

```text
data/processed/camels_us/<basin_id>.csv
data/metadata/normalization_stats.json
data/metadata/basin_metadata.csv
```

The processed files contain 15 dynamic variables from Daymet, Maurer, and NLDAS
(five variables from each product), 13 static catchment attributes, streamflow
in mm/day, a missing-flow mask, and normalized training-period features.

## Repository Structure

```text
.
|-- download_camels.py
|-- configs/
|   `-- data_config.yaml
|-- src/
|   |-- data/
|   |   |-- data_preprocessing.py
|   |   |-- dataset.py
|   |   `-- compute_perbasin_stats.py
|   |-- losses/
|   |-- models/
|   `-- utils.py
|-- experiments/
|   |-- baseline/
|   |-- scarce/
|   |-- uncertainty/
|   |-- physics_guided/
|   |-- analysis/
|   `-- orchestrator.py
|-- results/
|   |-- tables/
|   `-- figures/
|-- REPRODUCIBILITY.md
`-- requirements.txt
```

The manuscript source and compiled submission files are handled as release or
submission artifacts, not as part of the public code checkout.

## Reproducing the Experiments

Run the full pipeline:

```bash
python experiments/orchestrator.py
```

Or run the main stages manually:

```bash
python experiments/baseline/train_model.py --model lpu_stream
python experiments/baseline/train_xgboost.py
python experiments/analysis/eval_point_perbasin.py

python experiments/uncertainty/train_quantile.py
python experiments/uncertainty/retrain_ensembles_correct.py
python experiments/uncertainty/eval_fair_671.py

python experiments/scarce/train_data_scarce.py --years 1
python experiments/scarce/train_data_scarce.py --years 3
python experiments/scarce/train_data_scarce.py --years 5
python experiments/scarce/train_data_scarce.py --years 15

python experiments/analysis/diagnose_1yr.py
python experiments/analysis/cross_region_validation.py
python experiments/analysis/stability_1yr.py
python experiments/analysis/confidence_levels.py
python experiments/analysis/enhanced_robustness.py
python experiments/analysis/basin_representativeness.py
python experiments/analysis/make_figures.py
python experiments/analysis/verify_manuscript.py
```

`verify_manuscript.py` checks stored numerical results against the manuscript
source when `paper/manuscript.tex` is available. In a code-only checkout, it
still audits the machine-readable JSON results and skips text-presence checks.

## Model Summary

LPU-Stream uses a 128-unit LSTM for the dynamic meteorological sequence and a
static-basin MLP that maps 13 catchment descriptors to a 32-dimensional
embedding. The quantile model predicts the 0.05, 0.50, and 0.95 quantiles with
pinball loss and has 104,099 trainable parameters. The point-prediction variant
has 103,969 parameters. CQR is applied post hoc on the held-out calibration
period and does not retrain the network.

## Reproducibility Notes

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for temporal splits, seeds,
calibration settings, figure regeneration, and the recommended release archive
contents.

## License

MIT. See [LICENSE](LICENSE).
