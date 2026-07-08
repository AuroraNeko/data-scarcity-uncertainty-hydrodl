# Reproducibility Guide

This document records the minimum information needed to reproduce and audit the
main results of the manuscript:

**How Data Scarcity Compromises Uncertainty Estimates in Hydrological Deep
Learning and How Conformal Calibration Mitigates It**.

The repository supports two audit levels:

1. Reproduce the complete experiments from raw CAMELS-US data.
2. Audit the submitted figures and tables from stored result files.

The second path is the fastest way to verify that the reported manuscript
numbers match the machine-readable outputs.

## 1. Environment

The experiments were run with Python 3.11, PyTorch 2.x, and a single NVIDIA RTX
5060 Ti GPU. Training is GPU-recommended. Figure generation and manuscript
verification can run on CPU once the result files are present.

Install dependencies with:

```bash
pip install -r requirements.txt
```

The required dependency families are PyTorch, NumPy, pandas, matplotlib,
XGBoost, and SciPy.

## 2. Data

The study uses CAMELS-US version 2.0 with 671 basins. The preprocessing
pipeline uses three meteorological forcing products: Daymet, Maurer, and NLDAS.
From each forcing product, five daily variables are used: precipitation,
minimum and maximum temperature, solar radiation, and vapor pressure. The model
also uses 13 static catchment attributes covering topography, climate, soil,
vegetation, and geology.

Download and preprocess the data with:

```bash
python download_camels.py
python src/data/data_preprocessing.py
```

The processed basin files are expected under:

```text
data/processed/camels_us/
```

Raw and processed CAMELS-US data are not tracked in git because of their size.
The canonical data settings are summarized in `configs/data_config.yaml`.

## 3. Temporal Splits

The splits are water-year aligned:

| Split | Period | Role |
|---|---|---|
| Training | 1 Oct 1980 to 30 Sep 1995 | Model fitting; truncated to 1, 3, or 5 water years for scarcity experiments |
| Validation / calibration | 1 Oct 1995 to 30 Sep 2000 | CQR calibration and validation monitoring for point baselines |
| Test | 1 Oct 2000 to 30 Sep 2010 | Final held-out evaluation |

## 4. Main Experimental Settings

| Component | Setting |
|---|---|
| Scarcity basin subset | 50 basins sampled once with NumPy `RandomState(42)` and reused for all scarcity durations |
| Sequence length | 365 days for full 15-year models; 30, 90, 180, and 365 days for 1-, 3-, 5-, and 15-year scarcity runs |
| Quantile levels | 0.05, 0.50, 0.95 |
| CQR target | 90% intervals, `alpha = 0.10` |
| Optimizer | Adam, learning rate 0.001, betas 0.9 and 0.999 |
| Batch size | 1024 |
| Gradient clipping | 1.0 |
| Early stopping | patience 5, maximum 30 epochs |
| Deep Ensemble seeds | 42, 123, 456, 789, 999 |
| 1-year stability seeds | 42, 123, 456 |
| Bootstrap | Basin-cluster bootstrap with 500 resamples for full-dataset uncertainty metrics |

## 5. Full Reproduction Path

The full pipeline can be launched with:

```bash
python experiments/orchestrator.py
```

The main stages can also be run manually:

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
```

Some stages require trained checkpoints under `results/checkpoints/`. These
checkpoint files are large and should be regenerated or retrieved from the
archived release associated with the submitted manuscript.

## 6. Fast Audit Path

The result files used by the manuscript are stored under:

```text
results/tables/
results/figures/
```

Regenerate figures from stored tables:

```bash
python experiments/analysis/make_figures.py
```

Verify that manuscript numbers match the stored JSON/CSV outputs:

```bash
python experiments/analysis/verify_manuscript.py
```

If `paper/manuscript.tex` is present, the verification script also checks
selected manuscript text strings. In a code-only checkout, it skips those text
checks and still audits the machine-readable result files. A fully passing run
reports `ALL CHECKS PASSED`.

## 7. Files To Archive With A Release

For a public code release, include at least:

| Artifact | Purpose |
|---|---|
| `src/` and `experiments/` | Model, training, uncertainty, and analysis code |
| `configs/data_config.yaml` | Canonical data split and preprocessing reference |
| `requirements.txt` | Python dependency families |
| `results/tables/*.json` and `*.csv` | Machine-readable numerical results |
| `results/figures/*.pdf` and `*.png` | Generated manuscript figures |
| `REPRODUCIBILITY.md` | Audit trail and rerun instructions |

For a manuscript submission archive, additionally include:

| Artifact | Purpose |
|---|---|
| `paper/manuscript.tex` and `paper/supplement.tex` | Manuscript source |
| `paper/manuscript.pdf` and `paper/supplement.pdf` | Compiled submission files |

Do not archive raw CAMELS-US data unless the dataset license and repository
policy allow it. Instead, cite the original CAMELS-US source and provide the
preprocessing scripts and exact data paths.

## 8. Traceability Notes

The manuscript uses the full 671-basin model for method comparison and
aridity-based diagnostics. The matched data-scarcity gradient uses one fixed
50-basin seed-42 subset. Supplement Tables S1--S2 list the basin IDs and show
that this subset is not significantly different from the full CAMELS-US set
across 13 tested static attributes.

The supplement also includes Table S3, a compact reproducibility checklist for
splits, seeds, sequence lengths, calibration, and verification scripts.
