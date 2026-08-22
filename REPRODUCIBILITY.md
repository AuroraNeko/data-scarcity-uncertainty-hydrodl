# Reproducibility Guide

This guide describes the corrected experiments for:

**Data Scarcity in Deep Learning Streamflow Forecasting: Prediction Skill,
Interval Reliability, and Conformal Calibration**

Versioned code and compact result snapshots are maintained under the Zenodo
concept DOI [10.5281/zenodo.21256955](https://doi.org/10.5281/zenodo.21256955).

The project supports two audit levels:

1. Re-run training and inference after installing CAMELS-US locally.
2. Recompute summaries and figures from compact result tables, then compare
   them with the manuscript in an author checkout.

Large upstream data, checkpoints, and per-sample predictions are intentionally
outside ordinary Git history. Result metadata record their SHA-256 hashes.

## 1. Reference Environment

The corrected experiments used:

| Component | Version or setting |
|---|---|
| Python | 3.11 |
| PyTorch | 2.11.0+cu128 |
| GPU | NVIDIA RTX 5060 Ti, 8 GB |
| OS used for reference run | Windows / PowerShell |
| Model seeds | 42, 123, 456 for scarcity; 42, 123, 456, 789, 999 for the full ensemble |

Install declared Python dependencies with:

```bash
pip install -r requirements.txt
```

The portable file uses lower bounds so users can select a CPU or compatible
CUDA build. The exact packages used for the reported CUDA 12.8 runs are pinned
separately:

```bash
pip install -r requirements-reference-cu128.txt
```

Deterministic helpers seed Python, NumPy, CPU PyTorch, and CUDA PyTorch. The
corrected scarcity runs use deterministic algorithms where available. Exact
floating-point identity across GPU architectures is not guaranteed; the
stored artifacts and hashes identify the reference run.

## 2. Source Data and Processed-Data Audit

The project combines:

- CAMELS-US basin forcing and streamflow archive v1.2;
- CAMELS catchment attributes v2.0;
- 671 basins;
- Daymet, Maurer, and NLDAS basin-mean forcing products.

Source locations:

- [Zenodo record 15529996](https://zenodo.org/records/15529996)
- [CAMELS-US DOI 10.5065/D6MW2F4D](https://doi.org/10.5065/D6MW2F4D)

Download and preprocess:

```bash
python download_camels.py
python src/data/data_preprocessing.py
```

The common date range is `1980-01-01` through `2008-12-31`. Maurer ends in
2008; the corrected pipeline does not fill a later tail with standardized zero.
Within the common period, each forcing channel is linearly interpolated with a
seven-value limit; subsequent forward and backward fills each use a three-value
limit.

The source files expose five named columns per forcing product, or 15 supplied
channels. Maurer Tmax/Tmin are exact duplicates, and NLDAS Tmax/Tmin are exact
duplicates, leaving 13 distinct daily series. The columns remain in place for
checkpoint compatibility. This property is verified rather than hidden:

```bash
python experiments/analysis/audit_processed_data.py \
  --processed-dir data/processed/camels_us \
  --expected-start 1980-01-01 \
  --expected-end 2008-12-31 \
  --test-start 2000-10-01 \
  --output results/tables/processed_data_audit.json
```

The reference audit reports:

| Check | Reference result |
|---|---:|
| Basin files | 671 |
| Total daily rows | 7,107,903 |
| Duplicate-date basins | 0 |
| Discontinuous-date basins | 0 |
| Maximum dynamic missing fraction | 0 |
| Maximum test-period dynamic missing fraction | 0 |
| Missing static values | Geologic porosity in 3 basins; all other selected static values present |
| Calibration target-date cells / observed flows | 1,225,917 / 1,225,916 |
| Test target-date cells / observed flows | 2,022,394 / 2,020,930 |
| Maurer Tmax/Tmin unequal rows | 0 |
| NLDAS Tmax/Tmin unequal rows | 0 |
| Daymet Tmax/Tmin unequal rows | greater than 0 |

Missing static values are set to zero after standardization, which is
equivalent to the corresponding training-basin mean. None of the three affected
basins belongs to the fixed 50-basin scarcity subset.

## 3. Controlled Scarcity Protocol

The `scarcity_v2_fixed_end_fixed_memory` protocol changes one experimental
factor: the number of supervised target years.

| Condition | Training target period | Valid labels in reference subset |
|---|---|---:|
| 1 year | 1994-10-01 to 1995-09-30 | 18,250 |
| 3 years | 1992-10-01 to 1995-09-30 | 54,750 |
| 5 years | 1990-10-01 to 1995-09-30 | 91,300 |
| 15 years | 1980-10-01 to 1995-09-30 | 266,322 |

Controls shared by all 12 runs:

| Component | Fixed value |
|---|---|
| Basin set | 50 basins selected once with basin seed 42 |
| Sequence length | 180 days |
| Batch size | 512 |
| Epoch budget | 30 |
| Checkpoint rule | final checkpoint at fixed epoch budget |
| Calibration targets | 1995-10-01 to 2000-09-30, 91,350 observed basin-days |
| Test targets | 2000-10-01 to 2008-12-31, 150,700 grid cells / 149,239 observed flows |
| Model seeds | 42, 123, 456 |
| Quantiles | 0.05, 0.50, 0.95 |
| CQR target | 90%, `alpha = 0.10` |
| Optimizer | Adam, learning rate 0.001 |
| Gradient clipping | 1.0 |

Antecedent context begins before each declared target boundary. Consequently,
calibration and test sample IDs and dates are identical across every scarcity
condition; the first 180 target days are not silently removed.

Each condition fits normalization statistics only from its allowed 50 training
basins and training target years. Predictions and targets are de-standardized
to `log1p(mm/day)` before interval metrics. Basin NSE is computed after inversion
to nonnegative physical streamflow.

The corrected quantile head predicts a median and nonnegative softplus offsets,
so lower <= median <= upper by construction. The CQR finite-sample quantile is
the exact order statistic at rank
`min(n, ceil((n + 1) * (1 - alpha)))`; no linear interpolation is used.

## 4. Running the Scarcity Gradient

Bash:

```bash
for years in 1 3 5 15; do
  for seed in 42 123 456; do
    python experiments/scarce/train_scarcity_corrected.py \
      --years "$years" --basin-seed 42 --model-seed "$seed" \
      --seq-len 180 --batch-size 512 --epochs 30
  done
done
python experiments/analysis/analyze_corrected_scarcity.py
```

PowerShell:

```powershell
foreach ($years in 1,3,5,15) {
  foreach ($seed in 42,123,456) {
    python experiments/scarce/train_scarcity_corrected.py `
      --years $years --basin-seed 42 --model-seed $seed `
      --seq-len 180 --batch-size 512 --epochs 30
  }
}
python experiments/analysis/analyze_corrected_scarcity.py
```

Authoritative compact outputs:

```text
results/tables/corrected_scarcity/scarcity_summary.json
results/tables/corrected_scarcity/scarcity_runs.csv
results/tables/corrected_scarcity/basin_endpoint_effects.csv
results/tables/corrected_scarcity/scarcity_<years>yr_basinseed42_modelseed<seed>.json
```

Local/release artifacts referenced by those JSON files:

```text
results/checkpoints/corrected_scarcity/*.pt
results/predictions/corrected_scarcity/*.npz
```

`analyze_corrected_scarcity.py` verifies every checkpoint and prediction hash,
confirms sample identity across runs, exports the basin-level paired endpoint
table, and performs 5000 whole-basin bootstrap resamples after averaging over
model seeds.

The fixed 50-basin subset is compared with the disjoint set of 621 remaining
basins using 13 two-sample KS tests, Holm correction, and Cohen's $d$:

```bash
python experiments/analysis/basin_representativeness.py
```

This script writes compact diagnostics and a LaTeX row fragment; it does not
overwrite the manuscript or supplement.

## 5. Scarcity Robustness Analyses

The conditional analyses reuse the saved predictions from the 12 primary
scarcity runs and do not retrain a model:

```bash
python experiments/analysis/analyze_conditional_robustness.py
```

This evaluates exact within-basin highest-flow shares of 100%, 33%, 20%, 10%,
5%, and 2%; exports per-basin calibration distributions; and applies 5000
paired whole-basin resamples after averaging each basin over model seeds.

Create the endpoint-replication subsets and run the training sensitivities:

```bash
python experiments/analysis/make_scarcity_replication_subsets.py
python experiments/run_scarcity_robustness.py --stage replications
python experiments/run_scarcity_robustness.py --stage equal-updates
python experiments/run_scarcity_robustness.py --stage architecture
python experiments/analysis/sanitize_scarcity_checkpoint_metadata.py
python experiments/analysis/analyze_scarcity_robustness_experiments.py
```

The robustness runner skips a fit only when its protocol identity, budget,
checkpoint, prediction archive, and both recorded SHA-256 hashes match. A
mismatched partial result is reported rather than silently reused; `--force`
is required to replace it.

The metadata sanitizer removes the known machine-local subset-list path from
checkpoint command arguments before release, rejects any remaining Windows,
UNC, or POSIX absolute path, writes a stable SHA-256 over model-state tensors,
verifies that this digest is unchanged by the metadata rewrite, and updates the
compact checkpoint-file hash. It never changes learned parameters.

The replication design adds four mutually disjoint 50-basin subsets sampled
proportionally from joint quartiles of aridity and `log1p` drainage area. Only
the 1- and 15-year endpoints are repeated, with the same three model seeds,
giving 24 new fits. Together with the primary sample, the analysis spans five
non-overlapping subsets and 250 unique basins. Each basin is averaged over
model seeds before pairing. The combined 95% interval resamples the five
training subsets and then basins within each selected subset.

The equal-update sensitivity gives both primary-subset endpoints 1,080 Adam
updates. This equals the 1-year 30-epoch budget; it exposes the 15-year model to
only about two complete passes and is intentionally a compute-constrained
sensitivity rather than an alternative estimate of the fixed-epoch effect.

The architecture sensitivity adds six fixed-epoch endpoint fits from a
standard 128-unit LSTM. It shares inputs, static attributes, dropout, quantile
head, optimizer, normalization, dates, and evaluation with LPU-Stream, but
fuses static attributes only in the output head rather than injecting a learned
embedding at each time step.

Authoritative compact outputs:

```text
results/tables/corrected_scarcity/conditional_robustness.json
results/tables/corrected_scarcity/conditional_robustness_by_run.csv
results/tables/corrected_scarcity/basin_calibration_distribution.csv
results/tables/scarcity_replications/replication_subsets.csv
results/tables/scarcity_replications/scarcity_replication_summary.json
results/tables/scarcity_replications/scarcity_replication_basin_effects.csv
results/tables/scarcity_equal_updates/equal_update_summary.json
results/tables/scarcity_standard_lstm/architecture_sensitivity_summary.json
```

## 6. Separate 671-Basin Method Comparison

This comparison is not used to estimate the scarcity effect. It reuses the
full-basin point and quantile checkpoints and evaluates all methods on the same
complete target-date grid. Missing observed streamflow is masked consistently:

- calibration: `1995-10-01` to `2000-09-30`;
- test: `2000-10-01` to `2008-12-31`;
- sequence length: 365 days;
- methods: MC Dropout, single quantile regression, and five-member
  mean-quantile ensemble;
- variants: raw and global-CQR for every base method;
- uncertainty: 1000 paired whole-basin bootstrap resamples.

The retained full-basin checkpoints use forcing from 1980-10-01, but their
legacy training loader begins supervised targets only after the first 365-day
sequence. The effective first training target is therefore 1981-10-01. This
warm-up is shared by all methods and disclosed in the paper; it is not described
as a full 15 target-year protocol.

The squared-error point checkpoint is selected by minimum validation MSE. The
seed-42 quantile checkpoint and four additional ensemble members are selected
by minimum training pinball loss within a 30-epoch budget. The comparison thus
standardizes the evaluation grid and post-processing, but it is not interpreted
as a controlled causal comparison of training or checkpoint-selection rules.

The point and seed-42 quantile training scripts write completion sidecars only
after the training loop finishes. The orchestrator skips those stages only when
the checkpoint configuration, completion flag, stored path, epoch, and SHA-256
all match:

```text
results/tables/full_checkpoint_training/point_seed42.json
results/tables/full_checkpoint_training/quantile_seed42.json
```

Run corrected inference:

```bash
python experiments/uncertainty/evaluate_corrected_methods.py \
  --batch-size 1024 --n-mc 50 --bootstrap 1000
```

The script caches calibration and test predictions separately. Cache identity
includes checkpoint hashes, the normalization-statistics hash, dates, sequence
length, MC count, and batch size.
MC Dropout uploads each batch once and executes 50 stochastic passes while the
batch remains on the GPU.

Authoritative compact outputs:

```text
results/tables/corrected_full_671/method_comparison.json
results/tables/corrected_full_671/method_comparison.csv
results/tables/corrected_full_671/climate_quartiles.csv
```

Large local/release caches:

```text
results/predictions/corrected_full_671/calibration_predictions.npz
results/predictions/corrected_full_671/test_predictions.npz
```

## 7. Figures, Manuscript Fragments, and Verification

```bash
python experiments/analysis/make_corrected_figures.py
python experiments/analysis/export_manuscript_results.py  # author checkout only
python experiments/analysis/prepare_submission_sources.py  # author checkout only
python experiments/analysis/verify_manuscript.py
```

`make_corrected_figures.py` reads only corrected result JSON/CSV files and writes
to `results/figures/corrected/`. `export_manuscript_results.py` generates LaTeX
macros and table rows under `paper/generated/`, preventing independent manual
copies of headline numbers. `prepare_submission_sources.py` expands those values,
table rows, and references into one HESS-compatible manuscript source, removes
custom-command and extra-package dependencies, and copies numbered figure files
to the ignored `submission/hess/` directory. The verifier checks protocol fields,
compact results, available artifact hashes, generated fragments, obsolete-claim
absence, citation keys, and expected figures. Paper checks are skipped in the
public code checkout because `/paper/` is excluded there.

## 8. Public Repository and Release Policy

Appropriate for ordinary Git history:

| Artifact | Reason |
|---|---|
| `src/`, `experiments/`, `tests/` | Executable methods and protocol tests |
| `configs/` | Human-readable configuration references |
| `results/tables/**/*.json` and `*.csv` | Compact machine-readable evidence |
| `results/figures/corrected/*.pdf` and `*.png` | Regenerable publication figures |
| README, reproducibility guide, citation metadata | Public use and traceability |

Excluded from ordinary Git history:

| Path | Reason |
|---|---|
| `/data/` | Large upstream and processed CAMELS-US data |
| `/results/checkpoints/` | Large model weights |
| `/results/predictions/` | Large per-sample intermediate arrays |
| `/paper/`, `/submission/` | Author and journal submission materials |
| `/pdf/` | Potentially copyrighted third-party papers |
| logs and local tool state | Machine-specific, non-scientific intermediates |

Checkpoints and per-sample predictions are scientifically useful release
artifacts even though they are poor Git artifacts. A formal tagged archive can
include them when storage permits. Compact JSON metadata keep paths, dates,
protocol identifiers, and hashes so an archive copy can be matched exactly.

## 9. Automated Tests

Run protocol unit tests with:

```bash
python -m unittest discover -s tests -v
```

The tests cover the exact conformal order statistic, monotonic LPU-Stream and
standard-LSTM quantile parameterizations, separation of context dates from
target dates, disjoint replication subsets, fixed optimizer-step metadata,
runner-stage isolation, checkpoint absolute-path rejection, and
conditional-analysis row counts and reference values.
