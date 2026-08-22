[English](README.md) | [简体中文](README.zh-CN.md)

# Data Scarcity and Interval Reliability in Hydrological Deep Learning

Companion code for the paper:

**Data Scarcity in Deep Learning Streamflow Forecasting: Prediction Skill,
Interval Reliability, and Conformal Calibration**

This repository studies how supervised streamflow record length changes point
skill and prediction-interval reliability. The main experiment is a controlled
CAMELS-US gradient: only the number of training target years changes. The basin
set, record end, sequence memory, batch size, epoch budget, calibration targets,
and test targets remain fixed.

The corrected result is deliberately more nuanced than the original draft.
Reducing the record from 15 to 1 year substantially lowers median basin NSE
(`0.666` to `0.511`). Raw 90% coverage also falls (`0.821` to `0.776`), but the
short-record intervals are wider rather than narrower. Their worse Winkler
score shows that forecast error grows faster than learned interval width adapts.
Global CQR stabilizes marginal coverage near 0.90, while high-flow conditional
coverage remains lower and benefits only partly from predicted-flow
stratification.

Versioned code and compact result snapshots are maintained under the Zenodo
concept DOI [10.5281/zenodo.21256955](https://doi.org/10.5281/zenodo.21256955).

## Corrected Main Results

All rows below use the same 50 basins, 180-day memory, batch size 512, 30-epoch
final checkpoint, 1995--2000 calibration targets, and 2000--2008 test targets.
Values are mean ± sample standard deviation over model seeds 42, 123, and 456.

| Training targets | Valid labels | Median basin NSE | Raw PICP | CQR PICP | Raw MPIW | Raw Winkler |
|---|---:|---:|---:|---:|---:|---:|
| 1 year | 18,250 | 0.511 ± 0.006 | 0.776 ± 0.010 | 0.896 ± 0.002 | 0.442 | 1.137 |
| 3 years | 54,750 | 0.570 ± 0.014 | 0.806 ± 0.013 | 0.906 ± 0.007 | 0.388 | 0.885 |
| 5 years | 91,300 | 0.612 ± 0.023 | 0.815 ± 0.014 | 0.903 ± 0.005 | 0.376 | 0.808 |
| 15 years | 266,322 | 0.666 ± 0.005 | 0.821 ± 0.009 | 0.901 ± 0.007 | 0.358 | 0.709 |

The paired median 15-minus-1-year basin NSE effect is `0.140` (95% basin
bootstrap CI `[0.104, 0.173]`), and 94% of basins improve with the longer
record. The raw PICP endpoint effect is 4.6 percentage points (95% CI
`[1.5, 7.7]`).

Targeted robustness checks preserve the point-skill conclusion without hiding
where interval behavior changes. Across five mutually disjoint 50-basin
subsets, the combined paired NSE effect is `0.154` (two-stage bootstrap 95% CI
`[0.137, 0.170]`), with 96.8% of 250 basins positive. The combined raw-PICP
effect is 3.72 percentage points (`[2.11, 5.32]`), but only 68% of basins are
positive and one subset interval includes zero. With both endpoints limited to
1,080 optimizer updates, the NSE effect remains positive at `0.047`
(`[0.022, 0.090]`) while the 15-year raw interval over-covers. A standard
83,523-parameter LSTM gives an NSE effect of `0.134` (`[0.095, 0.187]`). These
experiments support a repeatable record-length effect on point skill and a more
budget- and basin-sensitive interval response.

For the 1-year condition, mean coverage by observed within-basin flow tercile is:

| Flow tercile | Raw QR | Global CQR | Predicted-regime CQR | Observed-regime reference |
|---|---:|---:|---:|---:|
| Low | 0.783 | 0.932 | 0.896 | 0.865 |
| Middle | 0.797 | 0.910 | 0.917 | 0.886 |
| High | 0.749 | 0.841 | 0.867 | 0.899 |

Observed-regime CQR uses the unknown test observation for group assignment and
is diagnostic only. Predicted-regime CQR uses calibration-period predicted
median thresholds and is deployable for the seen basins in this experiment.

![Corrected scarcity effects](results/figures/corrected/fig1_scarcity_effects.png)

![Corrected conditional coverage](results/figures/corrected/fig2_conditional_coverage.png)

![Replicated endpoint effects](results/figures/corrected/figS5_subset_replication.png)

![High-flow coverage stress test](results/figures/corrected/figS3_high_flow_thresholds.png)

## Extended Calibration and Method Analyses

Beyond the corrected core pipeline, the repository carries the extended
analyses of the paper's supplement (S13--S19). They run offline from saved
predictions and compact tables, and their scripts live in
[`revision_work/`](revision_work/):

- **Calibration-side variants** (`tier1b_*.py`): sequential rolling- and
  expanding-window recalibration, adaptive conformal inference, asymmetric
  per-side radii, aridity-quartile groupwise radii, recalibration-window
  sensitivity (1--5 water years, `tables/scarcity_windows/`), and a faithful
  EnCQR comparison calibrated per ensemble member
  (`tables/corrected_full_671/encqr_comparison.json`).
- **Localization diagnostics** (`tier1b_*.py`): per-basin calibration pools,
  tertile-versus-decile predicted-flow strata, regime-ordered residual
  heterogeneity, and the deployable median-bias-correction experiment.
- **Extended method suite** (`tier3_*.py`): five-component mixture density
  network and UMAL-style continuous quantile heads, noise-augmented and
  checkpoint-reselected MC Dropout retrains, multi-level quantile reliability
  (50--99% nominal levels), and the offline CRPS comparison
  (`tables/tier3_671/`).
- **Spatial transfer** (`tier2_ungauged*.py`): frozen-model endpoint
  evaluation on the 621 basins never used by the 50-basin pipeline
  (`tables/ungauged_scarcity/ungauged_summary.json`; per-seed intermediates
  are reproducibility artifacts and stay untracked).

Together these analyses show that no tested calibration-side variant removes
the scarcity gradient in interval quality, that first-order median-bias
correction does not close the high-flow conditional gap, and that the gap is a
conditional-dispersion problem rather than a marginal-distribution one.

## Installation

```bash
git clone https://github.com/AuroraNeko/data-scarcity-uncertainty-hydrodl.git
cd data-scarcity-uncertainty-hydrodl
pip install -r requirements.txt
```

For the exact CUDA 12.8 package versions used by the reference runs:

```bash
pip install -r requirements-reference-cu128.txt
```

The corrected runs used Python 3.11, PyTorch 2.11.0+cu128, and one NVIDIA RTX
5060 Ti. Training and the 50-pass MC Dropout evaluation are GPU-recommended.
Analysis and figure generation are CPU-compatible once result tables exist.

## Data Setup

```bash
python download_camels.py
python src/data/data_preprocessing.py
python experiments/analysis/audit_processed_data.py \
  --processed-dir data/processed/camels_us \
  --output results/tables/processed_data_audit.json
```

The download helper uses [Zenodo record
15529996](https://zenodo.org/records/15529996); the source dataset DOI is
[10.5065/D6MW2F4D](https://doi.org/10.5065/D6MW2F4D).

The project combines the CAMELS-US v1.2 forcing/streamflow archive with the
attribute set v2.0. The common processed dates are `1980-01-01` through
`2008-12-31`, the endpoint of the Maurer forcing. No 2009--2010 Maurer tail is
silently mean-imputed.

For each forcing channel, linear interpolation uses a seven-value limit,
followed by forward and backward fills with a three-value limit each. Three
basins lack geologic porosity; standardized zero imputation gives the
corresponding training-basin mean, and none of those basins is in the fixed
50-basin scarcity subset.

Each processed file exposes 15 supplied forcing channels: five named columns
from Daymet, Maurer, and NLDAS. In the upstream archive, Maurer Tmax/Tmin are
identical and NLDAS Tmax/Tmin are identical, so these channels contain 13
distinct daily series. The duplicate columns are retained for compatibility
with the published checkpoints and are reported by the data-audit script.

## Reproducing the Corrected Pipeline

The orchestrator is resumable and does not terminate unrelated processes:

```bash
python experiments/orchestrator.py
```

A scarcity run is skipped only when its protocol fields, checkpoint, prediction
archive, and both SHA-256 hashes match. The full-basin evaluator always starts;
it reuses a split cache only after validating the checkpoint, normalization,
date, sequence-length, MC-count, and batch-size identity. Full-basin point and
seed-42 quantile training are skipped only when their completion sidecars also
match the checkpoint epoch and SHA-256.

To use already available full-basin checkpoints:

```bash
python experiments/orchestrator.py --skip-full-training
```

To use an existing compact robustness snapshot without retraining the 24
replicated-subset, six equal-update, and six standard-LSTM endpoint runs:

```bash
python experiments/orchestrator.py --skip-full-training --skip-robustness-training
```

The controlled scarcity runs can also be launched explicitly:

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

PowerShell equivalent:

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

Run the saved-prediction diagnostics and the targeted endpoint sensitivities:

```bash
python experiments/analysis/analyze_conditional_robustness.py
python experiments/analysis/make_scarcity_replication_subsets.py
python experiments/run_scarcity_robustness.py --stage all
python experiments/analysis/sanitize_scarcity_checkpoint_metadata.py
python experiments/analysis/analyze_scarcity_robustness_experiments.py
```

The robustness runner resumes only from results whose protocol fields and
checkpoint/prediction SHA-256 hashes match. Incomplete or mismatched artifacts
are reported and require an explicit `--force` rerun.

The separate full-basin method comparison uses existing point, quantile, and
ensemble checkpoints on the complete corrected target-date grid. Missing
observed streamflow remains masked and is excluded from metrics. The point
checkpoint is selected by validation MSE, whereas the quantile checkpoints are
selected by training pinball loss within a 30-epoch budget; this is therefore a
common-grid evaluation, not a causal comparison of training algorithms:

```bash
python experiments/uncertainty/evaluate_corrected_methods.py \
  --batch-size 1024 --n-mc 50 --bootstrap 1000
```

The 50-basin subset diagnostic uses an independent comparison group rather
than testing the subset against a full set that contains it:

```bash
python experiments/analysis/basin_representativeness.py
```

Generate compact manuscript outputs and verify consistency:

```bash
python experiments/analysis/make_corrected_figures.py
python experiments/analysis/export_manuscript_results.py  # author checkout
python experiments/analysis/verify_manuscript.py
```

## Repository Structure

[`experiments/README.md`](experiments/README.md) separates current manuscript
entry points from development-history and provenance-only scripts. Only the
current entry points write authoritative `corrected_*` outputs.

```text
.
|-- src/                              model, losses, and corrected data loader
|-- experiments/
|   |-- scarce/train_scarcity_corrected.py
|   |-- uncertainty/evaluate_corrected_methods.py
|   `-- analysis/                     audits, summaries, figures, verification
|-- revision_work/                    extended supplement analyses (S13--S19)
|-- results/
|   |-- tables/corrected_scarcity/
|   |-- tables/corrected_full_671/
|   |-- tables/full_checkpoint_training/
|   |-- tables/scarcity_replications/
|   |-- tables/scarcity_equal_updates/
|   |-- tables/scarcity_standard_lstm/
|   |-- tables/scarcity_windows/      recalibration-window sensitivity
|   |-- tables/tier3_671/             dense heads, retrains, reliability, CRPS
|   |-- tables/ungauged_scarcity/     spatial transfer to 621 unseen basins
|   |-- figures/corrected/
|   `-- legacy/                        superseded outputs, provenance only
|-- REPRODUCIBILITY.md
|-- requirements-reference-cu128.txt
`-- requirements.txt
```

Raw/processed data, checkpoints, and per-sample prediction archives are not Git
artifacts. Compact JSON/CSV result tables and generated figures are suitable for
Git. Prediction archives and checkpoints should be included with a formal
tagged release when storage permits; result metadata retain their SHA-256
hashes so local or release copies can be audited.

## Model Summary

LPU-Stream contains a 128-unit LSTM and a static MLP that embeds 13 catchment
descriptors into 32 dimensions. The corrected scarcity model predicts the
0.05, 0.50, and 0.95 quantiles with pinball loss and a monotonic
median-plus-softplus parameterization, giving zero quantile crossings by
construction. The quantile model has 104,099 trainable parameters. CQR is a
post-hoc calibration step and does not retrain the network.

## Reproducibility and Citation

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for protocol boundaries, artifact
policy, and audit commands. Cite the versioned archive corresponding to the
code used; the release series has concept DOI
[10.5281/zenodo.21256955](https://doi.org/10.5281/zenodo.21256955).

## License

MIT. See [LICENSE](LICENSE).
