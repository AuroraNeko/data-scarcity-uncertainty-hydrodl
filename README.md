[English](README.md) · [简体中文](README.zh-CN.md)

# Training-record length, forecast skill and interval reliability

Code for **How supervised training-record length affects point skill and interval reliability in streamflow forecasting**.

How do point predictions and uncertainty intervals change when a regional streamflow model has fewer years of training labels? On the primary CAMELS-US experiment, longer records improve point skill and interval scores. CQR brings pooled 90% coverage close to nominal, while reliability still varies across basins and forecast conditions.

[Experiment and results](#experiment-and-results) · [Reproduce](#reproduce) · [Repository guide](#repository-guide) · [Protocol and artifact guide](REPRODUCIBILITY.md)

## Experiment and results

![Training, calibration and test periods for the primary experiment](docs/figures/design_timeline.png)

The primary experiment fits **one regional model across the same 50 basins** for each combination of 1, 3, 5 or 15 supervised years and three model seeds: **12 fits**. All training ends on 30 September 1995. Calibration uses October 1995–September 2000; testing uses October 2000–December 2008.

Each forecast uses the preceding 180 days of weather and 13 static attributes, with no observed streamflow in the dynamic input sequences. Five of the static attributes are CAMELS cross-period climate indices, basin constants derived by the dataset authors from 1989–2009 records and not recalculated per supervised window; their potential evapotranspiration term was calibrated per basin against observed upstream discharge. See the manuscript data section for details. The LSTM predicts three ordered quantiles, 0.05/0.50/0.95. **Softplus offsets enforce their ordering**; a masked pinball loss trains them. The 15 supplied weather channels contain 13 distinct series because two upstream temperature pairs are duplicated.

The observation budget matters: **one year of training labels still comes with five separate years of calibration flow observations**. Nested windows also change the years represented and the number of optimizer updates under a fixed epoch budget. The study evaluates this historical allocation of observations, with sensitivity experiments to test its limits.

### Point skill and interval quality

![Point skill and pooled raw versus global-CQR coverage across four training lengths](docs/figures/fig1_scarcity_effects.png)

Median basin NSE rises from **0.511 to 0.666** between the one- and fifteen-year conditions. Raw 90% coverage rises from **0.776 to 0.821**, remaining below nominal. Short-record intervals are wider and have worse interval scores. Error bars above show sample SD across three model seeds; paired confidence intervals are reported separately below.

### Pooled coverage and subgroup reliability

![One-year model coverage in calibration-predicted groups and retrospective observed-flow groups](docs/figures/fig2_conditional_coverage.png)

The left panel groups days using calibration-period predicted-flow thresholds, so group membership is available before observing the outcome. In the high predicted-flow group, stratified CQR improves coverage from **0.832 to 0.882**, leaving a shortfall below 0.90. The right panel selects groups by realized test flow: these are retrospective stress tests, whose coverage can be reduced by outcome selection itself. The two groupings answer different questions.

| Quantity | Result and interpretation |
|---|---|
| Primary NSE, 1 → 15 years | 0.511 → 0.666; paired median difference **0.140 [0.104, 0.173]**; 94% of basins improve |
| Raw pooled 90% coverage | 0.776 → 0.821; equal-basin difference **+4.6 percentage points [1.5, 7.7]**; 72% of basins improve |
| One-year high predicted-flow coverage | Global → stratified CQR: **0.832 → 0.882**; gain +5.0 percentage points, positive under basin, HUC2 and synchronous-year resampling |
| Common 13-point distribution score | MDN 0.077511; Gaussian head 0.078314; noise-MC 0.081902. These are finite-grid IQS on the log-flow scale; complete Gaussian CRPS is 0.080759 |
| Coverage counterexamples | At nominal 50%, raw coverage is 0.420 / 0.370 for 1 / 15 years. TCN raw 90% intervals over-cover at both endpoints: 0.930 / 0.922 |
| Matched 365 − 180-day memory effect on NSE | Short record: +0.0002 [−0.0022, 0.0070]; long record: +0.0072 [0.0002, 0.0125] |

Brackets are 95% paired whole-basin bootstrap intervals conditional on fitted models and the selected historical periods. HUC2 intervals provide an additional dependence sensitivity; they do not measure independent retraining uncertainty. A difference between endpoint medians and a median of paired basin differences have different aggregation orders.

The extended evidence includes four additional disjoint 50-basin subsets (**250 unique basins in total**, with endpoint replication), a separate **671-basin comparison of eight interval constructions**, and **48 additional fits** covering alternative years, independent validation and update budgets, multiple nominal levels, TCN and input memory. A standard-LSTM endpoint check is also available. Point-skill and interval-score directions are more consistent than raw-coverage directions across these checks.

## Reproduce

Python 3.10 or newer is required; the reference runs used Python 3.11, PyTorch 2.11.0+cu128 and an RTX 5060 Ti with 8 GB VRAM. [Pinned reference packages](requirements-reference-cu128.txt) accompany the portable dependency list.

```bash
pip install -r requirements.txt
python download_camels.py
python src/data/data_preprocessing.py

python run_pipeline.py --list       # stages and their scripts
python run_pipeline.py --dry-run    # validate script paths without execution
python run_pipeline.py              # train, evaluate, audit, summarize and plot
```

Full reproduction involves GPU training and can take several GPU-days. Stages 2, 3 and 5 train models; stage 6 includes GPU inference and timing. The [reproducibility guide](REPRODUCIBILITY.md) distinguishes a full rerun from re-evaluating frozen predictions, specifies calibration information and precision settings, and explains artifact identities.

Data, weights, prediction arrays and numerical outputs are built under gitignored directories. The Git repository contains code and the README figure copies. A frozen numerical release additionally needs its source/results and prediction/checkpoint archives with SHA-256 manifests; publication of those attachments is a separate author action.

## Repository guide

| Path | Purpose |
|---|---|
| [`run_pipeline.py`](run_pipeline.py) | Nine-stage execution order and entry point |
| [`src/data/`](src/data/) | CAMELS preprocessing, training-only normalization and target-date windows |
| [`src/models/`](src/models/) and [`src/heads/`](src/heads/) | Regional LSTM, standard LSTM, TCN and probabilistic heads |
| [`src/evaluation/protocol.py`](src/evaluation/protocol.py) | CQR ranks, group boundaries, interval scores, finite-grid IQS and analytic CRPS |
| [`src/sensitivity.py`](src/sensitivity.py) | The 48 declared sensitivity configurations |
| [`experiments/`](experiments/) | Preparation, training, evaluation, audit, analysis and figure scripts |
| [`docs/figures/`](docs/figures/) | README figures and their input-hash provenance |
| `results/` (generated) | Tables, figures, checkpoints, predictions, caches and QA reports |

Dependencies flow from `run_pipeline.py` to `experiments/` to `src/`. Stage scripts share library functions rather than importing one another. Figure generation writes to `results/`; the author manuscript workspace is maintained separately.

To regenerate the README figures after recomputing the numerical tables:

```bash
python experiments/figures/make_readme_figures.py --copy-to-docs
```

## Citation and license

[CITATION.cff](CITATION.cff) provides software citation metadata; the code uses the [MIT license](LICENSE). The earlier release series has Zenodo concept DOI [10.5281/zenodo.21256955](https://doi.org/10.5281/zenodo.21256955). Cite the version and artifact manifest used for a reproduction, since the concept record groups multiple versions.
