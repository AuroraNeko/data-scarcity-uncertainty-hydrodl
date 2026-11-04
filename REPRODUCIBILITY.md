# Reproducibility guide

`run_pipeline.py` defines nine stages: prepare → train primary → train
benchmark → correct → train sensitivity → evaluate → audit → analyze →
figures. Dependencies flow from the entry point to `experiments/` and then
to `src/`; stage scripts do not import one another.

## Environment and compute

The reference runs used Python 3.11, PyTorch 2.11.0+cu128 and an RTX 5060 Ti
with 8 GB VRAM. Install portable dependencies with
`pip install -r requirements.txt`; exact reference package versions are in
`requirements-reference-cu128.txt`. The portable entry points require
Python 3.10 or newer.

Stages 2, 3 and 5 train models. Full reproduction can take several GPU-days.
Stage 6 includes checkpoint inference, and `export_dense_taus.py` needs CUDA
when grids and timings must be rebuilt. Stages 1, 4, 7, 8 and 9 perform
preparation, offline analysis, verification and plotting on CPU.

The 48 additional sensitivity fits use float32 tensors with
`torch.backends.cudnn.allow_tf32=True` and
`torch.backends.cuda.matmul.allow_tf32=False`. Fixed seeds and deterministic
settings support repeatability on the reference backend; CPU and different
GPU kernels can produce different floating-point results. They do not imply
bitwise-identical retraining across hardware or library versions.

Sensitivity fits bind completed records and optimizer checkpoints to the raw
cache bytes, cache source manifest, ordered input-column definitions, declared
protocol and fitted normalization. Changed or missing identities stop reuse
before model outputs are overwritten. Preserve an incompatible run and start
in a fresh output tree. Legacy optimizer states are never silently adopted.
The version-1.1.3 completed records include an explicit audit binding for the
48 archived fits; their weights and prediction arrays retain their original
hashes. This binding is not a claim that those old checkpoint files originally
contained an input fingerprint.

## Data and full rerun

```bash
python download_camels.py
python src/data/data_preprocessing.py
python run_pipeline.py --list
python run_pipeline.py --dry-run
python run_pipeline.py
```

The download contains a roughly 3.4 GB time-series ZIP, about 14 GB
extracted, and small attribute files. Preprocessing writes basin CSVs under
`data/processed/camels_us/` and statistics under `data/metadata/`.

Missing streamflow is masked and excluded from normalization, losses,
calibration and scoring. Three forcing products supply 15 channels; the
Maurer and NLDAS Tmax/Tmin pairs are duplicated upstream, leaving 13
distinct daily series. This input layout is retained for checkpoint
compatibility. Stage 7 independently parses the raw source files to verify
forcing values, observation masks, cfs-to-mm/day conversion, log1p and
affine normalization across all 671 processed basins.

## Frozen predictions and offline evaluation

After extracting both revision archives, run
`python experiments/prepare/verify_submission_snapshot.py --output verification.json`.
This CPU check recomputes the two primary endpoints, a masked within-basin
high-flow group and the full-basin common-grid score directly from delivered
arrays. It uses the two archived metadata files and needs no raw CAMELS download.
Git blob hashes and ZIP file hashes describe different byte representations
where checkout line endings differ; use the corresponding manifest.

The Git checkout contains source code and README figure copies. Numerical
tables, raw/processed data, checkpoints and per-sample arrays are generated
or supplied as separate artifacts. For the fixed submission results, extract
the source/results and prediction/checkpoint packages at the repository
root, verify their manifests, and supply the upstream data/metadata needed
by the requested analysis. The package readme gives extraction details.

Examples that re-score existing predictions without training:

```bash
python run_pipeline.py --only coverage_groups.py
python run_pipeline.py --only distribution_scores.py
python run_pipeline.py --only method_comparison.py
python run_pipeline.py --stage 7
python run_pipeline.py --from 8
```

These commands need the earlier-stage arrays, tables, data and metadata they
consume. `--from` and `--stage` do not reconstruct missing prerequisites.
Method costs in the paper are archived reference measurements; fresh timing
results are stored separately because workstation load changes elapsed time.

## Training selection and calibration information

- The primary scarcity fits use condition-specific training-only statistics,
  four nested windows ending 30 September 1995, 180-day context, batch 512,
  30 epochs and seeds 42/123/456. Each fit jointly trains on its basin set.
- The primary comparison retains a separate five-year calibration period.
  One supervised year therefore requires more than one year of total flow
  observations. Alternative training years also change recency and available
  labels. A fixed epoch count gives longer windows more updates.
- The additional independent-validation fits reserve WY1996 for selection,
  followed by four years for CQR. Fixed-final and validation-selected weights
  are reported separately. Their test data never select weights.
- Full-basin quantile and dense baselines select minimum training loss with
  disclosed early stopping. The archived spread-only MC point checkpoint used
  the later calibration period for selection; its row is explicitly an
  exception to a clean independent split. The training-tail selection control
  is reported separately.
- Sequential calibration additionally uses flow observations arriving during
  testing. Static and sequential results have different information budgets.

## Verification and cache reuse

Stage 7 checks processed-data integrity, raw-source transformations,
checkpoint metadata and every recorded checkpoint/prediction SHA-256.
Primary runs validate completed artifacts before skipping; mismatched
primary results require an explicit `--force` rerun. Ensemble and sensitivity
drivers validate their recorded artifacts. Some other benchmark trainers
always retrain.

Dense-grid reuse is bound to current checkpoints, three-quantile predictions,
normalization, reference arrays and grid hashes. Older exports are adopted
only after a full shared-quantile check. Distribution scoring independently
checks the overlapping quantile columns. Publication figures require all 48
sensitivity fits; `--preview` is reserved for incomplete author layout work.

## Scoring and interpretation

`src/evaluation/protocol.py` defines signed CQR scores, non-interpolated
finite-sample ranks, within-basin 0.33/0.67 cuts with ties in the lower group,
and midpoint projection for empty intervals. Predicted thresholds are fitted
on calibration predictions. Observed-flow groups use realized outcomes and
are retrospective diagnostics.

Interval width, Winkler and distribution scores use the log-flow coordinate;
NSE uses physical mm/day after back-transformation. Finite-grid integrated
quantile scores (IQS) and complete analytic Gaussian CRPS are different
estimands. Legacy IDs `umal` and `encqr` identify a Gaussian quantile head
and static memberwise split-CQR averaging, respectively. They do not
represent the original UMAL or EnCQR algorithms.

Resampling intervals condition on the fitted models and historical periods.
Model-seed SD, basin bootstrap and HUC2-cluster sensitivity answer different
questions. Pooled coverage near nominal does not establish reliability in
every basin, season or flow group.

Some archived fields retain historical definitions: `coverage_*_flow` in
per-run metadata pools regimes across basins, while current paper subgroup
results are recomputed by `coverage_groups.py`; `cluster_bootstrap_ci95`
in old benchmark tables denotes a basin bootstrap. Current HUC2 intervals
are labelled `huc2_cluster_ci95` in the authoritative result tables.

## Outputs and release identity

Outputs live under gitignored `results/`: `tables/`, `figures/`,
`predictions/`, `checkpoints/`, `cache/`, `logs/`, `drafts/` and `qa/`.
`results/tables/` is the current numerical location. Figure scripts do not
modify author manuscripts. `make_readme_figures.py --copy-to-docs` explicitly
updates the committed README PNGs and their input-hash provenance.

Publication prose and control-table values read the same result JSON files.
`results/tables/publication_control_macro_bindings.json` records the source
field, formatting and scale for the retained architecture, update-budget and
spatial-replication prose macros. Missing source fields stop generation.

The local version-1.1.3 submission attachment set has three separate ZIPs:
source/results, predictions/checkpoints, and author submission materials.
Each carries a file manifest; the local `release_index.json` records archive
SHA-256 hashes. Public candidate source packages include the current pipeline
entry point and documentation assets; author/reviewer material is separate.
A Git tag fixes code only. Publishing release attachments or a new version
DOI is a separate author action, and is not performed by the pipeline.
