# Experiment Entry Points

The current manuscript is generated from these entry points:

- `scarce/train_scarcity_corrected.py`
- `analysis/analyze_corrected_scarcity.py`
- `analysis/analyze_conditional_robustness.py`
- `analysis/make_scarcity_replication_subsets.py`
- `run_scarcity_robustness.py`
- `analysis/sanitize_scarcity_checkpoint_metadata.py`
- `analysis/analyze_scarcity_robustness_experiments.py`
- `analysis/basin_representativeness.py`
- `uncertainty/evaluate_corrected_methods.py`
- `analysis/audit_processed_data.py`
- `analysis/make_corrected_figures.py`
- `analysis/export_manuscript_results.py`
- `analysis/prepare_submission_sources.py`
- `analysis/verify_manuscript.py`
- `orchestrator.py`

Other scripts remain for development history, baseline checkpoint creation, or
legacy-result provenance. In particular, `scarce/train_data_scarce.py`,
`uncertainty/eval_fair_671.py`, `analysis/diagnose_1yr.py`,
`analysis/cross_region_validation.py`, and `analysis/make_figures.py` do not
implement the corrected manuscript protocol and must not overwrite or be mixed
with `results/tables/corrected_*` outputs.

The separate full-basin checkpoints are created by
`baseline/train_model.py --model lpu_stream`, `uncertainty/train_quantile.py`,
and `uncertainty/retrain_ensembles_correct.py`. Those scripts create model
checkpoints and compact training metadata; the authoritative comparison is
always produced by `uncertainty/evaluate_corrected_methods.py`.

`analysis/analyze_conditional_robustness.py` reads the saved primary-run
prediction archives and performs no training. `run_scarcity_robustness.py` is
resumable at the JSON-result level and separates replicated-subset and
equal-update outputs into distinct protocol directories. Do not merge those
sensitivity artifacts into `corrected_scarcity/`: they answer different
sampling and optimization-budget questions.

The runner's `architecture` stage uses the quantile-capable standard LSTM in
`src/models/lstm.py` for the same primary 1/15-year endpoints. It writes only
to `scarcity_standard_lstm/` and is an architecture sensitivity, not a new
full-basin leaderboard.

`analysis/sanitize_scarcity_checkpoint_metadata.py` is a release audit, not a
training or scoring stage. It replaces a known machine-local subset-list path
with its repository-relative form, rejects any other absolute filesystem path,
proves model tensors unchanged via a stable tensor digest, and updates the
compact checkpoint-file hash.

`analysis/prepare_submission_sources.py` expands generated numerical macros,
inlines generated table rows and references, removes authoring-only list/package
conveniences, and writes a self-contained HESS source plus correctly numbered
figure files under the ignored `submission/hess/` directory. It does not change
the auditable authoring source in `paper/`.
