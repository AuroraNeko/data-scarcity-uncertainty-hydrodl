[English](README.md) | [简体中文](README.zh-CN.md)

# How supervised training-record length affects point skill and interval reliability in streamflow forecasting

Companion code for a Journal of Hydrology revision. The intervention restricts
**model-training labels while retaining a separate calibration record**. A
one-year model also needs calibration observations; it does not represent a
basin with only one year of all available flow measurements.

The primary experiment uses 50 CAMELS-US basins, four nested target windows
(1/3/5/15 years ending September 1995), and model seeds 42/123/456. Four additional
disjoint 50-basin subsets repeat only the endpoints: 250 unique basins overall.
Year composition and, under fixed epochs, optimizer exposure change with length.

Primary median basin NSE rises from 0.511 to 0.666. The paired median effect is
0.140 (95% basin interval [0.104, 0.173]); 94% of basins improve. Raw pooled 90%
coverage rises from 0.776 to 0.821, with an equal-basin paired effect of 4.6
percentage points ([1.5, 7.7]); only 72% of basins have a positive effect.
Short-record intervals are wider and have worse interval scores. Global CQR
brings pooled coverage near nominal without guaranteeing subgroup coverage.

Using calibration-fitted within-basin predicted-flow groups, one-year high-group
coverage improves from 0.832 to 0.882. The 5.0-percentage-point gain remains
positive under basin, HUC2 and synchronous water-year resampling. Retrospective
groups selected using realized flows answer a different question: exact
conditional intervals need not cover 90% after selecting large outcomes.

The revision adds 48 fits: 18 alternative-year, nine independent-validation,
six multi-level, six TCN, and nine matched-target 180/365-day-memory fits.
The TCN over-covers at both endpoints, and the 13-quantile 50% comparison
reverses the primary raw-coverage direction. Score and point-skill evidence is
more consistent than raw coverage; no universal coverage gradient is claimed.

## Revision identity and corrections

The source/compact-result submission archive is `JOH-revision-20261030`, with
a SHA-256 manifest in `revise/release/`. This identifier does not imply a new
public upload. The direct repository is
[AuroraNeko/data-scarcity-uncertainty-hydrodl](https://github.com/AuroraNeko/data-scarcity-uncertainty-hydrodl).
The earlier release series has [Zenodo concept DOI 10.5281/zenodo.21256955](https://doi.org/10.5281/zenodo.21256955);
that concept record alone does not identify the revision. Large prediction and
checkpoint availability is documented separately in the frozen package.

- Distributional scoring now de-standardizes **both** targets and predictions.
  All constructors share a three-point integrated quantile score (IQS); dense
  heads also share a 13-point grid. Neither finite grid is called full CRPS.
  Gaussian analytical CRPS is 0.080759; its 13-point IQS is 0.078314.
- Legacy `umal` denotes a **Gaussian location-scale quantile head**, not
  asymmetric-Laplacian UMAL. Legacy `encqr` denotes **static memberwise split-CQR
  averaging**, not the cited EnCQR algorithm. Compatibility IDs only trace arrays.
- Observed groups use within-basin 0.33/0.67 cuts, ties in lower groups and strict
  high-flow exceedance. Predictive thresholds are fitted within basin from
  calibration predictions, with cross-basin relative-stratum score pools.

`src/evaluation_protocol.py` defines signed scores, exact non-interpolated ranks,
group boundaries, empty-set midpoint projection and interval/distribution scores.
Resampling intervals condition on the historical periods and fitted models;
model-seed SD and basin/cluster uncertainty answer different questions.

## Reproduce

Install dependencies and obtain/preprocess CAMELS-US:

```bash
pip install -r requirements.txt
python download_camels.py
python src/data/data_preprocessing.py
```

The reference environment uses Python 3.11, PyTorch 2.11.0+cu128 and RTX 5060 Ti
(8 GB); see `requirements-reference-cu128.txt`. Complete reconstruction and
audit commands are in [revise/REPRODUCE.md](revise/REPRODUCE.md) and
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

```bash
python revise/code/prepare_training_cache.py
python revise/code/run_training_queue.py
python revise/code/analyze_revision_training.py --require-complete
python revise/code/publication_outputs.py --require-complete
python -m unittest discover -s tests -v
```

The queue checks protocol identities and artifact hashes before reuse. Data,
weights and per-sample arrays are excluded from ordinary Git history. Offline
scoring needs the prediction artifacts identified in the manifest; compact
summaries alone cannot regenerate per-sample metrics. Legacy evaluation scripts
in `revision_work/` remain for provenance. Authoritative revision evaluations
are in `revise/code/`; author manuscripts and reviewer correspondence are not
included in the public source/compact-result archive. See [LICENSE](LICENSE).
