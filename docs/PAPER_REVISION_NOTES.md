# Paper ↔ Code Reconciliation Notes / 论文 ↔ 代码 对照修订清单

This document lists places where the **paper text** (`paper/manuscript.tex`) does
not match what the **code actually does**. The code in this repository is the
artifact that produced the reported results; the paper prose should be edited to
match it. None of the discrepancies change the paper's conclusions — they are
reporting/wording mismatches in the Methods section.

本文档列出 **论文正文**（`paper/manuscript.tex`）与 **代码实际行为** 不一致之处。本仓库的
代码是产出论文结果的真实代码；应以代码为准，修订论文措辞。下列各项均**不影响论文结论**，
只是 Methods 部分的描述性出入。

---

## 1. Target transform / 目标变换  🔴

- **Paper (§2.1, line 89):** "Target streamflow is log-transformed as
  $y_{\text{log}} = \ln(y + 0.01)$".
- **Code:** `src/data/data_preprocessing.py` uses `TARGET_TRANSFORM = "log1p"`,
  i.e. `numpy.log1p` → $\ln(1 + y)$, **not** $\ln(y + 0.01)$.
- **Suggested fix:** Change the paper formula to $y_{\text{log}} = \ln(1 + y)$.

## 2. Dropout / Dropout 配置  🟡

- **Paper (§2.2, line 103):** "Dropout (p=0.4) is applied to the LSTM
  hidden-to-hidden connections."
- **Code:** `LPUStreamModel` uses a **single LSTM layer**, so the LSTM's internal
  dropout is a no-op. Dropout `p=0.3` is applied to the final hidden state and
  inside the prediction head (`src/models/lpu_stream.py`; CONFIG `dropout: 0.3`).
- **Suggested fix:** Reword to "Dropout (p=0.3) is applied to the LSTM output and
  the prediction head." (Single-layer LSTMs have no hidden-to-hidden recurrence
  to apply dropout to.)

## 3. Batch size / 批大小  🟡

- **Paper (§2.2, line 103):** "batch size 256".
- **Code:** LPU-Stream and EA-LSTM use **batch size 1024**
  (`experiments/uncertainty/train_quantile.py`, `train_model.py`).
  Only the plain-LSTM and TCN baselines use 256.
- **Suggested fix:** State batch size 1024 for LPU-Stream (and note 256 for the
  LSTM/TCN baselines).

## 4. Early stopping / 早停策略  🔴

- **Paper (§2.2, line 103):** "early stopping on validation NSE with patience of
  10 epochs (maximum 50 epochs)".
- **Code:** For the quantile and data-scarcity training
  (`train_quantile.py`, `train_data_scarce.py`) early stopping and LR scheduling
  monitor the **training** loss, with **patience 5** and **max 30 epochs**. The
  baseline (`train_model.py`) and ensemble (`train_ensembles_fair.py`) scripts do
  monitor validation loss (still patience 5 / max 30).
- **Suggested fix:** Reword to "early stopping (patience 5, maximum 30 epochs)";
  note that quantile/scarce runs stop on training loss while baselines stop on
  validation loss.

## 5. Sequence length & spin-up / 序列长度与 spin-up  🔴

- **Paper (§2.2, line 103):** "a sequence length of 365 days with a 365-day
  spin-up period for cell-state initialization".
- **Code:** There is **no separate spin-up**; the sliding window directly predicts
  the day after the 365-day input (`src/data/dataset.py`). Furthermore, the
  data-scarcity experiments use a **variable** sequence length —
  `seqlen_map = {1: 30, 3: 90, 5: 180, 15: 365}` in
  `experiments/scarce/train_data_scarce.py`.
- **Suggested fix:** Remove the spin-up claim, and state that the scarcity
  experiments use sequence lengths 30/90/180/365 days scaled to the training
  horizon.

## 6. Basin sampling for the scarcity experiment / 稀缺实验流域抽样  🔴

- **Paper (§2.1, line 99):** "50 basins were randomly selected …, stratified by
  aridity quartile to ensure hydroclimatic diversity."
- **Code:** Basins are selected by **uniform** random choice
  (`rng.choice(...)` in `train_data_scarce.py`). The paper's own Discussion (§4.4)
  already acknowledges the resulting humid/low-elevation bias.
- **Suggested fix:** Change "stratified by aridity quartile" to "randomly selected"
  in Methods (the Discussion already describes the limitation honestly).

## 7. Day-of-year input / Day-of-year 输入  🟢

- **Paper (Fig. 1 caption, line 108):** lists "day-of-year" among the dynamic
  inputs.
- **Code:** The model uses 5 dynamic forcings — `prcp, tmin, tmax, srad, vp`
  (`data_preprocessing.py`, `n_dynamic=5`). No day-of-year feature is used.
- **Suggested fix:** Remove "day-of-year" from the Fig. 1 caption (or add it to
  the inputs if intended — currently absent).

## 8. Flow-regime thresholds / 流量分级阈值  🟡

- **Paper:** Methods (§2.4, line 158) says low/normal/high are the **<25th /
  25–75th / >75th** percentiles; Results (§3.4, line 264) refers to high flows as
  the **top 10%**.
- **Code:** `compute_uncertainty_metrics` in `src/losses/cqr.py` splits regimes at
  the **33rd / 67th** percentiles (tertiles).
- **Suggested fix:** Pick one definition and state it consistently; the code uses
  33/67.

---

## Result-file provenance note / 结果文件来源说明

The committed JSON files under `results/tables/` are example outputs from the
actual experimental runs. They are **seed-sensitive**:

- `scarce_1yr_results.json` reports `test_nse = 0.684`, whereas the paper's
  Table 2 headline for the 1-year row is `NSE = 0.665`. The README multi-seed
  table explains this: `0.665` is seed 42, `0.684` is seed 456 — i.e. this
  particular JSON was produced by a different seed than the Table 2 headline run.
  The paper's headline numbers are the authoritative ones; the JSON files are
  included to illustrate output format and ballpark magnitudes.

`results/tables/` 下的 JSON 是实际实验运行的示例输出，且**对随机种子敏感**：
`scarce_1yr_results.json` 的 `test_nse = 0.684` 与论文 Table 2 头条 `NSE = 0.665` 不一致
——0.665 对应 seed 42、0.684 对应 seed 456（见 README 多 seed 表）。论文头条数字为权威值，
JSON 仅用于展示输出格式与量级。
