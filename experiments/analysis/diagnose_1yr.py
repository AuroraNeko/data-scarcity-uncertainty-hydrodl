"""diagnose_1yr.py — Retrain the 1-year scarcity model and compute the coverage-
degradation diagnosis from REAL predictions, saving results/tables/diagnosis_1yr.json
so Fig 3 can be regenerated from an artifact instead of the current (incorrect)
hard-coded values.

Reuses the exact training procedure of train_data_scarce.py (train-loss selection,
50 basins, seed 42, seq_len 30) so the model matches Table 2's 1-year row.
"""
import sys
import json
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'experiments' / 'scarce'))

from src.models.lpu_stream import LPUStreamModel
from src.data.dataset import CamelsDataset, get_basin_list
from src.data.data_preprocessing import VAL_START, VAL_END, TEST_START, TEST_END
from src.utils import get_device, set_seed
from train_data_scarce import train_model, predict

QUANTILES = [0.05, 0.5, 0.95]
device = get_device()
set_seed(42)

# --- identical setup to `train_data_scarce.py --years 1 --n_basins 50 --seed 42` ---
all_basins = get_basin_list()
rng = np.random.RandomState(42)
selected = rng.choice(all_basins, 50, replace=False).tolist()
seq_len, train_start, train_end = 30, "1980-10-01", "1981-09-30"
print(f"1-yr scarcity model: {len(selected)} basins, seq_len={seq_len}", flush=True)

train_ds = CamelsDataset(selected, train_start, train_end, seq_len)
val_ds = CamelsDataset(selected, VAL_START, VAL_END, seq_len)
test_ds = CamelsDataset(selected, TEST_START, TEST_END, seq_len)
bs = 512
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=bs, shuffle=False, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=bs, shuffle=False, pin_memory=True)

model = LPUStreamModel(quantiles=QUANTILES).to(device)
best_loss, best_epoch = train_model(model, train_loader, val_loader, device, epochs=30)
print(f"trained: best_epoch={best_epoch} loss={best_loss:.4f}", flush=True)

# --- predictions ---
tp, tt, tm = predict(model, test_loader, device)
v = tm.flatten() > 0
q05, q50, q95 = tp[v, 0], tp[v, 1], tp[v, 2]
y = tt.flatten()[v]
width = q95 - q05

nse = float(1 - np.sum((y - q50) ** 2) / np.sum((y - y.mean()) ** 2))
picp = float(((y >= q05) & (y <= q95)).mean())
mpiw = float(width.mean())
print(f"NSE={nse:.4f} PICP={picp:.4f} MPIW={mpiw:.4f}", flush=True)

# --- flow regime split (33/67 pct of observations) ---
q33, q67 = np.percentile(y, [33, 67])
masks = {'low': y <= q33, 'normal': (y > q33) & (y <= q67), 'high': y > q67}


def regime_stats(mask):
    w = width[mask]; yy = y[mask]; lo = q05[mask]; hi = q95[mask]; med = q50[mask]
    err_std = float(np.std(yy - med))
    wr = float(w.mean() / (2 * 1.645 * err_std)) if err_std > 0 else float('nan')
    return {'picp': float(((yy >= lo) & (yy <= hi)).mean()),
            'mpiw': float(w.mean()), 'width_ratio': wr, 'n': int(mask.sum())}


regime = {k: regime_stats(m) for k, m in masks.items()}
err_std_all = float(np.std(y - q50))
width_ratio_overall = float(mpiw / (2 * 1.645 * err_std_all))

# --- calibration curve: PICP by predicted-width quartile (4 bins) ---
wq = np.percentile(width, [25, 50, 75])
bins = np.digitize(width, wq)
cal_curve = []
for b in range(4):
    msk = bins == b
    if msk.sum() > 0:
        cal_curve.append(float(((y[msk] >= q05[msk]) & (y[msk] <= q95[msk])).mean()))
    else:
        cal_curve.append(float('nan'))

# --- coverage by regime across years (1yr from this run; 3/5/15 from JSONs) ---
def uncov(yrs):
    if yrs == 1:
        return {k: regime[k]['picp'] for k in ['low', 'normal', 'high']}
    f = {'3': 'scarce_3yr_results.json', '5': 'scarce_5yr_results.json',
         '15': 'scarce_15yr_results.json'}[str(yrs)]
    u = json.load(open(PROJECT_ROOT / 'results' / 'tables' / f))['test_uncalibrated']
    return {'low': u['coverage_low_flow'], 'normal': u['coverage_normal_flow'],
            'high': u['coverage_high_flow']}


across_years = {str(y): uncov(y) for y in [1, 3, 5, 15]}

# --- CQR calibration: does global CQR fix high-flow conditional coverage? ---
# (global CQR restores MARGINAL coverage; conditional/high-flow is the question)
vp, vt, vm = predict(model, val_loader, device)
vv = vm.flatten() > 0
v05, v50, v95 = vp[vv, 0], vp[vv, 1], vp[vv, 2]
yv = vt.flatten()[vv]
alpha = 0.1
conf = np.maximum(v05 - yv, yv - v95)          # nonconformity scores on val
n_cal = len(conf)
q_cal_global = float(np.quantile(conf, (1 - alpha) * (1 + 1 / n_cal)))


def cqr_picp(lo, hi, yy, qc):
    L, H = lo - qc, hi + qc
    return float(((yy >= L) & (yy <= H)).mean())


# global CQR applied to test, coverage by (test) flow regime
cqr_global_regime = {k: cqr_picp(q05[m], q95[m], y[m], q_cal_global)
                     for k, m in masks.items()}
# per-regime (conditional) CQR: calibrate separately per val regime, apply to test regime
vq33, vq67 = np.percentile(yv, [33, 67])
vmasks = {'low': yv <= vq33, 'normal': (yv > vq33) & (yv <= vq67), 'high': yv > vq67}
q_cal_regime, cqr_perregime = {}, {}
for k, m in vmasks.items():
    c = conf[m]
    q_cal_regime[k] = float(np.quantile(c, (1 - alpha) * (1 + 1 / len(c))) if len(c) > 0 else 0.0)
    cqr_perregime[k] = cqr_picp(q05[masks[k]], q95[masks[k]], y[masks[k]], q_cal_regime[k])

# deployable conditional CQR: use predicted median-flow tertiles instead of
# observed-flow tertiles. Calibration and test-time assignment both depend only
# on model outputs, so this variant can be used before observations arrive.
pred_q33, pred_q67 = np.percentile(v50, [33, 67])
pred_vmasks = {
    'low': v50 <= pred_q33,
    'normal': (v50 > pred_q33) & (v50 <= pred_q67),
    'high': v50 > pred_q67,
}
pred_tmasks = {
    'low': q50 <= pred_q33,
    'normal': (q50 > pred_q33) & (q50 <= pred_q67),
    'high': q50 > pred_q67,
}
q_cal_predregime = {}
qc_by_sample = np.zeros_like(y, dtype=np.float32)
for k, m in pred_vmasks.items():
    c = conf[m]
    q_cal_predregime[k] = float(np.quantile(c, (1 - alpha) * (1 + 1 / len(c))) if len(c) > 0 else 0.0)
    qc_by_sample[pred_tmasks[k]] = q_cal_predregime[k]
pred_lo, pred_hi = q05 - qc_by_sample, q95 + qc_by_sample
cqr_predregime_overall = {
    'picp': float(((y >= pred_lo) & (y <= pred_hi)).mean()),
    'mpiw': float((pred_hi - pred_lo).mean()),
}
cqr_predregime_by_observed_regime = {
    k: float(((y[m] >= pred_lo[m]) & (y[m] <= pred_hi[m])).mean())
    for k, m in masks.items()
}
cqr_predregime_by_predicted_regime = {
    k: float(((y[m] >= pred_lo[m]) & (y[m] <= pred_hi[m])).mean())
    for k, m in pred_tmasks.items()
}
predregime_counts = {k: {'cal': int(pred_vmasks[k].sum()), 'test': int(pred_tmasks[k].sum())}
                     for k in pred_vmasks}
print(f"\nCQR global q_cal={q_cal_global:.4f}; per-regime q_cal={ {k:round(v,4) for k,v in q_cal_regime.items()} }", flush=True)
print(f"per-regime PICP  raw={ {k:round(regime[k]['picp'],3) for k in regime} }", flush=True)
print(f"                 CQR-global={ {k:round(v,3) for k,v in cqr_global_regime.items()} }", flush=True)
print(f"                 CQR-perregime={ {k:round(v,3) for k,v in cqr_perregime.items()} }", flush=True)
print(f"                 CQR-predregime={ {k:round(v,3) for k,v in cqr_predregime_by_observed_regime.items()} }", flush=True)

out = {
    'model': '1yr_scarcity_diagnosis', 'n_basins': 50, 'seq_len': seq_len,
    'nse': nse, 'picp': picp, 'mpiw': mpiw,
    'width_ratio_overall': width_ratio_overall,
    'width_ratio_by_regime': regime,
    'calibration_curve_by_width_quartile': cal_curve,
    'coverage_by_regime_across_years': across_years,
    'cqr_global_q_cal': q_cal_global,
    'cqr_perregime_q_cal': q_cal_regime,
    'cqr_predregime_q_cal': q_cal_predregime,
    'cqr_predregime_thresholds': {'q50_33': float(pred_q33), 'q50_67': float(pred_q67)},
    'cqr_predregime_counts': predregime_counts,
    'coverage_by_regime_cqr_global': cqr_global_regime,
    'coverage_by_regime_cqr_perregime': cqr_perregime,
    'coverage_by_regime_cqr_predregime': cqr_predregime_by_observed_regime,
    'coverage_by_predicted_regime_cqr_predregime': cqr_predregime_by_predicted_regime,
    'cqr_predregime_overall': cqr_predregime_overall,
}
with open(PROJECT_ROOT / 'results' / 'tables' / 'diagnosis_1yr.json', 'w') as f:
    json.dump(out, f, indent=2)
print("\n=== diagnosis_1yr.json ===")
print(json.dumps(out, indent=2))
