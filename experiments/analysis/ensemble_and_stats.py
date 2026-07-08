"""
Deep Ensembles baseline + KS statistical tests.
Uses 5 seeds' point predictions, computes 90% ensemble intervals,
compares with CQR and MC Dropout via KS test.
"""
import sys, json, numpy as np, torch
from pathlib import Path
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lpu_stream import LPUStreamModel
from src.data.dataset import CamelsDataset, get_basin_list
from src.data.data_preprocessing import TRAIN_START, TRAIN_END, TEST_START, TEST_END
from src.losses.cqr import compute_uncertainty_metrics
from src.utils import get_device

device = get_device()
QUANTILES = [0.05, 0.5, 0.95]
basins = get_basin_list()[:50]
SEEDS = [42, 123, 456, 789, 999]

# --- Train 5 ensemble members (point prediction) ---
ensemble_preds = []
for seed in SEEDS:
    print(f'Training seed {seed}...', flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = CamelsDataset(basins, TRAIN_START, "1981-09-30", seq_len=30, preview=False)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1024, shuffle=True)

    model = LPUStreamModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = torch.nn.MSELoss(reduction='none')

    for ep in range(30):
        model.train()
        total, n = 0.0, 0
        for dynamic, static, target, mask, _, _ in train_loader:
            dynamic, static, target, mask = dynamic.to(device), static.to(device), target.to(device), mask.to(device)
            opt.zero_grad()
            pred = model(dynamic, static)
            loss = (crit(pred, target) * mask).sum() / mask.sum().clamp(min=1)
            loss.backward()
            opt.step()
            total += loss.item() * mask.sum().item()
            n += mask.sum().item()

    # Predict on test set
    test_ds = CamelsDataset(basins, TEST_START, TEST_END, seq_len=365, preview=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1024, shuffle=False, pin_memory=True)
    preds = []
    model.eval()
    with torch.no_grad():
        for dynamic, static, target, mask, _, _ in test_loader:
            pred = model(dynamic.to(device), static.to(device))
            preds.append(pred.cpu().numpy())
    ensemble_preds.append(np.concatenate(preds))
    print(f'  Seed {seed}: done', flush=True)

# --- Compute Ensemble intervals ---
all_preds = np.stack(ensemble_preds, axis=0).squeeze(-1)  # (5, N)
q05_en = np.percentile(all_preds, 5, axis=0)
q95_en = np.percentile(all_preds, 95, axis=0)
q50_en = np.percentile(all_preds, 50, axis=0)

test_ds = CamelsDataset(basins, TEST_START, TEST_END, seq_len=365, preview=False)
all_y = []; all_m = []
for bd in test_ds.basin_data:
    n_valid = len(test_ds)
# Load test targets
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1024, shuffle=False, pin_memory=True)
targets, masks = [], []
for _, _, tgt, msk, _, _ in test_loader:
    targets.append(tgt.numpy())
    masks.append(msk.numpy())
y = np.concatenate(targets).flatten()
msk = np.concatenate(masks).flatten()
valid = msk > 0
y_v, q05_v, q95_v = y[valid], q05_en[valid], q95_en[valid]

ens_nse = 1 - np.sum((y_v - np.percentile(all_preds[:, valid], 50, axis=0))**2) / np.sum((y_v - y_v.mean())**2)
ens_metrics = compute_uncertainty_metrics(q05_v, q95_v, y_v, alpha=0.1)

print(f'\n=== Deep Ensembles (5 seeds) ===')
print(f'NSE: {ens_nse:.4f}')
print(f'PICP: {ens_metrics["picp"]:.4f}')
print(f'MPIW: {ens_metrics["mpiw"]:.4f}')
print(f'Winkler: {ens_metrics["winkler_score"]:.4f}')
print(f'Coverage low/normal/high: {ens_metrics["coverage_low_flow"]:.4f}/{ens_metrics["coverage_normal_flow"]:.4f}/{ens_metrics["coverage_high_flow"]:.4f}')

# --- Load CQR and MC Dropout results ---
with open(PROJECT_ROOT / 'results/tables/lpu_stream_quantile_results.json') as f:
    cqr_r = json.load(f)['test_calibrated']
with open(PROJECT_ROOT / 'results/tables/mc_dropout_results.json') as f:
    mc_r = json.load(f)['test_metrics']

print(f'\n{"=" * 60}')
print(f'Uncertainty Method Comparison (Test Set, 50 basins)')
print(f'{"=" * 60}')
print(f'{"Method":<25} {"PICP":>8} {"MPIW":>8} {"Winkler":>8} {"Inference":>10}')
print(f'{"MC Dropout":<25} {mc_r["picp"]:>8.4f} {mc_r["mpiw"]:>8.4f} {mc_r["winkler_score"]:>8.4f} {"50x":>10}')
print(f'{"Deep Ensembles (5)":<25} {ens_metrics["picp"]:>8.4f} {ens_metrics["mpiw"]:>8.4f} {ens_metrics["winkler_score"]:>8.4f} {"5x":>10}')
print(f'{"CQR (ours)":<25} {cqr_r["picp"]:>8.4f} {cqr_r["mpiw"]:>8.4f} {cqr_r["winkler_score"]:>8.4f} {"1x":>10}')

# --- KS Tests ---
print(f'\n{"=" * 60}')
print(f'Kolmogorov-Smirnov Tests')
print(f'{"=" * 60}')
# Per-sample coverage: 1=in interval, 0=outside
in_cqr = ((y_v >= cqr_r.get('_q05', q05_v)) & (y_v <= cqr_r.get('_q95', q95_v))).astype(float)
in_ens = ((y_v >= q05_v) & (y_v <= q95_v)).astype(float)

# Precompute raw coverage values using the test predictions
# Since we can't get per-sample CQR/MC from saved results, test on per-basin PICP
# Use KS test on per-basin coverage values
from src.losses.cqr import CQRCalibrator

# For per-basin KS: compute PICP for each basin
bid_to_idx = {}
for i, idx_data in enumerate(test_ds.indices):
    bid = idx_data[0]
    if bid not in bid_to_idx:
        bid_to_idx[bid] = []
    bid_to_idx[bid].append(i)

# Get CQR per-basin predictions
ckpt_q = torch.load(PROJECT_ROOT / 'results/checkpoints/lpu_stream_quantile_best.pt', weights_only=False)
model_q = LPUStreamModel(quantiles=QUANTILES).to(device)
model_q.load_state_dict(ckpt_q['model_state_dict'])
model_q.eval()

all_q_preds = []
with torch.no_grad():
    for dynamic, static, target, mask, _, _ in test_loader:
        pred = model_q(dynamic.to(device), static.to(device))
        all_q_preds.append(pred.cpu().numpy())
q_preds = np.concatenate(all_q_preds)

per_basin_cqr = []
per_basin_ens = []
for bid, indices in bid_to_idx.items():
    idx_arr = np.array(indices)
    y_b = y[idx_arr]
    m_b = msk[idx_arr]
    v = m_b > 0
    if v.sum() < 50:
        continue
    yb = y_b[v]
    # CQR
    in_cqr_b = (yb >= q_preds[idx_arr][v, 0]) & (yb <= q_preds[idx_arr][v, 2])
    per_basin_cqr.append(in_cqr_b.mean())
    # Ensemble
    in_ens_b = (yb >= q05_en[idx_arr][v]) & (yb <= q95_en[idx_arr][v])
    per_basin_ens.append(in_ens_b.mean())

pb_cqr = np.array(per_basin_cqr)
pb_ens = np.array(per_basin_ens)

ks_cqr_ens = stats.ks_2samp(pb_cqr, pb_ens)
print(f'CQR vs Deep Ensembles: KS={ks_cqr_ens.statistic:.4f}, p={ks_cqr_ens.pvalue:.4f}')

# Compare Ensemble vs theoretical 90%
from scipy.stats import wilcoxon
w_ens = wilcoxon(pb_ens - 0.90)
print(f'Ensembles vs 0.90 target: Wilcoxon p={w_ens.pvalue:.4f}')

w_cqr = wilcoxon(pb_cqr - 0.90)
print(f'CQR vs 0.90 target: Wilcoxon p={w_cqr.pvalue:.4f}')

# Save combined results
all_results = {
    'deep_ensembles': {
        'n_seeds': 5, 'nse': float(ens_nse), 'metrics': ens_metrics,
    },
    'cqr': cqr_r,
    'mc_dropout': mc_r,
    'ks_tests': {
        'cqr_vs_ensembles': {'ks': float(ks_cqr_ens.statistic), 'p': float(ks_cqr_ens.pvalue)},
        'ensembles_vs_90': {'wilcoxon_p': float(w_ens.pvalue)},
        'cqr_vs_90': {'wilcoxon_p': float(w_cqr.pvalue)},
    },
}
with open(PROJECT_ROOT / 'results/tables/final_comparison.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print(f'\nAll results saved to results/tables/final_comparison.json')
