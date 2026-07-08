"""
analyze_degradation.py  -  Analyze why uncertainty coverage degrades with less data.

Questions:
  1. Are extreme events underrepresented in scarce training data?
  2. Are the intervals too narrow (overconfidence) or biased?
  3. Which basins degrade most? What predicts degradation?
  4. Is there a seasonal pattern?
"""
import sys, json, numpy as np, torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lpu_stream import LPUStreamModel
from src.data.dataset import CamelsDataset, get_basin_list
from src.data.data_preprocessing import TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END
from src.utils import get_device

QUANTILES = [0.05, 0.5, 0.95]
device = get_device()

# Load quantile checkpoint
ckpt = torch.load(PROJECT_ROOT / 'results/checkpoints/lpu_stream_quantile_best.pt', weights_only=False)
cfg = ckpt['config']
print(f"Using checkpoint: epoch {ckpt['epoch']}, Val NSE {ckpt['val_nse']:.4f}")

# --- Load static attributes for analysis ---
import pandas as pd
basins = get_basin_list()
attr_data = {}
for bid in basins:
    csv_path = PROJECT_ROOT / 'data/processed/camels_us' / f'{bid}.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        attr_data[bid] = {
            'aridity': df.get('aridity_norm', df.get('aridity', pd.Series([0]))).iloc[0],
            'area': df.get('area_gages2_norm', df.get('area_gages2', pd.Series([0]))).iloc[0],
            'p_mean': df.get('p_mean_norm', df.get('p_mean', pd.Series([0]))).iloc[0],
        }

# --- 1. Extreme event representation in training data ---
print("\n" + "=" * 60)
print("1. EXTREME EVENT REPRESENTATION IN TRAINING DATA")
print("=" * 60)

year_map = {1: "1981-09-30", 3: "1983-09-30", 5: "1985-09-30", 15: TRAIN_END}
for years, end_date in year_map.items():
    ds = CamelsDataset(basins[:50], TRAIN_START, end_date, seq_len=30)
    q95 = ds.compute_q95()
    all_t = []
    for bd in ds.basin_data:
        valid = bd['mask'] > 0
        all_t.append(bd['target'][valid])
    all_t = np.concatenate(all_t)
    pct_above_q95 = (all_t > q95).mean() * 100
    n_extreme = (all_t > q95).sum()
    print(f"  {years:2d}yr: Q95={q95:.4f}, samples={len(all_t):,}, extreme={n_extreme:,} ({pct_above_q95:.1f}%)")

# Load full training data Q95
full_ds = CamelsDataset(basins[:50], TRAIN_START, TRAIN_END, seq_len=365)
full_q95 = full_ds.compute_q95()
print(f"  Full: Q95={full_q95:.4f}")

# --- 2. Per-basin coverage analysis ---
print("\n" + "=" * 60)
print("2. PER-BASIN COVERAGE VS BASIN ATTRIBUTES")
print("=" * 60)

def predict_for_basins(model, basin_list, start, end, seq_len=365):
    ds = CamelsDataset(basin_list, start, end, seq_len, preview=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=False, pin_memory=True)
    preds, targets, masks = [], [], []
    model.eval()
    with torch.no_grad():
        for dynamic, static, target, mask, _, _ in loader:
            pred = model(dynamic.to(device), static.to(device))
            preds.append(pred.cpu().numpy())
            targets.append(target.numpy())
            masks.append(mask.numpy())
    return np.concatenate(preds), np.concatenate(targets), np.concatenate(masks)

model = LPUStreamModel(quantiles=QUANTILES).to(device)
model.load_state_dict(ckpt['model_state_dict'])

test_basins = basins[:50]
test_preds, test_targets, test_masks = predict_for_basins(model, test_basins, TEST_START, TEST_END)

per_basin = {}
for i, bid in enumerate(test_basins):
    if i < len(ds := CamelsDataset([bid], TEST_START, TEST_END, 365)):
        p, t, m = predict_for_basins(model, [bid], TEST_START, TEST_END)
        valid = m.flatten() > 0
        q05, q95 = p[valid, 0], p[valid, 2]
        y = t.flatten()[valid]
        in_ci = (y >= q05) & (y <= q95)
        picp = in_ci.mean()
        mpw = (q95 - q05).mean()
        nse_median = 1 - np.sum((y - p[valid, 1])**2) / np.sum((y - y.mean())**2)
        per_basin[bid] = {
            'picp': picp, 'mpw': mpw, 'nse': nse_median, 'n': len(y),
            **attr_data.get(bid, {}),
        }

bids_sorted = sorted(per_basin.keys(), key=lambda b: per_basin[b]['picp'])
print(f"  Best 5 basins (PICP):")
for b in bids_sorted[-5:]:
    print(f"    {b}: PICP={per_basin[b]['picp']:.3f}, NSE={per_basin[b]['nse']:.3f}")
print(f"  Worst 5 basins (PICP):")
for b in bids_sorted[:5]:
    print(f"    {b}: PICP={per_basin[b]['picp']:.3f}, NSE={per_basin[b]['nse']:.3f}")

# Correlation: PICP vs aridity
aridities = np.array([per_basin[b].get('aridity', 0) for b in bids_sorted])
picps = np.array([per_basin[b]['picp'] for b in bids_sorted])
nses = np.array([per_basin[b]['nse'] for b in bids_sorted])
valid_a = np.isfinite(aridities) & np.isfinite(picps)
print(f"\n  PICP vs Aridity correlation: {np.corrcoef(aridities[valid_a], picps[valid_a])[0,1]:.3f}")
print(f"  PICP vs NSE correlation: {np.corrcoef(nses[valid_a], picps[valid_a])[0,1]:.3f}")

# --- 3. Overconfidence vs bias analysis ---
print("\n" + "=" * 60)
print("3. OVERCONFIDENCE VS BIAS")
print("=" * 60)
# For each data scenario, check: intervals too narrow? or biased?
# "Overconfidence" = intervals narrower than needed with low coverage
# "Bias" = intervals shifted from true distribution

for yr_tag, (years, end_date, seq_len) in [
    ("1yr", 1, "1981-09-30", 30),
    ("5yr", 5, "1985-09-30", 180),
]:
    # Train quick model
    train_ds = CamelsDataset(basins[:50], TRAIN_START, end_date, seq_len, preview=False)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1024, shuffle=True)

    m = LPUStreamModel(quantiles=QUANTILES).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    from src.losses.pinball_loss import PinballLoss
    crit = PinballLoss(QUANTILES)
    for ep in range(10):
        m.train()
        for dyn, stat, tgt, msk, _, _ in train_loader:
            dyn, stat, tgt, msk = dyn.to(device), stat.to(device), tgt.to(device), msk.to(device)
            opt.zero_grad()
            loss = crit(m(dyn, stat), tgt, msk)
            loss.backward()
            opt.step()

    p, t, mk = predict_for_basins(m, basins[:50], TEST_START, TEST_END)
    valid = mk.flatten() > 0
    q05, q50, q95 = p[valid, 0], p[valid, 1], p[valid, 2]
    y = t.flatten()[valid]

    # Interval width distribution
    widths = q95 - q05
    in_ci = (y >= q05) & (y <= q95)
    picp = in_ci.mean()

    # Bias: median of errors (Q_0.5 - y)
    bias = np.median(q50 - y)

    # Overconfidence check: mean interval width vs standard deviation of errors
    error_std = np.std(y - q50)

    print(f"  {yr_tag}: PICP={picp:.3f}, IntervalWidth={widths.mean():.3f}, "
          f"Bias(median)={bias:.4f}, ErrorStd={error_std:.3f}, "
          f"Ratio(width/2*errorStd)={widths.mean()/(2*1.645*error_std):.3f}")

print("\nNote: Ratio < 1 = intervals too narrow (overconfidence), > 1 = too wide (overcautious)")

# --- 4. Seasonality of coverage failure ---
print("\n" + "=" * 60)
print("4. SEASONAL PATTERN OF COVERAGE FAILURES")
print("=" * 60)

# Use full model predictions
valid = test_masks.flatten() > 0
preds_v = test_preds[valid]
y_v = test_targets.flatten()[valid]

# Simulate dates by using day-of-year indices
n_valid = len(y_v)
# Map indices to approximate month (using test period length)
month_sim = np.tile(np.arange(1, 13), n_valid // 12 + 1)[:n_valid]
in_ci_all = (y_v >= preds_v[:, 0]) & (y_v <= preds_v[:, 2])

for month in range(1, 13):
    mask_m = month_sim == month
    if mask_m.sum() > 100:
        picp_m = in_ci_all[mask_m].mean()
        width_m = (preds_v[mask_m, 2] - preds_v[mask_m, 0]).mean()
        print(f"  Month {month:2d}: PICP={picp_m:.4f}, MPIW={width_m:.4f}, n={mask_m.sum():,}")

print("\n=== Analysis complete ===")
