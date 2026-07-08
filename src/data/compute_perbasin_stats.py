"""compute_perbasin_stats.py — Per-basin target normalization statistics
(mean/std of log1p(flow)) from the TRAINING period only, for each basin.

Per-basin target normalization is the CAMELS literature convention and removes
the inter-basin variance that inflates pooled-normalized NSE.
"""
import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / 'data' / 'processed' / 'camels_us'
TRAIN_START, TRAIN_END = '1980-10-01', '1995-09-30'

basins = pd.read_csv(ROOT / 'data' / 'metadata' / 'basin_metadata.csv',
                     dtype={'basin_id': str})['basin_id'].tolist()
stats = {}
for bid in basins:
    csv_path = PROC / f'{bid}.csv'
    if not csv_path.exists():
        continue
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    tr = df.loc[TRAIN_START:TRAIN_END]
    m = tr['flow_mask'] == 1
    t = tr.loc[m, 'target']  # log1p(flow)
    if len(t) > 30:
        stats[bid] = {'mean': float(t.mean()), 'std': float(t.std())}

out = ROOT / 'data' / 'metadata' / 'per_basin_target_stats.json'
with open(out, 'w') as f:
    json.dump(stats, f, indent=2)
print(f"Computed per-basin target stats for {len(stats)} basins -> {out}")
# quick peek at spread
import numpy as np
means = [v['mean'] for v in stats.values()]
print(f"basin log-flow mean: min={min(means):.3f} median={np.median(means):.3f} max={max(means):.3f}")
