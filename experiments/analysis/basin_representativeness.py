"""
basin_representativeness.py — Check whether the 50 scarcity-experiment basins
are representative of the full 671-basin CAMELS-US set.

Compares: aridity, p_mean, pet_mean, elev_mean, frac_forest, slope_mean,
area_gages2, frac_snow, p_seasonality, soil_depth, soil_porosity.
Uses Kolmogorov-Smirnov tests and visual comparison.
"""

import sys, json, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.dataset import PROCESSED_DIR, METADATA_DIR

# Get 50 scarcity basins (from the experiment config)
# The first 50 basins in sorted order were used
all_basins = sorted([f.stem for f in PROCESSED_DIR.glob('*.csv')])
scarcity_basins = set(all_basins[:50])

print(f"All basins: {len(all_basins)}")
print(f"Scarcity basins: {len(scarcity_basins)}")
print(f"Scarcity basins IDs: {sorted(scarcity_basins)[:5]}...{sorted(scarcity_basins)[-3:]}")

# Read static attributes from each basin's CSV
attrs_all = {bid: {} for bid in all_basins}
static_cols = ['elev_mean', 'slope_mean', 'area_gages2', 'p_mean', 'pet_mean',
               'aridity', 'frac_snow', 'p_seasonality', 'soil_depth_pelletier',
               'soil_porosity', 'frac_forest', 'lai_diff', 'geol_porostiy']

for csv in PROCESSED_DIR.glob('*.csv'):
    bid = csv.stem
    df = pd.read_csv(csv, index_col=0, parse_dates=True)
    for col in static_cols:
        if col in df.columns:
            attrs_all[bid][col] = float(df[col].iloc[0])

# Build arrays
attr_data = {col: {'all': [], 'scarcity': []} for col in static_cols}
for bid, attrs in attrs_all.items():
    for col in static_cols:
        if col in attrs:
            attr_data[col]['all'].append(attrs[col])
            if bid in scarcity_basins:
                attr_data[col]['scarcity'].append(attrs[col])

# Compare distributions
print(f"\n{'='*80}")
print(f"{'Attribute Representativeness: Scarcity 50 vs All 671':^80}")
print(f"{'='*80}")
print(f"{'Attribute':<25} {'All Mean':>10} {'All Std':>10} {'S50 Mean':>10} {'S50 Std':>10} {'KS p-val':>10} {'Diff':>10}")
print(f"{'-'*80}")

results = {}
for col in static_cols:
    all_arr = np.array(attr_data[col]['all'])
    s50_arr = np.array(attr_data[col]['scarcity'])
    if len(all_arr) == 0 or len(s50_arr) == 0:
        continue
    
    all_mean, all_std = all_arr.mean(), all_arr.std()
    s50_mean, s50_std = s50_arr.mean(), s50_arr.std()
    
    # KS test
    ks_stat, ks_p = stats.ks_2samp(all_arr, s50_arr)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((all_std**2 + s50_std**2) / 2)
    cohens_d = (all_mean - s50_mean) / pooled_std if pooled_std > 0 else 0
    
    results[col] = {'all_mean': all_mean, 'all_std': all_std, 
                    's50_mean': s50_mean, 's50_std': s50_std,
                    'ks_p': ks_p, 'cohens_d': cohens_d,
                    'diff_pct': (s50_mean - all_mean) / abs(all_mean) * 100 if all_mean != 0 else 0}
    
    sig = " ***" if ks_p < 0.001 else (" **" if ks_p < 0.01 else (" *" if ks_p < 0.05 else ""))
    print(f"{col:<25} {all_mean:>10.4f} {all_std:>10.4f} {s50_mean:>10.4f} {s50_std:>10.4f} {ks_p:>10.4f}{sig} {results[col]['diff_pct']:>+9.1f}%")

print(f"{'-'*80}")
print(f"Significance: * p<0.05, ** p<0.01, *** p<0.001")

# Count significant differences
sig_count = sum(1 for r in results.values() if r['ks_p'] < 0.05)
print(f"\nSignificantly different attributes (p<0.05): {sig_count}/{len(results)}")

if sig_count == 0:
    print(">>> The 50 scarcity basins are statistically representative of the full 671-basin set.")
else:
    print(f">>> {sig_count} attributes show significant differences. Review carefully.")
    for col, r in results.items():
        if r['ks_p'] < 0.05:
            print(f"    - {col}: All={r['all_mean']:.4f} vs S50={r['s50_mean']:.4f} (diff={r['diff_pct']:+.1f}%)")

# Save results
out_path = PROJECT_ROOT / "results" / "tables" / "basin_representativeness.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")
