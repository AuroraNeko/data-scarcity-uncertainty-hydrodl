"""
train_climatology.py — Climatology baseline for CAMELS-US.

Predicts streamflow using the historical mean for each day-of-year.
Simplest possible baseline — required by hydrology reviewers.

Usage:
    python experiments/baseline/train_climatology.py
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "camels_us"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

TRAIN_START, TRAIN_END = "1980-10-01", "1995-09-30"
TEST_START, TEST_END = "2000-10-01", "2010-09-30"


def compute_nse(preds, targets):
    valid = ~np.isnan(targets) & ~np.isnan(preds)
    if valid.sum() == 0:
        return 0.0
    p, t = preds[valid], targets[valid]
    return 1.0 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2) if np.sum((t - t.mean()) ** 2) > 0 else 0.0


def main():
    print("=" * 60)
    print("Climatology Baseline — CAMELS-US")
    print("=" * 60)

    basins = pd.read_csv(METADATA_DIR / "basin_metadata.csv", dtype={"basin_id": str})["basin_id"].tolist()
    print(f"Basins: {len(basins)}")

    basin_nses = []

    for i, bid in enumerate(basins):
        csv_path = PROCESSED_DIR / f"{bid}.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        train = df.loc[TRAIN_START:TRAIN_END]
        test = df.loc[TEST_START:TEST_END]

        # Historical daily mean streamflow (mm/day) per day-of-year
        train_obs = train[train["flow_mask"] == 1]["streamflow_mm"]
        if len(train_obs) < 100:
            continue

        climatology = train_obs.groupby(train_obs.index.dayofyear).mean()

        # Predict test period using climatology
        test_obs_mask = test["flow_mask"] == 1
        if test_obs_mask.sum() == 0:
            continue

        test_doy = test.index.dayofyear
        test_actual = test.loc[test_obs_mask, "streamflow_mm"]
        test_pred = test_doy[test_obs_mask].map(climatology)

        # Drop DOYs not in climatology (Feb 29 edge case)
        valid = test_pred.notna() & test_actual.notna()
        if valid.sum() < 30:
            continue

        nse = compute_nse(test_pred[valid].values, test_actual[valid].values)
        basin_nses.append(nse)

    basin_nses = np.array(basin_nses)
    print(f"\nBasins evaluated: {len(basin_nses)}")
    print(f"Median NSE:  {np.median(basin_nses):.4f}")
    print(f"Mean NSE:    {np.mean(basin_nses):.4f}")
    print(f"NSE > 0:     {(basin_nses > 0).sum()}/{len(basin_nses)}")
    print(f"NSE > 0.5:   {(basin_nses > 0.5).sum()}/{len(basin_nses)}")

    results = {
        "model": "Climatology",
        "n_basins": len(basin_nses),
        "median_nse": float(np.median(basin_nses)),
        "mean_nse": float(np.mean(basin_nses)),
        "nse_gt0": int((basin_nses > 0).sum()),
        "nse_gt05": int((basin_nses > 0.5).sum()),
        "timestamp": datetime.now().isoformat(),
    }
    results_path = PROJECT_ROOT / "results" / "tables" / "climatology_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
