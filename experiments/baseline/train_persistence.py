"""
train_persistence.py  -  Persistence baseline for CAMELS-US.

Predicts today's streamflow as yesterday's streamflow.
Simple reference baseline for hydrology model comparison.

Usage:
    python experiments/baseline/train_persistence.py
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

TEST_START, TEST_END = "2000-10-01", "2010-09-30"


def compute_nse(preds, targets):
    valid = ~np.isnan(targets) & ~np.isnan(preds)
    if valid.sum() == 0:
        return 0.0
    p, t = preds[valid], targets[valid]
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def main():
    print("=" * 60)
    print("Persistence Baseline  -  CAMELS-US")
    print("=" * 60)

    basins = pd.read_csv(METADATA_DIR / "basin_metadata.csv", dtype={"basin_id": str})["basin_id"].tolist()
    print(f"Basins: {len(basins)}")

    basin_nses = []

    for bid in basins:
        csv_path = PROCESSED_DIR / f"{bid}.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        test = df.loc[TEST_START:TEST_END]

        # Only use observed flow
        observed = test[test["flow_mask"] == 1]["streamflow_mm"]
        if len(observed) < 30:
            continue

        # Predict: today = yesterday
        actual = observed.values[1:]
        pred = observed.values[:-1]

        valid = ~np.isnan(actual) & ~np.isnan(pred)
        if valid.sum() < 30:
            continue

        nse = compute_nse(pred[valid], actual[valid])
        basin_nses.append(nse)

    basin_nses = np.array(basin_nses)
    print(f"\nBasins evaluated: {len(basin_nses)}")
    print(f"Median NSE:  {np.median(basin_nses):.4f}")
    print(f"Mean NSE:    {np.mean(basin_nses):.4f}")
    print(f"NSE > 0:     {(basin_nses > 0).sum()}/{len(basin_nses)}")
    print(f"NSE > 0.5:   {(basin_nses > 0.5).sum()}/{len(basin_nses)}")

    results = {
        "model": "Persistence",
        "n_basins": len(basin_nses),
        "median_nse": float(np.median(basin_nses)),
        "mean_nse": float(np.mean(basin_nses)),
        "nse_gt0": int((basin_nses > 0).sum()),
        "nse_gt05": int((basin_nses > 0.5).sum()),
        "timestamp": datetime.now().isoformat(),
    }
    results_path = PROJECT_ROOT / "results" / "tables" / "persistence_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
