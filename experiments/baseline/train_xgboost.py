"""
train_xgboost.py — XGBoost baseline for CAMELS-US streamflow prediction.

Memory-efficient: builds features basin-by-basin, saves to intermediate
parquet files, then trains XGBoost from disk.

Usage:
    python experiments/baseline/train_xgboost.py
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
FEAT_DIR = PROJECT_ROOT / "data" / "processed" / "xgboost_features"

DYNAMIC_VARS = ["prcp", "tmin", "tmax", "srad", "vp"]
STATIC_COLS = ["elev_mean", "slope_mean", "area_gages2", "p_mean", "pet_mean",
               "aridity", "frac_snow", "p_seasonality", "soil_depth_pelletier",
               "soil_porosity", "frac_forest", "lai_diff", "geol_porostiy"]

TRAIN_START, TRAIN_END = "1980-10-01", "1995-09-30"
VAL_START, VAL_END = "1995-10-01", "2000-09-30"
TEST_START, TEST_END = "2000-10-01", "2010-09-30"


def compute_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute hand-crafted lag features from dynamic variables."""
    features = pd.DataFrame(index=df.index)

    for var in DYNAMIC_VARS:
        vals = df[var]
        features[f"{var}_t0"] = vals
        for lag in [1, 3, 7, 30]:
            features[f"{var}_lag{lag}"] = vals.shift(lag)
        for window in [3, 7, 30]:
            features[f"{var}_rmean{window}"] = vals.rolling(window, min_periods=1).mean()
        features[f"{var}_rsum7"] = vals.rolling(7, min_periods=1).sum()
        features[f"{var}_rsum30"] = vals.rolling(30, min_periods=1).sum()

    doy = df.index.dayofyear
    features["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    features["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    features["month"] = df.index.month

    return features


def build_feature_files():
    """Build feature parquet files for each split, basin by basin."""
    FEAT_DIR.mkdir(parents=True, exist_ok=True)

    basins = pd.read_csv(METADATA_DIR / "basin_metadata.csv", dtype={"basin_id": str})["basin_id"].tolist()

    train_parts, val_parts, test_parts = [], [], []

    for i, bid in enumerate(basins):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(basins)} basins...")

        csv_path = PROCESSED_DIR / f"{bid}.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if len(df) < 365:
            continue

        lag_feats = compute_lag_features(df)

        # Add static attributes
        for col in STATIC_COLS:
            lag_feats[col] = df[col].iloc[0]

        lag_feats["target"] = df["target"]
        lag_feats["flow_mask"] = df["flow_mask"]

        # Split and filter observed only
        for split_name, start, end, parts_list in [
            ("train", TRAIN_START, TRAIN_END, train_parts),
            ("val", VAL_START, VAL_END, val_parts),
            ("test", TEST_START, TEST_END, test_parts),
        ]:
            split_df = lag_feats.loc[start:end]
            split_df = split_df[split_df["flow_mask"] == 1].dropna(subset=["target"])
            if len(split_df) > 0:
                parts_list.append(split_df)

    # Concatenate and save
    print("Saving feature files...")
    for name, parts in [("train", train_parts), ("val", val_parts), ("test", test_parts)]:
        combined = pd.concat(parts)
        combined.to_parquet(FEAT_DIR / f"{name}.parquet")
        print(f"  {name}: {len(combined)} samples")

    return len(train_parts)


def compute_nse(preds, targets):
    valid = ~np.isnan(targets) & ~np.isnan(preds)
    if valid.sum() == 0:
        return 0.0
    p, t = preds[valid], targets[valid]
    return 1.0 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2) if np.sum((t - t.mean()) ** 2) > 0 else 0.0


def main():
    print("=" * 60)
    print("XGBoost Baseline — CAMELS-US Task 1")
    print("=" * 60)

    try:
        import xgboost as xgb
    except ImportError:
        print("Installing xgboost...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost",
                               "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"])
        import xgboost as xgb

    # Build features if not cached
    if not (FEAT_DIR / "train.parquet").exists():
        print("\nBuilding features...")
        build_feature_files()
    else:
        print("\nUsing cached features.")

    # Load features
    print("\nLoading features...")
    train_df = pd.read_parquet(FEAT_DIR / "train.parquet")
    val_df = pd.read_parquet(FEAT_DIR / "val.parquet")
    test_df = pd.read_parquet(FEAT_DIR / "test.parquet")

    feature_cols = [c for c in train_df.columns if c not in ["target", "flow_mask"]]
    print(f"Features: {len(feature_cols)}")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Convert to float32 arrays
    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df["target"].to_numpy(dtype=np.float32)
    del train_df

    X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
    y_val = val_df["target"].to_numpy(dtype=np.float32)
    del val_df

    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_df["target"].to_numpy(dtype=np.float32)
    del test_df

    # XGBoost handles NaN natively — no imputation needed

    # Train
    print("\nTraining XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluation")
    print("=" * 60)

    train_nse = compute_nse(model.predict(X_train), y_train)
    val_nse = compute_nse(model.predict(X_val), y_val)
    test_nse = compute_nse(model.predict(X_test), y_test)

    print(f"Train NSE: {train_nse:.4f}")
    print(f"Val NSE:   {val_nse:.4f}")
    print(f"Test NSE:  {test_nse:.4f}")

    # Top features
    importance = model.feature_importances_
    top_idx = np.argsort(importance)[-10:][::-1]
    print(f"\nTop 10 features:")
    for idx in top_idx:
        print(f"  {feature_cols[idx]}: {importance[idx]:.4f}")

    # Save
    results = {
        "model": "XGBoost", "train_nse": float(train_nse),
        "val_nse": float(val_nse), "test_nse": float(test_nse),
        "n_features": len(feature_cols),
        "timestamp": datetime.now().isoformat(),
    }
    results_path = PROJECT_ROOT / "results" / "tables" / "xgboost_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
