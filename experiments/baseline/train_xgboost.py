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

# 15 dynamic features (5 vars x 3 forcing products) — matches the DL models'
# input for a fair comparison. Names must match processed CSV raw columns.
_FORCING_SOURCES = ["daymet", "maurer", "nldas"]
_BASE_FORCING_VARS = ["prcp", "srad", "tmax", "tmin", "vp"]
DYNAMIC_VARS = [f"{v}_{s}" for s in _FORCING_SOURCES for v in _BASE_FORCING_VARS]
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
        lag_feats["basin_id"] = bid

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

    # Daymet-only features: the 3 forcing products are redundant (maurer≈daymet≈nldas)
    # and XGBoost overfits all 166 lag features (per-basin NSE collapses to <0 or near-0).
    # Daymet-only (50 lag features + 13 static + 3 time = 66) gives a clean ~0.63
    # (verified independently). This is the standard single-forcing XGBoost config for
    # CAMELS; the paper notes the DL model additionally leverages all 3 forcing products.
    feature_cols = [c for c in train_df.columns if c not in ["target", "flow_mask", "basin_id"]
                    and ("_daymet" in c or c in STATIC_COLS or c in ("doy_sin", "doy_cos", "month"))]
    print(f"Features: {len(feature_cols)} (Daymet-only)")
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
    test_basin = test_df["basin_id"].values
    del test_df

    # XGBoost handles NaN natively — no imputation needed

    # Train. Daymet-only (66 features, no redundancy) — standard config: depth 8
    # + colsample 0.8 + early stopping. Verified to give per-basin median NSE ~0.63.
    print("\nTraining XGBoost (Daymet-only, standard config)...")
    model = xgb.XGBRegressor(
        n_estimators=800, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
        reg_lambda=1.0, tree_method="hist", random_state=42, n_jobs=-1,
        early_stopping_rounds=30,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluation")
    print("=" * 60)

    train_nse = compute_nse(model.predict(X_train), y_train)
    val_nse = compute_nse(model.predict(X_val), y_val)
    test_nse = compute_nse(model.predict(X_test), y_test)

    # Per-basin median NSE on raw flow (standard, literature-comparable metric)
    # + cluster-bootstrap 95% CI (resample basins, same protocol as the DL models)
    test_pred = model.predict(X_test)
    _eval = pd.DataFrame({"basin": test_basin, "y": np.expm1(y_test), "p": np.expm1(test_pred)})
    _pnse = []
    for _b, _g in _eval.groupby("basin"):
        if len(_g) > 5:
            _ss = ((_g["y"] - _g["y"].mean()) ** 2).sum()
            if _ss > 0:
                _pnse.append(1 - ((_g["y"] - _g["p"]) ** 2).sum() / _ss)
    _pnse = np.asarray(_pnse)
    test_nse_median = float(np.nanmedian(_pnse))
    _rng = np.random.RandomState(42); _B = 1000; _nb = len(_pnse)
    _boot = np.nanmedian(_pnse[_rng.randint(0, _nb, (_B, _nb))], axis=1)
    test_nse_perbasin_ci = [float(np.percentile(_boot, 2.5)), float(np.percentile(_boot, 97.5))]

    print(f"Train NSE: {train_nse:.4f}")
    print(f"Val NSE:   {val_nse:.4f}")
    print(f"Test NSE (pooled):  {test_nse:.4f}")
    print(f"Test NSE (per-basin median): {test_nse_median:.4f}")

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
        "test_nse_perbasin_median": test_nse_median,
        "test_nse_perbasin_ci": test_nse_perbasin_ci,
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
