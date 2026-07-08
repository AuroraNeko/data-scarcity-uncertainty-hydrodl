"""
data_preprocessing.py - CAMELS-US data preprocessing for LPU-Stream

Loads raw CAMELS-US data (Daymet + Maurer + NLDAS forcing, USGS streamflow, catchment attributes),
merges into unified per-basin DataFrames, handles missing values, applies transforms,
and saves processed data ready for model training.

Usage:
    python src/data/data_preprocessing.py
"""

import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "camels_us"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "camels_us"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

# CAMELS data paths (within the extracted zip)
FORCING_ROOT = RAW_DIR / "basin_dataset_public_v1p2" / "basin_mean_forcing"
STREAMFLOW_ROOT = RAW_DIR / "basin_dataset_public_v1p2" / "usgs_streamflow"

# Configuration
# Three CAMELS v1.2 forcing products (Daymet, Maurer, NLDAS). Each provides the
# same 5 meteorological variables; using all 3 gives 15 dynamic features, the
# standard input for strong CAMELS LSTM benchmarks (Kratzert et al.).
FORCING_SOURCES = ["daymet", "maurer", "nldas"]
BASE_FORCING_VARS = ["prcp", "srad", "tmax", "tmin", "vp"]  # 5 vars per source
DYNAMIC_VARS = [f"{v}_{s}" for s in FORCING_SOURCES for v in BASE_FORCING_VARS]  # 15 features
FORCING_SOURCE = "daymet"  # kept for basin enumeration / backward compatibility
STATIC_ATTR_FILES = {
    "camels_topo.txt": ["elev_mean", "slope_mean", "area_gages2"],
    "camels_clim.txt": ["p_mean", "pet_mean", "aridity", "frac_snow", "p_seasonality"],
    "camels_soil.txt": ["soil_depth_pelletier", "soil_porosity"],
    "camels_vege.txt": ["frac_forest", "lai_diff"],
    "camels_geol.txt": ["geol_porostiy"],  # note: typo in original dataset
}

# Date splits (paper Section 6)
TRAIN_START = "1980-10-01"
TRAIN_END = "1995-09-30"
VAL_START = "1995-10-01"
VAL_END = "2000-09-30"
TEST_START = "2000-10-01"
TEST_END = "2010-09-30"

# Transform for target variable (log1p = ln(1+y) via numpy.log1p; or "sqrt").
TARGET_TRANSFORM = "log1p"


def _build_forcing_index(source: str = "daymet") -> dict:
    """Build a mapping from basin_id to forcing file path.

    File-naming differs by source: daymet uses 'cida', maurer/nldas use their
    own tags (*_lump_<tag>_forcing_leap.txt).
    """
    tag = {"daymet": "cida", "maurer": "maurer", "nldas": "nldas"}.get(source, source)
    index = {}
    forcing_dir = FORCING_ROOT / source
    if not forcing_dir.exists():
        return index
    for huc_dir in forcing_dir.iterdir():
        if huc_dir.is_dir():
            for f in huc_dir.glob(f"*_lump_{tag}_forcing_leap.txt"):
                basin_id = f.name.split("_")[0]
                index[basin_id] = f
            for f in huc_dir.glob(f"*_lump_{tag}_forcing.txt"):
                basin_id = f.name.split("_")[0]
                if basin_id not in index:
                    index[basin_id] = f
    return index


def load_forcing_basin(basin_id: str, source: str = "daymet", f_index: dict = None) -> pd.DataFrame:
    """Load meteorological forcing data for a single basin.

    CAMELS forcing file format (tab/space-delimited, 4 header lines; identical for Daymet/Maurer/NLDAS):
        Year Mnth Day Hr dayl(s) prcp(mm/day) srad(W/m2) swe(mm) tmax(C) tmin(C) vp(Pa)
    """
    filepath = f_index.get(basin_id) if f_index else None
    if filepath is None or not filepath.exists():
        return pd.DataFrame()

    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        skiprows=4,
        header=None,
        names=["Year", "Mnth", "Day", "Hr", "dayl", "prcp", "srad", "swe", "tmax", "tmin", "vp"],
    )

    df["date"] = pd.to_datetime(
        dict(year=df["Year"].astype(int), month=df["Mnth"].astype(int), day=df["Day"].astype(int)),
        errors="coerce",
    )
    df = df.dropna(subset=["date"])
    df = df.set_index("date")
    df = df[["prcp", "tmin", "tmax", "srad", "vp"]]
    df = df.sort_index()

    return df


def _build_streamflow_index() -> dict:
    """Build a mapping from basin_id to streamflow file path."""
    index = {}
    if not STREAMFLOW_ROOT.exists():
        return index
    for huc_dir in STREAMFLOW_ROOT.iterdir():
        if huc_dir.is_dir():
            for f in huc_dir.glob("*_streamflow_qc.txt"):
                basin_id = f.name.split("_")[0]
                index[basin_id] = f
    return index


def load_streamflow_basin(basin_id: str, sf_index: dict = None) -> pd.DataFrame:
    """Load observed streamflow for a single basin.

    CAMELS streamflow files have format (space-delimited, no header):
        basin_id  Year  Mnth  Day  streamflow(cfs)  quality_flag
    """
    filepath = sf_index.get(basin_id) if sf_index else None
    if filepath is None or not filepath.exists():
        return pd.DataFrame()

    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        names=["basin_id", "Year", "Mnth", "Day", "streamflow_cfs", "flag"],
    )

    df["date"] = pd.to_datetime(
        dict(year=df["Year"].astype(int), month=df["Mnth"].astype(int), day=df["Day"].astype(int)),
        errors="coerce",
    )
    df = df.dropna(subset=["date"])
    df = df.set_index("date")
    df = df[["streamflow_cfs"]]

    # Replace -999 with NaN (CAMELS missing value marker)
    df["streamflow_cfs"] = df["streamflow_cfs"].replace(-999.0, np.nan)

    return df


def load_all_attributes() -> pd.DataFrame:
    """Load and merge all catchment attribute files."""
    attrs = {}
    for filename, columns in STATIC_ATTR_FILES.items():
        filepath = RAW_DIR / filename
        if not filepath.exists():
            print(f"  [WARN] Attribute file not found: {filepath}")
            continue

        df = pd.read_csv(filepath, sep=";", dtype={"gauge_id": str})
        df = df.rename(columns={"gauge_id": "basin_id"})

        # Select specified columns (keep basin_id)
        available = [c for c in columns if c in df.columns]
        if available:
            attrs[filename] = df[["basin_id"] + available]

    if not attrs:
        return pd.DataFrame()

    # Merge all attribute DataFrames on basin_id
    result = None
    for name, df in attrs.items():
        if result is None:
            result = df
        else:
            result = result.merge(df, on="basin_id", how="outer")

    return result


def cfs_to_mmday(streamflow_cfs: pd.Series, area_km2: float) -> pd.Series:
    """Convert streamflow from cfs to mm/day.

    1 cfs = 0.0283168 m^3/s
    mm/day = (cfs * 0.0283168 * 86400) / (area_km2 * 1e6) * 1000
           = cfs * 0.0283168 * 86400 / (area_km2 * 1e3)
           = cfs * 2446.57152 / (area_km2 * 1e3)
    """
    conversion = 0.0283168 * 86400.0 / (area_km2 * 1e6) * 1000.0  # = 2446.57 / area_km2
    return streamflow_cfs * conversion


def apply_transform(series: pd.Series, transform: str) -> pd.Series:
    """Apply target variable transform (``log1p`` = ln(1+y) via numpy.log1p)."""
    if transform == "log1p":
        return np.log1p(series)
    elif transform == "sqrt":
        return np.sqrt(series)
    return series


def inverse_transform(series: pd.Series, transform: str) -> pd.Series:
    """Inverse target variable transform."""
    if transform == "log1p":
        return np.expm1(series)
    elif transform == "sqrt":
        return np.square(series)
    return series


def fill_missing_dynamic(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing dynamic (forcing) variables with linear interpolation."""
    for col in DYNAMIC_VARS:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear", limit=7)
            df[col] = df[col].ffill(limit=3)
            df[col] = df[col].bfill(limit=3)
    return df


def get_all_basin_ids() -> list:
    """Get list of all basin IDs from the forcing directory."""
    forcing_dir = FORCING_ROOT / FORCING_SOURCE
    if not forcing_dir.exists():
        print(f"[ERROR] Forcing directory not found: {forcing_dir}")
        print("Please extract the CAMELS zip file first.")
        return []

    basin_ids = []
    for huc_dir in sorted(forcing_dir.iterdir()):
        if huc_dir.is_dir():
            for f in sorted(huc_dir.glob("*_lump_cida_forcing_leap.txt")):
                basin_id = f.name.split("_")[0]
                basin_ids.append(basin_id)
    return basin_ids


def preprocess_basin(basin_id: str, attributes: pd.DataFrame, sf_index: dict, f_index_all: dict) -> pd.DataFrame | None:
    """Full preprocessing pipeline for a single basin.

    Loads all 3 forcing products (daymet, maurer, nldas), renames each source's
    5 variables as ``<var>_<source>`` (15 dynamic features total), and merges
    them on the date index before joining streamflow. ``f_index_all`` maps
    source -> {basin_id -> path}.
    """
    # Load all 3 forcing products and merge on date
    frames = []
    for s in FORCING_SOURCES:
        fdf = load_forcing_basin(basin_id, s, f_index_all.get(s, {}))
        if fdf.empty:
            return None  # require all 3 sources present for this basin
        fdf = fdf.rename(columns={v: f"{v}_{s}" for v in BASE_FORCING_VARS})
        frames.append(fdf)
    forcing = frames[0].join(frames[1], how="outer").join(frames[2], how="outer")
    if forcing.empty:
        return None

    # Load streamflow
    streamflow = load_streamflow_basin(basin_id, sf_index)
    if streamflow.empty:
        return None

    # Merge forcing and streamflow
    df = forcing.join(streamflow, how="outer")

    # Fill missing forcing values
    df = fill_missing_dynamic(df)

    # Convert streamflow from cfs to mm/day
    basin_attrs = attributes[attributes["basin_id"] == basin_id]
    if basin_attrs.empty:
        return None

    area_km2 = basin_attrs["area_gages2"].values[0]
    if pd.isna(area_km2) or area_km2 <= 0:
        return None

    df["streamflow_mm"] = cfs_to_mmday(df["streamflow_cfs"], area_km2)

    # Create streamflow mask (1 = observed, 0 = missing)
    df["flow_mask"] = df["streamflow_mm"].notna().astype(int)

    # Apply transform to target
    df["target"] = apply_transform(df["streamflow_mm"], TARGET_TRANSFORM)

    # Add static attributes as constant columns
    for _, row in basin_attrs.iterrows():
        for col in row.index:
            if col != "basin_id":
                df[col] = row[col]

    # Filter to our target date range
    start = pd.Timestamp(TRAIN_START)
    end = pd.Timestamp(TEST_END)
    df = df.loc[start:end]

    # Drop rows with no forcing data at all
    df = df.dropna(subset=DYNAMIC_VARS, how="all")

    return df


def compute_normalization(basin_data: dict) -> dict:
    """Compute normalization statistics from training period across all basins."""
    all_forcing = []
    all_target = []

    for basin_id, df in basin_data.items():
        train_mask = (df.index >= TRAIN_START) & (df.index <= TRAIN_END)
        train_df = df.loc[train_mask]

        all_forcing.append(train_df[DYNAMIC_VARS].values)
        valid_target = train_df["target"].dropna()
        if len(valid_target) > 0:
            all_target.append(valid_target.values)

    forcing_arr = np.concatenate(all_forcing, axis=0)
    target_arr = np.concatenate(all_target, axis=0)

    stats = {}
    # Dynamic variable statistics
    for i, var in enumerate(DYNAMIC_VARS):
        col = forcing_arr[:, i]
        col = col[~np.isnan(col)]
        stats[f"{var}_mean"] = float(np.mean(col))
        stats[f"{var}_std"] = float(np.std(col)) + 1e-8

    # Target statistics
    stats["target_mean"] = float(np.mean(target_arr))
    stats["target_std"] = float(np.std(target_arr)) + 1e-8

    # Static attribute statistics
    static_cols = [c for c in basin_data[list(basin_data.keys())[0]].columns
                   if c not in DYNAMIC_VARS + ["streamflow_cfs", "streamflow_mm", "flow_mask", "target"]
                   and c not in ["basin_id"]]
    for col in static_cols:
        vals = []
        for df in basin_data.values():
            v = df[col].iloc[0]
            if pd.notna(v):
                vals.append(v)
        if vals:
            stats[f"{col}_mean"] = float(np.mean(vals))
            stats[f"{col}_std"] = float(np.std(vals)) + 1e-8

    return stats


def normalize_basin(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Normalize features using pre-computed statistics."""
    df = df.copy()

    # Normalize dynamic variables
    for var in DYNAMIC_VARS:
        mean = stats.get(f"{var}_mean", 0)
        std = stats.get(f"{var}_std", 1)
        df[f"{var}_norm"] = (df[var] - mean) / std

    # Normalize target
    mean = stats.get("target_mean", 0)
    std = stats.get("target_std", 1)
    df["target_norm"] = (df["target"] - mean) / std

    # Normalize static attributes
    static_cols = [c for c in df.columns
                   if c not in DYNAMIC_VARS + ["streamflow_cfs", "streamflow_mm", "flow_mask",
                                                "target", "target_norm"]
                   and c not in [f"{v}_norm" for v in DYNAMIC_VARS]
                   and c not in ["basin_id"]]
    for col in static_cols:
        mean = stats.get(f"{col}_mean", 0)
        std = stats.get(f"{col}_std", 1)
        df[f"{col}_norm"] = (df[col] - mean) / std

    return df


def main():
    print("=" * 60)
    print("CAMELS-US Data Preprocessing for LPU-Stream")
    print("=" * 60)

    # Check extraction
    if not FORCING_ROOT.exists():
        zip_path = RAW_DIR / "basin_timeseries_v1p2_metForcing_obsFlow.zip"
        if zip_path.exists():
            print(f"\n[INFO] Extracting {zip_path}...")
            print("This may take several minutes...")
            import zipfile
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(RAW_DIR)
            print("Extraction complete.")
        else:
            print(f"[ERROR] Neither extracted data nor zip found.")
            print(f"Expected: {FORCING_ROOT}")
            print(f"Or zip: {zip_path}")
            return

    # Get basin list
    basin_ids = get_all_basin_ids()
    print(f"\nFound {len(basin_ids)} basins with {FORCING_SOURCE} forcing data")

    if not basin_ids:
        print("[ERROR] No basins found. Check data extraction.")
        return

    # Load attributes
    print("\nLoading catchment attributes...")
    attributes = load_all_attributes()
    print(f"  Attributes loaded for {len(attributes)} basins")
    print(f"  Attribute columns: {list(attributes.columns)}")

    # Build streamflow file index
    print("\nBuilding streamflow file index...")
    sf_index = _build_streamflow_index()
    print(f"  Found {len(sf_index)} streamflow files")

    # Build forcing file index for all 3 sources
    print("Building forcing file index (daymet + maurer + nldas)...")
    f_index_all = {s: _build_forcing_index(s) for s in FORCING_SOURCES}
    for s in FORCING_SOURCES:
        print(f"  {s}: {len(f_index_all[s])} forcing files")

    # Process each basin
    print(f"\nProcessing {len(basin_ids)} basins...")
    basin_data = {}
    skipped = 0

    for i, basin_id in enumerate(basin_ids):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(basin_ids)} basins...")

        df = preprocess_basin(basin_id, attributes, sf_index, f_index_all)
        if df is not None and len(df) > 365:  # At least 1 year of data
            basin_data[basin_id] = df
        else:
            skipped += 1

    print(f"\n  Successfully processed: {len(basin_data)} basins")
    print(f"  Skipped: {skipped} basins")

    if not basin_data:
        print("[ERROR] No basins successfully processed.")
        return

    # Compute normalization statistics from training period
    print("\nComputing normalization statistics...")
    stats = compute_normalization(basin_data)

    # Save statistics
    os.makedirs(METADATA_DIR, exist_ok=True)
    stats_path = METADATA_DIR / "normalization_stats.json"
    import json
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved normalization stats to {stats_path}")

    # Normalize all basins and save
    print("\nNormalizing and saving processed data...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    for basin_id, df in basin_data.items():
        df_norm = normalize_basin(df, stats)
        out_path = PROCESSED_DIR / f"{basin_id}.csv"
        df_norm.to_csv(out_path)

    # Save basin metadata (list of valid basins, data coverage)
    print("\nSaving basin metadata...")
    metadata = []
    for basin_id, df in basin_data.items():
        train = df.loc[TRAIN_START:TRAIN_END]
        val = df.loc[VAL_START:VAL_END]
        test = df.loc[TEST_START:TEST_END]

        metadata.append({
            "basin_id": basin_id,
            "train_days": len(train),
            "val_days": len(val),
            "test_days": len(test),
            "train_flow_observed": int(train["flow_mask"].sum()),
            "val_flow_observed": int(val["flow_mask"].sum()),
            "test_flow_observed": int(test["flow_mask"].sum()),
            "flow_missing_pct": float((df["flow_mask"] == 0).sum() / len(df) * 100),
        })

    meta_df = pd.DataFrame(metadata)
    meta_path = METADATA_DIR / "basin_metadata.csv"
    meta_df.to_csv(meta_path, index=False)

    # Print summary
    print(f"\n{'='*60}")
    print("Preprocessing Complete!")
    print(f"{'='*60}")
    print(f"  Basins processed: {len(basin_data)}")
    print(f"  Date range: {TRAIN_START} to {TEST_END}")
    print(f"  Train: {TRAIN_START} to {TRAIN_END}")
    print(f"  Val:   {VAL_START} to {VAL_END}")
    print(f"  Test:  {TEST_START} to {TEST_END}")
    print(f"  Target transform: {TARGET_TRANSFORM}")
    print(f"  Dynamic variables: {DYNAMIC_VARS}")
    print(f"\n  Output: {PROCESSED_DIR}")
    print(f"  Metadata: {METADATA_DIR}")
    print(f"  Stats: {stats_path}")

    # Print data coverage stats
    avg_missing = meta_df["flow_missing_pct"].mean()
    print(f"\n  Avg flow missing: {avg_missing:.1f}%")
    print(f"  Basins with <5% missing: {(meta_df['flow_missing_pct'] < 5).sum()}")
    print(f"  Basins with >20% missing: {(meta_df['flow_missing_pct'] > 20).sum()}")


if __name__ == "__main__":
    main()
