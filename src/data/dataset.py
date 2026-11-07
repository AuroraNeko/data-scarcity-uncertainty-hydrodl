"""
dataset.py - PyTorch Dataset for CAMELS-US streamflow prediction.

Memory-efficient lazy loading: stores per-basin data arrays and creates
sliding window samples on-the-fly in __getitem__.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "camels_us"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

# 15 supplied channels: 5 named columns x 3 CAMELS forcing products. The
# upstream Maurer and NLDAS Tmax/Tmin pairs are exact duplicates, leaving 13
# distinct daily series. MUST match src/data/data_preprocessing.py DYNAMIC_VARS.
_FORCING_SOURCES = ["daymet", "maurer", "nldas"]
_BASE_FORCING_VARS = ["prcp", "srad", "tmax", "tmin", "vp"]
DYNAMIC_VARS = [f"{v}_{s}" for s in _FORCING_SOURCES for v in _BASE_FORCING_VARS]
STATIC_COLS = ["elev_mean", "slope_mean", "area_gages2", "p_mean", "pet_mean",
               "aridity", "frac_snow", "p_seasonality", "soil_depth_pelletier",
               "soil_porosity", "frac_forest", "lai_diff", "geol_porostiy"]


def compute_normalization_stats(
    basin_list: list[str],
    start_date: str,
    end_date: str,
) -> dict:
    """Compute normalization statistics from one experiment's training data.

    Dynamic inputs and the transformed streamflow target are summarized only
    over the requested dates and basins. This prevents short-record experiments
    from inheriting target statistics from excluded years or other basins.
    """
    dynamic_sum = np.zeros(len(DYNAMIC_VARS), dtype=np.float64)
    dynamic_sumsq = np.zeros(len(DYNAMIC_VARS), dtype=np.float64)
    dynamic_count = np.zeros(len(DYNAMIC_VARS), dtype=np.int64)
    target_sum = 0.0
    target_sumsq = 0.0
    target_count = 0
    static_values = {name: [] for name in STATIC_COLS}
    used_basins = []

    columns = ["date", *DYNAMIC_VARS, "target", "flow_mask", *STATIC_COLS]
    for basin_id in basin_list:
        csv_path = PROCESSED_DIR / f"{basin_id}.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path, usecols=columns, parse_dates=["date"])
        df = df.set_index("date").loc[start_date:end_date]
        if df.empty:
            continue

        dynamic = df[DYNAMIC_VARS].to_numpy(dtype=np.float64)
        finite = np.isfinite(dynamic)
        dynamic_sum += np.where(finite, dynamic, 0.0).sum(axis=0)
        dynamic_sumsq += np.where(finite, dynamic * dynamic, 0.0).sum(axis=0)
        dynamic_count += finite.sum(axis=0)

        target = df["target"].to_numpy(dtype=np.float64)
        target_valid = (df["flow_mask"].to_numpy() > 0) & np.isfinite(target)
        valid_target = target[target_valid]
        target_sum += valid_target.sum()
        target_sumsq += np.square(valid_target).sum()
        target_count += len(valid_target)

        first = df.iloc[0]
        for name in STATIC_COLS:
            value = first[name]
            if pd.notna(value):
                static_values[name].append(float(value))
        used_basins.append(basin_id)

    if not used_basins or target_count == 0 or np.any(dynamic_count == 0):
        raise ValueError(
            f"Cannot compute normalization statistics for {start_date} to {end_date}: "
            "no complete training data were found."
        )

    stats = {
        "fit_start": start_date,
        "fit_end": end_date,
        "n_basins": len(used_basins),
        "basin_ids": used_basins,
        "target_count": int(target_count),
    }
    for index, name in enumerate(DYNAMIC_VARS):
        mean = dynamic_sum[index] / dynamic_count[index]
        variance = max(dynamic_sumsq[index] / dynamic_count[index] - mean * mean, 0.0)
        stats[f"{name}_mean"] = float(mean)
        stats[f"{name}_std"] = float(np.sqrt(variance) + 1e-8)
        stats[f"{name}_count"] = int(dynamic_count[index])

    target_mean = target_sum / target_count
    target_variance = max(target_sumsq / target_count - target_mean * target_mean, 0.0)
    stats["target_mean"] = float(target_mean)
    stats["target_std"] = float(np.sqrt(target_variance) + 1e-8)

    for name, values in static_values.items():
        if not values:
            raise ValueError(f"No finite values found for static attribute {name}")
        values_array = np.asarray(values, dtype=np.float64)
        stats[f"{name}_mean"] = float(values_array.mean())
        stats[f"{name}_std"] = float(values_array.std() + 1e-8)

    return stats


class CamelsDataset(Dataset):
    """Memory-efficient sliding-window dataset for CAMELS-US.

    Stores index mapping (basin_idx, window_start) and loads windows on-the-fly.
    """

    def __init__(
        self,
        basin_list: list[str],
        start_date: str,
        end_date: str,
        seq_len: int = 365,
        preview: bool = False,
        normalization_stats: dict | None = None,
        target_start_date: str | None = None,
        target_end_date: str | None = None,
    ):
        self.seq_len = seq_len
        self.indices = []  # (basin_array_idx, window_start)
        self.basin_data = []  # list of dicts with numpy arrays
        self.basin_ids = []
        self.target_start_date = pd.Timestamp(target_start_date) if target_start_date else None
        self.target_end_date = pd.Timestamp(target_end_date) if target_end_date else None
        self.normalization_stats = normalization_stats
        self.dynamic_missing_count = np.zeros(len(DYNAMIC_VARS), dtype=np.int64)
        self.dynamic_value_count = np.zeros(len(DYNAMIC_VARS), dtype=np.int64)

        for basin_id in basin_list:
            csv_path = PROCESSED_DIR / f"{basin_id}.csv"
            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            df = df.loc[start_date:end_date]

            if len(df) < seq_len + 1:
                continue

            # Dynamic features. Corrected experiments provide condition-specific
            # training statistics; without them the stored normalized columns
            # are used as-is.
            if normalization_stats is not None:
                dynamic = df[DYNAMIC_VARS].values.astype(np.float32)
                for index, name in enumerate(DYNAMIC_VARS):
                    dynamic[:, index] = (
                        dynamic[:, index] - normalization_stats[f"{name}_mean"]
                    ) / normalization_stats[f"{name}_std"]
            else:
                dynamic_cols = [f"{v}_norm" for v in DYNAMIC_VARS]
                if not all(c in df.columns for c in dynamic_cols):
                    dynamic_cols = DYNAMIC_VARS
                dynamic = df[dynamic_cols].values.astype(np.float32)
            self.dynamic_missing_count += np.isnan(dynamic).sum(axis=0)
            self.dynamic_value_count += dynamic.shape[0]
            dynamic = np.nan_to_num(dynamic, nan=0.0)

            # A zero on the standardized scale is training-basin mean imputation.
            static_cols_norm = [f"{c}_norm" for c in STATIC_COLS]
            if normalization_stats is not None:
                static = df[STATIC_COLS].iloc[0].values.astype(np.float32)
                for index, name in enumerate(STATIC_COLS):
                    static[index] = (
                        static[index] - normalization_stats[f"{name}_mean"]
                    ) / normalization_stats[f"{name}_std"]
            elif all(c in df.columns for c in static_cols_norm):
                static = df[static_cols_norm].iloc[0].values.astype(np.float32)
            else:
                static = df[STATIC_COLS].iloc[0].values.astype(np.float32)
            static = np.nan_to_num(static, nan=0.0)

            # Target and mask
            if normalization_stats is not None:
                raw = df["target"].values.astype(np.float32)
                target = (
                    raw - normalization_stats["target_mean"]
                ) / normalization_stats["target_std"]
            else:
                target_col = "target_norm" if "target_norm" in df.columns else "target"
                target = df[target_col].values.astype(np.float32)
            mask = df["flow_mask"].values.astype(np.float32)
            target = np.nan_to_num(target, nan=0.0)

            # Water-year index carried alongside each sample.
            dates = df.index
            water_years = dates.year + (dates.month >= 10).astype(int)
            year_offset = water_years.min()
            year_idx = (water_years - year_offset).values.astype(np.int32)

            bidx = len(self.basin_data)
            self.basin_data.append({
                "dynamic": dynamic,
                "static": static,
                "target": target,
                "mask": mask,
                "year_idx": year_idx,
                "dates": dates.to_numpy(dtype="datetime64[D]"),
            })

            # Register windows by target date. Separating context dates from
            # target dates lets every experiment evaluate exactly the same days.
            n_windows = len(df) - seq_len
            for i in range(n_windows):
                target_date = dates[i + seq_len]
                if self.target_start_date is not None and target_date < self.target_start_date:
                    continue
                if self.target_end_date is not None and target_date > self.target_end_date:
                    continue
                self.indices.append((bidx, i))

            self.basin_ids.append(basin_id)

        if preview:
            print(f"  Dataset: {len(self.basin_ids)} basins, {len(self.indices)} samples")
            missing = self.dynamic_missing_fraction()
            print(f"  Dynamic missing fraction: max={missing.max():.6f}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        bidx, start = self.indices[idx]
        bd = self.basin_data[bidx]
        end = start + self.seq_len

        return (
            torch.tensor(bd["dynamic"][start:end], dtype=torch.float32),
            torch.tensor(bd["static"], dtype=torch.float32),
            torch.tensor([bd["target"][end]], dtype=torch.float32),
            torch.tensor([bd["mask"][end]], dtype=torch.float32),
            torch.tensor(bidx, dtype=torch.int32),
            torch.tensor(bd["year_idx"][end], dtype=torch.int32),
        )

    def dynamic_missing_fraction(self) -> np.ndarray:
        """Return missing fractions for the dynamic inputs before zero imputation."""
        denominator = np.maximum(self.dynamic_value_count, 1)
        return self.dynamic_missing_count / denominator

    def sample_metadata(self) -> tuple[np.ndarray, np.ndarray]:
        """Return basin IDs and target dates in dataset iteration order."""
        basin_ids = []
        target_dates = []
        for bidx, start in self.indices:
            basin_ids.append(self.basin_ids[bidx])
            target_dates.append(self.basin_data[bidx]["dates"][start + self.seq_len])
        return np.asarray(basin_ids), np.asarray(target_dates, dtype="datetime64[D]")

    def valid_target_count(self) -> int:
        """Count observed streamflow targets among the registered windows."""
        return int(sum(
            self.basin_data[bidx]["mask"][start + self.seq_len] > 0
            for bidx, start in self.indices
        ))


def get_basin_list() -> list[str]:
    """Load list of all valid basin IDs from metadata."""
    meta_path = METADATA_DIR / "basin_metadata.csv"
    if meta_path.exists():
        df = pd.read_csv(meta_path, dtype={"basin_id": str})
        return df["basin_id"].tolist()
    return [f.stem for f in sorted(PROCESSED_DIR.glob("*.csv"))]


SCARCITY_SELECTION_SEED = 42
N_SCARCITY_BASINS = 50


def primary_basin_ids(basins: list[str] | None = None) -> list[str]:
    """The fixed 50-basin primary set: seeded choice over the full basin list.

    This derivation depends only on ``data/metadata``, so every stage can
    reconstruct the primary set without reading another stage's outputs.
    """
    if basins is None:
        basins = get_basin_list()
    rng = np.random.RandomState(SCARCITY_SELECTION_SEED)
    return rng.choice(basins, N_SCARCITY_BASINS, replace=False).tolist()


def create_dataloaders(
    seq_len: int = 365,
    batch_size: int = 256,
    num_workers: int = 4,
    basin_list: list[str] | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders over all basins (temporal split).

    Used by the 671-basin benchmark trainers; the scarcity experiments
    construct CamelsDataset directly with condition-specific statistics.
    Windows register a target only after one full sequence of context, so
    the train split starts supervised targets on the declared warm-up day
    and the val/test splits omit the first seq_len days of their declared
    periods. Full-period calibration and test evaluation therefore uses
    the context-prepended mirrors built by
    experiments/train_benchmark/prepare_gpu.py instead of these loaders.
    """
    from src.data.data_preprocessing import TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END

    if basin_list is None:
        basin_list = get_basin_list()

    print(f"Creating dataloaders for {len(basin_list)} basins...")
    print(f"  Train: {TRAIN_START} to {TRAIN_END}")
    print(f"  Val:   {VAL_START} to {VAL_END}")
    print(f"  Test:  {TEST_START} to {TEST_END}")

    train_ds = CamelsDataset(basin_list, TRAIN_START, TRAIN_END, seq_len, preview=True)
    val_ds = CamelsDataset(basin_list, VAL_START, VAL_END, seq_len, preview=True)
    test_ds = CamelsDataset(basin_list, TEST_START, TEST_END, seq_len, preview=True)

    # Only the train loader (shuffled, multi-epoch) benefits from workers;
    # val/test are single/light passes; workers there just waste RAM.
    train_kw = dict(num_workers=num_workers, pin_memory=True)
    eval_kw = dict(num_workers=0, pin_memory=True)
    if num_workers > 0:
        train_kw['persistent_workers'] = True
        train_kw['prefetch_factor'] = 2
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **train_kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **eval_kw)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **eval_kw)

    return train_loader, val_loader, test_loader
