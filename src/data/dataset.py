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

# 15 dynamic features: 5 meteorological vars x 3 CAMELS forcing products
# (daymet, maurer, nldas). MUST match src/data/data_preprocessing.py DYNAMIC_VARS.
_FORCING_SOURCES = ["daymet", "maurer", "nldas"]
_BASE_FORCING_VARS = ["prcp", "srad", "tmax", "tmin", "vp"]
DYNAMIC_VARS = [f"{v}_{s}" for s in _FORCING_SOURCES for v in _BASE_FORCING_VARS]
STATIC_COLS = ["elev_mean", "slope_mean", "area_gages2", "p_mean", "pet_mean",
               "aridity", "frac_snow", "p_seasonality", "soil_depth_pelletier",
               "soil_porosity", "frac_forest", "lai_diff", "geol_porostiy"]


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
        per_basin_stats: dict | None = None,
    ):
        self.seq_len = seq_len
        self.indices = []  # (basin_array_idx, window_start)
        self.basin_data = []  # list of dicts with numpy arrays
        self.basin_ids = []

        for basin_id in basin_list:
            csv_path = PROCESSED_DIR / f"{basin_id}.csv"
            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            df = df.loc[start_date:end_date]

            if len(df) < seq_len + 1:
                continue

            # Dynamic features (normalized)
            dynamic_cols = [f"{v}_norm" for v in DYNAMIC_VARS]
            if not all(c in df.columns for c in dynamic_cols):
                dynamic_cols = DYNAMIC_VARS
            dynamic = df[dynamic_cols].values.astype(np.float32)
            dynamic = np.nan_to_num(dynamic, nan=0.0)

            # Static attributes: fill NaN with 0
            static_cols_norm = [f"{c}_norm" for c in STATIC_COLS]
            if all(c in df.columns for c in static_cols_norm):
                static = df[static_cols_norm].iloc[0].values.astype(np.float32)
            else:
                static = df[STATIC_COLS].iloc[0].values.astype(np.float32)
            static = np.nan_to_num(static, nan=0.0)

            # Target and mask
            if per_basin_stats is not None:
                # Per-basin normalization (CAMELS convention): normalizes log-flow
                # by each basin's own training-period mean/std. Removes inter-basin
                # variance and yields literature-comparable per-basin NSE.
                raw = df["target"].values.astype(np.float32)
                bs = per_basin_stats.get(basin_id)
                target = (raw - bs["mean"]) / bs["std"] if bs else raw
            else:
                target_col = "target_norm" if "target_norm" in df.columns else "target"
                target = df[target_col].values.astype(np.float32)
            mask = df["flow_mask"].values.astype(np.float32)
            target = np.nan_to_num(target, nan=0.0)

            # Water year index for physics loss grouping
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
            })

            # Register sliding window indices
            n_windows = len(df) - seq_len
            for i in range(n_windows):
                self.indices.append((bidx, i))

            self.basin_ids.append(basin_id)

        if preview:
            print(f"  Dataset: {len(self.basin_ids)} basins, {len(self.indices)} samples")

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

    def compute_q95(self) -> float:
        """Compute 95th percentile of observed target values (for extreme weighting)."""
        all_targets = []
        for bd in self.basin_data:
            valid = bd["mask"] > 0
            all_targets.append(bd["target"][valid])
        all_targets = np.concatenate(all_targets)
        return float(np.percentile(all_targets, 95))


def get_basin_list() -> list[str]:
    """Load list of all valid basin IDs from metadata."""
    meta_path = METADATA_DIR / "basin_metadata.csv"
    if meta_path.exists():
        df = pd.read_csv(meta_path, dtype={"basin_id": str})
        return df["basin_id"].tolist()
    return [f.stem for f in sorted(PROCESSED_DIR.glob("*.csv"))]


def create_dataloaders(
    seq_len: int = 365,
    batch_size: int = 256,
    num_workers: int = 4,
    basin_list: list[str] | None = None,
    per_basin_stats: dict | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders for Task 1 (seen basin temporal split)."""
    from src.data.data_preprocessing import TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END

    if basin_list is None:
        basin_list = get_basin_list()

    print(f"Creating dataloaders for {len(basin_list)} basins...")
    print(f"  Train: {TRAIN_START} to {TRAIN_END}")
    print(f"  Val:   {VAL_START} to {VAL_END}")
    print(f"  Test:  {TEST_START} to {TEST_END}")

    train_ds = CamelsDataset(basin_list, TRAIN_START, TRAIN_END, seq_len, preview=True, per_basin_stats=per_basin_stats)
    val_ds = CamelsDataset(basin_list, VAL_START, VAL_END, seq_len, preview=True, per_basin_stats=per_basin_stats)
    test_ds = CamelsDataset(basin_list, TEST_START, TEST_END, seq_len, preview=True, per_basin_stats=per_basin_stats)

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
