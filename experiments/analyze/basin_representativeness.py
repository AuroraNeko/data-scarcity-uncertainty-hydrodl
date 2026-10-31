"""Compare the fixed 50-basin scarcity subset with the other CAMELS basins.

The inferential comparison uses the 50 selected basins and the disjoint set of
621 remaining basins. This avoids applying a two-sample test to a subset and a
full set that contains that subset. The script writes compact JSON/CSV evidence
and a LaTeX row fragment; it never overwrites the supplement itself.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import PROCESSED_DIR, get_basin_list, primary_basin_ids


SELECTION_SEED = 42
N_SCARCITY_BASINS = 50
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
GENERATED_DIR = PROJECT_ROOT / "paper" / "generated"

STATIC_COLUMNS = [
    "elev_mean",
    "slope_mean",
    "area_gages2",
    "p_mean",
    "pet_mean",
    "aridity",
    "frac_snow",
    "p_seasonality",
    "soil_depth_pelletier",
    "soil_porosity",
    "frac_forest",
    "lai_diff",
    "geol_porostiy",
]

DISPLAY_NAMES = {
    "elev_mean": "Elevation (m)",
    "slope_mean": "Slope (m km$^{-1}$)",
    "area_gages2": "Drainage area (km$^2$)",
    "p_mean": "Mean precipitation (mm d$^{-1}$)",
    "pet_mean": "Mean PET (mm d$^{-1}$)",
    "aridity": "Aridity (PET/P)",
    "frac_snow": "Snow fraction",
    "p_seasonality": "Precipitation seasonality",
    "soil_depth_pelletier": "Soil depth (m)",
    "soil_porosity": "Soil porosity",
    "frac_forest": "Forest fraction",
    "lai_diff": "LAI seasonal difference",
    "geol_porostiy": "Geologic porosity",
}


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm family-wise-error adjusted p-values."""
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted_sorted = np.empty_like(values)
    running_max = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[index])
        running_max = max(running_max, candidate)
        adjusted_sorted[rank] = running_max
    adjusted = np.empty_like(values)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def pooled_standard_deviation(left: np.ndarray, right: np.ndarray) -> float:
    denominator = len(left) + len(right) - 2
    if denominator <= 0:
        return float("nan")
    variance = (
        (len(left) - 1) * np.var(left, ddof=1)
        + (len(right) - 1) * np.var(right, ddof=1)
    ) / denominator
    return float(np.sqrt(variance))


def format_number(value: float, digits: int = 3) -> str:
    if not math.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def format_p(value: float) -> str:
    if not math.isfinite(value):
        return "--"
    return f"{value:.1e}" if value < 0.001 else f"{value:.3f}"


def main() -> None:
    all_basins = get_basin_list()
    if len(all_basins) != 671:
        raise RuntimeError(f"Expected 671 basins, found {len(all_basins)}")

    rng = np.random.RandomState(SELECTION_SEED)
    selected_list = primary_basin_ids(all_basins)
    assert selected_list == rng.choice(all_basins, N_SCARCITY_BASINS, replace=False).tolist()
    selected = set(selected_list)
    remaining = set(all_basins) - selected

    records = []
    for basin_id in all_basins:
        path = PROCESSED_DIR / f"{basin_id}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        row = pd.read_csv(path, usecols=STATIC_COLUMNS, nrows=1).iloc[0].to_dict()
        row["basin_id"] = basin_id
        row["group"] = "scarcity50" if basin_id in selected else "remaining621"
        records.append(row)
    frame = pd.DataFrame(records)

    attributes = {}
    summary_rows = []
    raw_p_values = []
    for column in STATIC_COLUMNS:
        subset = frame.loc[frame["group"] == "scarcity50", column].to_numpy(dtype=float)
        comparison = frame.loc[frame["group"] == "remaining621", column].to_numpy(dtype=float)
        subset = subset[np.isfinite(subset)]
        comparison = comparison[np.isfinite(comparison)]
        if len(subset) < 2 or len(comparison) < 2:
            raise RuntimeError(f"Insufficient finite values for {column}")

        statistic, p_value = stats.ks_2samp(subset, comparison, method="auto")
        pooled_sd = pooled_standard_deviation(subset, comparison)
        effect = (
            float((subset.mean() - comparison.mean()) / pooled_sd)
            if pooled_sd > 0
            else 0.0
        )
        row = {
            "attribute_key": column,
            "attribute": DISPLAY_NAMES[column],
            "remaining621_mean": float(comparison.mean()),
            "remaining621_std": float(comparison.std(ddof=1)),
            "scarcity50_mean": float(subset.mean()),
            "scarcity50_std": float(subset.std(ddof=1)),
            "ks_statistic": float(statistic),
            "ks_p_raw": float(p_value),
            "cohens_d_subset_minus_remaining": effect,
        }
        summary_rows.append(row)
        raw_p_values.append(float(p_value))

    adjusted = holm_adjust(raw_p_values)
    for row, adjusted_p in zip(summary_rows, adjusted):
        row["ks_p_holm"] = adjusted_p
        attributes[row["attribute_key"]] = {
            key: value for key, value in row.items() if key != "attribute_key"
        }

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "selection_seed": SELECTION_SEED,
        "n_scarcity_basins": len(selected),
        "n_comparison_basins": len(remaining),
        "comparison_design": "disjoint scarcity subset versus remaining basins",
        "multiple_testing": "Holm family-wise-error adjustment over 13 KS tests",
        "basins": selected_list,
        "attributes": attributes,
    }
    (TABLE_DIR / "basin_representativeness.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    pd.DataFrame(summary_rows).to_csv(
        TABLE_DIR / "basin_representativeness_summary.csv", index=False
    )
    pd.DataFrame(
        {
            "selection_order": np.arange(1, len(selected_list) + 1),
            "basin_id": selected_list,
        }
    ).to_csv(TABLE_DIR / "scarcity_50_basin_ids.csv", index=False)

    tex_rows = []
    for row in summary_rows:
        tex_rows.append(
            f"{row['attribute']} & "
            f"{format_number(row['remaining621_mean'])} "
            f"({format_number(row['remaining621_std'])}) & "
            f"{format_number(row['scarcity50_mean'])} "
            f"({format_number(row['scarcity50_std'])}) & "
            f"{format_p(row['ks_p_raw'])} & "
            f"{format_number(row['cohens_d_subset_minus_remaining'], 2)} \\\\"
        )
    (GENERATED_DIR / "representativeness_rows.tex").write_text(
        "\n".join(tex_rows) + "\n", encoding="ascii"
    )

    smallest_adjusted = min(adjusted)
    largest_effect = max(abs(row["cohens_d_subset_minus_remaining"]) for row in summary_rows)
    print(f"Selected basins: {len(selected)}; independent comparison basins: {len(remaining)}")
    print(f"Smallest Holm-adjusted KS p-value: {smallest_adjusted:.3f}")
    print(f"Largest absolute Cohen's d: {largest_effect:.3f}")
    print(f"Saved representativeness artifacts to {TABLE_DIR}")


if __name__ == "__main__":
    main()
