"""Create four disjoint hydroclimatically stratified scarcity subsets.

The seed-42 random subset remains the primary gradient. Four additional
50-basin subsets are sampled without overlap from the remaining CAMELS-US
basins, stratified jointly by aridity and log drainage-area quartiles. They
are used only for the 1-year versus 15-year endpoint replication.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import PROCESSED_DIR, get_basin_list, primary_basin_ids  # noqa: E402


REPLICATE_SEEDS = [20260721, 20260722, 20260723, 20260724]
TABLE_DIR = PROJECT_ROOT / "results" / "tables" / "scarcity_replications"
CSV_PATH = TABLE_DIR / "replication_subsets.csv"
JSON_PATH = TABLE_DIR / "replication_subsets.json"


def load_attributes(basins: list[str]) -> pd.DataFrame:
    rows = []
    for basin_id in basins:
        path = PROCESSED_DIR / f"{basin_id}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        values = pd.read_csv(
            path,
            usecols=["aridity", "area_gages2"],
            nrows=1,
        ).iloc[0]
        rows.append({
            "basin_id": basin_id,
            "aridity": float(values["aridity"]),
            "area_gages2": float(values["area_gages2"]),
        })
    frame = pd.DataFrame(rows)
    if not np.isfinite(frame[["aridity", "area_gages2"]].to_numpy()).all():
        raise ValueError("Aridity and drainage area must be finite for every basin")
    frame["aridity_quartile"] = pd.qcut(
        frame["aridity"], 4, labels=False, duplicates="raise"
    ).astype(int)
    frame["area_quartile"] = pd.qcut(
        np.log1p(frame["area_gages2"]), 4, labels=False, duplicates="raise"
    ).astype(int)
    frame["stratum"] = (
        frame["aridity_quartile"].astype(str)
        + "_"
        + frame["area_quartile"].astype(str)
    )
    return frame


def proportional_allocation(frame: pd.DataFrame, sample_size: int) -> dict[str, int]:
    counts = frame.groupby("stratum").size().sort_index()
    exact = counts / counts.sum() * sample_size
    allocation = np.floor(exact).astype(int)
    remainder = sample_size - int(allocation.sum())
    order = sorted(
        counts.index,
        key=lambda name: (-(exact[name] - allocation[name]), name),
    )
    for name in order[:remainder]:
        allocation[name] += 1
    if int(allocation.sum()) != sample_size:
        raise RuntimeError("Stratified allocation did not reach the requested sample size")
    if (allocation > counts).any():
        raise RuntimeError("A stratum was allocated more basins than are available")
    return {str(name): int(value) for name, value in allocation.items()}


def select_stratified(
    candidates: pd.DataFrame,
    sample_size: int,
    seed: int,
) -> list[str]:
    allocation = proportional_allocation(candidates, sample_size)
    rng = np.random.default_rng(seed)
    selected = []
    for stratum in sorted(allocation):
        basin_ids = candidates.loc[
            candidates["stratum"] == stratum, "basin_id"
        ].to_numpy()
        chosen = rng.choice(basin_ids, size=allocation[stratum], replace=False)
        selected.extend(chosen.tolist())
    rng.shuffle(selected)
    if len(selected) != sample_size or len(set(selected)) != sample_size:
        raise RuntimeError("Invalid stratified subset")
    return selected


def subset_summary(frame: pd.DataFrame) -> dict:
    return {
        "n_basins": int(len(frame)),
        "aridity_mean": float(frame["aridity"].mean()),
        "aridity_sd": float(frame["aridity"].std(ddof=1)),
        "area_median_km2": float(frame["area_gages2"].median()),
        "stratum_counts": {
            str(key): int(value)
            for key, value in frame.groupby("stratum").size().sort_index().items()
        },
    }


def main() -> None:
    basins = get_basin_list()
    if len(basins) != 671:
        raise RuntimeError(f"Expected 671 basins, found {len(basins)}")
    frame = load_attributes(basins).set_index("basin_id", drop=False)

    primary = primary_basin_ids()
    available = set(basins) - set(primary)
    subsets: dict[str, list[str]] = {"primary_seed42": primary}
    selection_seeds: dict[str, int] = {"primary_seed42": 42}

    for index, seed in enumerate(REPLICATE_SEEDS, start=1):
        subset_id = f"stratified_{index}"
        candidates = frame.loc[sorted(available)].copy()
        selected = select_stratified(candidates, len(primary), seed)
        subsets[subset_id] = selected
        selection_seeds[subset_id] = seed
        available.difference_update(selected)

    all_selected = [basin for values in subsets.values() for basin in values]
    if len(all_selected) != 250 or len(set(all_selected)) != 250:
        raise RuntimeError("The five scarcity subsets must contain 250 distinct basins")

    rows = []
    summaries = {}
    for subset_id, selected in subsets.items():
        subset_frame = frame.loc[selected]
        summaries[subset_id] = subset_summary(subset_frame)
        for order, basin_id in enumerate(selected, start=1):
            value = frame.loc[basin_id]
            rows.append({
                "subset_id": subset_id,
                "subset_seed": selection_seeds[subset_id],
                "selection_order": order,
                "selection_method": (
                    "random_without_replacement"
                    if subset_id == "primary_seed42"
                    else "disjoint_aridity_area_stratified"
                ),
                "basin_id": basin_id,
                "aridity_quartile": int(value["aridity_quartile"]),
                "area_quartile": int(value["area_quartile"]),
                "aridity": float(value["aridity"]),
                "area_gages2": float(value["area_gages2"]),
            })

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(CSV_PATH, index=False)
    output = {
        "design": "one existing random subset plus four disjoint stratified replication subsets",
        "stratification": "joint quartiles of aridity and log1p drainage area",
        "n_subsets": len(subsets),
        "n_basins_per_subset": len(primary),
        "n_unique_basins": len(set(all_selected)),
        "remaining_unselected_basins": len(available),
        "subset_seeds": selection_seeds,
        "subsets": summaries,
        "csv": CSV_PATH.relative_to(PROJECT_ROOT).as_posix(),
    }
    JSON_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {JSON_PATH}")
    print("Subsets: 5 x 50 basins; overlap: 0")


if __name__ == "__main__":
    main()
