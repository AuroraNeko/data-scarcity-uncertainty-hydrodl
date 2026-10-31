"""Audit processed CAMELS files before running experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import DYNAMIC_VARS, STATIC_COLS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "processed" / "camels_us")
    parser.add_argument("--expected-start", default="1980-01-01")
    parser.add_argument("--expected-end", default="2008-12-31")
    parser.add_argument("--calibration-start", default="1995-10-01")
    parser.add_argument("--calibration-end", default="2000-09-30")
    parser.add_argument("--test-start", default="2000-10-01")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "results" / "tables" / "processed_data_audit.json")
    args = parser.parse_args()

    files = sorted(args.processed_dir.glob("*.csv"))
    missing = np.zeros(len(DYNAMIC_VARS), dtype=np.int64)
    values = np.zeros(len(DYNAMIC_VARS), dtype=np.int64)
    test_missing = np.zeros(len(DYNAMIC_VARS), dtype=np.int64)
    test_values = np.zeros(len(DYNAMIC_VARS), dtype=np.int64)
    first_dates = {}
    last_dates = {}
    discontinuous = []
    duplicate_dates = []
    total_rows = 0
    observed_targets = 0
    calibration_rows = 0
    calibration_observed_targets = 0
    test_rows = 0
    test_observed_targets = 0
    invalid_flow_masks = 0
    observed_nonfinite_targets = 0
    masked_finite_targets = 0
    static_missing_basin_counts = {name: 0 for name in STATIC_COLS}
    duplicate_pair_mismatches = {
        "tmax_maurer__tmin_maurer": 0,
        "tmax_nldas__tmin_nldas": 0,
        "tmax_daymet__tmin_daymet": 0,
    }

    columns = ["date", *DYNAMIC_VARS, "flow_mask", "target"]
    for index, path in enumerate(files, start=1):
        frame = pd.read_csv(path, usecols=columns, parse_dates=["date"])
        static_row = pd.read_csv(path, usecols=STATIC_COLS, nrows=1).iloc[0]
        for name in STATIC_COLS:
            static_missing_basin_counts[name] += int(pd.isna(static_row[name]))
        dates = frame["date"]
        first = dates.iloc[0].strftime("%Y-%m-%d")
        last = dates.iloc[-1].strftime("%Y-%m-%d")
        first_dates[first] = first_dates.get(first, 0) + 1
        last_dates[last] = last_dates.get(last, 0) + 1
        if dates.duplicated().any():
            duplicate_dates.append(path.stem)
        if not dates.diff().dropna().eq(pd.Timedelta(days=1)).all():
            discontinuous.append(path.stem)

        dynamic = frame[DYNAMIC_VARS].to_numpy(dtype=np.float64)
        for pair in duplicate_pair_mismatches:
            left, right = pair.split("__")
            duplicate_pair_mismatches[pair] += int(
                (~np.isclose(frame[left], frame[right], equal_nan=True)).sum()
            )
        missing += np.isnan(dynamic).sum(axis=0)
        values += dynamic.shape[0]
        calibration = (
            (frame["date"] >= pd.Timestamp(args.calibration_start))
            & (frame["date"] <= pd.Timestamp(args.calibration_end))
        ).to_numpy()
        test = (frame["date"] >= pd.Timestamp(args.test_start)).to_numpy()
        test_dynamic = dynamic[test]
        test_missing += np.isnan(test_dynamic).sum(axis=0)
        test_values += test_dynamic.shape[0]
        total_rows += len(frame)
        flow_mask = frame["flow_mask"].to_numpy()
        target = frame["target"].to_numpy(dtype=np.float64)
        observed = flow_mask == 1
        masked = flow_mask == 0
        observed_targets += int(observed.sum())
        calibration_rows += int(calibration.sum())
        calibration_observed_targets += int((observed & calibration).sum())
        test_rows += int(test.sum())
        test_observed_targets += int((observed & test).sum())
        invalid_flow_masks += int((~np.isin(flow_mask, [0, 1])).sum())
        observed_nonfinite_targets += int((observed & ~np.isfinite(target)).sum())
        masked_finite_targets += int((masked & np.isfinite(target)).sum())

        if index % 100 == 0:
            print(f"Audited {index}/{len(files)} files", flush=True)

    full_fraction = missing / np.maximum(values, 1)
    test_fraction = test_missing / np.maximum(test_values, 1)
    report = {
        "n_files": len(files),
        "total_rows": total_rows,
        "observed_targets": observed_targets,
        "missing_targets": total_rows - observed_targets,
        "calibration_rows": calibration_rows,
        "calibration_observed_targets": calibration_observed_targets,
        "calibration_missing_targets": calibration_rows - calibration_observed_targets,
        "test_rows": test_rows,
        "test_observed_targets": test_observed_targets,
        "test_missing_targets": test_rows - test_observed_targets,
        "invalid_flow_masks": invalid_flow_masks,
        "observed_nonfinite_targets": observed_nonfinite_targets,
        "masked_finite_targets": masked_finite_targets,
        "static_missing_basin_counts": static_missing_basin_counts,
        "static_imputation": (
            "NaN static values are set to zero after standardization, equivalent "
            "to the corresponding training-basin mean"
        ),
        "first_dates": first_dates,
        "last_dates": last_dates,
        "discontinuous_basins": discontinuous,
        "duplicate_date_basins": duplicate_dates,
        "full_missing_fraction": {
            name: float(value) for name, value in zip(DYNAMIC_VARS, full_fraction)
        },
        "test_missing_fraction": {
            name: float(value) for name, value in zip(DYNAMIC_VARS, test_fraction)
        },
        "max_full_missing_fraction": float(full_fraction.max()),
        "max_test_missing_fraction": float(test_fraction.max()),
        "channel_pair_mismatch_counts": duplicate_pair_mismatches,
        "exact_duplicate_channel_pairs": [
            pair for pair, count in duplicate_pair_mismatches.items() if count == 0
        ],
    }
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    failed = (
        len(files) != 671
        or first_dates != {args.expected_start: len(files)}
        or last_dates != {args.expected_end: len(files)}
        or discontinuous
        or duplicate_dates
        or test_fraction.max() > 0.01
        or invalid_flow_masks
        or observed_nonfinite_targets
        or masked_finite_targets
        or any(
            count != (3 if name == "geol_porostiy" else 0)
            for name, count in static_missing_basin_counts.items()
        )
        or duplicate_pair_mismatches["tmax_daymet__tmin_daymet"] == 0
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
