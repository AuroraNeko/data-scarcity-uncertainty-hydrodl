"""Deployable median-bias correction before CQR, on the archived scarcity runs.

This check asks whether correcting the median forecast (residual-mean
adjustment) before interval construction changes conditional coverage. Two
deployable schemes are evaluated offline on all 12 runs; correction
parameters use calibration-period information only:

- "basin": per-basin mean residual delta_b = mean_cal(y - q50) shifts the
  whole (q05, q50, q95) triple;
- "basin_tercile": the same mean within each (basin, calibration-period
  predicted-median stratum) cell, with 0.33/0.67 cuts fitted within each basin
  from calibration predictions (within-basin 0.33/0.67 cuts on calibration predictions).

After the shift, the global conformal radius is refitted on the shifted
calibration scores, and coverage/quality are reported pooled and on the
within-basin observed upper tercile (the within-basin observed upper tercile).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.corrections.common import (LENGTHS, SEEDS, conformal_radius, crosscheck,  # noqa: E402
                           load_run, metrics, observed_split)
from src.evaluation.protocol import fit_basin_thresholds,assign_basin_groups  # noqa: E402

OUT_CSV = ROOT / "results" / "tables" / "biasfix_check.csv"


def upper_tercile_mask(y, basin):
    """Within-basin observed upper-tercile rows, (upper tercile means y > the within-basin 0.67 quantile; ties stay in the middle tercile)."""
    keep = np.zeros(y.size, dtype=bool)
    for b in np.unique(basin):
        rows = np.where(basin == b)[0]
        cut = np.quantile(y[rows], 0.67)
        keep[rows[y[rows] > cut]] = True
    return keep


def basin_median_bias(bias_rows, basin, upper):
    """Revised S13 metric: per-basin mean over high-flow days, then the
    cross-basin median (bias in q50 - y convention; negative = under)."""
    per_basin = [np.mean(bias_rows[(basin == b) & upper])
                 for b in np.unique(basin)]
    return float(np.median(per_basin))


def shifted_metrics(cal_q, cal_y, cal_basin, test_q, test_y, test_basin,
                   scheme):
    if scheme == "none":
        d_cal = np.zeros(cal_y.size)
        d_test = np.zeros(test_y.size)
    elif scheme == "basin":
        residual = cal_y - cal_q[:, 1]
        delta = {b: residual[cal_basin == b].mean()
                 for b in np.unique(cal_basin)}
        d_cal = np.array([delta[b] for b in cal_basin])
        d_test = np.array([delta.get(b, 0.0) for b in test_basin])
    elif scheme == "basin_tercile":
        cuts = fit_basin_thresholds(cal_q[:,1],cal_basin)
        t_cal = assign_basin_groups(cal_q[:,1],cal_basin,cuts)
        t_test = assign_basin_groups(test_q[:,1],test_basin,cuts)
        residual = cal_y - cal_q[:, 1]
        delta = {}
        for b in np.unique(cal_basin):
            for t in range(3):
                rows = (cal_basin == b) & (t_cal == t)
                if rows.any():
                    delta[(b, t)] = residual[rows].mean()
        d_cal = np.array([delta.get((b, t), 0.0)
                          for b, t in zip(cal_basin, t_cal)])
        d_test = np.array([delta.get((b, t), 0.0)
                           for b, t in zip(test_basin, t_test)])
    else:
        raise KeyError(scheme)

    cq, tq = cal_q + d_cal[:, None], test_q + d_test[:, None]
    scores = np.maximum(cq[:, 0] - cal_y, cal_y - cq[:, 2])
    radius = conformal_radius(scores)

    upper = upper_tercile_mask(test_y, test_basin)
    raw = metrics(tq[:, 0], tq[:, 2], test_y)
    cqr = metrics(tq[:, 0] - radius, tq[:, 2] + radius, test_y)
    raw_u = metrics(tq[upper, 0], tq[upper, 2], test_y[upper])
    cqr_u = metrics(tq[upper, 0] - radius, tq[upper, 2] + radius,
                    test_y[upper])
    # q50 - y convention (negative = under-prediction), S13 basin-median metric
    bias = tq[:, 1] - test_y
    return {
        "radius": radius,
        "raw_picp": raw["picp"], "raw_winkler": raw["winkler"],
        "cqr_picp": cqr["picp"], "cqr_winkler": cqr["winkler"],
        "upper_picp_raw": raw_u["picp"], "upper_winkler_raw": raw_u["winkler"],
        "upper_picp_cqr": cqr_u["picp"], "upper_winkler_cqr": cqr_u["winkler"],
        "bias_upper_basinmedian": basin_median_bias(bias, test_basin, upper),
    }


def main():
    rows = []
    for length in LENGTHS:
        for seed in SEEDS:
            _, data, meta = load_run(length, seed)
            cal, test = observed_split(data)
            crosscheck(cal, test, meta)
            for scheme in ("none", "basin", "basin_tercile"):
                m = shifted_metrics(
                    cal["q"], cal["y"], cal["basin"],
                    test["q"], test["y"], test["basin"], scheme)
                rows.append({"length": length, "seed": seed, "scheme": scheme,
                             **m})
                print(f"{length} seed{seed} {scheme:14s} "
                      f"CQR picp={m['cqr_picp']:.3f} "
                      f"W={m['cqr_winkler']:.3f} "
                      f"upper picp={m['upper_picp_cqr']:.3f} "
                      f"upper bias={m['bias_upper_basinmedian']:+.3f}")

    with open(OUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", OUT_CSV)

    print("\n=== mean over seeds ===")
    import pandas as pd
    frame = pd.DataFrame(rows)
    print(frame.groupby(["scheme", "length"])[
        ["raw_picp", "cqr_picp", "cqr_winkler", "upper_picp_cqr",
         "upper_winkler_cqr", "bias_upper_basinmedian"]
    ].mean().round(4).to_string())


if __name__ == "__main__":
    main()
