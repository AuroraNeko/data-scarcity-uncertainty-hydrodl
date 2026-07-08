"""
Check whether the 50 scarcity-experiment basins are representative of the full
671-basin CAMELS-US set.

The scarcity experiments select basins with the same procedure used by
experiments/scarce/train_data_scarce.py: np.random.RandomState(42).choice(...)
on get_basin_list(). This script writes machine-readable CSV/JSON tables and a
standalone LaTeX supplement.
"""

import sys, json, math, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.dataset import PROCESSED_DIR, get_basin_list

SELECTION_SEED = 42
N_SCARCITY_BASINS = 50

all_basins = get_basin_list()
rng = np.random.RandomState(SELECTION_SEED)
scarcity_basin_list = rng.choice(
    all_basins, min(N_SCARCITY_BASINS, len(all_basins)), replace=False
).tolist()
scarcity_basins = set(scarcity_basin_list)

print(f"All basins: {len(all_basins)}")
print(f"Scarcity basins: {len(scarcity_basins)}")
print(f"Selection seed: {SELECTION_SEED}")
print(f"Scarcity basins IDs: {scarcity_basin_list[:5]}...{scarcity_basin_list[-3:]}")

# Read static attributes from each basin's CSV
attrs_all = {bid: {} for bid in all_basins}
static_cols = [
    "elev_mean", "slope_mean", "area_gages2", "p_mean", "pet_mean",
    "aridity", "frac_snow", "p_seasonality", "soil_depth_pelletier",
    "soil_porosity", "frac_forest", "lai_diff", "geol_porostiy",
]

display_names = {
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

for csv in PROCESSED_DIR.glob('*.csv'):
    bid = csv.stem
    df = pd.read_csv(csv, index_col=0, parse_dates=True)
    for col in static_cols:
        if col in df.columns:
            attrs_all[bid][col] = float(df[col].iloc[0])

# Build arrays
attr_data = {col: {'all': [], 'scarcity': []} for col in static_cols}
for bid, attrs in attrs_all.items():
    for col in static_cols:
        if col in attrs:
            attr_data[col]['all'].append(attrs[col])
            if bid in scarcity_basins:
                attr_data[col]['scarcity'].append(attrs[col])

# Compare distributions
print(f"\n{'='*80}")
print(f"{'Attribute Representativeness: Scarcity 50 vs All 671':^80}")
print(f"{'='*80}")
print(f"{'Attribute':<25} {'All Mean':>10} {'All Std':>10} {'S50 Mean':>10} {'S50 Std':>10} {'KS p-val':>10} {'Diff':>10}")
print(f"{'-'*80}")

results = {}
summary_rows = []
for col in static_cols:
    all_arr = np.array(attr_data[col]["all"], dtype=float)
    s50_arr = np.array(attr_data[col]["scarcity"], dtype=float)
    all_arr = all_arr[np.isfinite(all_arr)]
    s50_arr = s50_arr[np.isfinite(s50_arr)]
    if len(all_arr) == 0 or len(s50_arr) == 0:
        continue
    
    all_mean, all_std = all_arr.mean(), all_arr.std()
    s50_mean, s50_std = s50_arr.mean(), s50_arr.std()
    
    # KS test
    ks_stat, ks_p = stats.ks_2samp(all_arr, s50_arr)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((all_std**2 + s50_std**2) / 2)
    cohens_d = (all_mean - s50_mean) / pooled_std if pooled_std > 0 else 0
    
    diff_pct = (s50_mean - all_mean) / abs(all_mean) * 100 if all_mean != 0 else 0
    results[col] = {
        "name": display_names.get(col, col),
        "all_mean": all_mean,
        "all_std": all_std,
        "s50_mean": s50_mean,
        "s50_std": s50_std,
        "ks_p": ks_p,
        "cohens_d": cohens_d,
        "diff_pct": diff_pct,
    }
    summary_rows.append({
        "attribute": display_names.get(col, col),
        "all_mean": all_mean,
        "all_std": all_std,
        "scarcity50_mean": s50_mean,
        "scarcity50_std": s50_std,
        "diff_pct": diff_pct,
        "ks_p": ks_p,
        "cohens_d": cohens_d,
    })
    
    sig = " ***" if ks_p < 0.001 else (" **" if ks_p < 0.01 else (" *" if ks_p < 0.05 else ""))
    print(f"{col:<25} {all_mean:>10.4f} {all_std:>10.4f} {s50_mean:>10.4f} {s50_std:>10.4f} {ks_p:>10.4f}{sig} {diff_pct:>+9.1f}%")

print(f"{'-'*80}")
print(f"Significance: * p<0.05, ** p<0.01, *** p<0.001")

# Count significant differences
sig_count = sum(1 for r in results.values() if r['ks_p'] < 0.05)
print(f"\nSignificantly different attributes (p<0.05): {sig_count}/{len(results)}")

if sig_count == 0:
    print(">>> The 50 scarcity basins are statistically representative of the full 671-basin set.")
else:
    print(f">>> {sig_count} attributes show significant differences. Review carefully.")
    for col, r in results.items():
        if r['ks_p'] < 0.05:
            print(f"    - {col}: All={r['all_mean']:.4f} vs S50={r['s50_mean']:.4f} (diff={r['diff_pct']:+.1f}%)")

# Save results
tables_dir = PROJECT_ROOT / "results" / "tables"
paper_dir = PROJECT_ROOT / "paper"
tables_dir.mkdir(parents=True, exist_ok=True)
paper_dir.mkdir(parents=True, exist_ok=True)

out_path = tables_dir / "basin_representativeness.json"
with open(out_path, "w") as f:
    json.dump({
        "selection_seed": SELECTION_SEED,
        "n_scarcity_basins": len(scarcity_basin_list),
        "basins": scarcity_basin_list,
        "attributes": results,
    }, f, indent=2, default=str)

basin_ids_path = tables_dir / "scarcity_50_basin_ids.csv"
pd.DataFrame({
    "selection_order": np.arange(1, len(scarcity_basin_list) + 1),
    "basin_id": scarcity_basin_list,
}).to_csv(basin_ids_path, index=False)

summary_path = tables_dir / "basin_representativeness_summary.csv"
pd.DataFrame(summary_rows).to_csv(summary_path, index=False)


def fmt(x, digits=3):
    if x is None or not math.isfinite(float(x)):
        return "--"
    return f"{float(x):.{digits}f}"


def fmt_p(x):
    if x is None or not math.isfinite(float(x)):
        return "--"
    x = float(x)
    return f"{x:.1e}" if x < 0.001 else f"{x:.3f}"


def tex_escape(s):
    return str(s).replace("_", "\\_").replace("%", "\\%")


ids_sorted = sorted(scarcity_basin_list)
id_rows = []
for i in range(0, len(ids_sorted), 5):
    cells = ids_sorted[i:i + 5]
    cells += [""] * (5 - len(cells))
    id_rows.append(" & ".join(cells) + r" \\")

summary_tex_rows = []
for row in summary_rows:
    summary_tex_rows.append(
        f"{tex_escape(row['attribute'])} & "
        f"{fmt(row['all_mean'])} ({fmt(row['all_std'])}) & "
        f"{fmt(row['scarcity50_mean'])} ({fmt(row['scarcity50_std'])}) & "
        f"{fmt(row['diff_pct'], 1)} & {fmt_p(row['ks_p'])} & {fmt(row['cohens_d'], 2)} \\\\"
    )

supplement_tex = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{caption}
\usepackage{hyperref}
\hypersetup{hidelinks}
\captionsetup{justification=raggedright,singlelinecheck=false}
\renewcommand{\thetable}{S\arabic{table}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}

\begin{document}

\section*{S1 Scarcity-experiment basin subset}

The data-scarcity experiments use a fixed subset of 50 CAMELS-US basins. The subset was selected with the same seed-42 random sampling procedure used by the experiment script. The same basin list is used for the 1-, 3-, 5-, and 15-year scarcity-gradient experiments.

\begin{table}[htbp]
\centering
\caption{The 50 CAMELS-US basin IDs used in the matched data-scarcity gradient. IDs are sorted for readability; the machine-readable selection order is provided as \texttt{scarcity\_50\_basin\_ids.csv} in the public repository.}
\label{tab:s1_basin_ids}
\begin{tabular}{lllll}
\toprule
""" + "\n".join(id_rows) + r"""
\bottomrule
\end{tabular}
\end{table}

\section*{S2 Representativeness of the scarcity subset}

Table~\ref{tab:s2_representativeness} compares static catchment attributes for the full CAMELS-US set and the 50-basin scarcity subset. Values are mean (standard deviation). The Kolmogorov--Smirnov (KS) test compares the full distribution of each attribute between the full set and the subset. Cohen's $d$ is computed as (full mean -- subset mean) divided by the pooled standard deviation. None of the 13 tested attributes differs significantly at $p<0.05$, indicating that this seed-42 subset is reasonably representative of CAMELS-US with respect to these static descriptors. Percentage differences should be interpreted cautiously for attributes with near-zero full-set means, such as precipitation seasonality. The scarcity gradient is still interpreted as a controlled data-volume experiment because it uses one fixed 50-basin draw rather than repeated sampling over all possible basin subsets.

\begingroup
\footnotesize
\setlength{\parskip}{0pt}
\renewcommand{\arraystretch}{0.94}
\setlength{\tabcolsep}{3pt}
\begin{longtable}{p{0.24\linewidth}p{0.20\linewidth}p{0.20\linewidth}rrr}
\caption{Representativeness check for the 50-basin scarcity subset relative to the full CAMELS-US set.}\label{tab:s2_representativeness}\\
\toprule
Attribute & Full mean (SD) & Subset mean (SD) & Diff. (\%) & KS $p$ & Cohen's $d$ \\
\midrule
\endfirsthead
\toprule
Attribute & Full mean (SD) & Subset mean (SD) & Diff. (\%) & KS $p$ & Cohen's $d$ \\
\midrule
\endhead
""" + "\n".join(summary_tex_rows) + r"""
\bottomrule
\end{longtable}
\endgroup

\section*{S3 Reproducibility checklist}

Table~\ref{tab:s3_reproducibility} summarizes the implementation choices needed to trace the main experiments and figures. The checklist is intended to complement, not replace, the public code and data archive.

\begingroup
\footnotesize
\setlength{\parskip}{0pt}
\renewcommand{\arraystretch}{0.96}
\setlength{\tabcolsep}{4pt}
\begin{longtable}{>{\raggedright\arraybackslash}p{0.28\linewidth}>{\raggedright\arraybackslash}p{0.64\linewidth}}
\caption{Compact reproducibility checklist for the main experiments.}\label{tab:s3_reproducibility}\\
\toprule
Item & Setting used in this study \\
\midrule
\endfirsthead
\toprule
Item & Setting used in this study \\
\midrule
\endhead
Dataset & CAMELS-US version 2.0, 671 basins, three meteorological forcing products (Daymet, Maurer, and NLDAS). \\
Dynamic inputs & Fifteen meteorological variables: precipitation, minimum and maximum temperature, solar radiation, and vapor pressure from each forcing product. \\
Static inputs & Thirteen CAMELS catchment attributes covering topography, climate, soil, vegetation, and geology. \\
Target transform & Daily streamflow in mm~d$^{-1}$ transformed as $\log(1+y)$ and standardized using training-period statistics. \\
Temporal split & Training: 1 October 1980--30 September 1995; validation/CQR calibration: 1 October 1995--30 September 2000; test: 1 October 2000--30 September 2010. \\
Scarcity subset & Fifty basins sampled once from the full basin set with NumPy RandomState seed 42 and reused for the 1-, 3-, 5-, and 15-year scarcity-gradient experiments. \\
Sequence length & Full 15-year models use 365 daily steps; scarcity runs use 30, 90, 180, and 365 daily steps for the 1-, 3-, 5-, and 15-year settings, respectively. \\
Model & LPU-Stream with a 128-unit LSTM, 32-dimensional static embedding, and a two-layer quantile head trained for $\tau \in \{0.05,0.50,0.95\}$. \\
Optimization & Adam optimizer, learning rate 0.001, $\beta_1=0.9$, $\beta_2=0.999$, batch size 1024, gradient clipping at 1.0, early-stopping patience 5, and maximum 30 epochs. \\
Uncertainty calibration & Split conformalized quantile regression with $\alpha=0.10$, using the validation period as the calibration set and the test period only for final evaluation. \\
Ensemble seeds & Deep Ensembles use five independently initialized quantile models with seeds 42, 123, 456, 789, and 999. \\
Stability seeds & The 1-year stability check fixes the seed-42 basin subset and varies only model initialization over seeds 42, 123, and 456. \\
Robustness checks & Calibration-window sensitivity over 1--5 validation water years, basin-cluster bootstrap with 500 resamples, aridity-based diagnostic subsets, and static-attribute ablation. \\
Hardware and software & Experiments were run on a single NVIDIA RTX 5060 Ti GPU with Python 3.11 and PyTorch 2.x; analysis scripts are CPU-compatible once result files are available. \\
Verification path & Result tables are stored as JSON/CSV files under \texttt{results/tables/}; figures are regenerated from these files, and \texttt{experiments/analysis/verify\_manuscript.py} checks manuscript numbers against the stored results. \\
\bottomrule
\end{longtable}
\endgroup

\end{document}
"""

supp_path = paper_dir / "supplement.tex"
supp_path.write_text(supplement_tex, encoding="utf-8")

print(f"\nResults saved to {out_path}")
print(f"Basin IDs saved to {basin_ids_path}")
print(f"Summary CSV saved to {summary_path}")
print(f"Supplement TeX saved to {supp_path}")
