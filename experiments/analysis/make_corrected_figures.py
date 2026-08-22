"""Generate manuscript figures from the corrected experiment artifacts."""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
SCARCITY_PATH = ROOT / "results" / "tables" / "corrected_scarcity" / "scarcity_summary.json"
BASIN_EFFECT_PATH = ROOT / "results" / "tables" / "corrected_scarcity" / "basin_endpoint_effects.csv"
CONDITIONAL_ROBUSTNESS_PATH = ROOT / "results" / "tables" / "corrected_scarcity" / "conditional_robustness.json"
BASIN_CALIBRATION_PATH = ROOT / "results" / "tables" / "corrected_scarcity" / "basin_calibration_distribution.csv"
METHOD_PATH = ROOT / "results" / "tables" / "corrected_full_671" / "method_comparison.json"
REPLICATION_PATH = ROOT / "results" / "tables" / "scarcity_replications" / "scarcity_replication_summary.json"
EQUAL_UPDATE_PATH = ROOT / "results" / "tables" / "scarcity_equal_updates" / "equal_update_summary.json"
ARCHITECTURE_PATH = ROOT / "results" / "tables" / "scarcity_standard_lstm" / "architecture_sensitivity_summary.json"
OUTPUT_DIR = ROOT / "results" / "figures" / "corrected"
PAPER_DIR = ROOT / "paper" / "figures"
STALE_FIGURE_STEMS = (
    "fig1_degradation",
    "fig2_method_comparison",
    "fig3_calibration",
    "fig4_architecture",
    "fig5_cross_region",
    "fig6_stability_ablation",
    "fig7_robustness_sensitivity",
    "figS1_climate_quartiles",
)

BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
AMBER = "#E69F00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
GRAY = "#62676D"
LIGHT_GRAY = "#D7DADD"
INK = "#202428"


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "axes.titleweight": "semibold",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.7,
        "axes.edgecolor": "#5D6268",
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": LIGHT_GRAY,
        "grid.linewidth": 0.55,
        "grid.alpha": 0.8,
        "lines.linewidth": 1.8,
        "lines.markersize": 5.2,
        "figure.dpi": 180,
        "savefig.dpi": 450,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required result artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def panel_label(axis, label: str) -> None:
    axis.text(
        -0.14,
        1.06,
        label,
        transform=axis.transAxes,
        fontsize=10,
        fontweight="bold",
        color=INK,
        va="top",
    )


def centered_panel_label(axis, label: str) -> None:
    axis.text(
        0.5,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=10,
        fontweight="bold",
        color=INK,
        ha="center",
        va="bottom",
    )


def edge_panel_label(axis, label: str) -> None:
    axis.text(
        -0.055,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=10,
        fontweight="bold",
        color=INK,
        ha="center",
        va="bottom",
    )


def save_figure(figure, stem: str, *, crop: bool = True) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUTPUT_DIR / f"{stem}.pdf"
    png = OUTPUT_DIR / f"{stem}.png"
    options = {"facecolor": "white"}
    if crop:
        options.update({"bbox_inches": "tight", "pad_inches": 0.04})
    figure.savefig(pdf, **options)
    figure.savefig(png, **options)
    shutil.copy2(pdf, PAPER_DIR / pdf.name)
    shutil.copy2(png, PAPER_DIR / png.name)
    plt.close(figure)


def remove_stale_figure_aliases() -> None:
    """Remove superseded numbered copies that could be mistaken for current figures."""
    for directory in (OUTPUT_DIR, PAPER_DIR):
        for stem in STALE_FIGURE_STEMS:
            for suffix in (".pdf", ".png"):
                path = directory / f"{stem}{suffix}"
                if path.exists():
                    path.unlink()


def series(summary: dict, metric: str, field: str = "mean") -> np.ndarray:
    return np.asarray([summary["by_year"][str(year)][metric][field] for year in [1, 3, 5, 15]])


def load_basin_effects(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Required basin-level result table not found: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 50 or len({row["basin_id"] for row in rows}) != 50:
        raise ValueError("The basin endpoint table must contain 50 unique basins")
    return rows


def load_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Required result table not found: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def date_number(value: str) -> float:
    return mdates.date2num(datetime.fromisoformat(value))


def draw_period(axis, y: float, start: str, end: str, color: str, height: float = 0.54) -> None:
    left = date_number(start)
    right = date_number(end) + 1
    axis.barh(y, right - left, left=left, height=height, color=color, edgecolor="white", linewidth=0.6)


def draw_context(axis, y: float, start: str, height: float = 0.54) -> None:
    target_start = datetime.fromisoformat(start)
    context_start = target_start - timedelta(days=180)
    left = mdates.date2num(context_start)
    right = mdates.date2num(target_start)
    axis.barh(
        y,
        right - left,
        left=left,
        height=height,
        color=LIGHT_GRAY,
        edgecolor=GRAY,
        linewidth=0.45,
        hatch="////",
    )


def make_protocol_figure(summary: dict, methods: dict) -> None:
    protocol = summary["protocol"]
    figure = plt.figure(figsize=(7.25, 4.45))
    grid = figure.add_gridspec(
        2,
        1,
        height_ratios=[3.15, 1.0],
        left=0.16,
        right=0.985,
        bottom=0.075,
        top=0.89,
        hspace=0.40,
    )
    axis = figure.add_subplot(grid[0])

    positions = {15: 4.2, 5: 3.2, 3: 2.2, 1: 1.2}
    for years in [15, 5, 3, 1]:
        start, end = protocol["training_target_periods"][str(years)]
        draw_context(axis, positions[years], start)
        draw_period(axis, positions[years], start, end, BLUE)

    calibration_start, calibration_end = protocol["calibration_target_period"]
    test_start, test_end = protocol["test_target_period"]
    evaluation_y = -0.05
    draw_context(axis, evaluation_y, calibration_start)
    draw_period(axis, evaluation_y, calibration_start, calibration_end, GREEN)
    draw_context(axis, evaluation_y, test_start)
    draw_period(axis, evaluation_y, test_start, test_end, AMBER)

    axis.text(
        (date_number(calibration_start) + date_number(calibration_end)) / 2,
        evaluation_y,
        "Calibration",
        ha="center",
        va="center",
        fontsize=7.6,
        color="white",
        fontweight="semibold",
    )
    axis.text(
        (date_number(test_start) + date_number(test_end)) / 2,
        evaluation_y,
        "Test",
        ha="center",
        va="center",
        fontsize=7.6,
        color=INK,
        fontweight="semibold",
    )

    common_end = date_number(calibration_start)
    axis.axvline(common_end, color=INK, linestyle="--", linewidth=0.9, zorder=0)
    axis.text(
        common_end - 120,
        4.82,
        "Common training end",
        ha="right",
        va="bottom",
        fontsize=7.2,
        color=INK,
    )
    axis.axhline(0.58, color=LIGHT_GRAY, linewidth=0.7)
    axis.set_yticks(
        [positions[15], positions[5], positions[3], positions[1], evaluation_y],
        ["15-year training", "5-year training", "3-year training", "1-year training", "Shared evaluation"],
    )
    axis.set_xlim(date_number("1979-12-15"), date_number("2009-02-01"))
    axis.set_ylim(-0.55, 5.25)
    axis.xaxis.set_major_locator(mdates.YearLocator(5))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axis.grid(axis="x")
    axis.tick_params(axis="y", length=0)
    axis.legend(
        handles=[
            Patch(facecolor=BLUE, label="Supervised targets"),
            Patch(facecolor=LIGHT_GRAY, edgecolor=GRAY, hatch="////", label="180-day input context"),
            Patch(facecolor=GREEN, label="Calibration targets"),
            Patch(facecolor=AMBER, label="Test targets"),
        ],
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.005),
        ncol=4,
        borderaxespad=0,
        columnspacing=1.15,
        handlelength=1.4,
    )
    edge_panel_label(axis, "a")

    axis = figure.add_subplot(grid[1])
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.axvline(0.5, ymin=0.05, ymax=0.95, color=LIGHT_GRAY, linewidth=0.9)
    axis.plot([0.015, 0.015], [0.12, 0.88], color=BLUE, linewidth=3.2, solid_capstyle="butt")
    axis.plot([0.535, 0.535], [0.12, 0.88], color=GREEN, linewidth=3.2, solid_capstyle="butt")
    axis.text(0.035, 0.80, "Primary controlled gradient", fontsize=8.4, fontweight="semibold", color=INK)
    axis.text(
        0.035,
        0.51,
        f"{summary['n_basins']} fixed basins x {len(summary['years'])} windows x "
        f"{len(summary['model_seeds'])} seeds",
        fontsize=7.8,
        color=INK,
    )
    axis.text(0.035, 0.29, f"= {protocol['n_model_fits']} quantile-model fits", fontsize=7.8, color=INK)
    axis.text(0.035, 0.08, "Only supervised record length changes", fontsize=7.4, color=GRAY)
    axis.text(0.555, 0.80, "Independent method comparison", fontsize=8.4, fontweight="semibold", color=INK)
    axis.text(0.555, 0.51, f"{methods['protocol']['n_basins']} basins x 3 base methods", fontsize=7.8, color=INK)
    axis.text(0.555, 0.29, f"Raw/CQR = {len(methods['metrics'])} evaluated variants", fontsize=7.8, color=INK)
    axis.text(0.555, 0.08, "Common grid; 365-day sequences", fontsize=7.4, color=GRAY)
    edge_panel_label(axis, "b")

    save_figure(figure, "fig_protocol_timeline", crop=False)


def make_scarcity_figure(summary: dict) -> None:
    labels = ["1", "3", "5", "15"]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(2, 2, figsize=(7.25, 5.75), constrained_layout=True)

    axis = axes[0, 0]
    nse = series(summary, "median_nse")
    nse_sd = series(summary, "median_nse", "sd")
    axis.errorbar(x, nse, yerr=nse_sd, color=BLUE, marker="o", capsize=3)
    axis.set_ylabel("Median basin NSE")
    axis.set_ylim(0.45, 0.70)
    axis.set_xticks(x, labels)
    axis.grid(axis="y")
    effect = summary["per_basin_endpoint_effect"]["across_seed_average"]
    ci = effect["median_delta_ci95"]
    axis.text(
        0.03,
        0.07,
        f"15 minus 1 yr: {effect['median_delta_nse_15_minus_1']:.3f}\n"
        f"95% CI [{ci[0]:.3f}, {ci[1]:.3f}]; "
        f"{100 * effect['fraction_improved_with_15_years']:.0f}% basins improve",
        transform=axis.transAxes,
        fontsize=7.4,
        color=INK,
        bbox={"facecolor": "white", "edgecolor": LIGHT_GRAY, "pad": 3.5},
    )
    panel_label(axis, "a")

    axis = axes[0, 1]
    raw = series(summary, "raw_picp")
    raw_sd = series(summary, "raw_picp", "sd")
    cqr = series(summary, "cqr_picp")
    cqr_sd = series(summary, "cqr_picp", "sd")
    axis.errorbar(x, raw, yerr=raw_sd, color=VERMILLION, marker="o", capsize=3, label="Raw QR")
    axis.errorbar(x, cqr, yerr=cqr_sd, color=GREEN, marker="s", capsize=3, label="Global CQR")
    axis.axhline(0.90, color=GRAY, linestyle="--", linewidth=1.1, label="Nominal 90%")
    axis.set_ylabel("PICP")
    axis.set_ylim(0.73, 0.93)
    axis.set_xticks(x, labels)
    axis.grid(axis="y")
    endpoint = summary["basin_cluster_endpoint_intervals"]["metrics"]["raw_picp"]
    delta = endpoint["difference_15_minus_1"]
    axis.text(
        0.03,
        0.06,
        f"Raw 15 minus 1 yr: {100 * delta['mean']:.1f} pp\n"
        f"95% CI [{100 * delta['ci95'][0]:.1f}, {100 * delta['ci95'][1]:.1f}] pp",
        transform=axis.transAxes,
        fontsize=7.4,
        color=INK,
        bbox={"facecolor": "white", "edgecolor": LIGHT_GRAY, "pad": 3.5},
    )
    axis.legend(loc="lower right", frameon=False)
    panel_label(axis, "b")

    axis = axes[1, 0]
    raw_width = series(summary, "raw_mpiw_log")
    raw_width_sd = series(summary, "raw_mpiw_log", "sd")
    cqr_width = series(summary, "cqr_mpiw_log")
    cqr_width_sd = series(summary, "cqr_mpiw_log", "sd")
    axis.errorbar(x, raw_width, yerr=raw_width_sd, color=VERMILLION, marker="o", capsize=3, label="Raw QR")
    axis.errorbar(x, cqr_width, yerr=cqr_width_sd, color=GREEN, marker="s", capsize=3, label="Global CQR")
    axis.set_ylabel("Mean interval width")
    axis.set_xlabel("Supervised training record (years)")
    axis.set_xticks(x, labels)
    axis.grid(axis="y")
    axis.text(
        0.97,
        0.93,
        "log1p(mm day$^{-1}$) scale",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
        color=GRAY,
    )
    panel_label(axis, "c")

    axis = axes[1, 1]
    raw_score = series(summary, "raw_winkler_log")
    raw_score_sd = series(summary, "raw_winkler_log", "sd")
    cqr_score = series(summary, "cqr_winkler_log")
    cqr_score_sd = series(summary, "cqr_winkler_log", "sd")
    axis.errorbar(x, raw_score, yerr=raw_score_sd, color=VERMILLION, marker="o", capsize=3, label="Raw QR")
    axis.errorbar(x, cqr_score, yerr=cqr_score_sd, color=GREEN, marker="s", capsize=3, label="Global CQR")
    axis.set_ylabel("Winkler score (lower is better)")
    axis.set_xlabel("Supervised training record (years)")
    axis.set_xticks(x, labels)
    axis.grid(axis="y")
    axis.text(
        0.97,
        0.93,
        "log1p(mm day$^{-1}$) scale",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
        color=GRAY,
    )
    panel_label(axis, "d")

    save_figure(figure, "fig1_scarcity_effects")


def make_conditional_figure(summary: dict) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.25, 3.35), constrained_layout=True)
    labels = ["1", "3", "5", "15"]
    x = np.arange(len(labels))
    colors = {"low": BLUE, "normal": AMBER, "high": VERMILLION}
    display = {"low": "Low", "normal": "Middle", "high": "High"}

    axis = axes[0]
    for regime in ["low", "normal", "high"]:
        values = np.asarray([
            summary["conditional_cqr_by_year"][str(year)]["global_cqr"][regime]["mean"]
            for year in [1, 3, 5, 15]
        ])
        errors = np.asarray([
            summary["conditional_cqr_by_year"][str(year)]["global_cqr"][regime]["sd"]
            for year in [1, 3, 5, 15]
        ])
        axis.errorbar(
            x, values, yerr=errors, color=colors[regime], marker="o", capsize=2.5,
            label=f"{display[regime]} observed flow",
        )
    axis.axhline(0.90, color=GRAY, linestyle="--", linewidth=1.1)
    axis.set_xticks(x, labels)
    axis.set_xlabel("Supervised training record (years)")
    axis.set_ylabel("PICP after global CQR")
    axis.set_ylim(0.81, 0.96)
    axis.grid(axis="y")
    axis.legend(frameon=False, loc="lower right")
    panel_label(axis, "a")

    axis = axes[1]
    regimes = ["low", "normal", "high"]
    method_keys = ["raw", "global_cqr", "predicted_regime_cqr", "oracle_observed_regime_cqr"]
    method_labels = ["Raw QR", "Global CQR", "Predicted-regime CQR", "Observed-regime reference"]
    method_colors = [VERMILLION, BLUE, GREEN, GRAY]
    width = 0.19
    for index, (method, label, color) in enumerate(zip(method_keys, method_labels, method_colors)):
        values = [
            summary["conditional_cqr_by_year"]["1"][method][regime]["mean"]
            for regime in regimes
        ]
        errors = [
            summary["conditional_cqr_by_year"]["1"][method][regime]["sd"]
            for regime in regimes
        ]
        offset = (index - 1.5) * width
        axis.bar(
            np.arange(3) + offset,
            values,
            width=width,
            yerr=errors,
            capsize=2,
            color=color,
            alpha=0.9,
            label=label,
            edgecolor="white",
            linewidth=0.5,
            hatch="///" if method == "oracle_observed_regime_cqr" else None,
        )
    axis.axhline(0.90, color=GRAY, linestyle="--", linewidth=1.1)
    axis.set_xticks(np.arange(3), ["Low", "Middle", "High"])
    axis.set_xlabel("Observed test-flow tercile")
    axis.set_ylabel("PICP, 1-year condition")
    axis.set_ylim(0.70, 0.97)
    axis.grid(axis="y")
    axis.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        columnspacing=1.0,
    )
    panel_label(axis, "b")

    save_figure(figure, "fig2_conditional_coverage")


def asymmetric_error(point: float, interval: dict) -> np.ndarray:
    return np.asarray([[point - interval["low"]], [interval["high"] - point]])


def make_basin_endpoint_figure(summary: dict, rows: list[dict]) -> None:
    nse = np.asarray([float(row["delta_nse_15_minus_1"]) for row in rows])
    raw_picp_pp = 100 * np.asarray([
        float(row["delta_raw_picp_15_minus_1"])
        for row in rows
    ])
    nse_effect = summary["per_basin_endpoint_effect"]["across_seed_average"]
    raw_effect = summary["basin_cluster_endpoint_intervals"]["metrics"]["raw_picp"]["difference_15_minus_1"]

    figure, axes = plt.subplots(1, 2, figsize=(7.25, 3.55))
    figure.subplots_adjust(left=0.105, right=0.985, bottom=0.19, top=0.90, wspace=0.25)

    specifications = [
        (
            axes[0],
            nse,
            nse_effect["median_delta_nse_15_minus_1"],
            nse_effect["median_delta_ci95"],
            "NSE difference (15 yr - 1 yr; symmetric-log scale)",
            "Point skill",
            f"Median = {nse_effect['median_delta_nse_15_minus_1']:+.3f}\n"
            f"95% CI [{nse_effect['median_delta_ci95'][0]:+.3f}, "
            f"{nse_effect['median_delta_ci95'][1]:+.3f}]\n"
            f"{int((nse > 0).sum())}/50 basins > 0",
        ),
        (
            axes[1],
            raw_picp_pp,
            100 * raw_effect["mean"],
            [100 * value for value in raw_effect["ci95"]],
            "Raw PICP difference (percentage points)",
            "Raw interval coverage",
            f"Mean = {100 * raw_effect['mean']:+.1f} pp\n"
            f"95% CI [{100 * raw_effect['ci95'][0]:+.1f}, "
            f"{100 * raw_effect['ci95'][1]:+.1f}] pp\n"
            f"{int((raw_picp_pp > 0).sum())}/50 basins > 0",
        ),
    ]
    for index, (axis, values, aggregate, interval, ylabel, title, annotation) in enumerate(specifications):
        ordered = np.sort(values)
        ranks = np.arange(1, len(ordered) + 1)
        colors = np.where(ordered > 0, BLUE, VERMILLION)
        axis.axhspan(interval[0], interval[1], color=BLUE, alpha=0.10, linewidth=0)
        axis.axhline(aggregate, color=BLUE, linestyle="--", linewidth=1.15)
        axis.axhline(0, color=INK, linewidth=0.9)
        axis.scatter(ranks, ordered, c=colors, s=24, edgecolor="white", linewidth=0.45, zorder=3)
        axis.set_xlim(0, 51)
        axis.set_xticks([1, 10, 20, 30, 40, 50])
        axis.set_xlabel("Basin rank (ascending paired effect)")
        axis.set_ylabel(ylabel)
        axis.set_title(title, pad=7)
        if index == 0:
            axis.set_yscale("symlog", linthresh=0.10, linscale=1.0)
            axis.set_yticks([-0.3, -0.1, 0, 0.1, 0.3, 1, 3])
            axis.set_yticklabels(["-0.3", "-0.1", "0", "0.1", "0.3", "1", "3"])
        axis.grid(axis="y")
        axis.text(
            0.035,
            0.96,
            annotation,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            color=INK,
            bbox={"facecolor": "white", "edgecolor": LIGHT_GRAY, "pad": 3.2, "alpha": 0.94},
        )
        axis.text(
            -0.06,
            1.055,
            chr(ord("a") + index),
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            color=INK,
            va="top",
        )

    save_figure(figure, "figS1_basin_endpoint_effects", crop=False)


def make_method_figure(result: dict) -> None:
    names = [
        "mc_dropout_raw",
        "mc_dropout_cqr",
        "single_qr_raw",
        "single_qr_cqr",
        "ensemble_raw",
        "ensemble_cqr",
    ]
    labels = ["MC Dropout", "MC Dropout + CQR", "Single QR", "Single QR + CQR", "Ensemble", "Ensemble + CQR"]
    colors = [PURPLE, PURPLE, BLUE, BLUE, AMBER, AMBER]
    markers = ["o", "s", "o", "s", "o", "s"]
    y = np.arange(len(names))
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(7.25, 3.65),
        gridspec_kw={"width_ratios": [1.14, 1.10, 1.04]},
    )
    figure.subplots_adjust(left=0.195, right=0.985, bottom=0.22, top=0.91, wspace=0.30)

    for panel, metric, xlabel in [
        (axes[0], "picp", "PICP"),
        (axes[1], "winkler_score", "Winkler score (lower is better)"),
    ]:
        for row, (name, color, marker) in enumerate(zip(names, colors, markers)):
            point = result["metrics"][name][metric]
            interval = result["cluster_bootstrap_95ci"][name][metric]
            panel.errorbar(
                point,
                row,
                xerr=asymmetric_error(point, interval),
                color=color,
                marker=marker,
                capsize=2.5,
                linestyle="none",
                markersize=5,
            )
        panel.set_yticks(y)
        panel.grid(axis="x")
        panel.set_xlabel(xlabel)
        panel.invert_yaxis()
    axes[0].set_yticklabels(labels)
    axes[1].tick_params(axis="y", which="both", left=False, labelleft=False)
    axes[0].axvline(0.90, color=GRAY, linestyle="--", linewidth=1.1)
    # keep the widest error bar clear of the y-axis label area
    axes[0].set_xlim(0.64, 0.93)
    axes[0].xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
    centered_panel_label(axes[0], "a")
    centered_panel_label(axes[1], "b")

    axis = axes[2]
    unique_names = ["single_qr_raw", "ensemble_raw", "mc_dropout_raw"]
    unique_labels = ["Single QR", "Ensemble", "MC Dropout"]
    unique_colors = [BLUE, AMBER, PURPLE]
    costs = [result["metrics"][name]["relative_forward_time"] for name in unique_names]
    cost_y = np.arange(3)
    axis.axvline(1.0, color=LIGHT_GRAY, linewidth=1.0, zorder=0)
    for row, value, color in zip(cost_y, costs, unique_colors):
        axis.hlines(row, 1.0, value, color=color, linewidth=3.0, alpha=0.8)
        axis.scatter(value, row, color=color, s=34, zorder=3)
    axis.set_xscale("log")
    axis.set_yticks(cost_y)
    axis.set_yticklabels([])
    axis.tick_params(axis="y", which="both", left=False)
    axis.spines["left"].set_visible(False)
    axis.invert_yaxis()
    axis.set_xlabel("Relative forward time\n(single QR = 1; log scale)")
    axis.grid(axis="x")
    axis.set_xlim(0.28, max(costs) * 1.35)
    for row, label in zip(cost_y, unique_labels):
        axis.text(
            0.78,
            row,
            label,
            transform=axis.transData,
            ha="right",
            va="center",
            fontsize=7.7,
            color=INK,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.90},
            zorder=4,
        )
    for row, value in zip(cost_y, costs):
        is_largest = np.isclose(value, max(costs))
        axis.annotate(
            f"{value:.1f}x",
            xy=(value, row),
            xytext=((-4, 9) if is_largest else (5, 0)),
            textcoords="offset points",
            va=("bottom" if is_largest else "center"),
            ha=("right" if is_largest else "left"),
            fontsize=7.7,
            color=INK,
            clip_on=True,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.90},
        )
    centered_panel_label(axis, "c")

    save_figure(figure, "fig3_method_comparison", crop=False)


def make_climate_figure(result: dict) -> None:
    rows = result["climate_groups"]["rows"]
    quartiles = rows[:4]
    labels = ["Q1 most humid", "Q2", "Q3", "Q4 driest"]
    y = np.arange(4)
    colors = [BLUE, SKY, AMBER, VERMILLION]
    figure, axes = plt.subplots(1, 3, figsize=(7.25, 2.8), constrained_layout=True)
    for axis, metric, xlabel in [
        (axes[0], "median_basin_nse", "Median basin NSE"),
        (axes[1], "picp", "PICP"),
        (axes[2], "mpiw", "Mean interval width"),
    ]:
        values = [row[metric] for row in quartiles]
        axis.barh(y, values, color=colors, height=0.58, alpha=0.9)
        axis.set_yticks(y)
        axis.invert_yaxis()
        axis.set_xlabel(xlabel)
        axis.grid(axis="x")
    axes[0].set_yticklabels(labels)
    axes[1].set_yticklabels([])
    axes[2].set_yticklabels([])
    axes[1].axvline(0.90, color=GRAY, linestyle="--", linewidth=1.1)
    axes[2].text(
        0.98,
        0.05,
        "log1p(mm day$^{-1}$)",
        transform=axes[2].transAxes,
        ha="right",
        fontsize=7.2,
        color=GRAY,
    )
    for label, axis in zip(["a", "b", "c"], axes):
        panel_label(axis, label)
    save_figure(figure, "figS2_climate_quartiles")


def make_high_flow_threshold_figure(result: dict) -> None:
    fractions = np.asarray(result["top_flow_fractions"], dtype=float)
    percentages = 100 * fractions
    figure, axes = plt.subplots(1, 2, figsize=(7.25, 3.15))
    figure.subplots_adjust(left=0.09, right=0.985, bottom=0.20, top=0.87, wspace=0.24)

    year_styles = [
        (1, VERMILLION, "o"),
        (3, AMBER, "s"),
        (5, SKY, "^"),
        (15, BLUE, "D"),
    ]
    for years, color, marker in year_styles:
        entries = [
            result["threshold_curve"][str(years)]["global_cqr"][f"{fraction:.2f}"]["picp"]
            for fraction in fractions
        ]
        values = np.asarray([entry["value"] for entry in entries])
        low = np.asarray([entry["ci95"][0] for entry in entries])
        high = np.asarray([entry["ci95"][1] for entry in entries])
        axes[0].plot(percentages, values, color=color, marker=marker, label=f"{years} years")
        axes[0].fill_between(percentages, low, high, color=color, alpha=0.10, linewidth=0)

    method_styles = [
        ("raw", VERMILLION, "o", "Raw QR"),
        ("global_cqr", BLUE, "s", "Global CQR"),
        ("predicted_regime_cqr", GREEN, "^", "Predicted-regime CQR"),
    ]
    for method, color, marker, label in method_styles:
        entries = [
            result["threshold_curve"]["1"][method][f"{fraction:.2f}"]["picp"]
            for fraction in fractions
        ]
        values = np.asarray([entry["value"] for entry in entries])
        low = np.asarray([entry["ci95"][0] for entry in entries])
        high = np.asarray([entry["ci95"][1] for entry in entries])
        axes[1].plot(percentages, values, color=color, marker=marker, label=label)
        axes[1].fill_between(percentages, low, high, color=color, alpha=0.10, linewidth=0)

    for axis in axes:
        axis.axhline(0.90, color=GRAY, linestyle="--", linewidth=1.0)
        axis.set_xscale("log")
        axis.invert_xaxis()
        axis.set_xticks(percentages)
        axis.set_xticklabels(["100", "33", "20", "10", "5", "2"])
        axis.set_xlabel("Highest-flow observations retained\nwithin each basin (%)")
        axis.set_ylabel("PICP")
        axis.set_ylim(0.48, 0.93)
        axis.grid(axis="y")
        axis.legend(frameon=False, loc="lower left")
    axes[0].set_title("Global CQR across record lengths", pad=7)
    axes[1].set_title("Calibration strategies at 1 year", pad=7)
    edge_panel_label(axes[0], "a")
    edge_panel_label(axes[1], "b")
    save_figure(figure, "figS3_high_flow_thresholds", crop=False)


def make_basin_calibration_figure(rows: list[dict], summary: dict) -> None:
    methods = ["raw", "global_cqr", "predicted_regime_cqr"]
    labels = ["Raw QR", "Global CQR", "Predicted-\nregime CQR"]
    colors = [VERMILLION, BLUE, GREEN]
    figure, axes = plt.subplots(1, 2, figsize=(7.25, 3.15), sharey=True)
    figure.subplots_adjust(left=0.09, right=0.985, bottom=0.30, top=0.86, wspace=0.16)
    rng = np.random.default_rng(20260731)

    for axis, years, label in zip(axes, [1, 15], ["1-year record", "15-year record"]):
        distributions = []
        for method in methods:
            values = np.asarray([
                float(row["picp"])
                for row in rows
                if int(row["years"]) == years and row["method"] == method
            ])
            if len(values) != 50:
                raise ValueError(f"Expected 50 basin values for {years} years and {method}")
            distributions.append(values)
        boxes = axis.boxplot(
            distributions,
            positions=np.arange(3),
            widths=0.48,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": INK, "linewidth": 1.25},
            whiskerprops={"color": GRAY, "linewidth": 0.9},
            capprops={"color": GRAY, "linewidth": 0.9},
        )
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.22)
            patch.set_edgecolor(color)
        for position, (values, color) in enumerate(zip(distributions, colors)):
            jitter = rng.uniform(-0.13, 0.13, size=len(values))
            axis.scatter(
                position + jitter,
                values,
                s=15,
                color=color,
                alpha=0.64,
                edgecolor="white",
                linewidth=0.3,
                zorder=3,
            )
        axis.axhspan(0.85, 0.95, color=LIGHT_GRAY, alpha=0.30, linewidth=0)
        axis.axhline(0.90, color=GRAY, linestyle="--", linewidth=1.0)
        axis.set_xticks(np.arange(3))
        axis.set_xticklabels(labels)
        axis.set_title(label, pad=7)
        axis.set_ylim(0.40, 1.015)
        axis.grid(axis="y")
        within = [
            100 * summary[str(years)][method]["fraction_within_0p85_0p95"]
            for method in methods
        ]
        axis.text(
            0.5,
            -0.27,
            "Within 0.85-0.95: " + ", ".join(f"{value:.0f}%" for value in within),
            transform=axis.transAxes,
            fontsize=7.2,
            color=GRAY,
            ha="center",
            va="top",
            clip_on=False,
        )
    axes[0].set_ylabel("Per-basin PICP")
    edge_panel_label(axes[0], "a")
    edge_panel_label(axes[1], "b")
    save_figure(figure, "figS4_basin_calibration", crop=False)


def make_replication_figure(summary: dict) -> None:
    subset_ids = [
        "primary_seed42",
        "stratified_1",
        "stratified_2",
        "stratified_3",
        "stratified_4",
    ]
    labels = [
        "Primary sample",
        "Stratified sample 1",
        "Stratified sample 2",
        "Stratified sample 3",
        "Stratified sample 4",
        "Combined (250 basins)",
    ]
    y = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(7.25, 3.45), sharey=True)
    figure.subplots_adjust(left=0.205, right=0.985, bottom=0.19, top=0.88, wspace=0.24)

    panels = [
        ("nse", 1.0, "Paired NSE effect (15 yr - 1 yr)", "Median within subset"),
        ("raw_picp", 100.0, "Raw PICP effect (percentage points)", "Mean within subset"),
    ]
    for panel_index, (axis, (metric, scale, xlabel, title)) in enumerate(zip(axes, panels)):
        entries = [
            summary["by_subset"][subset_id]["effects_15_minus_1"][metric]
            for subset_id in subset_ids
        ]
        entries.append(summary["combined_effects_15_minus_1"][metric])
        for row, entry in enumerate(entries):
            point = scale * entry["estimate"]
            low, high = [scale * value for value in entry["ci95"]]
            color = VERMILLION if row == len(entries) - 1 else BLUE
            marker = "D" if row == len(entries) - 1 else "o"
            axis.errorbar(
                point,
                row,
                xerr=np.asarray([[point - low], [high - point]]),
                color=color,
                marker=marker,
                capsize=2.8,
                linestyle="none",
                markersize=5.2,
                zorder=3,
            )
        axis.axvline(0, color=INK, linewidth=0.9)
        axis.axhline(4.5, color=LIGHT_GRAY, linewidth=0.8)
        axis.set_yticks(y)
        axis.set_ylim(len(labels) - 0.5, -0.5)
        axis.set_xlabel(xlabel)
        axis.set_title(title, pad=7)
        axis.grid(axis="x")
        edge_panel_label(axis, chr(ord("a") + panel_index))
    axes[0].set_yticklabels(labels)
    axes[1].tick_params(axis="y", which="both", left=False, labelleft=False)
    save_figure(figure, "figS5_subset_replication", crop=False)


def make_equal_update_figure(summary: dict) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(7.25, 5.25))
    figure.subplots_adjust(left=0.10, right=0.97, bottom=0.11, top=0.91, hspace=0.42, wspace=0.24)
    x = np.arange(2)
    styles = [
        ("fixed_epochs", BLUE, "o", "Fixed 30 epochs"),
        ("fixed_updates", VERMILLION, "s", "Fixed 1,080 updates"),
    ]
    panels = [
        ("median_nse", "Median basin NSE", "Point skill"),
        ("raw_picp", "Raw PICP", "Marginal coverage"),
        ("raw_mpiw_log", "Mean interval width", "Sharpness"),
        ("raw_winkler_log", "Winkler score (lower is better)", "Interval quality"),
    ]
    for panel_index, (axis, (metric, ylabel, title)) in enumerate(zip(axes.flat, panels)):
        for group, color, marker, label in styles:
            entries = [summary["training_budgets"][group]["by_year"][str(year)][metric] for year in [1, 15]]
            values = np.asarray([entry["mean"] for entry in entries])
            errors = np.asarray([entry["sd"] for entry in entries])
            axis.errorbar(
                x,
                values,
                yerr=errors,
                color=color,
                marker=marker,
                capsize=2.8,
                linestyle="-" if group == "fixed_epochs" else "--",
                markerfacecolor="white" if group == "fixed_updates" else color,
                label=label,
            )
        if metric == "raw_picp":
            axis.axhline(0.90, color=GRAY, linestyle=":", linewidth=1.1)
            axis.text(
                0.03,
                0.90,
                "Nominal 90%",
                transform=axis.get_yaxis_transform(),
                ha="left",
                va="bottom",
                fontsize=7.2,
                color=GRAY,
            )
        axis.set_xticks(x, ["1 year", "15 years"])
        axis.set_ylabel(ylabel)
        axis.set_title(title, pad=7)
        axis.grid(axis="y")
        edge_panel_label(axis, chr(ord("a") + panel_index))
        if metric in {"raw_mpiw_log", "raw_winkler_log"}:
            annotation_x = 0.03 if metric == "raw_mpiw_log" else 0.98
            axis.text(
                annotation_x,
                0.95,
                "log1p(mm day$^{-1}$) scale",
                transform=axis.transAxes,
                ha="left" if metric == "raw_mpiw_log" else "right",
                va="top",
                fontsize=7.0,
                color=GRAY,
            )
    axes[0, 0].legend(frameon=False, loc="upper left")
    save_figure(figure, "figS6_equal_updates", crop=False)


def make_architecture_figure(summary: dict) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(7.25, 5.25))
    figure.subplots_adjust(left=0.10, right=0.97, bottom=0.11, top=0.91, hspace=0.42, wspace=0.24)
    x = np.arange(2)
    styles = [
        ("lpu_stream", BLUE, "o", "LPU-Stream"),
        ("standard_lstm", GREEN, "s", "Standard LSTM"),
    ]
    panels = [
        ("median_nse", "Median basin NSE", "Point skill"),
        ("raw_picp", "Raw PICP", "Marginal coverage"),
        ("raw_mpiw_log", "Mean interval width", "Sharpness"),
        ("raw_winkler_log", "Winkler score (lower is better)", "Interval quality"),
    ]
    for panel_index, (axis, (metric, ylabel, title)) in enumerate(zip(axes.flat, panels)):
        for architecture, color, marker, label in styles:
            architecture_result = summary["architectures"][architecture]
            entries = [architecture_result["by_year"][str(year)][metric] for year in [1, 15]]
            values = np.asarray([entry["mean"] for entry in entries])
            errors = np.asarray([entry["sd"] for entry in entries])
            parameter_label = f"{label} ({architecture_result['n_parameters'] / 1000:.0f}k params)"
            axis.errorbar(
                x,
                values,
                yerr=errors,
                color=color,
                marker=marker,
                capsize=2.8,
                label=parameter_label,
            )
        if metric == "raw_picp":
            axis.axhline(0.90, color=GRAY, linestyle=":", linewidth=1.1)
            axis.text(
                0.03,
                0.90,
                "Nominal 90%",
                transform=axis.get_yaxis_transform(),
                ha="left",
                va="bottom",
                fontsize=7.2,
                color=GRAY,
            )
        axis.set_xticks(x, ["1 year", "15 years"])
        axis.set_ylabel(ylabel)
        axis.set_title(title, pad=7)
        axis.grid(axis="y")
        edge_panel_label(axis, chr(ord("a") + panel_index))
        if metric in {"raw_mpiw_log", "raw_winkler_log"}:
            axis.text(
                0.98,
                0.95,
                "log1p(mm day$^{-1}$) scale",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=7.0,
                color=GRAY,
            )
    axes[0, 0].legend(frameon=False, loc="upper left")
    save_figure(figure, "figS7_architecture_sensitivity", crop=False)


def main() -> None:
    configure_style()
    remove_stale_figure_aliases()
    scarcity = load_json(SCARCITY_PATH)
    methods = load_json(METHOD_PATH)
    basin_effects = load_basin_effects(BASIN_EFFECT_PATH)
    conditional_robustness = load_json(CONDITIONAL_ROBUSTNESS_PATH)
    basin_calibration = load_csv_rows(BASIN_CALIBRATION_PATH)
    replication = load_json(REPLICATION_PATH)
    equal_updates = load_json(EQUAL_UPDATE_PATH)
    architecture = load_json(ARCHITECTURE_PATH)
    make_protocol_figure(scarcity, methods)
    make_scarcity_figure(scarcity)
    make_conditional_figure(scarcity)
    make_method_figure(methods)
    make_basin_endpoint_figure(scarcity, basin_effects)
    make_climate_figure(methods)
    make_high_flow_threshold_figure(conditional_robustness)
    make_basin_calibration_figure(
        basin_calibration,
        conditional_robustness["basin_calibration_summary"],
    )
    make_replication_figure(replication)
    make_equal_update_figure(equal_updates)
    make_architecture_figure(architecture)
    print(f"Corrected figures written to {OUTPUT_DIR} and synchronized to {PAPER_DIR}")


if __name__ == "__main__":
    main()
