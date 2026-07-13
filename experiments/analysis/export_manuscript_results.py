"""Export authoritative corrected results as LaTeX fragments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCARCITY_DIR = ROOT / "results" / "tables" / "corrected_scarcity"
METHOD_DIR = ROOT / "results" / "tables" / "corrected_full_671"
REPLICATION_DIR = ROOT / "results" / "tables" / "scarcity_replications"
EQUAL_UPDATE_DIR = ROOT / "results" / "tables" / "scarcity_equal_updates"
ARCHITECTURE_DIR = ROOT / "results" / "tables" / "scarcity_standard_lstm"
OUTPUT_DIR = ROOT / "paper" / "generated"


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def ci_text(interval: dict, digits: int = 3) -> str:
    return f"[{interval['low']:.{digits}f}, {interval['high']:.{digits}f}]"


def export_macros(
    scarcity: dict,
    methods: dict,
    conditional_robustness: dict,
    replications: dict,
    equal_updates: dict,
    architecture_sensitivity: dict,
) -> None:
    by_year = scarcity["by_year"]
    basin_effect = scarcity["per_basin_endpoint_effect"]["across_seed_average"]
    endpoint = scarcity["basin_cluster_endpoint_intervals"]["metrics"]
    conditional = scarcity["conditional_cqr_by_year"]
    method_metrics = methods["metrics"]
    raw_picp_endpoint = endpoint["raw_picp"]["difference_15_minus_1"]
    cqr_picp_endpoint = endpoint["cqr_picp"]["difference_15_minus_1"]
    raw_high_endpoint = endpoint["raw_high_flow_picp"]["difference_15_minus_1"]
    cqr_high_endpoint = endpoint["cqr_high_flow_picp"]["difference_15_minus_1"]
    predicted_overall = np.mean([
        scarcity["conditional_cqr_by_run"][f"1yr_seed{seed}"]["overall"]
        ["predicted_regime_cqr"]["picp"]
        for seed in [42, 123, 456]
    ])
    width_increase = (
        by_year["1"]["raw_mpiw_log"]["mean"]
        / by_year["15"]["raw_mpiw_log"]["mean"]
        - 1
    )
    winkler_increase = (
        by_year["1"]["raw_winkler_log"]["mean"]
        / by_year["15"]["raw_winkler_log"]["mean"]
        - 1
    )
    replicated_nse = replications["combined_effects_15_minus_1"]["nse"]
    replicated_raw_picp = replications["combined_effects_15_minus_1"]["raw_picp"]
    threshold_33_global = conditional_robustness["threshold_curve"]["1"]["global_cqr"]["0.33"]["picp"]
    threshold_33_predicted = conditional_robustness["threshold_curve"]["1"]["predicted_regime_cqr"]["0.33"]["picp"]
    threshold_02_global = conditional_robustness["threshold_curve"]["1"]["global_cqr"]["0.02"]["picp"]
    threshold_02_predicted = conditional_robustness["threshold_curve"]["1"]["predicted_regime_cqr"]["0.02"]["picp"]
    threshold_02_delta = conditional_robustness["predicted_minus_global"]["1"]["0.02"]
    basin_calibration = conditional_robustness["basin_calibration_summary"]
    equal = equal_updates["training_budgets"]["fixed_updates"]
    equal_effect = equal["effects_15_minus_1"]["nse"]
    standard_lstm = architecture_sensitivity["architectures"]["standard_lstm"]
    standard_effect = standard_lstm["effects_15_minus_1"]["nse"]

    values = [
        macro("OneYearMedianNSE", f"{by_year['1']['median_nse']['mean']:.3f}"),
        macro("FifteenYearMedianNSE", f"{by_year['15']['median_nse']['mean']:.3f}"),
        macro("ScarcityMedianNSEDelta", f"{basin_effect['median_delta_nse_15_minus_1']:.3f}"),
        macro("ScarcityMedianNSEDeltaCI", f"[{basin_effect['median_delta_ci95'][0]:.3f}, {basin_effect['median_delta_ci95'][1]:.3f}]"),
        macro("ScarcityFractionBasinsImproved", f"{100 * basin_effect['fraction_improved_with_15_years']:.0f}\\%"),
        macro("OneYearRawPICP", f"{by_year['1']['raw_picp']['mean']:.3f}"),
        macro("FifteenYearRawPICP", f"{by_year['15']['raw_picp']['mean']:.3f}"),
        macro("OneYearCQRPICP", f"{by_year['1']['cqr_picp']['mean']:.3f}"),
        macro("FifteenYearCQRPICP", f"{by_year['15']['cqr_picp']['mean']:.3f}"),
        macro("RawPICPEndpointPP", f"{100 * endpoint['raw_picp']['difference_15_minus_1']['mean']:.1f}"),
        macro("RawPICPEndpointPPCI", f"[{100 * endpoint['raw_picp']['difference_15_minus_1']['ci95'][0]:.1f}, {100 * endpoint['raw_picp']['difference_15_minus_1']['ci95'][1]:.1f}]"),
        macro("RawPICPPositiveBasins", f"{100 * raw_picp_endpoint['fraction_basins_positive']:.0f}\\%"),
        macro("CQRPICPEndpointPP", f"{100 * cqr_picp_endpoint['mean']:.1f}"),
        macro("CQRPICPEndpointPPCI", f"[{100 * cqr_picp_endpoint['ci95'][0]:.1f}, {100 * cqr_picp_endpoint['ci95'][1]:.1f}]"),
        macro("OneYearRawWidth", f"{by_year['1']['raw_mpiw_log']['mean']:.3f}"),
        macro("FifteenYearRawWidth", f"{by_year['15']['raw_mpiw_log']['mean']:.3f}"),
        macro("OneYearRawWinkler", f"{by_year['1']['raw_winkler_log']['mean']:.3f}"),
        macro("FifteenYearRawWinkler", f"{by_year['15']['raw_winkler_log']['mean']:.3f}"),
        macro("OneYearRawWidthIncreasePct", f"{100 * width_increase:.0f}\\%"),
        macro("OneYearRawWinklerIncreasePct", f"{100 * winkler_increase:.0f}\\%"),
        macro("OneYearRawHighPICP", f"{by_year['1']['raw_high_flow_picp']['mean']:.3f}"),
        macro("FifteenYearRawHighPICP", f"{by_year['15']['raw_high_flow_picp']['mean']:.3f}"),
        macro("OneYearCQRHighPICP", f"{conditional['1']['global_cqr']['high']['mean']:.3f}"),
        macro("OneYearCQRLowPICP", f"{conditional['1']['global_cqr']['low']['mean']:.3f}"),
        macro("OneYearPredCQRHighPICP", f"{conditional['1']['predicted_regime_cqr']['high']['mean']:.3f}"),
        macro("OneYearOracleCQRHighPICP", f"{conditional['1']['oracle_observed_regime_cqr']['high']['mean']:.3f}"),
        macro("OneYearPredCQROverallPICP", f"{predicted_overall:.3f}"),
        macro("RawHighPICPEndpointPP", f"{100 * raw_high_endpoint['mean']:.1f}"),
        macro("RawHighPICPEndpointPPCI", f"[{100 * raw_high_endpoint['ci95'][0]:.1f}, {100 * raw_high_endpoint['ci95'][1]:.1f}]"),
        macro("CQRHighPICPEndpointPP", f"{100 * cqr_high_endpoint['mean']:.1f}"),
        macro("CQRHighPICPEndpointPPCI", f"[{100 * cqr_high_endpoint['ci95'][0]:.1f}, {100 * cqr_high_endpoint['ci95'][1]:.1f}]"),
        macro("FullSingleRawPICP", f"{method_metrics['single_qr_raw']['picp']:.3f}"),
        macro("FullSingleCQRPICP", f"{method_metrics['single_qr_cqr']['picp']:.3f}"),
        macro("FullEnsembleRawPICP", f"{method_metrics['ensemble_raw']['picp']:.3f}"),
        macro("FullEnsembleCQRPICP", f"{method_metrics['ensemble_cqr']['picp']:.3f}"),
        macro("FullMCRawPICP", f"{method_metrics['mc_dropout_raw']['picp']:.3f}"),
        macro("FullMCCQRPICP", f"{method_metrics['mc_dropout_cqr']['picp']:.3f}"),
        macro("FullMCCQRPICPCI", ci_text(methods["cluster_bootstrap_95ci"]["mc_dropout_cqr"]["picp"])),
        macro("FullSingleCQRWinkler", f"{method_metrics['single_qr_cqr']['winkler_score']:.3f}"),
        macro("FullEnsembleCQRWinkler", f"{method_metrics['ensemble_cqr']['winkler_score']:.3f}"),
        macro("FullMCCQRWinkler", f"{method_metrics['mc_dropout_cqr']['winkler_score']:.3f}"),
        macro("FullSingleRelativeTime", f"{method_metrics['single_qr_raw']['relative_forward_time']:.1f}"),
        macro("FullEnsembleRelativeTime", f"{method_metrics['ensemble_raw']['relative_forward_time']:.1f}"),
        macro("FullMCRelativeTime", f"{method_metrics['mc_dropout_raw']['relative_forward_time']:.1f}"),
        macro("ReplicatedNSEDelta", f"{replicated_nse['estimate']:.3f}"),
        macro("ReplicatedNSEDeltaCI", f"[{replicated_nse['ci95'][0]:.3f}, {replicated_nse['ci95'][1]:.3f}]"),
        macro("ReplicatedNSEPositiveBasins", f"{100 * replicated_nse['fraction_basins_positive']:.0f}\\%"),
        macro("ReplicatedRawPICPEndpointPP", f"{100 * replicated_raw_picp['estimate']:.1f}"),
        macro("ReplicatedRawPICPEndpointPPCI", f"[{100 * replicated_raw_picp['ci95'][0]:.1f}, {100 * replicated_raw_picp['ci95'][1]:.1f}]"),
        macro("ReplicatedRawPICPPositiveBasins", f"{100 * replicated_raw_picp['fraction_basins_positive']:.0f}\\%"),
        macro("OneYearGlobalTopThirdPICP", f"{threshold_33_global['value']:.3f}"),
        macro("OneYearPredTopThirdPICP", f"{threshold_33_predicted['value']:.3f}"),
        macro("OneYearGlobalTopTwoPICP", f"{threshold_02_global['value']:.3f}"),
        macro("OneYearPredTopTwoPICP", f"{threshold_02_predicted['value']:.3f}"),
        macro("OneYearPredTopTwoGainPP", f"{100 * threshold_02_delta['mean_delta_picp']:.1f}"),
        macro("OneYearPredTopTwoGainPPCI", f"[{100 * threshold_02_delta['mean_delta_picp_ci95'][0]:.1f}, {100 * threshold_02_delta['mean_delta_picp_ci95'][1]:.1f}]"),
        macro("OneYearGlobalBasinBandPct", f"{100 * basin_calibration['1']['global_cqr']['fraction_within_0p85_0p95']:.0f}\\%"),
        macro("OneYearPredBasinBandPct", f"{100 * basin_calibration['1']['predicted_regime_cqr']['fraction_within_0p85_0p95']:.0f}\\%"),
        macro("FifteenYearGlobalBasinBandPct", f"{100 * basin_calibration['15']['global_cqr']['fraction_within_0p85_0p95']:.0f}\\%"),
        macro("EqualUpdateOneYearNSE", f"{equal['by_year']['1']['median_nse']['mean']:.3f}"),
        macro("EqualUpdateFifteenYearNSE", f"{equal['by_year']['15']['median_nse']['mean']:.3f}"),
        macro("EqualUpdateNSEDelta", f"{equal_effect['estimate']:.3f}"),
        macro("EqualUpdateNSEDeltaCI", f"[{equal_effect['ci95'][0]:.3f}, {equal_effect['ci95'][1]:.3f}]"),
        macro("EqualUpdateFifteenYearRawPICP", f"{equal['by_year']['15']['raw_picp']['mean']:.3f}"),
        macro("EqualUpdateFifteenYearRawWidth", f"{equal['by_year']['15']['raw_mpiw_log']['mean']:.3f}"),
        macro("EqualUpdateFifteenYearRawWinkler", f"{equal['by_year']['15']['raw_winkler_log']['mean']:.3f}"),
        macro("FixedEpochFifteenYearSteps", f"{equal_updates['training_budgets']['fixed_epochs']['optimizer_budget']['15']['optimizer_steps']:,}"),
        macro("EqualUpdateSteps", f"{equal_updates['design']['fixed_update_budget']:,}"),
        macro("StandardLSTMParameters", f"{standard_lstm['n_parameters']:,}"),
        macro("StandardLSTMOneYearNSE", f"{standard_lstm['by_year']['1']['median_nse']['mean']:.3f}"),
        macro("StandardLSTMFifteenYearNSE", f"{standard_lstm['by_year']['15']['median_nse']['mean']:.3f}"),
        macro("StandardLSTMNSEDelta", f"{standard_effect['estimate']:.3f}"),
        macro("StandardLSTMNSEDeltaCI", f"[{standard_effect['ci95'][0]:.3f}, {standard_effect['ci95'][1]:.3f}]"),
        macro("StandardLSTMFifteenYearRawPICP", f"{standard_lstm['by_year']['15']['raw_picp']['mean']:.3f}"),
        macro("StandardLSTMFifteenYearRawWidth", f"{standard_lstm['by_year']['15']['raw_mpiw_log']['mean']:.3f}"),
        macro("StandardLSTMFifteenYearRawWinkler", f"{standard_lstm['by_year']['15']['raw_winkler_log']['mean']:.3f}"),
    ]
    (OUTPUT_DIR / "result_macros.tex").write_text("\n".join(values) + "\n", encoding="ascii")


def export_scarcity_table(scarcity: dict) -> None:
    target_counts = {}
    for years in [1, 3, 5, 15]:
        result = read_json(SCARCITY_DIR / f"scarcity_{years}yr_basinseed42_modelseed42.json")
        target_counts[years] = result["train_observed_targets"]
    rows = []
    for years in [1, 3, 5, 15]:
        values = scarcity["by_year"][str(years)]
        rows.append(
            f"{years} & {target_counts[years]:,} & "
            f"{values['median_nse']['mean']:.3f} $\\pm$ {values['median_nse']['sd']:.3f} & "
            f"{values['raw_picp']['mean']:.3f} $\\pm$ {values['raw_picp']['sd']:.3f} & "
            f"{values['cqr_picp']['mean']:.3f} $\\pm$ {values['cqr_picp']['sd']:.3f} & "
            f"{values['raw_mpiw_log']['mean']:.3f} & "
            f"{values['raw_winkler_log']['mean']:.3f} \\\\"
        )
    (OUTPUT_DIR / "scarcity_rows.tex").write_text("\n".join(rows) + "\n", encoding="ascii")


def export_method_table(methods: dict) -> None:
    definitions = [
        ("MC Dropout", "mc_dropout_raw"),
        ("MC Dropout + CQR", "mc_dropout_cqr"),
        ("Single QR", "single_qr_raw"),
        ("Single QR + CQR", "single_qr_cqr"),
        ("Mean-quantile ensemble", "ensemble_raw"),
        ("Mean-quantile ensemble + CQR", "ensemble_cqr"),
    ]
    rows = []
    for label, key in definitions:
        value = methods["metrics"][key]
        interval = methods["cluster_bootstrap_95ci"][key]["picp"]
        rows.append(
            f"{label} & {value['median_basin_nse']:.3f} & "
            f"{value['picp']:.3f} {ci_text(interval)} & "
            f"{value['mpiw']:.3f} & {value['winkler_score']:.3f} & "
            f"{value['forward_passes_per_sample']} \\\\"
        )
    (OUTPUT_DIR / "method_rows.tex").write_text("\n".join(rows) + "\n", encoding="ascii")

    zh_labels = [
        "MC Dropout",
        "MC Dropout + CQR",
        "单模型分位数回归",
        "单模型分位数回归 + CQR",
        "均值分位数集成",
        "均值分位数集成 + CQR",
    ]
    zh_rows = []
    for label, (_, key) in zip(zh_labels, definitions):
        value = methods["metrics"][key]
        interval = methods["cluster_bootstrap_95ci"][key]["picp"]
        zh_rows.append(
            f"{label} & {value['median_basin_nse']:.3f} & "
            f"{value['picp']:.3f} {ci_text(interval)} & "
            f"{value['mpiw']:.3f} & {value['winkler_score']:.3f} & "
            f"{value['forward_passes_per_sample']} \\\\"
        )
    (OUTPUT_DIR / "method_rows_zh.tex").write_text("\n".join(zh_rows) + "\n", encoding="utf-8")


def export_conditional_table(scarcity: dict) -> None:
    labels = [("Low", "low"), ("Middle", "normal"), ("High", "high")]
    methods = ["raw", "global_cqr", "predicted_regime_cqr", "oracle_observed_regime_cqr"]
    rows = []
    for label, regime in labels:
        values = [
            scarcity["conditional_cqr_by_year"]["1"][method][regime]["mean"]
            for method in methods
        ]
        rows.append(
            f"{label} & {values[0]:.3f} & {values[1]:.3f} & "
            f"{values[2]:.3f} & {values[3]:.3f} \\\\"
        )
    (OUTPUT_DIR / "conditional_rows.tex").write_text("\n".join(rows) + "\n", encoding="ascii")
    zh_rows = []
    for label, regime in [("低流量", "low"), ("中等流量", "normal"), ("高流量", "high")]:
        values = [
            scarcity["conditional_cqr_by_year"]["1"][method][regime]["mean"]
            for method in methods
        ]
        zh_rows.append(
            f"{label} & {values[0]:.3f} & {values[1]:.3f} & "
            f"{values[2]:.3f} & {values[3]:.3f} \\\\"
        )
    (OUTPUT_DIR / "conditional_rows_zh.tex").write_text("\n".join(zh_rows) + "\n", encoding="utf-8")


def export_climate_table(methods: dict) -> None:
    rows = []
    for row in methods["climate_groups"]["rows"]:
        rows.append(
            f"{row['group']} & {row['n_basins']} & "
            f"{row['aridity_min']:.2f}--{row['aridity_max']:.2f} & "
            f"{row['median_basin_nse']:.3f} & {row['picp']:.3f} & {row['mpiw']:.3f} \\\\"
        )
    (OUTPUT_DIR / "climate_rows.tex").write_text("\n".join(rows) + "\n", encoding="ascii")
    zh_labels = ["Q1 最湿润", "Q2 湿润过渡", "Q3 干旱过渡", "Q4 最干旱", "全部 CAMELS-US"]
    zh_rows = []
    for label, row in zip(zh_labels, methods["climate_groups"]["rows"]):
        zh_rows.append(
            f"{label} & {row['n_basins']} & "
            f"{row['aridity_min']:.2f}--{row['aridity_max']:.2f} & "
            f"{row['median_basin_nse']:.3f} & {row['picp']:.3f} & {row['mpiw']:.3f} \\\\"
        )
    (OUTPUT_DIR / "climate_rows_zh.tex").write_text("\n".join(zh_rows) + "\n", encoding="utf-8")


def export_sensitivity_table(scarcity: dict) -> None:
    rows = []
    by_seed = scarcity["calibration_window_sensitivity_15yr"]
    for years in [1, 2, 3, 4, 5]:
        values = [by_seed[str(seed)][str(years)] for seed in [42, 123, 456]]
        n_cal = [value["n_calibration"] for value in values]
        q_cal = np.asarray([value["q_cal_log"] for value in values])
        picp = np.asarray([value["picp"] for value in values])
        mpiw = np.asarray([value["mpiw"] for value in values])
        winkler = np.asarray([value["winkler_score"] for value in values])
        rows.append(
            f"{years} & {int(np.mean(n_cal)):,} & "
            f"{q_cal.mean():.3f} $\\pm$ {q_cal.std(ddof=1):.3f} & "
            f"{picp.mean():.3f} $\\pm$ {picp.std(ddof=1):.3f} & "
            f"{mpiw.mean():.3f} & {winkler.mean():.3f} \\\\"
        )
    (OUTPUT_DIR / "sensitivity_rows.tex").write_text("\n".join(rows) + "\n", encoding="ascii")


def export_replication_table(replications: dict) -> None:
    labels = [
        ("Primary sample", "primary_seed42"),
        ("Stratified sample 1", "stratified_1"),
        ("Stratified sample 2", "stratified_2"),
        ("Stratified sample 3", "stratified_3"),
        ("Stratified sample 4", "stratified_4"),
    ]
    rows = []
    for label, subset_id in labels:
        effects = replications["by_subset"][subset_id]["effects_15_minus_1"]
        nse = effects["nse"]
        picp = effects["raw_picp"]
        rows.append(
            f"{label} & {nse['estimate']:.3f} [{nse['ci95'][0]:.3f}, {nse['ci95'][1]:.3f}] & "
            f"{100 * nse['fraction_basins_positive']:.0f}\\% & "
            f"{100 * picp['estimate']:.1f} [{100 * picp['ci95'][0]:.1f}, {100 * picp['ci95'][1]:.1f}] & "
            f"{100 * picp['fraction_basins_positive']:.0f}\\% \\\\"
        )
    effects = replications["combined_effects_15_minus_1"]
    nse = effects["nse"]
    picp = effects["raw_picp"]
    rows.append(
        f"Combined & {nse['estimate']:.3f} [{nse['ci95'][0]:.3f}, {nse['ci95'][1]:.3f}] & "
        f"{100 * nse['fraction_basins_positive']:.0f}\\% & "
        f"{100 * picp['estimate']:.1f} [{100 * picp['ci95'][0]:.1f}, {100 * picp['ci95'][1]:.1f}] & "
        f"{100 * picp['fraction_basins_positive']:.0f}\\% \\\\"
    )
    (OUTPUT_DIR / "replication_rows.tex").write_text("\n".join(rows) + "\n", encoding="ascii")


def export_equal_update_table(equal_updates: dict) -> None:
    labels = [("Fixed 30 epochs", "fixed_epochs"), ("Fixed 1,080 updates", "fixed_updates")]
    rows = []
    for label, group in labels:
        values = equal_updates["training_budgets"][group]
        for years in [1, 15]:
            by_year = values["by_year"][str(years)]
            rows.append(
                f"{label} & {years} & {values['optimizer_budget'][str(years)]['optimizer_steps']:,} & "
                f"{by_year['median_nse']['mean']:.3f} $\\pm$ {by_year['median_nse']['sd']:.3f} & "
                f"{by_year['raw_picp']['mean']:.3f} & {by_year['cqr_picp']['mean']:.3f} & "
                f"{by_year['raw_mpiw_log']['mean']:.3f} & {by_year['raw_winkler_log']['mean']:.3f} \\\\"
            )
    (OUTPUT_DIR / "equal_update_rows.tex").write_text("\n".join(rows) + "\n", encoding="ascii")


def export_architecture_table(architecture_sensitivity: dict) -> None:
    labels = [("LPU-Stream", "lpu_stream"), ("Standard LSTM", "standard_lstm")]
    rows = []
    for label, architecture in labels:
        result = architecture_sensitivity["architectures"][architecture]
        for years in [1, 15]:
            by_year = result["by_year"][str(years)]
            rows.append(
                f"{label} & {years} & {result['n_parameters']:,} & "
                f"{by_year['median_nse']['mean']:.3f} $\\pm$ {by_year['median_nse']['sd']:.3f} & "
                f"{by_year['raw_picp']['mean']:.3f} & {by_year['cqr_picp']['mean']:.3f} & "
                f"{by_year['raw_mpiw_log']['mean']:.3f} & {by_year['raw_winkler_log']['mean']:.3f} \\\\"
            )
    (OUTPUT_DIR / "architecture_rows.tex").write_text("\n".join(rows) + "\n", encoding="ascii")


def main() -> None:
    scarcity = read_json(SCARCITY_DIR / "scarcity_summary.json")
    methods = read_json(METHOD_DIR / "method_comparison.json")
    conditional_robustness = read_json(SCARCITY_DIR / "conditional_robustness.json")
    replications = read_json(REPLICATION_DIR / "scarcity_replication_summary.json")
    equal_updates = read_json(EQUAL_UPDATE_DIR / "equal_update_summary.json")
    architecture_sensitivity = read_json(ARCHITECTURE_DIR / "architecture_sensitivity_summary.json")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    export_macros(
        scarcity,
        methods,
        conditional_robustness,
        replications,
        equal_updates,
        architecture_sensitivity,
    )
    export_scarcity_table(scarcity)
    export_conditional_table(scarcity)
    export_method_table(methods)
    export_climate_table(methods)
    export_sensitivity_table(scarcity)
    export_replication_table(replications)
    export_equal_update_table(equal_updates)
    export_architecture_table(architecture_sensitivity)
    print(f"Generated manuscript fragments in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
