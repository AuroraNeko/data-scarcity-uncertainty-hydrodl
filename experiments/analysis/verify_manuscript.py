"""Verify corrected result artifacts and manuscript consistency."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[2]
SCARCITY_DIR = ROOT / "results" / "tables" / "corrected_scarcity"
METHOD_DIR = ROOT / "results" / "tables" / "corrected_full_671"
REPLICATION_DIR = ROOT / "results" / "tables" / "scarcity_replications"
EQUAL_UPDATE_DIR = ROOT / "results" / "tables" / "scarcity_equal_updates"
ARCHITECTURE_DIR = ROOT / "results" / "tables" / "scarcity_standard_lstm"
PAPER_DIR = ROOT / "paper"
GENERATED_DIR = PAPER_DIR / "generated"
MANUSCRIPT = PAPER_DIR / "manuscript.tex"
SUPPLEMENT = PAPER_DIR / "supplement.tex"
REFERENCES = PAPER_DIR / "references.tex"
SUBMISSION_DIR = ROOT / "submission" / "hess"
README = ROOT / "README.md"
README_ZH = ROOT / "README.zh-CN.md"
ZENODO_METADATA = ROOT / ".zenodo.json"


failures: list[str] = []


def fail(message: str) -> None:
    failures.append(message)


def require(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok  {message}")
    else:
        fail(message)


def close(label: str, actual: float, expected: float, tolerance: float = 5e-4) -> None:
    require(abs(actual - expected) <= tolerance, f"{label}: {actual:.6f} ~= {expected:.6f}")


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_path(value: str) -> Path:
    """Resolve artifact paths written by either Windows or POSIX runs."""
    parts = value.replace("\\", "/").split("/")
    if (
        not value
        or any(part in {"", ".", ".."} or ":" in part for part in parts)
    ):
        raise ValueError(f"Invalid repository-relative path: {value!r}")
    path = ROOT.joinpath(*parts)
    try:
        path.resolve().relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Path escapes repository root: {value!r}") from exc
    return path


def verify_scarcity() -> dict:
    print("== Corrected scarcity artifacts ==")
    summary = read_json(SCARCITY_DIR / "scarcity_summary.json")
    require(summary["protocol_version"] == "scarcity_v2_fixed_end_fixed_memory", "correct scarcity protocol")
    require(summary["years"] == [1, 3, 5, 15], "four declared training-record lengths")
    require(summary["model_seeds"] == [42, 123, 456], "three model seeds")
    require(summary["n_basins"] == 50, "fixed 50-basin subset")
    protocol = summary["protocol"]
    require(
        protocol["training_target_periods"] == {
            "1": ["1994-10-01", "1995-09-30"],
            "3": ["1992-10-01", "1995-09-30"],
            "5": ["1990-10-01", "1995-09-30"],
            "15": ["1980-10-01", "1995-09-30"],
        },
        "nested training target periods with a common endpoint",
    )
    require(protocol["n_model_fits"] == 12, "four record lengths by three model seeds")

    expected_targets = {1: 18250, 3: 54750, 5: 91300, 15: 266322}
    for years in [1, 3, 5, 15]:
        for seed in [42, 123, 456]:
            path = SCARCITY_DIR / f"scarcity_{years}yr_basinseed42_modelseed{seed}.json"
            result = read_json(path)
            require(result["years"] == years and result["model_seed"] == seed, f"identity {path.name}")
            require(result["seq_len"] == 180 and result["batch_size"] == 512, f"fixed memory/batch {path.name}")
            require(result["epochs"] == 30, f"fixed epoch budget {path.name}")
            require(result["checkpoint_selection"] == "fixed_epoch_budget_final_checkpoint", f"fixed checkpoint rule {path.name}")
            require(result["calibration_target_period"] == ["1995-10-01", "2000-09-30"], f"calibration dates {path.name}")
            require(result["test_target_period"] == ["2000-10-01", "2008-12-31"], f"test dates {path.name}")
            require(result["train_observed_targets"] == expected_targets[years], f"valid labels {path.name}")
            require(result["calibration_windows"] == 91350 and result["test_windows"] == 150700, f"fixed evaluation samples {path.name}")
            require(result["calibration_observed_targets"] == 91350, f"observed calibration targets {path.name}")
            require(result["test_observed_targets"] == 149239, f"observed test targets {path.name}")
            require(result["max_dynamic_missing_fraction"] == 0.0, f"no dynamic missingness {path.name}")
            require(result["raw_interval_metrics"]["crossing_rate"] == 0.0, f"zero raw crossing {path.name}")
            for artifact_key, hash_key in [("checkpoint", "checkpoint_sha256"), ("predictions", "predictions_sha256")]:
                require(
                    "\\" not in result[artifact_key],
                    f"portable repository path: {path.name} {artifact_key}",
                )
                artifact = repository_path(result[artifact_key])
                if artifact.exists():
                    require(sha256(artifact) == result[hash_key], f"SHA-256 matches: {artifact.name}")
                else:
                    print(f"  skip release artifact not present in code checkout: {artifact}")

    by_year = summary["by_year"]
    close("1-year median NSE", by_year["1"]["median_nse"]["mean"], 0.510628)
    close("15-year median NSE", by_year["15"]["median_nse"]["mean"], 0.665768)
    close("1-year raw PICP", by_year["1"]["raw_picp"]["mean"], 0.776424)
    close("15-year raw PICP", by_year["15"]["raw_picp"]["mean"], 0.820782)
    close("1-year CQR PICP", by_year["1"]["cqr_picp"]["mean"], 0.895749)
    close("15-year CQR PICP", by_year["15"]["cqr_picp"]["mean"], 0.900946)

    effect = summary["per_basin_endpoint_effect"]["across_seed_average"]
    close("paired median basin NSE effect", effect["median_delta_nse_15_minus_1"], 0.140083)
    close("fraction basins improved", effect["fraction_improved_with_15_years"], 0.94, 1e-8)
    endpoint = summary["basin_cluster_endpoint_intervals"]["metrics"]
    require(endpoint["raw_picp"]["difference_15_minus_1"]["ci95"][0] > 0, "raw PICP endpoint CI excludes zero")
    require(endpoint["cqr_picp"]["difference_15_minus_1"]["ci95"][0] < 0 < endpoint["cqr_picp"]["difference_15_minus_1"]["ci95"][1], "CQR endpoint CI contains zero")
    require(endpoint["raw_mpiw_log"]["difference_15_minus_1"]["mean"] < 0, "1-year raw intervals are wider than 15-year intervals")

    basin_table = SCARCITY_DIR / "basin_endpoint_effects.csv"
    with basin_table.open(newline="", encoding="utf-8") as stream:
        basin_rows = list(csv.DictReader(stream))
    basin_ids = [row["basin_id"] for row in basin_rows]
    require(len(basin_rows) == 50 and len(set(basin_ids)) == 50, "50 unique basin endpoint rows")
    require(all(len(value) == 8 and value.isdigit() for value in basin_ids), "eight-digit basin IDs retained")
    nse_deltas = [float(row["delta_nse_15_minus_1"]) for row in basin_rows]
    raw_picp_deltas = [float(row["delta_raw_picp_15_minus_1"]) for row in basin_rows]
    close("basin table median NSE effect", median(nse_deltas), 0.140083)
    close("basin table mean raw PICP effect", mean(raw_picp_deltas), 0.045695)
    close("basin table NSE-positive fraction", sum(value > 0 for value in nse_deltas) / 50, 0.94, 1e-8)
    close("basin table raw-PICP-positive fraction", sum(value > 0 for value in raw_picp_deltas) / 50, 0.72, 1e-8)
    require(
        set(basin_ids) == set(summary["per_basin_endpoint_effect"]["by_basin"]),
        "basin table and summary use the same point-effect basins",
    )
    return summary


def verify_readme_results(summary: dict) -> None:
    print("\n== README result tables ==")
    readme = README.read_text(encoding="utf-8")
    readme_zh = README_ZH.read_text(encoding="utf-8")
    valid_targets = {"1": "18,250", "3": "54,750", "5": "91,300", "15": "266,322"}
    for years in ["1", "3", "5", "15"]:
        metrics = summary["by_year"][years]
        values = [
            f'{metrics["median_nse"]["mean"]:.3f} ± {metrics["median_nse"]["sd"]:.3f}',
            f'{metrics["raw_picp"]["mean"]:.3f} ± {metrics["raw_picp"]["sd"]:.3f}',
            f'{metrics["cqr_picp"]["mean"]:.3f} ± {metrics["cqr_picp"]["sd"]:.3f}',
            f'{metrics["raw_mpiw_log"]["mean"]:.3f}',
            f'{metrics["raw_winkler_log"]["mean"]:.3f}',
        ]
        result_cells = " | ".join(values)
        english_label = f"{years} year" if years == "1" else f"{years} years"
        expected_english = (
            f"| {english_label} | {valid_targets[years]} | {result_cells} |"
        )
        expected_chinese = f"| {years} 年 | {valid_targets[years]} | {result_cells} |"
        require(expected_english in readme, f"English README {years}-year result row")
        require(expected_chinese in readme_zh, f"Chinese README {years}-year result row")


def verify_release_metadata() -> None:
    print("\n== Public release metadata ==")
    expected_title = (
        "Data Scarcity in Hydrological Deep Learning: Prediction Skill, "
        "Interval Reliability, and Conformal Calibration"
    )
    concept_doi = "10.5281/zenodo.21256955"
    obsolete_version_doi = "10.5281/zenodo.21256956"
    metadata = read_json(ZENODO_METADATA)
    require(metadata["title"] == expected_title, "Zenodo title matches the manuscript")
    require(metadata["version"] == "2.0.0", "corrected archive is a new major version")
    require(
        [creator["name"] for creator in metadata["creators"]]
        == ["AuroraNeko", "AuroraNeko"],
        "Zenodo creators match the manuscript authors",
    )
    require(metadata["license"] == "mit", "Zenodo uses the MIT license vocabulary ID")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    public_docs = "\n".join(
        [
            citation,
            README.read_text(encoding="utf-8"),
            README_ZH.read_text(encoding="utf-8"),
            (ROOT / "REPRODUCIBILITY.md").read_text(encoding="utf-8"),
        ]
    )
    require(expected_title in citation, "CITATION title matches the manuscript")
    require("license: MIT" in citation, "CITATION uses the MIT SPDX identifier")
    require(concept_doi in public_docs, "public metadata uses the Zenodo concept DOI")
    require(
        obsolete_version_doi not in public_docs,
        "public metadata does not present the superseded v1.0.0 DOI as current",
    )


def verify_processed_data() -> None:
    print("\n== Processed-data audit ==")
    audit = read_json(ROOT / "results" / "tables" / "processed_data_audit.json")
    require(audit["n_files"] == 671, "671 processed basin files")
    require(audit["total_rows"] == 7107903, "expected common-period row count")
    require(audit["first_dates"] == {"1980-01-01": 671}, "common first date")
    require(audit["last_dates"] == {"2008-12-31": 671}, "common last date")
    require(not audit["discontinuous_basins"], "no date discontinuities")
    require(not audit["duplicate_date_basins"], "no duplicate dates")
    require(audit["max_full_missing_fraction"] == 0.0, "no missing dynamic inputs")
    require(audit["calibration_observed_targets"] == 1225916, "full calibration observed targets")
    require(audit["test_observed_targets"] == 2020930, "full test observed targets")
    require(audit["invalid_flow_masks"] == 0, "binary flow masks")
    require(audit["observed_nonfinite_targets"] == 0, "observed targets are finite")
    require(audit["masked_finite_targets"] == 0, "masked targets remain missing")
    static_missing = audit["static_missing_basin_counts"]
    require(
        static_missing["geol_porostiy"] == 3
        and all(value == 0 for key, value in static_missing.items() if key != "geol_porostiy"),
        "only three geologic-porosity values are missing",
    )
    pairs = audit["channel_pair_mismatch_counts"]
    require(pairs["tmax_maurer__tmin_maurer"] == 0, "Maurer temperature pair duplicated upstream")
    require(pairs["tmax_nldas__tmin_nldas"] == 0, "NLDAS temperature pair duplicated upstream")
    require(pairs["tmax_daymet__tmin_daymet"] > 0, "Daymet temperature pair is distinguishable")
    preprocessing = (ROOT / "src" / "data" / "data_preprocessing.py").read_text(
        encoding="utf-8"
    )
    config = (ROOT / "configs" / "data_config.yaml").read_text(encoding="utf-8")
    require(
        'interpolate(method="linear", limit=7)' in preprocessing
        and ".ffill(limit=3)" in preprocessing
        and ".bfill(limit=3)" in preprocessing,
        "implemented forcing-fill limits are 7/3/3",
    )
    require(
        "dynamic_interpolation_limit: 7" in config
        and "dynamic_directional_fill_limit: 3" in config,
        "configuration documents implemented forcing-fill limits",
    )


def verify_representativeness() -> None:
    print("\n== Fixed-basin representativeness diagnostic ==")
    result = read_json(ROOT / "results" / "tables" / "basin_representativeness.json")
    require(result["n_scarcity_basins"] == 50, "50 scarcity basins")
    require(result["n_comparison_basins"] == 621, "disjoint 621-basin comparison group")
    require("disjoint" in result["comparison_design"], "independent comparison design")
    require(len(result["attributes"]) == 13, "13 static-attribute diagnostics")
    adjusted = [row["ks_p_holm"] for row in result["attributes"].values()]
    effects = [abs(row["cohens_d_subset_minus_remaining"]) for row in result["attributes"].values()]
    require(min(adjusted) > 0.05, "no Holm-adjusted KS rejection")
    require(max(effects) < 0.20, "all absolute Cohen's d values below 0.20")


def verify_methods() -> dict:
    print("\n== Corrected full-basin method comparison ==")
    result = read_json(METHOD_DIR / "method_comparison.json")
    protocol = result["protocol"]
    require(protocol["n_basins"] == 671, "671 evaluated basins")
    require(protocol["sequence_length_days"] == 365, "365-day full-basin sequence")
    require(protocol["calibration_targets"] == ["1995-10-01", "2000-09-30"], "full calibration dates")
    require(protocol["test_targets"] == ["2000-10-01", "2008-12-31"], "full test dates")
    require(protocol["calibration_windows"] == 1225917, "complete calibration target-date grid")
    require(protocol["calibration_observed_targets"] == 1225916, "observed calibration targets")
    require(protocol["test_windows"] == 2022394, "complete test target-date grid")
    require(protocol["test_observed_targets"] == 2020930, "observed test targets")
    normalization_path = ROOT / "data" / "metadata" / "normalization_stats.json"
    if normalization_path.exists():
        require(
            sha256(normalization_path) == result["normalization_stats_sha256"],
            "full-method normalization hash",
        )
    expected = {
        "mc_dropout_raw",
        "mc_dropout_cqr",
        "single_qr_raw",
        "single_qr_cqr",
        "ensemble_raw",
        "ensemble_cqr",
    }
    require(set(result["metrics"]) == expected, "all six raw/CQR method variants")

    training_dir = ROOT / "results" / "tables" / "full_checkpoint_training"
    for metadata_name, hash_key, selection in [
        ("point_seed42.json", "point_seed42", "minimum_validation_mse"),
        ("quantile_seed42.json", "quantile_seed42", "minimum_training_pinball_loss"),
    ]:
        metadata = read_json(training_dir / metadata_name)
        require(metadata["training_run_complete"] is True, f"completed training metadata: {metadata_name}")
        require(metadata["model_seed"] == 42, f"training seed metadata: {metadata_name}")
        require(metadata["checkpoint_selection"] == selection, f"checkpoint rule metadata: {metadata_name}")
        require("\\" not in metadata["checkpoint"], f"portable checkpoint path: {metadata_name}")
        require(
            metadata["checkpoint_sha256"] == result["checkpoint_sha256"][hash_key],
            f"method result uses declared checkpoint: {metadata_name}",
        )
        checkpoint = repository_path(metadata["checkpoint"])
        if checkpoint.exists():
            require(sha256(checkpoint) == metadata["checkpoint_sha256"], f"checkpoint SHA-256: {metadata_name}")

    for name, metrics in result["metrics"].items():
        require(
            metrics["n_valid"] == protocol["test_observed_targets"],
            f"masked observed test targets: {name}",
        )
        require(0 <= metrics["picp"] <= 1, f"valid PICP: {name}")
        require(metrics["mpiw"] >= 0, f"nonnegative width: {name}")
        require(metrics["forward_passes_per_sample"] in {1, 5, 50}, f"declared passes: {name}")
        require(name in result["cluster_bootstrap_95ci"], f"cluster CI present: {name}")
    for method in ["mc_dropout", "single_qr", "ensemble"]:
        raw = result["metrics"][f"{method}_raw"]
        calibrated = result["metrics"][f"{method}_cqr"]
        close(f"point NSE unchanged by CQR: {method}", raw["median_basin_nse"], calibrated["median_basin_nse"], 1e-12)
        require(abs(calibrated["picp"] - 0.90) <= abs(raw["picp"] - 0.90), f"CQR does not worsen PICP target error: {method}")

    climate = result["climate_groups"]
    rows = climate["rows"]
    require("Mutually exclusive quartiles" in climate["metadata"]["definition"], "mutually exclusive climate definition")
    require(sum(row["n_basins"] for row in rows[:4]) == 671, "climate quartiles partition 671 basins")
    require(rows[-1]["group"] == "All CAMELS-US" and rows[-1]["n_basins"] == 671, "all-basin climate row matches protocol")
    for split, expected_hash in result["prediction_sha256"].items():
        path = ROOT / "results" / "predictions" / "corrected_full_671" / f"{split}_predictions.npz"
        if path.exists():
            require(sha256(path) == expected_hash, f"prediction cache hash: {split}")
        else:
            print(f"  skip release prediction cache not present: {split}")
    return result


def verify_robustness_experiments() -> None:
    print("\n== Scarcity robustness experiments ==")
    conditional = read_json(SCARCITY_DIR / "conditional_robustness.json")
    require(
        conditional["protocol_version"] == "conditional_robustness_v1_saved_predictions",
        "saved-prediction conditional protocol",
    )
    require(
        conditional["top_flow_fractions"] == [1.0, 0.33, 0.2, 0.1, 0.05, 0.02],
        "six predeclared within-basin high-flow shares",
    )
    close(
        "1-year global CQR top-third PICP",
        conditional["threshold_curve"]["1"]["global_cqr"]["0.33"]["picp"]["value"],
        0.844071,
    )
    close(
        "1-year global CQR top-2% PICP",
        conditional["threshold_curve"]["1"]["global_cqr"]["0.02"]["picp"]["value"],
        0.648571,
    )
    top_two_gain = conditional["predicted_minus_global"]["1"]["0.02"]
    require(top_two_gain["mean_delta_picp_ci95"][0] > 0, "predicted-regime top-2% gain CI excludes zero")
    require(
        conditional["predicted_minus_global"]["1"]["1.00"]["mean_delta_picp_ci95"][0]
        < 0
        < conditional["predicted_minus_global"]["1"]["1.00"]["mean_delta_picp_ci95"][1],
        "predicted-regime overall PICP difference CI includes zero",
    )
    with (SCARCITY_DIR / "basin_calibration_distribution.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        basin_rows = list(csv.DictReader(stream))
    require(len(basin_rows) == 300, "300 per-basin calibration rows")
    groups = {(row["years"], row["method"]) for row in basin_rows}
    require(len(groups) == 6, "two endpoints by three calibration methods")
    require(
        all(
            len({
                row["basin_id"]
                for row in basin_rows
                if (row["years"], row["method"]) == group
            }) == 50
            for group in groups
        ),
        "50 unique-basin rows per calibration group",
    )

    with (REPLICATION_DIR / "replication_subsets.csv").open(newline="", encoding="utf-8") as stream:
        subset_rows = list(csv.DictReader(stream))
    subset_ids = ["primary_seed42", "stratified_1", "stratified_2", "stratified_3", "stratified_4"]
    require(
        all(sum(row["subset_id"] == subset_id for row in subset_rows) == 50 for subset_id in subset_ids),
        "five 50-basin replication subsets",
    )
    all_subset_basins = [row["basin_id"] for row in subset_rows if row["subset_id"] in subset_ids]
    require(len(all_subset_basins) == len(set(all_subset_basins)) == 250, "250 disjoint replication basins")

    new_runs = list(REPLICATION_DIR.glob("scarcity_*yr_subsetstratified_*_modelseed*.json"))
    require(len(new_runs) == 24, "24 new replicated-subset endpoint fits")
    for path in new_runs:
        result = read_json(path)
        require(
            result["protocol_version"] == "scarcity_v3_replicated_subsets_fixed_epochs",
            f"replication protocol: {path.name}",
        )
        require(
            result["training_budget"] == "fixed_epochs" and result["epochs"] == 30,
            f"fixed-epoch replication budget: {path.name}",
        )
        require(
            result.get("checkpoint_metadata_sanitized") is True
            and len(result.get("checkpoint_tensor_sha256", "")) == 64,
            f"release-safe checkpoint metadata: {path.name}",
        )

    replication = read_json(REPLICATION_DIR / "scarcity_replication_summary.json")
    require(replication["design"]["n_unique_basins"] == 250, "replication summary uses 250 basins")
    require(replication["design"]["new_model_fits"] == 24, "replication summary declares 24 new fits")
    with (REPLICATION_DIR / "scarcity_replication_basin_effects.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        effect_rows = list(csv.DictReader(stream))
    require(
        len(effect_rows) == 250
        and len({(row["subset_id"], row["basin_id"]) for row in effect_rows}) == 250,
        "250 unique replicated basin-effect rows",
    )
    nse_values = [float(row["delta_nse_15_minus_1"]) for row in effect_rows]
    picp_values = [float(row["delta_raw_picp_15_minus_1"]) for row in effect_rows]
    close(
        "replication CSV median NSE effect",
        median(nse_values),
        replication["combined_effects_15_minus_1"]["nse"]["estimate"],
        1e-12,
    )
    close(
        "replication CSV mean raw PICP effect",
        mean(picp_values),
        replication["combined_effects_15_minus_1"]["raw_picp"]["estimate"],
        1e-12,
    )

    equal_runs = list(EQUAL_UPDATE_DIR.glob("scarcity_*yr_subsetprimary_seed42_modelseed*.json"))
    require(len(equal_runs) == 6, "six equal-update endpoint fits")
    for path in equal_runs:
        result = read_json(path)
        require(result["protocol_version"] == "scarcity_v3_equal_optimizer_updates", f"equal-update protocol: {path.name}")
        require(result["optimizer_steps"] == 1080, f"1,080 optimizer steps: {path.name}")
        require(
            result.get("checkpoint_metadata_sanitized") is True
            and len(result.get("checkpoint_tensor_sha256", "")) == 64,
            f"release-safe checkpoint metadata: {path.name}",
        )
    equal = read_json(EQUAL_UPDATE_DIR / "equal_update_summary.json")
    require(equal["design"]["fixed_update_budget"] == 1080, "equal-update summary budget")
    close(
        "equal-update 15-year mean NSE",
        equal["training_budgets"]["fixed_updates"]["by_year"]["15"]["median_nse"]["mean"],
        0.582984,
    )
    require(
        equal["training_budgets"]["fixed_updates"]["effects_15_minus_1"]["nse"]["estimate"] > 0,
        "equal-update point-skill endpoint effect remains positive",
    )

    architecture_runs = list(
        ARCHITECTURE_DIR.glob("scarcity_*yr_subsetprimary_seed42_modelseed*.json")
    )
    require(len(architecture_runs) == 6, "six standard-LSTM endpoint fits")
    for path in architecture_runs:
        result = read_json(path)
        require(
            result["protocol_version"] == "scarcity_v3_standard_lstm_endpoint_sensitivity",
            f"standard-LSTM protocol: {path.name}",
        )
        require(result["architecture"] == "standard_lstm", f"standard-LSTM identity: {path.name}")
        require(
            result["training_budget"] == "fixed_epochs" and result["epochs"] == 30,
            f"standard-LSTM fixed-epoch budget: {path.name}",
        )
        require(
            result.get("checkpoint_metadata_sanitized") is True
            and len(result.get("checkpoint_tensor_sha256", "")) == 64,
            f"release-safe checkpoint metadata: {path.name}",
        )
    architecture = read_json(ARCHITECTURE_DIR / "architecture_sensitivity_summary.json")
    require(
        architecture["architectures"]["standard_lstm"]["effects_15_minus_1"]["nse"]["estimate"] > 0,
        "standard-LSTM point-skill endpoint effect remains positive",
    )


def verify_generated_and_text() -> None:
    print("\n== Generated LaTeX and manuscript text ==")
    if not all(path.exists() for path in [MANUSCRIPT, SUPPLEMENT, REFERENCES]):
        print("  skip paper-source checks (paper directory is excluded from the public code release)")
        return
    required_generated = [
        "result_macros.tex",
        "scarcity_rows.tex",
        "conditional_rows.tex",
        "conditional_rows_zh.tex",
        "method_rows.tex",
        "method_rows_zh.tex",
        "climate_rows.tex",
        "climate_rows_zh.tex",
        "representativeness_rows.tex",
        "sensitivity_rows.tex",
        "replication_rows.tex",
        "equal_update_rows.tex",
        "architecture_rows.tex",
    ]
    for name in required_generated:
        path = GENERATED_DIR / name
        require(path.exists() and path.stat().st_size > 0, f"generated fragment: {name}")

    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    manuscript_zh_path = PAPER_DIR / "manuscript_zh.tex"
    manuscript_zh = manuscript_zh_path.read_text(encoding="utf-8")
    supplement = SUPPLEMENT.read_text(encoding="utf-8")
    references = REFERENCES.read_text(encoding="utf-8")
    for source_name, source in [
        ("English manuscript", manuscript),
        ("Chinese manuscript", manuscript_zh),
        ("supplement", supplement),
    ]:
        labels = re.findall(r"\\label\{([^}]+)\}", source)
        references_used = re.findall(r"\\(?:ref|pageref)\{([^}]+)\}", source)
        duplicate_labels = sorted({label for label in labels if labels.count(label) > 1})
        require(not duplicate_labels, f"{source_name} has no duplicate labels: {duplicate_labels}")
        require(
            not (set(references_used) - set(labels)),
            f"{source_name} cross-references resolve: {sorted(set(references_used) - set(labels))}",
        )
        stack: list[str] = []
        environment_error = ""
        for action, environment in re.findall(r"\\(begin|end)\{([^}]+)\}", source):
            if action == "begin":
                stack.append(environment)
            elif not stack or stack.pop() != environment:
                environment_error = f"mismatched end for {environment}"
                break
        if not environment_error and stack:
            environment_error = f"unclosed environments: {stack}"
        require(not environment_error, f"{source_name} environments are balanced: {environment_error}")
    for required in [
        "Short-record intervals widen",
        "paired whole-basin",
        "predicted-regime CQR",
        "time-series archive v1.2",
        "13 distinct daily series",
        "31 December 2008",
        "figures/fig_protocol_timeline.pdf",
        "Supplement Sect.~S4 and Fig.~S1",
        "Supplement Sects.~S10--S12",
        "five disjoint 50-basin subsets",
        "same 1080 optimizer updates",
        "standard 128-unit LSTM",
        "highest 2\\%",
        "$\\log(1+\\mathrm{drainage\\ area})$",
        "training-target endpoint",
        "linear interpolation uses a seven-value limit",
        "averaged over $\\tau",
        "same 1995--2000 period subsequently used for CQR calibration",
        "rather than a clean split-conformal benchmark",
        "OpenAI Codex was used",
        "https://github.com/AuroraNeko/data-scarcity-uncertainty-hydrodl",
        "https://doi.org/10.5281/zenodo.21256955",
    ]:
        require(required in manuscript, f"manuscript statement: {required}")
    require(
        "$\\log(1+\\mathrm{drainage\\ area})$" in supplement,
        "supplement states the exact log1p drainage-area stratifier",
    )
    require(
        "$\\log(1+\\text{流域面积})$" in manuscript_zh,
        "Chinese manuscript states the exact log1p drainage-area stratifier",
    )
    require("训练目标终点" in manuscript_zh, "Chinese manuscript states the shared training endpoint")
    require("严格独立的 split-conformal 基准" in manuscript_zh, "Chinese manuscript discloses calibration reuse")
    require(
        "Masked pinball losses are averaged" in supplement,
        "supplement states the implemented multi-quantile aggregation",
    )
    require(
        "\\href{https://github.com/AuroraNeko/"
        "data-scarcity-uncertainty-hydrodl}{public GitHub repository}"
        in manuscript,
        "review manuscript links the public repository without layout overflow",
    )
    for forbidden in [
        "0.884",
        "0.719",
        "0.671",
        "systematic interval narrowing",
        "30, 90, 180, and 365",
        "30 September 2010",
        "PICP = 0.504",
        "same 50 basins, target dates",
        "summed over $\\tau",
        "three edge days",
        "10.5281/zenodo.21256956",
    ]:
        require(forbidden not in manuscript + supplement, f"obsolete text absent: {forbidden}")

    citation_keys = set()
    for command in re.findall(r"\\cite[tp]?\{([^}]+)\}", manuscript):
        citation_keys.update(key.strip() for key in command.split(","))
    citation_keys_zh = set()
    for command in re.findall(r"\\cite[tp]?\{([^}]+)\}", manuscript_zh):
        citation_keys_zh.update(key.strip() for key in command.split(","))
    bib_keys = set(re.findall(r"\\bibitem(?:\[[^]]*\])?\{([^}]+)\}", references))
    require(not (citation_keys - bib_keys), f"all citation keys resolved: {sorted(citation_keys - bib_keys)}")
    require(not (bib_keys - citation_keys), f"no unused bibliography entries: {sorted(bib_keys - citation_keys)}")
    require(citation_keys_zh == citation_keys, "English and Chinese citation sets match")
    dois = re.findall(r"https://doi\.org/([^}]+)", references, flags=re.IGNORECASE)
    require(len(dois) == len(set(value.lower() for value in dois)), "no duplicate bibliography DOIs")

    for name in [
        "architecture_tikz.pdf",
        "fig_protocol_timeline.pdf",
        "fig1_scarcity_effects.pdf",
        "fig2_conditional_coverage.pdf",
        "fig3_method_comparison.pdf",
        "figS1_basin_endpoint_effects.pdf",
        "figS2_climate_quartiles.pdf",
        "figS3_high_flow_thresholds.pdf",
        "figS4_basin_calibration.pdf",
        "figS5_subset_replication.pdf",
        "figS6_equal_updates.pdf",
        "figS7_architecture_sensitivity.pdf",
    ]:
        path = PAPER_DIR / "figures" / name
        require(path.exists() and path.stat().st_size > 1000, f"figure exists: {name}")

    for directory in [ROOT / "results" / "figures" / "corrected", PAPER_DIR / "figures"]:
        for suffix in [".pdf", ".png"]:
            stale = directory / f"figS1_climate_quartiles{suffix}"
            require(not stale.exists(), f"stale figure alias absent: {stale.relative_to(ROOT)}")

    submission_main = SUBMISSION_DIR / "manuscript.tex"
    submission_supplement = SUBMISSION_DIR / "supplement.tex"
    require(submission_main.exists(), "flattened HESS manuscript source exists")
    require(submission_supplement.exists(), "flattened supplement source exists")
    if submission_main.exists():
        flattened = submission_main.read_text(encoding="utf-8")
        for forbidden in ["\\newcommand", "\\input{", "\\inputtablerows", "\\usepackage", "[nosep]"]:
            require(forbidden not in flattened, f"submission source excludes {forbidden}")
        require("\\begin{thebibliography}" in flattened, "submission source inlines references")
        require("figures/" not in flattened, "submission source uses flat numbered figure paths")
    for index in range(1, 6):
        path = SUBMISSION_DIR / f"f{index:02d}.pdf"
        require(path.exists() and path.stat().st_size > 1000, f"numbered submission figure: {path.name}")
    short_summary = SUBMISSION_DIR / "short_summary.txt"
    require(short_summary.exists(), "HESS short-summary text exists")
    if short_summary.exists():
        require(
            len(short_summary.read_text(encoding="utf-8").strip()) <= 500,
            "HESS short summary is at most 500 characters",
        )


def main() -> None:
    verify_processed_data()
    scarcity_summary = verify_scarcity()
    verify_readme_results(scarcity_summary)
    verify_release_metadata()
    verify_representativeness()
    verify_methods()
    verify_robustness_experiments()
    verify_generated_and_text()
    print("\n" + "=" * 64)
    if failures:
        print(f"{len(failures)} verification failure(s):")
        for message in failures:
            print(f"  FAIL {message}")
        raise SystemExit(1)
    print("All corrected manuscript checks passed.")


if __name__ == "__main__":
    main()
