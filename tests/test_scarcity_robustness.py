import csv
import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from experiments.train_primary import run_endpoint_replications as robustness_runner
from experiments.analyze.conditional_robustness import top_flow_mask
from experiments.prepare.audit_checkpoint_metadata import (
    contains_absolute_path,
    tensor_state_sha256 as sanitizer_tensor_sha256,
)
from experiments.train_primary.train_scarcity_corrected import (
    tensor_state_sha256 as training_tensor_sha256,
)
from src.models.lstm import LSTMModel


ROOT = Path(__file__).resolve().parents[1]


class ScarcityRobustnessArtifactTests(unittest.TestCase):
    def test_checkpoint_path_audit_rejects_absolute_paths(self):
        self.assertTrue(contains_absolute_path({"args": {"file": r"D:\\private\\data.csv"}}))
        self.assertTrue(contains_absolute_path({"args": {"file": "/home/user/data.csv"}}))
        self.assertTrue(contains_absolute_path({"args": {"file": r"\\\\server\\share\\data.csv"}}))
        self.assertFalse(
            contains_absolute_path(
                {"args": {"file": "results/tables/scarcity_replications/replication_subsets.csv"}}
            )
        )

    def test_runner_keeps_budget_and_architecture_stages_separate(self):
        with (
            mock.patch.object(robustness_runner, "result_complete", return_value=False),
            mock.patch.object(Path, "exists", return_value=False),
            mock.patch.object(robustness_runner, "run_logged") as run_logged,
        ):
            robustness_runner.run_equal_updates(force=False)

        self.assertEqual(run_logged.call_count, 6)
        for call in run_logged.call_args_list:
            command = call.args[0]
            self.assertEqual(command[command.index("--output-group") + 1], "scarcity_equal_updates")
            self.assertEqual(command[command.index("--training-budget") + 1], "fixed_updates")
            self.assertEqual(command[command.index("--architecture") + 1], "lpu_stream")

        with (
            mock.patch.object(robustness_runner, "result_complete", return_value=False),
            mock.patch.object(Path, "exists", return_value=False),
            mock.patch.object(robustness_runner, "run_logged") as run_logged,
        ):
            robustness_runner.run_architecture_sensitivity(force=False)

        self.assertEqual(run_logged.call_count, 6)
        for call in run_logged.call_args_list:
            command = call.args[0]
            self.assertEqual(command[command.index("--output-group") + 1], "scarcity_standard_lstm")
            self.assertEqual(command[command.index("--training-budget") + 1], "fixed_epochs")
            self.assertEqual(command[command.index("--architecture") + 1], "standard_lstm")

    def test_top_flow_mask_selects_exact_share_per_basin(self):
        target = np.asarray([1.0, 2.0, 3.0, 4.0, 8.0, 8.0, 9.0, 10.0])
        basins = np.asarray(["a"] * 4 + ["b"] * 4)
        selected = top_flow_mask(target, basins, 0.25)
        self.assertEqual(int(selected[:4].sum()), 1)
        self.assertEqual(int(selected[4:].sum()), 1)
        self.assertTrue(selected[3])
        self.assertTrue(selected[7])

    def test_checkpoint_tensor_digest_is_stable_across_writer_and_auditor(self):
        model = LSTMModel(
            quantiles=[0.05, 0.50, 0.95],
            enforce_monotonic=True,
        )
        self.assertEqual(
            training_tensor_sha256(model.state_dict()),
            sanitizer_tensor_sha256(model.state_dict()),
        )

    @unittest.skipUnless(
        (ROOT / "results" / "tables" / "scarcity_replications" / "replication_subsets.csv").exists(),
        "recorded artifacts not present in a fresh clone",
    )
    def test_replication_subsets_are_disjoint_and_match_primary_run(self):
        subset_path = (
            ROOT
            / "results"
            / "tables"
            / "scarcity_replications"
            / "replication_subsets.csv"
        )
        with subset_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        subset_ids = [
            "primary_seed42",
            "stratified_1",
            "stratified_2",
            "stratified_3",
            "stratified_4",
        ]
        by_subset = {
            subset_id: [row for row in rows if row["subset_id"] == subset_id]
            for subset_id in subset_ids
        }
        self.assertTrue(all(len(selected) == 50 for selected in by_subset.values()))
        all_basins = [row["basin_id"] for selected in by_subset.values() for row in selected]
        self.assertEqual(len(set(all_basins)), 250)

        primary = sorted(
            by_subset["primary_seed42"], key=lambda row: int(row["selection_order"])
        )
        result = json.loads(
            (
                ROOT
                / "results"
                / "tables"
                / "corrected_scarcity"
                / "scarcity_1yr_basinseed42_modelseed42.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual([row["basin_id"] for row in primary], result["selected_basins"])

    @unittest.skipUnless(
        (ROOT / "results" / "tables" / "scarcity_equal_updates").is_dir(),
        "recorded artifacts not present in a fresh clone",
    )
    def test_equal_update_runs_have_identical_step_budgets(self):
        directory = ROOT / "results" / "tables" / "scarcity_equal_updates"
        results = []
        for years in [1, 15]:
            for seed in [42, 123, 456]:
                path = directory / f"scarcity_{years}yr_subsetprimary_seed42_modelseed{seed}.json"
                result = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(result["protocol_version"], "scarcity_v3_equal_optimizer_updates")
                self.assertEqual(result["training_budget"], "fixed_updates")
                self.assertEqual(result["optimizer_steps"], 1080)
                self.assertEqual(result["steps_per_round"], 36)
                results.append(result)
        basin_sets = {tuple(result["selected_basins"]) for result in results}
        self.assertEqual(len(basin_sets), 1)

    @unittest.skipUnless(
        (ROOT / "results" / "tables" / "corrected_scarcity" / "conditional_robustness.json").exists(),
        "recorded artifacts not present in a fresh clone",
    )
    def test_conditional_robustness_outputs_expected_units_and_counts(self):
        directory = ROOT / "results" / "tables" / "corrected_scarcity"
        result = json.loads(
            (directory / "conditional_robustness.json").read_text(encoding="utf-8")
        )
        self.assertEqual(result["top_flow_fractions"], [1.0, 0.33, 0.2, 0.1, 0.05, 0.02])
        top_third = result["threshold_curve"]["1"]["global_cqr"]["0.33"]["picp"]
        top_two = result["threshold_curve"]["1"]["global_cqr"]["0.02"]["picp"]
        predicted_top_two = result["threshold_curve"]["1"]["predicted_regime_cqr"]["0.02"]["picp"]
        self.assertAlmostEqual(top_third["value"], 0.8440705, places=6)
        self.assertAlmostEqual(top_two["value"], 0.6485711, places=6)
        self.assertAlmostEqual(predicted_top_two["value"], 0.6914929, places=6)

        with (directory / "basin_calibration_distribution.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 300)
        combinations = {(row["years"], row["method"]) for row in rows}
        self.assertEqual(len(combinations), 6)
        for combination in combinations:
            selected = [row for row in rows if (row["years"], row["method"]) == combination]
            self.assertEqual(len(selected), 50)


if __name__ == "__main__":
    unittest.main()
