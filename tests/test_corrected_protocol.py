import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import src.data.dataset as dataset_module
from src.data.dataset import CamelsDataset, DYNAMIC_VARS, STATIC_COLS, compute_normalization_stats
from src.losses.cqr import finite_sample_quantile
from src.losses.pinball_loss import PinballLoss
from src.models.lpu_stream import LPUStreamModel
from src.models.lstm import LSTMModel
from experiments.evaluate.evaluate_corrected_methods import (
    destandardize_log_scale,
    protocol_identifier,
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
from experiments.analyze.summarize_scarcity import (
    repository_path as scarcity_repository_path,
)
from experiments.analyze.summarize_replications import (
    repository_path as robustness_repository_path,
)


class CorrectedProtocolTests(unittest.TestCase):
    def test_finite_sample_quantile_is_order_statistic(self):
        scores = np.arange(1.0, 11.0)
        self.assertEqual(finite_sample_quantile(scores, alpha=0.1), 10.0)
        self.assertEqual(finite_sample_quantile(scores, alpha=0.5), 6.0)

    def test_multi_quantile_pinball_loss_is_averaged(self):
        criterion = PinballLoss([0.05, 0.50, 0.95])
        predictions = torch.zeros((2, 3), dtype=torch.float32)
        targets = torch.ones((2, 1), dtype=torch.float32)
        masks = torch.ones((2, 1), dtype=torch.float32)
        self.assertAlmostEqual(
            criterion(predictions, targets, masks).item(),
            (0.05 + 0.50 + 0.95) / 3,
        )

    def test_monotonic_quantile_parameterization(self):
        model = LPUStreamModel(
            n_dynamic=15,
            n_static=13,
            quantiles=[0.05, 0.5, 0.95],
            enforce_monotonic=True,
        )
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(8, 12, 15), torch.randn(8, 13))
        self.assertTrue(torch.all(output[:, 0] <= output[:, 1]))
        self.assertTrue(torch.all(output[:, 1] <= output[:, 2]))

    def test_monotonic_quantile_parameterization_rejects_bad_order(self):
        with self.assertRaises(ValueError):
            LPUStreamModel(
                n_dynamic=15,
                n_static=13,
                quantiles=[0.95, 0.5, 0.05],
                enforce_monotonic=True,
            )

    def test_standard_lstm_quantile_parameterization(self):
        model = LSTMModel(
            n_dynamic=15,
            n_static=13,
            quantiles=[0.05, 0.50, 0.95],
            enforce_monotonic=True,
        )
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(8, 12, 15), torch.randn(8, 13))
        self.assertEqual(tuple(output.shape), (8, 3))
        self.assertTrue(torch.all(output[:, 0] <= output[:, 1]))
        self.assertTrue(torch.all(output[:, 1] <= output[:, 2]))

    def test_standard_lstm_point_interface_remains_backward_compatible(self):
        model = LSTMModel(n_dynamic=15, n_static=13)
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(8, 12, 15), torch.randn(8, 13))
        self.assertEqual(tuple(output.shape), (8, 1))

    def test_context_and_target_dates_are_separate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            original = dataset_module.PROCESSED_DIR
            dataset_module.PROCESSED_DIR = Path(temp_dir)
            try:
                dates = pd.date_range("2000-01-01", periods=10, freq="D")
                frame = pd.DataFrame(index=dates)
                frame.index.name = "date"
                for index, name in enumerate(DYNAMIC_VARS):
                    frame[name] = np.arange(10, dtype=float) + index
                frame["target"] = np.arange(10, dtype=float)
                frame["flow_mask"] = 1
                for index, name in enumerate(STATIC_COLS):
                    frame[name] = float(index + 1)
                frame.to_csv(Path(temp_dir) / "00000001.csv")

                stats = compute_normalization_stats(
                    ["00000001"], "2000-01-05", "2000-01-08"
                )
                self.assertEqual(stats["target_count"], 4)
                self.assertAlmostEqual(stats["target_mean"], 5.5)

                data = CamelsDataset(
                    ["00000001"],
                    "2000-01-01",
                    "2000-01-10",
                    seq_len=3,
                    normalization_stats=stats,
                    target_start_date="2000-01-05",
                    target_end_date="2000-01-08",
                )
                self.assertEqual(len(data), 4)
                basin_ids, target_dates = data.sample_metadata()
                self.assertEqual(set(basin_ids), {"00000001"})
                np.testing.assert_array_equal(
                    target_dates,
                    np.arange(
                        np.datetime64("2000-01-05"),
                        np.datetime64("2000-01-09"),
                    ),
                )
                self.assertEqual(data.valid_target_count(), 4)
            finally:
                dataset_module.PROCESSED_DIR = original

    def test_full_method_evaluation_destandardizes_log_flow(self):
        arrays = {
            "target": np.asarray([0.0, 1.0], dtype=np.float32),
            "mask": np.asarray([1, 1], dtype=np.uint8),
            "basin_index": np.asarray([0, 0], dtype=np.int16),
            "single_qr": np.asarray([[0.0, 0.5, 1.0], [1.0, 1.5, 2.0]], dtype=np.float32),
            "ensemble": np.asarray([[0.1, 0.6, 1.1], [1.1, 1.6, 2.1]], dtype=np.float32),
            "mc_dropout": np.asarray([[-0.1, 0.4, 0.9], [0.9, 1.4, 1.9]], dtype=np.float32),
        }
        converted = destandardize_log_scale(
            arrays, {"target_mean": 2.0, "target_std": 3.0}
        )
        np.testing.assert_allclose(converted["target"], [2.0, 5.0])
        np.testing.assert_allclose(converted["single_qr"][0], [2.0, 3.5, 5.0])
        np.testing.assert_array_equal(converted["mask"], arrays["mask"])
        np.testing.assert_array_equal(converted["basin_index"], arrays["basin_index"])

    def test_full_method_cache_identity_includes_normalization(self):
        checkpoints = {"point": "abc", "quantile": "def"}
        first = protocol_identifier(50, 1024, checkpoints, "stats-a")
        second = protocol_identifier(50, 1024, checkpoints, "stats-b")
        self.assertNotEqual(first, second)

    def test_repository_artifact_paths_are_cross_platform(self):
        expected = REPOSITORY_ROOT / "results" / "tables" / "result.json"
        resolvers = (
            scarcity_repository_path,
            robustness_repository_path,
        )
        for resolver in resolvers:
            with self.subTest(resolver=resolver.__module__):
                self.assertEqual(resolver("results/tables/result.json"), expected)
                self.assertEqual(resolver(r"results\tables\result.json"), expected)
                for invalid in ("../outside.json", "/outside.json", r"C:\outside.json"):
                    with self.assertRaises(ValueError):
                        resolver(invalid)


if __name__ == "__main__":
    unittest.main()
