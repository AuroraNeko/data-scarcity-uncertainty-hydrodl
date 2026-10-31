"""Guard: every script referenced by reproduce.py exists and the stage table
stays contiguous and ordered."""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reproduce", ROOT / "reproduce.py")
reproduce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproduce)


class ReproduceStageTests(unittest.TestCase):
    def test_stage_numbers_are_contiguous_from_one(self):
        self.assertEqual(sorted(reproduce.STAGES), list(range(1, len(reproduce.STAGES) + 1)))

    def test_every_referenced_script_exists(self):
        missing = []
        for number, (_title, scripts) in reproduce.STAGES.items():
            for s in scripts:
                if not (ROOT / s).is_file():
                    missing.append((number, s))
        self.assertEqual(missing, [])

    def test_every_script_lives_under_experiments(self):
        for _number, (_title, scripts) in reproduce.STAGES.items():
            for s in scripts:
                self.assertTrue(s.startswith("experiments/"), s)


if __name__ == "__main__":
    unittest.main()
