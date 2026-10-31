"""Repository-wide structural contracts from the 2026-10 full audit.

These checks need no dataset and no GPU: they keep the one-way import rule
(run_pipeline -> experiments -> src), the bare-invocation CLI contract for
every stage script, byte-compilation, the sensitivity spec table, run_pipeline
--only resolution, and the shared path/hash helpers honest.
"""
import ast
import hashlib
import importlib.util
import os
import py_compile
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("run_pipeline", ROOT / "run_pipeline.py")
run_pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_pipeline)

PY_ROOTS = [ROOT / "experiments", ROOT / "src", ROOT / "tests"]
PY_FILES = sorted(
    [p for root in PY_ROOTS for p in root.rglob("*.py") if "__pycache__" not in p.parts]
    + [ROOT / "run_pipeline.py", ROOT / "download_camels.py"]
)


class CompilationTests(unittest.TestCase):
    def test_every_script_byte_compiles(self):
        for path in PY_FILES:
            with self.subTest(path=path.relative_to(ROOT)):
                py_compile.compile(str(path), doraise=True)


class BareInvocationTests(unittest.TestCase):
    """Every script listed in STAGES must accept being run with no arguments."""

    @staticmethod
    def _required_cli_arguments(path: Path) -> list[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        offenders = []

        def visit_call(node: ast.Call) -> None:
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
                return
            keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}
            required = keywords.get("required")
            if isinstance(required, ast.Constant) and required.value is True:
                offenders.append(ast.unparse(node))
                return
            if not node.args:
                return
            first = node.args[0]
            name = first.value if isinstance(first, ast.Constant) else None
            if isinstance(name, str) and not name.startswith("-"):
                nargs = keywords.get("nargs")
                nargs_value = nargs.value if isinstance(nargs, ast.Constant) else None
                if nargs_value not in ("?", "*"):
                    offenders.append(ast.unparse(node))

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                visit_call(node)
        return offenders

    def test_stage_scripts_have_no_required_cli_arguments(self):
        offenders = {}
        for _number, (_title, scripts) in run_pipeline.STAGES.items():
            for script in scripts:
                found = self._required_cli_arguments(ROOT / script)
                if found:
                    offenders[script] = found
        self.assertEqual(offenders, {})


class ImportGraphTests(unittest.TestCase):
    """experiments/ scripts never import each other; src/ never imports back."""

    @staticmethod
    def _imported_names(path: Path) -> set[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names.add(node.module)
        return names

    def test_experiments_never_import_each_other(self):
        violations = []
        for path in sorted((ROOT / "experiments").rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            sibling_stems = {p.stem for p in path.parent.glob("*.py")} - {path.stem}
            for name in self._imported_names(path):
                top = name.split(".")[0]
                if top == "experiments" or top in sibling_stems:
                    violations.append(f"{path.relative_to(ROOT)}: {name}")
        self.assertEqual(violations, [])

    def test_src_never_imports_experiments(self):
        violations = []
        for path in sorted((ROOT / "src").rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            for name in self._imported_names(path):
                if name.split(".")[0] == "experiments":
                    violations.append(f"{path.relative_to(ROOT)}: {name}")
        self.assertEqual(violations, [])


class PipelineResolutionTests(unittest.TestCase):
    def test_only_resolves_unique_basenames(self):
        self.assertEqual(
            run_pipeline.match_by_basename("memberwise_splitcqr.py"),
            "experiments/evaluate/memberwise_splitcqr.py",
        )
        self.assertIsNone(run_pipeline.match_by_basename("does_not_exist.py"))

    def test_only_rejects_ambiguous_basenames(self):
        original = run_pipeline.STAGES
        try:
            run_pipeline.STAGES = {
                1: ("a", ["experiments/evaluate/coverage_groups.py"]),
                2: ("b", ["experiments/analyze/coverage_groups.py"]),
            }
            with self.assertRaises(SystemExit):
                run_pipeline.match_by_basename("coverage_groups.py")
        finally:
            run_pipeline.STAGES = original


class SensitivitySpecTests(unittest.TestCase):
    def test_48_unique_ids_every_one_queue_positioned(self):
        from src.sensitivity import queue_position, specs

        rows = specs()
        ids = [row["id"] for row in rows]
        self.assertEqual(len(rows), 48)
        self.assertEqual(len(set(ids)), 48)
        for row in rows:
            self.assertIsInstance(queue_position(row), int)
        with self.assertRaises(ValueError):
            queue_position({"id": "undeclared_prefix_s42"})


class ContinueQueueTests(unittest.TestCase):
    def test_newest_activity_watches_status_resume_and_log(self):
        module = importlib.import_module("experiments.train_sensitivity.continue_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "results/checkpoints/sensitivity").mkdir(parents=True)
            (root / "results/logs").mkdir(parents=True)
            status = root / "results/training_queue_status.json"
            resume = root / "results/checkpoints/sensitivity/tag_resume.pt"
            log = root / "results/logs/tag.log"
            status.write_text("{}", encoding="utf-8")
            resume.write_bytes(b"x")
            log.write_bytes(b"x")
            stamp = {"status": 100.0, "resume": 200.0, "log": 300.0}
            for path, key in ((status, "status"), (resume, "resume"), (log, "log")):
                os.utime(path, (stamp[key], stamp[key]))

            saved = (module.status_path, module.ROOT)
            try:
                module.status_path = status
                module.ROOT = root
                self.assertEqual(module.newest_activity({"running": "tag"}), 300.0)
                self.assertEqual(module.newest_activity({}), 100.0)
                self.assertEqual(module.newest_activity({"running": "gone"}), 100.0)
            finally:
                module.status_path, module.ROOT = saved


class UtilTests(unittest.TestCase):
    def test_sha256_file_matches_hashlib(self):
        from src.utils import sha256_file

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.bin"
            payload = b"lpu-stream\n" * 1000
            path.write_bytes(payload)
            self.assertEqual(sha256_file(path), hashlib.sha256(payload).hexdigest())

    def test_repository_path_vectors(self):
        from src.utils import repository_path

        root = Path("D:/repo").resolve() if os.name == "nt" else Path("/repo")
        self.assertEqual(repository_path(root, "results/tables"), root / "results/tables")
        self.assertEqual(repository_path(root, "results\\tables"), root / "results/tables")
        for bad in ("", ".", "..", "a/../b", "results//tables", "results/./tables", "D:/elsewhere", "/absolute"):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    repository_path(root, bad)


if __name__ == "__main__":
    unittest.main()
