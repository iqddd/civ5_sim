import json
import tempfile
import unittest
from pathlib import Path

from civ5rl.engine.resources import analyze_telemetry
from civ5rl.experiments.evaluation import paired_world_bootstrap
from civ5rl.provenance.lineage import verify_parent_lineage


class ExperimentRefactorTests(unittest.TestCase):
    def test_time_weighted_windows_and_phase_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resources.jsonl"
            path.with_name("resources-schema.json").write_text(
                json.dumps({"interval_seconds": 2})
            )
            rows = [
                {
                    "elapsed_seconds": t,
                    "phase": "iterative_training",
                    "gpu_core_load_percent": 50 if t < 30 else 90,
                }
                for t in range(0, 62, 2)
            ]
            path.write_text("\n".join(map(json.dumps, rows)))
            result = analyze_telemetry(path)
            self.assertEqual(len(result["windows"]), 1)
            self.assertEqual(result["windows"][0]["core_average"], 70)
            rows[15]["phase"] = "compilation"
            path.write_text("\n".join(map(json.dumps, rows)))
            self.assertEqual(analyze_telemetry(path)["windows"], [])

    def test_bootstrap_requires_pairs_and_records_arguments(self):
        before = {(i, j, j % 2): 0 for i in range(16) for j in range(4)}
        after = {key: 1 for key in before}
        result = paired_world_bootstrap(before, after, samples=29, seed=77)
        self.assertEqual((result["delta"], result["CI95"]), (1, [1, 1]))
        self.assertEqual(
            (result["bootstrap_samples"], result["bootstrap_seed"]), (29, 77)
        )
        after.pop(next(iter(after)))
        with self.assertRaises(ValueError):
            paired_world_bootstrap(before, after, samples=29, seed=77)

    def test_lineage_missing_environment_field_fails_closed(self):
        import zipfile

        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            with zipfile.ZipFile(parent / "sources.zip", "w"):
                pass
            with self.assertRaises(ValueError):
                verify_parent_lineage(
                    parent,
                    {"source_hashes": {}, "gpu": None},
                    {"source_hashes": {}},
                    set(),
                )


if __name__ == "__main__":
    unittest.main()
