import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from civ5rl import checkpoint, migration


class ProvenanceRefactorTests(unittest.TestCase):
    def test_discovery_and_passport_archive_match(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            contents = {
                "civ5rl/nested/feature.py": b"untracked nested code",
                "configs/rl-new-experiment.json": b"{}",
                "docs/rl-new-experiment.md": b"experiment",
                "experiments/demo/spec.yaml": b"extra",
                "configs/rl-migration-policies.json": b"policy",
            }
            for name, data in contents.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            config = {"passport_artifacts": ["experiments/**/*.yaml", "civ5rl/**/*.py"]}
            out = root / "output"
            with (
                patch.object(checkpoint, "ROOT", root),
                patch.object(
                    checkpoint.subprocess, "check_output", side_effect=["commit\n", b""]
                ),
            ):
                result = checkpoint.passport(out, config, {"manifest": 1})
                with zipfile.ZipFile(out / "sources.zip") as archive:
                    names = archive.namelist()
                    self.assertEqual(len(names), len(set(names)))
                    self.assertEqual(set(names), set(contents))
                    for name in names:
                        self.assertEqual(
                            result["source_hashes"][name],
                            hashlib.sha256(archive.read(name)).hexdigest(),
                        )
                with self.assertRaises(ValueError):
                    checkpoint.artifact_sources({"passport_artifacts": ["../*.py"]})

    def test_fixed_migration_policies_reject_model_and_rules(self):
        identity = {"source_hashes": {"civ5rl/model.py": "old"}, "corpus_hash": "same"}
        changed = {"source_hashes": {"civ5rl/model.py": "new"}, "corpus_hash": "same"}
        with self.assertRaises(ValueError):
            migration.validate_audit_migration({}, {}, identity, changed)
        with self.assertRaises(ValueError):
            migration.validate_audit_migration(
                {"model": {"width": 1}}, {"model": {"width": 2}}, identity, identity
            )
        old = {"spec_version": 1}
        new = {"spec_version": 2}
        before = {
            "source_hashes": {"civ5rl/collector.py": "old"},
            "corpus_hash": "same",
        }
        after = {"source_hashes": {"civ5rl/collector.py": "new"}, "corpus_hash": "same"}
        self.assertEqual(
            migration.validate_audit_migration(old, new, before, after),
            {
                "config_changes": ["spec_version"],
                "source_changes": ["civ5rl/collector.py"],
            },
        )
        self.assertTrue(
            checkpoint.same_execution_identity(
                {**identity, "git_commit": "a"}, {**identity, "git_commit": "b"}
            )
        )
        self.assertFalse(checkpoint.same_execution_identity(identity, changed))

    def test_library_import_does_not_load_reporting_stack(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import civ5rl.diagnostic_one; assert 'matplotlib' not in sys.modules; assert 'tools.report_rl_one' not in sys.modules",
            ],
            check=True,
        )

    def test_report_is_separate_and_supports_selected_seed(self):
        from tools.report_rl_one import report_one

        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            run = directory / "seed-23"
            initial = run / "evaluation/initial-D0"
            initial.mkdir(parents=True)
            (run / "run-result.json").write_text(
                json.dumps(
                    {
                        "status": "paused",
                        "update": 0,
                        "decisions": 0,
                        "training_seconds": 0,
                        "evaluation_seconds": 0,
                    }
                )
            )
            (initial / "summary.json").write_text(
                json.dumps(
                    {
                        "wins": 1,
                        "draws": 0,
                        "losses": 0,
                        "win_rate": 1,
                        "first_winning_attack_probability": 1,
                    }
                )
            )
            report_one(directory, seed=23)
            self.assertIn(
                "seed-23", (directory / "report.md").read_text(encoding="utf-8")
            )
            self.assertTrue((directory / "progress.png").is_file())
