import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from civ5rl import checkpoint


class ProvenanceRefactorTests(unittest.TestCase):
    def test_execution_identity_ignores_documentation_but_rejects_experiment_changes(self):
        before = {
            "git_commit": "old",
            "source_hashes": {"civ5rl/model.py": "code", "docs/rl-plan.md": "old"},
            "corpus_hash": "corpus",
            "packages": {"torch": "same"},
        }
        after = {
            **before,
            "git_commit": "new",
            "source_hashes": {"civ5rl/model.py": "code", "docs/rl-new.md": "new"},
        }
        self.assertTrue(checkpoint.same_execution_identity(before, after))
        for change in (
            {"source_hashes": {**after["source_hashes"], "civ5rl/model.py": "changed"}},
            {"source_hashes": {**after["source_hashes"], "configs/rl.json": "new"}},
            {"corpus_hash": "changed"},
            {"packages": {"torch": "changed"}},
        ):
            self.assertFalse(checkpoint.same_execution_identity(before, {**after, **change}))

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
