import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from civ5rl import checkpoint


class ProvenanceRefactorTests(unittest.TestCase):
    def test_diagnostic_limit_cannot_cross_trusted_provenance(self):
        # A signed case with the wrong source must not silently alter the horizon.
        from civ5rl.protocols.diagnostic import DiagnosticFactory
        from civ5rl.provenance import bound_cases, case_manifest
        from civ5sim.engine import digest

        spec = {"scenario": {"seed": 1}, "learner_decision_limit": 1}
        generic = case_manifest([spec], source="other", split="train", namespace="test")
        self.assertEqual(bound_cases(generic)[0]["learner_decision_limit"], 1)
        forged = {
            **generic,
            "source": "rl01.diagnostic.D0",
            "split": "diagnostic_train",
        }
        forged["hash"] = digest({k: v for k, v in forged.items() if k != "hash"})
        with self.assertRaises(ValueError):
            bound_cases(forged)
        factory = DiagnosticFactory({"D0": {"train": forged}}, 17)
        with self.assertRaises(ValueError):
            factory.make(0, "D0")
        valid = {**spec, "diagnostic_protocol": "D0-one-decision"}
        manifest = case_manifest(
            [valid],
            source="rl01.diagnostic.D0-one-decision",
            split="diagnostic_train",
            namespace="test",
        )
        self.assertEqual(bound_cases(manifest)[0]["learner_decision_limit"], 1)

    def test_execution_identity_ignores_documentation_but_rejects_experiment_changes(
        self,
    ):
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
            self.assertFalse(
                checkpoint.same_execution_identity(before, {**after, **change})
            )

    def test_rng_capture_restore_repeats_all_streams(self):
        # Divergent random streams silently break exact checkpoint resumption.
        import random

        import numpy as np
        import torch

        from civ5rl.provenance.rng import capture_rng, restore_rng

        shuffle = random.Random(42)
        original = capture_rng(shuffle)
        try:
            for cuda_enabled in (False, torch.cuda.is_available()):
                with patch.object(
                    torch.cuda, "is_available", return_value=cuda_enabled
                ):
                    saved = capture_rng(shuffle)

                    def draw(cuda_enabled=cuda_enabled):
                        return (
                            random.random(),
                            np.random.random(),
                            torch.rand(3),
                            torch.rand(3, device="cuda").cpu()
                            if cuda_enabled
                            else None,
                            shuffle.random(),
                        )

                    expected = draw()
                    restore_rng(saved, shuffle)
                    actual = draw()
                    self.assertEqual(expected[:2], actual[:2])
                    torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)
                    if cuda_enabled:
                        torch.testing.assert_close(
                            expected[3], actual[3], rtol=0, atol=0
                        )
                    self.assertEqual(expected[4], actual[4])
        finally:
            restore_rng(original, shuffle)

    def test_migration_identity_is_strict_about_markdown(self):
        from civ5rl.provenance.migration import validate_audit_migration

        old = {"source_hashes": {"docs/rl-plan.md": "old"}}
        new = {"source_hashes": {"docs/rl-plan.md": "changed"}}
        self.assertTrue(checkpoint.same_execution_identity(old, new))
        with self.assertRaises(ValueError):
            validate_audit_migration({}, {}, old, new)

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
