"""Silent provenance failures: allowing changed training code or trusting broken archives."""

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from civ5rl.provenance import checkpoint


class PassportSourceRoleTests(unittest.TestCase):
    def fixture(self, root):
        names = {
            "civ5rl/core/model.py": "model = 1\n",
            "civ5rl/engine/ppo.py": "loss = 1\n",
            "civ5rl/protocols/corpus.py": "cases = 1\n",
            "native/combat.cpp": "// simulator\n",
            "civ5rl/experiments/evaluation.py": "match_execution = 1\n",
            "civ5rl/reporting/snapshot.py": "graph = 1\n",
            "civ5rl/reporting/evaluation_audit.py": "bootstrap = 1\n",
            "civ5rl/deploy/remote.py": "inspect = 1\n",
            "civ5rl/provenance/audit.py": "audit = 1\n",
            "tools/report_rl.py": "plot = 1\n",
            "tools/ctl_rl.py": "audit = 1\n",
            "tools/run_ppo.py": "training = 1\n",
            "tools/_build_common.py": "compiler = 1\n",
            "tools/_imitation_benchmark.py": "demonstrations = 1\n",
            "tests/test_loss.py": "test = 1\n",
        }
        for name, text in names.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return names

    def test_discovery_and_legacy_identity_share_source_roles(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            contents = self.fixture(root)
            with patch.object(checkpoint, "ROOT", root):
                files = {
                    p.relative_to(root).as_posix()
                    for p in checkpoint.artifact_sources(
                        {"passport_artifacts": ["tools/**/*.py", "tests/**/*.py"]}
                    )
                }
            protected = {
                "civ5rl/core/model.py",
                "civ5rl/engine/ppo.py",
                "civ5rl/protocols/corpus.py",
                "native/combat.cpp",
                "civ5rl/experiments/evaluation.py",
                "tools/run_ppo.py",
                "tools/_build_common.py",
                "tools/_imitation_benchmark.py",
            }
            self.assertEqual(files, protected)
            legacy = {
                "source_hashes": dict.fromkeys(contents, "old"),
                "corpus_hash": "fixed",
            }
            current = {**legacy, "source_hashes": dict.fromkeys(protected, "old")}
            self.assertTrue(checkpoint.same_execution_identity(legacy, current))
            for name in protected:
                changed = {
                    **current,
                    "source_hashes": {**current["source_hashes"], name: "changed"},
                }
                self.assertFalse(
                    checkpoint.same_execution_identity(legacy, changed), name
                )

    def test_analysis_uses_frozen_archive_but_execution_checks_live_core(self):
        config = {"passport_artifacts": []}
        manifest = {"cases": [1, 2]}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            out = root / "saved"
            with (
                patch.object(checkpoint, "ROOT", root),
                patch.object(
                    checkpoint.subprocess,
                    "check_output",
                    side_effect=lambda args, **kw: (
                        "commit\n" if args[1] == "rev-parse" else b""
                    ),
                ),
            ):
                frozen = checkpoint.passport(out, config, manifest)
                # Old passports also contain reporting files; verify their saved
                # bytes even though they are no longer part of resume identity.
                name = "tools/report_rl.py"
                old_report = (root / name).read_bytes()
                with zipfile.ZipFile(out / "sources.zip", "a") as archive:
                    archive.writestr(name, old_report)
                frozen["source_hashes"][name] = hashlib.sha256(old_report).hexdigest()
                (out / "passport.json").write_text(json.dumps(frozen), encoding="utf-8")
                frozen_bytes = (out / "passport.json").read_bytes()
                (root / name).write_text("new plotting code\n")
                (root / "tools/new_report.py").write_text("new = 1\n")
                checkpoint.verify_execution_passport(out, config, manifest)
                (root / "civ5rl/core/model.py").write_text("changed model\n")
                self.assertEqual(
                    checkpoint.verify_frozen_passport(
                        out, config=config, manifest=manifest
                    ),
                    frozen,
                )
                with self.assertRaisesRegex(ValueError, "Training source"):
                    checkpoint.verify_execution_passport(out, config, manifest)
                self.assertEqual((out / "passport.json").read_bytes(), frozen_bytes)
                # Archive-only tooling still counts as immutable evidence.
                with zipfile.ZipFile(out / "sources.zip") as archive:
                    saved = {n: archive.read(n) for n in archive.namelist()}
                saved[name] = b"corrupt historical reporting code"
                with zipfile.ZipFile(out / "sources.zip", "w") as archive:
                    for n, data in saved.items():
                        archive.writestr(n, data)
                with self.assertRaisesRegex(ValueError, "archive corrupted"):
                    checkpoint.verify_frozen_passport(
                        out, config=config, manifest=manifest
                    )

    def test_frozen_check_rejects_changed_config_and_corpus(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            out = root / "saved"
            with (
                patch.object(checkpoint, "ROOT", root),
                patch.object(
                    checkpoint.subprocess, "check_output", side_effect=["commit\n", b""]
                ),
            ):
                checkpoint.passport(out, {"lr": 1}, {"cases": [1]})
            with self.assertRaisesRegex(ValueError, "configuration"):
                checkpoint.verify_frozen_passport(out, config={"lr": 2})
            (out / "resolved-config.json").write_text('{"lr": 2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "configuration hash"):
                checkpoint.verify_frozen_passport(out)
            (out / "resolved-config.json").write_text('{"lr": 1}', encoding="utf-8")
            corpus = out / "corpus-manifest.json"
            corpus.rename(out / "manifest.json")
            checkpoint.verify_frozen_passport(out)
            (out / "manifest.json").rename(corpus)
            (out / "corpus-manifest.json").write_text(
                '{"cases": [2]}', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "corpus manifest"):
                checkpoint.verify_frozen_passport(out)
