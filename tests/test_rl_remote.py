"""Focused integrity checks for portable RL capsules."""

import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

from civ5rl.deploy.remote import _sha, _validate_zip, package_capsule, safe_extract


class RemoteTests(unittest.TestCase):
    def test_package_inherits_config_and_includes_untracked_sources_and_open_maps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.test",
                    "commit",
                    "--allow-empty",
                    "-qm",
                    "initial",
                ],
                cwd=root,
                check=True,
            )

            def put(name, data):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(data, encoding="utf-8")

            for name in (
                "build/combat_rules.h",
                "build/combat_rules.json",
                "build/cp_geometry.inc",
                "parent.pt",
                "open.json",
                "helper.py",
            ):
                put(name, name)
            put(
                "manifest.json",
                json.dumps(
                    {
                        "maps": [
                            {"path": "open.json", "split": "train"},
                            {"path": "closed.json", "split": "test"},
                        ]
                    }
                ),
            )
            put(
                "base.json",
                json.dumps(
                    {
                        "experiment": {
                            "parent": "parent.pt",
                            "parent_manifest": "manifest.json",
                        }
                    }
                ),
            )
            put("config.json", json.dumps({"base_config": "base.json"}))
            out = Path(tmp) / "capsule"
            package_capsule("config.json", out, root=root)
            transfer = json.loads((out / "transfer.json").read_text(encoding="utf-8"))
            for name, meta in transfer["archives"].items():
                self.assertEqual(_sha(out / name), meta["sha256"])
                _validate_zip(out / name, meta["members"])
            self.assertIn("helper.py", transfer["archives"]["source.zip"]["members"])
            inputs = transfer["archives"]["inputs.zip"]["members"]
            self.assertIn("open.json", inputs)
            self.assertIn("base.json", inputs)
            self.assertNotIn("closed.json", inputs)

    def test_extract_rejects_traversal_collisions_and_conflicts_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "files.zip"
            out = base / "out"
            for members in (
                ("ok.txt", "../bad"),
                ("A.py", "a.py"),
                ("Dir/a", "dir/b"),
                ("a", "A/b"),
            ):
                with zipfile.ZipFile(archive, "w") as z:
                    for name in members:
                        z.writestr(name, "data")
                with self.assertRaises(ValueError):
                    safe_extract(archive, out)
                self.assertFalse(out.exists())
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("a.txt", "new")
                z.writestr("b.txt", "second")
            out.mkdir()
            (out / "a.txt").write_text("old")
            with self.assertRaises(ValueError):
                safe_extract(archive, out)
            self.assertFalse((out / "b.txt").exists())
            self.assertEqual((out / "a.txt").read_text(), "old")
            safe_extract(archive, out, overwrite=True)
            self.assertEqual((out / "a.txt").read_text(), "new")

    def test_member_hash_corruption_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "x.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("a", "incorrect")
            out = Path(tmp) / "out"
            with self.assertRaises(ValueError):
                safe_extract(archive, out, expected={"a": "0" * 64})
            self.assertFalse(out.exists())
