"""Catch silent scientific data loss/cross-operation restores in archive transport."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from civ5rl.deploy.archive import pack, seven_zip, unpack, _inventory


class ArchiveTests(unittest.TestCase):
    def test_storage_preserves_scientific_bytes_and_rejects_damaged_archive(self):
        # Deleting unpacked results before verifying a durable copy silently loses evidence.
        from civ5rl.reporting.evaluation_storage import archive_evaluations, restore_evaluations, readable_evaluations
        from civ5rl.engine.control import RunLock
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            source = run / "evaluation"
            source.mkdir()
            (source / "replay.json").write_bytes(b'{"steps":[]}\n')
            before = _inventory(source)
            archive_evaluations(run)
            self.assertFalse(source.exists())
            with RunLock(run), readable_evaluations(run):
                self.assertEqual(_inventory(source), before)
            self.assertFalse(source.exists())
            archive = run / "evaluation.storage/json.7z"
            original = archive.read_bytes()
            archive.write_bytes(original + b'damage')
            with self.assertRaises(ValueError):
                restore_evaluations(run)
            self.assertFalse(source.exists())
            archive.write_bytes(original)
            restore_evaluations(run)
            self.assertEqual(_inventory(source), before)

    def test_ppmd_roundtrip_exact_tree_and_stale_files(self):
        # A transport that reports success but omits/mutates matches invalidates later audits.
        seven_zip()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "source"
            (source / "evaluation" / "empty").mkdir(parents=True)
            (source / "evaluation" / "данные.json").write_text(
                '{"terrain":"холмы"}\n', encoding="utf-8"
            )
            (source / "steps.jsonl").write_bytes(b'{"step":1}\n{"step":2}\n')
            (source / "checkpoint.pt").write_bytes(bytes(range(256)))
            with patch("civ5rl.deploy.archive.JSON_MIN_FILES", 2):
                manifest = pack(
                    source, base / "packet", token="job-1", direction="download"
                )
            self.assertEqual(len(manifest["archived"]), 2)
            destination = base / "destination"
            (destination / "stale" / "nested").mkdir(parents=True)
            (destination / "stale" / "nested" / "old.json").write_text("old")
            # A file can become a directory, and a directory can become a file.
            (destination / "evaluation").write_text("old file")
            (destination / "checkpoint.pt").mkdir()
            receipt = unpack(
                base / "packet", destination, token="job-1", direction="download"
            )
            self.assertEqual(receipt["verified_files"], 3)
            self.assertEqual(_inventory(source), _inventory(destination))
            self.assertEqual(
                unpack(
                    base / "packet", destination, token="job-1", direction="download"
                ),
                receipt,
            )

    def test_damage_or_wrong_operation_cannot_replace_existing_results(self):
        # Unlike a traceback during a run, a falsely accepted old packet silently loses training.
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "source"
            source.mkdir()
            (source / "state.json").write_text('{"decisions":100}')
            pack(source, base / "packet", token="current", direction="download")
            destination = base / "destination"
            destination.mkdir()
            (destination / "keep.json").write_text("existing results")
            before = _inventory(destination)
            for token, direction in (("previous", "download"), ("current", "upload")):
                with self.assertRaises(ValueError):
                    unpack(
                        base / "packet", destination, token=token, direction=direction
                    )
                self.assertEqual(_inventory(destination), before)
            (base / "packet" / "files" / "state.json").write_text("damaged")
            with self.assertRaises(ValueError):
                unpack(
                    base / "packet", destination, token="current", direction="download"
                )
            self.assertEqual(_inventory(destination), before)
            with patch("civ5rl.deploy.archive.JSON_MIN_FILES", 1):
                pack(source, base / "packet", token="current", direction="download")
            with (base / "packet" / "json.7z").open("ab") as stream:
                stream.write(b"corruption")
            with self.assertRaises(ValueError):
                unpack(
                    base / "packet", destination, token="current", direction="download"
                )
            self.assertEqual(_inventory(destination), before)


if __name__ == "__main__":
    unittest.main()
