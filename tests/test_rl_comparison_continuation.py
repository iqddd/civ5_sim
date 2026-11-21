"""Keep a continued fit tied to its accepted Adam state and frozen dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch

from civ5rl.engine.comparison.continuation import inherited_state, validate_continuation
from civ5rl.provenance.common import file_sha256, write_json
from civ5sim.engine import digest


class ComparisonContinuationTest(TestCase):
    def test_continuation_preserves_state_with_new_operational_settings(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, collection = root / "predecessor", root / "collection"
            dataset = {"anchors": [[0, 0], [0, 1]], "pairs": []}
            write_json(source / "dataset.json", dataset)
            write_json(collection / "dataset.json", dataset)
            parent_path = root / "parent.pt"
            model = {"action.weight": torch.ones(1, 2), "memory.weight": torch.zeros(2, 2)}
            torch.save({"variant": "C2P", "model": model}, parent_path)
            training = {
                "mode": "frozen_heads_step_kl",
                "max_accepted_steps": 1,
                "anchors_per_step": 1,
                "initial_lr": 1e-5,
                "kl_limit": 0.015,
                "max_retries": 2,
            }
            predecessor = {
                "kind": "rl01.rollout_comparison",
                "evaluation_protocol": "paired_development_only",
                "training": training,
                "output": str(source),
                "training_seconds": 10,
                "parents": [str(parent_path)],
                "parent_sha256": [file_sha256(parent_path)],
                "workers": 2,
            }
            manifest = {"training": [0, 1]}
            passport = {
                "config_hash": self._record_hash(predecessor),
                "corpus_hash": self._record_hash(manifest),
            }
            write_json(source / "passport.json", passport)
            write_json(source / "corpus-manifest.json", manifest)
            write_json(source / "resolved-config.json", predecessor)
            write_json(source / "status.json", {"status": "complete"})
            write_json(source / "phase-budgets/training.json", {"seconds": 9.0})
            accepted_model = {**model, "action.weight": torch.full((1, 2), 0.5)}
            candidate = source / "candidate.pt"
            torch.save({"variant": "C2P", "model": accepted_model}, candidate)
            write_json(
                source / "training.json",
                {
                    "status": "complete",
                    "training_steps": 1,
                    "seconds": 9.5,
                    "candidate_sha256": file_sha256(candidate),
                },
            )
            moments = {
                "step": torch.tensor(1.0),
                "exp_avg": torch.zeros(1, 2),
                "exp_avg_sq": torch.ones(1, 2),
            }
            state = {
                "identity": {"passport": passport, "dataset_hash": digest(dataset)},
                "mode": training["mode"],
                "stop_reason": "accepted_step_limit",
                "steps": 1,
                "attempts": 1,
                "history": [{"accepted": True}],
                "anchor_cursor": 1,
                "anchor_order": [1, 0],
                "pending": None,
                "model": accepted_model,
                "lr": training["initial_lr"],
                "seconds": 9.5,
                "optimizer": {
                    "param_groups": [{"lr": training["initial_lr"], "params": [0]}],
                    "state": {0: moments},
                },
            }
            torch.save(state, source / "training.pt")
            paths = (
                "training.pt",
                "candidate.pt",
                "training.json",
                "status.json",
                "resolved-config.json",
                "dataset.json",
                "phase-budgets/training.json",
            )
            hashes = {name: file_sha256(source / name) for name in paths}
            config = {
                **predecessor,
                "workers": 5,
                "output": str(root / "continued"),
                "training_seconds": 20,
                "training": {**training, "max_accepted_steps": 2},
                "collection_input": {
                    "run": str(collection),
                    "sha256": {"dataset.json": file_sha256(collection / "dataset.json")},
                },
                "fit_continuation": {"run": str(source), "sha256": hashes},
            }
            restored = inherited_state(config, {"dataset_hash": digest(dataset)})
            self.assertEqual(
                (restored["steps"], restored["seconds"], restored["anchor_order"]), (1, 9.5, [1, 0])
            )
            self.assertIsNone(restored["stop_reason"])
            torch.testing.assert_close(restored["optimizer"]["state"][0]["exp_avg_sq"], moments["exp_avg_sq"])

            moments["step"] = torch.tensor(1.5)
            torch.save(state, source / "training.pt")
            hashes["training.pt"] = file_sha256(source / "training.pt")
            with self.assertRaisesRegex(ValueError, "head Adam moments or clock"):
                validate_continuation(config)

    @staticmethod
    def _record_hash(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
