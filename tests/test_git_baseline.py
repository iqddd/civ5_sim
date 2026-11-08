"""Prevent silently benchmarking live code as Git baseline or mixing replay state."""

import inspect
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from civ5rl.engine.benchmark import run_replay_benchmark
from civ5rl.provenance import common
from civ5rl.provenance.rng import capture_rng


class GitBaselineTests(unittest.TestCase):
    def test_git_blob_isolated_from_live_source_and_failed_import_is_removed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            old = "from dataclasses import dataclass\n@dataclass\nclass Item:\n    value: int = 7\ndef result():\n    return Item().value\n"
            (root / "source.py").write_text(old, encoding="utf-8")
            (root / "broken.py").write_text(
                "raise RuntimeError('broken baseline')\n", encoding="utf-8"
            )
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.test",
                    "commit",
                    "-qm",
                    "baseline",
                ],
                cwd=root,
                check=True,
            )
            (root / "source.py").write_text("working_tree = 999\n", encoding="utf-8")
            with patch.object(common, "ROOT", root):
                baseline = common.load_module_from_git(
                    "_test_git_baseline", "source.py"
                )
                try:
                    self.assertEqual(baseline.result(), 7)
                    self.assertIn(
                        "return Item().value", inspect.getsource(baseline.result)
                    )
                    self.assertTrue(baseline.__file__.startswith("git:"))
                    self.assertEqual(
                        (root / "source.py").read_text(), "working_tree = 999\n"
                    )
                    self.assertEqual(
                        {p.name for p in root.glob("*.py")}, {"source.py", "broken.py"}
                    )
                    with self.assertRaisesRegex(ValueError, "already loaded"):
                        common.load_module_from_git("_test_git_baseline", "source.py")
                    self.assertIs(sys.modules["_test_git_baseline"], baseline)
                    with self.assertRaisesRegex(RuntimeError, "broken baseline"):
                        common.load_module_from_git(
                            "_test_broken_baseline", "broken.py"
                        )
                    self.assertNotIn("_test_broken_baseline", sys.modules)
                finally:
                    sys.modules.pop("_test_git_baseline", None)

    def test_replay_abba_resets_rng_and_isolates_each_cache(self):
        policy = torch.nn.Linear(1, 1)
        policy.variant = "C"
        rng = capture_rng()
        calls = []
        caches = []
        episodes = [{"chosen": [0], "joint_commands": [{"kind": "end_turn"}]}]

        class Cache:
            def __init__(self, eps, variant, device, resident):
                self.episodes = eps
                caches.append(self)

        def implementation(name, difference=0):
            def measure(policy, cache, sequences, *, full_distribution, temperature=1):
                calls.append(name)
                return {
                    "sampled_kl": random.random()
                    + np.random.rand()
                    + torch.rand(()).item()
                    + difference
                }

            return SimpleNamespace(RolloutCache=Cache, measure_replay=measure)

        try:
            result = run_replay_benchmark(
                policy=policy,
                episodes=episodes,
                implementations={
                    "git": implementation("A"),
                    "working": implementation("B"),
                },
                config={},
                device="cpu",
                repeats=1,
                warmup=1,
            )
            self.assertEqual(calls, list("ABABBA"))
            self.assertIsNot(caches[0].episodes, caches[1].episodes)
            self.assertIsNot(caches[0].episodes, episodes)
            self.assertEqual(result["order"], ["git", "working", "working", "git"])
            self.assertEqual(result["Adam_steps"], 0)
            self.assertEqual(result["new_matches"], 0)
            self.assertTrue(policy.training)
            self.assertEqual(random.getstate(), rng["python_rng"])
            self.assertTrue(
                np.array_equal(np.random.get_state()[1], rng["numpy_rng"][1])
            )
            self.assertTrue(torch.equal(torch.get_rng_state(), rng["torch_rng"]))
            with self.assertRaises(AssertionError):
                run_replay_benchmark(
                    policy=policy,
                    episodes=episodes,
                    implementations={
                        "git": implementation("A"),
                        "working": implementation("B", 1),
                    },
                    config={},
                    device="cpu",
                    repeats=1,
                    warmup=1,
                )
        finally:
            common_random = rng["python_rng"]
            random.setstate(common_random)
            np.random.set_state(rng["numpy_rng"])
            torch.set_rng_state(rng["torch_rng"])

    def test_old_replay_cannot_be_silently_used_for_temperature_half(self):
        def old(policy, cache, sequences, *, full_distribution):
            raise AssertionError("Unsupported baseline should not run")

        with self.assertRaisesRegex(ValueError, "temperature support"):
            run_replay_benchmark(
                policy=torch.nn.Linear(1, 1),
                episodes=[{"policy_temperature": 0.5}],
                implementations={"git": SimpleNamespace(measure_replay=old)},
                config={"policy_temperature": 0.5},
                device="cpu",
            )
