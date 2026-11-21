"""Verify repeat metrics and state isolation for saved rollout replay."""

from __future__ import annotations

import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from civ5rl.experiments.benchmarks import ppo
from civ5rl.provenance.rng import capture_rng, restore_rng


class ReplayBenchmarkTests(unittest.TestCase):
    def test_replay_resets_rng_and_checks_repeat_metrics(self):
        policy = torch.nn.Linear(1, 1)
        policy.variant = "C"
        rng = capture_rng()
        self.addCleanup(restore_rng, rng)
        episodes = [
            {
                "chosen": [0],
                "joint_commands": [{"kind": "end_turn"}],
                "policy_temperature": 0.5,
            }
        ]
        replay_forward = object()
        calls = []
        divergence = 0

        def measure(
            policy, cache, sequences, *, full_distribution, temperature, forward
        ):
            self.assertIsNot(cache, episodes)
            self.assertIsNot(cache[0], episodes[0])
            self.assertFalse(policy.training)
            self.assertTrue(full_distribution)
            self.assertEqual(temperature, 0.5)
            self.assertIs(forward, replay_forward)
            calls.append(random.random() + np.random.rand() + torch.rand(()).item())
            return {"sampled_kl": calls[-1] + divergence * len(calls)}

        def run():
            return ppo.run_replay_benchmark(
                policy=policy,
                episodes=episodes,
                config={"policy_temperature": 0.5},
                device="cpu",
                repeats=2,
                warmup=1,
            )

        with (
            patch.object(ppo, "RolloutCache", side_effect=lambda eps, *a, **kw: eps),
            patch.object(ppo, "measure_replay", measure),
            patch.object(ppo, "ppo_replay_forward", return_value=replay_forward),
        ):
            result = run()
            self.assertEqual(calls, [calls[0]] * 3)
            self.assertEqual(result["order"], ["working"] * 2)
            self.assertEqual(result["results"]["working"]["passes"], 2)
            self.assertEqual(result["Adam_steps"], 0)
            self.assertEqual(result["new_matches"], 0)
            divergence = 1
            with self.assertRaisesRegex(AssertionError, "sampled_kl"):
                run()

        self.assertTrue(policy.training)
        self.assertEqual(random.getstate(), rng["python_rng"])
        self.assertTrue(np.array_equal(np.random.get_state()[1], rng["numpy_rng"][1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng["torch_rng"]))
