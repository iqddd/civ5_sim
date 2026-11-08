import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from civ5rl.engine.benchmark import assert_update_parity, run_full_cycle_benchmark


class BenchmarkTest(unittest.TestCase):
    def test_abba_restarts_rng_and_detects_trajectory_changes(self):
        calls = []
        reset = lambda: None
        shuffle = random.Random(91)
        shuffle_state = shuffle.getstate()

        def result(name):
            calls.append(name)
            return {"model": {"x": torch.tensor([random.random(), np.random.rand(),
                                                   torch.rand(()).item(), shuffle.random()])},
                    "optimizer": {}, "gradients": {"x": torch.tensor([1.0])},
                    "sampled_kl": 0.01, "commands": [{"kind": "end_turn"}]}

        assert_update_parity(lambda: result("A"), lambda: result("B"), reset=reset,
                             shuffle_rng=shuffle)
        self.assertEqual(calls, list("ABBA"))
        self.assertEqual(shuffle.getstate(), shuffle_state)
        def changed():
            value = result("B")
            value["commands"] = []
            return value
        with self.assertRaisesRegex(AssertionError, "Trajectory"):
            assert_update_parity(lambda: result("A"), changed, reset=reset,
                                 shuffle_rng=shuffle)

    def test_trials_reset_model_adam_and_shuffle(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        shuffle = random.Random(11)
        initial_model = {k: v.clone() for k, v in model.state_dict().items()}
        initial_rng = shuffle.getstate()
        seen = []

        class Collector:
            def collect(self, policy, *args, **kwargs):
                seen.append((policy.weight.item(), len(optimizer.state), shuffle.random()))
                return [{"chosen": [0], "joint_commands": [], "result": {}}], None, {}

        def fake_update(policy, optim, episodes, config, decisions, device, rng):
            policy(torch.ones(1, 1)).sum().backward()
            optim.step()
            optim.zero_grad()
            return {"optimizer_steps": 1}

        with (tempfile.TemporaryDirectory() as tmp,
              patch("civ5rl.engine.benchmark.update", fake_update),
              patch("civ5rl.engine.benchmark.save_rollout", return_value={"sha256": "fixture"})):
            result = run_full_cycle_benchmark(policy=model, optimizer=optimizer,
                shuffle_rng=shuffle, collector=Collector(), config={}, specs=[1],
                manifest={}, output=tmp, device="cpu", repeats=2)
        self.assertEqual(len(seen), 3)
        self.assertTrue(all(x == seen[0] for x in seen))
        self.assertEqual(shuffle.getstate(), initial_rng)
        self.assertFalse(optimizer.state)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, initial_model[key], rtol=0, atol=0)
        self.assertIsNotNone(result["seconds_per_accepted_adam_amortized"])
