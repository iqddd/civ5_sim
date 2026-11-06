"""Token-based Adam steps, per-step KL and frozen D1 initialization."""

import random
import tempfile
import unittest
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import torch
from civ5rl.config import load_config
from civ5rl.common import configure, read_json
from civ5rl.ppo import token_minibatches, update
from civ5rl.validation import validate_config
from test_rl_ppo_refactor import FakeBackend, ReverseRng

from civ5rl.protocols.diagnostic import gate
from civ5rl.protocols.diagnostic_d1 import load_weights, next_state_d1, prepare_d1


class LowKLBackend(FakeBackend):
    def kl(self):
        self.kl_calls += 1
        return 0.001


class D1Tests(unittest.TestCase):
    def setUp(self):
        configure(17)
        self.config = load_config("configs/rl-diagnostic-d1.json")

    def test_token_partition_short_long_mixed_and_remainder(self):
        for lengths in ([1] * 8192, [128] * 64, [13, 123, 1, 128] * 9):
            order = [(i, 0, n) for i, n in enumerate(lengths)]
            random.Random(18).shuffle(order)
            batches = list(token_minibatches(order, 1024))
            sizes = [sum(b - a for _, a, b in spans) for spans in batches]
            self.assertTrue(all(n == 1024 for n in sizes[:-1]))
            self.assertEqual(sum(sizes), sum(lengths))
            expected = [(i, j) for i, a, b in order for j in range(a, b)]
            actual = [
                (i, j) for spans in batches for i, a, b in spans for j in range(a, b)
            ]
            self.assertEqual(actual, expected)
            self.assertTrue(
                all(0 < b - a <= 128 for spans in batches for _, a, b in spans)
            )

    def test_per_step_stop_and_full_epoch_counts(self):
        episodes = [
            {"chosen": [0] * 8192, "old_value": [0.0] * 8192, "result": {"reward": 1}}
        ]
        for backend, expected_steps, completed in (
            (FakeBackend, 1, 0),
            (LowKLBackend, 32, 4),
        ):
            model = torch.nn.Linear(1, 1, bias=False)
            optimizer = torch.optim.Adam(model.parameters())
            with patch(
                "torch.nn.utils.clip_grad_norm_", wraps=torch.nn.utils.clip_grad_norm_
            ) as clip:
                result = update(
                    model,
                    optimizer,
                    episodes,
                    self.config,
                    0,
                    "cpu",
                    ReverseRng(),
                    _backend=backend,
                )
            self.assertEqual(result["optimizer_steps"], expected_steps)
            self.assertEqual(result["epochs_completed"], completed)
            self.assertEqual(clip.call_count, expected_steps)
            self.assertEqual(FakeBackend.instance.kl_calls, expected_steps)
            self.assertEqual(
                [r["valid_tokens"] for r in result["optimizer_step_records"]],
                [1024] * expected_steps,
            )
            self.assertEqual(int(optimizer.state[model.weight]["step"]), expected_steps)
            self.assertEqual(
                result["stop_reason"],
                "stop_kl" if expected_steps == 1 else "epochs_completed",
            )

    def test_reference_batched_loss_and_gradients_with_split_prefixes(self):
        from civ5rl.model import Policy
        from test_rl import RLTests

        case = RLTests()
        policy = Policy()
        episodes = case._short_rollout(policy)
        fast = deepcopy(policy)
        config = deepcopy(self.config)
        config["ppo"].update(
            epochs=1,
            sequence_length=3,
            valid_tokens_per_minibatch=4,
            sequences_per_microbatch=2,
        )
        config["performance"] = {"batched_ppo": False}
        a = update(
            policy,
            torch.optim.SGD(policy.parameters(), lr=3e-4),
            episodes,
            config,
            0,
            "cpu",
            random.Random(18),
        )
        config["performance"]["batched_ppo"] = True
        b = update(
            fast,
            torch.optim.SGD(fast.parameters(), lr=3e-4),
            episodes,
            config,
            0,
            "cpu",
            random.Random(18),
        )
        for key in ("policy_loss", "value_loss", "entropy"):
            self.assertAlmostEqual(a[key], b[key], places=6)
        self.assertEqual(
            [r["valid_tokens"] for r in a["optimizer_step_records"]],
            [r["valid_tokens"] for r in b["optimizer_step_records"]],
        )
        for k, v in policy.state_dict().items():
            torch.testing.assert_close(v, fast.state_dict()[k], rtol=2e-5, atol=1e-7)

    def test_d1_initialization_data_and_gate(self):
        from civ5rl.model import Policy

        validate_config(self.config)
        with tempfile.TemporaryDirectory() as tmp:
            data = prepare_d1(tmp, self.config)
        original = read_json(self.config["diagnostic"]["source_dataset"])
        self.assertEqual(data["D1"], original["D1"])
        policy = Policy()
        proof = load_weights(policy, self.config)
        self.assertTrue(proof["all_model_parameters_bitwise_equal"])
        self.assertEqual(proof["previous_decisions"], 49152)
        self.assertEqual(len(torch.optim.Adam(policy.parameters()).state), 0)
        base = [{"layout_id": i // 2, "side": i % 2, "win": False} for i in range(128)]
        for wins, passed in ((102, False), (103, True)):
            rows = [{**r, "win": i < wins} for i, r in enumerate(base)]
            self.assertEqual(
                gate("D1", {"win_rate": wins / 128}, rows, base)["passed"], passed
            )
        state = {"status": "running", "stage_updates": 7, "consecutive": 0}
        self.assertEqual(next_state_d1(state, True)["consecutive"], 1)
        self.assertEqual(
            next_state_d1({**state, "consecutive": 1}, False)["consecutive"], 0
        )
        self.assertEqual(
            next_state_d1({**state, "stage_updates": 8}, False)["status"], "failed"
        )
        self.assertEqual(
            next_state_d1({**state, "stage_updates": 8, "consecutive": 1}, True)[
                "status"
            ],
            "passed",
        )

    def test_evaluation_rng_isolation_even_on_failure(self):
        from tools.archive import run_rl_diagnostic as runner

        def consume(*args):
            random.random()
            np.random.random()
            torch.rand(5)
            raise RuntimeError("evaluation failure")

        before = random.getstate(), np.random.get_state(), torch.get_rng_state()
        with (
            patch.object(runner, "_evaluate_stage", side_effect=consume),
            self.assertRaisesRegex(RuntimeError, "evaluation failure"),
        ):
            runner.evaluate_stage(None, None, None, None, None, None)
        self.assertEqual(before[0], random.getstate())
        np.testing.assert_equal(before[1], np.random.get_state())
        torch.testing.assert_close(before[2], torch.get_rng_state(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
