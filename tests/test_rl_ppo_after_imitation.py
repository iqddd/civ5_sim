"""Silent risks: forced-diluted KL, deadline overruns, and confirmation leakage."""

import random
import tempfile
import unittest
from copy import deepcopy
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch

from civ5rl.engine.ppo import token_minibatches, update
from civ5rl.engine.ppo_fast import measure_replay
from civ5rl.experiments.evaluation import paired_world_bootstrap
from civ5rl.protocols.ppo_after_imitation import TrainingFactory, prepare_cases
from civ5rl.provenance.common import configure, read_json
from civ5rl.provenance.validation import validate_config


class PPOAfterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure(17)
        cls.config = read_json("configs/rl-ppo-after-imitation.json")

    def test_packed_rollout_preserves_legal_order_and_ragged_behavior(self):
        # PPO's in-memory check cannot detect a corrupt diagnostic serialization.
        from civ5rl.experiments.ppo_artifacts import load_rollout, save_rollout
        from civ5sim import Action

        end, attack = Action("end_turn"), Action("ranged", 0, 5)
        episodes = [{
            "observations": [{"turn": 0}, {"turn": 1}],
            "actions": [(attack, end), (end,)],
            "chosen": [1, 0],
            "old_log_probs": [np.log(np.asarray([0.25, 0.75], np.float32)), np.zeros(1, np.float32)],
            "old_logp": [float(np.log(np.float32(0.75))), 0.0],
            "old_value": [0.125, 0.25],
            "result": {"reward": 0},
        }]
        with tempfile.TemporaryDirectory() as directory:
            save_rollout(episodes, directory, 1)
            restored = load_rollout(Path(directory) / "first.pt")
        for key in ("observations", "actions", "chosen", "old_logp", "old_value", "result"):
            self.assertEqual(episodes[0][key], restored[0][key])
        for expected, actual in zip(episodes[0]["old_log_probs"], restored[0]["old_log_probs"]):
            np.testing.assert_array_equal(expected, actual)
            self.assertEqual(expected.dtype, actual.dtype)

    def test_profile_split_balance_and_confirmation_are_frozen(self):
        validate_config(self.config)
        wrong = deepcopy(self.config)
        wrong["reward"]["loss"] = -1
        with self.assertRaisesRegex(ValueError, "reward.loss"):
            validate_config(wrong)
        plan = prepare_cases(self.config)
        self.assertEqual(len(plan["development"]), 64)
        self.assertEqual(len(plan["confirmation"]), 256)
        from civ5rl.protocols.imitation_own_states import own_state_physical_hash

        old = {own_state_physical_hash(c["scenario"]) for c in plan["development"]}
        new = {c["physical_hash"] for c in plan["confirmation"]}
        self.assertEqual(len(new), 128)
        self.assertFalse(old & new)
        factory = TrainingFactory(self.config, plan["base"])
        cases = [factory.make(i) for i in range(128)]
        factory.validate_specs(cases)
        from collections import Counter

        self.assertEqual(
            set(Counter((c["learner_side"], c["placement"]) for c in cases).values()),
            {32},
        )
        self.assertFalse(
            {c["parent_world"] for c in cases}
            & set(plan["base"]["worlds"]["validation"])
        )
        self.assertTrue(
            all(
                c["opponent"] == "cp" and c["reward_profile"] == "win_probability"
                for c in cases
            )
        )

    def test_free_KL_has_correct_denominator_and_all_forced_is_null(self):
        class Model:
            variant = "B"
            spatial_cuda_graphs = False

            def forward_sequences(self, batch, lengths, hidden):
                return batch["logits"], torch.zeros(sum(lengths)), None

        class Cache:
            device = "cpu"
            has_behavior_distribution = False
            episodes = [{}]
            encoded = [[None] * 3]
            offsets = np.asarray([0, 3])
            action_counts = np.asarray([2, 1, 1])

            def batch(self, spans):
                return (
                    {
                        "logits": torch.tensor([[0.0, 0.0], [0.0, -1e9], [0.0, -1e9]]),
                        "action_mask": torch.tensor(
                            [[True, True], [True, False], [True, False]]
                        ),
                    },
                    {
                        "indices": torch.zeros(3, dtype=torch.long),
                        "old_logp": torch.tensor([-1.0, 0.0, 0.0]),
                        "returns": torch.zeros(3),
                    },
                    [3],
                )

        result = measure_replay(Model(), Cache())
        self.assertAlmostEqual(
            result["sampled_kl_free"], result["sampled_kl"] * 3, places=6
        )
        self.assertEqual(result["free_decisions"], 1)
        cache = Cache()
        cache.action_counts = np.ones(3, dtype=int)
        original = cache.batch

        def forced(spans):
            batch, labels, lengths = original(spans)
            batch["action_mask"] = torch.tensor([[True, False]] * 3)
            batch["logits"] = torch.tensor([[0.0, -1e9]] * 3)
            labels["old_logp"].zero_()
            return batch, labels, lengths

        cache.batch = forced
        self.assertIsNone(measure_replay(Model(), cache)["sampled_kl_free"])

    def test_deadline_stops_before_step_or_after_required_KL_and_remainder(self):
        config = deepcopy(self.config)
        episodes = [
            {"chosen": [0] * 1027, "old_value": [0.0] * 1027, "result": {"reward": 1}}
        ]
        clock = [0.0]

        class Backend:
            prefix_seconds = backward_seconds = 0.0
            instance = None

            def __init__(self, policy, *args):
                self.policy = policy
                self.steps = 0
                self.kl_calls = 0
                Backend.instance = self

            def precheck(self):
                return 0.0, 0.0

            def backward(self, batch, count, entropy):
                self.policy.weight.sum().backward()
                self.steps += 1
                clock[0] = 101.0
                return count

            def record_gradient(self, norm):
                pass

            def kl(self):
                self.kl_calls += 1
                return 0.001

            def statistics(self, tokens):
                return {}, None, None

            def extras(self, seconds):
                return {}

        model = torch.nn.Linear(1, 1, bias=False)
        opt = torch.optim.Adam(model.parameters())
        with patch("civ5rl.engine.ppo.perf_counter", side_effect=lambda: clock[0]):
            result = update(
                model,
                opt,
                episodes,
                config,
                0,
                "cpu",
                random.Random(17),
                _backend=Backend,
                deadline=100.0,
            )
        self.assertEqual((result["optimizer_steps"], Backend.instance.kl_calls), (1, 1))
        self.assertFalse(result["update_complete"])
        self.assertEqual(result["stop_reason"], "active_time_limit")
        self.assertEqual(result["optimizer_step_records"][0]["valid_tokens"], 1024)
        clock[0] = 101.0
        with patch("civ5rl.engine.ppo.perf_counter", side_effect=lambda: clock[0]):
            result = update(
                model,
                opt,
                episodes,
                config,
                0,
                "cpu",
                random.Random(17),
                _backend=Backend,
                deadline=100.0,
            )
        self.assertEqual((result["optimizer_steps"], Backend.instance.kl_calls), (0, 0))
        batches = list(token_minibatches([(0, 0, 1027)], 1024))
        self.assertEqual(
            [sum(b - a for _, a, b in spans) for spans in batches], [1024, 3]
        )

    def test_confirmation_bootstrap_keeps_sixteen_cases_per_world(self):
        cases = {
            (w, i, s): {"placement": "contact" if i < 4 else "separated"}
            for w in range(16)
            for i in range(8)
            for s in (0, 1)
        }
        a = dict.fromkeys(cases, 0)
        b = dict.fromkeys(cases, 1)
        result = paired_world_bootstrap(
            a, b, samples=10000, seed=20261103, cases=cases, cases_per_world=16
        )
        self.assertEqual(result["CI95"], [1.0, 1.0])
        b.pop(next(iter(b)))
        with self.assertRaises(ValueError):
            paired_world_bootstrap(a, b, samples=10, seed=17, cases_per_world=16)


if __name__ == "__main__":
    unittest.main()
