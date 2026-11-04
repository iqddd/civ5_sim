import unittest

import torch
from civ5rl.ppo import update


class ReverseRng:
    def __init__(self):
        self.calls = 0

    def shuffle(self, order):
        self.calls += 1
        order.reverse()


class FakeBackend:
    instance = None

    def __init__(self, policy, episodes, config, device, monitor):
        FakeBackend.instance = self
        self.policy = policy
        self.prefix_seconds = self.backward_seconds = 0.0
        self.batches = []
        self.gradients = []
        self.kl_calls = 0

    def precheck(self):
        return 0.0, 0.0

    def backward(self, minibatch, count, entropy_coefficient):
        self.batches.append((list(minibatch), count, entropy_coefficient))
        self.policy.weight.sum().backward()
        return count

    def kl(self):
        self.kl_calls += 1
        return 0.1

    def record_gradient(self, norm):
        self.gradients.append(norm.item())

    def statistics(self, tokens):
        return (
            {"policy_loss": 2.0},
            sum(self.gradients) / len(self.gradients),
            max(self.gradients),
        )

    def extras(self, optimizer_seconds):
        return {}


class PPOControlTests(unittest.TestCase):
    def test_epoch_stop_schedules_and_token_aggregation(self):
        episodes = [
            {"chosen": [0] * 5, "old_value": [0.0] * 5, "result": {"reward": 1}},
            {"chosen": [0] * 2, "old_value": [0.5] * 2, "result": {"reward": -1}},
        ]
        config = {
            "ppo": {
                "sequence_length": 2,
                "sequences_per_minibatch": 2,
                "epochs": 3,
                "stop_kl": 0.05,
                "max_gradient_norm": 10,
                "entropy_schedule": {
                    "initial": 0.1,
                    "final": 0.0,
                    "linear_decay_decisions": 100,
                },
            },
            "optimizer": {
                "lr_schedule": {
                    "initial": 0.2,
                    "final": 0.1,
                    "linear_decay_decisions": 100,
                }
            },
        }
        policy = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(policy.parameters(), lr=1)
        rng = ReverseRng()
        result = update(
            policy, optimizer, episodes, config, 50, "cpu", rng, _backend=FakeBackend
        )
        backend = FakeBackend.instance
        self.assertEqual([x[1] for x in backend.batches], [3, 4])
        self.assertTrue(all(x[2] == 0.05 for x in backend.batches))
        self.assertAlmostEqual(result["lr"], 0.15)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.15)
        self.assertEqual(
            (
                result["epochs"],
                result["optimizer_steps"],
                result["objective_valid_tokens"],
            ),
            (1, 2, 7),
        )
        self.assertEqual((backend.kl_calls, result["policy_loss"]), (1, 2.0))
