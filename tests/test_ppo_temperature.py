"""Silent risks: wrong temperature gradients, twice-scaled behavior, mixed rollouts."""

import unittest
from copy import deepcopy

import numpy as np
import torch
from test_rl import context, scenario

from civ5rl.config import load_config
from civ5rl.core.distributions import policy_logits, require_rollout_temperature
from civ5rl.core.model import Policy, batch_inputs
from civ5rl.engine.ppo import encode_episode, objective, targets
from civ5rl.engine.ppo_fast import BatchedBackend, categorical_distance
from civ5rl.provenance.common import configure
from civ5sim import Action, Battle


class TemperatureTests(unittest.TestCase):
    def setUp(self):
        configure(17, "cpu")

    def test_loss_gradient_entropy_and_kl_match_explicit_cold_policy(self):
        # Zero replay KL alone cannot detect a missing 1/T actor gradient.
        current = torch.tensor(
            [[0.7, -0.1, -float("inf")], [0.0, -float("inf"), -float("inf")]],
            requires_grad=True,
        )
        old = torch.tensor(
            [[0.4, 0.1, -float("inf")], [0.0, -float("inf"), -float("inf")]]
        )
        old_logp = (old / 0.5).log_softmax(-1)
        indices = torch.tensor([0, 0])
        values = torch.tensor([0.3, 0.2], requires_grad=True)
        returns, advantages = torch.tensor([1.0, 0.0]), torch.tensor([0.8, -0.6])
        loss, stats = objective(
            current,
            values,
            indices,
            old_logp[:, 0],
            returns,
            advantages,
            0.03,
            temperature=0.5,
        )
        reference = current.detach().clone().requires_grad_()
        reference_values = values.detach().clone().requires_grad_()
        logp = (reference / 0.5).log_softmax(-1)
        probs = logp.exp()
        entropy = -(probs * logp.masked_fill(~torch.isfinite(logp), 0)).sum(-1)
        normalized = entropy / torch.tensor([np.log(2), 1.0], dtype=torch.float32)
        ratio = (logp[:, 0] - old_logp[:, 0]).exp()
        explicit = (
            -torch.minimum(
                ratio * advantages, ratio.clamp(0.8, 1.2) * advantages
            ).mean()
            + 0.5 * (reference_values - returns).square().mean()
            - 0.03 * normalized.mean()
        )
        loss.backward()
        explicit.backward()
        torch.testing.assert_close(loss, explicit, rtol=0, atol=0)
        torch.testing.assert_close(current.grad, reference.grad, rtol=0, atol=0)
        torch.testing.assert_close(values.grad, reference_values.grad, rtol=0, atol=0)
        self.assertTrue(torch.equal(current.grad[1], torch.zeros(3)))
        torch.testing.assert_close(stats["entropy"], entropy.mean())
        kl, _ = categorical_distance(
            old_logp, policy_logits(current.detach(), 0.5), torch.isfinite(current)
        )
        expected = (
            old_logp.double().exp()
            * (old_logp.double() - logp.detach().double()).masked_fill(
                ~torch.isfinite(logp), 0
            )
        ).sum(-1)
        torch.testing.assert_close(kl, expected, rtol=1e-5, atol=1e-7)
        default, _ = objective(
            current.detach(),
            values.detach(),
            indices,
            old_logp[:, 0],
            returns,
            advantages,
            0.03,
        )
        one, _ = objective(
            current.detach(),
            values.detach(),
            indices,
            old_logp[:, 0],
            returns,
            advantages,
            0.03,
            temperature=1.0,
        )
        self.assertTrue(torch.equal(default, one))

    def test_accelerated_backend_gradients_and_full_replay(self):
        # Exercise the actual backend with recurrent episodes, a mask and forced actions.
        config = load_config("configs/rl-ppo-target-bc-t05.json")
        config["performance"]["compile"] = False
        original = scenario()
        with Battle(original) as battle:
            observation, actions = battle.observe(0), tuple(battle.available_actions())
        policy = Policy("C")
        episodes = []
        for length, reward, legal in ((3, 1, actions), (1, 0, (Action("end_turn"),))):
            ep = {
                "spec": {
                    "scenario": original,
                    "armies": context(original)["armies"],
                    "learner_side": 0,
                    "max_rounds": 3,
                },
                "policy_temperature": 0.5,
                "observations": [observation] * length,
                "actions": [legal] * length,
                "chosen": [0] * length,
                "old_value": [0.0] * length,
                "result": {"reward": reward},
            }
            encoded = encode_episode(ep, "C")
            with torch.no_grad():
                logits, value, _ = policy.forward_sequences(
                    batch_inputs(encoded, "cpu"), [length]
                )
            behavior = (logits / 0.5).log_softmax(-1).numpy()
            ep.update(
                old_log_probs=[r.copy() for r in behavior],
                old_logp=behavior[:, 0].tolist(),
                old_value=value.tolist(),
            )
            episodes.append(ep)
        backend = BatchedBackend(policy, episodes, config, "cpu")
        pre_kl, error = backend.precheck()
        self.assertLess(abs(pre_kl), 1e-7)
        self.assertLess(error, 1e-5)
        self.assertLess(abs(backend.replay["full_kl_free"]), 1e-7)
        # Move weights before comparing nonzero gradients and full old||current KL.
        with torch.no_grad():
            next(policy.parameters()).add_(0.001)
        reference = deepcopy(policy)
        spans = [(0, 0, 3), (1, 0, 1)]
        backend.backward(spans, 4, 0.03)
        encoded = [item for ep in episodes for item in encode_episode(ep, "C")]
        logits, value, _ = reference.forward_sequences(
            batch_inputs(encoded, "cpu"), [3, 1]
        )
        old = torch.tensor([v for ep in episodes for v in ep["old_logp"]])
        ret = torch.tensor([1.0, 1.0, 1.0, 0.0])
        adv = torch.from_numpy(np.concatenate(targets(episodes)))
        loss, _ = objective(
            logits / 0.5, value, torch.zeros(4, dtype=torch.long), old, ret, adv, 0.03
        )
        loss.backward()
        for (name, p), (_, q) in zip(
            policy.named_parameters(), reference.named_parameters()
        ):
            torch.testing.assert_close(p.grad, q.grad, rtol=3e-5, atol=2e-6, msg=name)
        current = (logits / 0.5).log_softmax(-1)
        behavior = torch.from_numpy(np.stack(episodes[0]["old_log_probs"])).double()
        explicit = (behavior.exp() * (behavior - current[:3].double())).sum(-1).mean()
        self.assertAlmostEqual(backend.kl(), float(explicit.detach()), places=6)
        with self.assertRaisesRegex(ValueError, "temperature"):
            BatchedBackend(
                policy, episodes, {**config, "policy_temperature": 1.0}, "cpu"
            )
        with self.assertRaisesRegex(ValueError, "temperature"):
            require_rollout_temperature([{}], 0.5)
        require_rollout_temperature([{}], 1.0)


if __name__ == "__main__":
    unittest.main()
