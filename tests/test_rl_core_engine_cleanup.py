import subprocess
import sys
import unittest

import torch
from test_rl import context, scenario

from civ5rl.core.context import context_for, context_from_spec
from civ5rl.core.model import Policy, batch_inputs
from civ5rl.core.observation import ObservationEncoder, ordered_actions
from civ5sim import Battle


class CoreEngineCleanupTests(unittest.TestCase):
    def test_context_defaults_and_overrides(self):
        s = scenario()
        original = context(s)
        spec = {
            "scenario": s,
            "armies": original["armies"],
            "learner_side": 0,
            "max_rounds": original["max_rounds"],
        }
        self.assertEqual(
            context_from_spec(spec),
            context_for(s, spec["armies"], 0, spec["max_rounds"]),
        )
        self.assertEqual(context_from_spec(spec, side=1, seed=7)["side"], 1)
        self.assertEqual(context_from_spec(spec, side=1, seed=7)["seed"], 7)

    def test_worker_imports_do_not_load_torch(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import civ5rl.engine.collector, civ5rl.core.env; "
                    "assert 'torch' not in sys.modules"
                ),
            ],
            check=True,
        )

    def test_feedforward_forward_matches_previous_computation_and_gradients(self):
        s = scenario()
        with Battle(s) as battle:
            obs, actions = (
                battle.observe(0),
                ordered_actions(battle.available_actions()),
            )
        encoded = ObservationEncoder(context(s)).encode(obs, actions)
        batch = batch_inputs([encoded], "cpu")
        for variant in ("A", "B"):
            with self.subTest(variant=variant):
                policy = Policy(variant)
                rng = torch.get_rng_state().clone()
                logits, values, hidden = policy(batch)
                (logits.sum() + values.sum()).backward()
                gradients = [p.grad.clone() for p in policy.parameters()]
                policy.zero_grad(set_to_none=True)
                x, ctx = policy.encode_spatial(batch)
                with policy.autocast(ctx):
                    h = policy.memory(ctx).float()
                expected_logits, expected_values = policy.score_heads(batch, x, h)
                (expected_logits.sum() + expected_values.sum()).backward()
                self.assertTrue(torch.equal(logits, expected_logits))
                self.assertTrue(torch.equal(values, expected_values))
                self.assertIsNone(hidden)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                for actual, parameter in zip(gradients, policy.parameters()):
                    self.assertTrue(torch.equal(actual, parameter.grad))


if __name__ == "__main__":
    unittest.main()
