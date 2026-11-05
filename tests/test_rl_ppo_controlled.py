"""Silent risks: wrong KL population and rejected Adam updates leaking into policy."""

import random
import unittest
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import torch

from civ5rl.engine.optimizer_transaction import AdamSnapshot, require_finite_state
from civ5rl.engine.ppo import update
from civ5rl.engine.ppo_fast import measure_replay
from civ5rl.protocols.ppo_after_imitation import advance_controlled_rollout, prepare_cases
from civ5rl.provenance.common import read_json, configure
from civ5rl.provenance.validation import validate_config


class ControlledPPOTests(unittest.TestCase):
    def setUp(self):
        configure(17)
        self.config = read_json("configs/rl-ppo-controlled-updates.json")

    def test_exact_kl_uses_legal_actions_and_free_denominator(self):
        # A sampled estimator or forced-diluted mean silently changes acceptance.
        class Model:
            variant = "B"
            spatial_cuda_graphs = False
            def forward_sequences(self, batch, lengths, hidden):
                return batch["logits"], torch.zeros(sum(lengths)), None
        class Cache:
            device = "cpu"
            has_behavior_distribution = True
            episodes = [{}]
            encoded = [[None] * 3]
            offsets = np.array([0, 3])
            action_counts = np.array([2, 1, 1])
            def batch(self, spans):
                return ({"logits": torch.tensor([[0., 1., -1e9], [0., -1e9, -1e9], [0., -1e9, -1e9]]),
                         "action_mask": torch.tensor([[True, True, False], [True, False, False], [True, False, False]])},
                        {"indices": torch.zeros(3, dtype=torch.long),
                         "old_logp": torch.tensor([np.log(.75), 0, 0]),
                         "old_log_probs": torch.tensor([[np.log(.75), np.log(.25), -1e9], [0., -1e9, -1e9], [0., -1e9, -1e9]]),
                         "returns": torch.zeros(3)}, [3])
        result = measure_replay(Model(), Cache(), full_distribution=True)
        q = torch.tensor([0., 1.], dtype=torch.float64).softmax(0).numpy()
        expected = sum(p * np.log(p / qi) for p, qi in zip([.75, .25], q))
        self.assertAlmostEqual(result["full_kl_free"], expected, places=7)
        self.assertAlmostEqual(result["full_kl"] * 3, expected, places=7)
        c = Cache()
        c.action_counts = np.ones(3, dtype=int)
        original = c.batch
        def forced(spans):
            batch, labels, lengths = original(spans)
            batch["logits"][:] = torch.tensor([0., -1e9, -1e9])
            batch["action_mask"][:] = torch.tensor([True, False, False])
            labels["old_log_probs"][:] = torch.tensor([0., -1e9, -1e9])
            labels["old_logp"].zero_()
            return batch, labels, lengths
        c.batch = forced
        self.assertIsNone(measure_replay(Model(), c, full_distribution=True)["full_kl_free"])

    def test_rejected_attempt_restores_empty_and_initialized_adam_in_place(self):
        # Policy equality alone misses momentum/counter drift after rejection.
        model = torch.nn.Linear(2, 2)
        model.register_buffer("counter", torch.zeros(1))
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        for initialized in (False, True):
            before = deepcopy(model.state_dict())
            states = deepcopy(optimizer.state_dict())
            refs = {p: {k: v for k, v in s.items() if torch.is_tensor(v)} for p, s in optimizer.state.items()}
            snap = AdamSnapshot(model, optimizer)
            rng = torch.get_rng_state().clone()
            optimizer.zero_grad(set_to_none=True)
            model(torch.ones(1, 2)).sum().backward()
            optimizer.step()
            model.counter.add_(1)
            torch.rand(4)
            snap.restore()
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            for k, v in model.state_dict().items():
                self.assertTrue(torch.equal(v, before[k]))
            restored = optimizer.state_dict()
            self.assertEqual(restored["param_groups"], states["param_groups"])
            self.assertEqual(restored["state"].keys(), states["state"].keys())
            for key, values in states["state"].items():
                for name, value in values.items():
                    self.assertTrue(torch.equal(restored["state"][key][name], value))
            for param, values in refs.items():
                for key, ref in values.items():
                    self.assertIs(optimizer.state[param][key], ref)
            optimizer.zero_grad(set_to_none=True)
            model(torch.ones(1, 2)).sum().backward()
            optimizer.step()
        require_finite_state(model, optimizer)
        with torch.no_grad():
            model.weight[0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            require_finite_state(model, optimizer)

    def test_accept_then_reject_preserves_first_step_and_rollout_accounting(self):
        # The normal run cannot reveal an incorrectly retained rejected step.
        config = deepcopy(self.config)
        config["ppo"]["valid_tokens_per_minibatch"] = 1
        episodes = [{"chosen": [0, 0, 0], "old_value": [0., 0., 0.], "result": {"reward": 1}}]
        class Backend:
            prefix_seconds = backward_seconds = 0.
            def __init__(self, model, *args):
                self.model = model
                self.replay = {"full_kl_free": 0.}
                self.calls = iter([.005, .02, .005])
            def precheck(self): return 0., 0.
            def backward(self, batch, count, entropy):
                self.model.weight.sum().backward()
                return count
            def record_gradient(self, norm): pass
            def kl(self):
                result = next(self.calls)
                self.last_replay = {"full_kl_free": result}
                return result
            def statistics(self, tokens): return {}, None, None
            def extras(self, seconds): return {}
        model = torch.nn.Linear(1, 1, bias=False)
        reference = deepcopy(model)
        opt = torch.optim.Adam(model.parameters(), lr=1e-5, eps=1e-5)
        ref_opt = torch.optim.Adam(reference.parameters(), lr=1e-5, eps=1e-5)
        reference.weight.sum().backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), .5)
        ref_opt.step()
        result = update(model, opt, episodes, config, 0, "cpu", random.Random(17), _backend=Backend)
        self.assertEqual((result["attempted_optimizer_steps"], result["optimizer_steps"], result["rejected_optimizer_steps"]), (2, 1, 1))
        self.assertEqual(result["stop_reason"], "rejected_full_kl")
        self.assertEqual(result["final_accepted_kl"], .005)
        self.assertTrue(torch.equal(model.weight, reference.weight))
        for a, b in zip(opt.state.values(), ref_opt.state.values()):
            for key in a:
                self.assertTrue(torch.equal(a[key], b[key]))
        state = dict(updates=0, attempted_steps=0, rejected_steps=0, accepted_updates=0,
                     zero_accepted_streak=0, first_accepted_evaluated=False, decisions=8192, active_seconds=10.)
        for number in range(1, 4):
            advance_controlled_rollout(state, {"optimizer_steps": 0, "attempted_optimizer_steps": 1, "rejected_optimizer_steps": 1}, number)
        self.assertEqual((state["updates"], state["zero_accepted_streak"], state["decisions"], state["active_seconds"]), (3, 3, 8192, 10.))
        advance_controlled_rollout(state, {"optimizer_steps": 1, "attempted_optimizer_steps": 1}, 4)
        self.assertEqual(state["zero_accepted_streak"], 0)
        validate_config(self.config)
        new = prepare_cases(self.config)
        old = read_json("artifacts/rl/ppo-after-imitation/manifest.json")
        self.assertEqual(new["development"], old["development"])
        self.assertEqual(new["confirmation"], old["confirmation"])

    def test_time_limit_still_finishes_attempt_kl_and_rollback(self):
        # A timeout after Adam must not bypass acceptance and leak an oversized step.
        clock = [0.]
        class Backend:
            prefix_seconds = backward_seconds = 0.
            def __init__(self, model, *args):
                self.model = model
                self.replay = {"full_kl_free": 0.}
                self.calls = 0
            def precheck(self): return 0., 0.
            def backward(self, spans, count, entropy):
                self.model.weight.sum().backward()
                clock[0] = 101.
                return count
            def kl(self):
                self.calls += 1
                value = .02 if self.calls == 1 else 0.
                self.last_replay = {"full_kl_free": value}
                return value
            def record_gradient(self, norm): pass
            def statistics(self, tokens): return {}, None, None
            def extras(self, seconds): return {}
        model = torch.nn.Linear(1, 1, bias=False)
        expected = model.weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters())
        episodes = [{"chosen": [0], "old_value": [0.], "result": {"reward": 1}}]
        with patch("civ5rl.engine.ppo.perf_counter", side_effect=lambda: clock[0]):
            result = update(model, optimizer, episodes, self.config, 0, "cpu",
                            random.Random(17), _backend=Backend, deadline=100.)
            before = update(model, optimizer, episodes, self.config, 0, "cpu",
                            random.Random(17), _backend=Backend, deadline=100.)
        self.assertEqual(result["stop_reason"], "active_time_limit")
        self.assertEqual((result["attempted_optimizer_steps"], result["optimizer_steps"]), (1, 0))
        self.assertEqual(before["attempted_optimizer_steps"], 0)
        self.assertTrue(torch.equal(model.weight, expected))
        self.assertFalse(optimizer.state)


if __name__ == "__main__":
    unittest.main()
