"""Resource accounting must not silently lose or double-apply live extensions."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from civ5rl.engine.budget import ActiveBudget, deadline_reached
from civ5rl.engine.control import RunLock
from civ5rl.provenance.common import read_json, write_json
from civ5rl.reporting.operations import extend_budget


class LiveBudgetTests(unittest.TestCase):
    def prepare(self, root, kind="rl01.ppo_after_target_bc"):
        cfg = {"experiment": {"active_seconds": 10}, "training_seconds": 10}
        write_json(root / "run.json", dict(kind=kind, run_id="r", config=cfg))
        write_json(root / "state.json", dict(active_seconds=4, updates=1, epoch=1, Adam_steps=7))
        write_json(root / "status.json", dict(status="training"))
        return cfg

    def test_live_writer_does_not_block_and_retry_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.prepare(root)
            original = (root / "state.json").read_bytes(), (root / "run.json").read_bytes()
            with RunLock(root):
                first = extend_budget(root, 60, "user", request_id="one")
                retry = extend_budget(root, 60, "user", request_id="one")
                second = extend_budget(root, 30, "again", request_id="two")
            self.assertFalse(first["applied"])
            self.assertEqual(first, retry)
            self.assertEqual(second["additional_active_seconds"], 90)
            self.assertEqual(original, ((root / "state.json").read_bytes(), (root / "run.json").read_bytes()))
            with self.assertRaises(ValueError):
                extend_budget(root, 10, "user", request_id="one")

    def test_limits_refresh_during_work_and_after_evaluation(self):
        kinds = ["rl01.ppo_after_target_bc", "rl01.cp_imitation_target_scale",
                 "rl01.cp_imitation_target_more_data", "rl01.cp_imitation_target_c2"]
        for kind in kinds:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                self.prepare(root, kind)
                with patch("civ5rl.engine.budget.monotonic", return_value=0) as tick:
                    budget = ActiveBudget(root)
                    self.assertEqual(budget.limit(), 10)
                    request = extend_budget(root, 20, "during update", request_id="one")
                    self.assertEqual(budget.limit(), 10)  # throttle is one second
                    tick.return_value = 2
                    with patch("civ5rl.engine.budget.perf_counter", return_value=15):
                        self.assertFalse(deadline_reached(budget.limit))
                    self.assertEqual(budget.limit(), 30)
                    receipt = read_json(root / "budget-applied.json")
                    self.assertEqual(receipt["revision"], request["requested_revision"])
                    # Evaluation changes no spent time. A second request remains
                    # pending until the next safe training check.
                    extend_budget(root, 5, "during evaluation", request_id="two")
                    tick.return_value = 4
                    self.assertEqual(budget.limit(), 35)

    def test_terminal_training_cannot_be_resurrected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.prepare(root)
            write_json(root / "state.json", dict(stop_reason="decision_limit"))
            with self.assertRaises(ValueError):
                extend_budget(root, 60, "too late")

    def test_extension_inside_ppo_update_preserves_adam_and_rng(self):
        import torch
        from civ5rl.engine.ppo import update
        from tests.test_rl_ppo_refactor import FakeBackend, ReverseRng

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.prepare(root)
            policy = torch.nn.Linear(1, 1, bias=False)
            optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
            rng = ReverseRng()
            config = dict(ppo=dict(sequence_length=2, sequences_per_minibatch=1, epochs=1,
                                   stop_kl=1, max_gradient_norm=10, entropy_coefficient=0,
                                   kl_measurement="whole_rollout_after_optimizer_step"), optimizer=dict(lr=0.01))
            episodes = [dict(chosen=[0] * 6, old_value=[0.] * 6, result=dict(reward=1))]
            torch_rng = torch.get_rng_state().clone()
            with patch("civ5rl.engine.budget.monotonic", return_value=0) as tick, \
                    patch("civ5rl.engine.budget.perf_counter", return_value=9) as elapsed:
                budget = ActiveBudget(root)

                def on_step(record):
                    if record["accepted_step"] == 1:
                        extend_budget(root, 20, "inside update", request_id="live")
                        tick.return_value = 2
                        elapsed.return_value = 15  # the original deadline passed

                result = update(policy, optimizer, episodes, config, 0, "cpu", rng,
                                _backend=FakeBackend, deadline=budget.limit, on_step=on_step)
            self.assertEqual(result["optimizer_steps"], 3)
            self.assertEqual({int(v["step"]) for v in optimizer.state.values()}, {3})
            self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))
            self.assertEqual(rng.calls, 1)


if __name__ == "__main__":
    unittest.main()
