"""Horizon-only D0 change, frozen inputs, initialization, gates and resume."""

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from civ5rl.common import configure, read_json
from civ5rl.diagnostic import DecisionEnvContext, DiagnosticFactory
from civ5rl.diagnostic_one import gate_one, load_initial_weights, next_state_one
from civ5rl.model import Policy
from civ5rl.provenance import bound_cases
from civ5rl.validation import validate_config
from civ5sim import Action


class OneDecisionTests(unittest.TestCase):
    def setUp(self):
        configure(17)
        self.config = read_json("configs/rl-diagnostic-one-decision.json")
        path = Path("artifacts/rl/diagnostic-one-decision/manifests/manifest.json")
        if not path.exists():
            self.skipTest("Run --prepare-only for one-decision dataset")
        self.dataset = read_json(path)

    def test_all_dev_controls_and_no_opponent_after_first_command(self):
        for spec in bound_cases(self.dataset["D0"]["dev"]):
            for kind in ("ranged", "end_turn", "move", "skip", "fortify"):
                with DecisionEnvContext(spec) as env:
                    state = env.request()
                    action = next((a for a in state["actions"] if a.kind == kind), None)
                    if action is None:
                        continue
                    env.prepare(action)
                    result = env.battle.step(action)
                    env.applied(action, result)
                    count = env.commands
                    with patch.object(
                        env.opponent,
                        "act",
                        side_effect=AssertionError("opponent acted after horizon"),
                    ):
                        final = env.request()
                        again = env.request()
                    self.assertTrue(final["done"])
                    self.assertTrue(again["done"])
                    self.assertEqual(
                        final["result"]["reward"], 1 if kind == "ranged" else 0
                    )
                    self.assertEqual(
                        final["result"]["reason"],
                        "army_destroyed"
                        if kind == "ranged"
                        else "diagnostic_decision_limit",
                    )
                    self.assertEqual(env.commands, count)
                    self.assertEqual(env.learner_decisions, 1)

    def test_horizon_rejects_wrong_protocol_and_technical_errors_are_errors(self):
        spec = bound_cases(self.dataset["D0"]["dev"])[0]
        bad = deepcopy(spec)
        bad.pop("diagnostic_protocol")
        with self.assertRaises(ValueError):
            DecisionEnvContext(bad)
        with DecisionEnvContext(spec) as env:
            env.request()
            action = Action("ranged", 999, 999)
            result = env.battle.step(action)
            with self.assertRaises(RuntimeError):
                env.applied(action, result)
            self.assertEqual(env.learner_decisions, 0)

    def test_inputs_manifests_and_original_initial_weights(self):
        validate_config(self.config)
        old = read_json("artifacts/rl/diagnostic-results/manifests/manifest.json")
        for split in ("train", "dev"):
            self.assertNotEqual(
                self.dataset["D0"][split]["hash"], old["D0"][split]["hash"]
            )
            for a, b in zip(
                old["D0"][split]["cases"], self.dataset["D0"][split]["cases"]
            ):
                self.assertEqual(
                    a,
                    {
                        k: v
                        for k, v in b.items()
                        if k not in ("learner_decision_limit", "diagnostic_protocol")
                    },
                )
        policy = Policy()
        proof = load_initial_weights(policy, self.config)
        self.assertTrue(proof["all_model_parameters_bitwise_equal"])
        source = torch.load(
            self.config["diagnostic"]["initial_checkpoint"],
            map_location="cpu",
            weights_only=False,
        )
        for section in ("model", "optimizer", "ppo", "numerics", "collection"):
            self.assertEqual(self.config[section], source["config"][section])
        for k, v in policy.state_dict().items():
            torch.testing.assert_close(v, source["model"][k], rtol=0, atol=0)
        optimizer = torch.optim.Adam(policy.parameters())
        self.assertEqual(len(optimizer.state), 0)

    def test_gates_and_eight_update_limit(self):
        self.assertFalse(gate_one({"games": 128, "wins": 121})["passed"])
        self.assertTrue(gate_one({"games": 128, "wins": 122})["passed"])
        with self.assertRaises(ValueError):
            gate_one({"games": 127, "wins": 127})
        state = {
            "stage": "D0",
            "stage_updates": 1,
            "consecutive": 0,
            "status": "running",
        }
        first = next_state_one(state, True)
        self.assertEqual(first["status"], "running")
        reset = next_state_one(first, False)
        self.assertEqual(reset["consecutive"], 0)
        self.assertEqual(next_state_one(first, True)["status"], "passed")
        self.assertEqual(
            next_state_one({**state, "stage_updates": 8}, False)["status"], "failed"
        )
        self.assertEqual(
            next_state_one({**state, "stage_updates": 8, "consecutive": 1}, True)[
                "status"
            ],
            "passed",
        )

    def test_collector_one_token_labels_zero_memory_and_ppo_counts(self):
        import random

        from civ5rl.collector import Collector
        from civ5rl.ppo_fast import RolloutCache, update_fast

        config = deepcopy(self.config)
        config["collection"].update(
            num_envs=4,
            cpu_workers=2,
            envs_per_worker=2,
            min_learner_decisions_per_update=16,
        )
        config["ppo"].update(epochs=1)
        policy = Policy().eval()
        load_initial_weights(policy, config)
        factory = DiagnosticFactory(self.dataset, 17)
        with (
            tempfile.TemporaryDirectory() as tmp,
            Collector(config, factory, "cpu", Path(tmp)) as c,
        ):
            episodes, _, _ = c.collect(policy, "D0", 0)
        self.assertEqual(len(episodes), 16)
        for ep in episodes:
            for key in (
                "observations",
                "actions",
                "chosen",
                "old_logp",
                "old_log_probs",
                "old_value",
                "encoded",
            ):
                self.assertEqual(len(ep[key]), 1)
            self.assertEqual(ep["old_hidden"], [None])
            self.assertEqual(ep["result"]["learner_decisions"], 1)
        cache = RolloutCache(episodes, "C", "cpu")
        np.testing.assert_array_equal(
            cache.labels["returns"].numpy(), [ep["result"]["reward"] for ep in episodes]
        )
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.0003)
        stats = update_fast(
            policy, optimizer, episodes, config, 0, "cpu", random.Random(18)
        )
        self.assertEqual(stats["optimizer_steps"], 2)
        self.assertEqual(stats["objective_valid_tokens"], 16)
        self.assertTrue(all(int(x["step"]) == 2 for x in optimizer.state.values()))

    def test_one_decision_pending_evaluation_resume_does_not_update_again(self):
        from tools import run_rl_diagnostic as runner

        config = deepcopy(self.config)
        config["collection"].update(
            num_envs=2,
            cpu_workers=1,
            envs_per_worker=2,
            min_learner_decisions_per_update=8,
        )
        config["ppo"].update(epochs=1)
        evaluate = runner.evaluate_stage
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seed-17"

            def interrupted(*args):
                if Path(args[4]).name.startswith("update-"):
                    raise RuntimeError("pending evaluation interruption")
                return evaluate(*args)

            with (
                patch.object(runner, "evaluate_stage", side_effect=interrupted),
                self.assertRaisesRegex(RuntimeError, "pending evaluation"),
            ):
                runner.run_seed(config, self.dataset, path, 17, device="cpu")
            latest = read_json(path / "latest.json")
            self.assertEqual(latest["update"], 1)
            before = torch.load(
                path / "checkpoints" / latest["checkpoint"], weights_only=False
            )
            (path / "stop.request").touch()
            with patch(
                "civ5rl.ppo.update", side_effect=AssertionError("repeat training")
            ):
                result = runner.run_seed(
                    config, self.dataset, path, 17, device="cpu", resume=True
                )
            self.assertEqual(result["status"], "paused")
            self.assertEqual(result["update"], 1)
            self.assertIsNone(result["pending_evaluation"])
            after = torch.load(
                path / "checkpoints" / latest["checkpoint"], weights_only=False
            )
            for k, v in before["model"].items():
                torch.testing.assert_close(v, after["model"][k], rtol=0, atol=0)
            self.assertEqual(len((path / "metrics.jsonl").read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
