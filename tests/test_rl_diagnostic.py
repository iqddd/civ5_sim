"""Diagnostic prerequisites, gates, provenance and attack metrics."""

import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from civ5rl.config import load_config
from civ5rl.common import configure, read_json
from civ5rl.diagnostic import (
    AttackOrEndTurn,
    gate,
    may_start_update,
    next_state,
    physical_hash,
    reference_win,
)
from civ5rl.diagnostic_metrics import decision_metrics, summarize_decisions
from civ5rl.ppo import objective
from civ5rl.validation import validate_config

from civ5rl.provenance import bound_cases, case_manifest, validate_cases
from civ5sim import Action


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        configure(17)
        self.config = load_config("configs/rl-diagnostic.json")

    def dataset(self):
        path = Path("artifacts/rl/diagnostic-results/manifests/manifest.json")
        if not path.exists():
            self.skipTest("Diagnostic corpus not prepared")
        return read_json(path)

    def test_loss_legacy_values_exact_and_config_values_change_gradients(self):
        logits = torch.tensor([[0.7, -0.5], [0.4, 0.3]], requires_grad=True)
        value = torch.tensor([0.2, -0.1], requires_grad=True)
        indices = torch.tensor([0, 1])
        old = torch.tensor([-2.0, -0.2])
        returns = torch.tensor([1.0, -1.0])
        adv = torch.tensor([1.0, -1.0])
        loss, stats = objective(logits, value, indices, old, returns, adv, 0.01)
        lp = logits.log_softmax(-1)
        p = lp.exp()
        ratio = (lp[range(2), indices] - old).exp()
        reference = (
            -torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()
            + 0.5 * (value - returns).square().mean()
            - 0.01 * (-(p * lp).sum(-1) / np.log(2)).mean()
        )
        torch.testing.assert_close(loss, reference, rtol=0, atol=0)
        a = torch.autograd.grad(loss, (logits, value), retain_graph=True)
        b = torch.autograd.grad(reference, (logits, value), retain_graph=True)
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        changed, other = objective(
            logits,
            value,
            indices,
            old,
            returns,
            adv,
            0.01,
            clip_ratio=0.7,
            value_coefficient=0.1,
        )
        c = torch.autograd.grad(changed, (logits, value))
        self.assertFalse(torch.equal(a[1], c[1]))
        self.assertNotEqual(loss.item(), changed.item())
        self.assertEqual(
            stats["clip_fraction"].item(),
            ((ratio - 1).abs() > 0.2).float().mean().item(),
        )
        self.assertEqual(
            other["clip_fraction"].item(),
            ((ratio - 1).abs() > 0.7).float().mean().item(),
        )

    def test_config_rejects_unsupported_before_workers(self):
        validate_config(self.config)
        for section, key, value in (
            ("model", "hex_blocks", 5),
            ("reward", "win", 2),
            ("ppo", "gamma", 0.99),
            ("ppo", "gae_lambda", 0.95),
            ("ppo", "value_clip", True),
            ("numerics", "gru_precision", "bfloat16"),
            ("ppo", "advantage_epsilon", 0.1),
        ):
            config = deepcopy(self.config)
            config[section][key] = value
            from civ5rl.collector import Collector

            with (
                patch(
                    "multiprocessing.get_context",
                    side_effect=AssertionError("worker creation reached"),
                ),
                self.assertRaisesRegex(ValueError, section + "." + key),
            ):
                Collector(config, None, "cpu", Path("unused"))

    def test_provenance_rejects_spoofs_and_closed_manifests(self):
        d = self.dataset()
        m = d["D0"]["dev"]
        cases = bound_cases(m)
        validate_cases(cases, m)
        for mutate in (
            lambda s: s["scenario"]["units"][0].update(hp=50),
            lambda s: s["provenance"].update(split="test"),
            lambda s: s.update(split="test"),
            lambda s: s.pop("provenance"),
        ):
            spec = deepcopy(cases[0])
            mutate(spec)
            with self.assertRaises(ValueError):
                validate_cases([spec], m)
        with self.assertRaises(ValueError):
            validate_cases(cases, None)
        for split in ("test", "diagnostic_test"):
            with self.assertRaises(ValueError):
                case_manifest(m["cases"], source="x", split=split, namespace="x")
            from civ5rl.evaluation import require_split

            with self.assertRaises(ValueError):
                require_split(
                    self.config, split, mode="final", frozen_plan={"allowed": True}
                )
            from civ5rl.evaluation import evaluate

            with self.assertRaisesRegex(ValueError, "Closed split"):
                evaluate(
                    None, object(), [{"split": split}], self.config, Path("unused")
                )
        from civ5rl.collector import Collector

        # The check happens before touching the policy or worker pipes.
        with self.assertRaises(ValueError):
            Collector.collect(object(), None, None, 0, fixed_specs=cases)

    def test_admission_and_train_dev_independence(self):
        d = self.dataset()
        for stage in ("D0", "D1"):
            train = d[stage]["train"]["cases"]
            dev = d[stage]["dev"]["cases"]
            self.assertEqual(len(dev), 128)
            self.assertEqual(len({s["physical_hash"] for s in dev}), 64)
            self.assertFalse(
                {s["physical_hash"] for s in train} & {s["physical_hash"] for s in dev}
            )
            for spec in (dev[0], dev[1], train[0], train[1]):
                self.assertTrue(reference_win(spec))
                from civ5rl.env import DecisionEnv

                env = DecisionEnv(spec)
                try:
                    state = env.request()
                    self.assertTrue(any(a.kind == "ranged" for a in state["actions"]))
                finally:
                    env.close()
            scenario = dev[0]["scenario"]
            swapped = deepcopy(scenario)
            swapped["seed"] += 1
            swapped["units"].reverse()
            for u in swapped["units"]:
                u["side"] ^= 1
            self.assertEqual(physical_hash(scenario), physical_hash(swapped))
        from civ5rl.diagnostic import d0_certificate

        for spec in d["D0"]["dev"]["cases"]:
            proof = d0_certificate(spec)
            self.assertTrue(proof)
            self.assertTrue(
                all(
                    p["minimum_damage"] >= 1 and p["rolls_checked"] == 1200
                    for p in proof
                )
            )

    def test_attack_opponent_and_diagnostics(self):
        agent = AttackOrEndTurn()
        agent.reset({"seed": 17})
        actions = (
            Action("move", 0, 5),
            Action("ranged", 0, 1),
            Action("melee", 1, 2),
            Action("end_turn"),
        )
        choices = [agent.act({}, actions) for _ in range(100)]
        self.assertEqual(set(choices), set(actions[1:3]))
        self.assertEqual(agent.act({}, (actions[0], actions[3])), Action("end_turn"))
        row = decision_metrics(actions, 3, np.array([0.2, 0.3, 0.4, 0.1]))
        summary = summarize_decisions([{"decision_metrics": [row]}])
        self.assertAlmostEqual(summary["attack"]["probability_when_available"], 0.7)
        self.assertEqual(summary["attack"]["chosen"], 0)
        self.assertEqual(summary["end_turn_when_attack_available"], 1)

    def test_gates_consecutive_transitions_limits_and_bootstrap(self):
        self.assertFalse(
            gate("D0", {"win_rate": 0.95, "first_winning_attack_rate": 0.89}, [])[
                "passed"
            ]
        )
        state = {
            "status": "running",
            "stage": "D0",
            "stage_updates": 1,
            "consecutive": 0,
            "training_seconds": 0.0,
            "pending_evaluation": None,
        }
        one = next_state(state, True)
        self.assertEqual(one["stage"], "D0")
        reset = next_state(one, False)
        self.assertEqual(reset["consecutive"], 0)
        two = next_state(one, True)
        self.assertEqual(two["stage"], "D1")
        self.assertEqual(two["stage_updates"], 0)
        self.assertEqual(
            next_state({**state, "stage_updates": 4}, False)["status"], "failed"
        )
        self.assertFalse(may_start_update({**state, "training_seconds": 1800}))
        self.assertFalse(may_start_update(state, stopping=True))
        self.assertFalse(
            may_start_update({**state, "pending_evaluation": {"stage": "D0"}})
        )
        rows = [
            {"layout_id": i // 2, "side": i % 2, "win": i < 104} for i in range(128)
        ]
        baseline = [{**r, "win": i < 60} for i, r in enumerate(rows)]
        result = gate("D1", {"win_rate": 104 / 128}, rows, baseline)
        self.assertTrue(result["passed"])
        self.assertGreater(result["paired_bootstrap_ci95"][0], 0)
        self.assertEqual(result, gate("D1", {"win_rate": 104 / 128}, rows, baseline))
        with self.assertRaises(ValueError):
            gate("D1", {"win_rate": 1}, rows[:-1], baseline)

    def test_actual_damage_caps_at_remaining_target_HP(self):
        from civ5rl.env import DecisionEnv

        spec = bound_cases(self.dataset()["D0"]["dev"])[0]
        env = DecisionEnv(spec)
        try:
            state = env.request()
            shot = next(a for a in state["actions"] if a.kind == "ranged")
            env.prepare(shot)
            env.applied(shot, env.battle.step(shot))
            done = env.request()
            self.assertTrue(done["done"])
            self.assertEqual(done["result"]["reward"], 1)
            self.assertEqual(done["result"]["damage_dealt"][spec["learner_side"]], 1)
        finally:
            env.close()

    def test_winning_shot_reward_pushes_probability_in_correct_direction(self):
        from civ5rl.env import DecisionEnv

        spec = bound_cases(self.dataset()["D0"]["dev"])[0]
        env = DecisionEnv(spec)
        try:
            state = env.request()
            index = next(
                i for i, a in enumerate(state["actions"]) if a.kind == "ranged"
            )
            action = state["actions"][index]
            env.prepare(action)
            env.applied(action, env.battle.step(action))
            reward = env.request()["result"]["reward"]
        finally:
            env.close()
        logits = torch.zeros((1, len(state["actions"])), requires_grad=True)
        old = logits.detach().log_softmax(-1)[:, index]
        loss, _ = objective(
            logits,
            torch.zeros(1),
            torch.tensor([index]),
            old,
            torch.tensor([float(reward)]),
            torch.tensor([float(reward)]),
            0.0,
        )
        loss.backward()
        self.assertEqual(reward, 1)
        self.assertLess(logits.grad[0, index], 0)
        updated = logits.detach() - 0.1 * logits.grad
        self.assertGreater(updated.softmax(-1)[0, index], logits.softmax(-1)[0, index])


if __name__ == "__main__":
    unittest.main()
