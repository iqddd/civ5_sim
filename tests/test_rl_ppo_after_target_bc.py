"""Silent risks in the new critic handoff and economical stopping rules."""

import unittest
from copy import deepcopy

import numpy as np
import torch

from civ5rl.config import load_config
from civ5rl.core.model import Policy, batch_inputs
from civ5rl.engine.ppo import encode_episode, targets
from civ5rl.engine.ppo_fast import RolloutCache
from civ5rl.experiments.ppo_after_imitation import optimizer_for
from civ5rl.experiments.ppo_after_target_bc import behavior_digest, prepare_critic
from civ5rl.provenance.common import configure
from civ5rl.protocols.ppo_after_target_bc import (
    check_development_gain,
    record_collection,
    record_optimization,
)
from civ5sim import Battle
from test_rl import context, scenario


class TargetBCPPOTests(unittest.TestCase):
    def setUp(self):
        configure(17)
        self.config = load_config("configs/rl-ppo-after-target-bc.json")

    def episodes(self, policy):
        initial = scenario()
        with Battle(initial) as battle:
            observation = battle.observe(0)
            actions = tuple(battle.available_actions())
        episodes = []
        for length, reward in ((5, 1), (3, 0)):
            episode = {
                "spec": {
                    "scenario": initial,
                    "armies": context(initial)["armies"],
                    "learner_side": 0,
                    "max_rounds": 3,
                },
                "observations": [observation] * length,
                "actions": [actions] * length,
                "chosen": [0] * length,
                "old_value": [0.0] * length,
                "result": {"reward": reward},
            }
            encoded = encode_episode(episode, "C")
            with torch.no_grad():
                logits, values, _ = policy(batch_inputs(encoded, "cpu"), sequence=True)
            distributions = logits.log_softmax(-1).numpy()
            episode["old_log_probs"] = [row.copy() for row in distributions]
            episode["old_logp"] = distributions[:, 0].tolist()
            episode["old_value"] = values.tolist()
            episodes.append(episode)
        return episodes

    def test_value_only_fit_refreshes_ppo_inputs_without_optimizer_or_behavior_leak(
        self,
    ):
        # Training can run successfully while silently using stale BC values/moments.
        policy = Policy()
        episodes = self.episodes(policy)
        before = deepcopy(policy.state_dict())
        original_behavior = behavior_digest(episodes)
        original_values = np.concatenate([ep["old_value"] for ep in episodes])
        batch = batch_inputs(encode_episode(episodes[0], "C"), "cpu")
        with torch.no_grad():
            logits_before = policy(batch, sequence=True)[0]
        settings = {**self.config["critic"], "batch_size": 4}
        result = prepare_critic(policy, episodes, settings, "cpu")
        self.assertEqual((result["epochs_completed"], result["steps"]), (2, 4))
        for name, value in policy.state_dict().items():
            if not name.startswith("value."):
                self.assertTrue(torch.equal(value, before[name]), name)
        with torch.no_grad():
            torch.testing.assert_close(
                policy(batch, sequence=True)[0], logits_before, rtol=0, atol=0
            )
        self.assertEqual(behavior_digest(episodes), original_behavior)
        refreshed = np.concatenate([ep["old_value"] for ep in episodes])
        self.assertFalse(np.array_equal(refreshed, original_values))
        cache = RolloutCache(episodes, "C", "cpu", resident=False, cpu_labels=True)
        np.testing.assert_array_equal(
            cache.labels["advantages"].numpy(), np.concatenate(targets(episodes))
        )
        optimizer = optimizer_for(policy, self.config, "cpu")
        self.assertEqual(len(optimizer.state), 0)

    def test_winless_skip_and_optimization_streak_are_independent(self):
        state = dict(
            rollouts=0,
            decisions=0,
            winless_streak=0,
            zero_accepted_streak=2,
            updates=0,
            optimizer_steps=0,
            attempted_steps=0,
            rejected_steps=0,
            stop_reason=None,
        )
        for _ in range(4):
            self.assertTrue(
                record_collection(state, {"decisions": 8192, "W": 0}, self.config)
            )
        self.assertFalse(
            record_collection(state, {"decisions": 8192, "W": 0}, self.config)
        )
        self.assertEqual(state["decisions"], 40960)
        self.assertEqual(state["winless_streak"], 5)
        self.assertEqual(state["zero_accepted_streak"], 2)
        self.assertEqual(state["updates"], 0)
        self.assertEqual(state["stop_reason"], "no_sampling_wins")
        state["stop_reason"] = None
        record_optimization(
            state,
            dict(
                optimizer_steps=0,
                attempted_optimizer_steps=1,
                rejected_optimizer_steps=1,
            ),
            self.config,
        )
        self.assertEqual(state["stop_reason"], "no_accepted_updates")

    def test_two_development_points_required_and_one_win_allows_continuation(self):
        state = dict(
            decisions=524288,
            baseline_wins=27,
            best={"wins": 27},
            development_points={"262144": "first"},
            stop_reason=None,
        )
        check_development_gain(state, self.config)
        self.assertIsNone(state["stop_reason"])
        state["development_points"]["524288"] = "second"
        check_development_gain(state, self.config)
        self.assertEqual(state["stop_reason"], "no_development_gain")
        state.update(stop_reason=None, best={"wins": 28})
        check_development_gain(state, self.config)
        self.assertIsNone(state["stop_reason"])
