"""Silent risks in the new critic handoff and economical stopping rules."""

import unittest
from copy import deepcopy

import numpy as np
import torch

from civ5rl.config import load_config
from civ5rl.core.model import Policy, batch_inputs
from civ5rl.engine.counters import TrainingCounters
from civ5rl.engine.ppo import encode_episode, targets
from civ5rl.engine.ppo_fast import RolloutCache
from civ5rl.experiments.ppo_after_imitation import (
    optimizer_for,
    train as ordinary_ppo_train,
)
from civ5rl.experiments.ppo_after_target_bc import (
    behavior_digest,
    heal_interrupted_update,
    prepare_critic,
)
from civ5rl.protocols.ppo_after_imitation import KIND as ORDINARY_PPO_KIND
from civ5rl.protocols.ppo_after_target_bc import (
    KIND,
    check_development_gain,
    record_collection,
    record_optimization,
)
from civ5rl.provenance import checkpoint
from civ5rl.provenance.audit import verify_ppo_ledger
from civ5rl.provenance.common import configure, write_json
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

    def test_rejected_and_accepted_counter_translation_and_interrupted_ledger_healing(
        self,
    ):
        import random
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            write_json(tmp / "run.json", {"run_id": "test_counters", "config": self.config})
            state = {
                **TrainingCounters().to_state_dict(protocol=KIND),
                "updates": 0,
                "decisions": 0,
                "winless_streak": 0,
                "zero_accepted_streak": 0,
                "stop_reason": None,
            }
            record_collection(state, {"decisions": 2048, "episodes": 4, "W": 1}, self.config)
            rejected_metrics = {
                "PPO_skipped": False,
                "optimizer_steps": 0,
                "accepted_optimizer_steps": 0,
                "attempted_optimizer_steps": 2,
                "rejected_optimizer_steps": 2,
                "optimizer_step_records": [
                    {"accepted": False, "attempted_kl": 0.03},
                    {"accepted": False, "attempted_kl": 0.025},
                ],
            }
            record_optimization(state, rejected_metrics, self.config)
            self.assertEqual(
                (
                    state["updates"],
                    state["rollouts_collected"],
                    state["rollouts_with_accepted_steps"],
                    state["adam_steps_attempted"],
                    state["adam_steps_accepted"],
                ),
                (1, 1, 0, 2, 0),
            )
            # Round-trip with updates=1 on a 0-accepted rollout must not inflate rollouts_with_accepted_steps.
            self.assertEqual(
                TrainingCounters.from_dict(state, protocol=KIND).rollouts_with_accepted_steps,
                0,
            )
            row = {
                "rollout": state["rollouts"],
                "decisions": 2048,
                "total_decisions": state["decisions"],
                "ppo": rejected_metrics,
            }
            state["pending_update"] = row
            write_json(tmp / "state.json", state)
            self.assertTrue(heal_interrupted_update(tmp, state))
            self.assertFalse(heal_interrupted_update(tmp, {**state, "pending_update": row}))
            audit_result = verify_ppo_ledger(tmp)
            self.assertEqual(
                audit_result["totals"],
                {
                    "updates": 1,
                    "decisions": 2048,
                    "accepted": 0,
                    "attempted": 2,
                    "rejected": 2,
                },
            )

            # Ordinary PPO train() without legacy attempted_steps/accepted_updates must advance counters and preserve updates on 0-step rollouts.
            ord_dir = tmp / "ordinary"
            (ord_dir / "checkpoints").mkdir(parents=True)
            ord_cfg = load_config("configs/rl-ppo-after-imitation.json")
            policy = Policy()
            optimizer = optimizer_for(policy, ord_cfg, "cpu")
            identity = {"run_id": "ord"}
            write_json(ord_dir / "passport.json", identity)
            write_json(ord_dir / "preflight.json", {"passed": True})
            write_json(ord_dir / "baseline-policy.json", {"hash": "dummy"})
            ord_state = {
                **TrainingCounters().to_state_dict(protocol=ORDINARY_PPO_KIND),
                "decisions": 0,
                "next_episode": 0,
                "active_seconds": 0.0,
                "evaluation_seconds": 0.0,
                "save_seconds": 0.0,
                "preparation_seconds": 0.0,
                "evaluated": {},
                "best": {"path": str(ord_dir / "checkpoints/initial.pt"), "wins": 1, "update": 0},
                "latest": "checkpoints/initial.pt",
                "pending_evaluation": False,
                "stop_reason": None,
            }
            checkpoint.save(
                ord_dir / "checkpoints/initial.pt",
                policy,
                optimizer,
                ord_state,
                random.Random(17),
                ord_cfg,
                identity,
            )
            write_json(ord_dir / "state.json", ord_state)
            dummy_ep = {
                "result": {"winner": 0, "reward": 1},
                "spec": {"learner_side": 0},
                "actions": [(1, 2)] * 10,
                "chosen": [0] * 10,
                "observations": [{}] * 10,
                "old_logp": [-0.5] * 10,
                "old_value": [0.1] * 10,
            }
            call_idx = [0]

            class FakeCollector:
                def __init__(self, *args, **kwargs):
                    pass

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

                def collect(self, *args, **kwargs):
                    call_idx[0] += 1
                    return [dummy_ep], call_idx[0], {"collection_seconds": 0.1}

            def fake_update(*args, **kwargs):
                steps = 0 if call_idx[0] == 2 else 4
                return {
                    "optimizer_steps": steps,
                    "attempted_optimizer_steps": 4,
                    "rejected_optimizer_steps": 4 - steps,
                    "stop_reason": "active_time_limit"
                    if call_idx[0] == 3
                    else "epochs_complete",
                    "update_complete": True,
                }

            plan = {"base": {"worlds": {"training": [10055]}, "maps": []}}
            with (
                patch(
                    "civ5rl.experiments.ppo_after_imitation.create_policy",
                    return_value=policy,
                ),
                patch(
                    "civ5rl.experiments.ppo_after_imitation.Collector", FakeCollector
                ),
                patch(
                    "civ5rl.experiments.ppo_after_imitation.update",
                    side_effect=fake_update,
                ),
                patch("civ5rl.experiments.ppo_after_imitation_checks.warmup"),
                patch(
                    "civ5rl.experiments.ppo_after_imitation.development",
                    side_effect=lambda cfg, pl, o, st: st.update(
                        pending_evaluation=False
                    ),
                ),
                patch(
                    "civ5rl.experiments.ppo_after_imitation.save_rollout",
                    return_value={"sha256": "abc"},
                ),
            ):
                final_state = ordinary_ppo_train(ord_cfg, plan, ord_dir, "cpu")

            self.assertEqual(
                (
                    final_state["rollouts_collected"],
                    final_state["rollouts_with_accepted_steps"],
                    final_state["adam_steps_attempted"],
                    final_state["adam_steps_accepted"],
                    final_state["optimizer_steps"],
                    final_state["updates"],
                ),
                (3, 2, 12, 8, 8, 3),
            )
            self.assertNotIn("accepted_updates", final_state)
            self.assertNotIn("attempted_steps", final_state)


