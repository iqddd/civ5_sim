"""Silent risks in the new critic handoff and economical stopping rules."""

import unittest
from copy import deepcopy

import numpy as np
import torch

from civ5rl.config import load_config
from civ5rl.core.model import Policy, batch_inputs
from civ5rl.engine.counters import TrainingCounters
from civ5rl.engine.ppo import (
    encode_episode,
    episode_returns,
    raw_gae_advantages,
    targets,
    update,
)
from civ5rl.engine.ppo_fast import BatchedBackend, RolloutCache
from civ5rl.engine.replay_metrics import StepReplay
from civ5rl.experiments.imitation_helpers import assert_equal
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
    compute_time_budget_check,
    load_config as load_target_bc_config,
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
            torch.testing.assert_close(policy(batch, sequence=True)[0], logits_before, rtol=0, atol=0)
        self.assertEqual(behavior_digest(episodes), original_behavior)
        refreshed = np.concatenate([ep["old_value"] for ep in episodes])
        self.assertFalse(np.array_equal(refreshed, original_values))
        cache = RolloutCache(episodes, "C", "cpu", resident=False, cpu_labels=True)
        np.testing.assert_array_equal(cache.labels["advantages"].numpy(), np.concatenate(targets(episodes)))
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
            self.assertTrue(record_collection(state, {"decisions": 8192, "W": 0}, self.config))
        self.assertFalse(record_collection(state, {"decisions": 8192, "W": 0}, self.config))
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
            legacy_zero_accepted = {"updates": 2, "optimizer_steps": 0}
            for proto in (KIND, "ppo_controlled_updates"):
                self.assertEqual(
                    TrainingCounters.from_dict(
                        legacy_zero_accepted, protocol=proto
                    ).rollouts_with_accepted_steps,
                    0,
                )
            conflicting = {
                "updates": 5,
                "rollout_number": 3,
                "optimizer_steps": 10,
                "steps_performed": 7,
                "attempted_steps": 14,
                "attempted_optimizer_steps": 12,
            }
            self.assertEqual(
                TrainingCounters.from_dict(conflicting, protocol=KIND),
                TrainingCounters(
                    rollouts_collected=3,
                    rollouts_with_accepted_steps=5,
                    adam_steps_attempted=14,
                    adam_steps_accepted=10,
                ),
            )
            self.assertEqual(
                TrainingCounters.from_dict(conflicting, protocol="ppo_controlled_updates"),
                TrainingCounters(
                    rollouts_collected=5,
                    rollouts_with_accepted_steps=5,
                    adam_steps_attempted=14,
                    adam_steps_accepted=10,
                ),
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

            # Ordinary PPO train() without legacy attempted_steps/accepted_updates
            # must advance counters and preserve updates on 0-step rollouts.
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
                    "stop_reason": "active_time_limit" if call_idx[0] == 3 else "epochs_complete",
                    "update_complete": True,
                }

            plan = {"base": {"worlds": {"training": [10055]}, "maps": []}}
            with (
                patch(
                    "civ5rl.experiments.ppo_after_imitation.create_policy",
                    return_value=policy,
                ),
                patch("civ5rl.experiments.ppo_after_imitation.Collector", FakeCollector),
                patch(
                    "civ5rl.experiments.ppo_after_imitation.update",
                    side_effect=fake_update,
                ),
                patch("civ5rl.experiments.ppo_after_imitation_checks.warmup"),
                patch(
                    "civ5rl.experiments.ppo_after_imitation.development",
                    side_effect=lambda cfg, pl, o, st: st.update(pending_evaluation=False),
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

    def test_discounted_returns_and_gae_math(self):
        ep_win = {
            "chosen": [0, 1, 0],
            "old_value": [0.2, 0.5, 0.8],
            "result": {"reward": 1.0},
        }
        ep_loss = {
            "chosen": [1, 0],
            "old_value": [0.4, 0.3],
            "result": {"reward": 0.0},
        }
        episodes = [ep_win, ep_loss]

        # Legacy gamma=1, lambda=1 exact equivalence
        mc_ret = episode_returns(episodes, gamma=1.0)
        np.testing.assert_allclose(mc_ret[0], [1.0, 1.0, 1.0], rtol=0, atol=0)
        np.testing.assert_allclose(mc_ret[1], [0.0, 0.0], rtol=0, atol=0)
        mc_adv = raw_gae_advantages(episodes, gamma=1.0, gae_lambda=1.0)
        np.testing.assert_allclose(mc_adv[0], [0.8, 0.5, 0.2], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(mc_adv[1], [-0.4, -0.3], rtol=1e-6, atol=1e-6)

        # Discounted returns and GAE(gamma=0.997, lambda=0.95)
        gamma, lam = 0.997, 0.95
        disc_ret = episode_returns(episodes, gamma=gamma)
        np.testing.assert_allclose(
            disc_ret[0],
            np.asarray([gamma**2, gamma**1, 1.0], dtype=np.float32),
            rtol=1e-6,
            atol=1e-7,
        )
        np.testing.assert_allclose(disc_ret[1], [0.0, 0.0], rtol=0, atol=0)

        # Hand-computed backward GAE for ep_win: V = [0.2, 0.5, 0.8], R = 1.0
        d2 = 1.0 + gamma * 0.0 - 0.8
        a2 = d2
        d1 = 0.0 + gamma * 0.8 - 0.5
        a1 = d1 + gamma * lam * a2
        d0 = 0.0 + gamma * 0.5 - 0.2
        a0 = d0 + gamma * lam * a1

        # Hand-computed backward GAE for ep_loss: V = [0.4, 0.3], R = 0.0
        ld1 = 0.0 + gamma * 0.0 - 0.3
        la1 = ld1
        ld0 = 0.0 + gamma * 0.3 - 0.4
        la0 = ld0 + gamma * lam * la1

        gae_adv = raw_gae_advantages(episodes, gamma=gamma, gae_lambda=lam)
        np.testing.assert_allclose(
            gae_adv[0], np.asarray([a0, a1, a2], dtype=np.float32), rtol=1e-6, atol=1e-7
        )
        np.testing.assert_allclose(
            gae_adv[1], np.asarray([la0, la1], dtype=np.float32), rtol=1e-6, atol=1e-7
        )

        # Propagation of gamma and gae_lambda through prepare_critic and RolloutCache
        policy = Policy()
        real_eps = self.episodes(policy)
        settings = {**self.config["critic"], "batch_size": 4}
        warmup = prepare_critic(
            policy, real_eps, settings, "cpu", gamma=gamma, gae_lambda=lam
        )
        expected_disc_returns = np.concatenate(episode_returns(real_eps, gamma=gamma))
        expected_raw_gae = np.concatenate(
            raw_gae_advantages(real_eps, gamma=gamma, gae_lambda=lam)
        )
        self.assertAlmostEqual(
            warmup["after"]["return_mean"], float(expected_disc_returns.mean()), places=6
        )
        self.assertNotAlmostEqual(
            warmup["after"]["return_mean"], 5.0 / 8.0, places=4
        )
        self.assertAlmostEqual(
            warmup["raw_advantage_std"], float(np.std(expected_raw_gae)), places=6
        )
        cache = RolloutCache(
            real_eps,
            "C",
            "cpu",
            resident=False,
            cpu_labels=True,
            gamma=gamma,
            gae_lambda=lam,
        )
        np.testing.assert_allclose(
            cache.labels["returns"].numpy(), expected_disc_returns, rtol=0, atol=0
        )
        np.testing.assert_allclose(
            cache.labels["advantages"].numpy(),
            np.concatenate(targets(real_eps, gamma=gamma, gae_lambda=lam)),
            rtol=0,
            atol=0,
        )

    def test_epoch_or_spike_kl_commit_and_rollback(self):
        import random
        import tempfile
        from pathlib import Path

        policy = Policy()
        episodes = self.episodes(policy)
        cfg = deepcopy(self.config)
        cfg["performance"]["compile"] = False
        cfg["ppo"].update(
            epochs=2,
            sequence_length=4,
            sequences_per_minibatch=1,
            sequences_per_microbatch=1,
            valid_tokens_per_minibatch=4,
            kl_measurement="whole_rollout_after_epoch_or_minibatch_spike",
            minibatch_kl_spike_threshold=0.01,
            target_kl=0.01,
            stop_kl=0.05,
            gamma=0.997,
            gae_lambda=0.95,
        )
        optimizer = optimizer_for(policy, cfg, "cpu")
        replay_calls = [0]

        class CountingBackend(BatchedBackend):
            def measure_step_replay(self):
                replay_calls[0] += 1
                return super().measure_step_replay()

        metrics = update(
            policy,
            optimizer,
            episodes,
            cfg,
            0,
            "cpu",
            random.Random(17),
            _backend=CountingBackend,
        )
        # 2 epochs * 2 minibatches = 4 steps, but only 2 full-rollout KL checks (1 per epoch).
        self.assertEqual(metrics["attempted_optimizer_steps"], 4)
        self.assertEqual(metrics["accepted_optimizer_steps"], 4)
        self.assertEqual(metrics["rejected_optimizer_steps"], 0)
        self.assertEqual(replay_calls[0], 2)
        self.assertEqual(
            [r["full_kl_checked"] for r in metrics["optimizer_step_records"]],
            [False, True, False, True],
        )
        self.assertEqual(
            [r["committed_accepted_steps"] for r in metrics["optimizer_step_records"]],
            [2, 2, 4, 4],
        )
        first_commit_record = [
            r for r in metrics["optimizer_step_records"] if r["full_kl_checked"]
        ][0]
        self.assertEqual(
            first_commit_record["accepted_step"],
            first_commit_record["committed_accepted_steps"],
        )

        # Full rollback at epoch end restores exact model & Adam state (including exp_avg/exp_avg_sq).
        saved_weights = deepcopy(policy.state_dict())
        saved_adam = deepcopy(optimizer.state_dict())
        strict_cfg = deepcopy(cfg)
        strict_cfg["ppo"]["stop_kl"] = 1e-12
        rollback_metrics = update(
            policy,
            optimizer,
            episodes,
            strict_cfg,
            0,
            "cpu",
            random.Random(19),
            _backend=CountingBackend,
        )
        self.assertEqual(rollback_metrics["stop_reason"], "rejected_full_kl")
        self.assertEqual(rollback_metrics["accepted_optimizer_steps"], 0)
        self.assertEqual(rollback_metrics["attempted_optimizer_steps"], 2)
        self.assertEqual(rollback_metrics["rejected_optimizer_steps"], 2)
        assert_equal(policy.state_dict(), saved_weights)
        assert_equal(optimizer.state_dict(), saved_adam)

        # Mid-epoch spike triggers immediate check, latches yellow_zone, and rolls back
        # step 4 to the intermediate snapshot committed after step 3.
        step_counter = [0]
        scripted_minibatch_kl = [0.002, 0.003, 0.015, 0.001]
        snapshots_after_step = {}

        class SpikeAndLatchBackend(BatchedBackend):
            def backward(self, minibatch, count, entropy_coefficient):
                tokens = super().backward(minibatch, count, entropy_coefficient)
                self.last_minibatch_approx_kl = scripted_minibatch_kl[step_counter[0]]
                step_counter[0] += 1
                return tokens

            def measure_step_replay(self):
                real = super().measure_step_replay()
                if step_counter[0] == 4 and "step4_rejected" not in snapshots_after_step:
                    snapshots_after_step["step4_rejected"] = True
                    return StepReplay(
                        acceptance_kl=cfg["ppo"]["stop_kl"] + 0.02,
                        diagnostics=real.diagnostics,
                    )
                return real

        def capture_intermediate(record):
            if record["accepted"] and record["full_kl_checked"] and record["step"] == 3:
                snapshots_after_step["weights_3"] = deepcopy(policy.state_dict())
                snapshots_after_step["adam_3"] = deepcopy(optimizer.state_dict())

        partial_metrics = update(
            policy,
            optimizer,
            episodes,
            cfg,
            0,
            "cpu",
            random.Random(23),
            on_step=capture_intermediate,
            _backend=SpikeAndLatchBackend,
        )
        self.assertEqual(partial_metrics["stop_reason"], "rejected_full_kl")
        self.assertEqual(partial_metrics["attempted_optimizer_steps"], 4)
        self.assertEqual(partial_metrics["accepted_optimizer_steps"], 3)
        self.assertEqual(partial_metrics["rejected_optimizer_steps"], 1)
        self.assertEqual(partial_metrics["epochs_completed"], 1)
        self.assertEqual(
            [r["full_kl_checked"] for r in partial_metrics["optimizer_step_records"]],
            [False, True, True, True],
        )
        self.assertEqual(
            [r["accepted"] for r in partial_metrics["optimizer_step_records"]],
            [True, True, True, False],
        )
        self.assertEqual(
            [
                r["committed_accepted_steps"]
                for r in partial_metrics["optimizer_step_records"]
            ],
            [2, 2, 3, 3],
        )
        assert_equal(policy.state_dict(), snapshots_after_step["weights_3"])
        assert_equal(optimizer.state_dict(), snapshots_after_step["adam_3"])
        self.assertFalse(
            all(
                torch.equal(policy.state_dict()[k], saved_weights[k])
                for k in saved_weights
            )
        )

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            write_json(tmp / "run.json", {"run_id": "kl_batch", "config": cfg})
            state = {
                **TrainingCounters(
                    rollouts_collected=3,
                    rollouts_with_accepted_steps=2,
                    adam_steps_attempted=10,
                    adam_steps_accepted=7,
                ).to_state_dict(protocol=KIND),
                "updates": 3,
                "decisions": 24,
                "rejected_steps": 3,
                "phase": "collect",
            }
            write_json(tmp / "state.json", state)
            rows = [
                {
                    "rollout": 1,
                    "decisions": 8,
                    "total_decisions": 8,
                    "ppo": {**metrics, "PPO_skipped": False},
                },
                {
                    "rollout": 2,
                    "decisions": 8,
                    "total_decisions": 16,
                    "ppo": {**rollback_metrics, "PPO_skipped": False},
                },
                {
                    "rollout": 3,
                    "decisions": 8,
                    "total_decisions": 24,
                    "ppo": {**partial_metrics, "PPO_skipped": False},
                },
            ]
            with (tmp / "updates.jsonl").open("w", encoding="utf-8") as f:
                import json

                for r in rows:
                    f.write(json.dumps(r) + "\n")
            verified = verify_ppo_ledger(tmp)
            self.assertEqual(verified["status"], "verified")
            self.assertEqual(verified["totals"]["accepted"], 7)
            self.assertEqual(verified["totals"]["rejected"], 3)

    def test_c2_h60_gae_config_and_early_gain_gate(self):
        c2_cfg = load_target_bc_config("configs/rl-ppo-c2-h60-gae.json")
        budget = compute_time_budget_check(c2_cfg)
        self.assertEqual(budget["nominal_optimizer_steps"], 16384)
        self.assertEqual(budget["partial_minibatch_steps_allowance"], 1024)
        self.assertEqual(budget["terminal_overshoot_steps_allowance"], 16)
        self.assertEqual(budget["planned_optimizer_steps_upper_bound"], 17424)
        self.assertEqual(budget["planned_kl_checks_upper_bound"], 1536)
        self.assertTrue(budget["fits_within_active_seconds"])
        over_budget = deepcopy(c2_cfg)
        over_budget["ppo"]["kl_measurement"] = "whole_rollout_after_optimizer_step"
        with self.assertRaises(ValueError):
            compute_time_budget_check(over_budget)
        state = {
            "decisions": 1048576,
            "baseline_wins": 193,
            "best": {"wins": 203},
            "development_points": {"1048576": "hash_1m"},
            "stop_reason": None,
        }
        # +10 wins on 512 matches (+1.95 pp < +2.0 pp) triggers early stop
        check_development_gain(state, c2_cfg)
        self.assertEqual(state["stop_reason"], "no_development_gain")
        # +11 wins on 512 matches (+2.15 pp >= +2.0 pp) continues
        state.update(stop_reason=None, best={"wins": 204})
        check_development_gain(state, c2_cfg)
        self.assertIsNone(state["stop_reason"])


if __name__ == "__main__":
    unittest.main()
