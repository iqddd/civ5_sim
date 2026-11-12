"""Catch silent Adam loss/repeated preparation and changed behavior in gamma=1."""

import random
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from civ5rl.core.model import Policy, batch_inputs
from civ5rl.engine.ppo import episode_returns, raw_gae_advantages, update
from civ5rl.engine.ppo_adam import check_counters, grouped_optimizer
from civ5rl.engine.ppo_fast import BatchedBackend
from civ5rl.experiments import ppo_after_target_bc as runner
from civ5rl.experiments.imitation_helpers import assert_equal
from civ5rl.experiments.ppo_after_imitation import optimizer_for
from civ5rl.experiments.ppo_critic_adaptation import adapt_critic
from civ5rl.protocols.ppo_after_target_bc import load_config, validate_profile
from civ5rl.provenance import checkpoint
from civ5rl.provenance.audit import verify_checkpoint_chain
from civ5rl.provenance.common import configure, read_json, write_json
import test_rl_ppo_continuation as continuation


class GammaOneTests(unittest.TestCase):
    def test_failed_extraction_does_not_renew_preparation_budget(self):
        # A compile failure before the first feature yield otherwise silently
        # grants another 300s on resume, despite the cumulative fit limit.
        cfg = load_config("configs/rl-ppo-c2-h60-gae-gamma1.json")
        policy = Policy()
        optimizer = grouped_optimizer(policy, cfg, "cpu")
        episodes = [dict(chosen=[0], old_value=[0.0], result=dict(reward=1))]
        state = dict(active_seconds=0.0, origin_Adam_counter=0, optimizer_steps=0)
        current = [0.0]

        def interrupted(*args):
            current[0] += 250
            raise RuntimeError("compile interrupted")
            yield

        def resumed(*args):
            current[0] += 60
            yield 0, 0, torch.zeros((1, policy.hidden_size))

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_json(out / "run.json", dict(config=cfg, kind=cfg["kind"], run_id="budget-test"))
            clock = runner.ActiveClock(state, out)
            kwargs = dict(behavior_hash=lambda _: "fixed", statistics=runner.critic_statistics,
                          policy_hash=runner.policy_hash)
            from civ5rl.experiments import ppo_critic_adaptation as adaptation
            with patch.object(adaptation, "perf_counter", side_effect=lambda: current[0]):
                with self.assertRaisesRegex(RuntimeError, "compile interrupted"):
                    adapt_critic(policy, optimizer, episodes, cfg, "cpu", out, state,
                                 random.Random(1), {}, clock, None, features_for=interrupted, **kwargs)
                self.assertEqual(read_json(out / "state.json")["critic_fit_seconds"], 250)
                result = adapt_critic(policy, optimizer, episodes, cfg, "cpu", out, state,
                                      random.Random(1), {}, clock, None, features_for=resumed, **kwargs)
            self.assertEqual(result["steps"], 0)
            self.assertEqual(result["fit_seconds"], 310)

    def test_undiscounted_returns_and_lambda_recursion_include_forced(self):
        # gamma=1 must not silently imply lambda=1 or skip forced transitions.
        episode = dict(chosen=[0, 0, 0], actions=[[0, 1], [0], [0, 1]],
                       old_value=[0.2, 0.3, 0.4], result=dict(reward=1))
        np.testing.assert_array_equal(episode_returns([episode], gamma=1)[0], [1, 1, 1])
        expected = [0.1 + 0.95 * (0.1 + 0.95 * 0.6), 0.1 + 0.95 * 0.6, 0.6]
        np.testing.assert_allclose(raw_gae_advantages([episode], gamma=1, gae_lambda=0.95)[0], expected)
        validate_profile(load_config("configs/rl-ppo-c2-h60-gae-gamma1.json"))

    def test_adaptation_resume_then_ppo_rollback_and_checkpoint(self):
        # A successful run could silently discard shared moments, refit twice,
        # or restore only one group's Adam step after rejecting a PPO update.
        configure(17)
        cfg = load_config("configs/rl-ppo-c2-h60-gae-gamma1.json")
        cfg["model"]["variant"] = "C"
        cfg["performance"].update(compile=False, resident_rollout=False)
        cfg["critic"].update(batch_size=4, epochs=2)
        cfg["ppo"].update(epochs=1, sequence_length=4, valid_tokens_per_minibatch=8,
                          sequences_per_microbatch=1, stop_kl=0.05)
        policy = Policy()
        parent_adam = optimizer_for(policy, cfg, "cpu")
        for p in policy.parameters():
            p.grad = torch.full_like(p, 0.01)
        parent_adam.step()
        parent_adam.zero_grad(set_to_none=True)
        for item in parent_adam.state.values():
            item["step"].fill_(12644)
        parent_adam.param_groups[0]["foreach"] = True
        saved = dict(model=deepcopy(policy.state_dict()), optimizer=deepcopy(parent_adam.state_dict()),
                     variant="C", state=dict(optimizer_steps=12644, critic_complete=True, phase="collect"))
        adam = runner.load_initial_optimizer(policy, saved, cfg)
        self.assertTrue(all(g["foreach"] for g in adam.param_groups))
        shared = {n: deepcopy(adam.state[p]) for n, p in policy.named_parameters() if not n.startswith("value.")}
        self.assertTrue(all(p not in adam.state for p in policy.value.parameters()))
        episodes = continuation.PPOContinuationTests.episodes(self, policy)
        original_behavior = runner.behavior_digest(episodes)
        original_weights = deepcopy(policy.state_dict())
        state = dict(optimizer_steps=0, attempted_steps=0, rejected_steps=0, updates=0,
                     phase="critic", decisions=8, origin_Adam_counter=12644,
                     active_seconds=0.0, latest="checkpoints/pending.pt")
        identity = dict(run_id="gamma1-test")
        rng = random.Random(789)
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            write_json(out / "run.json", dict(config=cfg, kind=cfg["kind"], run_id=identity["run_id"]))
            clock = runner.ActiveClock(state, out)
            calls = dict(features_for=runner.recurrent_features, behavior_hash=runner.behavior_digest,
                         statistics=runner.critic_statistics, policy_hash=runner.policy_hash)
            from civ5rl.experiments import ppo_critic_adaptation as adaptation
            real_save = adaptation.atomic_save

            def fail_after_first_commit(value, path):
                real_save(value, path)
                if Path(path).name == "critic-adaptation.pt" and value["progress"]["steps"] == 1:
                    raise RuntimeError("simulated interruption after durable batch")

            with patch.object(adaptation, "atomic_save", side_effect=fail_after_first_commit):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    adapt_critic(policy, adam, episodes, cfg, "cpu", out, state, rng,
                                 identity, clock, None, **calls)
            # Simulate reloading the earlier full boundary; the capsule must
            # restore its own value weights, Adam and RNG before continuing.
            policy.load_state_dict(saved["model"])
            adam = runner.load_initial_optimizer(policy, saved, cfg)
            result = adapt_critic(policy, adam, episodes, cfg, "cpu", out, state, rng,
                                  identity, clock, None, **calls)
            self.assertEqual(result["steps"], 4)
            self.assertEqual(result["positions_processed"], 16)
            self.assertEqual(check_counters(adam, state), dict(shared_actor=12644, value=4))
            for n, p in policy.named_parameters():
                if not n.startswith("value."):
                    assert_equal(adam.state[p], shared[n])
                    self.assertTrue(torch.equal(p, original_weights[n]))
            self.assertEqual(runner.behavior_digest(episodes), original_behavior)
            self.assertEqual(adam.param_groups[1]["lr"], 1e-5)
            # Refresh must actually use the adapted value head.
            encoded = runner.encode_episode(episodes[0], "C")
            with torch.no_grad():
                logits_after, values, _ = policy(batch_inputs(encoded, "cpu"), sequence=True)
            np.testing.assert_allclose(episodes[0]["old_value"], values.numpy(), atol=2e-7)
            parent_policy = Policy()
            parent_policy.load_state_dict(saved["model"])
            with torch.no_grad():
                logits_before = parent_policy(batch_inputs(encoded, "cpu"), sequence=True)[0]
            torch.testing.assert_close(logits_after, logits_before, rtol=0, atol=0)
            metrics = update(policy, adam, episodes, cfg, 0, "cpu", rng, _backend=BatchedBackend)
            self.assertEqual(metrics["optimizer_steps"], 1)
            state.update(optimizer_steps=1, attempted_steps=1, updates=1, phase="collect",
                         latest="checkpoints/update-0001.pt")
            self.assertEqual(check_counters(adam, state), dict(shared_actor=12645, value=5))
            weights, moments = deepcopy(policy.state_dict()), deepcopy(adam.state_dict())
            rejected_cfg = deepcopy(cfg)
            rejected_cfg["ppo"]["stop_kl"] = 1e-12
            rejected = update(policy, adam, continuation.PPOContinuationTests.episodes(self, policy), rejected_cfg,
                              0, "cpu", rng, _backend=BatchedBackend)
            self.assertEqual(rejected["optimizer_steps"], 0)
            assert_equal(policy.state_dict(), weights)
            assert_equal(adam.state_dict(), moments)
            runner.save_boundary(out, policy, adam, state, rng, cfg, identity)
            restored = Policy()
            restored_adam = grouped_optimizer(restored, cfg, "cpu")
            checkpoint.restore(out / state["latest"], restored, restored_adam, rng, cfg, identity)
            assert_equal(restored_adam.state_dict(), moments)
            self.assertEqual(verify_checkpoint_chain(out, adam_offset=12644)["status"], "verified")


if __name__ == "__main__":
    unittest.main()
