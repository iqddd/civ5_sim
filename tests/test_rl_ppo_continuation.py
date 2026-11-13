"""Continuation must preserve Adam/RNG semantics and the fixed spending gate."""

import random
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import torch

from civ5rl.core.model import Policy
from civ5rl.engine.evaluation import evaluation_options, runtime_for
from civ5rl.engine.ppo import update
from civ5rl.engine.ppo_fast import BatchedBackend
from civ5rl.experiments import ppo_after_target_bc as runner
from civ5rl.experiments.imitation_helpers import assert_equal, isolated_rng
from civ5rl.experiments.ppo_after_imitation import optimizer_for
from civ5rl.protocols.ppo_after_target_bc import check_development_gain, load_config
from civ5rl.provenance import checkpoint
from civ5rl.provenance.audit import verify_checkpoint_chain, verify_ppo_ledger
from civ5rl.provenance.common import configure, read_json, write_json
import test_rl_ppo_after_target_bc as fixtures


class PPOContinuationTests(unittest.TestCase):
    def test_two_inherited_adam_origins_survive_update_rollback_and_resume(self):
        # A single origin silently corrupts the adapted critic's Adam clock.
        from civ5rl.engine.ppo_adam import grouped_optimizer, check_counters, expected_counters

        configure(17)
        cfg = load_config("configs/rl-ppo-c2-h60-gae-gamma1-continuation.json")
        cfg["model"]["variant"] = "C"
        cfg["performance"].update(compile=False, resident_rollout=False)
        cfg["ppo"].update(epochs=1, sequence_length=4, valid_tokens_per_minibatch=8,
                          sequences_per_microbatch=1, stop_kl=0.05)
        policy = Policy()
        adam = grouped_optimizer(policy, cfg, "cpu")
        for p in policy.parameters():
            p.grad = torch.full_like(p, 0.01)
        adam.step()
        adam.zero_grad(set_to_none=True)
        origins = cfg["experiment"]["parent_optimizer_counters"]
        for group in adam.param_groups:
            for p in group["params"]:
                adam.state[p]["step"].fill_(origins[group["name"]])
        saved = dict(model=deepcopy(policy.state_dict()), optimizer=deepcopy(adam.state_dict()),
                     variant="C", state=dict(optimizer_steps=12527, critic_complete=True, phase="collect",
                                             Adam_counters=origins))
        inherited = runner.load_initial_optimizer(policy, saved, cfg)
        assert_equal(inherited.state_dict(), saved["optimizer"])
        rng = random.Random(987)
        metrics = update(policy, inherited, self.episodes(policy), cfg, 0, "cpu", rng, _backend=BatchedBackend)
        self.assertEqual(metrics["optimizer_steps"], 1)
        state = dict(optimizer_steps=1, attempted_steps=1, rejected_steps=0, updates=1,
                     phase="collect", decisions=8, origin_Adam_counters=origins,
                     latest="checkpoints/update-0001.pt", critic_complete=True)
        state["Adam_counters"] = expected_counters(state)
        self.assertEqual(check_counters(inherited, state), dict(shared_actor=25172, value=12616))
        weights, moments = deepcopy(policy.state_dict()), deepcopy(inherited.state_dict())
        rejecting = deepcopy(cfg)
        rejecting["ppo"]["stop_kl"] = 1e-12
        rejected = update(policy, inherited, self.episodes(policy), rejecting, 0, "cpu", rng, _backend=BatchedBackend)
        self.assertEqual(rejected["optimizer_steps"], 0)
        assert_equal(policy.state_dict(), weights)
        assert_equal(inherited.state_dict(), moments)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            identity = {"run_id": "two-origin-test"}
            checkpoint.save(out / state["latest"], policy, inherited, state, rng, cfg, identity)
            rng_before = checkpoint.capture_rng(rng)
            restored_policy = Policy()
            restored_adam = grouped_optimizer(restored_policy, cfg, "cpu")
            restored_state = checkpoint.restore(out / state["latest"], restored_policy, restored_adam,
                                                 rng, cfg, identity)
            assert_equal(checkpoint.capture_rng(rng), rng_before)
            assert_equal(restored_policy.state_dict(), weights)
            assert_equal(restored_adam.state_dict(), moments)
            self.assertEqual(check_counters(restored_adam, restored_state), state["Adam_counters"])
            write_json(out / "state.json", state)
            write_json(out / "run.json", dict(config=cfg, parent_Adam_counters=origins))
            (out / "updates.jsonl").write_text(__import__("json").dumps(
                dict(rollout=1, decisions=8, total_decisions=8, ppo=metrics)) + "\n")
            self.assertEqual(verify_ppo_ledger(out)["adam"]["counters"], state["Adam_counters"])
            self.assertEqual(verify_checkpoint_chain(out)["status"], "verified")

    def episodes(self, policy):
        episodes = fixtures.TargetBCPPOTests.episodes(self, policy)
        for ep in episodes:
            ep["policy_temperature"] = 0.5
            ep["old_log_probs"] = [(torch.tensor(row) / 0.5).log_softmax(-1).numpy() for row in ep["old_log_probs"]]
            ep["old_logp"] = [float(row[choice]) for row, choice in zip(ep["old_log_probs"], ep["chosen"])]
        return episodes

    def test_preserved_adam_first_step_rollback_and_resume(self):
        # Resetting Adam silently changes training even if weights and LR match.
        configure(17)
        cfg = load_config("configs/rl-ppo-c2-h60-gae-continuation.json")
        cfg["model"]["variant"] = "C"
        cfg["performance"].update(compile=False, resident_rollout=False)
        cfg["ppo"].update(epochs=1, sequence_length=4, valid_tokens_per_minibatch=8,
                          sequences_per_microbatch=1, stop_kl=0.05)
        policy = Policy()
        adam = optimizer_for(policy, cfg, "cpu")
        # Populate every moment (including critic), then use the checked origin.
        for p in policy.parameters():
            p.grad = torch.full_like(p, 0.01)
        adam.step()
        adam.zero_grad(set_to_none=True)
        for item in adam.state.values():
            item["step"].fill_(12644)
        saved = dict(model=deepcopy(policy.state_dict()), optimizer=deepcopy(adam.state_dict()),
                     variant="C", state=dict(optimizer_steps=12644, critic_complete=True, phase="collect"))
        inherited = runner.load_initial_optimizer(policy, saved, cfg)
        assert_equal(inherited.state_dict(), saved["optimizer"])
        episodes = self.episodes(policy)
        rng = random.Random(987)
        seen = []

        def accepted(row):
            if row["accepted"] and row["full_kl_checked"]:
                seen.append({int(v["step"]) for v in inherited.state.values()})

        metrics = update(policy, inherited, episodes, cfg, 0, "cpu", rng,
                         _backend=BatchedBackend, on_step=accepted)
        self.assertEqual(metrics["optimizer_steps"], 1)
        self.assertEqual(seen, [{12645}])
        self.assertEqual(metrics["optimizer_step_records"][0]["accepted_step"], 1)
        before_weights, before_adam = deepcopy(policy.state_dict()), deepcopy(inherited.state_dict())
        rollback_cfg = deepcopy(cfg)
        rollback_cfg["ppo"]["stop_kl"] = 1e-12
        # Fresh behavior labels ensure that the rejection is for this update.
        episodes = self.episodes(policy)
        rejected = update(policy, inherited, episodes, rollback_cfg, 0, "cpu", rng, _backend=BatchedBackend)
        self.assertEqual(rejected["optimizer_steps"], 0)
        assert_equal(policy.state_dict(), before_weights)
        assert_equal(inherited.state_dict(), before_adam)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            state = dict(optimizer_steps=1, attempted_steps=1, rejected_steps=0, updates=1,
                         phase="collect", decisions=8, origin_Adam_counter=12644,
                         Adam_counter_total=12645, latest="checkpoints/update-0001.pt")
            identity = {"run_id": "continuation-test"}
            checkpoint.save(out / state["latest"], policy, inherited, state, rng, cfg, identity)
            expected_rng = checkpoint.capture_rng(rng)
            configure(456)
            restored_policy = Policy()
            restored_adam = optimizer_for(restored_policy, cfg, "cpu")
            restored_rng = random.Random()
            restored_state = checkpoint.restore(out / state["latest"], restored_policy, restored_adam,
                                                 restored_rng, cfg, identity)
            assert_equal(restored_policy.state_dict(), before_weights)
            assert_equal(restored_adam.state_dict(), before_adam)
            assert_equal(checkpoint.capture_rng(restored_rng), expected_rng)
            self.assertEqual(restored_state, state)
            write_json(out / "state.json", state)
            write_json(out / "run.json", dict(config=cfg, parent_Adam_counter=12644))
            row = dict(rollout=1, decisions=8, total_decisions=8, ppo=metrics)
            (out / "updates.jsonl").write_text(__import__("json").dumps(row) + "\n")
            self.assertEqual(verify_ppo_ledger(out)["adam"]["total"], 12645)
            self.assertEqual(verify_checkpoint_chain(out, adam_offset=12644)["status"], "verified")

    def test_spending_gate_and_fixed_runtime(self):
        cfg = load_config("configs/rl-ppo-c2-h60-gae-continuation.json")
        state = dict(decisions=2097152, baseline_wins=282, early_gain_required_wins=293,
                     development_points={"1048576": "one"}, best=dict(wins=293), stop_reason=None)
        check_development_gain(state, cfg)
        self.assertIsNone(state["stop_reason"])  # incomplete 2 Mi is not a zero score
        state["development_points"]["2097152"] = "two"
        check_development_gain(state, cfg)
        self.assertIsNone(state["stop_reason"])  # best can be the 1 Mi checkpoint
        state["best"]["wins"] = 292
        check_development_gain(state, cfg)
        self.assertEqual(state["stop_reason"], "no_development_gain")
        with tempfile.TemporaryDirectory() as tmp, patch("torch.cuda.is_available", return_value=True):
            runtime = runtime_for(tmp, cfg)
            self.assertEqual((runtime.device, runtime.workers, runtime.max_batch, runtime.wait_seconds),
                             ("cuda", 8, 32, 0.001))
            with evaluation_options(device="cpu"), self.assertRaises(ValueError):
                runtime_for(tmp, cfg)
        with tempfile.TemporaryDirectory() as tmp, patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                runtime_for(tmp, cfg)

    def test_development_recovery_and_single_confirmation(self):
        cfg = load_config("configs/rl-ppo-c2-h60-gae-continuation.json")
        state = dict(latest="checkpoints/initial.pt", baseline_wins=None, evaluation_seconds=0,
                     evaluated={}, decisions=0, updates=0, development_points={},
                     best=dict(path="checkpoints/initial.pt", wins=-1), stop_reason=None)
        def result(path, purpose, wins):
            return dict(policy_hash=path, checkpoint=path, runs=dict(argmax=dict(cp=dict(
                directory=purpose + "/" + Path(path).stem, summary=dict(W=wins)))))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            rng = random.Random(19)
            before_rng = checkpoint.capture_rng(rng)
            with patch.object(runner, "evaluate", side_effect=RuntimeError("GPU failure")), \
                    self.assertRaises(RuntimeError):
                runner.development(cfg, {}, out, state, rng)
            self.assertIsNone(state["baseline_wins"])
            self.assertEqual(state["development_points"], {})
            with patch.object(runner, "report"), patch.object(runner, "evaluate", return_value=result(state["latest"], "dev", 282)):
                runner.development(cfg, {}, out, state, rng)
            self.assertEqual(state["early_gain_required_wins"], 293)
            assert_equal(checkpoint.capture_rng(rng), before_rng)
            # A +1 candidate still receives Additional: there is no old +5pp gate.
            state.update(latest="checkpoints/update-0001.pt", decisions=30, updates=1, stop_reason="active_time_limit")
            with patch.object(runner, "report"), patch.object(runner, "evaluate", return_value=result(state["latest"], "dev", 283)):
                runner.development(cfg, {}, out, state, rng)
            calls = []
            def evaluate(config, plan, directory, path, purpose):
                calls.append((path, purpose))
                entry = result(path, purpose, 1)
                case = dict(parent_world=1, case_id=0, learner_side=0)
                match = out / entry["runs"]["argmax"]["cp"]["directory"] / "case-0"
                write_json(match / "case.json", case)
                write_json(match / "metrics.json", dict(winner=0, side=0))
                return entry
            with patch.object(runner, "evaluate", side_effect=evaluate), \
                    patch.object(runner, "paired_world_bootstrap", return_value={"delta": 0}):
                runner.finish(cfg, {}, out)
                runner.finish(cfg, {}, out)
            self.assertEqual(calls, [("checkpoints/initial.pt", "confirmation"),
                                     ("checkpoints/update-0001.pt", "confirmation")])
            self.assertEqual(read_json(out / "selected.json")["wins"], 283)


if __name__ == "__main__":
    unittest.main()
