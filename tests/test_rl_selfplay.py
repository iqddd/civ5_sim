"""Focused correctness tests for mixed self-play PPO, 3-group Adam inheritance, and dual H2H selection."""

from collections import Counter
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch

from civ5rl.config import load_config
from civ5rl.core.model import Policy
from civ5rl.engine.collection.collector import Collector
from civ5rl.engine.collection.opponents import (
    OpponentPool,
    maybe_archive_snapshot,
    select_opponent,
)
from civ5rl.engine.ppo.adam import check_counters, expected_counters, grouped_optimizer
from civ5rl.engine.ppo.backend import BatchedBackend
from civ5rl.experiments.h2h_evaluation import (
    dual_promotion_decision,
    evaluate_h2h,
    select_dual_milestone,
)
from civ5rl.experiments.ppo_after_imitation import summarize_rollout
from civ5rl.experiments.ppo_after_target_bc import load_initial_optimizer
from civ5rl.experiments.ppo_artifacts import prune_checkpoints
from civ5rl.protocols.common import bootstrap_ci95
from civ5rl.protocols.memory import flat_map
from civ5rl.protocols.ppo_after_target_bc import validate_profile
from civ5rl.provenance import bound_cases, case_manifest
from civ5rl.provenance.common import ROOT, configure, write_json
from civ5rl.reporting.evaluation_audit import h2h_world_bootstrap, load_eval_rows
from civ5rl.reporting.operations import _confirmation_check
from civ5sim.engine import digest
from tests.test_combat import scenario, unit


class SelfPlayCorrectnessTests(unittest.TestCase):
    def test_opponent_pool_mix_and_archive_retention(self):
        cfg = load_config(ROOT / "configs/rl-ppo-c2p-h60-selfplay.json")
        validate_profile(cfg)

        # Empty archive: 50% cp, 30% current, 20% u0
        empty_counts = Counter(select_opponent(110860 + i, cfg, []) for i in range(4000))
        self.assertAlmostEqual(empty_counts[("cp", None)] / 4000, 0.50, delta=0.025)
        self.assertAlmostEqual(empty_counts[("policy", "current")] / 4000, 0.30, delta=0.025)
        self.assertAlmostEqual(empty_counts[("policy", "u0")] / 4000, 0.20, delta=0.025)

        # Populated archive: 50% cp, 30% current, 10% u0, 10% split across archive
        archive = [{"id": "archive:10"}, {"id": "archive:20"}]
        pop_counts = Counter(select_opponent(110860 + i, cfg, archive) for i in range(4000))
        self.assertAlmostEqual(pop_counts[("cp", None)] / 4000, 0.50, delta=0.025)
        self.assertAlmostEqual(pop_counts[("policy", "current")] / 4000, 0.30, delta=0.025)
        self.assertAlmostEqual(pop_counts[("policy", "u0")] / 4000, 0.10, delta=0.02)
        arch_share = (pop_counts[("policy", "archive:10")] + pop_counts[("policy", "archive:20")]) / 4000
        self.assertAlmostEqual(arch_share, 0.10, delta=0.02)

        # Archive gating: zero steps and duplicate policy hashes must be skipped
        state = {
            "updates": 10,
            "decisions": 330000,
            "optimizer_steps": 0,
            "self_play_archive": [],
            "self_play_last_archived_optimizer_steps": 0,
            "latest": "checkpoints/update-0020.pt",
            "best": {"path": "checkpoints/initial.pt"},
            "evaluated": {},
        }
        self.assertIsNone(
            maybe_archive_snapshot(
                state, cfg, checkpoint_relpath="checkpoints/update-0010.pt", policy_hash_value="h0"
            )
        )
        state["optimizer_steps"] = 51
        added = maybe_archive_snapshot(
            state, cfg, checkpoint_relpath="checkpoints/update-0010.pt", policy_hash_value="h10"
        )
        self.assertIsNotNone(added)
        self.assertEqual(added["id"], "archive:10")
        self.assertEqual(state["self_play_last_archived_optimizer_steps"], 51)
        state["updates"] = 20
        state["optimizer_steps"] = 99
        self.assertIsNone(
            maybe_archive_snapshot(
                state, cfg, checkpoint_relpath="checkpoints/update-0020.pt", policy_hash_value="h10"
            )
        )
        self.assertIsNone(
            maybe_archive_snapshot(
                state,
                cfg,
                checkpoint_relpath="checkpoints/update-0020.pt",
                policy_hash_value=cfg["experiment"]["parent_policy_hash"],
            )
        )

        # Archived checkpoints must survive prune_checkpoints()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            ckpt_dir = out / "checkpoints"
            ckpt_dir.mkdir()
            for name in ("initial.pt", "update-0009.pt", "update-0010.pt", "update-0019.pt", "update-0020.pt"):
                (ckpt_dir / name).write_bytes(b"ckpt")
            prune_checkpoints(out, state)
            remaining = sorted(p.name for p in ckpt_dir.glob("*.pt"))
            self.assertEqual(remaining, ["initial.pt", "update-0010.pt", "update-0019.pt", "update-0020.pt"])

    def test_three_group_adam_inheritance_lr_override_and_value_reset(self):
        configure(17, "cpu")
        cfg = load_config(ROOT / "configs/rl-ppo-c2p-h60-selfplay.json")
        parent_policy = Policy("C2P")
        parent_cfg = deepcopy(cfg)
        parent_cfg["optimizer"]["lr"] = 1e-5
        parent_opt = grouped_optimizer(parent_policy, parent_cfg, "cpu")
        origins = cfg["experiment"]["parent_optimizer_counters"]
        for group in parent_opt.param_groups:
            step_val = float(origins[group["name"]])
            for p in group["params"]:
                parent_opt.state[p] = {
                    "step": torch.tensor(step_val),
                    "exp_avg": torch.full_like(p, 0.123),
                    "exp_avg_sq": torch.full_like(p, 0.456),
                }
        saved = {
            "variant": "C2P",
            "model": parent_policy.state_dict(),
            "optimizer": parent_opt.state_dict(),
            "state": {"critic_complete": True, "phase": "collect", "Adam_counters": dict(origins)},
        }
        student = Policy("C2P")
        student.load_state_dict(saved["model"])
        opt = load_initial_optimizer(student, saved, cfg)
        self.assertEqual([g["name"] for g in opt.param_groups], ["shared_actor", "value", "attack_preview"])
        for group in opt.param_groups:
            self.assertEqual(group["lr"], 5e-6)
        state = {
            "origin_Adam_counters": {**origins, "value": 0},
            "value_optimizer": "reset_then_preserve",
            "optimizer_steps": 0,
            "critic_progress": {"steps": 0},
        }
        self.assertEqual(check_counters(opt, state), {"shared_actor": 41882, "value": 0, "attack_preview": 12492})
        state["critic_progress"]["steps"] = 4
        self.assertEqual(
            expected_counters(state, extra_ppo_steps=1),
            {"shared_actor": 41883, "value": 5, "attack_preview": 12493},
        )

    def test_streaming_selfplay_slot_reuse_isolation_and_replay_parity(self):
        configure(17, "cpu")
        cfg = load_config(ROOT / "configs/rl-ppo-c2p-h60-selfplay.json")

        base_spec = scenario(unit(x=0, y=0), unit(side=1, x=1, y=1), terrain=flat_map(3))
        base_spec["fog"] = True
        base_spec["random_damage"] = True
        base_spec["turn_reset_mode"] = "end_turn"
        specs = []
        opponents = [("policy", "u0"), ("policy", "archive:10"), ("policy", "current"), ("cp", None)]
        for i, (opp, opp_id) in enumerate(opponents):
            s = {
                "episode_id": 110860 + i,
                "scenario": deepcopy(base_spec),
                "armies": [["warrior"], ["warrior"]],
                "learner_side": i % 2,
                "max_rounds": 4,
                "max_calls_per_turn": 200,
                "opponent": opp,
                "reward_profile": "win_probability",
                "agent_seeds": [100 + 2 * i, 101 + 2 * i],
                "split": "train",
                "scenario_hash": digest(base_spec),
            }
            if opp == "cp":
                s["opponent_search"] = deepcopy(cfg["teacher_search"])
            else:
                s["opponent_id"] = opp_id
            specs.append(s)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            ckpt_dir = out / "checkpoints"
            ckpt_dir.mkdir()
            policy = Policy("C2P")
            u0 = deepcopy(policy)
            arch = deepcopy(policy)
            with torch.no_grad():
                for p in arch.parameters():
                    p.mul_(0.95)
            torch.save({"variant": "C2P", "model": u0.state_dict()}, ckpt_dir / "initial.pt")
            torch.save({"variant": "C2P", "model": arch.state_dict()}, ckpt_dir / "update-0010.pt")

            pool_obj = OpponentPool(cfg, "cpu", out)
            pool_obj.sync_archive([{"id": "archive:10", "path": "checkpoints/update-0010.pt"}])

            runner = pool_obj.model_for("archive:10", policy)
            for p1, p2 in zip(runner.parameters(), arch.parameters()):
                self.assertTrue(torch.equal(p1, p2))

            trusted = case_manifest(specs, source="test_selfplay", split="train", namespace="test_sp")
            # num_envs=2 across 4 episodes forces real slot reuse (assign over finished self-play slots)
            with Collector(
                cfg, None, "cpu", out / "errors",
                cpu_workers=1, num_envs=2, reply_size=1,
                collection_scheduler="streaming",
                opponent_pool=pool_obj,
            ) as collector:
                episodes, _, _ = collector.collect(
                    policy, None, 0, fixed_specs=bound_cases(trusted), trusted_manifest=trusted
                )

            self.assertEqual(len(episodes), 4)
            for ep in episodes:
                self.assertEqual(len(ep["chosen"]), ep["result"]["learner_decisions"])
                self.assertLessEqual(len(ep["chosen"]), ep["result"]["commands"])
                for obs in ep["observations"]:
                    self.assertEqual(obs["side"], ep["spec"]["learner_side"])
            backend = BatchedBackend(policy, episodes, cfg, "cpu")
            replay = backend.precheck()
            self.assertLessEqual(replay["full_kl"], 1e-6)
            self.assertLessEqual(replay["max_total_variation"], 0.005)
            summary = summarize_rollout(episodes)
            self.assertEqual(set(summary["by_opponent"]), {"cp", "current", "u0", "archive"})

    def test_dual_selection_h2h_symmetry_and_audit_semantics(self):
        # 1. Dual selection & promotion rules
        base = dict(path="checkpoints/initial.pt", update=0, decisions=0, policy_hash="h0",
                    cp_wins=361, h2h_wins=240, h2h_losses=240, cp_summary={}, h2h_summary={})
        cp_only = dict(path="checkpoints/update-0030.pt", update=30, decisions=1048576, policy_hash="h1",
                       cp_wins=370, h2h_wins=230, h2h_losses=240, cp_summary={}, h2h_summary={})
        h2h_only = dict(path="checkpoints/update-0060.pt", update=60, decisions=2097152, policy_hash="h2",
                        cp_wins=360, h2h_wins=270, h2h_losses=210, cp_summary={}, h2h_summary={})
        both_good = dict(path="checkpoints/update-0090.pt", update=90, decisions=3145728, policy_hash="h3",
                         cp_wins=368, h2h_wins=260, h2h_losses=220, cp_summary={}, h2h_summary={})
        tie_more_cp = dict(path="checkpoints/update-0120.pt", update=120, decisions=4194304, policy_hash="h4",
                           cp_wins=371, h2h_wins=257, h2h_losses=220, cp_summary={}, h2h_summary={})
        self.assertEqual(select_dual_milestone([base, cp_only, h2h_only], 361)["path"], "checkpoints/initial.pt")
        chosen = select_dual_milestone([base, cp_only, h2h_only, both_good, tie_more_cp], 361)
        self.assertEqual((chosen["path"], chosen["J"], chosen["delta_w_cp"], chosen["delta_w_h2h"]),
                         ("checkpoints/update-0120.pt", 47, 10, 37))

        self.assertTrue(
            dual_promotion_decision({"delta": 0.02, "CI95": [-0.01, 0.05]}, {"delta": 0.06, "CI95": [0.01, 0.11]})["promoted"]
        )
        self.assertFalse(
            dual_promotion_decision({"delta": -0.01, "CI95": [-0.04, 0.02]}, {"delta": 0.08, "CI95": [0.03, 0.13]})["promoted"]
        )
        self.assertFalse(
            dual_promotion_decision({"delta": 0.02, "CI95": [-0.01, 0.05]}, {"delta": 0.02, "CI95": [-0.01, 0.05]})["promoted"]
        )

        # 2. Decisive mirrored H2H zero-point symmetry on CPU with fresh Policy("C2P") (no artifacts/CUDA)
        configure(17, "cpu")
        cfg = deepcopy(load_config(ROOT / "configs/rl-ppo-c2p-h60-selfplay.json"))
        cfg["evaluation_runtime"] = {"device": "cpu", "workers": 1, "max_batch": 2, "wait_seconds": 0.001}
        base_sc = scenario(
            unit(x=0, y=0, hp=100),
            unit(side=1, x=1, y=0, hp=1),
            terrain=flat_map(3),
        )
        base_sc["fog"] = True
        base_sc["random_damage"] = True
        base_sc["turn_reset_mode"] = "end_turn"
        cases = []
        for side in (0, 1):
            c = {
                "case_id": 0,
                "layout_id": 0,
                "parent_world": 20000,
                "scenario": deepcopy(base_sc),
                "armies": [["warrior"], ["warrior"]],
                "learner_side": side,
                "max_rounds": 6,
                "placement": "contact",
                "map_size": 3,
                "army_bin": 0,
                "split": "validation",
                "agent_seeds": [101, 102],
            }
            c["case_hash"] = digest(c)
            cases.append(c)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "checkpoints").mkdir()
            policy = Policy("C2P")
            # Bias attack_preview weight so argmax attacks the adjacent 1-HP enemy deterministically
            with torch.no_grad():
                policy.attack_preview.weight.fill_(10.0)
            torch.save({"variant": "C2P", "model": policy.state_dict()}, out / "checkpoints/initial.pt")
            write_json(out / "run.json", {"evaluation_runtime_default": "auto"})
            res = evaluate_h2h(
                cfg, {"development_512": cases}, out,
                "checkpoints/initial.pt", "checkpoints/initial.pt", "development_512",
            )
            self.assertEqual((res["summary"]["W"], res["summary"]["D"], res["summary"]["L"]), (1, 0, 1))
            self.assertEqual(res["summary"]["delta_wins"], 0)
            self.assertEqual(len(load_eval_rows(out / res["directory"], expected_n=2)), 2)

            # 3. In-memory h2h_world_bootstrap matches shared bootstrap_ci95
            rows = {}
            for w in range(16):
                for side in (0, 1):
                    win = side if (w % 4 != 0 or side == 0) else (1 - side)
                    rows[(20000 + w, w, side)] = ({"winner": win, "side": side}, {}, Path("."))
            boot = h2h_world_bootstrap(rows, 32, samples=500, seed=20261103, clusters=16)
            expected_ci = bootstrap_ci95(
                [boot["world_deltas"][str(20000 + w)] for w in range(16)],
                samples=500,
                seed=20261103,
            )
            self.assertAlmostEqual(boot["CI95"][0], float(expected_ci[0]), places=12)
            self.assertAlmostEqual(boot["CI95"][1], float(expected_ci[1]), places=12)

            # 4. Dual _confirmation_check: fallback to initial.pt allowed when dW_cp > 0 but dW_h2h < 0
            base_cp = {"matches": 512, "W": 361, "D": 31, "L": 120}
            base_h2h = {"matches": 512, "W": 240, "D": 32, "L": 240}
            cand_cp = {"matches": 512, "W": 370, "D": 22, "L": 120}
            cand_h2h_bad = {"matches": 512, "W": 230, "D": 42, "L": 240}
            best = select_dual_milestone(
                [
                    dict(path="checkpoints/initial.pt", update=0, decisions=0, policy_hash="h0",
                         cp_wins=361, h2h_wins=240, h2h_losses=240, cp_summary=base_cp, h2h_summary=base_h2h),
                    dict(path="checkpoints/update-0030.pt", update=30, decisions=1048576, policy_hash="h1",
                         cp_wins=370, h2h_wins=230, h2h_losses=240, cp_summary=cand_cp, h2h_summary=cand_h2h_bad),
                ],
                361,
            )
            state = {
                "stop_reason": "decision_limit",
                "best": best,
                "evaluated": {
                    "h0": {
                        "checkpoint": "checkpoints/initial.pt",
                        "policy_hash": "h0",
                        "decisions": 0,
                        "update": 0,
                        "runs": {"argmax": {"cp": {"summary": base_cp}}},
                        "h2h": {"summary": base_h2h},
                    },
                    "h1": {
                        "checkpoint": "checkpoints/update-0030.pt",
                        "policy_hash": "h1",
                        "decisions": 1048576,
                        "update": 30,
                        "runs": {"argmax": {"cp": {"summary": cand_cp}}},
                        "h2h": {"summary": cand_h2h_bad},
                    },
                },
            }
            write_json(out / "selected.json", best)
            checked = _confirmation_check(out, state, {"status": "not_run_parent_selected"}, cfg)
            self.assertEqual(checked["selection_rule"], "dual_nonregression_max_J")
            # Making candidate admissible (dW_h2h > 0) must cause skipped confirmation to fail
            state["evaluated"]["h1"]["h2h"]["summary"] = {"matches": 512, "W": 260, "D": 12, "L": 240}
            with self.assertRaises(ValueError):
                _confirmation_check(out, state, {"status": "not_run_parent_selected"}, cfg)


if __name__ == "__main__":
    unittest.main()
