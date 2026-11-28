"""Keep the three-opponent acceptance and continuation gates reproducible."""

import unittest
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch

from civ5rl.engine.collection.opponents import select_opponent
from civ5rl.protocols import three_opponents as protocol


def entry(step, v2, cp, balance, *, cp_contact=None):
    cp_contact = cp // 2 if cp_contact is None else cp_contact
    return {
        "checkpoint": "checkpoints/initial.pt" if step == 0 else f"checkpoints/update-{step:04}.pt",
        "policy_hash": f"policy-{step}",
        "update": step,
        "runs": {"argmax": {"cp": {"summary": {"W": cp}}}},
        "slices": {"placement": {"contact": {"W": cp_contact}, "separated": {"W": cp - cp_contact}}},
        "cpv2": {"runs": {"argmax": {"cpv2": {"summary": {"W": v2}}}},
                  "slices": {"placement": {"contact": {"W": v2 // 2}, "separated": {"W": (v2 + 1) // 2}}}},
        "h2h": {"summary": {"W": 230 + balance, "L": 230}},
    }


class ThreeOpponentTests(unittest.TestCase):
    def test_pre_gelu_pilot_uses_committed_r3_and_can_recover_missing_diagnostic(self):
        from civ5rl.experiments.ppo.after_target_bc import diagnostic
        from civ5rl.core.model import Policy
        from civ5rl.provenance.common import file_sha256, read_json, write_json
        from civ5rl.provenance.jsonl import append_jsonl

        with TemporaryDirectory() as directory:
            out = Path(directory)
            policy = Policy("C2PA-pre-GELU")
            checkpoint = out / "checkpoints/update-0003.pt"
            checkpoint.parent.mkdir()
            torch.save({"variant": policy.variant}, checkpoint)
            rollout = out / "rollouts/latest.pt"
            rollout.parent.mkdir()
            torch.save([{"chosen": [0]}], rollout)
            write_json(rollout.with_suffix(".json"), {"update": 3, "sha256": file_sha256(rollout)})
            write_json(out / "run.json", {"config": {"model": {"variant": policy.variant}, "critic": {"epochs": 4}}})
            row = {"cpv2_contact_pre": {"explained_variance": 0.4}}
            write_json(out / "diagnostic-rollouts.json", {"2": row, "3": row})
            write_json(out / "critic-preparation.json", {
                "epochs_completed": 4, "cpv2_contact_after": {"explained_variance": 0.5},
            })
            for index in (1, 2, 3):
                append_jsonl(out / "updates.jsonl", {"rollout": index, "ppo": {
                    "optimizer_steps": 57, "stop_reason": "epochs_completed", "PPO_skipped": False,
                }})
            session = SimpleNamespace(out=out, policy=policy, device="cpu", shuffle=Random(17), state={
                "rollouts": 3, "phase": "collect", "latest": "checkpoints/update-0003.pt", "optimizer_steps": 171,
            })
            self.assertFalse(diagnostic.pilot_report(out)["passed"])
            with patch.object(diagnostic, "_attention_metrics", side_effect=RuntimeError("interrupted")):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    diagnostic.record_post_r3(session)
            self.assertNotIn("3_post", read_json(out / "diagnostic-rollouts.json"))
            self.assertTrue(policy.training)

            with patch.object(diagnostic, "_attention_metrics", return_value={
                "attention_rms": 0.02, "argmax_switch_rate": 0.003,
                "qkv_relative_change": 0.001, "value_context_rms": 0.02,
            }) as measure:
                diagnostic.record_post_r3(session)
                diagnostic.record_post_r3(session)
                measure.assert_called_once()
            report = diagnostic.pilot_report(out)
            self.assertTrue(report["passed"])
            self.assertEqual(report["R3_post"]["accepted_steps"], 171)
            self.assertEqual(report["R3_post"]["checkpoint_sha256"], file_sha256(checkpoint))
            self.assertNotIn("retry_overrides", report)

    def test_critic_capsule_restores_context_without_touching_actor(self):
        from copy import deepcopy

        from civ5rl.config import load_config
        from civ5rl.core.model import Policy
        from civ5rl.engine.ppo.adam import check_counters, grouped_optimizer
        from civ5rl.experiments.ppo import critic
        from civ5rl.experiments.ppo.after_target_bc.boundary import ActiveClock
        from civ5rl.provenance.common import write_json
        from civ5rl.provenance.identity import policy_hash

        config = load_config("configs/rl-ppo-c2pa-h60-cpv2.json")
        config["critic"].update(epochs=2, batch_size=2)
        policy = Policy("C2PA")
        weights = deepcopy(policy.state_dict())
        actor = policy_hash(weights)
        features = torch.rand((4, policy.hidden_size + 2 * policy.hex_dim))
        episodes = [dict(chosen=[0, 0], old_value=[0., 0.], result={"reward": reward}) for reward in (1, 0)]
        state = {"active_seconds": 0., "optimizer_steps": 0, "value_optimizer": "reset_then_preserve",
                 "origin_Adam_counters": {g: 0 for g in ("shared_actor", "value", "attack_preview", "action_context", "value_actions")}}
        rng = Random(17)

        def feature_blocks(*_args):
            yield 0, 0, features[:2]
            yield 1, 0, features[2:]

        with TemporaryDirectory() as directory:
            out = Path(directory)
            write_json(out / "run.json", {"kind": config["kind"], "config": config})
            clock = ActiveClock(state, out)
            save = critic.atomic_save

            def stop_after_capsule(value, path):
                save(value, path)
                if path.name == "critic-adaptation.pt":
                    raise RuntimeError("interrupt after saving critic capsule")

            def adapt(optimizer):
                return critic.adapt_critic(policy, optimizer, episodes, config, "cpu", out, state,
                                          rng, {}, clock, None, features_for=feature_blocks,
                                          behavior_hash=lambda _: "fixed")

            with patch.object(critic, "atomic_save", side_effect=stop_after_capsule):
                with self.assertRaisesRegex(RuntimeError, "interrupt after saving"):
                    adapt(grouped_optimizer(policy, config, "cpu"))
            policy.load_state_dict(weights)
            optimizer = grouped_optimizer(policy, config, "cpu")
            self.assertEqual(adapt(optimizer)["steps"], 4)
            self.assertEqual(check_counters(optimizer, state)["value_actions"], 4)
            self.assertEqual(policy_hash(policy.state_dict()), actor)
            self.assertGreater(torch.count_nonzero(policy.value_actions.weight).item(), 0)
            with torch.no_grad():
                self.assertEqual(torch.cat([policy.value_from_features(x) for x in features.split(2)]).tolist(),
                                 [value for episode in episodes for value in episode["old_value"]])

    def test_selection_and_resume_only_use_admissible_milestones(self):
        points = {"0": entry(0, 255, 378, 0), "1": entry(1, 381, 378, 40)}
        self.assertEqual(protocol.select(points)["milestone"], 1)
        points["2"] = entry(2, 385, 384, 0)
        points["3"] = entry(3, 385, 384, 0, cp_contact=170)
        self.assertEqual(protocol.select(points)["milestone"], 2)
        points["4"] = entry(4, 385, 384, 0)
        self.assertEqual(protocol.select(points)["milestone"], 4)
        self.assertEqual(protocol.select(points, continuation={"milestone": 4})["milestone"], 0)
        points["5"] = {**entry(5, 385, 384, 0), "policy_hash": points["4"]["policy_hash"]}
        self.assertEqual(protocol.select(points, continuation={"milestone": 4})["milestone"], 0)
        points["6"] = entry(6, 385, 384, 0)
        self.assertEqual(protocol.select(points, continuation={"milestone": 4})["milestone"], 6)

    def test_stops_and_promotion_check_the_right_opponent(self):
        baseline = entry(0, 255, 378, 0)
        self.assertEqual(protocol.stop_reason({"0": baseline, "1": entry(1, 279, 390, 0), "2": entry(2, 279, 400, 10)},
                                              [1, 2, 3], gain=25, gain_boundary=2), "no_development_gain")
        points = {"0": baseline, "1": entry(1, 392, 392, -1), "2": entry(2, 392, 392, 0)}
        self.assertIsNone(protocol.stop_reason(points, [1, 2, 3], gain=25, gain_boundary=2))
        points["3"] = entry(3, 392, 392, 0)
        self.assertEqual(protocol.stop_reason(points, [1, 2, 3], gain=25, gain_boundary=2), protocol.EARLY_GOAL)
        self.assertTrue(protocol.allows_continuation(entry(1, 188, 193, 0)))
        self.assertFalse(protocol.allows_continuation(entry(1, 187, 193, 0)))
        self.assertFalse(protocol.allows_continuation(entry(1, 191, 193, -1)))
        v2 = {"delta": 0.1, "CI95": [0.05, 0.14]}
        cp = {"delta": 0}
        self.assertEqual(protocol.initial_promotion(v2, cp, entry(1, 191, 192, 0)),
                         {"promoted": True, "dual_target_confirmed": False})

    def test_archive_share_returns_to_u0_until_first_snapshot(self):
        config = {"self_play": {"cpv2_share": .5, "cp_share": .1, "current_share": .25,
                               "parent_share": .05, "archive_share": .1}}
        archive = [{"id": "archive:10"}]
        without = [select_opponent(i, config, []) for i in range(256)]
        with_archive = [select_opponent(i, config, archive) for i in range(256)]
        self.assertEqual(set(without), {("cp", None), ("cpv2", None), ("policy", "current"), ("policy", "u0")})
        self.assertIn(("policy", "archive:10"), with_archive)
        self.assertEqual(without, [("policy", "u0") if key == "archive:10" else (opponent, key)
                                   for opponent, key in with_archive])

    def test_adaptive_gain_threshold_and_saved_stop(self):
        from civ5rl.provenance.common import write_json
        from civ5rl.provenance.jsonl import append_jsonl
        from civ5rl.reporting.operations import _three_opponent_confirmation_check

        rule = protocol.ADAPTIVE_CPV2_GAIN
        for baseline, required in ((245, 285), (255, 288), (275, 303), (300, 321), (320, 340), (390, 384)):
            with self.subTest(baseline=baseline):
                self.assertEqual(protocol.required_gain_wins(baseline, None, gain_rule=rule), required)

        milestones = [1, 2, 3, 4]
        points = {"0": entry(0, 255, 393, 0), "1": entry(1, 288, 393, -1), "2": entry(2, 287, 393, -1)}
        stop = {"gain": None, "gain_boundary": 3, "gain_rule": rule}
        self.assertIsNone(protocol.stop_reason(points, milestones, **stop))
        points["3"] = entry(3, 287, 393, -1)
        self.assertIsNone(protocol.stop_reason(points, milestones, **stop))
        points["1"], points["2"] = entry(1, 287, 393, -1), entry(2, 288, 393, -1)
        self.assertIsNone(protocol.stop_reason(points, milestones, **stop))
        points["2"] = entry(2, 287, 393, -1)
        self.assertEqual(protocol.stop_reason(points, milestones, **stop), protocol.NO_DEVELOPMENT_GAIN)

        for point, result in points.items():
            result["decisions"] = int(point)
            result["cpv2"].update(purpose="development_512", checkpoint=result["checkpoint"],
                                  policy_hash=result["policy_hash"])
        experiment = {"milestones": milestones, "early_gain_decisions": 3, "early_gain_rule": rule}
        state = {"three_opponent_points": points,
                 "development_points": {key: points[key]["policy_hash"] for key in ("1", "2", "3")},
                 "decisions": 3, "latest": points["3"]["checkpoint"], "early_gain_required_wins": 288,
                 "stop_reason": protocol.NO_DEVELOPMENT_GAIN, "best": protocol.select(points)}
        confirmation = {"attempt": 1, "status": "not_run_parent_selected"}
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            write_json(directory / "manifest.json", {})
            write_json(directory / "selected.json", state["best"])
            append_jsonl(directory / "updates.jsonl", {"rollout": 3, "total_decisions": 3})
            _three_opponent_confirmation_check(directory, state, confirmation, experiment)
            with self.assertRaisesRegex(ValueError, "threshold differs"):
                _three_opponent_confirmation_check(directory, {**state, "early_gain_required_wins": 287},
                                                   confirmation, experiment)
            with self.assertRaisesRegex(ValueError, "stop differs"):
                _three_opponent_confirmation_check(directory, {**state, "stop_reason": "decision_limit"},
                                                   confirmation, experiment)
            with self.assertRaisesRegex(ValueError, "last committed rollout"):
                _three_opponent_confirmation_check(directory, {**state, "decisions": 4}, confirmation, experiment)

    def test_audit_rejects_a_conflicting_cpv2_milestone_summary(self):
        from civ5rl.provenance.common import write_json
        from civ5rl.reporting.operations import _evaluation_check

        point = entry(0, 255, 393, 0)
        series = point["cpv2"]["runs"]["argmax"]["cpv2"]
        summary = {"matches": 512, "W": 255, "D": 0, "L": 257}
        series.update(directory="evaluation/cpv2", summary={**summary, "W": 254, "L": 258})
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            write_json(directory / "manifest.json", {"development_512": []})
            with self.assertRaisesRegex(ValueError, "Saved summaries disagree"):
                _evaluation_check(directory, [{**series, "summary": summary}], {"0": point})
