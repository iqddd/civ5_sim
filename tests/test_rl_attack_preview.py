"""Public attack features must not leak information or corrupt inherited Adam."""

import random
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from civ5rl.core.model import Policy, batch_inputs
from civ5rl.core.observation import ObservationEncoder, ordered_actions
from civ5rl.core.preview import load_parent_model
from civ5rl.engine.optimizer_transaction import AdamSnapshot
from civ5rl.engine.ppo import encode_episode
from civ5rl.engine.ppo_adam import check_counters, grouped_optimizer
from civ5rl.engine.shared import EncodedSlots
from civ5rl.experiments.imitation_helpers import assert_equal
from civ5rl.experiments.ppo_after_target_bc import load_initial_optimizer
from civ5rl.experiments.ppo_artifacts import pack_rollout, unpack_rollout
from civ5rl.protocols.ppo_after_target_bc import load_config
from civ5rl.provenance import checkpoint
from civ5rl.provenance.common import configure
from civ5sim import Battle
from test_combat import scenario, unit
from test_rl import context

CONFIG = "configs/rl-ppo-c2-h60-gae-gamma1-lambda099-attack-preview.json"


class AttackPreviewTests(unittest.TestCase):
    def test_c2p_continuation_preserves_all_three_adam_origins(self):
        # Dropping the third group's moments silently changes a continuation's optimization.
        cfg = load_config("configs/rl-ppo-c2p-h60-end-turn.json")
        parent = Policy("C2P")
        old = grouped_optimizer(parent, cfg, "cpu")
        origins = cfg["experiment"]["parent_optimizer_counters"]
        for parameter in parent.parameters():
            parameter.grad = torch.full_like(parameter, .01)
        old.step()
        for group in old.param_groups:
            for parameter in group["params"]:
                old.state[parameter]["step"].fill_(origins[group["name"]])
        saved = dict(variant="C2P", model=deepcopy(parent.state_dict()),
                     optimizer=deepcopy(old.state_dict()),
                     state=dict(critic_complete=True, phase="collect", Adam_counters=origins))
        child = Policy("C2P")
        load_parent_model(child, saved, cfg)
        adam = load_initial_optimizer(child, saved, cfg)
        assert_equal(adam.state_dict(), saved["optimizer"])
        transaction = AdamSnapshot(child, adam)
        for parameter in child.parameters():
            parameter.grad = torch.full_like(parameter, .01)
        adam.step()
        self.assertEqual(check_counters(adam, dict(origin_Adam_counters=origins, optimizer_steps=1)),
                         {key: value + 1 for key, value in origins.items()})
        transaction.restore()
        assert_equal(child.state_dict(), saved["model"])
        assert_equal(adam.state_dict(), saved["optimizer"])

    def test_end_turn_plan_changes_physics_without_changing_layouts(self):
        # Unchanged case hashes could silently reuse legacy evaluation results.
        from civ5rl.protocols.turn_reset import evaluation_plan, validate_case
        from civ5sim.engine import digest
        case = dict(scenario=scenario(unit(), unit(side=1, x=3)), split="validation")
        case["case_hash"] = digest(case)
        parent = dict(development_512=[case], hash="source")
        cfg = {"environment": {"turn_reset_mode": "end_turn"}}
        plan = evaluation_plan(parent, cfg)
        converted = plan["development_512"][0]
        self.assertNotEqual(converted["case_hash"], case["case_hash"])
        self.assertNotIn("turn_reset_mode", case["scenario"])
        self.assertEqual(converted["scenario"]["units"], case["scenario"]["units"])
        validate_case(converted, cfg)
        with self.assertRaises(ValueError):
            validate_case(case, cfg)

    def test_public_batch_matches_scalar_without_state_or_hidden_information(self):
        # A fast native path could accidentally use omniscient strengths or mutate RNG.
        left = scenario(unit(hp=24), unit(side=1, x=3, hp=15), unit(side=1, x=8),
                        fog=True, random_damage=True)
        right = deepcopy(left)
        right["units"][2].update(tile=[9, 2], hp=1)
        with Battle(left) as a, Battle(right) as b:
            self.assertEqual(a.observe(0), b.observe(0))
            before = a.snapshot()
            actions, features = a.actions_with_previews()
            other_actions, other_features = b.actions_with_previews()
            self.assertEqual(actions, ordered_actions(a.available_actions()))
            self.assertEqual(actions, other_actions)
            np.testing.assert_array_equal(features, other_features)
            visible = a.observe(0)["units"]
            for action, row in zip(actions, features):
                if action.kind not in ("melee", "ranged"):
                    np.testing.assert_array_equal(row, np.zeros(5))
                    continue
                scalar = a.preview_attack(action)
                defender = next(u for u in visible if u["tile"] == action.target)
                attacker = next(u for u in visible if u["id"] == action.unit)
                np.testing.assert_array_equal(row, np.asarray([
                    1, scalar.damage / 100, scalar.retaliation / 100,
                    scalar.damage >= defender["hp"], scalar.retaliation >= attacker["hp"]], dtype=np.float32))
            self.assertEqual(before, a.snapshot())

    def test_zero_extension_inherits_policy_adam_and_recorded_replay(self):
        # Cold-started old Adam or shifted action rows silently alter the experiment.
        configure(17, "cpu")
        cfg = load_config(CONFIG)
        parent = Policy("C2")
        old_cfg = deepcopy(cfg)
        old_cfg["model"]["variant"] = "C2"
        old_adam = grouped_optimizer(parent, old_cfg, "cpu")
        for p in parent.parameters():
            p.grad = torch.full_like(p, .01)
        old_adam.step()
        origins = cfg["experiment"]["parent_optimizer_counters"]
        for group in old_adam.param_groups:
            for p in group["params"]:
                old_adam.state[p]["step"].fill_(origins[group["name"]])
        saved = dict(variant="C2", model=deepcopy(parent.state_dict()), optimizer=deepcopy(old_adam.state_dict()),
                     state=dict(critic_complete=True, phase="collect", Adam_counters=origins))
        policy = Policy("C2P")
        load_parent_model(policy, saved, cfg)
        adam = load_initial_optimizer(policy, saved, cfg)
        self.assertNotIn(policy.attack_preview.weight, adam.state)
        initial = scenario(unit(), unit(side=1, x=3))
        with Battle(initial) as battle:
            actions, previews = battle.actions_with_previews()
            observation = battle.observe(0)
            observation["attack_previews"] = previews
        encoded = ObservationEncoder(context(initial), "C2P").encode(observation, actions)
        old_encoded = ObservationEncoder(context(initial), "C2").encode(observation, actions)
        batch = batch_inputs([encoded], "cpu")
        with torch.no_grad():
            for a, b in zip(parent(batch_inputs([old_encoded], "cpu")), policy(batch)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        slots = EncodedSlots(1, 1024)
        try:
            published = slots.write(0, encoded)
            np.testing.assert_array_equal(slots.read(0, published).attack_preview, previews)
        finally:
            slots.close()
        episode = dict(spec=dict(scenario=initial, armies=context(initial)["armies"], learner_side=0,
                                 max_rounds=3, agent_seeds=[42, 43]), observations=[observation],
                       actions=[actions], chosen=[0])
        restored = unpack_rollout(pack_rollout([episode]))[0]
        np.testing.assert_array_equal(encode_episode(restored, "C2P", reuse_cached=False)[0].attack_preview, previews)
        first = AdamSnapshot(policy, adam)
        attack_index = next(i for i, a in enumerate(actions) if a.kind == "melee")
        logits, values, _ = policy(batch)
        (logits[0, attack_index] + values.sum()).backward()
        self.assertGreater(float(policy.attack_preview.weight.grad.abs().sum()), 0)
        adam.step()
        first.restore()
        self.assertNotIn(policy.attack_preview.weight, adam.state)
        adam.zero_grad(set_to_none=True)
        logits, values, _ = policy(batch)
        (logits[0, attack_index] + values.sum()).backward()
        adam.step()
        state = dict(origin_Adam_counters={**origins, "attack_preview": 0}, optimizer_steps=1)
        self.assertEqual(check_counters(adam, state), dict(shared_actor=29391, value=16835, attack_preview=1))
        weights, moments = deepcopy(policy.state_dict()), deepcopy(adam.state_dict())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            shuffle = random.Random(7)
            checkpoint.save(path, policy, adam, state, shuffle, cfg, {"run_id": "preview-test"})
            fresh = Policy("C2P")
            fresh_adam = grouped_optimizer(fresh, cfg, "cpu")
            checkpoint.restore(path, fresh, fresh_adam, shuffle, cfg, {"run_id": "preview-test"})
            assert_equal(fresh.state_dict(), weights)
            assert_equal(fresh_adam.state_dict(), moments)
