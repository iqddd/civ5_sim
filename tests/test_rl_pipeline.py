"""Ownership, public-history equivalence and asynchronous complete episodes."""

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from civ5rl.config import load_config
from civ5rl.collector import Collector
from civ5rl.common import configure
from civ5rl.corpus import context_for
from civ5rl.model import Policy
from civ5rl.observation import ObservationEncoder, ordered_actions
from civ5rl.ppo import encode_episode
from civ5rl.shared import EncodedSlots
from test_rl import scenario

from civ5sim import Battle
from civ5sim.greedy_agent import GreedyAgent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        configure(17)

    def specs(self, count):
        result = []
        for i in range(count):
            s = scenario()
            s["seed"] = 100 + i
            armies = [
                [u["type"] for u in s["units"] if u["side"] == side] for side in (0, 1)
            ]
            result.append(
                {
                    "scenario": s,
                    "armies": armies,
                    "learner_side": i % 2,
                    "max_rounds": 3,
                    "max_calls_per_turn": 2000,
                    "opponent": "greedy",
                    "agent_seeds": [17 + i, 99 + i],
                    "episode_id": i,
                }
            )
        return result

    def test_shared_copy_survives_next_worker_write_and_capacity_fallback(self):
        spec = self.specs(1)[0]
        e = ObservationEncoder(context_for(spec["scenario"], spec["armies"], 0, 3, 0))
        with Battle(spec["scenario"]) as b:
            x = e.encode(b.observe(0), ordered_actions(b.available_actions()))
        for capacity in (1, 4096):
            with self.subTest(capacity=capacity):
                slots = EncodedSlots(1, capacity)
                attached = EncodedSlots(descriptor=slots.descriptor)
                try:
                    state = attached.write(0, x)
                    value = slots.read(0, state)
                    for key in vars(x):
                        np.testing.assert_array_equal(
                            getattr(value, key), getattr(x, key)
                        )
                    original = value.cells.copy()
                    changed = deepcopy(x)
                    changed.cells[:] = 42
                    attached.write(0, changed)
                    np.testing.assert_array_equal(value.cells, original)
                finally:
                    attached.close()
                    slots.close()

    def test_worker_history_and_commands_are_independent_of_scheduler(self):
        config = load_config("configs/rl-v1-throughput.json")
        config["collection"].update(num_envs=4, cpu_workers=2, envs_per_worker=2)
        specs = self.specs(8)
        from civ5rl.provenance import bound_cases, case_manifest

        manifest = case_manifest(
            specs, source="test", split="train", namespace="unit_test"
        )
        specs = bound_cases(manifest)
        with tempfile.TemporaryDirectory() as tmp:
            for variant in ("A", "C"):
                configure(17)
                policy = Policy(variant).eval()
                reference = None
                for scheduler in ("barrier", "ready") if variant == "C" else ("ready",):
                    config["performance"]["collection_scheduler"] = scheduler
                    with Collector(config, None, "cpu", Path(tmp)) as c:
                        episodes, next_episode, _times = c.collect(
                            policy,
                            None,
                            0,
                            fixed_specs=specs,
                            trusted_manifest=manifest,
                        )
                    self.assertEqual(next_episode, 8)
                    self.assertEqual(len(episodes), 8)
                    commands = [e["joint_commands"] for e in episodes]
                    if reference is None:
                        reference = commands
                    self.assertEqual(commands, reference)
                    for ep in episodes:
                        raw = encode_episode(ep, variant, reuse_cached=False)
                        cached = encode_episode(ep, variant)
                        self.assertGreater(len(raw), 1)
                        for a, b in zip(raw, cached):
                            for key in vars(a):
                                np.testing.assert_array_equal(
                                    getattr(a, key), getattr(b, key)
                                )

    def test_greedy_cached_geometry_matches_fresh_public_probe(self):
        class FreshProbe(GreedyAgent):
            def _attack_approach(self, *args):
                self.close()
                return super()._attack_approach(*args)

        s = scenario()
        s["units"][2]["tile"] = 14
        s["units"][3]["tile"] = 34
        with Battle(s) as battle:
            agents = []
            for side in (0, 1):
                pair = [GreedyAgent(), FreshProbe()]
                for agent in pair:
                    agent.reset({"map": s["map"], "side": side})
                agents.append(pair)
            try:
                side = 0
                for _ in range(150):
                    if any(battle.army_status()["destroyed"]):
                        break
                    obs, actions = (
                        battle.observe(side),
                        tuple(battle.available_actions()),
                    )
                    a, b = [agent.act(obs, actions) for agent in agents[side]]
                    self.assertEqual(a, b)
                    self.assertTrue(battle.step(a).ok)
                    if a.kind == "end_turn":
                        side = 1 - side
            finally:
                for pair in agents:
                    for agent in pair:
                        agent.close()

    def test_pipeline_migration_rejects_learning_and_model_changes(self):
        from civ5rl.migration import validate_pipeline_migration

        old = load_config("configs/rl-v1.json")
        new = load_config("configs/rl-v1-throughput.json")
        identity = {"corpus_hash": "same", "source_hashes": {"civ5rl/model.py": "same"}}
        validate_pipeline_migration(old, new, identity, identity)
        for section, key, value in (
            ("ppo", "epochs", 8),
            ("numerics", "tf32", True),
            ("reward", "win", 2),
            ("collection", "min_learner_decisions_per_update", 16384),
        ):
            changed = deepcopy(new)
            changed[section][key] = value
            with self.assertRaises(ValueError):
                validate_pipeline_migration(old, changed, identity, identity)
        changed_identity = {
            **identity,
            "source_hashes": {"civ5sim/engine.py": "changed"},
        }
        with self.assertRaises(ValueError):
            validate_pipeline_migration(old, new, identity, changed_identity)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "Requires CUDA BF16",
    )
    def test_bf16_spatial_results_stable_across_inference_batch_sizes(self):
        from civ5rl.model import batch_inputs

        spec = self.specs(1)[0]
        encoder = ObservationEncoder(
            context_for(spec["scenario"], spec["armies"], 0, 3, 0)
        )
        with Battle(spec["scenario"]) as battle:
            item = encoder.encode(
                battle.observe(0), ordered_actions(battle.available_actions())
            )
        items = []
        for i in range(65):
            value = deepcopy(item)
            value.cells[:, 35:40] = np.random.default_rng(i).random(
                (100, 5), dtype=np.float32
            )
            items.append(value)
        policy = Policy().cuda().eval()
        policy.enable_amp(spatial_batch_size=32)
        policy.enable_compile()
        with torch.no_grad():
            reference = policy.encode_spatial(batch_inputs(items, "cuda"))
            for size in (1, 7, 32):
                parts = [
                    policy.encode_spatial(batch_inputs(items[i : i + size], "cuda"))
                    for i in range(0, len(items), size)
                ]
                for index in (0, 1):
                    torch.testing.assert_close(
                        torch.cat([x[index] for x in parts]),
                        reference[index],
                        rtol=0,
                        atol=2e-6,
                    )


if __name__ == "__main__":
    unittest.main()
