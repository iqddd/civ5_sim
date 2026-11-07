"""Ownership, public-history equivalence and asynchronous complete episodes."""

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from civ5rl.collector import Collector
from civ5rl.common import configure
from civ5rl.corpus import context_for
from civ5rl.model import Policy
from civ5rl.observation import ObservationEncoder, ordered_actions
from civ5rl.ppo import encode_episode
from civ5rl.shared import EncodedSlots
from test_rl import scenario

from civ5rl.config import load_config
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
        specs = self.specs(7)
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
                # Partial replies can otherwise silently lose, duplicate, or mix a slot's history.
                modes = (
                    [
                        ("barrier", 0, True, None),
                        ("ready", 0, True, None),
                        ("ready", 1, True, None),
                        ("ready", 1, True, 4),
                        ("ready", 1, True, 3),
                        ("streaming", 1, True, 3),
                        ("streaming", 2, True, 2),
                    ]
                    if variant == "C"
                    else [
                        ("ready", 0, True, None),
                        ("ready", 1, False, None),
                        ("streaming", 1, False, 3),
                    ]
                )
                for scheduler, reply_size, worker_encoding, workers in modes:
                    config["performance"]["collection_scheduler"] = (
                        "ready" if scheduler == "streaming" else scheduler
                    )
                    config["performance"]["worker_reply_size"] = reply_size
                    config["performance"]["worker_encoding"] = worker_encoding
                    with Collector(
                        config,
                        None,
                        "cpu",
                        Path(tmp),
                        cpu_workers=workers,
                        reply_size=reply_size,
                        num_envs=5
                        if scheduler == "streaming"
                        else (6 if workers == 3 else None),
                        collection_scheduler=scheduler,
                    ) as c:
                        if workers is not None:
                            c.inference_min_batch = 4
                            c.inference_wait_seconds = 0.002
                        episodes, next_episode, _times = c.collect(
                            policy,
                            None,
                            0,
                            fixed_specs=specs,
                            trusted_manifest=manifest,
                        )
                        # A single-slot wave leaves whole workers idle; the next
                        # full wave must reset all slots and recurrent histories.
                        short_manifest = case_manifest(
                            [specs[0]],
                            source="test",
                            split="train",
                            namespace="unit_test",
                        )
                        short_specs = bound_cases(short_manifest)
                        short, short_next, _ = c.collect(
                            policy,
                            None,
                            100,
                            fixed_specs=short_specs,
                            trusted_manifest=short_manifest,
                        )
                        repeated, repeated_next, _ = c.collect(
                            policy,
                            None,
                            200,
                            fixed_specs=specs,
                            trusted_manifest=manifest,
                        )
                    self.assertEqual(next_episode, 7)
                    self.assertEqual(short_next, 101)
                    self.assertEqual(repeated_next, 207)
                    self.assertEqual(len(episodes), 7)
                    self.assertEqual(
                        [ep["spec"]["episode_id"] for ep in episodes], list(range(7))
                    )
                    self.assertEqual(
                        short[0]["joint_commands"], episodes[0]["joint_commands"]
                    )
                    self.assertEqual(
                        [ep["joint_commands"] for ep in repeated],
                        [ep["joint_commands"] for ep in episodes],
                    )
                    for original, again in zip(episodes, repeated):
                        np.testing.assert_allclose(
                            original["old_value"],
                            again["old_value"],
                            rtol=1e-5,
                            atol=1e-5,
                        )
                        for before, after in zip(
                            original["old_hidden"], again["old_hidden"]
                        ):
                            if before is None:
                                self.assertIsNone(after)
                            else:
                                # Ready batching changes GEMM shapes. FP32 BLAS
                                # rounding varies across Linux/Windows backends.
                                np.testing.assert_allclose(
                                    before, after, rtol=1e-5, atol=1e-5
                                )
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

    def test_training_tail_wave_preserves_complete_episode_prefix_and_history(self):
        # Fixed-case checks miss training-only overcollection and stale GRU state
        # when a reduced final wave is followed by another collect on the same slots.
        from civ5rl.provenance import bound_cases, case_manifest

        config = load_config("configs/rl-v1-throughput.json")
        manifest = case_manifest(
            self.specs(8), source="test", split="train", namespace="unit_test"
        )
        specs = bound_cases(manifest)
        policy = Policy("C").eval()

        class Factory:
            def __init__(self):
                self.issued = []

            def make(self, episode_id, stage):
                self.issued.append(episode_id)
                return specs[episode_id]

            def validate_specs(self, values):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            for scheduler, reply_size in (("barrier", 0), ("ready", 0), ("ready", 1)):
                with self.subTest(scheduler=scheduler, reply_size=reply_size):
                    factory = Factory()
                    with Collector(
                        config,
                        factory,
                        "cpu",
                        Path(tmp),
                        num_envs=6,
                        cpu_workers=3,
                        reply_size=reply_size,
                        collection_scheduler=scheduler,
                    ) as collector:
                        reference, _, _ = collector.collect(
                            policy,
                            None,
                            0,
                            fixed_specs=specs,
                            trusted_manifest=manifest,
                        )
                        target = sum(len(ep["chosen"]) for ep in reference[:6]) + 1
                        episodes, following, timing = collector.collect(
                            policy, None, 0, max_decisions=target, episode_length_hint=1
                        )
                        repeated, repeated_next, repeated_timing = collector.collect(
                            policy, None, following, max_decisions=1
                        )
                    self.assertEqual(timing["wave_sizes"], [6, 1])
                    self.assertEqual(timing["target_decisions"], target)
                    self.assertEqual(following, 7)
                    self.assertEqual(repeated_next, 8)
                    self.assertEqual(repeated_timing["wave_sizes"], [1])
                    self.assertEqual(factory.issued, list(range(8)))
                    self.assertEqual(
                        timing["rollout_overshoot"],
                        sum(len(ep["chosen"]) for ep in episodes) - target,
                    )
                    for actual, expected in zip(episodes + repeated, reference):
                        self.assertEqual(
                            actual["joint_commands"], expected["joint_commands"]
                        )
                        self.assertIn("result", actual)
                        self.assertIsNone(actual["old_hidden"][0])
                        for before, after in zip(
                            actual["old_hidden"][1:], expected["old_hidden"][1:]
                        ):
                            np.testing.assert_allclose(
                                before, after, rtol=1e-5, atol=1e-5
                            )

    def test_streaming_budget_drains_every_started_episode(self):
        # Dropping slow episodes biases the rollout, while stale slot history
        # can silently train PPO on another episode's recurrent state.
        from time import perf_counter

        config = load_config("configs/rl-v1-throughput.json")
        specs = self.specs(40)

        class Factory:
            def make(self, episode_id, stage):
                return specs[episode_id]

            def validate_specs(self, values):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            policy = Policy("C").eval()
            with Collector(
                config,
                Factory(),
                "cpu",
                Path(tmp),
                num_envs=5,
                cpu_workers=2,
                reply_size=1,
                collection_scheduler="streaming",
            ) as collector:
                empty, following, _ = collector.collect(
                    policy,
                    None,
                    0,
                    deadline=perf_counter() - 1,
                )
                self.assertEqual((empty, following), ([], 0))
                episodes, following, timing = collector.collect(
                    policy,
                    None,
                    0,
                    max_decisions=12,
                    episode_length_hint=1,
                )
        self.assertEqual(
            [ep["spec"]["episode_id"] for ep in episodes], list(range(following))
        )
        self.assertEqual(len(episodes), timing["admitted_episodes"])
        self.assertGreaterEqual(sum(len(ep["chosen"]) for ep in episodes), 12)
        self.assertEqual(timing["admission_stop_reason"], "target")
        for episode in episodes:
            self.assertIn("result", episode)
            if episode["old_hidden"]:
                self.assertIsNone(episode["old_hidden"][0])

    def test_large_streaming_refills_preserve_all_episode_histories(self):
        # Large scenario refills and replies can fill both directions of a Pipe.
        # The former deadlock only appears with several slots on one worker.
        from civ5rl.provenance import bound_cases, case_manifest

        specs = self.specs(40)
        manifest = case_manifest(
            specs, source="test", split="train", namespace="unit_test"
        )
        specs = bound_cases(manifest)
        config = load_config("configs/rl-v1-throughput.json")
        policy = Policy("A").eval()
        results = []
        with tempfile.TemporaryDirectory() as tmp:
            for scheduler in ("ready", "streaming"):
                with Collector(
                    config,
                    None,
                    "cpu",
                    Path(tmp),
                    num_envs=16,
                    cpu_workers=1,
                    reply_size=1,
                    collection_scheduler=scheduler,
                ) as collector:
                    episodes, following, _ = collector.collect(
                        policy,
                        None,
                        0,
                        fixed_specs=specs,
                        trusted_manifest=manifest,
                    )
                self.assertEqual(following, 40)
                self.assertEqual(
                    [ep["spec"]["episode_id"] for ep in episodes], list(range(40))
                )
                results.append([ep["joint_commands"] for ep in episodes])
        self.assertEqual(*results)

    def test_container_cpu_limits_include_ancestors_and_fractional_quota(self):
        # Host CPU counts can silently choose an unusable parallelism in Docker.
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "collection_hardware",
            Path(__file__).resolve().parents[1] / "tools/rl_collection_hardware.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.effective_cpu_budget(range(32), 6.4, range(16)), 6.4)
        self.assertEqual(module.effective_cpu_budget(range(8), 16), 8)
        self.assertEqual(module.effective_cpu_budget(range(32), cpuset=range(8, 16)), 8)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proc = root / "proc/self"
            proc.mkdir(parents=True)
            mount = root / "unified"
            child = mount / "parent/child"
            child.mkdir(parents=True)
            (proc / "cgroup").write_text("0::/parent/child\n")
            (proc / "mountinfo").write_text(
                f"1 0 0:1 / {mount.as_posix()} rw - cgroup2 cgroup rw\n"
            )
            (child / "cpu.max").write_text("max 100000")
            (child.parent / "cpu.max").write_text("640000 100000")
            (mount / "cpu.max").write_text("1200000 100000")
            (child / "cpuset.cpus.effective").write_text("2-9,12")
            (child.parent / "memory.max").write_text("1000")
            (child.parent / "memory.current").write_text("400")
            limits = module.cgroup_limits(root / "proc")
            self.assertEqual(limits["cpu_quota"], 6.4)
            self.assertEqual(limits["cpuset"], [*range(2, 10), 12])
            self.assertEqual(limits["memory_available_bytes"], 600)
            # v1 reports the same restrictions in separate controller files.
            (proc / "cgroup").write_text(
                "1:cpu,cpuacct:/parent/child\n2:cpuset:/parent/child\n"
            )
            (proc / "mountinfo").write_text(
                f"1 0 0:1 / {mount.as_posix()} rw - cgroup cgroup rw,cpu,cpuacct,cpuset\n"
            )
            (child / "cpu.cfs_quota_us").write_text("-1")
            (child / "cpu.cfs_period_us").write_text("100000")
            (child.parent / "cpu.cfs_quota_us").write_text("800000")
            (child.parent / "cpu.cfs_period_us").write_text("100000")
            self.assertEqual(module.cgroup_limits(root / "proc")["cpu_quota"], 8)

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
