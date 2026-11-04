"""Information boundaries, numerical PPO behavior, restart and native integration."""

import random
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from civ5rl import checkpoint
from civ5rl.common import configure, read_json, seed_for
from civ5rl.corpus import ScenarioFactory, context_for, validate_corpus
from civ5rl.env import DecisionEnv
from civ5rl.evaluation import require_split
from civ5rl.memory import _history, flat_map
from civ5rl.model import Policy, batch_inputs, distribution_stats
from civ5rl.observation import ObservationEncoder, distance, ordered_actions
from civ5rl.ppo import objective, prefix_hidden, targets
from civ5sim import Action, Battle, MatchConfig, run_match


def scenario():
    return {
        "schema": 1,
        "kind": "civ5sim.scenario",
        "map": flat_map(10),
        "fog": True,
        "random_damage": True,
        "seed": 17,
        "active_side": 0,
        "units": [
            {"type": "warrior", "side": 0, "tile": 11},
            {"type": "archer", "side": 0, "tile": 21},
            {"type": "horseman", "side": 1, "tile": 88},
            {"type": "archer", "side": 1, "tile": 98},
        ],
    }


def context(s, side=0):
    armies = [[u["type"] for u in s["units"] if u["side"] == side_] for side_ in (0, 1)]
    return context_for(s, armies, side, 3, 42)


class RLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure(17)
        cls.config = read_json("configs/rl-v1.json")

    def test_hidden_worlds_same_public_history_inputs_actions_outputs(self):
        left, right = scenario(), scenario()
        right["units"][2].update(tile=78, hp=23, moves=0)
        right["map"]["metadata"] = {"hidden_seed": 999}
        policy = Policy().eval()
        encoders = [ObservationEncoder(context(s)) for s in (left, right)]
        with Battle(left) as a, Battle(right) as b:
            for decision in range(3):
                observations = [x.observe(0) for x in (a, b)]
                actions = [ordered_actions(x.available_actions()) for x in (a, b)]
                self.assertEqual(observations[0], observations[1])
                self.assertEqual(actions[0], actions[1])
                encoded = [
                    e.encode(o, acts)
                    for e, o, acts in zip(encoders, observations, actions)
                ]
                np.testing.assert_array_equal(encoded[0].cells, encoded[1].cells)
                np.testing.assert_array_equal(encoded[0].globals, encoded[1].globals)
                with torch.no_grad():
                    logits, value, _ = policy(batch_inputs(encoded, "cpu"))
                torch.testing.assert_close(logits[0], logits[1], rtol=0, atol=0)
                torch.testing.assert_close(value[0], value[1], rtol=0, atol=0)
                action = Action("skip", 0) if decision == 0 else Action("end_turn")
                for e, o in zip(encoders, observations):
                    e.remember_action(action, o)
                for battle in (a, b):
                    battle.step(action)
                    if action.kind == "end_turn":
                        battle.step(Action("end_turn"))

    def test_geometric_distance_and_rivers_match_kernel(self):
        for parity in (0, 1):
            s = scenario()
            m = s["map"]
            m["row_parity"] = parity
            m["tiles"][0]["rivers"] = [0, 1, 2]
            m["boundary_rivers"] = [{"x": 0, "y": 0, "direction": 4, "flow": 0}]
            e = ObservationEncoder(context(s))
            with Battle(s) as b:
                encoded_tiles = b._encode_tiles()
                for i, tile in enumerate(encoded_tiles):
                    for d in range(6):
                        self.assertEqual(
                            bool(e.static[i, 20 + d]), bool(tile.rivers & (1 << d))
                        )
                        n = e.neighbors[i, d]
                        if n >= 0:
                            self.assertEqual(distance(m, i, int(n)), 1)
                            self.assertEqual(e.neighbors[n, (d + 3) % 6], i)

    def test_contacts_enemy_clock_hidden_death_and_reset(self):
        s = scenario()
        e = ObservationEncoder(context(s))
        with Battle(s) as battle:
            obs = battle.observe(0)
        enemy = {
            "id": 2,
            "type": "horseman",
            "side": 1,
            "tile": 12,
            "hp": 75,
            "fortified": False,
        }
        obs["visibility"][12] = True
        obs["units"].append(enemy)
        actions = (Action("end_turn"),)
        e.encode(obs, actions)
        obs["units"].pop()
        x = e.encode(obs, actions)
        self.assertEqual(x.cells[12, 58 + 4], 1 / 8)
        self.assertEqual(x.cells[12, 63 + 4], 0)
        obs["turn"] = 2
        x = e.encode(obs, actions)
        self.assertEqual(x.cells[12, 63 + 4], 1 / 32)
        self.assertIn(2, e.contacts)
        fresh = ObservationEncoder(context(s))
        self.assertFalse(fresh.contacts)
        for side in (0, 1):
            e = ObservationEncoder(context(s, side))
            self.assertEqual((side + side) // 2, side)

    def test_parameter_count_padding_and_recurrent_replay(self):
        policy = Policy().eval()
        self.assertEqual(sum(p.numel() for p in policy.parameters()), 1375362)
        s = scenario()
        with Battle(s) as b:
            obs, actions = b.observe(0), ordered_actions(b.available_actions())
        e = ObservationEncoder(context(s))
        encoded = [e.encode(obs, actions)] * 5
        with torch.no_grad():
            logits, values, hidden = policy(batch_inputs(encoded, "cpu"), sequence=True)
            prefix = prefix_hidden(policy, encoded, 3, "cpu")
            split_logits, split_values, split_hidden = policy(
                batch_inputs(encoded[3:], "cpu"), prefix, sequence=True
            )
            torch.testing.assert_close(split_logits, logits[3:])
            torch.testing.assert_close(split_values, values[3:])
            torch.testing.assert_close(split_hidden, hidden)
            old = prefix.clone()
            policy.memory.bias_ih_l0.add_(0.1)
            new = prefix_hidden(policy, encoded, 3, "cpu")
            self.assertGreater((old - new).abs().max().item(), 0.01)
        short = deepcopy(encoded[0])
        short.sources, short.targets, short.action_features = (
            short.sources[:1],
            short.targets[:1],
            short.action_features[:1],
        )
        with torch.no_grad():
            single, _, _ = policy(batch_inputs([short], "cpu"))
            padded, _, _ = policy(batch_inputs([short, encoded[0]], "cpu"))
        torch.testing.assert_close(single[0, 0], padded[0, 0])
        self.assertTrue(torch.isneginf(padded[0, 1:]).all())

    def _short_rollout(self, policy):
        from civ5rl.ppo import encode_episode

        s = scenario()
        with Battle(s) as battle:
            observation = battle.observe(0)
            actions = ordered_actions(battle.available_actions())
        spec = {
            "scenario": s,
            "armies": context(s)["armies"],
            "learner_side": 0,
            "max_rounds": 3,
        }
        episodes = []
        for size, reward in ((5, 1), (3, -1), (2, 0)):
            ep = {
                "spec": spec,
                "observations": [observation] * size,
                "actions": [actions] * size,
                "chosen": [actions.index(Action("end_turn"))] * size,
                "old_logp": [0.0] * size,
                "old_value": [0.0] * size,
                "result": {"reward": reward},
            }
            encoded = encode_episode(ep, "C")
            with torch.no_grad():
                logits, value, _ = policy(batch_inputs(encoded, "cpu"), sequence=True)
                ep["old_logp"] = distribution_stats(logits, torch.tensor(ep["chosen"]))[
                    0
                ].tolist()
                ep["old_value"] = value.tolist()
            episodes.append(ep)
        return episodes

    def test_batched_gru_has_no_cross_episode_or_padding_effect(self):
        from civ5rl.ppo_fast import RolloutCache, prefixes

        policy = Policy()
        episodes = self._short_rollout(policy)
        cache = RolloutCache(episodes, "C", "cpu")
        spans = [(0, 0, 2), (1, 1, 3), (0, 3, 5)]
        h = prefixes(policy, cache, spans)
        batch, _, lengths = cache.batch(spans)
        logits, values, last = policy.forward_sequences(batch, lengths, h)
        offset = 0
        for i, (ep, start, stop) in enumerate(spans):
            prefix = prefix_hidden(policy, cache.encoded[ep], start, "cpu")
            expected_logits, expected_values, expected_h = policy(
                batch_inputs(cache.encoded[ep][start:stop], "cpu"),
                prefix,
                sequence=True,
            )
            torch.testing.assert_close(
                logits[offset : offset + stop - start], expected_logits
            )
            torch.testing.assert_close(
                values[offset : offset + stop - start], expected_values
            )
            torch.testing.assert_close(last[:, i : i + 1], expected_h)
            offset += stop - start

    def test_compile_padding_single_decision_and_single_action_is_invisible(self):
        policy = Policy().eval()
        s = scenario()
        with Battle(s) as battle:
            obs = battle.observe(0)
        encoded = ObservationEncoder(context(s)).encode(obs, (Action("end_turn"),))
        batch = batch_inputs([encoded], "cpu")
        with torch.no_grad():
            expected = policy(batch)
            # Execute the compile boundary through eager callables to inspect the
            # padding contract independently of an installed compiler backend.
            policy._spatial_runner, policy._heads_runner = policy.spatial, policy.heads
            actual = policy(batch)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)
        self.assertEqual(actual[0].shape, (1, 1))

    def test_bf16_amp_rejects_cpu_instead_of_silent_fallback(self):
        policy = Policy()
        with self.assertRaises(ValueError):
            policy.enable_amp()

    @unittest.skipUnless(
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported(including_emulation=False),
        "Requires native CUDA BF16",
    )
    def test_bf16_dense_regions_keep_gru_logits_loss_and_gradients_fp32(self):
        policy = Policy().cuda()
        policy.enable_amp()
        observations = {}
        hooks = [
            policy.projection.register_forward_hook(
                lambda module, args, output: observations.update(encoder=output.dtype)
            ),
            policy.action[0].register_forward_hook(
                lambda module, args, output: observations.update(action=output.dtype)
            ),
            policy.blocks[0]
            .mlp[0]
            .register_forward_hook(
                lambda module, args, output: observations.update(hex_dense=output.dtype)
            ),
            policy.memory.register_forward_pre_hook(
                lambda module, args: observations.update(gru_input=args[0].dtype)
            ),
        ]
        s = scenario()
        with Battle(s) as battle:
            obs, actions = (
                battle.observe(0),
                ordered_actions(battle.available_actions()),
            )
        encoder = ObservationEncoder(context(s))
        batch = batch_inputs([encoder.encode(obs, actions)] * 3, "cuda")
        logits, values, hidden = policy(batch, sequence=True)
        indices = torch.zeros(3, dtype=torch.long, device="cuda")
        logp, _, _ = distribution_stats(logits, indices)
        loss = -logp.mean() + values.square().mean()
        loss.backward()
        self.assertEqual(observations["encoder"], torch.float32)
        self.assertEqual(observations["action"], torch.float32)
        self.assertEqual(observations["hex_dense"], torch.bfloat16)
        self.assertEqual(observations["gru_input"], torch.float32)
        for tensor in (logits, values, hidden, logp, loss):
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertTrue(torch.isfinite(tensor).all())
        for parameter in policy.parameters():
            self.assertEqual(parameter.dtype, torch.float32)
            if parameter.grad is not None:
                self.assertEqual(parameter.grad.dtype, torch.float32)
                self.assertTrue(torch.isfinite(parameter.grad).all())
        for handle in hooks:
            handle.remove()

    def test_batched_ppo_weighted_gradients_match_reference_with_partial_chunks(self):
        from civ5rl.ppo import update

        configure(17)
        policy = Policy()
        episodes = self._short_rollout(policy)
        fast = deepcopy(policy)
        config = deepcopy(self.config)
        config["ppo"].update(
            epochs=1,
            sequence_length=3,
            sequences_per_minibatch=8,
            sequences_per_microbatch=4,
        )
        config["performance"] = {"batched_ppo": False}
        # SGD exposes the gradient comparison directly, without Adam's sensitivity
        # to tiny cancellation errors in near-zero gradients.
        optimizer = torch.optim.SGD(policy.parameters(), lr=3e-4)
        fast_optimizer = torch.optim.SGD(fast.parameters(), lr=3e-4)
        a = update(policy, optimizer, episodes, config, 0, "cpu", random.Random(18))
        config["performance"]["batched_ppo"] = True
        b = update(fast, fast_optimizer, episodes, config, 0, "cpu", random.Random(18))
        self.assertAlmostEqual(a["policy_loss"], b["policy_loss"], places=6)
        self.assertAlmostEqual(a["value_loss"], b["value_loss"], places=6)
        for name, tensor in policy.state_dict().items():
            torch.testing.assert_close(
                tensor, fast.state_dict()[name], rtol=2e-5, atol=1e-7
            )

    def test_clone_initial_snapshot_shares_static_internally_public_copies_detach(self):
        with Battle(scenario()) as root, root.clone(log=True) as branch:
            initial = branch.snapshot()
            self.assertIs(branch._initial["map"], root._map)
            public = branch.snapshot()
            public["map"]["tiles"][0]["terrain"] = "desert"
            self.assertEqual(root.snapshot()["map"]["tiles"][0]["terrain"], "grass")
            branch.step(Action("end_turn"))
            replay = branch.replay()
            replay["initial"]["map"]["tiles"][0]["terrain"] = "snow"
            self.assertEqual(branch._map["tiles"][0]["terrain"], "grass")
            branch.reset()
            self.assertEqual(branch.snapshot(), initial)

    def test_fused_hex_projection_matches_directional_reference_gradients(self):
        from torch.nn import functional as f

        from civ5rl.model import HexBlock

        torch.manual_seed(42)
        block = HexBlock()
        for weight in block.weight:
            torch.nn.init.xavier_uniform_(weight)
        x = torch.randn(2, 8, 128, requires_grad=True)
        neighbors = torch.randint(-1, 8, (2, 8, 6))
        mask = torch.ones(2, 8, dtype=torch.bool)
        optimized = block(x, neighbors, mask)
        message = f.linear(x, block.weight[0], block.bias)
        for direction in range(6):
            idx = neighbors[:, :, direction]
            gathered = x[torch.arange(2)[:, None], idx.clamp_min(0)] * (
                idx >= 0
            ).unsqueeze(-1)
            message = message + f.linear(gathered, block.weight[direction + 1])
        u = x + f.gelu(block.norm1(message))
        reference = u + block.mlp(block.norm2(u))
        torch.testing.assert_close(optimized, reference, atol=2e-6, rtol=2e-5)
        grad_a = torch.autograd.grad(
            optimized.square().mean(), (x, block.weight), retain_graph=True
        )
        grad_b = torch.autograd.grad(reference.square().mean(), (x, block.weight))
        for a, b in zip(grad_a, grad_b):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)

    def test_analytic_ppo_loss_and_weighted_accumulation(self):
        logits = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, -torch.inf]], requires_grad=True
        )
        values = torch.tensor([0.2, -0.3, 0.1], requires_grad=True)
        indices = torch.tensor([0, 1, 0])
        old = distribution_stats(logits.detach(), indices)[0]
        returns = torch.tensor([1.0, -1.0, 0.0])
        advantages = torch.tensor([0.5, -0.5, 0.0])
        loss, _stats = objective(logits, values, indices, old, returns, advantages, 0)
        expected = 0.5 * ((values - returns).square().mean())
        torch.testing.assert_close(loss, expected)
        loss.backward()
        full_grad = logits.grad.clone()
        logits.grad = None
        values.grad = None
        for start, stop in ((0, 2), (2, 3)):
            part, _ = objective(
                logits[start:stop],
                values[start:stop],
                indices[start:stop],
                old[start:stop],
                returns[start:stop],
                advantages[start:stop],
                0,
            )
            (part * ((stop - start) / 3)).backward()
        torch.testing.assert_close(logits.grad, full_grad)
        self.assertTrue(torch.isfinite(loss))

    def test_terminal_targets_sign_and_normalization(self):
        eps = [
            {"result": {"reward": r}, "chosen": [0, 0], "old_value": [0, 0]}
            for r in (-1, 0, 1)
        ]
        adv = targets(eps)
        self.assertLess(adv[0][0], 0)
        self.assertEqual(adv[1][0], 0)
        self.assertGreater(adv[2][0], 0)
        self.assertAlmostEqual(np.concatenate(adv).std(), 1, places=6)

    def test_precision_specific_replay_checks_reject_systematic_drift(self):
        from civ5rl.ppo_fast import validate_behavior_replay

        self.assertEqual(
            validate_behavior_replay(2.8e-8, 0.0016, amp_enabled=True, numerics={}),
            0.005,
        )
        with self.assertRaises(RuntimeError):
            validate_behavior_replay(2.8e-8, 0.0016, amp_enabled=False, numerics={})
        # All logp shifted by 0.003 passes the outlier bound, but fails mean KL.
        with self.assertRaises(RuntimeError):
            validate_behavior_replay(4.5e-6, 0.003, amp_enabled=True, numerics={})
        with self.assertRaises(RuntimeError):
            validate_behavior_replay(0, 0.006, amp_enabled=True, numerics={})

    def test_full_distribution_audit_measures_mass_and_rejects_real_drift(self):
        from civ5rl.ppo_fast import categorical_distance, validate_behavior_replay

        p = torch.tensor([[0.999, 0.001, 0], [1, 0, 0]], dtype=torch.float64)
        q = torch.tensor([[0.99900685, 0.00099315, 0], [1, 0, 0]], dtype=torch.float64)
        mask = p > 0
        kl, tv = categorical_distance(p.log(), q.log(), mask)
        torch.testing.assert_close(
            tv, torch.tensor([0.00000685, 0], dtype=torch.float64), atol=1e-14, rtol=0
        )
        self.assertTrue(torch.isfinite(kl).all())
        self.assertGreater((p[0, 1].log() - q[0, 1].log()).abs().item(), 0.005)
        limits = {"require_full_behavior_distribution": True}
        # Actual independent audit of the pilot's failed update 14.
        self.assertIsNone(
            validate_behavior_replay(
                6.99e-9,
                0.006882,
                amp_enabled=True,
                numerics=limits,
                full_kl=4.60e-9,
                max_total_variation=0.000705,
            )
        )
        for full_kl, max_tv in ((2e-6, 0.001), (1e-9, 0.006), (float("nan"), 0)):
            with self.assertRaises(RuntimeError):
                validate_behavior_replay(
                    0,
                    0,
                    amp_enabled=True,
                    numerics=limits,
                    full_kl=full_kl,
                    max_total_variation=max_tv,
                )
        with self.assertRaises(RuntimeError):
            validate_behavior_replay(0, 0, amp_enabled=True, numerics=limits)

    def test_behavior_distribution_cannot_replace_chosen_probability(self):
        from civ5rl.ppo_fast import RolloutCache

        policy = Policy()
        episodes = self._short_rollout(policy)
        for ep in episodes:
            ep["old_log_probs"] = [
                np.full(len(a), -np.log(len(a)), np.float32) for a in ep["actions"]
            ]
        with self.assertRaisesRegex(ValueError, "chosen logp"):
            RolloutCache(episodes, "C", "cpu")

    def test_audit_migration_rejects_learning_model_and_environment_changes(self):
        from civ5rl.migration import validate_audit_migration

        old = {"spec_version": "1.3", "ppo": {"epochs": 4}}
        new = {
            **old,
            "spec_version": "1.4",
            "numerics": {"require_full_behavior_distribution": True},
        }
        # Include the previously absent numerics dict explicitly for leaf diffs.
        old["numerics"] = {}
        identity = {"corpus_hash": "same", "source_hashes": {"civ5rl/model.py": "same"}}
        validate_audit_migration(old, new, identity, identity)
        changed = deepcopy(new)
        changed["ppo"]["epochs"] = 8
        with self.assertRaises(ValueError):
            validate_audit_migration(old, changed, identity, identity)
        for candidate in (
            {**identity, "corpus_hash": "different"},
            {**identity, "source_hashes": {"civ5rl/model.py": "different"}},
        ):
            with self.assertRaises(ValueError):
                validate_audit_migration(old, new, identity, candidate)

    def test_control_follows_recorded_continuation_and_rejects_cycle(self):
        from civ5rl.common import write_json
        from tools.control_rl import resolve_directory

        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / "first", Path(tmp) / "second"
            second.mkdir()
            write_json(
                first / "continuation.json",
                {"kind": "rl01.continuation", "directory": str(second)},
            )
            self.assertEqual(resolve_directory(first), second.resolve())
            write_json(
                second / "continuation.json",
                {"kind": "rl01.continuation", "directory": str(first)},
            )
            with self.assertRaises(ValueError):
                resolve_directory(first)

    def test_env_and_prepared_match_same_commands_and_result(self):
        s = scenario()
        armies = context(s)["armies"]
        for side in (0, 1):
            spec = {
                "scenario": s,
                "armies": armies,
                "learner_side": side,
                "max_rounds": 3,
                "max_calls_per_turn": 2000,
                "opponent": "end_turn",
                "agent_seeds": [1, 2],
            }
            env = DecisionEnv(spec)
            state = env.request()
            while not state["done"]:
                a = Action("end_turn")
                env.applied(a, env.battle.step(a))
                state = env.request()
            config = MatchConfig(s["map"], armies, 3)
            match = run_match(
                config, ["end_turn", "end_turn"], scenario=s, agent_seeds=[1, 2]
            )
            self.assertEqual(state["result"]["reason"], match.result["reason"])
            self.assertEqual(env.history, [x["action"] for x in match.replay["steps"]])
            self.assertEqual(env.battle.snapshot(), match.final)
            env.close()

    def test_reachable_memory_pairs_and_current_only_ablation(self):
        for shared in (False, True):
            for side in (0, 1):
                pair = [
                    _history(0, 2 if shared else 0, side, branch, shared)
                    for branch in (0, 1)
                ]
                self.assertEqual(pair[0]["current"], pair[1]["current"])
                self.assertEqual(pair[0]["actions"], pair[1]["actions"])
                outputs = []
                for case in pair:
                    e = ObservationEncoder(
                        context_for(case["scenario"], case["armies"], side, 60, 17), "A"
                    )
                    for step in case["history"]:
                        e.encode(step["observation"], step["actions"])
                        e.remember_action(step["action"], step["observation"])
                    outputs.append(e.encode(case["current"], case["actions"]))
                np.testing.assert_array_equal(outputs[0].cells, outputs[1].cells)
                np.testing.assert_array_equal(outputs[0].globals, outputs[1].globals)

    def test_checkpoint_restores_next_optimizer_step_and_all_rng(self):
        configure(42)
        policy = Policy("A")
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
        rng = random.Random(73)

        def step():
            optimizer.zero_grad()
            loss = sum(
                (
                    p.square().sum()
                    * (
                        random.random()
                        + float(np.random.rand())
                        + torch.rand(()).item()
                        + rng.random()
                    )
                )
                for p in policy.parameters()
            )
            loss.backward()
            optimizer.step()

        step()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            checkpoint.save(path, policy, optimizer, {"update": 1}, rng, {}, {})
            step()
            expected = deepcopy(policy.state_dict())
            checkpoint.restore(path, policy, optimizer, rng, {}, {})
            step()
            for name, value in policy.state_dict().items():
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0)

    def test_checkpoint_restore_accepts_commit_only_change_and_rejects_execution_changes(
        self,
    ):
        policy = Policy("A")
        optimizer = torch.optim.Adam(policy.parameters())
        rng = random.Random(17)
        identity = {
            "git_commit": "before",
            "source_hashes": {"model.py": "same"},
            "corpus_hash": "same",
            "packages": {"torch": "same"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            checkpoint.save(
                path, policy, optimizer, {"update": 0}, rng, {"seed": 17}, identity
            )
            updated = {**identity, "git_commit": "after"}
            state = checkpoint.restore(
                path, policy, optimizer, rng, {"seed": 17}, updated
            )
            self.assertEqual(state, {"update": 0})
            for key, value in (
                ("source_hashes", {"model.py": "changed"}),
                ("corpus_hash", "changed"),
                ("packages", {"torch": "changed"}),
            ):
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    checkpoint.restore(
                        path,
                        policy,
                        optimizer,
                        rng,
                        {"seed": 17},
                        {**updated, key: value},
                    )
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                checkpoint.restore(path, policy, optimizer, rng, {"seed": 42}, updated)

    def test_complete_cpu_training_matches_interrupted_resume(self):
        from civ5rl.collector import Collector
        from tools.train_rl import run

        corpus = Path("artifacts/rl/corpus/manifest.json")
        if not corpus.exists():
            self.skipTest("Corpus not prepared")
        config = deepcopy(self.config)
        config["collection"].update(
            num_envs=2,
            cpu_workers=1,
            envs_per_worker=2,
            min_learner_decisions_per_update=8,
        )
        config["ppo"].update(
            epochs=1,
            sequence_length=4,
            sequences_per_minibatch=2,
            sequences_per_microbatch=2,
        )
        config["curriculum"] = [{**config["curriculum"][0], "max_rounds": 2}]
        manifest = read_json(corpus)
        with tempfile.TemporaryDirectory() as tmp:
            continuous, resumed = Path(tmp) / "continuous", Path(tmp) / "resumed"
            run(
                config,
                manifest,
                continuous,
                device="cpu",
                budget=100000,
                max_updates=2,
                evaluate_every=False,
                replays=False,
            )
            collect = Collector.collect

            def stop_during_collection(collector, *args, **kwargs):
                result = collect(collector, *args, **kwargs)
                (resumed / "stop.request").touch()
                return result

            with patch.object(Collector, "collect", stop_during_collection):
                run(
                    config,
                    manifest,
                    resumed,
                    device="cpu",
                    budget=100000,
                    max_updates=2,
                    evaluate_every=False,
                    replays=False,
                )
            self.assertEqual(read_json(resumed / "run-result.json")["status"], "paused")
            self.assertEqual(read_json(resumed / "latest.json")["update"], 1)
            first = resumed / "checkpoints/update-000001.pt"
            self.assertTrue(first.exists())
            (resumed / "stop.request").unlink()
            run(
                config,
                manifest,
                resumed,
                device="cpu",
                budget=100000,
                max_updates=1,
                evaluate_every=False,
                replays=False,
                resume="latest",
            )
            a = torch.load(
                continuous / "checkpoints/update-000002.pt", weights_only=False
            )
            b = torch.load(resumed / "checkpoints/update-000002.pt", weights_only=False)
            for name, value in a["model"].items():
                torch.testing.assert_close(value, b["model"][name], rtol=0, atol=0)
            self.assertEqual(a["state"]["next_episode"], b["state"]["next_episode"])
            self.assertEqual(a["state"]["decisions"], b["state"]["decisions"])
            self.assertEqual(a["shuffle_rng"], b["shuffle_rng"])
            self.assertEqual(
                a["optimizer"]["param_groups"], b["optimizer"]["param_groups"]
            )
            for key, values in a["optimizer"]["state"].items():
                for name, value in values.items():
                    torch.testing.assert_close(
                        value, b["optimizer"]["state"][key][name], rtol=0, atol=0
                    )
            torch.testing.assert_close(a["torch_rng"], b["torch_rng"], rtol=0, atol=0)
            self.assertEqual(a["python_rng"], b["python_rng"])

    def test_training_run_lock_prevents_concurrent_writer_and_releases(self):
        from civ5rl.control import RunLock, is_active

        with tempfile.TemporaryDirectory() as tmp:
            with RunLock(tmp):
                self.assertTrue(is_active(tmp))
                with self.assertRaises(RuntimeError), RunLock(tmp):
                    pass
            self.assertFalse(is_active(tmp))

    def test_seed_namespaces_and_closed_test(self):
        self.assertEqual(seed_for(17, 2, "battle"), seed_for(17, 2, "battle"))
        self.assertNotEqual(
            seed_for(17, 2, "battle"), seed_for(17, 2, "battle", namespace="validation")
        )
        for split in ("test", "diagnostic_test", "cp_stress"):
            with self.assertRaises(ValueError):
                require_split(self.config, split)
        require_split(self.config, "validation")
        require_split(self.config, "memory_dev")

    def test_resource_metrics_nvml_units_and_unavailable_fields(self):
        from time import monotonic
        from types import SimpleNamespace

        from civ5rl.resources import ResourceMonitor

        monitor = ResourceMonitor("unused")
        monitor.started = monotonic()
        monitor.gpu = object()
        monitor.nvml = SimpleNamespace(
            nvmlDeviceGetUtilizationRates=lambda _: SimpleNamespace(gpu=81, memory=62),
            nvmlDeviceGetMemoryInfo=lambda _: SimpleNamespace(
                used=3 * 2**30, total=16 * 2**30
            ),
            nvmlDeviceGetPowerUsage=lambda _: 132000,
            nvmlDeviceGetPowerManagementDefaultLimit=lambda _: 165000,
            nvmlDeviceGetEnforcedPowerLimit=lambda _: 150000,
        )
        row = monitor.sample()
        self.assertEqual(row["gpu_core_load_percent"], 81)
        self.assertEqual(row["gpu_memory_controller_load_percent"], 62)
        self.assertEqual(row["gpu_power_watts"], 132)
        self.assertEqual(row["gpu_power_percent_tdp"], 80)
        self.assertEqual(row["gpu_enforced_power_limit_watts"], 150)
        self.assertGreater(row["ram_tree_rss_bytes"], 0)
        monitor.gpu = None
        unavailable = monitor.sample()
        self.assertIsNone(unavailable["gpu_core_load_percent"])
        self.assertIsNone(unavailable["gpu_power_percent_tdp"])

    def test_corpus_disjoint_hashes_and_scenario_repeatability(self):
        path = Path("artifacts/rl/corpus/manifest.json")
        if not path.exists():
            self.skipTest("Corpus not prepared")
        manifest = read_json(path)
        validate_corpus(manifest)
        factory = ScenarioFactory(self.config, manifest)
        a = factory.make(17, self.config["curriculum"][0])
        self.assertEqual(a, factory.make(17, self.config["curriculum"][0]))
        from civ5rl.evaluation import validation_specs

        specs = validation_specs(factory, self.config)
        self.assertEqual(len(specs), 256)
        for i in range(0, len(specs), 4):
            group = specs[i : i + 4]
            self.assertEqual({s["learner_side"] for s in group}, {0, 1})
            self.assertEqual({s["opponent"] for s in group}, {"random", "greedy"})
            self.assertTrue(all(s["scenario"] == group[0]["scenario"] for s in group))
        modified = deepcopy(manifest)
        modified["maps"].append({**modified["maps"][0], "split": "test"})
        with self.assertRaises(ValueError):
            validate_corpus(modified)


if __name__ == "__main__":
    unittest.main()
