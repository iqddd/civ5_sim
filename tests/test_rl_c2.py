"""Focused correctness checks for C2 architecture, hex distances, and time encoding."""

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from civ5rl.core.model import (
    Policy,
    batch_inputs,
    hex_distance_buckets_from_neighbors,
    unit_cell_mask_from_cells,
)
from civ5rl.core.observation import (
    ObservationEncoder,
    convert_globals_time_to_c2,
    distance,
    ordered_actions,
)
from civ5rl.core.schema import GLOBAL
from civ5rl.engine.imitation_stream import (
    PreparedDemonstrations,
    prepare_inputs,
)
from civ5rl.experiments.imitation_target_scale import (
    _format_corpus_entry,
    load_config,
)
from civ5rl.protocols.corpus import context_for
from civ5rl.protocols.imitation import dump_episode
from civ5rl.protocols.memory import flat_map
from civ5rl.protocols.ppo_target_scale import (
    HOLDOUT_WORLDS,
    development_512_cases,
    holdout_126_cases,
)
from civ5rl.provenance.common import ROOT, configure, read_json
from civ5sim import Action, Battle


def make_scenario(size=10, row_parity=0):
    m = flat_map(size)
    m["row_parity"] = row_parity
    return {
        "schema": 1,
        "kind": "civ5sim.scenario",
        "map": m,
        "fog": True,
        "random_damage": True,
        "seed": 17,
        "active_side": 0,
        "units": [
            {"type": "warrior", "side": 0, "tile": 11},
            {"type": "archer", "side": 0, "tile": 21},
            {"type": "horseman", "side": 1, "tile": size * size - 12},
            {"type": "archer", "side": 1, "tile": size * size - 2},
        ],
    }


def make_context(s, side=0, max_rounds=30):
    armies = [
        [u["type"] for u in s["units"] if u["side"] == side_] for side_ in (0, 1)
    ]
    return context_for(s, armies, side, max_rounds, 42)


class C2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure(17, "cpu")

    def test_c2_no_hidden_enemy_leakage_in_cells_attention_and_outputs(self):
        """Hidden enemy positions/HP must have zero effect on C2 inputs, attention, and outputs."""
        left = make_scenario(12, row_parity=0)
        right = make_scenario(12, row_parity=0)
        # Move hidden enemy units far in the fog of war and alter their HP/moves.
        right["units"][2].update(tile=115, hp=19, moves=0)
        right["units"][3].update(tile=130, hp=64, moves=1)
        right["map"]["metadata"] = {"hidden_seed": 999}

        policy = Policy("C2").eval()
        self.assertEqual(sum(p.numel() for p in policy.parameters()), 2166722)
        encoders = [
            ObservationEncoder(make_context(s, side=0, max_rounds=30), "C2")
            for s in (left, right)
        ]

        with Battle(left) as a, Battle(right) as b:
            hidden_states = [None, None]
            for decision in range(3):
                observations = [x.observe(0) for x in (a, b)]
                # On decision 1, inject a identical visible enemy contact that disappears on decision 2
                if decision == 1:
                    for obs in observations:
                        obs["visibility"][23] = True
                        obs["units"].append(
                            {
                                "id": 99,
                                "type": "horseman",
                                "side": 1,
                                "tile": 23,
                                "hp": 80,
                                "fortified": False,
                            }
                        )
                actions = [
                    ordered_actions(x.available_actions()) for x in (a, b)
                ]
                self.assertEqual(observations[0], observations[1])
                self.assertEqual(actions[0], actions[1])
                encoded = [
                    e.encode(o, acts)
                    for e, o, acts in zip(encoders, observations, actions)
                ]
                np.testing.assert_array_equal(encoded[0].cells, encoded[1].cells)
                np.testing.assert_array_equal(
                    encoded[0].globals, encoded[1].globals
                )

                inputs = [batch_inputs([enc], "cpu") for enc in encoded]
                masks = [
                    unit_cell_mask_from_cells(inp["cells"], inp["cell_mask"])
                    for inp in inputs
                ]
                self.assertTrue(torch.equal(masks[0], masks[1]))
                # Own units are on tiles 11 and 21; on decision >= 1 tile 23 is visible or remembered contact
                expected_tiles = {11, 21} if decision == 0 else {11, 21, 23}
                active_tiles = set(
                    torch.nonzero(masks[0][0], as_tuple=False)
                    .squeeze(-1)
                    .tolist()
                )
                self.assertEqual(active_tiles, expected_tiles)

                with torch.no_grad():
                    for idx in (0, 1):
                        inp = inputs[idx]
                        x = policy.projection(inp["cells"]) * inp[
                            "cell_mask"
                        ].unsqueeze(-1)
                        for block in policy.blocks:
                            x = block(x, inp["neighbors"], inp["cell_mask"])
                        attn_out, attn_weights, unit_mask = (
                            policy.unit_attention.forward_with_weights(
                                x,
                                inp["cells"],
                                inp["neighbors"],
                                inp["cell_mask"],
                            )
                        )
                        self.assertTrue(torch.equal(unit_mask, masks[idx]))
                        if idx == 0:
                            ref_attn_out, ref_attn_weights = (
                                attn_out,
                                attn_weights,
                            )
                        else:
                            self.assertTrue(torch.equal(ref_attn_out, attn_out))
                            self.assertTrue(
                                torch.equal(ref_attn_weights, attn_weights)
                            )
                            # Zero attention weight on non-unit key columns
                            non_unit_cols = ~masks[idx][0]
                            self.assertEqual(
                                attn_weights[0, :, :, non_unit_cols]
                                .abs()
                                .max()
                                .item(),
                                0.0,
                            )

                    out_l = policy(inputs[0], hidden_states[0])
                    out_r = policy(inputs[1], hidden_states[1])
                    self.assertTrue(torch.equal(out_l[0], out_r[0]))
                    self.assertTrue(torch.equal(out_l[1], out_r[1]))
                    self.assertTrue(torch.equal(out_l[2], out_r[2]))
                    hidden_states = [out_l[2], out_r[2]]

                action = (
                    Action("skip", 0) if decision == 0 else Action("end_turn")
                )
                for e, o in zip(encoders, observations):
                    if decision == 1:
                        o = dict(o)
                        o["units"] = [u for u in o["units"] if u["id"] != 99]
                    e.remember_action(action, o)
                for battle in (a, b):
                    battle.step(action)
                    if action.kind == "end_turn":
                        battle.step(Action("end_turn"))

    def test_hex_distance_buckets_match_exact_offset_hex_distance(self):
        """All-pairs hex distance buckets from neighbors must match exact offset-coordinate distance."""
        for size in (10, 12, 16):
            for parity in (0, 1):
                s = make_scenario(size, row_parity=parity)
                m = s["map"]
                enc = ObservationEncoder(make_context(s))
                with Battle(s) as b:
                    obs = b.observe(0)
                    actions = ordered_actions(b.available_actions())
                step = enc.encode(obs, actions)
                for min_cells in (0, 256):
                    inp = batch_inputs([step], "cpu", min_cells=min_cells)
                    buckets = hex_distance_buckets_from_neighbors(
                        inp["neighbors"], inp["cell_mask"]
                    )[0].numpy()
                    n_tiles = size * size
                    expected = np.zeros((n_tiles, n_tiles), dtype=np.int64)
                    for i in range(n_tiles):
                        for j in range(n_tiles):
                            expected[i, j] = min(distance(m, i, j), 15)
                    np.testing.assert_array_equal(
                        buckets[:n_tiles, :n_tiles], expected
                    )
                    if min_cells > n_tiles:
                        self.assertTrue(
                            np.all(buckets[n_tiles:, :] == 0)
                            and np.all(buckets[:, n_tiles:] == 0)
                        )

    def test_c2_bounded_time_encoding_and_lossless_npz_conversion(self):
        """C2 time encoding must stay in [0, 1] for max_rounds=30, 60, 100 and match converted C globals."""
        idx_turn = GLOBAL["turn"]
        idx_rem = GLOBAL["max_rounds"]
        for max_rounds in (30, 60, 100):
            s = make_scenario(10)
            enc_c = ObservationEncoder(
                make_context(s, side=0, max_rounds=max_rounds), "C"
            )
            enc_c2 = ObservationEncoder(
                make_context(s, side=0, max_rounds=max_rounds), "C2"
            )
            with Battle(s) as b:
                obs = b.observe(0)
                actions = ordered_actions(b.available_actions())
            for turn in (0, 1, 15, 29, 30, 59, 60, 61, 120, 199, 200):
                obs_t = dict(obs, turn=turn)
                g_c = enc_c.encode(obs_t, actions).globals
                g_c2 = enc_c2.encode(obs_t, actions).globals
                converted = convert_globals_time_to_c2(g_c)
                np.testing.assert_allclose(
                    converted, g_c2, rtol=0, atol=1e-6
                )
                self.assertGreaterEqual(float(g_c2[idx_turn]), 0.0)
                self.assertLessEqual(float(g_c2[idx_turn]), 1.0)
                self.assertGreaterEqual(float(g_c2[idx_rem]), 0.0)
                self.assertLessEqual(float(g_c2[idx_rem]), 1.0)
                expected_elapsed = min(turn / 120.0, 1.0)
                expected_remaining = min(
                    max(0.0, (2 * max_rounds - turn) / 120.0), 1.0
                )
                self.assertAlmostEqual(
                    float(g_c2[idx_turn]), expected_elapsed, places=6
                )
                self.assertAlmostEqual(
                    float(g_c2[idx_rem]), expected_remaining, places=6
                )

        # Verify PreparedDemonstrations converts .npz on load for variant="C2" while keeping "C" intact
        s = make_scenario(10)
        with Battle(s) as b:
            obs0 = b.observe(0)
            legal0 = tuple(b.available_actions())
        obs1 = dict(obs0, turn=17)
        episode = {
            "match_id": 0,
            "side": 0,
            "context": {
                "map": s["map"],
                "side": 0,
                "armies": [["warrior", "archer"], ["horseman", "archer"]],
                "max_rounds": 30,
                "fog": True,
                "random_damage": True,
            },
            "decisions": [
                {
                    "observation": obs0,
                    "actions": [asdict(a) for a in legal0],
                    "chosen": 0,
                    "supervised": True,
                },
                {
                    "observation": obs1,
                    "actions": [asdict(a) for a in legal0],
                    "chosen": 0,
                    "supervised": True,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ep_path = root / "data/match-000000/demonstration-side-0.json.gz"
            dump_episode(ep_path, episode)
            entry = _format_corpus_entry(
                {
                    "match_id": 0,
                    "side": 0,
                    "positions": 2,
                    "labels": 2,
                    "forced": 0,
                    "supervised": [True, True],
                },
                "old",
                "data/match-000000/demonstration-side-0.json.gz",
                10,
                2,
                sha256="dummy",
            )
            prep = prepare_inputs(root, [entry], workers=1)["entries"]
            cache_c = PreparedDemonstrations(
                root, prep, "cpu", 64 * 1024**2, variant="C"
            )
            cache_c2 = PreparedDemonstrations(
                root, prep, "cpu", 64 * 1024**2, variant="C2"
            )
            arr_c = cache_c.get_arrays(0)["globals"]
            arr_c2 = cache_c2.get_arrays(0)["globals"]
            self.assertAlmostEqual(float(arr_c[1, idx_turn]), 17 / 60.0, places=6)
            self.assertAlmostEqual(float(arr_c[1, idx_rem]), 30 / 60.0, places=6)
            self.assertAlmostEqual(
                float(arr_c2[1, idx_turn]), 17 / 120.0, places=6
            )
            self.assertAlmostEqual(
                float(arr_c2[1, idx_rem]), (60 - 17) / 120.0, places=6
            )

    def test_development_512_and_holdout_126_disjointness(self):
        """Development-512 must be disjoint from dev-128/conf-256; Holdout-126 must use only the 7 holdout worlds."""
        cfg = load_config(ROOT / "configs/rl-cp-imitation-target-c2.json")
        orig = read_json(ROOT / cfg["target_manifest"])
        base = orig["base"]
        existing = [*orig["development"], *orig["confirmation"]]
        dev512 = development_512_cases(cfg, base, existing_cases=existing)
        self.assertEqual(len(dev512), 512)

        def layout_sig(case):
            s = case["scenario"]
            m = s["map"]
            return (
                int(case["parent_world"]),
                int(m["width"]),
                int(m["height"]),
                int(m.get("crop", {}).get("x", -1)),
                int(m.get("crop", {}).get("y", -1)),
                tuple(
                    (u["side"], u["type"], int(u["tile"])) for u in s["units"]
                ),
            )

        existing_sigs = {layout_sig(c) for c in existing}
        dev512_sigs = {layout_sig(c) for c in dev512}
        self.assertFalse(existing_sigs & dev512_sigs)
        self.assertEqual(len(dev512_sigs), 256)

        corpus_manifest = read_json(ROOT / "artifacts/rl/corpus/manifest.json")
        holdout126 = holdout_126_cases(
            cfg, corpus_manifest, training_worlds=base["worlds"]["training"]
        )
        self.assertEqual(len(holdout126), 126)
        self.assertEqual(
            {int(c["parent_world"]) for c in holdout126}, set(HOLDOUT_WORLDS)
        )


if __name__ == "__main__":
    unittest.main()
