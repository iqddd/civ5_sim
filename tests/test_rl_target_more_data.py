"""Focused checks for silent risks in rl01.cp_imitation_target_more_data."""

import random
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from civ5rl.engine.imitation import label_minibatches
from civ5rl.engine.imitation_stream import (
    PreparedDemonstrations,
    prepare_episode,
    prepare_inputs,
)
from civ5rl.engine.spans import Span
from civ5rl.experiments.imitation_target_scale import (
    _format_corpus_entry,
    load_config,
    select_best_candidate,
)
from civ5rl.protocols.imitation import dump_episode
from civ5rl.protocols.ppo_target_scale import TargetFactory
from civ5rl.provenance.common import ROOT, configure, read_json
from civ5sim import Action, Battle


class TargetMoreDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure(17, "cpu")
        cls.config = load_config(ROOT / "configs/rl-cp-imitation-target-more-data.json")
        cls.plan = read_json(ROOT / cls.config["target_manifest"])

    def test_old_and_new_histories_with_same_match_and_side_never_collide(self):
        """Old and new episodes with match_id=0, side=0 must stay strictly isolated."""
        factory = TargetFactory(self.config, self.plan["base"])
        case_old = factory.case(
            0, self.plan["base"]["worlds"]["training"][0], 10, 0, True, 0, "old_ns", n=5
        )
        case_new = factory.case(
            0, self.plan["base"]["worlds"]["training"][1], 10, 1, False, 0, "new_ns", n=5
        )
        with Battle(case_old["scenario"]) as b_old:
            obs_old = b_old.observe(0)
            legal_old = tuple(b_old.available_actions())
        with Battle(case_new["scenario"]) as b_new:
            obs_new = b_new.observe(0)
            legal_new = tuple(b_new.available_actions())

        ep_old = {
            "match_id": 0,
            "side": 0,
            "context": {
                "map": case_old["scenario"]["map"],
                "side": 0,
                "armies": case_old["armies"],
                "max_rounds": 30,
                "fog": True,
                "random_damage": True,
            },
            "decisions": [
                {
                    "observation": obs_old,
                    "actions": [asdict(a) for a in legal_old],
                    "chosen": 0,
                    "supervised": True,
                },
                {
                    "observation": obs_old,
                    "actions": [asdict(Action("end_turn"))],
                    "chosen": 0,
                    "supervised": False,
                },
            ],
        }
        ep_new = {
            "match_id": 0,
            "side": 0,
            "context": {
                "map": case_new["scenario"]["map"],
                "side": 0,
                "armies": case_new["armies"],
                "max_rounds": 30,
                "fog": True,
                "random_damage": True,
            },
            "decisions": [
                {
                    "observation": obs_new,
                    "actions": [asdict(a) for a in legal_new],
                    "chosen": len(legal_new) - 1,
                    "supervised": True,
                },
                {
                    "observation": obs_new,
                    "actions": [asdict(a) for a in legal_new],
                    "chosen": 0,
                    "supervised": True,
                },
                {
                    "observation": obs_new,
                    "actions": [asdict(a) for a in legal_new],
                    "chosen": 0,
                    "supervised": True,
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_path = root / "old/data/match-000000/demonstration-side-0.json.gz"
            new_path = root / "new/data/match-000000/demonstration-side-0.json.gz"
            dump_episode(old_path, ep_old)
            dump_episode(new_path, ep_new)

            entry_old = _format_corpus_entry(
                {
                    "match_id": 0,
                    "side": 0,
                    "positions": 2,
                    "labels": 1,
                    "supervised": [True, False],
                },
                "old",
                str(old_path.relative_to(root).as_posix()),
                10,
                5,
            )
            entry_new = _format_corpus_entry(
                {
                    "match_id": 0,
                    "side": 0,
                    "positions": 3,
                    "labels": 3,
                    "supervised": [True, True, True],
                },
                "new",
                str(new_path.relative_to(root).as_posix()),
                10,
                5,
            )
            self.assertNotEqual(entry_old["entry_id"], entry_new["entry_id"])
            prepared_info = prepare_inputs(root, [entry_old, entry_new], workers=1)
            prepared_entries = prepared_info["entries"]
            self.assertEqual(len(prepared_entries), 2)
            self.assertNotEqual(
                prepared_entries[0]["prepared"], prepared_entries[1]["prepared"]
            )
            self.assertIn("old-match-000000-side-0", prepared_entries[0]["prepared"])
            self.assertIn("new-match-000000-side-0", prepared_entries[1]["prepared"])

            cache = PreparedDemonstrations(root, prepared_entries, "cpu")
            self.assertEqual(cache.masks, [[True, False], [True, True, True]])
            _, labels_0, lengths_0 = cache.batch([Span(0, 0, 2)])
            _, labels_1, lengths_1 = cache.batch([Span(1, 0, 3)])
            self.assertEqual(lengths_0, [2])
            self.assertEqual(lengths_1, [3])
            self.assertEqual(labels_0["indices"].tolist(), [0, 0])
            self.assertEqual(
                labels_1["indices"].tolist(), [len(legal_new) - 1, 0, 0]
            )

            # Duplicate entry_id must be rejected immediately.
            with self.assertRaisesRegex(
                ValueError, "Duplicate episode entry keys in BC corpus"
            ):
                prepare_inputs(root, [entry_old, entry_old], workers=1)

            # Mismatched side in the underlying episode file must be rejected.
            bad_entry = {**entry_new, "side": 1, "entry_id": "new:match-000000:side-1"}
            with self.assertRaisesRegex(
                ValueError, "Prepared episode side differs from collection entry"
            ):
                prepare_episode((str(root), bad_entry, str(root / "bad.npz")))

    def test_mid_epoch_resume_processes_each_free_label_once(self):
        """Resuming from saved epoch_order and batch_cursor must neither skip nor repeat labels."""
        masks = [
            [True, False, True, True, False, True],
            [False, True, True, True, False, True, True],
            [True, True, False, True, True, True],
        ]
        chunks = [Span(i, 0, len(m)) for i, m in enumerate(masks)]
        rng = random.Random(17)
        order = list(chunks)
        rng.shuffle(order)

        full_batches = list(label_minibatches(order, masks, size=4))
        self.assertGreaterEqual(len(full_batches), 3)

        # Simulate interrupting after cursor=2 batches and serializing epoch_order.
        cursor = 2
        serialized_order = [[int(s.episode), int(s.start), int(s.stop)] for s in order]
        restored_order = [Span(int(a), int(b), int(c)) for a, b, c in serialized_order]
        resumed_batches = list(label_minibatches(restored_order, masks, size=4))[cursor:]

        combined_batches = full_batches[:cursor] + resumed_batches
        self.assertEqual(combined_batches, full_batches)

        # Verify every supervised position across all episodes is covered exactly once.
        covered = []
        for spans, count in combined_batches:
            batch_supervised = 0
            for ep, start, stop in spans:
                for pos in range(start, stop):
                    if masks[ep][pos]:
                        covered.append((ep, pos))
                        batch_supervised += 1
            self.assertEqual(batch_supervised, count)

        expected = [
            (ep, pos)
            for ep, mask in enumerate(masks)
            for pos, is_sup in enumerate(mask)
            if is_sup
        ]
        self.assertEqual(len(covered), len(expected))
        self.assertEqual(sorted(covered), sorted(expected))

    def test_candidate_selection_and_tie_breaking_rules(self):
        """Verify baseline preference, PPO update 37 preference, and earlier BC checkpoint preference."""
        def point(name, wins, steps=0, epoch=0):
            return {
                "checkpoint": name,
                "policy_hash": f"hash_{name}",
                "epoch": epoch,
                "Adam_steps": steps,
                "runs": {
                    "argmax": {
                        "cp": {
                            "directory": f"eval/{name}",
                            "summary": {
                                "matches": 128,
                                "W": wins,
                                "D": 30,
                                "L": 98 - wins,
                                "win_rate": wins / 128,
                            },
                        }
                    }
                },
            }

        bc_base = point("checkpoints/initial.pt", 27, steps=0, epoch=0)
        ppo_ref = point("checkpoints/reference-update-0037.pt", 32, steps=0, epoch=0)

        # Case 1: New candidates reach at most 32 wins (tie with PPO update 37) -> baseline retained (PPO update 37).
        cand_e2 = point("checkpoints/epoch-02.pt", 31, steps=1950, epoch=2)
        cand_e5 = point("checkpoints/epoch-05.pt", 32, steps=4875, epoch=5)
        res1 = select_best_candidate(bc_base, ppo_ref, [cand_e2, cand_e5])
        self.assertTrue(res1["baseline_retained"])
        self.assertEqual(res1["best"]["checkpoint"], "checkpoints/reference-update-0037.pt")
        self.assertEqual(res1["best_wins"], 32)
        self.assertEqual(res1["required_new_candidate_wins"], 33)

        # Case 2: Two new candidates tie at 34 wins -> earlier checkpoint (epoch 5) is selected.
        cand_e5_34 = point("checkpoints/epoch-05.pt", 34, steps=4875, epoch=5)
        cand_e10_34 = point("checkpoints/epoch-10.pt", 34, steps=9750, epoch=10)
        res2 = select_best_candidate(bc_base, ppo_ref, [cand_e10_34, cand_e5_34])
        self.assertFalse(res2["baseline_retained"])
        self.assertEqual(res2["best"]["checkpoint"], "checkpoints/epoch-05.pt")
        self.assertEqual(res2["best_wins"], 34)

        # Case 3: Baselines tie (e.g. both 30 wins on re-evaluation) -> PPO update 37 is preferred over BC epoch 10.
        bc_tie = point("checkpoints/initial.pt", 30, steps=0, epoch=0)
        ppo_tie = point("checkpoints/reference-update-0037.pt", 30, steps=0, epoch=0)
        res3 = select_best_candidate(bc_tie, ppo_tie, [])
        self.assertTrue(res3["baseline_retained"])
        self.assertEqual(res3["best"]["checkpoint"], "checkpoints/reference-update-0037.pt")

        # Case 4: If BC epoch 10 exceeds PPO update 37 on re-evaluation -> BC epoch 10 becomes the active baseline threshold.
        bc_higher = point("checkpoints/initial.pt", 35, steps=0, epoch=0)
        res4 = select_best_candidate(bc_higher, ppo_ref, [cand_e5_34])
        self.assertTrue(res4["baseline_retained"])
        self.assertEqual(res4["best"]["checkpoint"], "checkpoints/initial.pt")
        self.assertEqual(res4["baseline_wins"], 35)
        self.assertEqual(res4["required_new_candidate_wins"], 36)


if __name__ == "__main__":
    unittest.main()
