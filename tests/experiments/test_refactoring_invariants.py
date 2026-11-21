"""Guard deterministic anchor ordering, paired statistics, and policy identity."""

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from civ5rl.experiments.comparison import rollout_collection, rollout_fit
from civ5rl.experiments.comparison.turn_collection import select_positions
from civ5rl.experiments.comparison.turn_confirmation import paired_statistics
from civ5rl.provenance.identity import policy_hash

EXPECTED_POLICY_HASH = (
    "73bf6194529610c9bfe231681ef203ca9a3e98a2095060607b22f2f62d9285e1"
)
EXPECTED_TURN_POSITIONS = [
    [301, 5, 0],
    [203, 3, 0],
    [102, 4, 0],
    [202, 5, 0],
    [101, 1, 0],
    [201, 4, 0],
    [301, 1, 0],
    [102, 2, 0],
    [202, 1, 0],
]
EXPECTED_PAIRED_STATISTICS = {
    "mean_win_delta": 0.14285714285714285,
    "ci95": [-0.75, 0.44886363636363596],
    "worlds": 4,
    "samples": 257,
    "seed": 3241,
    "conclusion": "uncertain",
}
EXPECTED_ANCHORS = {
    "rollout_fit": {
        "original": "93b3fb036843f5ae5f587b5151a4bfb3004888ab924be8a583b99384c5693d6d",
        "sorted": "242d870917db1247c9a4179caa4f372a228d5d6ceeafc59d7dc882e671578ad3",
    },
    "rollout_collection": {
        "original": "242d870917db1247c9a4179caa4f372a228d5d6ceeafc59d7dc882e671578ad3",
        "sorted": "242d870917db1247c9a4179caa4f372a228d5d6ceeafc59d7dc882e671578ad3",
    },
}


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _policy_state():
    return {
        "encoder.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4).T,
        "value.weight": torch.tensor([999.0]),
        "action.bias": torch.tensor([3.5, -2.0]),
        "value.bias": torch.tensor([998.0]),
    }


def _turn_positions():
    groups = {
        30: {301: (3, 1, 5)},
        10: {102: (2, 4), 101: (6, 1, 4)},
        20: {202: (5, 1), 203: (3,), 201: (4, 2, 3)},
    }
    return {
        world: {
            episode: [{"key": [episode, turn, 0]} for turn in turns]
            for episode, turns in episodes.items()
        }
        for world, episodes in groups.items()
    }


def _source_histories(root):
    episodes = [39, 7, 21]
    for episode, count in zip(episodes, (3100, 2800, 2400)):
        folder = root / "sources" / str(episode)
        folder.mkdir(parents=True)
        (folder / "case.json").write_text(json.dumps({"learner_side": episode % 2}))
        free_row = {"actions": [0, 1], "chosen": 0, "alternative": 1}
        forced_row = {"actions": [0], "chosen": 0, "alternative": None}
        rows = [forced_row, *([free_row] * count)]
        (folder / "history.json").write_text(json.dumps(rows))
    return episodes


def _anchor_digest(anchors):
    return _sha256(json.dumps(anchors, separators=(",", ":")).encode())


class ExperimentRefactoringInvariants(unittest.TestCase):
    def test_policy_hash_ignores_value_parameters_and_layout(self):
        state = _policy_state()
        self.assertFalse(state["encoder.weight"].is_contiguous())
        self.assertEqual(policy_hash(state), EXPECTED_POLICY_HASH)
        changed_value_state = {
            key: value.contiguous()
            for key, value in reversed(list(state.items()))
            if not key.startswith("value.")
        }
        changed_value_state["value.weight"] = torch.tensor([-11.0])
        self.assertEqual(
            policy_hash(changed_value_state),
            EXPECTED_POLICY_HASH,
        )

    def test_turn_selection_order(self):
        positions = _turn_positions()
        for groups in (positions, dict(reversed(list(positions.items())))):
            actual = select_positions(groups, maximum=9)
            self.assertEqual([row["key"] for row in actual], EXPECTED_TURN_POSITIONS)

    def test_turn_confirmation_statistics(self):
        actual = paired_statistics(
            ["w3", "w1", "w2", "w1", "w1", "w2", "w4"],
            [-1, 1, -0.25, 0.5, 0.125, 0.625, 0],
            samples=257,
            seed=3241,
        )
        self.assertEqual(actual, EXPECTED_PAIRED_STATISTICS)

    def test_rollout_comparison_anchor_selection(self):
        config = {"namespace": "test-refactoring-anchors", "seed": 17}
        modules = (rollout_fit, rollout_collection)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            episodes = _source_histories(root)
            for module in modules:
                name = module.__name__.rsplit(".", 1)[-1]
                for source_order, sources in (
                    ("original", episodes),
                    ("sorted", sorted(episodes)),
                ):
                    with self.subTest(module=name, source_order=source_order):
                        state = {
                            "sources": sources,
                            "forks": [{"episode": 21, "index": 0, "status": "pending"}],
                        }
                        anchors = module.build_dataset(config, root, state)["anchors"]
                        self.assertEqual(len(anchors), 8193)
                        self.assertIn((21, 0), anchors)
                        self.assertEqual(
                            _anchor_digest(anchors),
                            EXPECTED_ANCHORS[name][source_order],
                        )
                        persisted = json.loads((root / "anchors.json").read_text())
                        self.assertEqual(
                            persisted,
                            {
                                "positions": list(map(list, anchors)),
                                "selection_blind_to_continuation_outcomes": True,
                            },
                        )
