"""Freeze experiment protocol behavior that refactoring can silently change."""

import ast
import hashlib
import inspect
import json
import unittest
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from civ5rl.experiments import (
    evaluation,
    imitation_target_scale,
    ppo_after_imitation,
    ppo_target_scale,
    rollout_comparison,
    rollout_comparison_collection,
    target_bc_sampling,
    turn_comparison,
    turn_confirmation,
)

BASELINE = json.loads(
    Path(__file__).with_name("refactoring_baseline.json").read_text(encoding="utf-8")
)
CONFIG = {"teacher_search": {"width": 8, "depth": 2}, "policy_temperature": 0.5}
PLAN = {"development": [{"case_hash": "case-7"}, {"case_hash": "case-2"}]}
RUNTIME = {"device": "cpu", "dtype": "float32", "workers": 4}


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


def _native_hash(path):
    name = Path(path).name
    if name in ("combat_ai.dll", "libcombat_ai.so"):
        return "test-planner-sha256"
    if name in ("combat.dll", "libcombat.so"):
        return "test-kernel-sha256"
    raise AssertionError(f"Unexpected evaluator dependency: {path}")


class ExperimentRefactoringInvariants(unittest.TestCase):
    def test_historical_import_paths_and_names(self):
        for path, api in BASELINE["api"].items():
            with self.subTest(module=path):
                module = import_module(path)
                names = [*api["names"], *api.get("all", [])]
                missing = [name for name in names if not hasattr(module, name)]
                self.assertEqual(missing, [])
                if "all" in api:
                    self.assertEqual(list(module.__all__), api["all"])

    def test_target_scale_private_aliases(self):
        for path, aliases in BASELINE["aliases"].items():
            module = import_module(path)
            for private, public in aliases.items():
                with self.subTest(module=path, alias=private):
                    self.assertIs(getattr(module, private), getattr(module, public))

    def test_evaluator_source_fingerprints(self):
        current = Path(inspect.getfile(ppo_after_imitation)).read_text(encoding="utf-8")
        # ast.dump chooses its own representation of empty fields by Python version.
        # Construct the expected dump from the frozen evaluator on this interpreter.
        original = ast.parse(BASELINE["evaluation_task_source"]).body[0]
        expected = ast.dump(ast.Module(body=original.body, type_ignores=[]))
        self.assertEqual(ppo_after_imitation.match_execution(current), expected)
        self.assertIs(imitation_target_scale.play_case, evaluation.play_case)
        self.assertEqual(
            _sha256(inspect.getsource(evaluation.play_case).encode()),
            BASELINE["play_case_source_sha256"],
        )

    def test_policy_hash_ignores_value_parameters_and_layout(self):
        state = _policy_state()
        self.assertFalse(state["encoder.weight"].is_contiguous())
        self.assertEqual(
            ppo_after_imitation.policy_hash(state), BASELINE["policy_hash"]
        )
        changed_value_state = {
            key: value.contiguous()
            for key, value in reversed(list(state.items()))
            if not key.startswith("value.")
        }
        changed_value_state["value.weight"] = torch.tensor([-11.0])
        self.assertEqual(
            ppo_after_imitation.policy_hash(changed_value_state),
            BASELINE["policy_hash"],
        )

    def test_imitation_target_scale_evaluation_key(self):
        with patch.object(
            imitation_target_scale, "_evaluator_digest", return_value="test-evaluator"
        ):
            for name, runtime in (("legacy", None), ("with_runtime", RUNTIME)):
                with self.subTest(runtime=name):
                    actual = imitation_target_scale._evaluation_key(
                        CONFIG, PLAN, "development", BASELINE["policy_hash"], runtime
                    )
                    self.assertEqual(
                        actual, BASELINE["imitation_evaluation_keys"][name]
                    )

    def test_ppo_target_scale_evaluation_key(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save({"model": _policy_state()}, root / "policy.pt")
            with (
                patch.object(
                    ppo_target_scale.platform, "system", return_value="test-platform"
                ),
                patch.object(ppo_target_scale.torch, "__version__", "test-torch"),
                patch.object(ppo_target_scale, "file_sha256", side_effect=_native_hash),
            ):
                for path in ("cp", "policy.pt"):
                    with self.subTest(policy=path):
                        actual = ppo_target_scale.evaluation_key(
                            CONFIG, PLAN, root, path, "development", RUNTIME
                        )
                        self.assertEqual(
                            list(actual), BASELINE["ppo_evaluation_keys"][path]
                        )

    def test_target_bc_sampling_mode_key(self):
        identity = {"checkpoint": "policy-v1", "cases": ["case-2", "case-7"]}
        for mode, temperature, expected in BASELINE["sampling_mode_keys"]:
            with self.subTest(mode=mode, temperature=temperature):
                actual = target_bc_sampling.mode_key(
                    identity, {"mode": mode, "temperature": temperature}
                )
                self.assertEqual(actual, expected)

    def test_turn_selection_order(self):
        positions = _turn_positions()
        for groups in (positions, dict(reversed(list(positions.items())))):
            actual = turn_comparison.select_positions(groups, maximum=9)
            self.assertEqual([row["key"] for row in actual], BASELINE["turn_positions"])

    def test_turn_confirmation_statistics(self):
        actual = turn_confirmation.paired_statistics(
            ["w3", "w1", "w2", "w1", "w1", "w2", "w4"],
            [-1, 1, -0.25, 0.5, 0.125, 0.625, 0],
            samples=257,
            seed=3241,
        )
        self.assertEqual(actual, BASELINE["paired_statistics"])

    def test_rollout_comparison_anchor_selection(self):
        config = {"namespace": "test-refactoring-anchors", "seed": 17}
        modules = (rollout_comparison, rollout_comparison_collection)
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
                            BASELINE["anchor_sha256"][name][source_order],
                        )
                        persisted = json.loads((root / "anchors.json").read_text())
                        self.assertEqual(
                            persisted,
                            {
                                "positions": list(map(list, anchors)),
                                "selection_blind_to_continuation_outcomes": True,
                            },
                        )
