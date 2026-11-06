"""D0-M corpus identity, public contact metrics and terminal learning signals."""

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from civ5rl.config import load_config
from civ5rl.common import configure, read_json
from civ5rl.model import Policy
from civ5rl.validation import validate_config

from civ5rl.protocols.diagnostic_d1 import load_weights
from civ5rl.protocols.multishot import (
    STAGE,
    control_episode,
    factory,
    prepare,
    specification,
)
from civ5rl.protocols.multishot_metrics import (
    comparison,
    gate,
    sequence_metrics,
    summarize,
    transition,
)
from civ5rl.provenance import bound_cases, validate_cases
from civ5sim import Action


class MultishotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config("configs/rl-diagnostic-multishot.json")
        cls.dataset = read_json(
            "artifacts/rl/diagnostic-multishot/manifests/manifest.json"
        )

    def setUp(self):
        configure(17)

    def short_config(self):
        config = deepcopy(self.config)
        config["collection"].update(
            num_envs=2,
            cpu_workers=1,
            envs_per_worker=2,
            min_learner_decisions_per_update=16,
        )
        config["ppo"].update(epochs=1)
        return config

    def test_frozen_inputs_corpus_controls_and_native_certificates(self):
        validate_config(self.config)
        specification(self.config, full_config=True)
        # Exhaustive control games run during corpus preparation, not every test.
        directory = Path("artifacts/rl/diagnostic-multishot/manifests")
        dataset = prepare(directory, self.config)
        proof = read_json(directory / "construction-verification.json")
        self.assertEqual(dataset, self.dataset)
        self.assertEqual(proof["verified_cases"], 384)
        self.assertEqual(proof["control_games"], 6528)
        self.assertEqual(proof["passive_games"], 384)
        original = read_json("artifacts/rl/diagnostic-results/manifests/manifest.json")
        for split in ("train", "dev"):
            manifest = dataset[STAGE][split]
            validate_cases(bound_cases(manifest), manifest)
            for before, after in zip(original["D0"][split]["cases"], manifest["cases"]):
                self.assertEqual(before["agent_seeds"], after["agent_seeds"])
                self.assertEqual(before["scenario"]["seed"], after["scenario"]["seed"])
                self.assertEqual(before["layout_id"], after["layout_id"])
                self.assertNotEqual(before["physical_hash"], after["physical_hash"])
                self.assertEqual(after["max_rounds"], 4)
                self.assertNotIn("learner_decision_limit", after)
        for row in proof["cases"]:
            self.assertEqual(row["reference_wins"], 17)
            self.assertTrue(
                all(
                    p["max"] < p["target_hp"] == 40 and p["rolls"] == 1200
                    for p in row["initial_shots"]
                )
            )
        from civ5rl.protocols.diagnostic import DiagnosticFactory

        old = DiagnosticFactory(original, 17)
        new = factory(dataset)
        for i in range(10):
            a, b = old.make(i, "D0"), new.make(i, STAGE)
            self.assertEqual(a["layout_id"], b["layout_id"])
            self.assertEqual(a["agent_seeds"], b["agent_seeds"])
            self.assertEqual(a["scenario"]["seed"], b["scenario"]["seed"])

    def test_honest_multi_step_controls_terminal_labels_and_recurrent_memory(self):
        from civ5rl.collector import Collector
        from civ5rl.ppo_fast import RolloutCache

        spec = bound_cases(self.dataset[STAGE]["dev"])[0]
        ref = control_episode(spec)
        self.assertGreaterEqual(
            sum(m["ranged_chosen"] for m in ref["decision_metrics"]), 2
        )
        self.assertGreater(len(ref["chosen"]), 2)
        summary, rows = summarize([ref])
        self.assertEqual(summary["contact_immediate_use"]["value"], 1)
        self.assertEqual(rows[0]["reward"], 1)
        self.assertEqual(control_episode(spec, passive=True)["result"]["reward"], 0)
        policy = Policy()
        load_weights(policy, self.config)
        config = self.short_config()
        with (
            tempfile.TemporaryDirectory() as tmp,
            Collector(config, factory(self.dataset), "cpu", Path(tmp)) as collector,
        ):
            episodes, _, _ = collector.collect(policy, STAGE, 0)
        self.assertTrue(any(len(ep["chosen"]) > 1 for ep in episodes))
        for ep in episodes:
            self.assertIsNone(ep["old_hidden"][0])
            if len(ep["chosen"]) > 1:
                self.assertIsNotNone(ep["old_hidden"][1])
            self.assertNotEqual(ep["result"]["reason"], "diagnostic_decision_limit")
        cache = RolloutCache(episodes, "C", "cpu")
        np.testing.assert_array_equal(
            cache.labels["returns"].numpy(),
            [ep["result"]["reward"] for ep in episodes for _ in ep["chosen"]],
        )

    def test_contact_metrics_public_histories_and_null_denominators(self):
        shot, end, move = (
            Action("ranged", 0, 2),
            Action("end_turn"),
            Action("move", 0, 3),
        )

        def observation(turn, own=1, enemy=2, visible=True, hp=40):
            return {
                "turn": turn,
                "units": [{"id": 0, "side": 0, "hp": 100, "tile": own}]
                + ([{"id": 1, "side": 1, "hp": hp, "tile": enemy}] if visible else []),
            }

        def episode(observations, choices, actions=None):
            acts = actions or [(shot, end, move)] * len(choices)
            return {
                "observations": observations,
                "actions": acts,
                "chosen": choices,
                "old_log_probs": [np.log(np.array([0.6, 0.3, 0.1])) for _ in choices],
            }

        good = episode([observation(0), observation(0), observation(2)], [0, 1, 0])
        result = sequence_metrics(good)
        self.assertTrue(result["contact_opportunity"])
        self.assertTrue(result["contact_used"])
        self.assertAlmostEqual(result["contact_probability"], 0.6)
        skip = deepcopy(good)
        skip["chosen"][-1] = 1
        self.assertTrue(sequence_metrics(skip)["contact_opportunity"])
        self.assertFalse(sequence_metrics(skip)["contact_used"])
        for observations, reason in (
            ([observation(0), observation(0, own=3), observation(2)], "movement"),
            (
                [observation(0), observation(0, visible=False), observation(2)],
                "visibility_lost",
            ),
            ([observation(0), observation(2, hp=0)], "unit_destroyed"),
        ):
            ep = episode(observations, [0] + [1] * (len(observations) - 1))
            metric = sequence_metrics(ep)
            self.assertFalse(metric["contact_opportunity"])
            self.assertIn(reason, metric["contact_absence_reasons"])
            self.assertIsNone(metric["contact_probability"])
        self.assertIn(
            "episode_ended",
            sequence_metrics(episode([observation(0)], [0]))["contact_absence_reasons"],
        )
        spec = bound_cases(self.dataset[STAGE]["dev"])[0]
        summary, _ = summarize([control_episode(spec, passive=True)])
        self.assertEqual(summary["contact_immediate_use"]["denominator"], 0)
        self.assertIsNone(summary["contact_immediate_use"]["value"])
        self.assertIsNone(summary["contact_probability"]["value"])

    def test_gates_limits_and_descriptive_pairing(self):
        self.assertFalse(gate({"games": 128, "wins": 121}))
        self.assertTrue(gate({"games": 128, "wins": 122}))
        state = {
            "status": "running",
            "update": 7,
            "training_seconds": 0,
            "consecutive": 0,
        }
        one = transition(state, True)
        self.assertEqual(transition(one, False)["consecutive"], 0)
        self.assertEqual(
            transition({**one, "update": 8, "training_seconds": 1801}, True)["status"],
            "passed",
        )
        self.assertEqual(
            transition({**state, "update": 8}, True)["status"], "failed_update_limit"
        )
        self.assertEqual(
            transition({**state, "training_seconds": 1800}, False)["status"],
            "failed_training_time_limit",
        )
        base = [{"layout_id": i // 2, "side": i % 2, "win": i < 80} for i in range(128)]
        rows = [{**r, "win": i < 123} for i, r in enumerate(base)]
        self.assertTrue(comparison(rows, base)["descriptive_only"])
        with self.assertRaises(ValueError):
            comparison(rows[:-1], base)

    def test_pretraining_stops_have_no_Adam_steps(self):
        from tools.archive import run_rl_multishot as runner

        config = self.short_config()

        def fake_evaluate(policy, collector, dataset, directory, monitor, control=None):
            return {
                "games": 128,
                "wins": 122 if Path(directory).name == "transfer" else 0,
                "seconds": 0,
            }, []

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(runner, "evaluate", side_effect=fake_evaluate),
            patch(
                "torch.optim.Adam.step", side_effect=AssertionError("unexpected Adam")
            ),
        ):
            result = runner.run_seed(config, self.dataset, tmp, "cpu")
            self.assertEqual(result["status"], "solved_at_transfer")
            saved = torch.load(Path(tmp) / "checkpoints/initial.pt", weights_only=False)
            self.assertEqual(saved["optimizer"]["state"], {})
            self.assertEqual(result["decisions"], 0)

        def zero_evaluate(*args, **kwargs):
            return {"games": 128, "wins": 0, "seconds": 0}, []

        cases = bound_cases(self.dataset[STAGE]["train"])[:2]
        passive = [control_episode(s, passive=True) for s in cases]
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(runner, "evaluate", side_effect=zero_evaluate),
            patch.object(
                runner.Collector,
                "collect",
                return_value=(passive, 2, {"collection_seconds": 0}),
            ),
            patch(
                "civ5rl.ppo_fast.BatchedBackend",
                **{"return_value.replay": {"full_kl": 0.0}},
            ),
            patch(
                "torch.optim.Adam.step", side_effect=AssertionError("unexpected Adam")
            ),
        ):
            result = runner.run_seed(config, self.dataset, tmp, "cpu")
            self.assertEqual(result["status"], "no_positive_reward_in_initial_rollout")
            self.assertEqual(result["update"], 0)
            self.assertGreater(result["decisions"], 0)
            latest = read_json(Path(tmp) / "latest.json")
            saved = torch.load(
                Path(tmp) / "checkpoints" / latest["checkpoint"], weights_only=False
            )
            self.assertEqual(saved["optimizer"]["state"], {})
            self.assertTrue((Path(tmp) / latest["pending_rollout"]["path"]).exists())


if __name__ == "__main__":
    unittest.main()
