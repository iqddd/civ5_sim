"""Silent BC failure risks: label order, forced-token normalization and memory."""

import unittest
from copy import deepcopy
from dataclasses import asdict
from unittest.mock import patch

import numpy as np
import torch

from civ5rl.core.model import Policy
from civ5rl.core.observation import ObservationEncoder, ordered_actions
from civ5rl.engine.imitation import (
    DemonstrationCache,
    backward_minibatch,
    ce_sum,
    encode_demonstration,
    label_minibatches,
)
from civ5rl.protocols.imitation import (
    RecordingCP,
    make_case,
    prepare_manifest,
    search_config,
)
from civ5rl.provenance.common import configure, read_json
from civ5sim import Action, Battle
from civ5sim.cp_agent import CPAgent


class ImitationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure(17)
        cls.config = read_json("configs/rl-cp-imitation.json")
        cls.manifest = prepare_manifest(cls.config)

    def test_world_split_and_recorded_label_are_public_and_ordered(self):
        worlds = [
            set(self.manifest["worlds"][k])
            for k in ("training", "holdout", "validation")
        ]
        self.assertEqual([len(s) for s in worlds], [57, 7, 16])
        self.assertFalse(
            worlds[0] & worlds[1] or worlds[0] & worlds[2] or worlds[1] & worlds[2]
        )
        cases = [make_case(self.config, self.manifest, "training", i) for i in range(8)]
        self.assertEqual(sum(c["placement"] == "contact" for c in cases), 4)
        self.assertEqual([c["learner_side"] for c in cases], [0, 1] * 4)
        self.assertTrue(all(c["armies"] == [["warrior", "archer"]] * 2 for c in cases))
        self.assertEqual(cases[0], make_case(self.config, self.manifest, "training", 0))
        case = cases[0]
        with Battle(case["scenario"]) as battle:
            observation = battle.observe(0)
            actions = tuple(reversed(battle.available_actions()))
        teacher = RecordingCP(search_config(self.config))
        teacher.decisions = []
        selected = actions[-1]
        with patch.object(CPAgent, "act_with_planning", return_value=selected):
            self.assertEqual(
                teacher.act_with_planning(observation, actions, object()), selected
            )
        row = teacher.decisions[0]
        self.assertEqual(set(row), {"observation", "actions", "chosen", "supervised"})
        self.assertEqual(row["observation"], observation)
        self.assertEqual(row["actions"], [asdict(a) for a in ordered_actions(actions)])
        self.assertEqual(row["actions"][row["chosen"]], asdict(selected))

    def test_CE_gradient_accumulation_excludes_forced_labels_and_preserves_value(self):
        configure(17)
        case = make_case(self.config, self.manifest, "training", 0)
        context = {
            "map": case["scenario"]["map"],
            "side": 0,
            "armies": case["armies"],
            "max_rounds": 30,
            "random_damage": True,
            "fog": True,
        }
        with Battle(case["scenario"]) as battle:
            observation = battle.observe(0)
            legal = ordered_actions(battle.available_actions())
        free = {
            "observation": observation,
            "actions": [asdict(a) for a in legal],
            "chosen": legal.index(Action("end_turn")),
            "supervised": True,
        }
        forced = {
            **free,
            "actions": [asdict(Action("end_turn"))],
            "chosen": 0,
            "supervised": False,
        }
        # Synthetic sequences isolate the CE math, not simulator legality.
        episodes = [
            {"context": context, "decisions": [free, forced, free, forced, free]},
            {"context": context, "decisions": [forced, free]},
        ]
        cache = DemonstrationCache(episodes, "cpu")
        packed = list(label_minibatches(cache.chunks(), cache.masks, size=3))
        self.assertEqual([n for _, n in packed], [3, 1])
        self.assertEqual(
            sum(sum(cache.masks[i][a:b]) for spans, _ in packed for i, a, b in spans), 4
        )
        model = Policy("C")
        for parameter in model.value.parameters():
            parameter.requires_grad_(False)
        copied = deepcopy(model)
        spans = cache.chunks()
        batch, labels, lengths = cache.batch(spans)
        logits, _, _ = model.forward_sequences(batch, lengths)
        loss = ce_sum(logits, labels) / 4
        loss.backward()
        accumulated = backward_minibatch(copied, cache, spans, 4, micro_size=1)
        self.assertAlmostEqual(loss.item(), accumulated, places=6)
        for a, b in zip(model.parameters(), copied.parameters()):
            if a.requires_grad:
                torch.testing.assert_close(a.grad, b.grad, rtol=3e-5, atol=2e-7)
            else:
                self.assertIsNone(b.grad)
        value = {k: v.clone() for k, v in copied.value.state_dict().items()}
        torch.optim.Adam(
            [p for p in copied.parameters() if p.requires_grad], lr=1e-4
        ).step()
        self.assertTrue(
            all(torch.equal(v, copied.value.state_dict()[k]) for k, v in value.items())
        )
        # A new episode encoder starts fresh, even after another episode's history.
        together = cache.encoded[1]
        separate = encode_demonstration(episodes[1])
        for a, b in zip(together, separate):
            np.testing.assert_array_equal(a.previous, b.previous)
        fresh = ObservationEncoder(context, "C").encode(
            observation, (Action("end_turn"),)
        )
        np.testing.assert_array_equal(together[0].previous, fresh.previous)
        self.assertFalse(
            np.array_equal(cache.encoded[0][0].previous, cache.encoded[0][1].previous)
        )


if __name__ == "__main__":
    unittest.main()
