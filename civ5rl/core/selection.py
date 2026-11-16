"""Torch-free player-only action selection state and diagnostics."""

import math
import random

import numpy as np

from civ5rl.core.observation import ObservationEncoder
from civ5rl.core.variants import PolicyVariant
from civ5rl.provenance.common import choose


class ActionSelection:
    """Public encoder, selection RNG and diagnostics, without a model or CUDA."""

    def reset_public_state(self, context, variant):
        self.encoder = ObservationEncoder(context, PolicyVariant(variant))
        self.rng = random.Random(context["seed"])
        self.events = []
        self.selection_metrics = []

    def action_from_logits(self, logits, observation, actions):
        """Finish a decision after single-agent or shared batched inference."""
        if self.mode == "argmax" and not self.collect_selection_metrics:
            return self.action_from_index(int(logits[0].argmax()), observation, actions)
        from civ5rl.core.agent import action_probabilities

        probabilities = (
            action_probabilities(
                logits, 1.0 if self.mode == "argmax" else self.temperature
            )[0]
            .cpu()
            .numpy()
        )
        index = (
            int(logits[0].argmax())
            if self.mode == "argmax"
            else choose(probabilities, self.rng)
        )
        if self.collect_selection_metrics and len(actions) > 1:
            p = probabilities[: len(actions)].astype(np.float64)
            positive = p[p > 0]
            self.selection_metrics.append(
                {
                    "p_max": float(p.max()),
                    "argmax_selected": index == int(logits[0].argmax()),
                    "normalized_entropy": float(
                        -(positive * np.log(positive)).sum() / math.log(len(actions))
                    ),
                    "legal_actions": len(actions),
                }
            )
        return self.action_from_index(index, observation, actions)

    def action_from_index(self, index, observation, actions):
        """Commit a selected legal action without evaluating a distribution."""
        if not 0 <= index < len(actions):
            raise ValueError("Policy selected an invalid action index")
        self.encoder.remember_action(actions[index], observation)
        self.events.append(
            {
                "kind": "rl_contacts",
                "turn": observation["turn"],
                "contacts": [
                    {
                        "id": uid,
                        "type_index": c.type_index,
                        "tile": c.tile,
                        "last_hp": c.last_hp,
                        "last_enemy_clock": c.last_clock,
                    }
                    for uid, c in sorted(self.encoder.contacts.items())
                ],
            }
        )
        return actions[index]

    def drain_diagnostics(self):
        events = self.events
        self.events = []
        return events
