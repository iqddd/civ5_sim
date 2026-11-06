"""Checkpoint policy implementing the ordinary player-only Agent protocol."""

import random

import torch

from civ5rl.core.contracts import AgentContext
from civ5rl.core.model import Policy, batch_inputs
from civ5rl.core.observation import ObservationEncoder, ordered_actions
from civ5rl.provenance.common import choose


class RLAgent:
    def __init__(self, policy, device="cpu", mode="sample"):
        self.policy, self.device, self.mode = policy, device, mode
        self.policy.eval()

    @classmethod
    def from_checkpoint(cls, path, device="cpu", mode="sample"):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        policy = Policy(checkpoint["variant"]).to(device)
        policy.load_state_dict(checkpoint["model"])
        policy.spatial_cell_padding = checkpoint.get('config',{}).get('performance',{}).get('spatial_cell_padding',0)
        policy.enable_amp(
            checkpoint.get("config", {}).get("numerics", {}).get("amp", False)
            and str(device).startswith("cuda"),
            spatial_batch_size=checkpoint.get("config", {})
            .get("performance", {})
            .get("spatial_batch_size", 0),
        )
        return cls(policy, device, mode)

    def reset(self, context: AgentContext):
        self.encoder = ObservationEncoder(context, self.policy.variant)
        self.rng = random.Random(context["seed"])
        self.hidden = None
        self.events = []

    def _infer(self, observation, actions):
        encoded = self.encoder.encode(observation, actions)
        logits, _, self.hidden = self.policy(
            batch_inputs([encoded], self.device), self.hidden
        )
        return logits

    @torch.no_grad()
    def observe_step(self, observation, actions, forced_action):
        actions = ordered_actions(actions)
        if forced_action not in actions:
            raise ValueError("Forced action is not legal")
        self._infer(observation, actions)
        self.encoder.remember_action(forced_action, observation)

    @torch.no_grad()
    def act(self, observation, actions):
        actions = ordered_actions(actions)
        logits = self._infer(observation, actions)
        probabilities = logits.softmax(-1)[0].cpu().numpy()
        index = (
            int(logits[0].argmax())
            if self.mode == "argmax"
            else choose(probabilities, self.rng)
        )
        self.encoder.remember_action(actions[index], observation)
        self.events.append(
            {
                "kind": "rl_contacts",
                "turn": observation["turn"],
                "contacts": [
                    {
                        "id": uid,
                        "type_index": c[0],
                        "tile": c[1],
                        "last_hp": c[2],
                        "last_enemy_clock": c[3],
                    }
                    for uid, c in sorted(self.encoder.contacts.items())
                ],
            }
        )
        return actions[index]

    def drain_diagnostics(self):
        events, self.events = self.events, []
        return events
