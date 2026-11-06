"""Checkpoint policy implementing the ordinary player-only Agent protocol."""

import random

import torch

from civ5rl.core.contracts import AgentContext
from civ5rl.core.model import Policy, batch_inputs
from civ5rl.core.observation import ObservationEncoder, ordered_actions
from civ5rl.core.variants import PolicyVariant
from civ5rl.provenance.common import choose


class RLAgent:
    def __init__(self, policy, device="cpu", mode="sample"):
        self.policy = policy
        self.device = device
        self.mode = mode
        self.policy.eval()

    @classmethod
    def from_checkpoint(cls, path, device="cpu", mode="sample"):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        policy = Policy(PolicyVariant(checkpoint["variant"])).to(device)
        policy.load_state_dict(checkpoint["model"])
        config = checkpoint.get("config", {})
        performance = config.get("performance", {})
        numerics = config.get("numerics", {})
        policy.spatial_cell_padding = performance.get("spatial_cell_padding", 0)
        policy.enable_amp(
            numerics.get("amp", False) and str(device).startswith("cuda"),
            spatial_batch_size=performance.get("spatial_batch_size", 0),
        )
        return cls(policy, device, mode)

    def reset(self, context: AgentContext):
        self.encoder = ObservationEncoder(context, PolicyVariant(self.policy.variant))
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
