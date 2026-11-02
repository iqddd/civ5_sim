"""Player-only Python agent interface and small smoke-test policies."""

import importlib
import random
from typing import Protocol

from .engine import Action


class Agent(Protocol):
    """One instance per side; reset starts a match, act selects one legal command.

    Inputs are owned copies. Policies may retain observation history between calls.
    Python agents are trusted code, not isolated or subject to hard timeouts.
    """

    def reset(self, context: dict) -> None: ...

    def act(self, observation: dict, actions: tuple[Action, ...]) -> Action: ...


class RandomAgent:
    def reset(self, context):
        self.rng = random.Random(context["seed"])

    def act(self, observation, actions):
        return self.rng.choice(actions)


class EndTurnAgent:
    def reset(self, context):
        pass

    def act(self, observation, actions):
        return Action("end_turn")


def load_agent(spec):
    """Create a fresh builtin or a zero-argument module:factory agent."""
    if spec == "random":
        return RandomAgent()
    if spec == "end_turn":
        return EndTurnAgent()
    if spec == "cp":
        from .cp_agent import CPAgent

        return CPAgent()
    if spec == "greedy":
        from .greedy_agent import GreedyAgent

        return GreedyAgent()
    module, separator, name = spec.partition(":")
    if not separator or not module or not name:
        raise ValueError("Agent must be random, end_turn, greedy, cp or module:factory")
    return getattr(importlib.import_module(module), name)()
