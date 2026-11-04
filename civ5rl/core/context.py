"""Build the player-visible agent context from a scenario."""

from civ5rl.core.contracts import AgentContext, Scenario
from civ5rl.core.observation import public_map


def context_for(
    scenario: Scenario, armies: list[list[str]], side: int, max_rounds: int, seed: int
) -> AgentContext:
    return {
        "map": public_map(scenario["map"]),
        "side": side,
        "armies": armies,
        "max_rounds": max_rounds,
        "random_damage": scenario["random_damage"],
        "fog": scenario["fog"],
        "seed": seed,
    }
