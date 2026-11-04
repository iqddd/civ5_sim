"""Build the player-visible agent context from a scenario."""

from civ5rl.core.contracts import AgentContext, Scenario, ScenarioSpec
from civ5rl.core.observation import public_map


def context_for(
    scenario: Scenario,
    armies: list[list[str]],
    side: int,
    max_rounds: int,
    seed: int = 0,
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


def context_from_spec(
    spec: ScenarioSpec, *, side: int | None = None, seed: int = 0
) -> AgentContext:
    return context_for(
        spec["scenario"],
        spec["armies"],
        spec["learner_side"] if side is None else side,
        spec["max_rounds"],
        seed,
    )
