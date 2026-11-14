"""Reproducible deployment, player-only agent calls and finite match lifecycle."""

import json
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

from civ5map.maps import components, load, passable, validate

from .agents import load_agent
from .engine import UNIT_TYPES, Battle, action_value, boolean, digest, integer
from .telemetry import measure


@dataclass(frozen=True)
class MatchConfig:
    map: dict | str | Path
    armies: list[list[str]]
    max_rounds: int
    seed: int = 0
    fog: bool = True
    random_damage: bool = True
    max_calls_per_turn: int = 10_000
    turn_reset_mode: str = "legacy"


def load_match_config(path):
    """Read a CLI config; resolve its map relative to the config file."""
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Match config must be an object")
    value = value.copy()
    agents = value.pop("agents")
    if (
        not isinstance(agents, list)
        or len(agents) != 2
        or any(not isinstance(a, str) or not a for a in agents)
    ):
        raise ValueError("agents must contain two agent specifications")
    if isinstance(value.get("map"), str):
        value["map"] = path.parent / value["map"]
    return MatchConfig(**value), agents


def _seeds(seed):
    integer(seed, "seed", 0, 2**64 - 1)
    return {
        name: int(digest({"seed": seed, "stream": name})[:16], 16)
        for name in ("placement", "battle", "agent0", "agent1")
    }


def create_scenario(config):
    """Validate settings and deploy both rosters on a shared land component."""
    integer(config.max_rounds, "max_rounds", 1, (2**31 - 1) // 2)
    integer(config.max_calls_per_turn, "max_calls_per_turn", 1)
    boolean(config.fog, "fog")
    boolean(config.random_damage, "random_damage")
    seeds = _seeds(config.seed)
    if not isinstance(config.armies, (list, tuple)) or len(config.armies) != 2:
        raise ValueError("armies must contain two nonempty lists of unit types")
    for army in config.armies:
        if not isinstance(army, (list, tuple)) or not army:
            raise ValueError("Each army must be a nonempty list of unit types")
        if any(kind not in UNIT_TYPES for kind in army):
            raise ValueError(f"Army types must be from {UNIT_TYPES}")
    if sum(map(len, config.armies)) > 512:
        raise ValueError("Both armies together must have at most 512 units")
    m = deepcopy(config.map) if isinstance(config.map, dict) else load(config.map)
    validate(m)
    w, h = m["width"], m["height"]
    if w > 128 or h > 128 or w * h > 4096:
        raise ValueError("Combat map supports at most 4096 cells and 128 per axis")
    zones = m.get("spawn_candidates")
    if not isinstance(zones, list) or len(zones) != 2:
        raise ValueError("Map must have two spawn_candidates zones")
    cells = []
    for side, zone in enumerate(zones):
        if not isinstance(zone, list):
            raise TypeError("Spawn zone must be a list of [x, y] coordinates")
        indices = []
        for xy in zone:
            if not isinstance(xy, (list, tuple)) or len(xy) != 2:
                raise ValueError("Spawn candidate must be [x, y]")
            x = integer(xy[0], "spawn x", 0, w - 1)
            y = integer(xy[1], "spawn y", 0, h - 1)
            index = y * w + x
            if not passable(m["tiles"][index]):
                raise ValueError("Spawn candidates must be passable")
            indices.append(index)
        if len(set(indices)) != len(indices):
            raise ValueError("Spawn candidates must be unique")
        if len(indices) < len(config.armies[side]):
            raise ValueError(f"Not enough spawn candidates for side {side}")
        cells.append(set(indices))
    if cells[0] & cells[1]:
        raise ValueError("Spawn zones must not overlap")
    if not any(cells[0] | cells[1] <= group for group in components(m)):
        raise ValueError("Both spawn zones must belong to one land component")
    rng = random.Random(seeds["placement"])
    units = []
    for side, army in enumerate(config.armies):
        positions = rng.sample(sorted(cells[side]), len(army))
        units.extend(
            {"type": kind, "side": side, "tile": tile}
            for kind, tile in zip(army, positions)
        )
    return {
        "schema": 1,
        "kind": "civ5sim.scenario",
        "map": m,
        "units": units,
        "seed": seeds["battle"],
        "fog": config.fog,
        "random_damage": config.random_damage,
        **({"turn_reset_mode": config.turn_reset_mode} if config.turn_reset_mode != "legacy" else {}),
        "active_side": 0,
    }


@dataclass
class MatchOutcome:
    result: dict
    replay: dict
    final: dict
    diagnostic_recorder: object = None

    def save(self, directory):
        """Write replay and snapshot first, result last as the completion record."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("replay", "final", "result"):
            path = directory / f"{name}.json"
            path.write_text(
                json.dumps(getattr(self, name), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        if self.diagnostic_recorder is not None:
            self.diagnostic_recorder.save(directory, self.replay)


def run_match(config, agents, *, metrics=None, diagnostics="off", scenario=None, agent_seeds=None):
    """Run two agent objects/specifications and return privileged match artifacts.

    Invalid configuration raises before invoking agents. Agent failures instead
    return status=error with a replay of the successfully executed prefix.
    metrics, when supplied, receives wall-clock sample dictionaries separately
    from the deterministic result/replay. Collectors must not raise exceptions.
    """
    from .decision_diagnostics import recorder
    recording = recorder(diagnostics)
    if config.turn_reset_mode not in ("legacy", "end_turn"):
        raise ValueError("turn_reset_mode must be legacy or end_turn")
    if scenario is None:
        scenario = create_scenario(config)
    else:
        scenario = deepcopy(scenario)
        if scenario.get("kind") != "civ5sim.scenario" or scenario.get("active_side", 0) != 0:
            raise ValueError("Prepared match must start at side 0 in a scenario")
        if scenario.get("turn_reset_mode", "legacy") != config.turn_reset_mode:
            raise ValueError("Prepared scenario turn_reset_mode disagrees with MatchConfig")
        if scenario.get("fog") != config.fog or scenario.get("random_damage") != config.random_damage:
            raise ValueError("Prepared scenario flags disagree with MatchConfig")
        if [[u["type"] for u in scenario["units"] if u["side"] == s] for s in (0, 1)] != config.armies:
            raise ValueError("Prepared scenario rosters disagree with MatchConfig")
        integer(config.max_rounds, "max_rounds", 1, (2**31 - 1) // 2)
        integer(config.max_calls_per_turn, "max_calls_per_turn", 1)
    config = deepcopy(config)
    agents = list(agents)
    if len(agents) != 2:
        raise ValueError("Expected exactly two agents")
    if agents[0] is agents[1] and not isinstance(agents[0], str):
        raise ValueError("Each side needs an independent agent instance")
    names = [
        a if isinstance(a, str) else f"{type(a).__module__}:{type(a).__qualname__}"
        for a in agents
    ]
    seeds = _seeds(config.seed)
    if scenario["seed"] != seeds["battle"]:
        seeds.update(battle=scenario["seed"], placement=None)
    if agent_seeds is not None:
        if len(agent_seeds) != 2:
            raise ValueError("Expected two agent seeds")
        for side, value in enumerate(agent_seeds):
            seeds[f"agent{side}"] = integer(value, "agent seed", 0, 2**64 - 1)
    m = scenario["map"]
    public_map = {
        k: m[k] for k in ("schema", "width", "height", "row_parity", "wrap_x")
    }
    public_map["tiles"] = [
        {k: deepcopy(tile[k]) for k in ("terrain", "elevation", "feature", "rivers")}
        for tile in m["tiles"]
    ]
    if "boundary_rivers" in m:
        public_map["boundary_rivers"] = [
            {k: edge[k] for k in ("x", "y", "direction", "flow")}
            for edge in m["boundary_rivers"]
        ]
    status, outcome, winner, reason, error = "completed", None, None, None, None
    completed_turns = commands = calls_this_turn = 0
    agent_calls = [0, 0]
    ai_events = []

    def collect_events(side):
        drain = getattr(agents[side], "drain_diagnostics", None)
        if drain is not None:
            for event in drain():
                ai_events.append({**deepcopy(event), "side": side, "command": commands})

    with Battle(scenario, log=True) as battle:
        for side in range(2):
            try:
                if isinstance(agents[side], str):
                    agents[side] = load_agent(agents[side])
                if side == 1 and agents[0] is agents[1]:
                    raise ValueError(
                        "Agent factories must return independent instances"
                    )
                agents[side].reset(
                    {
                        "map": deepcopy(public_map),
                        "side": side,
                        "armies": deepcopy(config.armies),
                        "max_rounds": config.max_rounds,
                        "fog": config.fog,
                        "random_damage": config.random_damage,
                        "seed": seeds[f"agent{side}"],
                    }
                )
                if callable(getattr(agents[side], "act_with_planning", None)):
                    from .combat_ai import Planner

                    if not isinstance(getattr(agents[side], "planner", None), Planner):
                        raise TypeError("PlanningAgent must own a Planner after reset")
                    if not agents[side].planner._handle:
                        raise ValueError("PlanningAgent Planner is closed")
            except Exception as exc:  # noqa: BLE001 -- agent boundary saves failed matches
                if recording is not None:
                    recording.failure("reset", exc)
                status, reason = "error", "agent_error"
                error = {
                    "side": side,
                    "phase": "reset",
                    "message": f"{type(exc).__name__}: {exc}",
                }
                break
        while reason is None:
            destroyed = battle.army_status()["destroyed"]
            if any(destroyed):
                if all(destroyed):
                    outcome, reason = "draw", "mutual_destruction"
                else:
                    outcome, reason = "win", "army_destroyed"
                    winner = 1 if destroyed[0] else 0
                break
            if completed_turns >= 2 * config.max_rounds:
                outcome, reason = "draw", "turn_limit"
                break
            side = completed_turns % 2
            if calls_this_turn >= config.max_calls_per_turn:
                status, reason = "error", "action_limit"
                error = {
                    "side": side,
                    "phase": "act",
                    "message": "max_calls_per_turn exceeded",
                }
                break
            with measure(metrics, "prepare_actions", side=side, turn=completed_turns):
                previews = None
                if getattr(agents[side], "requires_attack_preview", False):
                    actions, previews = battle.actions_with_previews()
                else:
                    actions = tuple(battle.available_actions())
            calls_this_turn += 1
            agent_calls[side] += 1
            try:
                with measure(metrics, "prepare_observation", side=side, turn=completed_turns):
                    observation = battle.observe(side)
                    if previews is not None:
                        observation["attack_previews"] = previews.tolist()
                if recording is not None:
                    recording.begin(commands + 1, side, observation, agents[side])
                if callable(getattr(agents[side], "act_with_planning", None)):
                    from .combat_ai import CallPlanningServices, PlanningState

                    with measure(metrics, "prepare_planning", side=side, turn=completed_turns):
                        planning = CallPlanningServices(
                            agents[side].planner,
                            PlanningState.from_battle(battle, side, observation=observation),
                            side, observation["turn"], metrics=metrics,
                        )
                    try:
                        with measure(metrics, "act", side=side, turn=completed_turns):
                            selected = agents[side].act_with_planning(observation, actions, planning)
                    finally:
                        planning.expire()
                else:
                    with measure(metrics, "act", side=side, turn=completed_turns):
                        selected = agents[side].act(observation, actions)
                if recording is not None:
                    recording.end(selected)
                collect_events(side)
            except Exception as exc:  # noqa: BLE001 -- agent boundary saves failed matches
                if recording is not None:
                    recording.failure("act", exc)
                status, reason = "error", "agent_error"
                error = {
                    "side": side,
                    "phase": "act",
                    "message": f"{type(exc).__name__}: {exc}",
                }
                break
            try:
                action = action_value(selected)
                action.native()  # Strict integer/type validation before membership.
                if action not in actions:
                    raise ValueError(
                        "Agent returned an action outside available_actions"
                    )
            except (TypeError, ValueError) as exc:
                if recording is not None:
                    recording.failure("action_validation", exc)
                status, reason = "error", "invalid_action"
                error = {
                    "side": side,
                    "phase": "act",
                    "message": f"{type(exc).__name__}: {exc}",
                }
                break
            step = battle.step(action)
            if not step.ok:
                raise RuntimeError(f"Available action rejected by kernel: {step.error}")
            commands += 1
            if action.kind == "end_turn":
                completed_turns += 1
                calls_this_turn = 0
            try:
                callback = getattr(agents[side], "on_action_result", None)
                if callable(getattr(agents[side], "act_with_planning", None)) and callback:
                    with measure(metrics, "callback", side=side, turn=observation["turn"]):
                        callback(action, step, battle.observe(side))
                collect_events(side)
            except Exception as exc:  # noqa: BLE001 -- preserve the executed prefix
                if recording is not None:
                    recording.failure("on_action_result", exc)
                status, reason = "error", "agent_error"
                error = {"side": side, "phase": "on_action_result",
                         "message": f"{type(exc).__name__}: {exc}"}
        final = battle.snapshot()
        replay = battle.replay()
        if ai_events:
            replay["ai_events"] = ai_events
        settings = asdict(config)
        settings["map"] = deepcopy(scenario["map"])
        result = {
            "schema": 1,
            "kind": "civ5sim.match_result",
            "status": status,
            "outcome": outcome,
            "winner": winner,
            "reason": reason,
            "error": error,
            "config": settings,
            "agents": names,
            "seeds": seeds,
            "commands": commands,
            "completed_turns": completed_turns,
            "completed_rounds": completed_turns // 2,
            "agent_calls": agent_calls,
            "alive": battle.army_status()["alive"],
            "remaining_hp": [
                sum(u["hp"] for u in final["units"] if u["side"] == s) for s in range(2)
            ],
            "initial_state_sha256": digest(replay["initial"]),
            "final_state_sha256": digest(final),
        }
    return MatchOutcome(result, replay, final, recording)
