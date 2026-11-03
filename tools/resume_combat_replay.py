"""Continue one verified replay at a turn boundary with a fresh CP coordinator.

Physical state/RNG and the original command prefix are retained. CP observation
memory and greedy's own history are warmed from that prefix, without executing
new CP plans before the boundary. This never launches an evaluation batch.
"""

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from civ5sim import Action, Battle
from civ5sim.combat_ai import (
    Aggression,
    CallPlanningServices,
    PlanningState,
    SearchConfig,
)
from civ5sim.cp_agent import CPAgent
from civ5sim.engine import digest
from civ5sim.greedy_agent import GreedyAgent
from civ5sim.match import MatchOutcome


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resume(source, boundary, *, search_config=None, diagnostics="off"):
    from civ5sim.decision_diagnostics import recorder
    recording = recorder(diagnostics)
    source = Path(source)
    replay = read(source)
    previous = read(source.with_name("result.json"))
    if not 1 <= boundary <= len(replay["steps"]):
        raise ValueError("Boundary must follow an existing command")
    if replay["steps"][boundary-1]["action"]["kind"] != "end_turn":
        raise ValueError("Resume only after end_turn")
    aliases = {"civ5sim.cp_agent:CPAgent": "cp", "civ5sim.greedy_agent:GreedyAgent": "greedy"}
    specs = [aliases.get(name, name) for name in previous["agents"]]
    if len(specs) != 2 or set(specs) != {"cp", "greedy"}:
        raise ValueError("Expected a CP versus greedy source")
    cp_side = specs.index("cp")
    agents = [CPAgent(search_config) if name == "cp" else GreedyAgent() for name in specs]
    commands = completed_turns = 0
    calls = [0, 0]
    events = [deepcopy(e) for e in replay.get("ai_events", [])
              if e["command"] < boundary or (e["command"] == boundary and e["event"] == "action_result")]
    config = deepcopy(previous["config"])
    try:
        with Battle(replay["initial"], log=True) as battle:
            for side, agent in enumerate(agents):
                agent.reset({"map": deepcopy(config["map"]), "side": side,
                             "armies": config["armies"], "max_rounds": config["max_rounds"],
                             "seed": previous["seeds"][f"agent{side}"],
                             "fog": config["fog"], "random_damage": config["random_damage"]})
            for entry in replay["steps"][:boundary]:
                side = battle.snapshot()["state"]["side"]
                agent = agents[side]
                observation = battle.observe(side)
                if side == cp_side:
                    agent._observe(observation)
                else:
                    selected = agent.act(observation, tuple(battle.available_actions()))
                    if asdict(selected) != entry["action"]:
                        raise ValueError(f"Greedy history differs before command {commands+1}")
                result = battle.step(entry["action"])
                commands += 1
                calls[side] += 1
                if asdict(result) != entry["result"] or digest(battle.snapshot()) != entry["state_sha256"]:
                    raise ValueError(f"Source replay differs at command {commands}")
                if side == cp_side:
                    agent._observe(battle.observe(side))
                completed_turns += entry["action"]["kind"] == "end_turn"
            if battle.snapshot()["state"]["side"] != cp_side:
                raise ValueError("Boundary must start CP's turn, after greedy")
            checkpoint = battle.snapshot()
            agents[cp_side]._observe(battle.observe(cp_side))
            description = (f"Продолжение после команды {boundary}: прежние команды сохранены; "
                           "с этой границы работает новый координатор CP. Память наблюдений "
                           "восстановлена, прежние очереди и ограничения попыток сброшены.")
            continuation = {"kind": "continuation", "description": description,
                            "resume_after": boundary, "source": str(source.resolve()),
                            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                            "checkpoint_sha256": digest(checkpoint),
                            "port": agents[cp_side].planner.version,
                            "search_config": asdict(agents[cp_side].config)}
            events.append({"event": "continuation_started", "command": boundary,
                           "side": cp_side, "turn": checkpoint["state"]["turn"], **continuation})
            calls_this_turn = 0
            while True:
                destroyed = battle.army_status()["destroyed"]
                if any(destroyed):
                    outcome = "draw" if all(destroyed) else "win"
                    winner = None if all(destroyed) else int(destroyed[0])
                    reason = "mutual_destruction" if all(destroyed) else "army_destroyed"
                    break
                if completed_turns >= 2*config["max_rounds"]:
                    outcome, winner, reason = "draw", None, "turn_limit"
                    break
                if calls_this_turn >= config["max_calls_per_turn"]:
                    raise RuntimeError("Continuation exceeded max_calls_per_turn")
                side = battle.snapshot()["state"]["side"]
                agent = agents[side]
                before = battle.snapshot()
                obs = battle.observe(side)
                actions = tuple(battle.available_actions())
                if recording is not None:
                    recording.begin(commands + 1, side, obs, agent)
                if side == cp_side:
                    service = CallPlanningServices(agent.planner, PlanningState.from_battle(battle, side),
                                                   side, obs["turn"])
                    try:
                        action = agent.act_with_planning(obs, actions, service)
                    finally:
                        service.expire()
                    events.extend({**e, "side": side, "command": commands} for e in agent.drain_diagnostics())
                else:
                    action = agent.act(obs, actions)
                if recording is not None:
                    recording.end(action)
                if before != battle.snapshot():
                    raise AssertionError("Planning changed physical battle or RNG")
                if not isinstance(action, Action) or action not in actions:
                    raise ValueError("Agent returned an illegal continuation command")
                result = battle.step(action)
                if not result.ok:
                    raise AssertionError("Legal continuation command rejected")
                commands += 1
                calls[side] += 1
                calls_this_turn += 1
                if side == cp_side:
                    agent.on_action_result(action, result, battle.observe(side))
                    events.extend({**e, "side": side, "command": commands} for e in agent.drain_diagnostics())
                if action.kind == "end_turn":
                    completed_turns += 1
                    calls_this_turn = 0
            final = battle.snapshot()
            output = battle.replay()
            output.update(ai_events=events, experiment=continuation)
            result = {**previous, "status": "completed", "outcome": outcome, "winner": winner,
                      "reason": reason, "error": None, "commands": commands,
                      "completed_turns": completed_turns, "completed_rounds": completed_turns//2,
                      "agent_calls": calls, "alive": battle.army_status()["alive"],
                      "remaining_hp": [sum(u["hp"] for u in final["units"] if u["side"] == s) for s in (0, 1)],
                      "final_state_sha256": digest(final), "continuation": continuation}
        if output["steps"][:boundary] != replay["steps"][:boundary]:
            raise AssertionError("Continuation changed source prefix")
        with Battle(output["initial"]) as verification:
            for index, entry in enumerate(output["steps"], 1):
                actual = verification.step(entry["action"])
                if asdict(actual) != entry["result"] or digest(verification.snapshot()) != entry["state_sha256"]:
                    raise AssertionError(f"Continuation failed replay verification at {index}")
            if verification.snapshot() != final:
                raise AssertionError("Continuation final state differs")
        return MatchOutcome(result, output, final, recording)
    finally:
        agents[cp_side].close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--after", type=int, required=True)
    parser.add_argument("--search-config", type=Path, required=True)
    parser.add_argument("--diagnostics", choices=("off", "compact", "full"), default="off")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output directory; existing replays are never replaced")
    settings = read(args.search_config)
    if isinstance(settings.get("aggression"), str):
        settings["aggression"] = Aggression[settings["aggression"].upper()]
    outcome = resume(args.source, args.after, search_config=SearchConfig(**settings), diagnostics=args.diagnostics)
    outcome.save(args.output)
    checked = {"replay_verified": True, "prefix_verified": args.after,
               "continuation": True, "checkpoint_sha256": outcome.result["continuation"]["checkpoint_sha256"]}
    (args.output / "checked.json").write_text(json.dumps(checked, indent=2)+"\n", encoding="utf-8")
    (args.output.parent / "passport.json").write_text(
        json.dumps({"port": outcome.result["continuation"]["port"]}, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: outcome.result[k] for k in ("outcome", "winner", "commands", "remaining_hp")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
