"""Pilot-only validation, paired by physical side and parent world."""

from collections import defaultdict
from pathlib import Path

import numpy as np

from civ5sim.match import MatchConfig, run_match

from .agent import RLAgent
from .common import write_json


def require_split(config, split, mode="pilot", frozen_plan=None):
    if mode == "pilot" and split not in config["evaluation"]["pilot_allowed_splits"]:
        raise ValueError(f"Pilot cannot evaluate closed split {split}")
    if split in ("test", "diagnostic_test") and not frozen_plan:
        raise ValueError("Final test requires a frozen A/B/C evaluation plan")


def validation_specs(factory, config):
    require_split(config, "validation")
    stage = dict(config["evaluation"], contact_probability=0, opponents={"greedy": 1.0})
    master = config["evaluation"]["manifest_seeds"]["validation"]
    specs = []
    for i in range(config["evaluation"]["validation_cases"]):
        base = factory.make(
            i,
            stage,
            master=master,
            namespace="validation",
            split="validation",
            case_index=i,
        )
        for opponent in ("random", "greedy"):
            for side in (0, 1):
                specs.append(
                    {**base, "opponent": opponent, "learner_side": side, "case_id": i}
                )
    return specs


def summarize(episodes, samples=10000, seed=20261103):
    grouped = defaultdict(list)
    for e in episodes:
        s, r = e["spec"], e["result"]
        grouped[s["opponent"]].append((s, r))
    output = {}
    for opponent, rows in grouped.items():
        rewards = np.asarray([r["reward"] for _, r in rows])
        worlds = sorted({s["parent_world"] for s, _ in rows})
        world_scores = np.asarray(
            [
                np.mean(
                    [(r["reward"] + 1) / 2 for s, r in rows if s["parent_world"] == w]
                )
                for w in worlds
            ]
        )
        rng = np.random.default_rng(seed)
        bootstrap = world_scores[
            rng.integers(len(worlds), size=(samples, len(worlds)))
        ].mean(1)
        output[opponent] = {
            "games": len(rows),
            "wins": int((rewards == 1).sum()),
            "draws": int((rewards == 0).sum()),
            "losses": int((rewards == -1).sum()),
            "score": float(((rewards + 1) / 2).mean()),
            "ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "sides": {
                str(side): {
                    "games": sum(s["learner_side"] == side for s, _ in rows),
                    "wins": sum(
                        s["learner_side"] == side and r["reward"] == 1 for s, r in rows
                    ),
                    "draws": sum(
                        s["learner_side"] == side and r["reward"] == 0 for s, r in rows
                    ),
                    "losses": sum(
                        s["learner_side"] == side and r["reward"] == -1 for s, r in rows
                    ),
                    "score": float(
                        np.mean(
                            [
                                (r["reward"] + 1) / 2
                                for s, r in rows
                                if s["learner_side"] == side
                            ]
                        )
                    )
                    if any(s["learner_side"] == side for s, _ in rows)
                    else None,
                }
                for side in (0, 1)
            },
            "rounds_mean": float(np.mean([r["rounds"] for _, r in rows])),
            "command_mean": float(np.mean([r["commands"] for _, r in rows])),
            "turn_limit_draws": sum(r["reason"] == "turn_limit" for _, r in rows),
            "learner_hp_mean": float(
                np.mean([r["remaining_hp"][s["learner_side"]] for s, r in rows])
            ),
            "opponent_hp_mean": float(
                np.mean([r["remaining_hp"][1 - s["learner_side"]] for s, r in rows])
            ),
            "first_contact_turn_mean": float(
                np.mean(
                    [
                        r["first_contact_turn"]
                        for _, r in rows
                        if r["first_contact_turn"] is not None
                    ]
                )
            )
            if any(r["first_contact_turn"] is not None for _, r in rows)
            else None,
        }
    return output


def evaluate(policy, collector, specs, config, directory, monitor=None):
    directory = Path(directory)
    require_split(config, "validation")
    if monitor:
        monitor.mark("validation")
    episodes, _, times = collector.collect(
        policy, None, 0, monitor=None, fixed_specs=specs
    )
    summary = summarize(episodes, config["evaluation"]["bootstrap"]["samples"])
    write_json(directory / "summary.json", {"metrics": summary, "timings": times})
    write_json(
        directory / "outcomes.json",
        [
            {
                "case_id": e["spec"]["case_id"],
                "side": e["spec"]["learner_side"],
                "opponent": e["spec"]["opponent"],
                "parent_world": e["spec"]["parent_world"],
                "result": e["result"],
            }
            for e in episodes
        ],
    )
    return summary


def save_replays(policy, specs, device, directory, count=8, outcomes=None):
    from tools.view_combat import prepare_replay

    selected = []
    if outcomes:
        for reward in (1, -1, 0):
            matches = [o for o in outcomes if o["result"]["reward"] == reward]
            if matches:
                o = matches[0]
                selected.append(
                    next(
                        s
                        for s in specs
                        if s["case_id"] == o["case_id"]
                        and s["learner_side"] == o["side"]
                        and s["opponent"] == o["opponent"]
                    )
                )
    selected.extend(s for s in specs if s not in selected)
    paths = []
    for i, s in enumerate(selected[:count]):
        agents = [s["opponent"], s["opponent"]]
        agents[s["learner_side"]] = RLAgent(policy, device)
        config = MatchConfig(
            s["scenario"]["map"],
            s["armies"],
            s["max_rounds"],
            max_calls_per_turn=s["max_calls_per_turn"],
        )
        outcome = run_match(
            config, agents, scenario=s["scenario"], agent_seeds=s["agent_seeds"]
        )
        if outcome.result["status"] != "completed":
            raise RuntimeError(outcome.result["error"])
        path = Path(directory) / f"match-{i:02}"
        outcome.save(path)
        write_json(path / "manifest-case.json", s)
        prepare_replay(path / "replay.json")
        paths.append(str(path))
    return paths


def save_training_replay(episode, directory):
    """Replay exact collected commands, independently of subsequently updated weights."""
    from civ5sim.engine import Battle, digest
    from civ5sim.match import MatchOutcome

    s, r = episode["spec"], episode["result"]
    with Battle(s["scenario"], log=True) as battle:
        for action in episode["joint_commands"]:
            result = battle.step(action)
            if not result.ok:
                raise RuntimeError("Collected training replay diverged")
        replay, final = battle.replay(), battle.snapshot()
        result = {
            "schema": 1,
            "kind": "civ5sim.match_result",
            "status": "completed",
            "winner": r["winner"],
            "outcome": "draw" if r["winner"] is None else "win",
            "reason": r["reason"],
            "error": None,
            "commands": r["commands"],
            "agents": [
                "learner" if side == s["learner_side"] else s["opponent"]
                for side in (0, 1)
            ],
            "seeds": s["streams"],
            "initial_state_sha256": digest(replay["initial"]),
            "final_state_sha256": digest(final),
        }
        MatchOutcome(result, replay, final).save(directory)
    from tools.view_combat import prepare_replay

    prepare_replay(Path(directory) / "replay.json")
