"""Pilot-only validation, paired by physical side and parent world."""

from collections import defaultdict
from pathlib import Path

import numpy as np

from civ5rl.core.agent import RLAgent
from civ5rl.protocols.common import bootstrap_ci95, learner_outcome
from civ5rl.provenance import bound_cases, case_manifest, validate_cases
from civ5rl.provenance.common import write_json
from civ5sim.engine import Battle, digest
from civ5sim.match import MatchConfig, MatchOutcome, run_match


def require_split(config, split, mode="pilot", frozen_plan=None):
    if split in ("test", "diagnostic_test"):
        raise ValueError(f"Closed split: {split}; final evaluation is not implemented")
    if mode == "pilot" and split not in config["evaluation"]["pilot_allowed_splits"]:
        raise ValueError(f"Pilot cannot evaluate closed split {split}")


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
                specs.append({**base, "opponent": opponent, "learner_side": side, "case_id": i})
    return bound_cases(
        case_manifest(specs, source="rl01.validation", split="validation", namespace="validation")
    )


def _wdl_stats(outcomes):
    games = len(outcomes)
    wins = sum(o == "W" for o in outcomes)
    draws = sum(o == "D" for o in outcomes)
    losses = sum(o == "L" for o in outcomes)
    return {
        "games": games,
        "wins": int(wins),
        "draws": int(draws),
        "losses": int(losses),
        "score": float((wins + 0.5 * draws) / games) if games else None,
    }


def summarize(episodes, samples=10000, seed=20261103):
    grouped = defaultdict(list)
    for e in episodes:
        s, r = e["spec"], e["result"]
        grouped[s["opponent"]].append((s, r, learner_outcome(r["winner"], s["learner_side"])))
    output = {}
    for opponent, rows in grouped.items():
        outcomes = [o for _, _, o in rows]
        worlds = sorted({s["parent_world"] for s, _, _ in rows})
        world_scores = np.asarray(
            [
                np.mean(
                    [
                        1.0 if o == "W" else 0.5 if o == "D" else 0.0
                        for s, _, o in rows
                        if s["parent_world"] == w
                    ]
                )
                for w in worlds
            ]
        )
        ci = bootstrap_ci95(world_scores, samples=samples, seed=seed)
        output[opponent] = {
            **_wdl_stats(outcomes),
            "ci95": ci.tolist(),
            "sides": {
                str(side): _wdl_stats([o for s, _, o in rows if s["learner_side"] == side]) for side in (0, 1)
            },
            "rounds_mean": float(np.mean([r["rounds"] for _, r, _ in rows])),
            "command_mean": float(np.mean([r["commands"] for _, r, _ in rows])),
            "turn_limit_draws": sum(r["reason"] == "turn_limit" for _, r, _ in rows),
            "learner_hp_mean": float(np.mean([r["remaining_hp"][s["learner_side"]] for s, r, _ in rows])),
            "opponent_hp_mean": float(
                np.mean([r["remaining_hp"][1 - s["learner_side"]] for s, r, _ in rows])
            ),
            "first_contact_turn_mean": float(
                np.mean([r["first_contact_turn"] for _, r, _ in rows if r["first_contact_turn"] is not None])
            )
            if any(r["first_contact_turn"] is not None for _, r, _ in rows)
            else None,
        }
    return output


def evaluate(policy, collector, specs, config, directory, monitor=None, trusted_manifest=None):
    directory = Path(directory)
    require_split(config, "validation")
    for spec in specs:
        if spec.get("split") in ("test", "diagnostic_test") or spec.get("provenance", {}).get("split") in (
            "test",
            "diagnostic_test",
        ):
            raise ValueError("Closed split in evaluation cases")

    if trusted_manifest is None:
        trusted_manifest = case_manifest(
            validation_specs(collector.factory, config),
            source="rl01.validation",
            split="validation",
            namespace="validation",
        )
    if trusted_manifest["split"] != "validation":
        raise ValueError("Validation evaluator requires trusted validation cases")
    validate_cases(specs, trusted_manifest)
    if monitor:
        monitor.mark("validation")
    episodes, _, times = collector.collect(
        policy,
        None,
        0,
        monitor=None,
        fixed_specs=specs,
        trusted_manifest=trusted_manifest,
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
        for target in ("W", "L", "D"):
            matches = [o for o in outcomes if learner_outcome(o["result"]["winner"], o["side"]) == target]
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
        outcome = run_match(config, agents, scenario=s["scenario"], agent_seeds=s["agent_seeds"])
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
            "agents": ["learner" if side == s["learner_side"] else s["opponent"] for side in (0, 1)],
            "seeds": s.get("streams", {"battle": s["scenario"]["seed"], "agents": s["agent_seeds"]}),
            "initial_state_sha256": digest(replay["initial"]),
            "final_state_sha256": digest(final),
        }
        MatchOutcome(result, replay, final).save(directory)
    from tools.view_combat import prepare_replay

    prepare_replay(Path(directory) / "replay.json")
