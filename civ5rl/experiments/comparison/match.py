"""Torch-free worker match runner for source battles and exact-prefix continuations."""

from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path

from civ5rl.core.observation import ordered_actions
from civ5rl.engine.evaluation.proxy import evaluation_agent
from civ5rl.engine.evaluation.replay import verify_replay
from civ5rl.protocols.imitation import search_config
from civ5rl.provenance.common import read_json, write_json
from civ5sim import Action, MatchConfig, run_match
from civ5sim.cp_agent import CPAgent
from civ5sim.engine import digest
from civ5sim.telemetry import measure


class _Prefix:
    """Replay the source under both agents and reseed only at the branch boundary."""

    def __init__(self, branch, *, metrics, side):
        self.branch = branch
        self.source = read_json(Path(branch["source"]) / "replay.json") if branch else None
        self.boundary = branch["command"] if branch else None
        self.turn_branch = branch is not None and "turn_variant" in branch
        self.command = 0
        self.verified = False
        self.metrics = metrics
        self.side = side

    @property
    def in_prefix(self):
        return self.source is not None and self.command < self.boundary

    @property
    def at_boundary(self):
        return self.command == self.boundary

    def recorded_action(self):
        return self.source["steps"][self.command]["action"]

    def before_decision(self, battle, command):
        with measure(self.metrics, "prefix_hook", side=self.side):
            self.command = command
            if self.source is None or command > self.boundary:
                return
            expected = (
                self.source["steps"][command - 1]["state_sha256"]
                if command
                else digest(self.source["initial"])
            )
            # The simulator logs each executed command before calling this hook.
            actual = (
                battle._journal[-1]["state_sha256"]
                if command and len(battle._journal) == command
                else battle.state_digest()
            )
            if actual != expected:
                raise ValueError(f"Prefix state differs at {command}")
            if self.at_boundary:
                snapshot = battle.snapshot()
                snapshot["state"]["rng"] = self.branch["seed"]
                battle.reset(snapshot)
                self.verified = True


class _Learner:
    def __init__(self, checkpoint, prefix):
        self.prefix = prefix
        self.history = []
        self.changed_turn_done = False
        self.inner = evaluation_agent(
            checkpoint,
            "argmax",
            temperature=0.5,
            encoding_metrics=prefix.metrics,
            metrics_side=prefix.side,
        )

    def reset(self, context):
        self.inner.reset(context)
        self.requires_attack_preview = self.inner.requires_attack_preview

    def _set_mode(self, mode):
        setter = getattr(self.inner, "set_selection_mode", None)
        if setter:
            setter(mode)
        else:
            self.inner.mode = mode

    def _replay_action(self, observation, legal):
        prefix = self.prefix
        command = prefix.recorded_action() if prefix.in_prefix else prefix.branch["action"]
        action = Action(**command)
        if action not in legal:
            raise ValueError("Illegal prefix/alternative")
        self.inner.observe_step(observation, legal, action)
        if prefix.at_boundary and getattr(self.inner, "prefix_skip", 0):
            raise ValueError("Restored prefix does not match branch boundary")
        return action

    def _record(self, observation, legal, action, **extra):
        # The simulator returns a detached tree for each public observation.
        self.history.append(
            {
                "command": self.prefix.command,
                "observation": observation,
                "actions": [asdict(command) for command in legal],
                "chosen": legal.index(action),
                **extra,
            }
        )

    def act(self, observation, actions):
        prefix = self.prefix
        legal = ordered_actions(actions)
        if prefix.in_prefix or (prefix.at_boundary and not prefix.turn_branch):
            return self._replay_action(observation, legal)
        if prefix.turn_branch and prefix.at_boundary:
            self.inner.rng = random.Random(prefix.branch["sampling_seed"])
            self._set_mode("sample" if prefix.branch["turn_variant"] else "argmax")
        changed_turn = prefix.turn_branch and not self.changed_turn_done
        if changed_turn and len(legal) == 1:
            action = legal[0]
            self.inner.observe_step(observation, legal, action)
        else:
            action = self.inner.act(observation, legal)
        if changed_turn:
            self._record(observation, legal, action, forced=len(legal) == 1)
            if action.kind == "end_turn":
                self.changed_turn_done = True
                self._set_mode("argmax")
        if prefix.branch is None:
            proposal = next(event for event in self.inner.drain_diagnostics() if event["kind"] == "proposal")
            self._record(observation, legal, action, alternative=proposal["alternative"])
        return action


class _Opponent:
    def __init__(self, config, prefix):
        self.prefix = prefix
        self.inner = CPAgent(search_config(config))

    def reset(self, context):
        self.inner.reset(context)
        self.planner = self.inner.planner

    def act_with_planning(self, observation, actions, services):
        action = self.inner.act_with_planning(observation, actions, services)
        if self.prefix.in_prefix and asdict(action) != self.prefix.recorded_action():
            raise ValueError("CP prefix command differs")
        return action

    def on_action_result(self, *args):
        return self.inner.on_action_result(*args)


def _save_turn(directory, history, replay, branch):
    write_json(directory / "turn-history.json", history)
    write_json(directory / "branch.json", branch)
    # The reset starts the continuation journal at the reseeded snapshot. The
    # boundary hash describes physical state, independently of agent memory.
    last = history[-1]
    boundary_index = last["command"] - branch["command"]
    step = replay["steps"][boundary_index]
    if step["action"] != last["actions"][last["chosen"]]:
        raise ValueError("Changed-turn boundary does not match continuation replay")
    return {
        "turn_variant": branch["turn_variant"],
        "repetition": branch["repetition"],
        "position_key": branch["position_key"],
        "turn_commands": len(history),
        "damage_seed": branch["seed"],
        "sampling_seed": branch["sampling_seed"],
        "turn_sequence_sha256": digest([row["actions"][row["chosen"]] for row in history]),
        "turn_free_decisions": sum(not row["forced"] for row in history),
        "boundary_state_sha256": step["state_sha256"],
        "boundary_command": last["command"],
        "boundary_replay_index": boundary_index,
        "boundary_reason": "end_turn" if step["action"]["kind"] == "end_turn" else "battle_end",
    }


def match_task(task, *, metrics=None):
    """Run a source battle or continuation, optionally emitting worker timing samples.

    The callable receives simulator samples plus the prefix hook, replay check and
    proxy encoding scopes. Bind a picklable callable if sending it across a process.
    """
    config, case, directory, checkpoint, branch = task
    directory = Path(directory)
    side = case["learner_side"]
    prefix = _Prefix(branch, metrics=metrics, side=side)
    learner = _Learner(checkpoint, prefix)
    opponent = _Opponent(config, prefix)
    agents = [learner, opponent] if side == 0 else [opponent, learner]
    scenario = case["scenario"]
    match_config = MatchConfig(
        scenario["map"],
        case["armies"],
        case["max_rounds"],
        fog=scenario["fog"],
        random_damage=scenario["random_damage"],
        max_calls_per_turn=case["max_calls_per_turn"],
        turn_reset_mode="end_turn",
    )
    try:
        match = run_match(
            match_config,
            agents,
            scenario=scenario,
            agent_seeds=case["agent_seeds"],
            before_decision=prefix.before_decision,
            metrics=metrics,
        )
    finally:
        opponent.inner.close()
    directory.mkdir(parents=True, exist_ok=True)
    match.save(directory)
    write_json(directory / "case.json", case)
    if match.result["status"] != "completed":
        raise RuntimeError(match.result["error"])
    if branch and not prefix.verified:
        raise ValueError("Branch boundary not reached")
    with measure(metrics, "replay_verification", side=side):
        verify_replay(directory, project_relative=False)
        write_json(directory / "checked.json", {"replay_verified": True})
    row = {
        "winner": match.result["winner"],
        "side": side,
        "commands": match.result["commands"],
        "rounds": match.result["completed_rounds"],
        "prefix_verified": prefix.verified if branch else None,
    }
    if prefix.turn_branch:
        row.update(_save_turn(directory, learner.history, match.replay, branch))
    if branch is None:
        write_json(directory / "history.json", learner.history)
        if hasattr(learner.inner, "finish_source"):
            learner.inner.finish_source()
    write_json(directory / "completed.json", row)
    return str(directory)
