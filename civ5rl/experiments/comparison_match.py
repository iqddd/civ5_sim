"""Torch-free worker match runner for rollout-comparison source/continuation battles."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from civ5rl.core.observation import ordered_actions
from civ5rl.engine.evaluation.proxy import evaluation_agent
from civ5rl.protocols.imitation import search_config
from civ5rl.provenance.common import read_json, write_json
from civ5sim import Action, MatchConfig, run_match
from civ5sim.cp_agent import CPAgent
from civ5sim.engine import digest


def _verify(directory):
    # Continuation replays start at the reseeded snapshot, but retain the original horizon.
    from civ5rl.experiments.evaluation import verify_replay

    verify_replay(directory)
    write_json(directory / "checked.json", {"replay_verified": True})


def match_task(task):
    """Run a source battle or one exact-prefix, reseeded continuation.

    Re-execute CP commands and verify each prefix state hash. The organizer-only
    before_decision hook reseeds simulator damage immediately before the chosen
    alternative; agents never receive this hook or hidden snapshot. Each branch
    then consumes its own stream. The saved continuation replay starts from the
    reseeded snapshot while keeping the original 60-round horizon.
    """
    cfg, case, folder, checkpoint, branch = task
    folder = Path(folder)
    command = 0
    history = []
    source = read_json(Path(branch["source"]) / "replay.json") if branch else None
    boundary = branch["command"] if branch else None
    prefix_verified = False
    turn_branch = branch is not None and "turn_variant" in branch
    changed_turn_done = False

    def hook(battle, index):
        nonlocal command, prefix_verified
        command = index
        if source is None:
            return
        if index <= boundary:
            expected = source["initial"] if index == 0 else None
            expected_hash = digest(expected) if index == 0 else source["steps"][index - 1]["state_sha256"]
            # run_match logs every executed command before invoking this hook.
            # The full independent replay verification still runs after saving.
            actual_hash = (
                battle._journal[-1]["state_sha256"]
                if index and len(battle._journal) == index
                else battle.state_digest()
            )
            if actual_hash != expected_hash:
                raise ValueError(f"Prefix state differs at {index}")
        if index == boundary:
            saved = battle.snapshot()
            saved["state"]["rng"] = branch["seed"]
            battle.reset(saved)
            prefix_verified = True

    class Learner:
        def __init__(self):
            self.inner = evaluation_agent(checkpoint, "argmax", temperature=0.5)

        def reset(self, context):
            self.inner.reset(context)
            self.requires_attack_preview = self.inner.requires_attack_preview

        def mode(self, value):
            setter = getattr(self.inner, "set_selection_mode", None)
            if setter:
                setter(value)
            else:
                self.inner.mode = value

        def act(self, obs, actions):
            nonlocal changed_turn_done
            legal = ordered_actions(actions)
            if branch and (command < boundary or not turn_branch and command == boundary):
                action = Action(
                    **(source["steps"][command]["action"] if command < boundary else branch["action"])
                )
                if action not in legal:
                    raise ValueError("Illegal prefix/alternative")
                self.inner.observe_step(obs, legal, action)
                if command == boundary and getattr(self.inner, "prefix_skip", 0):
                    raise ValueError("Restored prefix does not match branch boundary")
                return action
            changed = turn_branch and not changed_turn_done
            if turn_branch and command == boundary:
                import random

                self.inner.rng = random.Random(branch["sampling_seed"])
                self.mode("sample" if branch["turn_variant"] else "argmax")
            if changed and len(legal) == 1:
                action = legal[0]
                self.inner.observe_step(obs, legal, action)
            else:
                action = self.inner.act(obs, legal)
            if changed:
                history.append(
                    {
                        "command": command,
                        "observation": obs,
                        "actions": [asdict(a) for a in legal],
                        "chosen": legal.index(action),
                        "forced": len(legal) == 1,
                    }
                )
                if action.kind == "end_turn":
                    changed_turn_done = True
                    self.mode("argmax")
            if not branch:
                events = self.inner.drain_diagnostics()
                alternative = next(e["alternative"] for e in events if e["kind"] == "proposal")
                # Battle.observe returns a new detached tree on each decision;
                # the public encoder/selection never mutates it afterwards.
                history.append(
                    {
                        "command": command,
                        "observation": obs,
                        "actions": [asdict(a) for a in legal],
                        "chosen": legal.index(action),
                        "alternative": alternative,
                    }
                )
            return action

    class Opponent:
        def __init__(self):
            self.inner = CPAgent(search_config(cfg))

        def reset(self, context):
            self.inner.reset(context)
            self.planner = self.inner.planner

        def act_with_planning(self, obs, actions, services):
            action = self.inner.act_with_planning(obs, actions, services)
            if branch and command < boundary and asdict(action) != source["steps"][command]["action"]:
                raise ValueError("CP prefix command differs")
            return action

        def on_action_result(self, *args):
            return self.inner.on_action_result(*args)

    learner, cp = Learner(), Opponent()
    agents = [None, None]
    side = case["learner_side"]
    agents[side] = learner
    agents[1 - side] = cp
    scenario = case["scenario"]
    config = MatchConfig(
        scenario["map"],
        case["armies"],
        cfg.get("_smoke_rounds", 60),
        fog=scenario["fog"],
        random_damage=scenario["random_damage"],
        max_calls_per_turn=case["max_calls_per_turn"],
        turn_reset_mode="end_turn",
    )
    try:
        match = run_match(
            config,
            agents,
            scenario=scenario,
            agent_seeds=case["agent_seeds"],
            before_decision=hook,
        )
    finally:
        cp.inner.close()
    folder.mkdir(parents=True, exist_ok=True)
    match.save(folder)
    write_json(folder / "case.json", case)
    if match.result["status"] != "completed":
        raise RuntimeError(match.result["error"])
    if branch and not prefix_verified:
        raise ValueError("Branch boundary not reached")
    _verify(folder)
    if turn_branch:
        write_json(folder / "turn-history.json", history)
        write_json(folder / "branch.json", branch)
        # reset() starts the continuation journal at the branch boundary.
        # Keep the physical boundary hash distinct from agent/GRU memory.
        last = history[-1]
        boundary_index = last["command"] - boundary
        step = match.replay["steps"][boundary_index]
        if step["action"] != last["actions"][last["chosen"]]:
            raise ValueError("Changed-turn boundary does not match continuation replay")
        turn_summary = {
            "turn_sequence_sha256": digest([r["actions"][r["chosen"]] for r in history]),
            "turn_free_decisions": sum(not r["forced"] for r in history),
            "boundary_state_sha256": step["state_sha256"],
            "boundary_command": last["command"],
            "boundary_replay_index": boundary_index,
            "boundary_reason": "end_turn" if step["action"]["kind"] == "end_turn" else "battle_end",
        }
    if not branch:
        write_json(folder / "history.json", history)
        if hasattr(learner.inner, "finish_source"):
            learner.inner.finish_source()
    row = dict(
        winner=match.result["winner"],
        side=side,
        commands=match.result["commands"],
        rounds=match.result["completed_rounds"],
        prefix_verified=prefix_verified if branch else None,
        **(
            dict(
                turn_variant=branch["turn_variant"],
                repetition=branch["repetition"],
                position_key=branch["position_key"],
                turn_commands=len(history),
                damage_seed=branch["seed"],
                sampling_seed=branch["sampling_seed"],
                **turn_summary,
            )
            if turn_branch
            else {}
        ),
    )
    write_json(folder / "completed.json", row)
    return str(folder)
