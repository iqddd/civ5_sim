"""Torch-free worker match runner for rollout-comparison source/continuation battles."""

from dataclasses import asdict
from pathlib import Path

from civ5rl.core.observation import ordered_actions
from civ5rl.engine.evaluation_proxy import evaluation_agent
from civ5rl.protocols.imitation import search_config
from civ5rl.provenance.common import read_json, write_json
from civ5sim import Action, MatchConfig, run_match
from civ5sim.cp_agent import CPAgent
from civ5sim.engine import digest


def _verify(directory):
    # Continuation replays start at the reseeded snapshot, but retain the original horizon.
    from civ5rl.experiments.evaluation import verify_replay

    verify_replay(directory)
    write_json(directory / "checked.json", dict(replay_verified=True))


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

    def hook(battle, index):
        nonlocal command, prefix_verified
        command = index
        if source is None:
            return
        if index <= boundary:
            expected = source["initial"] if index == 0 else None
            # Verification consumes the snapshot synchronously without mutation.
            actual = battle._snapshot(copy_static=False)
            expected_hash = digest(expected) if index == 0 else source["steps"][index - 1]["state_sha256"]
            if digest(actual) != expected_hash:
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

        def act(self, obs, actions):
            legal = ordered_actions(actions)
            if branch and command <= boundary:
                action = Action(**(source["steps"][command]["action"] if command < boundary else branch["action"]))
                if action not in legal:
                    raise ValueError("Illegal prefix/alternative")
                self.inner.observe_step(obs, legal, action)
                if command == boundary and getattr(self.inner,'prefix_skip',0):
                    raise ValueError('Restored prefix does not match branch boundary')
                return action
            action = self.inner.act(obs, legal)
            if not branch:
                events = self.inner.drain_diagnostics()
                alternative = next(e["alternative"] for e in events if e["kind"] == "proposal")
                # Battle.observe returns a new detached tree on each decision;
                # the public encoder/selection never mutates it afterwards.
                history.append(
                    dict(
                        command=command,
                        observation=obs,
                        actions=[asdict(a) for a in legal],
                        chosen=legal.index(action),
                        alternative=alternative,
                    )
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
        60,
        fog=scenario["fog"],
        random_damage=scenario["random_damage"],
        max_calls_per_turn=case["max_calls_per_turn"],
        turn_reset_mode="end_turn",
    )
    try:
        match = run_match(config, agents, scenario=scenario, agent_seeds=case["agent_seeds"], before_decision=hook)
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
    if not branch:
        write_json(folder / "history.json", history)
        if hasattr(learner.inner,'finish_source'):
            learner.inner.finish_source()
    row = dict(
        winner=match.result["winner"],
        side=side,
        commands=match.result["commands"],
        rounds=match.result["completed_rounds"],
        prefix_verified=prefix_verified if branch else None,
    )
    write_json(folder / "completed.json", row)
    return str(folder)
