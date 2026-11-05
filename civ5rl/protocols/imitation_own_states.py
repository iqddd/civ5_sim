"""Ordinary matches where policy executes and a replanning CP only advises."""

import random
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from civ5rl.core.observation import ordered_actions
from civ5rl.protocols.corpus import ScenarioFactory
from civ5rl.protocols.imitation import dump_episode, file_hash, search_config
from civ5rl.provenance.common import ROOT, write_json
from civ5sim import MatchConfig
from civ5sim.cp_agent import CPAgent
from civ5sim.engine import digest
from civ5sim.match import run_match


class ReplanningCPAdvisor(CPAgent):
    """Keep real public contacts/exploration; never continue unexecuted plans."""

    def discard_tactics(self):
        self._new_turn()
        self._turn = None
        for name in ("_objectives", "_blocked", "_combat_targets", "_target_blocked"):
            setattr(self, name, {})
        self._events = []

    def advise(self, observation, actions, planning):
        self.discard_tactics()
        try:
            return super().act_with_planning(deepcopy(observation), actions, planning)
        finally:
            self.discard_tactics()

    def observe_actual_result(self, action, result, observation):
        # Do not claim that an unexecuted assignment matched the policy action.
        self._observe(observation)
        self.discard_tactics()


class AdvisedPolicy:
    def __init__(self, learner, config):
        self.learner = learner
        self.advisor = ReplanningCPAdvisor(search_config(config))

    @property
    def planner(self):
        return self.advisor.planner

    def reset(self, context):
        self.public_context = deepcopy(context)
        self.learner.reset(deepcopy(context))
        self.advisor.reset(deepcopy(context))
        self.decisions, self.timings = [], []

    def act_with_planning(self, observation, actions, planning):
        public = deepcopy(observation)
        legal = ordered_actions(actions)
        start = perf_counter()
        try:
            teacher = self.advisor.advise(observation, actions, planning)
        finally:
            planning.expire()
        advised = perf_counter()
        executed = self.learner.act(deepcopy(public), legal)
        stopped = perf_counter()
        if teacher not in legal or executed not in legal:
            raise ValueError("Advice or policy command is outside the public legal list")
        self.decisions.append({
            "observation": public, "actions": [asdict(a) for a in legal],
            "teacher_action": legal.index(teacher), "executed_action": legal.index(executed),
            "supervised": len(legal) >= 2,
        })
        self.timings.append({"advice_seconds": advised-start, "policy_seconds": stopped-advised})
        return executed

    def on_action_result(self, action, result, observation):
        self.advisor.observe_actual_result(action, result, observation)

    def drain_diagnostics(self):
        return self.learner.drain_diagnostics()

    def close(self):
        self.advisor.close()


def physical_hash(scenario):
    # Seeds, side-to-play and unit list order do not create a new physical layout.
    units = [
        (u["side"], u["type"], u["tile"], u.get("hp", 100))
        for u in scenario["units"]
    ]
    flipped = [(1-s, k, t, hp) for s, k, t, hp in units]
    return digest({"map": digest(scenario["map"]), "units": min(sorted(units), sorted(flipped))})


def make_case(config, base, split, index, *, attempt=0):
    if split not in ("dev", "cycle-01", "cycle-02"):
        raise ValueError("Closed or unsupported own-state split")
    worlds = base["worlds"]["holdout" if split == "dev" else "training"]
    namespace = "cp_imitation_own_states_" + split
    strata = list(range(16))
    random.Random(int(digest([17, namespace, index//16, "balance"]), 16)).shuffle(strata)
    cell = strata[index % 16]
    enemy, mode, side, contact = (cell & 1, (cell >> 1) & 1, (cell >> 2) & 1, (cell >> 3) & 1)
    world = worlds[index % len(worlds)]
    maps = [m for m in base["maps"] if m["parent_seed"] == world and m["split"] == "train"]
    generator = ScenarioFactory({
        "seed": 17, "corpus": {"placement_rejection_limit": 100},
        "environment": {"max_calls_per_turn": config["max_calls_per_turn"]},
    }, {"schema": 1, "maps": maps})
    case = generator.make(
        index, {"map_size": [10, 10], "units_per_side": [2, 2],
                "roster": ["warrior", "archer"], "max_rounds": 30,
                "contact_probability": contact, "opponents": {"greedy": 1}},
        master=17, namespace=namespace+f"_attempt_{attempt}", split="train",
        case_index=index//len(worlds),
    )
    case.update(case_id=index, dataset_split=split, learner_side=side,
                opponent="cp" if enemy else "greedy", policy_mode="argmax" if mode else "sample",
                generation_attempt=attempt, source="rl01.cp_imitation_own_states")
    case["physical_hash"] = physical_hash(case["scenario"])
    case["case_hash"] = digest(case)
    return case


def match_task(task):
    config, base, case, checkpoint_path, directory = task
    from civ5rl.core.agent import RLAgent
    from civ5rl.provenance.common import configure

    directory = Path(directory)
    expected = make_case(config, base, case["dataset_split"], case["case_id"], attempt=case["generation_attempt"])
    if case != expected:
        raise ValueError("Own-state case differs from trusted deterministic source")
    configure(17, "cpu")
    write_json(directory/"case.json", case)
    learner = RLAgent.from_checkpoint(checkpoint_path, "cpu", case["policy_mode"])
    agent = AdvisedPolicy(learner, config)
    enemy = CPAgent(search_config(config)) if case["opponent"] == "cp" else "greedy"
    agents = [None, None]
    side = case["learner_side"]
    agents[side], agents[1-side] = agent, enemy
    timings = []
    started = perf_counter()
    try:
        outcome = run_match(
            MatchConfig(case["scenario"]["map"], case["armies"], 30,
                        max_calls_per_turn=config["max_calls_per_turn"]),
            agents, scenario=case["scenario"], agent_seeds=case["agent_seeds"], metrics=timings.append,
        )
        outcome.save(directory)
    finally:
        agent.close()
        if isinstance(enemy, CPAgent):
            enemy.close()
    if outcome.result["status"] != "completed":
        raise RuntimeError(f"Own-state match failed at {directory}: {outcome.result['error']}")
    episode = {
        "schema": 2, "kind": "rl01.CP_own_state_demonstration", "case_id": case["case_id"],
        "split": case["dataset_split"], "parent_world": case["parent_world"],
        "case_hash": case["case_hash"], "context": agent.public_context,
        "behavior_checkpoint_sha256": file_hash(checkpoint_path), "policy_mode": case["policy_mode"],
        "opponent": case["opponent"], "decisions": agent.decisions,
        "outcome": {"winner": outcome.result["winner"], "reason": outcome.result["reason"]},
    }
    dump_episode(directory/"demonstration.json.gz", episode)
    commands, teacher_commands = Counter(), Counter()
    labels = disagreements = 0
    for row in agent.decisions:
        commands[row["actions"][row["executed_action"]]["kind"]] += 1
        teacher_commands[row["actions"][row["teacher_action"]]["kind"]] += 1
        labels += row["supervised"]
        disagreements += row["supervised"] and row["teacher_action"] != row["executed_action"]
    native = defaultdict(float)
    for row in timings:
        native[f"{row['kind']}_side_{row.get('side', -1)}"] += row["ns"]/1e9
    result = {
        "case_id": case["case_id"], "case_hash": case["case_hash"], "physical_hash": case["physical_hash"],
        "side": side, "parent_world": case["parent_world"], "placement": case["placement"],
        "opponent": case["opponent"], "policy_mode": case["policy_mode"], "status": "completed",
        "seconds": perf_counter()-started, "labels": labels, "forced": len(agent.decisions)-labels,
        "disagreements": disagreements, "executed_actions": dict(commands), "teacher_actions": dict(teacher_commands),
        "winner": outcome.result["winner"], "reason": outcome.result["reason"],
        "rounds": outcome.result["completed_rounds"],
        "advice_seconds": sum(r["advice_seconds"] for r in agent.timings),
        "policy_seconds": sum(r["policy_seconds"] for r in agent.timings), "native_timings": dict(native),
        "demonstration_sha256": file_hash(directory/"demonstration.json.gz"),
    }
    write_json(directory/"metrics.json", result)
    return result
