"""Public CP demonstrations, deterministic world splits and ordinary matches.

This module intentionally avoids torch so CP-only workers never initialize CUDA.
"""

import gzip
import json
import os
import random
import sys
from collections import Counter, defaultdict
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from civ5rl.core.observation import ordered_actions
from civ5rl.protocols.common import self_digest
from civ5rl.protocols.corpus import ScenarioFactory, validate_corpus
from civ5rl.provenance.common import ROOT, file_sha256, read_json, write_json
from civ5sim import MatchConfig
from civ5sim.combat_ai import Aggression, SearchConfig
from civ5sim.cp_agent import CPAgent
from civ5sim.engine import digest
from civ5sim.match import run_match


def dump_episode(path, episode):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=1) as stream:
        json.dump(episode, stream, ensure_ascii=False, allow_nan=False)
    os.replace(tmp, path)


def load_episode(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def search_config(config):
    values = dict(config["teacher_search"])
    values["aggression"] = Aggression(values["aggression"])
    return SearchConfig(**values)


class RecordingCP(CPAgent):
    """Record only act inputs and the selected command; preserve CP callbacks."""

    def reset(self, context):
        super().reset(context)
        self.public_context = deepcopy(context)
        self.decisions = []

    def act_with_planning(self, observation, actions, planning):
        # CP uses its original legal-list order. Labels use the network's order.
        public_observation = deepcopy(observation)
        legal = ordered_actions(actions)
        action = super().act_with_planning(observation, actions, planning)
        if action not in legal:
            raise ValueError("CP command is outside the public legal list")
        self.decisions.append(
            {
                "observation": public_observation,
                "actions": [asdict(a) for a in legal],
                "chosen": legal.index(action),
                "supervised": len(legal) >= 2,
            }
        )
        return action


def imitation_scenario_factory(config, maps):
    sampling = {
        "seed": 17,
        "corpus": {"placement_rejection_limit": 100},
        "environment": {"max_calls_per_turn": config["max_calls_per_turn"]},
    }
    return ScenarioFactory(sampling, {"schema": 1, "maps": maps})


def imitation_stage(contact):
    return {
        "map_size": [10, 10],
        "units_per_side": [2, 2],
        "roster": ["warrior", "archer"],
        "max_rounds": 30,
        "contact_probability": contact,
        "opponents": {"greedy": 1},
    }


def _close_match_agent(close):
    pending = sys.exception()
    try:
        close()
    except Exception as error:
        if pending is None:
            raise
        pending.add_note(f"Agent cleanup failed: {error!r}")


def run_imitation_match(config, case, agents, directory):
    """Run and save one match, closing caller-owned agents on every exit."""
    timings = []
    started = perf_counter()
    with ExitStack() as stack:
        for agent in agents:
            close = getattr(agent, "close", None)
            if close is not None:
                stack.callback(_close_match_agent, close)
        outcome = run_match(
            MatchConfig(
                case["scenario"]["map"],
                case["armies"],
                30,
                max_calls_per_turn=config["max_calls_per_turn"],
            ),
            agents,
            scenario=case["scenario"],
            agent_seeds=case["agent_seeds"],
            metrics=timings.append,
        )
        outcome.save(directory)
    return outcome, timings, started


def summarize_match_timings(timings, *, by_side=False):
    totals = defaultdict(float)
    for row in timings:
        key = f"{row['kind']}_side_{row.get('side', -1)}" if by_side else row["kind"]
        totals[key] += row["ns"] / 1e9
    return dict(totals)


def make_case(config, manifest, split, index):
    if split not in ("training", "holdout", "validation"):
        raise ValueError("Closed or unsupported imitation split")
    worlds = manifest["worlds"][split]
    if split == "validation":
        world = worlds[(index // 2) % len(worlds)]
        contact = index % 2 == 0
    else:
        world = worlds[index % len(worlds)]
        contact = (index // 2) % 2 == 0
    corpus_split = "validation" if split == "validation" else "train"
    maps = [
        m
        for m in manifest["maps"]
        if m["parent_seed"] == world and m["split"] == corpus_split
    ]
    generator = imitation_scenario_factory(config, maps)
    stage = imitation_stage(int(contact))
    case = generator.make(
        index,
        stage,
        master=17,
        namespace="cp_imitation_" + split,
        split=corpus_split,
        case_index=index // len(worlds),
    )
    case.update(
        case_id=index,
        learner_side=index % 2,
        dataset_split=split,
        corpus_hash=manifest["source_corpus_sha256"],
    )
    case["case_hash"] = digest(case)
    return case


def prepare_manifest(config):
    source_path = ROOT / config["corpus"]
    original = read_json(source_path)
    # Validate only open maps; closed test data is never loaded for this phase.
    open_maps = [
        m
        for m in original["maps"]
        if m["split"] in ("train", "validation") and m["size"] == 10
    ]
    validate_corpus({**original, "maps": open_maps})
    train = sorted({m["parent_seed"] for m in open_maps if m["split"] == "train"})
    validation = sorted(
        {m["parent_seed"] for m in open_maps if m["split"] == "validation"}
    )
    if len(train) != 64 or len(validation) != 16:
        raise ValueError("Imitation requires 64 train and 16 validation worlds")
    random.Random(17).shuffle(train)
    worlds = {"training": train[7:], "holdout": train[:7], "validation": validation}
    # Also reject duplicated physical source maps across the new holdout split.
    owners = {}
    for split, selected in worlds.items():
        for entry in open_maps:
            if entry["parent_seed"] not in selected:
                continue
            for key in ("parent_hash", "crop_hash"):
                value = (key, entry[key])
                if value in owners and owners[value] != split:
                    raise ValueError(
                        "Physical source map leaked across imitation splits"
                    )
                owners[value] = split
    result = {
        "schema": 1,
        "kind": "rl01.cp_imitation",
        "source_corpus_sha256": file_sha256(source_path),
        "worlds": worlds,
        "worlds_hash": digest(worlds),
        "maps": open_maps,
        "case_generation": "make_case v1; seed=17; case ID determines world, side, contact and RNG",
    }
    result["validation_cases"] = [
        make_case(config, result, "validation", i) for i in range(32)
    ]
    result["hash"] = self_digest(result, key="hash")
    return result


def validate_case(case, config, manifest):
    split, index = case["dataset_split"], case["case_id"]
    expected = make_case(config, manifest, split, index)
    if split == "validation" and case["learner_side"] in (0, 1):
        expected["learner_side"] = case["learner_side"]
        expected["case_hash"] = self_digest(expected)
    if expected != case:
        raise ValueError("Imitation case differs from its trusted deterministic source")


def match_task(task):
    """One isolated ordinary match. No policy training or CP in a GPU collector."""
    config, manifest, case, directory, contender, opponent, device = task
    directory = Path(directory)
    validate_case(case, config, manifest)
    write_json(directory / "case.json", case)
    teacher = RecordingCP(search_config(config)) if contender == "teacher" else None
    agent = teacher
    if agent is None:
        from civ5rl.core.agent import RLAgent
        from civ5rl.provenance.common import configure

        configure(17, device)
        checkpoint_path = (
            contender["checkpoint"] if isinstance(contender, dict) else contender
        )
        mode = (
            contender.get("mode", "sample") if isinstance(contender, dict) else "sample"
        )
        if mode not in ("sample", "argmax"):
            raise ValueError("Unsupported imitation evaluation policy mode")
        agent = RLAgent.from_checkpoint(checkpoint_path, device=device, mode=mode)
    enemy = CPAgent(search_config(config)) if opponent == "cp" else "greedy"
    side = case["learner_side"]
    agents = [None, None]
    agents[side], agents[1 - side] = agent, enemy
    outcome, timings, start = run_imitation_match(config, case, agents, directory)
    seconds = perf_counter() - start
    by_kind = summarize_match_timings(timings)
    result = {
        "case_id": case["case_id"],
        "side": side,
        "seconds": seconds,
        "timings": by_kind,
        "agent_act_seconds": [
            sum(r["ns"] / 1e9 for r in timings if r["kind"] == "act" and r["side"] == s)
            for s in (0, 1)
        ],
        "status": outcome.result["status"],
        "winner": outcome.result["winner"],
        "reason": outcome.result["reason"],
        "rounds": outcome.result["completed_rounds"],
        "agent_calls": outcome.result["agent_calls"],
        "case_hash": case["case_hash"],
    }
    if outcome.result["status"] != "completed":
        write_json(directory / "metrics.json", result)
        raise RuntimeError(f"Match failed at {directory}: {outcome.result['error']}")
    if teacher:
        rows = teacher.decisions
        episode = {
            "schema": 1,
            "kind": "rl01.CP_demonstration",
            "case_id": case["case_id"],
            "split": case["dataset_split"],
            "parent_world": case["parent_world"],
            "case_hash": case["case_hash"],
            "context": teacher.public_context,
            "decisions": rows,
            "outcome": {
                "winner": outcome.result["winner"],
                "reason": outcome.result["reason"],
            },
        }
        dump_episode(directory / "demonstration.json.gz", episode)
        result.update(
            labels=sum(d["supervised"] for d in rows),
            forced=sum(not d["supervised"] for d in rows),
            actions_by_kind=dict(
                Counter(d["actions"][d["chosen"]]["kind"] for d in rows)
            ),
            demonstration_sha256=file_sha256(directory / "demonstration.json.gz"),
        )
    else:
        turn_side = 0
        kinds = Counter()
        for step in outcome.replay["steps"]:
            if turn_side == side:
                kinds[step["action"]["kind"]] += 1
            if step["action"]["kind"] == "end_turn":
                turn_side = 1 - turn_side
        result["actions_by_kind"] = dict(kinds)
    write_json(directory / "metrics.json", result)
    return result


def task_for(
    config,
    manifest,
    split,
    index,
    directory,
    contender="teacher",
    opponent="greedy",
    device="cpu",
):
    return (
        config,
        manifest,
        make_case(config, manifest, split, index),
        str(directory),
        contender,
        opponent,
        device,
    )


def worker_setup():
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
