"""Bounded D0/D1 experiment: construction, admission certificates and evaluation."""

import json
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

import numpy as np

from civ5rl.core.context import context_from_spec
from civ5rl.core.env import DecisionEnv
from civ5rl.protocols.common import (
    advance_stage_state,
    bootstrap_ci95,
    flat_map,
    learner_outcome,
    paired_layout_deltas,
    physical_hash,
)
from civ5rl.protocols.diagnostic_metrics import summarize_decisions
from civ5rl.provenance import bound_cases, case_manifest, validate_cases
from civ5rl.provenance.common import seed_for, verify_self_digest, write_json
from civ5sim import Action
from civ5sim import _native as native
from civ5sim.engine import digest
from civ5sim.greedy_agent import GreedyAgent


class AttackOrEndTurn:
    """Diagnostic opponent: uniform legal attack, otherwise end turn."""

    def reset(self, context):
        self.rng = random.Random(context["seed"])

    def act(self, observation, actions):
        attacks = [a for a in actions if a.kind in ("melee", "ranged")]
        return self.rng.choice(attacks) if attacks else Action("end_turn")


def make_spec(scenario, stage, side, index, *, seed=20261108):
    value = deepcopy(scenario)
    value["seed"] = seed_for(seed, index, "battle", namespace="diagnostic_" + stage)
    return {
        "episode_id": index * 2 + side,
        "case_id": index,
        "layout_id": index,
        "scenario": value,
        "armies": [[u["type"] for u in value["units"] if u["side"] == s] for s in (0, 1)],
        "learner_side": side,
        "max_rounds": 3 if stage == "D0" else 10,
        "max_calls_per_turn": 2000,
        "opponent": "end_turn"
        if stage == "D0"
        # The historical alias resolves via civ5rl.__init__._ALIASES; it is hashed.
        else "civ5rl.diagnostic:AttackOrEndTurn",
        "agent_seeds": [seed_for(seed, index, "sampling", s, "diagnostic_" + stage) for s in (0, 1)],
        "parent_world": index,
        "placement": "contact",
        "diagnostics": True,
        "diagnostic_stage": stage,
        "physical_hash": physical_hash(value),
    }


_SHOT_DAMAGE_ROLLS = 1200


def shot_damage_bounds(preview, range_error, *, library=None):
    lib = library if library is not None else native.library()
    if lib.sim_damage(preview.attacker_strength, preview.defender_strength, _SHOT_DAMAGE_ROLLS) != -1:
        raise ValueError(range_error)
    values = [
        lib.sim_damage(preview.attacker_strength, preview.defender_strength, roll)
        for roll in range(_SHOT_DAMAGE_ROLLS)
    ]
    return min(values), max(values)


def d0_certificate(spec):
    with DecisionEnv(spec) as env:
        state = env.request()
        shots = [a for a in state.get("actions", ()) if a.kind == "ranged"]
        if not shots:
            return None
        certificates = []
        for action in shots:
            preview = env.battle.preview_attack(action)
            minimum, _ = shot_damage_bounds(
                preview,
                "Native roll range changed; D0 exhaustive certificate must be revised",
            )
            if minimum < 1:
                return None
            certificates.append(
                {
                    "action": action.__dict__,
                    "minimum_damage": minimum,
                    "rolls_checked": _SHOT_DAMAGE_ROLLS,
                    "attacker_strength": preview.attacker_strength,
                    "defender_strength": preview.defender_strength,
                }
            )
        return certificates


def reference_win(spec):
    with DecisionEnv(spec) as env:
        agent = GreedyAgent()
        agent.reset(context_from_spec(spec))
        try:
            first = True
            while True:
                state = env.request()
                if state["done"]:
                    return state["result"]["reward"] == 1
                if first and not any(a.kind == "ranged" for a in state["actions"]):
                    return False
                first = False
                action = agent.act(state["observation"], state["actions"])
                env.step(action)
        finally:
            agent.close()


def admit_candidate(task):
    stage, index, scenario = task
    certificates = []
    for side in (0, 1):
        if stage == "D0":
            # Swap ownership so the learner is always the archer.
            value = deepcopy(scenario)
            for unit in value["units"]:
                unit["side"] ^= side
            proof = d0_certificate(make_spec(value, stage, side, index))
            if proof is None:
                return None
            certificates.append(proof)
        else:
            for trial in range(16):
                spec = make_spec(scenario, stage, side, index, seed=20261200 + trial)
                if not reference_win(spec):
                    return None
            certificates.append({"reference_wins": 16, "trials": 16})
    return {
        "scenario": scenario,
        "candidate": index,
        "physical_hash": physical_hash(scenario),
        "certificates": certificates,
    }


def candidate(stage, rng):
    tiles = rng.sample(range(36), 2 if stage == "D0" else 4)
    types = ("archer", "warrior") if stage == "D0" else ("warrior", "archer", "warrior", "archer")
    units = [
        {
            "type": kind,
            "side": int(i >= len(types) // 2),
            "tile": tile,
            "hp": 1 if stage == "D0" and i == 1 else 100,
        }
        for i, (kind, tile) in enumerate(zip(types, tiles))
    ]
    return {
        "schema": 1,
        "kind": "civ5sim.scenario",
        "map": flat_map(6),
        "units": units,
        "fog": True,
        "random_damage": True,
        "active_side": 0,
        "seed": 0,
    }


def prepare(directory, config):
    """Admission uses independent RNG and is frozen before any model is trained."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    if path.exists():
        value = json.loads(path.read_text())
        if value["preparation"] != config["diagnostic"]:
            raise ValueError("Diagnostic dataset hash/config mismatch")
        verify_self_digest(value, error_msg="Diagnostic dataset identity mismatch")
        for stage in ("D0", "D1"):
            for split in ("train", "dev"):
                bound_cases(value[stage][split])
            train = {s["physical_hash"] for s in value[stage]["train"]["cases"]}
            dev = {s["physical_hash"] for s in value[stage]["dev"]["cases"]}
            if train & dev:
                raise ValueError("Diagnostic train/dev layout overlap")
        return value
    result = {
        "schema": 1,
        "kind": "rl01.diagnostic_dataset",
        "preparation": config["diagnostic"],
    }
    for stage in ("D0", "D1"):
        rng = random.Random(20261108 + (stage == "D1"))
        accepted = []
        seen = set()
        attempts = 0
        needed = 64 + config["diagnostic"]["train_layouts"]
        with ProcessPoolExecutor(max_workers=4) as executor:
            while len(accepted) < needed and attempts < 10000:
                tasks = []
                for _ in range(min(64, 10000 - attempts)):
                    scenario = candidate(stage, rng)
                    key = physical_hash(scenario)
                    index = attempts
                    attempts += 1
                    if key not in seen:
                        seen.add(key)
                        tasks.append((stage, index, scenario))
                for value in executor.map(admit_candidate, tasks):
                    if value is not None and len(accepted) < needed:
                        accepted.append(value)
                write_json(
                    directory / (stage + "-preparation.json"),
                    {
                        "accepted": len(accepted),
                        "needed": needed,
                        "attempts": attempts,
                        "certificates": accepted,
                    },
                )
                print(
                    f"{stage}: accepted {len(accepted)}/{needed}, candidates {attempts}",
                    flush=True,
                )
        if len(accepted) < needed:
            write_json(
                directory / "preparation-failure.json",
                {
                    "stage": stage,
                    "attempts": attempts,
                    "accepted": len(accepted),
                    "needed": needed,
                },
            )
            raise RuntimeError(f"Insufficient admitted {stage} layouts after {attempts} candidates")
        # All candidates were admitted without seeing any policy outcomes.
        groups = {"dev": accepted[:64], "train": accepted[64:]}
        result[stage] = {}
        for split, layouts in groups.items():
            specs = []
            for i, layout in enumerate(layouts):
                for side in (0, 1):
                    scenario = deepcopy(layout["scenario"])
                    if stage == "D0":
                        for u in scenario["units"]:
                            u["side"] ^= side
                    specs.append(make_spec(scenario, stage, side, i, seed=20261120 + (split == "train")))
            result[stage][split] = case_manifest(
                specs,
                source="rl01.diagnostic." + stage,
                split="diagnostic_" + split,
                namespace=stage + "_" + split,
            )
        a = {s["physical_hash"] for s in result[stage]["train"]["cases"]}
        b = {s["physical_hash"] for s in result[stage]["dev"]["cases"]}
        assert not a & b
    result["hash"] = digest(result)
    write_json(path, result)
    return result


class DiagnosticFactory:
    def __init__(self, dataset, seed):
        self.dataset, self.seed = dataset, seed
        self._issued = {}

    def make(self, episode_id, stage):
        pool = self.dataset[stage]["train"]["cases"]
        rng = random.Random(seed_for(self.seed, episode_id, "case", namespace=stage))
        spec = deepcopy(pool[rng.randrange(len(pool))])
        spec["episode_id"] = episode_id
        spec["scenario"]["seed"] = seed_for(self.seed, episode_id, "battle", namespace=stage)
        spec["agent_seeds"] = [seed_for(self.seed, episode_id, "sampling", s, stage) for s in (0, 1)]
        manifest = case_manifest(
            [spec],
            source=self.dataset[stage]["train"]["source"],
            split="diagnostic_train",
            namespace=stage + "_train",
        )
        value = bound_cases(manifest)[0]
        self._issued[digest(value)] = manifest
        return value

    def validate_specs(self, specs):
        for spec in specs:
            manifest = self._issued.pop(digest(spec), None)
            validate_cases([spec], manifest)


def episode_rows(episodes):
    rows = []
    for ep in episodes:
        first = ep["decision_metrics"][0]
        r, s = ep["result"], ep["spec"]
        outcome = learner_outcome(r["winner"], s["learner_side"])
        rows.append(
            {
                "layout_id": s["layout_id"],
                "side": s["learner_side"],
                "reward": r["reward"],
                "outcome": outcome,
                "win": outcome == "W",
                "first_ranged": first["ranged_chosen"],
                "first_ranged_probability": first["ranged_probability"],
                "rounds": r["rounds"],
                "damage_dealt": r["damage_dealt"][s["learner_side"]],
                "units_lost": r["units_lost"][s["learner_side"]],
            }
        )
    return rows


def outcome_summary(episodes):
    rows = episode_rows(episodes)
    counts = Counter(r["outcome"] for r in rows)
    return {
        "games": len(rows),
        "wins": counts["W"],
        "draws": counts["D"],
        "losses": counts["L"],
        "win_rate": counts["W"] / len(rows),
        "first_winning_attack_rate": sum(r["first_ranged"] for r in rows) / len(rows),
        "first_winning_attack_probability": float(np.mean([r["first_ranged_probability"] for r in rows])),
        "rounds_mean": float(np.mean([r["rounds"] for r in rows])),
        "damage_dealt_mean": float(np.mean([r["damage_dealt"] for r in rows])),
        "units_lost_mean": float(np.mean([r["units_lost"] for r in rows])),
        "actions": summarize_decisions(episodes),
    }


def gate(stage, summary, rows, baseline_rows=None):
    if stage == "D0":
        return {"passed": summary["win_rate"] >= 0.95 and summary["first_winning_attack_rate"] >= 0.90}
    d = paired_layout_deltas(rows, baseline_rows)
    lower, upper = bootstrap_ci95(d)
    delta = float(d.mean())
    return {
        "passed": bool(summary["win_rate"] >= 0.8 and delta >= 0.2 and lower > 0),
        "win_rate_delta": delta,
        "paired_bootstrap_ci95": [float(lower), float(upper)],
        "clusters": 64,
    }


def next_state(state, passed):
    """Pure checkpointed state transition; no policy-dependent hidden decisions."""
    stage = state["stage"]
    success = (
        {"stage": "D1", "stage_updates": 0, "consecutive": 0}
        if stage == "D0"
        else {"status": "passed", "reason": "both_stages_passed"}
    )
    return advance_stage_state(
        state,
        passed,
        update_field="stage_updates",
        update_limit=4 if stage == "D0" else 8,
        success=success,
        failure={"status": "failed", "reason": stage + "_update_limit"},
    )


def may_start_update(state, stopping=False):
    return (
        state["status"] == "running"
        and not stopping
        and state["training_seconds"] < 1800
        and state["pending_evaluation"] is None
    )
