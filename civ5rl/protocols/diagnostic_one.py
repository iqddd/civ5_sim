"""Separate one-decision D0 protocol; original data/results remain immutable."""

from copy import deepcopy
from pathlib import Path

import numpy as np

from civ5rl.core.context import context_from_spec
from civ5rl.core.observation import ObservationEncoder
from civ5rl.protocols.common import advance_stage_state
from civ5rl.protocols.diagnostic import (
    _SHOT_DAMAGE_ROLLS,
    DecisionEnvContext,
    _minimum_shot_damage,
)
from civ5rl.provenance import bound_cases, case_manifest
from civ5rl.provenance.common import (
    ROOT,
    file_sha256,
    load_verified_checkpoint,
    read_json,
    verify_self_digest,
    write_json,
)
from civ5sim import Action
from civ5sim.engine import digest


def is_one(config):
    return config.get("diagnostic", {}).get("mode") == "D0-one-decision"


def prepare_one(directory, config):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source_path = ROOT / config["diagnostic"]["source_dataset"]
    source_hash = file_sha256(source_path)
    path = directory / "manifest.json"
    source = read_json(source_path)
    verify_self_digest(source, error_msg="Original diagnostic dataset hash mismatch")
    dataset = _expected_one_dataset(source, config, source_hash)
    if path.exists():
        cached = read_json(path)
        verify_self_digest(cached, error_msg="One-decision dataset identity mismatch")
        if cached != dataset:
            raise ValueError("One-decision dataset identity mismatch")
        for split in ("train", "dev"):
            bound_cases(cached["D0"][split])
        return cached
    proof = _certify_one_cases(source, dataset)
    write_json(
        directory / "construction-verification.json",
        {"source_sha256": source_hash, "cases": proof, "verified": len(proof)},
    )
    write_json(path, dataset)
    return dataset


def _expected_one_dataset(source, config, source_hash):
    dataset = {
        "schema": 1,
        "kind": "rl01.D0-one-decision",
        "source_sha256": source_hash,
        "preparation": config["diagnostic"],
        "D0": {},
    }
    for split in ("train", "dev"):
        cases = deepcopy(source["D0"][split]["cases"])
        for spec in cases:
            spec.update(diagnostic_protocol="D0-one-decision", learner_decision_limit=1)
        dataset["D0"][split] = case_manifest(
            cases,
            source="rl01.diagnostic.D0-one-decision",
            split="diagnostic_" + split,
            namespace="D0_one_decision_" + split,
        )
    dataset["hash"] = digest(dataset)
    return dataset


def _certify_one_cases(source, dataset):
    proof = []
    for split in ("train", "dev"):
        original = source["D0"][split]["cases"]
        current = bound_cases(dataset["D0"][split])
        for old, new in zip(original, current):
            proof.append({"split": split, **_certify_one_decision_case(old, new)})
    if {s["physical_hash"] for s in dataset["D0"]["train"]["cases"]} & {
        s["physical_hash"] for s in dataset["D0"]["dev"]["cases"]
    }:
        raise ValueError("One-decision train/dev overlap")
    return proof


def _certify_one_decision_case(old, new):
    assert {
        k: v
        for k, v in new.items()
        if k not in ("provenance", "diagnostic_protocol", "learner_decision_limit")
    } == old
    with DecisionEnvContext(old) as a, DecisionEnvContext(new) as b:
        x, y = a.request(), b.request()
        if x != y:
            raise ValueError("One-decision protocol changed the first player input")
        for variant in ("C",):
            enc = []
            for spec, state in ((old, x), (new, y)):
                encoder = ObservationEncoder(
                    context_from_spec({**spec, "max_rounds": 3}),
                    variant,
                )
                enc.append(encoder.encode(state["observation"], state["actions"]))
            for key in vars(enc[0]):
                np.testing.assert_array_equal(
                    getattr(enc[0], key), getattr(enc[1], key)
                )
        shots = [act for act in y["actions"] if act.kind == "ranged"]
        if not shots:
            raise ValueError("One-decision D0 has no initial ranged attack")
        minima = []
        for shot in shots:
            hp = next(
                u["hp"] for u in y["observation"]["units"] if u["tile"] == shot.target
            )
            preview = b.battle.preview_attack(shot)
            minimum = _minimum_shot_damage(preview, "Native random roll range changed")
            if minimum < hp:
                raise ValueError("Initial ranged attack is not guaranteed lethal")
            minima.append(
                {
                    "minimum_damage": minimum,
                    "target_hp": hp,
                    "rolls": _SHOT_DAMAGE_ROLLS,
                }
            )
        b.prepare(shots[0])
        b.applied(shots[0], b.battle.step(shots[0]))
        win = b.request()
        if not win["done"] or win["result"]["reward"] != 1:
            raise ValueError("Immediate shot control did not win")
    with DecisionEnvContext(new) as env:
        env.request()
        action = Action("end_turn")
        env.prepare(action)
        env.applied(action, env.battle.step(action))
        commands = env.commands
        done = env.request()
        if (
            not done["done"]
            or done["result"]["reward"] != 0
            or done["result"]["reason"] != "diagnostic_decision_limit"
            or env.commands != commands
        ):
            raise ValueError("Delayed shot control did not end at the first wait")
    return {
        "layout_id": new["layout_id"],
        "side": new["learner_side"],
        "raw_encoded_actions_equal": True,
        "immediate_win": True,
        "first_wait_draw": True,
        "shots": minima,
    }


def load_initial_weights(policy, config):
    path = ROOT / config["diagnostic"]["initial_checkpoint"]
    _, metadata = load_verified_checkpoint(
        policy,
        path,
        expected_update=0,
        source_error="One-decision initialization must be the original untrained C",
        equality_error="Initial model parameters differ",
    )
    return metadata


def gate_one(summary):
    if summary.get("games") != 128 or summary.get("wins") is None:
        raise ValueError("One-decision gate requires a complete 128-case evaluation")
    return {"passed": summary["wins"] >= 122, "minimum_wins": 122, "games": 128}


def next_state_one(state, passed):
    return advance_stage_state(
        state,
        passed,
        update_field="stage_updates",
        update_limit=8,
        success={
            "status": "passed",
            "reason": "D0_one_decision_two_successive_evaluations",
        },
        failure={"status": "failed", "reason": "D0_one_decision_update_limit"},
    )


def report_one(directory, seed=17):
    """Compatibility wrapper for the reporting tool."""
    from tools.archive.report_rl_one import report_one as render_report

    return render_report(directory, seed=seed)
