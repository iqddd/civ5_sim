"""Separate one-decision D0 protocol; original data/results remain immutable."""

import hashlib
from copy import deepcopy
from pathlib import Path

import numpy as np

from civ5rl.core.context import context_from_spec
from civ5rl.core.observation import ObservationEncoder
from civ5rl.protocols.diagnostic import (
    _SHOT_DAMAGE_ROLLS,
    DecisionEnvContext,
    _minimum_shot_damage,
)
from civ5rl.provenance import bound_cases, case_manifest
from civ5rl.provenance.common import ROOT, read_json, write_json
from civ5sim import Action
from civ5sim.engine import digest


def is_one(config):
    return config.get("diagnostic", {}).get("mode") == "D0-one-decision"


def prepare_one(directory, config):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source_path = ROOT / config["diagnostic"]["source_dataset"]
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    path = directory / "manifest.json"
    if path.exists():
        dataset = read_json(path)
        if (
            dataset["hash"] != digest({k: v for k, v in dataset.items() if k != "hash"})
            or dataset["source_sha256"] != source_hash
            or dataset["preparation"] != config["diagnostic"]
        ):
            raise ValueError("One-decision dataset identity mismatch")
        for split in ("train", "dev"):
            bound_cases(dataset["D0"][split])
        return dataset
    source = read_json(source_path)
    if source["hash"] != digest({k: v for k, v in source.items() if k != "hash"}):
        raise ValueError("Original diagnostic dataset hash mismatch")
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
    proof = []
    for split in ("train", "dev"):
        original = source["D0"][split]["cases"]
        current = bound_cases(dataset["D0"][split])
        for old, new in zip(original, current):
            assert {
                k: v
                for k, v in new.items()
                if k
                not in ("provenance", "diagnostic_protocol", "learner_decision_limit")
            } == old
            with DecisionEnvContext(old) as a, DecisionEnvContext(new) as b:
                x, y = a.request(), b.request()
                if x != y:
                    raise ValueError(
                        "One-decision protocol changed the first player input"
                    )
                for variant in ("C",):
                    enc = []
                    for spec, state in ((old, x), (new, y)):
                        encoder = ObservationEncoder(
                            context_from_spec({**spec, "max_rounds": 3}),
                            variant,
                        )
                        enc.append(
                            encoder.encode(state["observation"], state["actions"])
                        )
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
                        u["hp"]
                        for u in y["observation"]["units"]
                        if u["tile"] == shot.target
                    )
                    preview = b.battle.preview_attack(shot)
                    minimum = _minimum_shot_damage(
                        preview, "Native random roll range changed"
                    )
                    if minimum < hp:
                        raise ValueError(
                            "Initial ranged attack is not guaranteed lethal"
                        )
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
                    raise ValueError(
                        "Delayed shot control did not end at the first wait"
                    )
            proof.append(
                {
                    "split": split,
                    "layout_id": new["layout_id"],
                    "side": new["learner_side"],
                    "raw_encoded_actions_equal": True,
                    "immediate_win": True,
                    "first_wait_draw": True,
                    "shots": minima,
                }
            )
    if {s["physical_hash"] for s in dataset["D0"]["train"]["cases"]} & {
        s["physical_hash"] for s in dataset["D0"]["dev"]["cases"]
    }:
        raise ValueError("One-decision train/dev overlap")
    dataset["hash"] = digest(dataset)
    write_json(
        directory / "construction-verification.json",
        {"source_sha256": source_hash, "cases": proof, "verified": len(proof)},
    )
    write_json(path, dataset)
    return dataset


def load_initial_weights(policy, config):
    import torch

    path = ROOT / config["diagnostic"]["initial_checkpoint"]
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["variant"] != "C" or saved["state"]["update"] != 0:
        raise ValueError("One-decision initialization must be the original untrained C")
    policy.load_state_dict(saved["model"])
    if not all(
        torch.equal(v.detach().cpu(), saved["model"][k])
        for k, v in policy.state_dict().items()
    ):
        raise ValueError("Initial model parameters differ")
    tensors = hashlib.sha256()
    for k, v in sorted(saved["model"].items()):
        tensors.update(k.encode())
        tensors.update(v.contiguous().numpy().tobytes())
    return {
        "source_checkpoint": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "model_tensors_sha256": tensors.hexdigest(),
        "all_model_parameters_bitwise_equal": True,
        "source_optimizer_loaded": False,
        "source_training_rng_loaded": False,
    }


def gate_one(summary):
    if summary.get("games") != 128 or summary.get("wins") is None:
        raise ValueError("One-decision gate requires a complete 128-case evaluation")
    return {"passed": summary["wins"] >= 122, "minimum_wins": 122, "games": 128}


def next_state_one(state, passed):
    value = deepcopy(state)
    value["consecutive"] = value.get("consecutive", 0) + 1 if passed else 0
    if value["consecutive"] >= 2:
        value.update(
            status="passed", reason="D0_one_decision_two_successive_evaluations"
        )
    elif value["stage_updates"] >= 8:
        value.update(status="failed", reason="D0_one_decision_update_limit")
    return value


def report_one(directory, seed=17):
    """Compatibility wrapper for the reporting tool."""
    from tools.archive.report_rl_one import report_one as render_report

    return render_report(directory, seed=seed)
