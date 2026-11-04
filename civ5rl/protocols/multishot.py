"""Frozen multishot corpus construction and native scenario certificates."""

import hashlib
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np

from civ5rl.core.env import DecisionEnv
from civ5rl.protocols.diagnostic import DiagnosticFactory, physical_hash
from civ5rl.protocols.diagnostic_metrics import decision_metrics
from civ5rl.protocols.multishot_metrics import summarize
from civ5rl.provenance import bound_cases, case_manifest, validate_cases
from civ5rl.provenance.common import ROOT, read_json, seed_for, write_json
from civ5sim import Action, _native
from civ5sim.engine import digest

STAGE = "D0-M"


class InvalidDataset(ValueError):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def specification(config, *, full_config=False):
    item = config["diagnostic"]
    path = ROOT / item["specification"]
    if sha(path) != item["specification_sha256"]:
        raise InvalidDataset("Multishot specification hash mismatch")
    spec = read_json(path)
    for field in ("base_config", "source_dataset"):
        ref = spec[field]
        if sha(ROOT / ref["path"]) != ref["sha256"]:
            raise InvalidDataset(field + " SHA256 mismatch")
    if (
        sha(ROOT / spec["initialization"]["checkpoint"])
        != spec["initialization"]["sha256"]
    ):
        raise InvalidDataset("Initialization checkpoint SHA256 mismatch")
    if full_config:
        for section in (
            "model",
            "observation",
            "reward",
            "collection",
            "ppo",
            "optimizer",
            "numerics",
            "performance",
            "telemetry",
        ):
            if config[section] != spec[section]:
                raise InvalidDataset(
                    "Runtime config differs from frozen spec: " + section
                )
    return spec


def control_episode(spec, passive=False):
    ep = {
        "spec": spec,
        "observations": [],
        "actions": [],
        "chosen": [],
        "old_log_probs": [],
        "decision_metrics": [],
    }
    with DecisionEnv(spec) as env:
        while True:
            state = env.request()
            if state["done"]:
                ep.update(
                    result=state["result"], joint_commands=state["joint_commands"]
                )
                break
            acts = state["actions"]
            shot = next((a for a in acts if a.kind == "ranged"), None)
            action = Action("end_turn") if passive or shot is None else shot
            chosen = acts.index(action)
            probabilities = np.zeros(len(acts), dtype=np.float32)
            probabilities[chosen] = 1
            ep["observations"].append(state["observation"])
            ep["actions"].append(acts)
            ep["chosen"].append(chosen)
            ep["old_log_probs"].append(
                np.where(probabilities > 0, 0, -np.inf).astype(np.float32)
            )
            ep["decision_metrics"].append(decision_metrics(acts, chosen, probabilities))
            env.prepare(action)
            env.applied(action, env.battle.step(action))
    check_control(ep, passive)
    return ep


def check_control(ep, passive):
    r, s = ep["result"], ep["spec"]
    if (
        r["reward"] != (0 if passive else 1)
        or r["reason"] == "diagnostic_decision_limit"
    ):
        raise InvalidDataset("Multishot control outcome differs")
    if not passive and sum(m["ranged_chosen"] for m in ep["decision_metrics"]) < 2:
        raise InvalidDataset("Control won with fewer than two shots")
    events = r["diagnostic_events"]
    own = [
        i
        for i, u in enumerate(s["scenario"]["units"])
        if u["side"] == s["learner_side"]
    ]
    if any(
        event["side"] != s["learner_side"] and event["action"]["kind"] != "end_turn"
        for event in events
    ):
        raise InvalidDataset("Multishot opponent used a non-end-turn action")
    if any(
        event["hp_after"][i] < event["hp_before"][i] for event in events for i in own
    ):
        raise InvalidDataset("Multishot opponent damaged the learner")


def prepare(directory, config):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    spec = specification(config)
    source = read_json(ROOT / spec["source_dataset"]["path"])
    if source["hash"] != digest({k: v for k, v in source.items() if k != "hash"}):
        raise InvalidDataset("Original corpus hash mismatch")
    dataset = {
        "schema": 1,
        "kind": "rl01.D0-multishot",
        "preparation": config["diagnostic"],
        "source_sha256": spec["source_dataset"]["sha256"],
        STAGE: {},
    }
    for split in ("train", "dev"):
        old_manifest = source["D0"][split]
        validate_cases(bound_cases(old_manifest), old_manifest)
        cases = deepcopy(old_manifest["cases"])
        if len(cases) != (256 if split == "train" else 128):
            raise InvalidDataset("Multishot source case count differs")
        for case in cases:
            case.update(
                max_rounds=4, diagnostic_protocol="D0-multishot", diagnostic_stage=STAGE
            )
            case.pop("learner_decision_limit", None)
            for unit in case["scenario"]["units"]:
                if unit["side"] != case["learner_side"]:
                    unit["hp"] = 40
            case["physical_hash"] = physical_hash(case["scenario"])
        dataset[STAGE][split] = case_manifest(
            cases,
            source="rl01.diagnostic.D0-multishot",
            split="diagnostic_" + split,
            namespace=STAGE + "_" + split,
        )
    train, dev = (dataset[STAGE][s]["cases"] for s in ("train", "dev"))
    if {s["physical_hash"] for s in train} & {s["physical_hash"] for s in dev}:
        raise InvalidDataset("Multishot train/dev overlap")
    dataset["hash"] = digest(dataset)
    path = directory / "manifest.json"
    if path.exists():
        if read_json(path) != dataset:
            raise InvalidDataset("Multishot dataset identity mismatch")
        proof = read_json(directory / "construction-verification.json")
        if proof["dataset_hash"] != dataset["hash"] or proof["verified_cases"] != 384:
            raise InvalidDataset("Missing or incomplete multishot certificate")
        if proof.get("native_sha256") != sha(ROOT / "build/combat.dll"):
            raise InvalidDataset("Multishot certificate native version mismatch")
        if sha(directory / "controls.json") != proof["controls_sha256"]:
            raise InvalidDataset("Multishot controls certificate hash mismatch")
        return dataset
    lib = _native.library()
    records, controls = [], []
    for split in ("train", "dev"):
        for case in bound_cases(dataset[STAGE][split]):
            bounds = []
            with DecisionEnv(case) as env:
                state = env.request()
                shots = [a for a in state["actions"] if a.kind == "ranged"]
                if not shots:
                    raise InvalidDataset("No initial ranged attack")
                for action in shots:
                    hp = next(
                        u["hp"]
                        for u in state["observation"]["units"]
                        if u["tile"] == action.target
                    )
                    preview = env.battle.preview_attack(action)
                    values = [
                        lib.sim_damage(
                            preview.attacker_strength, preview.defender_strength, r
                        )
                        for r in range(1200)
                    ]
                    if (
                        hp != 40
                        or min(values) < 0
                        or max(values) >= hp
                        or lib.sim_damage(
                            preview.attacker_strength, preview.defender_strength, 1200
                        )
                        != -1
                    ):
                        raise InvalidDataset(
                            "Initial shot non-lethality certificate failed"
                        )
                    bounds.append(
                        {
                            "target_hp": hp,
                            "min": min(values),
                            "max": max(values),
                            "rolls": 1200,
                            "attacker_strength": preview.attacker_strength,
                            "defender_strength": preview.defender_strength,
                        }
                    )
            for trial in range(-1, 16):
                probe = deepcopy(case)
                if trial >= 0:
                    probe["scenario"]["seed"] = seed_for(
                        20261114,
                        case["layout_id"],
                        trial,
                        namespace="multishot_spec_probe_" + split,
                    )
                episode = control_episode(probe)
                _metrics, rows = summarize([episode])
                controls.append(
                    {
                        "split": split,
                        "episode_id": case["episode_id"],
                        "trial": trial,
                        "seed": probe["scenario"]["seed"],
                        "outcome": rows[0],
                        "hp_events": episode["result"]["diagnostic_events"],
                    }
                )
            control_episode(case, passive=True)
            records.append(
                {
                    "split": split,
                    "layout_id": case["layout_id"],
                    "side": case["learner_side"],
                    "initial_shots": bounds,
                    "reference_wins": 17,
                    "passive_draw": True,
                }
            )
    write_json(directory / "controls.json", controls)
    write_json(
        directory / "construction-verification.json",
        {
            "dataset_hash": dataset["hash"],
            "native_sha256": sha(ROOT / "build/combat.dll"),
            "verified_cases": len(records),
            "cases": records,
            "controls_sha256": sha(directory / "controls.json"),
            "seconds": perf_counter() - start,
            "control_games": len(controls),
            "passive_games": 384,
        },
    )
    write_json(path, dataset)  # Commit only after all certificates passed.
    return dataset


class MultishotFactory(DiagnosticFactory):
    def __init__(self, dataset):
        super().__init__({"D0": dataset[STAGE]}, 17)

    def make(self, episode_id, stage):
        if stage != STAGE:
            raise ValueError("Unexpected multishot stage")
        # Preserve the original D0 sampling/RNG derivation, using only new cases.
        return super().make(episode_id, "D0")


def factory(dataset):
    return MultishotFactory(dataset)
