"""Frozen P(win) PPO protocol, deterministic training and open confirmation cases."""

import hashlib
import json
import random
from copy import deepcopy

from civ5rl.protocols.common import self_digest
from civ5rl.protocols.corpus import ScenarioFactory, validate_corpus
from civ5rl.protocols.imitation_own_states import (
    own_state_physical_hash as physical_hash,
)
from civ5rl.protocols.ppo_validation import validate_bounded_training
from civ5rl.provenance.common import ROOT, read_json
from civ5rl.provenance.common import file_sha256 as file_hash
from civ5sim.engine import digest

KIND = "rl01.ppo_after_imitation"
CONTROLLED_KIND = "rl01.ppo_controlled_updates"


def controlled(config):
    return config.get("kind") == CONTROLLED_KIND


def validate_profile(config):
    validate_bounded_training(config, kinds=(KIND, CONTROLLED_KIND))
    ppo, experiment = config["ppo"], config["experiment"]
    if experiment["initial_value"] != "zero_last_layer":
        raise ValueError("PPO after imitation initializes the critic's last layer to zero")
    if "lr_schedule" in config["optimizer"] or "entropy_schedule" in ppo:
        raise ValueError("This protocol requires constant learning rate and entropy coefficient")
    expected_acceptance = "full_categorical_kl_rollback" if controlled(config) else None
    if ppo.get("step_acceptance") != expected_acceptance:
        raise ValueError("Step acceptance differs from the selected PPO protocol")
    if controlled(config) and (
        type(experiment["max_zero_accepted_rollouts"]) is not int
        or experiment["max_zero_accepted_rollouts"] < 1
    ):
        raise ValueError("max_zero_accepted_rollouts must be a positive integer")


def advance_controlled_rollout(state, metrics, number):
    """Collected work is never rolled back with rejected optimization attempts."""
    state["updates"] = number
    state["attempted_steps"] += metrics.get("attempted_optimizer_steps", 0)
    state["rejected_steps"] += metrics.get("rejected_optimizer_steps", 0)
    first = False
    if metrics["optimizer_steps"]:
        state["accepted_updates"] += 1
        state["zero_accepted_streak"] = 0
        first = not state["first_accepted_evaluated"]
    else:
        state["zero_accepted_streak"] += 1
    return first


def stage(contact):
    return {
        "map_size": [10, 10],
        "units_per_side": [2, 2],
        "roster": ["warrior", "archer"],
        "max_rounds": 30,
        "contact_probability": int(contact),
        "opponents": {"cp": 1},
    }


class TrainingFactory:
    def __init__(self, config, base):
        self.config, self.base = config, base
        self.issued = set()
        self.generators = {}
        for world in base["worlds"]["training"]:
            maps = [
                e
                for e in base["maps"]
                if e["parent_seed"] == world and e["split"] == "train"
            ]
            self.generators[world] = ScenarioFactory(
                config, {"schema": 1, "maps": maps}
            )

    def make(self, index, unused=None):
        worlds = self.base["worlds"]["training"]
        namespace = self.config.get("experiment", {}).get("training_namespace")
        cells = list(range(4)) * 32
        balance_key = [17, index // 128, "balance"]
        if namespace:
            balance_key.append(namespace)
        random.Random(int(digest(balance_key), 16)).shuffle(cells)
        cell = cells[index % 128]
        world = worlds[index % len(worlds)]
        spec = self.generators[world].make(
            index,
            stage(cell // 2),
            namespace=namespace or "ppo_after_imitation_train",
            split="train",
            case_index=index // len(worlds),
        )
        spec.update(
            learner_side=cell % 2,
            opponent="cp",
            opponent_search=deepcopy(self.config["teacher_search"]),
            reward_profile="win_probability",
        )
        self.issued.add(digest(spec))
        return spec

    def validate_specs(self, specs):
        for spec in specs:
            if (
                digest(spec) not in self.issued
                or spec["parent_world"] not in self.base["worlds"]["training"]
            ):
                raise ValueError("Untrusted training case")


def prepare_cases(config):
    parent = ROOT / config["experiment"]["parent_run"]
    if controlled(config):
        previous = read_json(parent / "manifest.json")
        if self_digest(previous, key="hash") != previous["hash"]:
            raise ValueError("Previous PPO manifest changed")
        if previous["kind"] != KIND:
            raise ValueError("Unexpected previous PPO manifest")
        recorded = read_json(parent / "passport.json")
        if (
            hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest()
            != recorded["corpus_hash"]
        ):
            raise ValueError("Previous PPO manifest differs from frozen passport")
        validate_corpus({"maps": previous["base"]["maps"]})
        if (
            file_hash(ROOT / config["experiment"]["corpus"])
            != previous["base"]["source_corpus_sha256"]
        ):
            raise ValueError("Corpus manifest changed")
        plan = deepcopy(previous)
        plan.update(
            kind=CONTROLLED_KIND,
            parent_sha256=config["experiment"]["parent_sha256"],
            reused_manifest_sha256=file_hash(parent / "manifest.json"),
            reused_training_streams=True,
        )
        plan.pop("hash")
        plan["hash"] = self_digest(plan, key="hash")
        return plan
    base = read_json(parent / "manifest.json")["base"]
    corpus = read_json(ROOT / config["experiment"]["corpus"])
    if file_hash(ROOT / config["experiment"]["corpus"]) != base["source_corpus_sha256"]:
        raise ValueError("Corpus manifest changed")
    if [len(base["worlds"][k]) for k in ("training", "holdout", "validation")] != [
        57,
        7,
        16,
    ]:
        raise ValueError("Unexpected world split")
    validate_corpus({"maps": base["maps"]})
    development = []
    for original in base["validation_cases"]:
        for side in (0, 1):
            case = {**deepcopy(original), "learner_side": side}
            case["case_hash"] = self_digest(case)
            development.append(case)
    used = {physical_hash(c["scenario"]) for c in development}
    confirmation = []
    for wi, world in enumerate(base["worlds"]["validation"]):
        local = {
            "schema": 1,
            "maps": [
                e
                for e in corpus["maps"]
                if e["parent_seed"] == world
                and e["size"] == 10
                and e["split"] == "validation"
            ],
        }
        generator = ScenarioFactory(config, local)
        for layout in range(8):
            index = wi * 8 + layout
            for attempt in range(100):
                case = generator.make(
                    index,
                    stage(layout < 4),
                    namespace=f"ppo_after_imitation_confirmation_{attempt}",
                    split="validation",
                    case_index=layout,
                )
                h = physical_hash(case["scenario"])
                if h not in used:
                    used.add(h)
                    break
            else:
                raise RuntimeError("Confirmation physical uniqueness exhausted")
            for side in (0, 1):
                row = {
                    **deepcopy(case),
                    "case_id": index,
                    "learner_side": side,
                    "physical_hash": h,
                    "source": KIND,
                    "purpose": "confirmation",
                }
                row["case_hash"] = digest(row)
                confirmation.append(row)
    plan = {
        "schema": 1,
        "kind": KIND,
        "base": base,
        "development": development,
        "confirmation": confirmation,
        "parent_sha256": config["experiment"]["parent_sha256"],
    }
    plan["hash"] = self_digest(plan, key="hash")
    return plan
