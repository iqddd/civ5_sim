"""Frozen P(win) PPO protocol, deterministic training and open confirmation cases."""

import hashlib
import json
import random
from copy import deepcopy

from civ5rl.protocols.common import require_config, self_digest
from civ5rl.protocols.corpus import ScenarioFactory, validate_corpus
from civ5rl.protocols.imitation_own_states import (
    own_state_physical_hash as physical_hash,
)
from civ5rl.provenance.common import ROOT, read_json
from civ5rl.provenance.common import file_sha256 as file_hash
from civ5sim.engine import digest

KIND = "rl01.ppo_after_imitation"
CONTROLLED_KIND = "rl01.ppo_controlled_updates"


def controlled(config):
    return config.get("kind") == CONTROLLED_KIND


def validate_profile(config):
    fixed = {
        "seed": 17,
        "workers": 4,
        "collection.num_envs": 128,
        "collection.cpu_workers": 4,
        "collection.envs_per_worker": 32,
        "collection.min_learner_decisions_per_update": 8192,
        "ppo.epochs": 4,
        "ppo.sequence_length": 128,
        "ppo.sequences_per_microbatch": 8,
        "ppo.valid_tokens_per_minibatch": 1024,
        "ppo.clip_ratio": 0.2,
        "ppo.value_coefficient": 0.5,
        "ppo.max_gradient_norm": 0.5,
        "ppo.stop_kl": 0.015,
        "ppo.entropy_coefficient": 0,
        "ppo.kl_population": "free",
        "ppo.kl_measurement": "whole_rollout_after_optimizer_step",
        "optimizer.lr": 1e-5 if controlled(config) else 1e-4,
        "optimizer.betas": [0.9, 0.999],
        "optimizer.eps": 1e-5,
        "optimizer.weight_decay": 0,
        "performance.spatial_batch_size": 32,
        "performance.collection_scheduler": "ready",
        "performance.compile": True,
        "performance.compile_dynamic": True,
        "environment.max_calls_per_turn": 2000,
        "experiment.decisions": 1048576,
        "experiment.active_seconds": 7200,
        "experiment.milestones": [131072, 262144, 524288, 1048576],
        "experiment.bootstrap_samples": 10000,
        "experiment.bootstrap_seed": 20261103,
        "experiment.confirmation_layouts_per_world": 8,
        "experiment.initial_value": "zero_last_layer",
        "experiment.parent_sha256": "035581230481bb5cc5cbd02f6a947549dc2ccb30d3797c7acbad25d5268ba7a1",
    }
    if controlled(config):
        fixed.update(
            {
                "ppo.step_acceptance": "full_categorical_kl_rollback",
                "experiment.parent": "artifacts/rl/ppo-after-imitation/checkpoints/initial.pt",
                "experiment.parent_run": "artifacts/rl/ppo-after-imitation",
                "experiment.parent_sha256": "ad1ec38bd730822c5a1328fddb8437cce641141251340ce45c8b64686bf4413e",
                "experiment.protocol": "docs/rl-ppo-controlled-updates.md",
                "experiment.max_zero_accepted_rollouts": 3,
            }
        )
    elif config["ppo"].get("step_acceptance") is not None:
        raise ValueError("Unsupported configuration: ppo.step_acceptance")
    require_config(config, fixed)
    if "lr_schedule" in config["optimizer"] or "entropy_schedule" in config["ppo"]:
        raise ValueError("This protocol has no schedules")
    if type(config["performance"].get("ppo_cached_replay", False)) is not bool:
        raise ValueError("Invalid configuration: performance.ppo_cached_replay")
    reply_size = config["performance"].get("worker_reply_size", 0)
    if type(reply_size) is not int or reply_size not in (0, 8, 16):
        raise ValueError("Invalid configuration: performance.worker_reply_size")
    parent = read_json(ROOT / "configs/rl-cp-imitation-own-states.json")
    if config["teacher_search"] != parent["teacher_search"]:
        raise ValueError("Unsupported configuration: teacher_search")


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
