"""The scale-up contract and public training coverage, without historical audits."""

from collections import Counter

from civ5rl.protocols.imitation_own_states import own_state_physical_hash
from civ5rl.protocols.ppo_after_imitation import TrainingFactory
from civ5rl.config import load_config as resolve_config
from civ5rl.provenance.common import ROOT, read_json
from civ5sim.engine import digest

KIND = "rl01.ppo_scale_up"


def load_config(path):
    config = resolve_config(path)
    validate_profile(config)
    return config


def validate_profile(config):
    reference = resolve_config(ROOT / "configs/rl-ppo-controlled-updates.json")
    for key in ("observation", "environment", "model", "reward", "collection", "ppo",
                "optimizer", "numerics", "performance", "teacher_search", "seed", "workers"):
        if config.get(key) != reference[key]:
            raise ValueError(f"Unsupported scale-up configuration: {key}")
    profiles = {
        "scale_up": "configs/rl-ppo-scale-up.json",
        "continuation": "configs/rl-ppo-scale-up-continuation.json",
    }
    profile = config.get("profile", "scale_up")
    if profile not in profiles:
        raise ValueError("Unsupported scale-up profile")
    expected = read_json(ROOT / profiles[profile])
    for key, value in expected["experiment"].items():
        if config["experiment"].get(key) != value:
            raise ValueError(f"Unsupported scale-up configuration: experiment.{key}")
    if config["kind"] != KIND:
        raise ValueError("Unsupported scale-up kind")
    if config["output"] != expected["output"]:
        raise ValueError("Unsupported scale-up output")


class ScaleUpFactory(TrainingFactory):
    def __init__(self, config, base):
        super().__init__(config, base)
        self.physical = set()
        self.duplicates = 0
        self.coverage = Counter()
        self.world_counts = Counter()
        self.crop_counts = Counter()

    def make(self, index, unused=None):
        spec = super().make(index, unused)
        h = own_state_physical_hash(spec["scenario"])
        self.duplicates += h in self.physical
        self.physical.add(h)
        self.coverage[f"{spec['learner_side']}/{spec['placement']}"] += 1
        self.world_counts[str(spec["parent_world"])] += 1
        self.crop_counts[digest(spec["scenario"]["map"])] += 1
        return spec

    def summary(self):
        return {"episodes": sum(self.coverage.values()), "physical_unique": len(self.physical),
                "physical_repeats": self.duplicates, "conditions": dict(self.coverage),
                "worlds": dict(self.world_counts), "crops": dict(self.crop_counts),
                "historical_physical_overlap": "not_measured"}
