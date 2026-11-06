"""The scale-up contract and public training coverage, without historical audits."""

from civ5rl.protocols.common import (
    CoverageTracker,
    own_state_physical_hash,
    require_config,
)
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
    keys = (
        "observation",
        "environment",
        "model",
        "reward",
        "collection",
        "ppo",
        "optimizer",
        "numerics",
        "performance",
        "teacher_search",
        "seed",
        "workers",
    )
    require_config(
        config,
        {key: reference[key] for key in keys},
        prefix="Unsupported scale-up configuration: ",
    )
    profiles = {
        "scale_up": "configs/rl-ppo-scale-up.json",
        "continuation": "configs/rl-ppo-scale-up-continuation.json",
    }
    profile = config.get("profile", "scale_up")
    if profile not in profiles:
        raise ValueError("Unsupported scale-up profile")
    expected = read_json(ROOT / profiles[profile])
    require_config(
        config["experiment"],
        expected["experiment"],
        prefix="Unsupported scale-up configuration: experiment.",
    )
    if config["kind"] != KIND:
        raise ValueError("Unsupported scale-up kind")
    if config["output"] != expected["output"]:
        raise ValueError("Unsupported scale-up output")


class ScaleUpFactory(TrainingFactory):
    def __init__(self, config, base):
        super().__init__(config, base)
        self._coverage_tracker = CoverageTracker()
        self.physical = self._coverage_tracker.physical
        self.coverage = self._coverage_tracker.conditions
        self.world_counts = self._coverage_tracker.worlds
        self.crop_counts = self._coverage_tracker.crops

    @property
    def duplicates(self):
        return self._coverage_tracker.duplicates

    def make(self, index, unused=None):
        spec = super().make(index, unused)
        h = own_state_physical_hash(spec["scenario"])
        self._coverage_tracker.record(
            h,
            f"{spec['learner_side']}/{spec['placement']}",
            str(spec["parent_world"]),
            digest(spec["scenario"]["map"]),
        )
        return spec

    def summary(self):
        return self._coverage_tracker.summary()
