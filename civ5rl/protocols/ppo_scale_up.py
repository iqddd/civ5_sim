"""The scale-up contract and public training coverage, without historical audits."""

from civ5rl.config import load_config as resolve_config
from civ5rl.protocols.common import (
    CoverageTracker,
    own_state_physical_hash,
)
from civ5rl.protocols.ppo_after_imitation import TrainingFactory
from civ5rl.protocols.ppo_validation import validate_bounded_training
from civ5sim.engine import digest

KIND = "rl01.ppo_scale_up"


def load_config(path):
    config = resolve_config(path)
    validate_profile(config)
    return config


def validate_profile(config):
    validate_bounded_training(config, kinds=(KIND,))
    if config["experiment"]["initial_value"] != "preserve_parent":
        raise ValueError("Continuing PPO requires the parent's critic and Adam")


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
