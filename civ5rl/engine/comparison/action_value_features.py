"""Read frozen public-history features for the action-value experiment."""

from __future__ import annotations

import torch

from civ5rl.core.schema import ACTION_INPUT_DIM
from civ5rl.engine.comparison.action_value import public_action_vector
from civ5rl.engine.comparison.frozen_features import FrozenFeatureCache, feature_positions
from civ5rl.engine.comparison.policy import ATTENTION_BACKEND, configure_heads
from civ5rl.engine.comparison.reference import target_logits


class FrozenActionFeatures:
    def __init__(self, histories, dataset, checkpoint):
        self.histories, self.dataset, self.checkpoint = histories, dataset, checkpoint
        self.statistics = {
            "cache_hits": 0,
            "cache_builds": 0,
            "bytes": 0,
            "policy_loads": 0,
            "parity_checks": 0,
        }
        self.cache = FrozenFeatureCache(histories, "cuda", ATTENTION_BACKEND, self.statistics)
        self.directory = self.cache.directory
        self.records = self.cache.records
        self.locations = self.cache.locations
        self.policy = None

    def prepare(self):
        self.cache.prepare(
            self.checkpoint,
            feature_positions(self.dataset),
            self._get_policy,
            verify_record=self._check_parity,
        )
        return self

    def _get_policy(self):
        if self.policy is None:
            self.policy = configure_heads(self.checkpoint).requires_grad_(False)
            self.statistics["policy_loads"] += 1
        return self.policy

    def _check_parity(self, policy, history, index, record):
        if self.statistics["parity_checks"]:
            return
        first_action_count = record["offsets"][1]
        vectors = torch.stack(
            [public_action_vector(record, 0, action) for action in range(first_action_count)]
        )
        vectors = vectors.to(self.cache.device)
        expected = target_logits(policy, history, index)
        base, preview = vectors[:, :ACTION_INPUT_DIM], vectors[:, ACTION_INPUT_DIM:]
        actual = policy.action[2](
            policy.action[1](policy.action[0](base) + policy.attack_preview(preview))
        ).flatten()
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
        self.statistics["parity_checks"] += 1
