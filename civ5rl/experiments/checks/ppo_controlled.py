"""Focused acceptance and reporting for the controlled PPO experiment."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from civ5rl.engine.ppo.adam import optimizer_for
from civ5rl.engine.ppo.backend import BatchedBackend
from civ5rl.engine.runtime.transactions import AdamSnapshot, require_finite_state
from civ5rl.experiments.checks.ppo_after_imitation import public_episodes
from civ5rl.experiments.imitation.helpers import assert_equal
from civ5rl.experiments.ppo.policy import create_policy
from civ5rl.provenance.common import configure, write_json


def preflight(config: dict[str, Any], plan: dict[str, Any], out: Path) -> None:
    # This checks live optimized CUDA graphs without another collection campaign.
    configure(17, "cuda")
    policy = create_policy(config, "cuda")
    optimizer = optimizer_for(policy, config, "cuda")
    episodes = public_episodes(policy, "cuda")
    backend = BatchedBackend(policy, episodes, config, "cuda")
    backend.precheck()
    spans = [(index, 1, len(episode["chosen"])) for index, episode in enumerate(episodes)]
    count = sum(stop - start for _, start, stop in spans)
    checks = []
    for adam_initialized in (False, True):
        print(f"Controlled CUDA rollback: initialized Adam={adam_initialized}", flush=True)
        optimizer.zero_grad(set_to_none=True)
        backend.backward(spans, count, 0)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5, error_if_nonfinite=True)
        snapshot = AdamSnapshot(policy, optimizer)
        expected_model = {name: value.detach().clone() for name, value in policy.state_dict().items()}
        expected_adam = deepcopy(optimizer.state_dict())
        baseline_kl = backend.kl()
        optimizer.step()
        require_finite_state(policy, optimizer)
        attempted_kl = backend.kl()
        snapshot.restore()
        restored_kl = backend.kl()
        assert_equal(policy.state_dict(), expected_model)
        assert_equal(optimizer.state_dict(), expected_adam)
        if baseline_kl is None:
            assert restored_kl is None
        else:
            torch.testing.assert_close(restored_kl, baseline_kl, rtol=1e-5, atol=1e-7)
        checks.append(
            {
                "initialized_adam": adam_initialized,
                "before_kl": baseline_kl,
                "attempted_kl": attempted_kl,
                "restored_kl": restored_kl,
                "bitwise_model_adam_equal": True,
            }
        )
        # Keep one real accepted initialization only in this disposable fixture.
        optimizer.step()
    torch.cuda.synchronize()
    write_json(
        out / "preflight.json",
        {
            "passed": True,
            "CUDA_BF16_rollback": checks,
            "research_updates": 0,
            "CP_collection_code_unchanged_in_this_change": True,
            "full_suite_and_resume_tests": False,
        },
    )
    del policy, optimizer, backend, episodes
    torch.cuda.empty_cache()
