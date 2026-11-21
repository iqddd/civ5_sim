"""Shared primitives for CP imitation experiments."""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch

from civ5rl.core.model import Policy
from civ5rl.core.preview import extends_parent, load_parent_model
from civ5rl.engine.runtime.shared import pool as pool
from civ5rl.engine.runtime.telemetry import CompilationClock as CompilationClock
from civ5rl.protocols.imitation import load_episode, make_case
from civ5rl.provenance.common import file_sha256, read_json
from civ5rl.provenance.rng import capture_rng
from civ5rl.provenance.rng import restore_rng as _restore_rng
from civ5rl.reporting.evaluation_audit import collect_summary as collect_summary


def adam(policy, config, device):
    values = dict(config["optimizer"])
    values["betas"] = tuple(values["betas"])
    return torch.optim.Adam(
        [p for p in policy.parameters() if p.requires_grad],
        **values,
        fused=str(device).startswith("cuda"),
    )


def load_policy(config, path, device):
    saved = torch.load(path, weights_only=False, map_location="cpu")
    variant = (
        config["model"]["variant"]
        if extends_parent(config)
        else saved.get("variant") or config.get("model", {}).get("variant", "C")
    )
    policy = Policy(variant)
    load_parent_model(policy, saved, config)
    policy = policy.to(device)
    performance = config["performance"]
    cuda = str(device).startswith("cuda")
    amp = cuda and config.get("numerics", {}).get("amp", False)
    policy.spatial_cell_padding = performance.get("spatial_cell_padding", 0)
    policy.enable_amp(amp, spatial_batch_size=performance.get("spatial_batch_size", 32))
    if performance.get("inverse_routing", False):
        policy.enable_inverse_routing()
    if performance.get("optimized_routing", False):
        policy.enable_inverse_routing()
        policy.enable_action_routing()
        policy.enable_maximum_routing()
    if cuda and performance["compile"]:
        policy.enable_compile(spatial_mode=performance.get("spatial_compile_mode"))
    for parameter in policy.value.parameters():
        parameter.requires_grad_(False)
    if cuda:
        if amp and performance.get("coalesced_spatial", False):
            policy.enable_coalesced_spatial()
        policy.coalesced_accumulation = performance.get("coalesced_accumulation", False)
        if performance.get("recurrent_cuda_graphs", False):
            policy.enable_recurrent_graphs(max_shapes=performance.get("recurrent_graph_max_shapes", 64))
    return policy


def demonstrations(config, manifest, out, split):
    episodes = []
    files = {}
    for path in sorted((out / "data" / split).glob("match-*/demonstration.json.gz")):
        case = read_json(path.parent / "case.json")
        metrics = read_json(path.parent / "metrics.json")
        if case != make_case(config, manifest, split, case["case_id"]):
            raise ValueError("Untrusted demonstration case")
        if metrics["status"] != "completed" or file_sha256(path) != metrics["demonstration_sha256"]:
            raise ValueError("Unfinished/corrupted demonstration")
        episode = load_episode(path)
        if episode["parent_world"] not in manifest["worlds"][split] or episode["split"] != split:
            raise ValueError("Demonstration world split leakage")
        episodes.append(episode)
        files[path.relative_to(out).as_posix()] = metrics["demonstration_sha256"]
    return episodes, files


def update_selection(state, epoch, ce, settings):
    """Best uses actual CE; the significant-improvement threshold only stops training."""
    result = dict(state)
    if ce < result["best_ce"]:
        result.update(best_ce=ce, best_epoch=epoch)
    if result["anchor_ce"] - ce >= settings["plateau_min_delta"]:
        result.update(anchor_ce=ce, plateau_count=0)
    else:
        result["plateau_count"] += 1
    result["plateau"] = (
        epoch >= settings["plateau_min_epoch"] and result["plateau_count"] >= settings["plateau_patience"]
    )
    return result


def restore_rng(saved, shuffle):
    _restore_rng(saved, shuffle)


@contextmanager
def isolated_rng(shuffle):
    state = capture_rng(shuffle)
    try:
        yield
    finally:
        restore_rng(state, shuffle)


def assert_equal(left, right):
    """Exact restored parameters/moments/RNG, including optimizer parameter groups."""
    if isinstance(left, torch.Tensor):
        assert torch.equal(left.detach().cpu(), right.detach().cpu())
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_equal(a, b)
    else:
        assert left == right
