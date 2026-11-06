"""Shared primitives for CP imitation experiments."""

from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from civ5rl.engine.pool import pool
from civ5rl.protocols.imitation import load_episode, make_case
from civ5rl.provenance.common import file_sha256, read_json
from civ5rl.provenance.rng import capture_rng, restore_rng as _restore_rng


def adam(policy, config, device):
    import torch

    values = dict(config["optimizer"])
    values["betas"] = tuple(values["betas"])
    return torch.optim.Adam(
        [p for p in policy.parameters() if p.requires_grad],
        **values,
        fused=str(device).startswith("cuda"),
    )


def load_policy(config, path, device):
    import torch

    from civ5rl.core.model import Policy

    saved = torch.load(path, weights_only=False, map_location="cpu")
    policy = Policy("C").to(device)
    policy.load_state_dict(saved["model"])
    policy.spatial_cell_padding = config['performance'].get('spatial_cell_padding',0)
    policy.enable_amp(str(device).startswith("cuda"), spatial_batch_size=32)
    if config["performance"].get("inverse_routing", False):
        policy.enable_inverse_routing()
    if config["performance"].get("optimized_routing", False):
        policy.enable_inverse_routing()
        policy.enable_action_routing()
        policy.enable_maximum_routing()
    if str(device).startswith("cuda") and config["performance"]["compile"]:
        policy.enable_compile(
            spatial_mode=config["performance"].get("spatial_compile_mode")
        )
    for parameter in policy.value.parameters():
        parameter.requires_grad_(False)
    if str(device).startswith("cuda"):
        if config["performance"].get("coalesced_spatial", False):
            policy.enable_coalesced_spatial()
        policy.coalesced_accumulation = config["performance"].get(
            "coalesced_accumulation", False
        )
        if config["performance"].get("recurrent_cuda_graphs", False):
            policy.enable_recurrent_graphs(
                max_shapes=config["performance"].get("recurrent_graph_max_shapes", 64)
            )
    return policy


def demonstrations(config, manifest, out, split):
    episodes = []
    files = {}
    for path in sorted((out / "data" / split).glob("match-*/demonstration.json.gz")):
        case = read_json(path.parent / "case.json")
        metrics = read_json(path.parent / "metrics.json")
        if case != make_case(config, manifest, split, case["case_id"]):
            raise ValueError("Untrusted demonstration case")
        if (
            metrics["status"] != "completed"
            or file_sha256(path) != metrics["demonstration_sha256"]
        ):
            raise ValueError("Unfinished/corrupted demonstration")
        episode = load_episode(path)
        if (
            episode["parent_world"] not in manifest["worlds"][split]
            or episode["split"] != split
        ):
            raise ValueError("Demonstration world split leakage")
        episodes.append(episode)
        files[path.relative_to(out).as_posix()] = metrics["demonstration_sha256"]
    return episodes, files


class CompilationClock:
    """Track compilation intervals and unregister callbacks on every exit."""

    def __init__(self):
        self.intervals = []
        self.active = None
        self.monitor = None

    def __enter__(self):
        from torch._dynamo.callback import callback_handler

        self.handler = callback_handler
        self.handler.register_start_callback(self.start)
        self.handler.register_end_callback(self.end)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.handler.remove_start_callback(self.start)
        finally:
            self.handler.remove_end_callback(self.end)
        if self.active is not None:
            self.end(None)

    def start(self, args):
        self.active = perf_counter()
        if self.monitor:
            self.previous_phase = dict(self.monitor.state)
            self.monitor.mark("compilation")

    def end(self, args):
        self.intervals.append((self.active, perf_counter()))
        self.active = None
        if self.monitor:
            self.monitor.mark(**self.previous_phase)

    def seconds(self, start, stop):
        return sum(max(0, min(stop, b) - max(start, a)) for a, b in self.intervals)


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
        epoch >= settings["plateau_min_epoch"]
        and result["plateau_count"] >= settings["plateau_patience"]
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


def collect_summary(directory):
    rows = [read_json(p) for p in sorted(Path(directory).glob("*/metrics.json"))]
    if len(rows) != 64 or any(r["status"] != "completed" for r in rows):
        raise ValueError(f"Incomplete battle evaluation: {directory}")
    outcomes = Counter(
        "D" if r["winner"] is None else "W" if r["winner"] == r["side"] else "L"
        for r in rows
    )
    return {
        "matches": len(rows),
        "W": outcomes["W"],
        "D": outcomes["D"],
        "L": outcomes["L"],
        "win_rate": outcomes["W"] / len(rows),
        "actions": dict(sum((Counter(r["actions_by_kind"]) for r in rows), Counter())),
        "draw_reasons": dict(Counter(r["reason"] for r in rows if r["winner"] is None)),
    }
