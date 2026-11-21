"""Short action-value smoke/profiling and human-readable offline results."""

from __future__ import annotations

import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from civ5rl.engine.comparison.action_value import (
    DEFAULT_WORLD_COUNTS,
    HEAD_WIDTHS,
    make_head,
    make_optimizer,
    outcome_loss,
    public_action_vector,
    split_worlds,
)
from civ5rl.engine.comparison.action_value_features import FrozenActionFeatures
from civ5rl.engine.comparison.heads import configure_heads
from civ5rl.engine.comparison.inputs import Histories
from civ5rl.engine.runtime.telemetry import ResourceMonitor
from civ5rl.engine.runtime.transactions import atomic_torch_save
from civ5rl.experiments.benchmarks.profiling import top_cuda_operators
from civ5rl.experiments.comparison.action_value import SHUFFLE_NAMESPACE, evaluate, train
from civ5rl.provenance.common import ROOT, configure, file_sha256, read_json, write_json


def smoke(cfg, out):

    out = Path(out)
    if any(p.name != "run.lock" and p.name != "derived" for p in out.iterdir()):
        raise ValueError("Use an empty smoke directory")
    head = make_head()
    opt = make_optimizer(head)
    features = torch.randn(256, 2, HEAD_WIDTHS[0], device="cuda")
    counts = torch.full((256, 2), 4.0, device="cuda")
    wins = torch.randint(0, 5, (256, 2), device="cuda").float()

    def step():
        opt.zero_grad(set_to_none=True)
        loss = outcome_loss(head(features.flatten(0, 1)).reshape(-1, 2), wins, counts)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        return loss

    times = []
    with ResourceMonitor(out / "telemetry") as monitor:
        monitor.mark("head_smoke")
        for iteration in range(8):
            torch.cuda.synchronize()
            start = perf_counter()
            step()
            torch.cuda.synchronize()
            if iteration >= 3:
                times.append(perf_counter() - start)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
            for _ in range(3):
                step()
        trace.export_chrome_trace(str(out / "trace.json"))
    atomic_torch_save({"model": head.state_dict(), "optimizer": opt.state_dict()}, out / "resume.pt")
    original = step().detach().clone()
    weights = {name: value.detach().clone() for name, value in head.state_dict().items()}
    saved = torch.load(out / "resume.pt", weights_only=False, map_location="cpu")
    head.load_state_dict(saved["model"])
    opt.load_state_dict(saved["optimizer"])
    torch.testing.assert_close(step(), original, atol=0, rtol=0)
    for name, value in head.state_dict().items():
        torch.testing.assert_close(value, weights[name], atol=0, rtol=0)
    controller = controller_smoke(cfg, out / "controller")
    cache = cached_vector_smoke(cfg)
    result = {
        "status": "passed",
        "parameters": sum(p.numel() for p in head.parameters()),
        "steady_step_seconds": sum(times) / len(times),
        "resume_exact": True,
        "controller": controller,
        "public_cache": cache,
        "top_kernels": top_cuda_operators(trace, 15),
    }
    write_json(out / "precheck.json", result)
    print(result, flush=True)
    return result


def controller_smoke(cfg, out):
    """Discardable synthetic fit: interrupted steps must equal an uninterrupted fit."""

    out = Path(out)
    seed = cfg["seed"]
    world_ids = range(10000, 10000 + sum(DEFAULT_WORLD_COUNTS))
    split = split_worlds(world_ids, seed=seed)
    worlds = np.repeat(world_ids, 2)
    configure(seed, "cuda")
    data = {
        "inputs": torch.randn(len(worlds), 2, HEAD_WIDTHS[0]),
        "wins": torch.randint(0, 5, (len(worlds), 2)).float(),
        "counts": torch.full((len(worlds), 2), 4.0),
        "first_wins": np.zeros((len(worlds), 2), dtype=int),
        "worlds": worlds,
        "keys": [[i, 0] for i in range(len(worlds))],
    }
    data["first_wins"] = data["wins"].numpy().astype(int)
    options = {**cfg, "epochs": 2, "batch_forks": 16}
    reference = out / "reference"
    resumed = out / "resumed"
    for directory in (reference, resumed):
        write_json(directory / "split.json", split)
        write_json(directory / "data.json", {"preparation_seconds": 0})
    configure(seed, "cuda")
    train(options, reference, data)
    calls = 0

    def interrupted():
        nonlocal calls
        calls += 1
        if calls == 7:
            raise InterruptedError("Smoke interruption in an uncommitted epoch")

    configure(seed, "cuda")
    try:
        train(options, resumed, data, after_step=interrupted)
    except InterruptedError:
        pass
    else:
        raise AssertionError("Smoke interruption did not occur")
    train(options, resumed, data)
    reference_state = torch.load(reference / "training.pt", weights_only=False, map_location="cpu")
    resumed_state = torch.load(resumed / "training.pt", weights_only=False, map_location="cpu")
    for key in ("epoch", "steps", "best", "cursor", "order"):
        assert reference_state[key] == resumed_state[key], key
    for name in reference_state["model"]:
        torch.testing.assert_close(
            reference_state["model"][name], resumed_state["model"][name], atol=0, rtol=0
        )

    configure(seed, "cuda")
    legacy = make_head()
    legacy_opt = make_optimizer(legacy)
    shuffle = random.Random(f"{SHUFFLE_NAMESPACE}|{seed}")
    training = np.flatnonzero(np.isin(data["worlds"], split["training"])).tolist()
    features, wins, counts = [data[key].cuda() for key in ("inputs", "wins", "counts")]
    for _ in range(2):
        order = training[:]
        shuffle.shuffle(order)
        for start in range(0, len(order), 16):
            ids = torch.tensor(order[start : start + 16], device="cuda")
            legacy_opt.zero_grad(set_to_none=True)
            outcome_loss(
                legacy(features[ids].flatten(0, 1)).reshape(-1, 2), wins[ids], counts[ids]
            ).backward()
            torch.nn.utils.clip_grad_norm_(legacy.parameters(), 1.0)
            legacy_opt.step()
    for name, value in legacy.state_dict().items():
        torch.testing.assert_close(value.cpu(), reference_state["model"][name], atol=0, rtol=0)

    for key, row in reference_state["optimizer"]["state"].items():
        for name, value in row.items():
            torch.testing.assert_close(value, resumed_state["optimizer"]["state"][key][name], atol=0, rtol=0)
    train(options, resumed, data)
    result = evaluate(options, resumed, data)
    receipt = (resumed / "selected.json").read_bytes()
    assert evaluate(options, resumed, data) == result
    assert (resumed / "selected.json").read_bytes() == receipt
    predictions = torch.load(resumed / "confirmation-predictions.pt", weights_only=False)
    probs = torch.as_tensor(predictions["logits"], dtype=torch.float64).sigmoid().numpy()
    np.testing.assert_array_equal(predictions["probs"], probs)
    np.testing.assert_array_equal(predictions["delta_prob"], probs[:, 1] - probs[:, 0])
    assert (
        resumed_state["best"]["epoch"] == 1
        or resumed_state["best"]["G"] > reference_state["history"][0]["selection"]["G"]
    )
    return {
        "resume_exact": True,
        "steps": resumed_state["steps"],
        "terminal_fit_stable": True,
        "confirmation_idempotent": True,
        "epoch_boundary_resume": True,
        "probabilities_exported": True,
        "original_update_math_exact": True,
    }


def cached_vector_smoke(cfg):
    """Compare the vector assembler to the frozen policy on real cached actions."""

    directory = Histories(ROOT / cfg["feature_cache_run"]).directory.parent / "head-features-fp32"
    dataset = read_json(ROOT / cfg["collection_input"]["run"] / "dataset.json")
    episodes = sorted({episode for episode, _ in dataset["anchors"]})
    cached = next(
        (
            (episode, receipt)
            for episode in episodes
            for receipt in sorted(directory.glob(f"{episode}-*.json"))
            if receipt.with_suffix(".pt").is_file()
            and read_json(receipt)["identity"]["common"]["parent"] == cfg["parent_sha256"]
        ),
        None,
    )
    if cached is None:
        return {
            "status": "cache_absent",
            "note": "Full preparation rebuilds features and checks ordinary-history parity",
        }
    episode, receipt = cached
    metadata = read_json(receipt)
    path = receipt.with_suffix(".pt")
    assert file_sha256(path) == metadata["sha256"]
    record = torch.load(path, weights_only=False, map_location="cpu", mmap=True)
    assert record["identity"] == metadata["identity"]

    subset = {
        **dataset,
        "pairs": [pair for pair in dataset["pairs"] if pair["episode"] == episode],
        "anchors": [key for key in dataset["anchors"] if key[0] == episode],
    }
    histories = Histories(ROOT / cfg["feature_cache_run"])
    # The normal preparation verifies these bytes against the source tree.
    histories.sources[episode] = dataset["source_hashes"][str(episode)]
    fast = FrozenActionFeatures(histories, subset, ROOT / cfg["parent"]).prepare()
    assert fast.statistics["cache_hits"] == 1 and fast.statistics["policy_loads"] == 0
    for name in ("x", "h"):
        torch.testing.assert_close(fast.records[episode][name], record[name], atol=0, rtol=0)
    policy = configure_heads(ROOT / cfg["parent"])
    policy.requires_grad_(False)
    start, stop = record["offsets"][:2]
    batch = {name: value[start:stop].unsqueeze(0).cuda() for name, value in record["actions"].items()}
    batch["action_mask"] = torch.ones(1, stop - start, dtype=torch.bool, device="cuda")
    spatial_context = record["x"][:1].cuda()
    recurrent_state = record["h"][:1].cuda()
    with torch.no_grad():
        logits, _ = policy.heads(batch, spatial_context, recurrent_state)
        vectors = torch.stack([public_action_vector(record, 0, a) for a in range(stop - start)]).cuda()
        manual = policy.action[2](
            policy.action[1](policy.action[0](vectors[:, :529]) + policy.attack_preview(vectors[:, 529:]))
        ).flatten()
    torch.testing.assert_close(manual, logits[0], atol=1e-4, rtol=1e-4)
    return {
        "episode": episode,
        "actions": stop - start,
        "max_logit_error": float((manual - logits[0]).abs().max()),
        "cache_policy_loads": fast.statistics["policy_loads"],
    }
