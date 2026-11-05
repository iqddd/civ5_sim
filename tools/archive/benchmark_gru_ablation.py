"""Temporary BC speed ablation: production GRU, matched-size FP32 MLP, bypass."""

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.nn import functional as F

from civ5rl.core.model import Policy
from civ5rl.engine.imitation import DemonstrationCache
from civ5rl.engine.imitation_performance import ResidentDemonstrations, backward_minibatch_async
from civ5rl.engine.ppo_fast import prefixes
from civ5rl.engine.resources import ResourceMonitor
from civ5rl.protocols.imitation import file_hash, load_episode
from civ5rl.provenance.common import configure, read_json, write_json
from tools.run_cp_imitation import adam


OUT = ROOT / "artifacts/rl/gru-ablation-20261105"


class BenchPolicy(Policy):
    def __init__(self, mode):
        super().__init__("C" if mode == "gru" else "B")
        self.bench_mode = mode
        self._ff_runner = None
        if mode == "bypass":
            self.memory = nn.Identity()

    def sequence_memory(self, ctx, lengths, hidden=None):
        if self.bench_mode == "gru":
            return super().sequence_memory(ctx, lengths, hidden)
        # Keep memory-block precision FP32 in both architectural alternatives.
        with torch.autocast(ctx.device.type, enabled=False):
            return ((self._ff_runner or self.memory)(ctx.float()), None)


def backward(policy, cache, spans, count):
    """Production BC, allowing None hidden for genuinely feedforward models."""
    torch.compiler.cudagraph_mark_step_begin()
    hidden = prefixes(policy, cache, spans)
    parameters = tuple(p for p in policy.parameters() if p.requires_grad)
    accumulated = None
    total = torch.zeros((), device=cache.device)
    for start in range(0, len(spans), 8):
        part = spans[start:start+8]
        if not any(any(cache.masks[ep][a:b]) for ep,a,b in part):
            continue
        batch, labels, lengths = cache.batch(part)
        h = hidden[:,start:start+len(part)].contiguous() if hidden is not None else None
        logits, _, _ = policy.forward_sequences(batch, lengths, h)
        loss = (F.cross_entropy(logits,labels["indices"],reduction="none")*labels["mask"]).sum()
        gradients = torch.autograd.grad(loss/count,parameters)
        if accumulated is None:
            accumulated = list(gradients)
            for p,g in zip(parameters,accumulated):
                p.grad = g
        else:
            torch._foreach_add_(accumulated,gradients)
        total.add_(loss.detach())
    result = total/count
    torch._assert_async(torch.isfinite(result),"Nonfinite CE")
    return result


def weighted(samples, end, field, seconds=60):
    start = end-seconds
    value = duration = 0.0
    for i,row in enumerate(samples):
        a = max(start,row["elapsed_seconds"])
        b = min(end,samples[i+1]["elapsed_seconds"] if i+1<len(samples) else end)
        if b>a and row[field] is not None:
            value += (b-a)*row[field]
            duration += b-a
    if duration < seconds-0.1:
        raise AssertionError(f"Insufficient {field} window: {duration}")
    return value/duration


def run(args):
    configure(17,"cuda")
    destination = OUT/f"{args.variant}-{args.repeat}"
    if (destination/"results.json").exists():
        raise FileExistsError(f"Do not overwrite completed probe: {destination}")
    destination.mkdir(parents=True,exist_ok=True)
    source = ROOT/"artifacts/rl/cp-imitation-more-data/checkpoints/best.pt"
    config = read_json(ROOT/"artifacts/rl/cp-imitation-more-data/resolved-config.json")
    fixture_path = ROOT/"artifacts/rl/bc-throughput-20261105/benchmark-inputs.json"
    fixture = read_json(fixture_path)
    assert file_hash(source) == fixture["checkpoint_sha256"]
    saved = torch.load(source,map_location="cpu",weights_only=False)
    episode_paths = [ROOT/f"artifacts/rl/cp-imitation/data/training/match-{i:06d}/demonstration.json.gz"
                     for i in fixture["episodes"]]
    batches = [([tuple(s) for s in b["spans"]],b["labels"]) for b in fixture["batches"]]
    assert len(batches)==8 and all(count==1024 for _,count in batches)
    write_json(destination/"inputs.json",{
        "checkpoint_sha256":file_hash(source),"fixture_sha256":file_hash(fixture_path),
        "episodes":{str(p.relative_to(ROOT)):file_hash(p) for p in episode_paths},
        "batches":fixture["batches"],"variant":args.variant,"seconds":args.seconds,
        "source_hashes":{p:file_hash(ROOT/p) for p in (
            "tools/archive/benchmark_gru_ablation.py","civ5rl/core/model.py","civ5rl/core/training_graphs.py",
            "civ5rl/engine/imitation.py","civ5rl/engine/imitation_performance.py",
            "civ5rl/engine/ppo_fast.py","civ5rl/engine/resources.py")}})
    print(f"{args.variant}: loading/encoding fixed 512 demonstrations",flush=True)
    with ResourceMonitor(destination/"telemetry") as monitor:
        monitor.mark("preparation")
        preparation_start = perf_counter()
        episodes = [load_episode(p) for p in episode_paths]
        cache = ResidentDemonstrations(DemonstrationCache(episodes,"cuda"))
        cache.enable_compiled_gather()
        cache.enable_async_transfers()
        policy = BenchPolicy(args.variant).cuda()
        shared = {k:v for k,v in saved["model"].items() if not k.startswith("memory.")}
        supplied = saved["model"] if args.variant=="gru" else shared
        missing,unexpected = policy.load_state_dict(supplied,strict=False)
        assert not unexpected
        assert all(k.startswith("memory.") for k in missing)
        assert all(torch.equal(policy.state_dict()[k].cpu(),v) for k,v in shared.items())
        for p in policy.value.parameters():
            p.requires_grad_(False)
        initial_weights = {k:v.detach().cpu().clone() for k,v in policy.state_dict().items()}
        policy.enable_amp(True,spatial_batch_size=32)
        policy.enable_inverse_routing()
        policy.enable_action_routing()
        policy.enable_maximum_routing()
        policy.enable_compile(dynamic=True,mode="default",spatial_mode="reduce-overhead")
        policy.enable_coalesced_spatial()
        policy.coalesced_accumulation = True
        if args.variant=="gru":
            policy.enable_recurrent_graphs(max_shapes=64)
        elif args.variant=="mlp":
            policy._ff_runner = torch.compile(policy.memory.forward,dynamic=True,fullgraph=True,mode="default")
        optimizer = adam(policy,config,"cuda")
        print(f"{args.variant}: compilation/warmup",flush=True)
        monitor.mark("compilation_warmup")
        # Check that conditional hidden support leaves the actual C backward unchanged.
        parity = None
        if args.variant=="gru" and args.repeat==1:
            optimizer.zero_grad(set_to_none=True)
            expected = backward_minibatch_async(policy,cache,*batches[0])
            expected_loss = float(expected)
            expected_grads = {k:p.grad.detach().clone() for k,p in policy.named_parameters() if p.grad is not None}
            optimizer.zero_grad(set_to_none=True)
            actual = backward(policy,cache,*batches[0])
            assert float(actual)==expected_loss
            assert all(torch.equal(p.grad,expected_grads[k]) for k,p in policy.named_parameters() if p.grad is not None)
            parity = {"production_backward_bitwise_equal":True,"loss":expected_loss}
            del expected_grads
        def step(i):
            optimizer.zero_grad(set_to_none=True)
            ce = backward(policy,cache,*batches[i%8])
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(),0.5,error_if_nonfinite=False)
            torch._assert_async(torch.isfinite(norm),"Nonfinite gradient norm")
            optimizer.step()
            return ce,norm
        for i in range(16):
            step(i)
            if i%8==7:
                torch.cuda.synchronize()
                print(f"{args.variant}: warmed {i+1}/16",flush=True)
        # No warmup updates enter the measured trajectory; all variants use new Adam.
        policy.load_state_dict(initial_weights)
        optimizer.state.clear()
        configure(17,"cuda")
        for i in range(8):
            step(i)
        torch.cuda.synchronize()
        captures_before = policy._memory_runner.captures if args.variant=="gru" else 0
        preparation = perf_counter()-preparation_start
        torch.cuda.reset_peak_memory_stats()
        monitor.mark("iterative")
        start = perf_counter()
        blocks = []
        steps = 0
        while perf_counter()-start < args.seconds:
            block_start = perf_counter()
            for _ in range(8):
                ce,norm = step(steps)
                steps += 1
            torch.cuda.synchronize()
            blocks.append((perf_counter()-block_start)/8)
        elapsed = perf_counter()-start
        phase_end = perf_counter()-monitor.started
        monitor.mark("postprocess")
        captures_after = policy._memory_runner.captures if args.variant=="gru" else 0
        assert captures_after==captures_before
        assert all(torch.equal(policy.state_dict()[k].cpu(),v) for k,v in initial_weights.items() if k.startswith("value."))
        result = {"variant":args.variant,"repeat":args.repeat,"steps":steps,
            "wall_seconds":elapsed,"mean_step_seconds":elapsed/steps,"labels_per_second":steps*1024/elapsed,
            "eight_step_block_mean_seconds":blocks,"block_mean_step_median":statistics.median(blocks),
            "block_mean_step_min":min(blocks),"block_mean_step_max":max(blocks),
            "preparation_seconds":preparation,"final_ce":float(ce),"final_gradient_norm":float(norm),
            "parameters":sum(p.numel() for p in policy.parameters()),
            "trainable_parameters":sum(p.numel() for p in policy.parameters() if p.requires_grad),
            "memory_parameters":sum(p.numel() for p in policy.memory.parameters()),
            "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved(),
            "GRU_captures_before":captures_before,"GRU_captures_after":captures_after,
            "Adam_steps":sorted(set(float(s["step"]) for s in optimizer.state.values())),
            "temporary_updates_discarded":True,"value_head_unchanged":True,"C_backward_parity":parity}
        assert result["Adam_steps"] == [steps+8]
    samples = [json.loads(line) for line in (destination/"telemetry/resources.jsonl").read_text().splitlines()]
    active = [r for r in samples if r["phase"]=="iterative"]
    result["iterative_samples"] = len(active)
    result["last_60_seconds"] = {k:weighted(active,phase_end,k) for k in (
        "gpu_core_load_percent","gpu_memory_controller_load_percent","gpu_power_watts","gpu_power_percent_tdp",
        "vram_used_bytes","cpu_tree_percent","ram_tree_rss_bytes")}
    result["iterative_gpu_core_mean"] = statistics.mean(r["gpu_core_load_percent"] for r in active)
    write_json(destination/"results.json",result)
    assert file_hash(source)==fixture["checkpoint_sha256"]
    print(json.dumps({k:v for k,v in result.items() if k!="eight_step_block_mean_seconds"},indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant",choices=("gru","mlp","bypass"),required=True)
    parser.add_argument("--repeat",type=int,default=1)
    parser.add_argument("--seconds",type=float,default=75)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
