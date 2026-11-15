"""Disposable fixed-shape PPO timing in an explicitly selected source checkout."""

import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True, help="Resolved config, independent of checkout")
    parser.add_argument("--template", type=Path, required=True, help="One saved public Encoded observation")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rollout-episodes", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args(argv)
    args.source_root = args.source_root.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.settings = args.settings.resolve()
    args.template = args.template.resolve()
    args.out = args.out.resolve()
    if args.rollout_episodes < 8 or args.rollout_episodes % 8 or args.warmup < 1 or args.steps < 1:
        parser.error("Need a multiple of 8 rollout episodes, and positive warmup/steps")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Choose a new empty output directory")
    if any(args.out == p or p.is_relative_to(args.out) for p in (args.checkpoint, args.settings, args.template)):
        parser.error("Output must not contain protected inputs")
    # This executable deliberately imports *all* execution code from the selected checkout.
    # Calling it in a fresh process avoids cached current modules contaminating the old version.
    sys.path.insert(0, str(args.source_root))
    os.chdir(args.source_root)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import copy
    import statistics
    import subprocess
    import numpy as np
    import torch
    from civ5rl.core.model import batch_inputs
    from civ5rl.engine.optimizer_transaction import AdamSnapshot, require_finite_state
    from civ5rl.engine.ppo_adam import grouped_optimizer
    from civ5rl.engine.ppo_fast import BatchedBackend
    from civ5rl.engine.resources import ResourceMonitor
    from civ5rl.engine.spans import Span
    from civ5rl.experiments.imitation_helpers import CompilationClock, load_policy, assert_equal
    from civ5rl.provenance.common import configure, file_sha256, write_json

    paths = (args.checkpoint, args.settings, args.template)
    hashes = {str(p): file_sha256(p) for p in paths}
    cfg = json.loads(args.settings.read_text(encoding="utf-8"))
    cfg = cfg.get("config", cfg)
    configure(17, "cuda")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    policy = load_policy(cfg, args.checkpoint, "cuda")
    for parameter in policy.parameters():
        parameter.requires_grad_(True)
    optimizer = grouped_optimizer(policy, cfg, "cuda")
    optimizer.load_state_dict(saved["optimizer"])
    initial = AdamSnapshot(policy, optimizer)
    encoded = torch.load(args.template, map_location="cpu", weights_only=False)
    if len(encoded.cells) != 256:
        raise ValueError("Fixed-shape template must contain 256 cells")
    actions = len(encoded.sources)
    temperature = float(cfg["policy_temperature"])
    # Every chunk is exactly128 decisions with identical action/cell dimensions.
    # This tests execution cost; repeated public positions are not real battles or learning evidence.
    repeated = [encoded] * 128
    policy.eval()
    with torch.no_grad():
        torch.compiler.cudagraph_mark_step_begin()
        logits, value, _ = policy.forward_sequences(batch_inputs(repeated, "cuda"), [128])
        logp = (logits.float() / temperature).log_softmax(-1).cpu().numpy()
        values = value.cpu().numpy()
    episodes = []
    rng = np.random.default_rng(17)
    for i in range(args.rollout_episodes):
        chosen = rng.integers(actions, size=128)
        episodes.append(dict(encoded=repeated, encoded_variant=policy.variant,
            chosen=chosen.tolist(), old_log_probs=list(logp.copy()),
            old_logp=logp[np.arange(128), chosen].tolist(), old_value=values.tolist(),
            policy_temperature=temperature, result=dict(reward=float(i % 2))))
    policy.train()
    args.out.mkdir(parents=True)
    backend = BatchedBackend(policy, episodes, cfg, "cuda", pinned_replay=True)
    precheck = backend.precheck()
    minibatch = [Span(i, 0, 128) for i in range(8)]
    batch, _, lengths = backend.cache.batch(minibatch, include_behavior=False)
    shapes = {k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in batch.items()}
    del batch
    assert lengths == [128] * 8
    records = []

    def graph_captures():
        import torch._inductor.cudagraph_trees as trees
        manager = trees.get_manager(torch.cuda.current_device(), create_if_none_exists=False)
        return (manager.graph_counter.__reduce__()[1][0] if manager else 0,
                getattr(policy._executor.memory_runner, "captures", 0))

    def step():
        optimizer.zero_grad(set_to_none=True)
        stamp = perf_counter()
        backend.backward(minibatch, 1024, float(cfg["ppo"]["entropy_coefficient"]))
        backward = perf_counter() - stamp
        norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg["ppo"]["max_gradient_norm"], error_if_nonfinite=True)
        stamp = perf_counter()
        optimizer.step()
        require_finite_state(policy, optimizer)
        adam = perf_counter() - stamp
        stamp = perf_counter()
        replay = backend.measure_step_replay()
        replay_seconds = perf_counter() - stamp
        if replay.acceptance_kl > cfg["ppo"]["stop_kl"]:
            raise RuntimeError("Synthetic five-step benchmark exceeded the unchanged KL limit")
        return dict(backward_seconds=backward, adam_finite_seconds=adam,
                    replay_seconds=replay_seconds, kl=replay.acceptance_kl,
                    gradient_norm=float(norm),
                    Adam_counters={g["name"]: sorted({int(optimizer.state[p]["step"]) for p in g["params"]})
                                   for g in optimizer.param_groups})

    with ResourceMonitor(args.out / "telemetry", interval=.1) as monitor:
        monitor.mark("warmup")
        with CompilationClock() as clock:
            for i in range(args.warmup):
                row = step()
                torch.cuda.synchronize()
                print(f"WARMUP {i+1} KL={row['kl']:.8f}", flush=True)
        initial.restore()
        optimizer.zero_grad(set_to_none=True)
        monitor.mark("measured_steps")
        for i in range(args.steps):
            torch.cuda.synchronize()
            graphs_before = graph_captures()
            with CompilationClock() as clock:
                start = perf_counter()
                row = step()
                torch.cuda.synchronize()
                stop = perf_counter()
            compilation = clock.seconds(start, stop)
            graphs_after = graph_captures()
            if compilation > 0 or graphs_before != graphs_after:
                raise RuntimeError("Compilation/CUDA Graph capture entered measured fixed-shape step; increase warmup")
            row.update(step=i+1, wall_seconds=stop-start, compilation_seconds=compilation,
                       graphs_before=graphs_before, graphs_after=graphs_after)
            records.append(row)
            print("MEASURED " + json.dumps(row), flush=True)
        # Profile separately; never blend Kineto overhead into the five reported steps.
        initial.restore()
        monitor.mark("trace")
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            step()
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(args.out / "step-trace.json"))
        events = prof.key_averages()
        summary = dict(sync=[dict(name=e.key,count=e.count,cpu_us=e.self_cpu_time_total)
                            for e in events if "Synchronize" in e.key or "record_data" in e.key],
                       top_kernels=sorted([dict(name=e.key,count=e.count,cuda_us=e.device_time_total)
                                          for e in events if e.device_time_total],key=lambda x:-x["cuda_us"])[:15])
        write_json(args.out / "trace-summary.json", summary)
    initial.restore()
    assert_equal(policy.state_dict(), saved["model"])
    # Device placement of Adam step scalars can differ; compare values on CPU.
    restored = copy.deepcopy(optimizer.state_dict())
    for state in restored["state"].values():
        for key, value in state.items():
            if torch.is_tensor(value): state[key] = value.cpu()
    assert_equal(restored, saved["optimizer"])
    if any(file_sha256(p) != checksum for p, checksum in hashes.items()):
        raise AssertionError("Scientific input changed")
    execution = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    names = ("civ5rl/core/model.py", "civ5rl/core/training_graphs.py", "civ5rl/engine/ppo.py", "civ5rl/engine/ppo_fast.py")
    result = dict(schema=1, source_commit=execution, source_root=str(args.source_root),
                  source_sha256={name:file_sha256(args.source_root/name) for name in names},
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  inputs_sha256=hashes, shapes=shapes, recurrent_lengths=lengths,
                  rollout_decisions=args.rollout_episodes*128, minibatch_decisions=1024,
                  warmup_steps=args.warmup, measured_steps=records,
                  median_seconds=statistics.median(r["wall_seconds"] for r in records),
                  precheck=precheck, inputs_model_adam_unchanged=True,
                  interpretation="Five fixed-shape optimizer+finite-state+full-KL steps; synthetic public histories; excludes collection, setup and compilation; not decisions/s of training")
    write_json(args.out / "benchmark.json", result)
    print("RESULT " + json.dumps({k:result[k] for k in ("source_commit", "median_seconds", "gpu")}), flush=True)


if __name__ == "__main__":
    main()
