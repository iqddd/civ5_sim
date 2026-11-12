"""Bounded CPU/GPU evaluation measurements on saved validation cases."""

from pathlib import Path
from time import perf_counter

import torch

from civ5rl.core.agent import RLAgent
from civ5rl.core.model import batch_inputs
from civ5rl.core.observation import ordered_actions
from civ5rl.engine.evaluation import BatchedInference, EvaluationExecutor, EvaluationRuntime
from civ5rl.engine.pool import pool
from civ5rl.engine.resources import ResourceMonitor
from civ5rl.experiments.ppo_target_scale import evaluation_task
from civ5rl.provenance.common import configure, file_sha256, read_json, write_json
from civ5rl.experiments.evaluation import verify_replay
from civ5sim import Battle
from tools._bc_benchmark import summarize_kineto_trace


def numeric_parity(checkpoint, case, *, steps=3):
    """Compare the same public history in CPU/GPU FP32 without new matches."""
    agents = [RLAgent.from_checkpoint(checkpoint, device, "argmax") for device in ("cpu", "cuda")]
    for agent in agents:
        agent.policy.enable_amp(False)
        agent.reset(dict(map=case["scenario"]["map"], side=0, armies=case["armies"], max_rounds=case["max_rounds"],
                         fog=case["scenario"]["fog"], random_damage=case["scenario"]["random_damage"], seed=17))
    with Battle(case["scenario"]) as battle:
        observation, actions = battle.observe(0), ordered_actions(battle.available_actions())
    max_logits = max_hidden = 0.0
    argmax_differences = 0
    with torch.inference_mode():
        for _ in range(steps):
            outputs = []
            for agent in agents:
                encoded = agent.encoder.encode(observation, actions)
                logits, _, agent.hidden = agent.policy(batch_inputs([encoded], agent.device), agent.hidden)
                outputs.append(logits.cpu())
            valid = torch.isfinite(outputs[0]) & torch.isfinite(outputs[1])
            max_logits = max(max_logits, float((outputs[0][valid] - outputs[1][valid]).abs().max()))
            if agents[0].hidden is not None:
                max_hidden = max(max_hidden, float((agents[0].hidden - agents[1].hidden.cpu()).abs().max()))
            torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-4, atol=1e-5)
            if agents[0].hidden is not None:
                torch.testing.assert_close(agents[0].hidden, agents[1].hidden.cpu(), rtol=1e-4, atol=1e-5)
            first = int(outputs[0][0].argmax())
            argmax_differences += int(first != int(outputs[1][0].argmax()))
            for agent in agents:
                agent.encoder.remember_action(actions[first], observation)
    return dict(steps=steps, max_abs_logits=max_logits, max_abs_hidden=max_hidden,
                argmax_differences=argmax_differences, rtol=1e-4, atol=1e-5,
                scope="Repeated fixed public observation; identical CPU-chosen action history")


def run_evaluation_benchmark(args, parser):
    if args.run is None or args.cases < 1 or args.workers < 1 or args.steps < 1 or args.warmup < 0 or args.trace_steps < 0:
        parser.error("evaluation requires --run and positive --cases/--workers/--steps")
    if not torch.cuda.is_available():
        parser.error("CPU/GPU comparison requires CUDA")
    root, out = args.run.resolve(), args.out.resolve()
    if out.exists():
        parser.error("benchmark output must not exist")
    record_path = root / "run.json"
    cfg_path = root / "resolved-config.json"
    cfg = read_json(cfg_path) if cfg_path.is_file() else read_json(record_path)["config"]
    checkpoint = args.checkpoint.resolve() if args.checkpoint else root / read_json(root / "state.json")["latest"]
    manifest_path = root / "manifest.json"
    plan = read_json(manifest_path)
    purpose = cfg.get("ppo", {}).get("evaluation_purpose", "development_512")
    cases = plan[purpose]
    # Spread the small sample over available map sizes before filling it.
    groups = {}
    for case in cases:
        groups.setdefault(case.get("map_size", 0), case)
    chosen = list(groups.values())
    if args.cases == 2 and len(chosen) > 2:
        chosen = [chosen[0], chosen[-1]]
    chosen = chosen[:args.cases]
    for case in cases:
        if len(chosen) >= args.cases:
            break
        if case not in chosen:
            chosen.append(case)
    protected = [p for p in (record_path, cfg_path, checkpoint, manifest_path, root / "state.json", root / "updates.jsonl") if p.exists()]
    before = {str(p): file_sha256(p) for p in protected}
    configure(args.seed, "cuda")
    out.mkdir(parents=True)
    results = {}
    gpu_executor = None
    for device in ("cpu", "cuda"):
        runtime = EvaluationRuntime(device, args.workers, args.max_batch, args.wait_seconds).resolve()
        tasks = [(cfg, case, str(checkpoint), str(out / device / f"case-{i:03d}")) for i, case in enumerate(chosen)]
        with ResourceMonitor(out / f"{device}-telemetry", interval=0.25) as monitor:
            monitor.mark("evaluation", device=device)
            started = perf_counter()
            if device == "cuda":
                gpu_executor = EvaluationExecutor(checkpoint, runtime)
                rows = list(gpu_executor.run(evaluation_task, tasks))
                statistics = gpu_executor.statistics()
            else:
                with pool(args.workers) as executor:
                    rows = list(executor.map(evaluation_task, tasks))
                statistics = None
            seconds = perf_counter() - started
        for _, _, _, directory in tasks:
            verify_replay(Path(directory))
        results[device] = dict(seconds=seconds, runtime=runtime.identity(), inference=statistics,
                               rows=rows, telemetry=read_json(out / f"{device}-telemetry/resources-summary.json"))
        print(f"Evaluation benchmark {device}: {seconds:.3f}s, {len(rows)} verified matches", flush=True)

    # Measure the inference component separately; full-match time above includes
    # spawning, CP, simulator and artifact writes. This is a repeated public
    # initial observation, not an additional scientific evaluation.
    case = chosen[0]
    with Battle(case["scenario"]) as battle:
        observation, actions = battle.observe(0), battle.available_actions()
    context = dict(map=case["scenario"]["map"], side=0, armies=case["armies"], max_rounds=case["max_rounds"],
                   fog=case["scenario"]["fog"], random_damage=case["scenario"]["random_damage"], seed=args.seed)
    policy = gpu_executor.inference.policy
    agents = [RLAgent(policy, "cuda", "argmax") for _ in range(min(args.workers, args.max_batch))]
    for agent in agents:
        agent.reset(context)
    requests = [(a, "act", observation, actions, None) for a in agents]
    inference = BatchedInference(policy, "cuda")
    for _ in range(args.warmup):
        inference.infer(requests)
    torch.cuda.synchronize()
    measured = []
    for _ in range(args.steps):
        started = perf_counter()
        inference.infer(requests)
        torch.cuda.synchronize()
        measured.append(perf_counter() - started)
    trace_summary = None
    if args.trace_steps:
        trace = out / "inference-trace.json.gz"
        started = perf_counter()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
            for _ in range(args.trace_steps):
                inference.infer(requests)
            torch.cuda.synchronize()
        seconds = perf_counter() - started
        profiler.export_chrome_trace(str(trace))
        trace_summary = summarize_kineto_trace(trace, seconds)
        write_json(out / "inference-trace-summary.json", trace_summary)
    after = {str(p): file_sha256(p) for p in protected}
    if before != after:
        raise RuntimeError("Protected scientific inputs changed")
    # Match by case identity, not asynchronous completion order.
    by_device = {d: {(r["case_hash"], r["side"]): r for r in v["rows"]} for d, v in results.items()}
    mismatches = [key for key in by_device["cpu"] if
                  (by_device["cpu"][key]["winner"], by_device["cpu"][key]["rounds"]) !=
                  (by_device["cuda"][key]["winner"], by_device["cuda"][key]["rounds"])]
    result = dict(results=results, cpu_over_cuda_wall_ratio=results["cpu"]["seconds"] / results["cuda"]["seconds"],
                  measured_inference_seconds=measured, mean_inference_seconds=sum(measured) / len(measured),
                  trace_summary=trace_summary, outcome_or_round_mismatches=mismatches,
                  protected_inputs_unchanged=True, protected_sha256=before,
                  scope="Bounded validation cases; no training. Inference-only timings reuse a public initial observation.")
    result["numeric_parity"] = numeric_parity(checkpoint, chosen[0])
    write_json(out / "benchmark.json", result)
    (out / "report.md").write_text(
        "# CPU/GPU evaluation benchmark\n\n"
        f"Cases: {len(chosen)}; workers: {args.workers}; max batch: {args.max_batch}; wait: {args.wait_seconds}s.\n\n"
        f"CPU wall: {results['cpu']['seconds']:.3f}s; CUDA wall: {results['cuda']['seconds']:.3f}s; "
        f"ratio: {result['cpu_over_cuda_wall_ratio']:.3f}. Includes startup, CP, simulator and file writes.\n\n"
        f"Outcome/round mismatches: {len(mismatches)}. All generated replays verified. Protected inputs unchanged.\n\n"
        f"Warm inference batch: {result['mean_inference_seconds'] * 1000:.3f} ms, "
        f"{len(measured)} steps after {args.warmup} warmup steps.\n",
        encoding="utf-8")
    return result
