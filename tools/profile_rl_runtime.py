"""Replay fixed episodes or collect fixed cases, without training or changing run state."""

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
from rl_collection_hardware import hardware


def behavior_hash(episodes):
    import numpy as np

    result = hashlib.sha256()
    for ep in episodes:
        for name in ("chosen", "old_logp", "old_value"):
            result.update(np.asarray(ep[name]).tobytes())
        for name in ("old_log_probs", "old_hidden"):
            for value in ep.get(name, []):
                result.update(b"none" if value is None else np.asarray(value).tobytes())
    return result.hexdigest()


def replay(episodes):
    from civ5rl.core.env import DecisionEnv
    from civ5sim.engine import digest

    results = []
    for ep in episodes:
        env = DecisionEnv(ep["spec"])
        try:
            for index, chosen in enumerate(ep["chosen"]):
                state = env.request()
                if (
                    state["actions"] != ep["actions"][index]
                    or state["observation"] != ep["observations"][index]
                ):
                    raise AssertionError(
                        "Replay changed legal actions or public observation"
                    )
                action = state["actions"][chosen]
                env.prepare(action)
                env.applied(action, env.battle.step(action))
            state = env.request()
            if not state["done"] or state["joint_commands"] != ep["joint_commands"]:
                raise AssertionError("Replay changed commands or episode boundary")
            result = {k: v for k, v in state["result"].items() if k != "timings"}
            expected = {k: v for k, v in ep["result"].items() if k != "timings"}
            if result != expected:
                raise AssertionError("Replay changed episode outcome")
            results.append(result)
        finally:
            env.close()
    return {
        "results_hash": digest(results),
        "commands_hash": digest([ep["joint_commands"] for ep in episodes]),
    }


def native_counters(library_path):
    import ctypes as C

    library = C.CDLL(str(library_path))
    if not hasattr(library, "cai_profile_count"):
        raise ValueError("Native counters require tools/build_combat_ai.py --profile")
    library.cai_profile_count.restype = C.c_int
    library.cai_profile_reset.argtypes = []
    library.cai_profile_reset.restype = None
    library.cai_profile_name.argtypes = [C.c_int]
    library.cai_profile_name.restype = C.c_char_p
    for name in ("cai_profile_ns", "cai_profile_calls"):
        getattr(library, name).argtypes = [C.c_int]
        getattr(library, name).restype = C.c_uint64
    return library


def load_collection_policy(config, checkpoint):
    import torch

    from civ5rl.core.model import Policy

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    policy = Policy(saved["variant"]).cuda().eval()
    policy.load_state_dict(saved["model"])
    performance = config.get("performance", {})
    policy.spatial_cell_padding = performance.get("spatial_cell_padding", 0)
    policy.enable_amp(
        config.get("numerics", {}).get("amp", False),
        spatial_batch_size=performance.get("spatial_batch_size", 0),
    )
    if performance.get("inverse_routing") or performance.get("optimized_routing"):
        policy.enable_inverse_routing()
    if performance.get("optimized_routing"):
        policy.enable_action_routing()
        policy.enable_maximum_routing()
    if performance.get("compile"):
        policy.enable_compile(spatial_mode=performance.get("spatial_compile_mode"))
    return policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("cpu", "collection"))
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--envs", type=int)
    parser.add_argument("--reply-size", type=int)
    parser.add_argument("--reference-trace", type=Path)
    parser.add_argument("--native-library", type=Path)
    parser.add_argument("--python-profile", action="store_true")
    args = parser.parse_args()
    started = perf_counter()
    if min(args.episodes, args.repeat) < 1 or args.warmup < 0:
        parser.error("Positive episode/repeat counts and nonnegative warmup required")
    sys.path.insert(0, str(args.source_root.resolve()))
    import torch

    from civ5rl.engine.resources import ResourceMonitor
    from civ5rl.experiments.ppo_artifacts import load_rollout
    from civ5rl.provenance.common import configure, file_sha256, write_json

    args.out.mkdir(parents=True, exist_ok=True)
    protected = [args.rollout.resolve()]
    if args.config:
        protected.append(args.config.resolve())
    if args.checkpoint:
        protected.append(args.checkpoint.resolve())
        for name in ("state.json", "status.json"):
            candidate = args.checkpoint.resolve().parent.parent / name
            if candidate.exists():
                protected.append(candidate)
    before = {str(path): file_sha256(path) for path in protected}
    configure(17, "cuda" if args.phase == "collection" else "cpu")
    episodes = load_rollout(args.rollout)[: args.episodes]
    if len(episodes) != args.episodes:
        parser.error("Rollout does not contain the requested episode count")
    counters = None
    if args.native_library:
        if args.phase != "cpu":
            parser.error("Native counter DLL is for sequential CPU replay")
        import civ5sim.combat_ai as planning

        planning.LIBRARY_PATH = args.native_library.resolve()
        counters = native_counters(planning.LIBRARY_PATH)
    result = {
        "hardware": hardware(),
        "phase": args.phase,
        "source_root": str(args.source_root.resolve()),
        "episodes": len(episodes),
        "positions": sum(len(ep["chosen"]) for ep in episodes),
        "protected_inputs": before,
        "warmup_repeats": args.warmup,
        "warmup_trials": [],
        "trials": [],
    }
    result["source_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.source_root, text=True
    ).strip()
    from civ5sim.combat_ai import LIBRARY_PATH

    result["native_library_sha256"] = file_sha256(LIBRARY_PATH)
    with ResourceMonitor(args.out) as monitor:
        collector = None
        try:
            if args.phase == "collection":
                if args.config is None or args.checkpoint is None:
                    parser.error("Collection requires --config and --checkpoint")
                from civ5rl.config import load_config
                from civ5rl.engine.collector import Collector
                from civ5rl.experiments.imitation_helpers import CompilationClock
                from civ5rl.provenance import bound_cases, case_manifest
                from civ5sim.engine import digest

                config = load_config(args.config)
                policy = load_collection_policy(config, args.checkpoint)
                manifest = case_manifest(
                    [ep["spec"] for ep in episodes],
                    source="runtime_profile",
                    split="train",
                    namespace="fixed_cases",
                )
                specs = bound_cases(manifest)
                # Runtime choices preserve scientific config and case specs.
                runtime = {"cpu_workers": args.workers}
                if args.envs is not None:
                    runtime["num_envs"] = args.envs
                if args.reply_size is not None:
                    runtime["reply_size"] = args.reply_size
                collector = Collector(
                    config, None, "cuda", args.out / "errors", **runtime
                )
                result.update(
                    workers=len(collector.workers),
                    num_envs=collector.num_envs,
                    reply_size=collector.reply_size,
                    GPU=torch.cuda.get_device_name(),
                    torch=torch.__version__,
                )
                reference_trace = None
                if args.reference_trace:
                    import numpy as np

                    with np.load(args.reference_trace) as archive:
                        reference_trace = {
                            name: archive[name] for name in archive.files
                        }

                def action():
                    nonlocal reference_trace
                    import numpy as np

                    with CompilationClock() as compiler:
                        compiler.monitor = monitor
                        values, _, timings = collector.collect(
                            policy,
                            None,
                            0,
                            fixed_specs=specs,
                            trusted_manifest=manifest,
                        )
                    timings["compilation_seconds"] = sum(
                        b - a for a, b in compiler.intervals
                    )
                    monitor.mark("verification")
                    trace = {}
                    for name in ("old_log_probs", "old_hidden", "old_value"):
                        parts = [
                            np.asarray(v).reshape(-1)
                            for ep in values
                            for v in ep[name]
                            if v is not None
                        ]
                        trace[name] = (
                            np.concatenate(parts) if parts else np.empty(0, np.float32)
                        )
                    if reference_trace is None:
                        reference_trace = trace
                    if not (args.out / "trace.npz").exists():
                        np.savez(args.out / "trace.npz", **trace)
                    errors = {
                        name: None
                        if v.shape != reference_trace[name].shape
                        else float(np.max(np.abs(v - reference_trace[name]), initial=0))
                        for name, v in trace.items()
                    }
                    return {
                        "commands_hash": digest(
                            [ep["joint_commands"] for ep in values]
                        ),
                        "results_hash": digest(
                            [
                                {
                                    k: v
                                    for k, v in ep["result"].items()
                                    if k != "timings"
                                }
                                for ep in values
                            ]
                        ),
                        "commands": sum(len(ep["joint_commands"]) for ep in values),
                        "positions": sum(len(ep["chosen"]) for ep in values),
                        "behavior_hash": behavior_hash(values),
                        "max_behavior_deviation": errors,
                        "timings": timings,
                    }
            else:

                def action():
                    return replay(episodes)

            reference = None
            result["setup_seconds"] = perf_counter() - started
            write_json(args.out / "result.json", result)
            for index in range(args.warmup + args.repeat):
                phase = "warmup" if index < args.warmup else args.phase
                monitor.mark(phase)
                if counters:
                    counters.cai_profile_reset()
                tick = perf_counter()
                outcome = action()
                elapsed = perf_counter() - tick
                seconds = outcome.get("timings", {}).get("collection_seconds", elapsed)
                # FP32 heads can round differently when asynchronous batching
                # changes shape. Keep hashes/deviations, require exact commands.
                signature = {
                    k: v
                    for k, v in outcome.items()
                    if k.endswith("hash") and k != "behavior_hash"
                }
                if reference is not None and reference != signature:
                    raise AssertionError(
                        "Repeated workload changed commands, behavior or results"
                    )
                reference = signature
                trial = {
                    "seconds": seconds,
                    "verification_seconds": elapsed - seconds,
                    **outcome,
                }
                if index >= args.warmup:
                    if counters:
                        trial["native_inclusive"] = {
                            counters.cai_profile_name(i).decode(): {
                                "seconds": counters.cai_profile_ns(i) / 1e9,
                                "calls": counters.cai_profile_calls(i),
                            }
                            for i in range(counters.cai_profile_count())
                        }
                    result["trials"].append(trial)
                    print(json.dumps(trial), flush=True)
                else:
                    result["warmup_trials"].append(trial)
                write_json(args.out / "result.json", result)
                monitor.mark("between_trials")
            if args.python_profile and args.phase == "cpu":
                import cProfile

                monitor.mark("python_profile")
                profiler = cProfile.Profile()
                profiler.runcall(action)
                profiler.dump_stats(str(args.out / "python.prof"))
        finally:
            if collector:
                collector.close()
            monitor.mark("finished")
    if any(file_sha256(path) != sha for path, sha in before.items()):
        raise AssertionError("Profiling modified a protected input/run state")
    result["median_seconds"] = statistics.median(t["seconds"] for t in result["trials"])
    write_json(args.out / "result.json", result)
    print(f"median_seconds={result['median_seconds']:.6f}", flush=True)


if __name__ == "__main__":
    main()
