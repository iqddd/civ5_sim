"""Compare identical fixed-rollout PPO workloads; optional CPU/CUDA trace."""

import argparse
import copy
import importlib.util
import os
import random
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")


def main():
    import torch

    from civ5rl.common import configure, read_json, write_json
    from civ5rl.model import Policy, batch_inputs
    from civ5rl.ppo import encode_episode, update
    from civ5rl.resources import ResourceMonitor

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("reference", "fast", "compiled"), required=True)
    p.add_argument("--microbatch", type=int, default=8)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--rollout",
        required=True,
    )
    p.add_argument(
        "--checkpoint", required=True, help="Checkpoint used to collect this rollout"
    )
    p.add_argument(
        "--reference-implementation",
        type=Path,
        help="Optional historical PPO Python source",
    )
    p.add_argument("--limit-episodes", type=int)
    p.add_argument("--trace", action="store_true")
    p.add_argument("--amp-bf16", action="store_true")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--dynamic", choices=("true", "auto", "false"), default="true")
    args = p.parse_args()
    directory = Path(args.out).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(read_json("configs/rl-v1.json"))
    config["ppo"]["sequences_per_microbatch"] = args.microbatch
    config["performance"] = {
        "batched_ppo": args.mode != "reference",
        "resident_rollout": True,
        "kl_sequences_per_batch": 8,
        "fused_adam": args.mode != "reference",
    }
    episodes = torch.load(args.rollout, map_location="cpu", weights_only=False)
    if args.limit_episodes:
        episodes = episodes[: args.limit_episodes]
    configure(config["seed"], "cuda")
    initial = torch.load(
        args.checkpoint,
        map_location="cuda",
        weights_only=False,
    )["model"]
    policy = Policy().cuda()
    policy.load_state_dict(initial)
    policy.enable_amp(args.amp_bf16)
    config["numerics"].update(
        amp=args.amp_bf16,
        precision="mixed_bfloat16" if args.amp_bf16 else "float32",
        # Historical fixed-rollout artifacts have only sampled behavior logp.
        require_full_behavior_distribution=False,
    )
    updater = update
    if args.mode == "compiled":
        policy.enable_compile(
            dynamic={"true": True, "false": False, "auto": None}[args.dynamic]
        )
    if args.mode == "reference" and args.reference_implementation:
        spec = importlib.util.spec_from_file_location(
            "civ5rl.reference_ppo",
            args.reference_implementation,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        updater = module.update
    write_json(directory / "resolved-config.json", config)
    try:
        with ResourceMonitor(directory) as monitor:
            monitor.mark("compile_warmup" if args.mode == "compiled" else "warmup")
            warmup_start = perf_counter()
            encoded = encode_episode(episodes[0], "C")[:32]
            batch = batch_inputs(encoded, "cuda")
            for _ in range(2):
                logits, value, _ = policy.forward_sequences(batch, [len(encoded)])
                loss = (
                    logits.masked_fill(~torch.isfinite(logits), 0).square().mean()
                    + value.square().mean()
                )
                loss.backward()
                policy.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            warmup_seconds = perf_counter() - warmup_start
            rows = []
            for repetition in range(args.repeats):
                policy.load_state_dict(initial)
                optimizer = torch.optim.Adam(
                    policy.parameters(),
                    lr=3e-4,
                    betas=(0.9, 0.999),
                    eps=1e-5,
                    fused=args.mode != "reference",
                )
                torch.cuda.reset_peak_memory_stats()
                monitor.mark("ppo_fixed_rollout", update=repetition + 1)
                torch.cuda.synchronize()
                start = perf_counter()
                if args.trace:
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        record_shapes=True,
                        profile_memory=True,
                    ) as profiler:
                        stats = updater(
                            policy,
                            optimizer,
                            episodes,
                            config,
                            0,
                            "cuda",
                            random.Random(config["seed"] + 1),
                            monitor,
                        )
                    profiler.export_chrome_trace(
                        str(directory / f"trace-{repetition}.json")
                    )
                    (directory / f"operators-{repetition}.txt").write_text(
                        profiler.key_averages().table(
                            sort_by="self_cuda_time_total", row_limit=35
                        ),
                        encoding="utf-8",
                    )
                else:
                    stats = updater(
                        policy,
                        optimizer,
                        episodes,
                        config,
                        0,
                        "cuda",
                        random.Random(config["seed"] + 1),
                        monitor,
                    )
                torch.cuda.synchronize()
                row = {
                    "repetition": repetition,
                    "wall_seconds": perf_counter() - start,
                    "stats": stats,
                    "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                }
                rows.append(row)
                write_json(
                    directory / "summary.json",
                    {
                        "mode": args.mode,
                        "microbatch": args.microbatch,
                        "decisions": sum(len(e["chosen"]) for e in episodes),
                        "warmup_seconds": warmup_seconds,
                        "repetitions": rows,
                        "torch": torch.__version__,
                        "cuda": torch.version.cuda,
                    },
                )
                print(row, flush=True)
            torch.save(policy.state_dict(), directory / "final-model.pt")
    except BaseException:
        import traceback

        write_json(directory / "failure.json", {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
