"""Frozen-policy equivalence probes or full-cycle PPO benchmark, with NVML.

--source-root can point to an archived civ5rl/civ5sim pair for an independent
baseline process. Native kernel and corpus stay fixed at the workspace root.
The explicitly recorded executed-source hashes are authoritative in that mode.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if "--source-root" in sys.argv:
    os.environ["CIV5RL_BENCH_SOURCE_ROOT"] = sys.argv[
        sys.argv.index("--source-root") + 1
    ]
SOURCE_ROOT = Path(os.environ.get("CIV5RL_BENCH_SOURCE_ROOT", ROOT)).resolve()
sys.path.insert(0, str(SOURCE_ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# Also executed in spawned workers; archived bindings use the same native DLL.
import civ5sim._native as native
from civ5rl import common

native.ROOT = common.ROOT = ROOT


def main():
    import torch

    from civ5rl.collector import Collector
    from civ5rl.corpus import ScenarioFactory
    from civ5rl.model import Policy
    from civ5rl.ppo import encode_episode
    from civ5rl.resources import ResourceMonitor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-root")
    parser.add_argument("--mode", choices=("collect", "full"), default="collect")
    parser.add_argument("--checkpoint")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--updates", type=int, default=10)
    args = parser.parse_args()
    config = common.read_json(args.config)
    manifest = common.read_json(ROOT / "artifacts/rl/corpus/manifest.json")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    executed_paths = [
        p
        for package in ("civ5rl", "civ5sim")
        for p in sorted((SOURCE_ROOT / package).glob("*.py"))
    ]
    executed_paths.append(SOURCE_ROOT / "tools/train_rl.py")
    common.write_json(
        out / "executed-sources.json",
        {
            "root": str(SOURCE_ROOT),
            "hashes": {
                str(p.relative_to(SOURCE_ROOT)): hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
                for p in executed_paths
            },
            "config": config,
        },
    )
    import zipfile

    with zipfile.ZipFile(
        out / "executed-sources.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in executed_paths:
            archive.write(path, str(path.relative_to(SOURCE_ROOT)))
        archive.write(Path(__file__), "tools/benchmark_rl_pipeline.py")
        archive.writestr("resolved-config.json", json.dumps(config, indent=2))
    if args.mode == "full":
        from tools.train_rl import run

        if (
            Path(run.__code__.co_filename).resolve()
            != (SOURCE_ROOT / "tools/train_rl.py").resolve()
        ):
            raise RuntimeError(
                "Training driver was not loaded from the selected source root"
            )

        if SOURCE_ROOT != ROOT:
            from civ5rl import checkpoint

            original_passport = checkpoint.passport

            def executed_passport(directory, config, manifest):
                identity = original_passport(directory, config, manifest)
                hashes = identity["source_hashes"]
                for key in list(hashes):
                    if key.startswith(("civ5rl/", "civ5sim/")):
                        del hashes[key]
                for path in executed_paths:
                    hashes[str(path.relative_to(SOURCE_ROOT)).replace("\\", "/")] = (
                        hashlib.sha256(path.read_bytes()).hexdigest()
                    )
                common.write_json(Path(directory) / "passport.json", identity)
                archive_path = Path(directory) / "sources.zip"
                with zipfile.ZipFile(archive_path) as archive:
                    retained = {
                        name: archive.read(name)
                        for name in archive.namelist()
                        if not name.replace("\\", "/").startswith(
                            ("civ5rl/", "civ5sim/")
                        )
                        and name.replace("\\", "/") != "tools/train_rl.py"
                    }
                with zipfile.ZipFile(
                    archive_path, "w", zipfile.ZIP_DEFLATED
                ) as archive:
                    for name, data in retained.items():
                        archive.writestr(name, data)
                    for path in executed_paths:
                        archive.write(path, str(path.relative_to(SOURCE_ROOT)))
                return identity

            checkpoint.passport = executed_passport

        rows = []
        for repeat in range(args.repetitions):
            directory = out / f"repeat-{repeat}"
            run(
                config,
                manifest,
                directory,
                device="cuda",
                budget=2**63,
                max_updates=args.warmup + args.updates,
                evaluate_every=False,
                replays=False,
                stage_cycle=[0, 1, 2],
            )
            metrics = [
                json.loads(line)
                for line in (directory / "metrics.jsonl").read_text().splitlines()
            ]
            measured = metrics[args.warmup :]
            row = {
                "repetition": repeat,
                "decisions": sum(x["rollout_decisions"] for x in measured),
                "seconds": sum(x["update_seconds"] for x in measured),
                "measured": measured,
            }
            row["decisions_per_second"] = row["decisions"] / row["seconds"]
            rows.append(row)
            common.write_json(
                out / "summary.json",
                {
                    "repetitions": rows,
                    "decisions_per_second": sum(x["decisions"] for x in rows)
                    / sum(x["seconds"] for x in rows),
                },
            )
        return

    if args.checkpoint is None:
        parser.error("--checkpoint is required for collection probes")
    common.configure(config["seed"], "cuda")
    policy = Policy().cuda().eval()
    policy.load_state_dict(
        torch.load(args.checkpoint, map_location="cuda", weights_only=False)["model"]
    )
    spatial_batch_size = config.get("performance", {}).get("spatial_batch_size", 0)
    if spatial_batch_size:
        policy.enable_amp(spatial_batch_size=spatial_batch_size)
    else:
        policy.enable_amp()
    policy.enable_compile()
    factory = ScenarioFactory(config, manifest)
    stage = config["curriculum"][2]
    specs = [factory.make(100000 + i, stage) for i in range(128)]
    from civ5rl.provenance import bound_cases, case_manifest

    trusted_manifest = case_manifest(
        specs, source="rl01.collection_benchmark", split="train", namespace="benchmark"
    )
    specs = bound_cases(trusted_manifest)
    config["collection"]["min_learner_decisions_per_update"] = 1
    common.write_json(out / "cases.json", specs)
    rows = []
    with (
        ResourceMonitor(out, interval=0.25) as monitor,
        Collector(config, factory, "cuda", out / "errors") as collector,
    ):
        monitor.mark("warmup")
        collector.collect(
            policy,
            stage,
            0,
            fixed_specs=specs[: config["collection"]["num_envs"]],
            trusted_manifest=trusted_manifest,
        )
        for repeat in range(args.repetitions):
            monitor.mark("collection_measured", repetition=repeat)
            torch.cuda.synchronize()
            start = perf_counter()
            episodes, _, timings = collector.collect(
                policy, stage, 0, fixed_specs=specs, trusted_manifest=trusted_manifest
            )
            torch.cuda.synchronize()
            seconds = perf_counter() - start
            monitor.mark("equivalence_check")
            commands = [ep["joint_commands"] for ep in episodes]
            common.write_json(out / f"commands-{repeat}.json", commands)
            cache_equal = None
            if config.get("performance", {}).get("reuse_encoded"):
                import numpy as np

                for ep in episodes:
                    raw = encode_episode(ep, "C", reuse_cached=False)
                    for a, b in zip(raw, ep["encoded"]):
                        for key in vars(a):
                            np.testing.assert_array_equal(
                                getattr(a, key), getattr(b, key)
                            )
                cache_equal = True
            decisions = sum(len(e["chosen"]) for e in episodes)
            row = {
                "repetition": repeat,
                "decisions": decisions,
                "seconds": seconds,
                "decisions_per_second": decisions / seconds,
                "timings": timings,
                "cache_equals_raw": cache_equal,
            }
            rows.append(row)
            common.write_json(
                out / "summary.json",
                {
                    "repetitions": rows,
                    "decisions_per_second": sum(x["decisions"] for x in rows)
                    / sum(x["seconds"] for x in rows),
                },
            )
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
