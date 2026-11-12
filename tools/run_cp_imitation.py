"""Prepare a map corpus or dispatch configured CP imitation phases."""

import argparse
import importlib
import os
import sys
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "prepare-corpus",
            "prepare",
            "benchmark",
            "collect",
            "train",
            "evaluate",
            "report",
            "run",
            "resume",
        ),
    )
    parser.add_argument("--config", default="configs/rl-cp-imitation.json")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cycle", type=int, choices=(1, 2))
    parser.add_argument("--game", type=Path)
    parser.add_argument("--evaluation-device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--evaluation-workers", type=int)
    parser.add_argument("--evaluation-max-batch", type=int)
    parser.add_argument("--evaluation-wait-seconds", type=float)
    args = parser.parse_args(argv)
    from civ5rl.config import load_config

    config = load_config(args.config)
    if args.phase == "prepare-corpus":
        if not args.out or args.game is None:
            parser.error("prepare-corpus requires --out and --game")
        if args.cycle is not None:
            parser.error("--cycle does not apply to prepare-corpus")
        if (
            not isinstance(config.get("corpus"), dict)
            or "parent_worlds" not in config["corpus"]
        ):
            parser.error(
                "prepare-corpus requires corpus.parent_worlds generation settings in --config"
            )
        workers = 4 if args.workers is None else args.workers
        if workers < 1:
            parser.error("workers must be positive")
        from civ5rl.protocols.corpus import prepare_corpus

        return prepare_corpus(config, args.out, args.game, workers)
    if args.game is not None:
        parser.error("--game applies only to prepare-corpus")
    kind = config.get("kind")
    modules = {
        "rl01.cp_imitation": "imitation_base",
        "rl01.cp_imitation_target_scale": "imitation_target_scale",
        "rl01.cp_imitation_target_more_data": "imitation_target_scale",
        "rl01.cp_imitation_target_c2": "imitation_target_scale",
        "rl01.cp_imitation_own_states": "imitation_own_states",
        "rl01.cp_imitation_more_data": "imitation_more_data",
        "rl01.cp_imitation_continuation": "imitation_continuation",
    }
    if kind not in modules:
        parser.error(f"Unsupported imitation kind: {kind!r}")
    if any(v is not None for v in (args.evaluation_device, args.evaluation_workers, args.evaluation_max_batch, args.evaluation_wait_seconds)) and kind not in ("rl01.cp_imitation_target_scale", "rl01.cp_imitation_target_more_data", "rl01.cp_imitation_target_c2"):
        parser.error("evaluation runtime options apply only to target-scale imitation")
    if args.cycle is not None and kind != "rl01.cp_imitation_own_states":
        parser.error("--cycle applies only to own-state imitation")
    if args.phase == "run" and kind not in (
        "rl01.cp_imitation_target_scale",
        "rl01.cp_imitation_target_more_data",
        "rl01.cp_imitation_target_c2",
        "rl01.cp_imitation_own_states",
    ):
        parser.error("run applies only to target-scale or own-state imitation")
    if args.phase == "resume" and kind not in (
        "rl01.cp_imitation_target_scale",
        "rl01.cp_imitation_target_more_data",
        "rl01.cp_imitation_target_c2",
    ):
        parser.error("resume applies only to target-scale imitation")
    out = (args.out or Path(config["output"])).resolve()
    workers = config["workers"] if args.workers is None else args.workers
    if workers < 1:
        parser.error("workers must be positive")
    module = importlib.import_module("civ5rl.experiments." + modules[kind])
    if kind in (
        "rl01.cp_imitation_target_scale",
        "rl01.cp_imitation_target_more_data",
        "rl01.cp_imitation_target_c2",
    ):
        config = module.load_config(args.config)
    from civ5rl.reporting.progress_process import ProcessProgress
    from civ5rl.engine.evaluation import evaluation_options

    progress = ProcessProgress(out, config) if args.phase in ("collect", "train", "run", "resume", "evaluate") else nullcontext()
    with evaluation_options(device=args.evaluation_device, workers=args.evaluation_workers,
                            max_batch=args.evaluation_max_batch, wait_seconds=args.evaluation_wait_seconds), progress:
        return _execute(args, config, kind, module, out, workers)


def _execute(args, config, kind, module, out, workers):
    if kind == "rl01.cp_imitation_own_states":
        return module.run(args.phase, config, out, workers, args.device, args.cycle)
    if kind != "rl01.cp_imitation":
        return module.run(args.phase, config, out, workers, args.device)
    from civ5rl.experiments.report_cp_imitation import report

    if args.phase != "report":
        manifest, identity = module.initialize(config, out)
        if args.phase == "benchmark":
            from tools._common import write_json
            from tools._imitation_benchmark import benchmark, latency

            benchmark(config, manifest, out, workers)
            latency(config, out)
            write_json(
                out / "status.json",
                {
                    "status": "benchmark_complete_stopped",
                    "next": "review_cost_report",
                    "research_training_started": False,
                    "full_evaluation_started": False,
                },
            )
        elif args.phase == "collect":
            module.collect(config, manifest, out, workers)
        elif args.phase == "train":
            module.train(config, manifest, identity, out, args.device)
        elif args.phase == "evaluate":
            module.evaluate(config, manifest, out, workers, args.device)
    return report(config, out)


if __name__ == "__main__":
    main()
