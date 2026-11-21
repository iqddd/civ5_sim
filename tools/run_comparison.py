"""Run bounded rollout, whole-turn, or action-value comparison workflows."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

RUNTIME_FLAGS = (
    "workers",
    "source_share",
    "buffer_limit",
    "pack_mib_per_second",
    "prefix_states",
)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("prepare", "collect", "train", "evaluate", "resume", "precheck"),
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/rl-c2p-rollout-comparison.json")
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--history-microbatch",
        type=int,
        help="Whole-turn fit: scored histories per GPU microbatch",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        help="Whole-turn fit: compile CUDA operators",
    )
    runtime = parser.add_argument_group(
        "collection-only runtime (rl01.rollout_comparison_collection)"
    )
    runtime.add_argument(
        "--runtime-config", type=Path, help="recommended-runtime.json from bench_rl.py"
    )
    runtime.add_argument("--workers", type=int)
    runtime.add_argument("--source-share", type=float)
    runtime.add_argument("--buffer-limit", type=int)
    runtime.add_argument("--pack-mib-per-second", type=float)
    runtime.add_argument("--prefix-states", action=argparse.BooleanOptionalAction)
    return parser


def _check_output(parser, args, cfg):
    if args.out is None or args.out.resolve() == (ROOT / cfg["output"]).resolve():
        parser.error("precheck requires a separate --out")


def _smoke(config, out, check, *, require_cuda=False, **options):
    from civ5rl.engine.runtime.control import RunLock
    from civ5rl.provenance.common import configure

    configure(config["seed"], "cuda")
    if require_cuda:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Action-value fitting requires CUDA")
    with RunLock(out):
        return check(config, out, **options)


def _launch(
    parser, args, module, started, *, check=None, check_requires_cuda=False, **options
):
    from civ5rl.provenance.execution import execution_session
    from civ5rl.reporting.progress_process import ProcessProgress

    cfg = module.load_config(args.config)
    if args.phase == "precheck":
        _check_output(parser, args, cfg)
    out = (args.out or ROOT / cfg["output"]).resolve()
    execution_session(out, args.phase)
    with ProcessProgress(out, cfg):
        if args.phase == "precheck":
            return _smoke(cfg, out, check, require_cuda=check_requires_cuda, **options)
        if started is not None:
            options["preparation_seconds"] = perf_counter() - started
        return module.run(args.phase, cfg, out, **options)


def main():
    parser = _parser()
    args = parser.parse_args()
    started = perf_counter()
    kind = json.loads(args.config.read_text(encoding="utf-8")).get("kind")
    runtime_options = {
        name: getattr(args, name)
        for name in RUNTIME_FLAGS
        if getattr(args, name) is not None
    }
    has_runtime_options = args.runtime_config is not None or bool(runtime_options)
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be positive")

    if kind == "rl01.turn_comparison_fit":
        if args.phase == "collect":
            parser.error("whole-turn fitting uses existing continuations only")
        if has_runtime_options:
            parser.error("collection options do not apply to whole-turn fitting")
        if args.history_microbatch is not None and args.history_microbatch < 1:
            parser.error("history microbatch must be positive")
        from civ5rl.experiments.checks.turn_fit import smoke
        from civ5rl.experiments.comparison import turn_fit

        return _launch(
            parser,
            args,
            turn_fit,
            started,
            check=smoke,
            microbatch=args.history_microbatch or 2,
            compile_model=args.compile is not False,
        )
    if args.history_microbatch is not None or args.compile is not None:
        parser.error(
            "history microbatch and compile options apply only to whole-turn fitting"
        )

    if kind == "rl01.turn_comparison_collection":
        if args.phase not in ("prepare", "precheck", "collect", "resume"):
            parser.error("whole-turn protocol is collection-only")
        if args.runtime_config is not None or runtime_options.keys() - {"workers"}:
            parser.error("whole-turn runtime accepts --workers only")
        from civ5rl.experiments.checks.turn_collection import smoke
        from civ5rl.experiments.comparison import turn_collection

        return _launch(
            parser, args, turn_collection, None, check=smoke, workers=args.workers
        )
    if kind == "rl01.action_value":
        if args.phase == "collect":
            parser.error("action-value fitting uses saved outcomes only")
        if has_runtime_options:
            parser.error(
                "collection runtime options do not apply to action-value fitting"
            )
        from civ5rl.experiments.checks.action_value import smoke
        from civ5rl.experiments.comparison import action_value

        return _launch(
            parser, args, action_value, None, check=smoke, check_requires_cuda=True
        )
    if kind == "rl01.rollout_comparison_collection":
        if args.phase not in ("prepare", "collect", "resume"):
            parser.error("collection-only runs support prepare, collect and resume")
        from civ5rl.experiments.comparison import rollout_collection

        runtime = {}
        if args.runtime_config:
            runtime = json.loads(args.runtime_config.read_text(encoding="utf-8"))[
                "runtime"
            ]
        resolved = rollout_collection.CollectionRuntime(
            **{**runtime, **runtime_options}
        ).resolved()
        return _launch(parser, args, rollout_collection, started, runtime=resolved)
    if has_runtime_options:
        parser.error(
            "collection runtime options apply only to rl01.rollout_comparison_collection"
        )
    from civ5rl.experiments.comparison import rollout_fit

    if args.phase == "precheck":
        from civ5rl.experiments.checks.comparison import precheck

        cfg = rollout_fit.load_config(args.config)
        _check_output(parser, args, cfg)
        return precheck(cfg, args.out)
    return _launch(parser, args, rollout_fit, started)


if __name__ == "__main__":
    main()
