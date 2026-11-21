"""Compare discarded PPO updates against a reusable, exact baseline."""

from __future__ import annotations

import copy
import hashlib
import pickle
import random
import statistics
import subprocess
from time import perf_counter

import numpy as np
import torch

from civ5rl.engine.ppo.adam import grouped_optimizer, optimizer_for, uses_grouped_adam
from civ5rl.engine.ppo.backend import BatchedBackend
from civ5rl.engine.ppo.core import encode_episode, update
from civ5rl.engine.ppo.storage import load_rollout
from civ5rl.engine.runtime.telemetry import CompilationClock
from civ5rl.experiments.imitation.helpers import load_policy
from civ5rl.provenance.common import ROOT, configure, file_sha256, read_json, write_json
from civ5rl.provenance.rng import capture_rng, restore_rng


def _fingerprint(value):
    """Hash tensor bytes (including BF16 and infinities) and their structure."""
    digest = hashlib.sha256()

    def feed(item):
        if isinstance(item, torch.Tensor):
            data = item.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy()
            digest.update(pickle.dumps((str(item.dtype), tuple(item.shape))))
            digest.update(data)
        elif isinstance(item, np.ndarray):
            digest.update(pickle.dumps((item.dtype.str, item.shape)))
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            digest.update(pickle.dumps(sorted(item)))
            for key in sorted(item):
                feed(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(pickle.dumps((type(item).__name__, len(item))))
            for child in item:
                feed(child)
        else:
            digest.update(pickle.dumps(item))

    feed(value)
    return digest.hexdigest()


def _metrics(result):
    fields = (
        "policy_loss",
        "value_loss",
        "entropy",
        "normalized_entropy",
        "clip_fraction",
        "lr",
        "entropy_coefficient",
        "kl",
        "pre_update_full_kl",
        "pre_update_max_total_variation",
        "attempted_optimizer_steps",
        "accepted_optimizer_steps",
        "rejected_optimizer_steps",
        "final_accepted_kl",
        "objective_valid_tokens",
        "stop_reason",
    )
    return {
        **{key: result[key] for key in fields if key in result},
        "steps": [
            {k: v for k, v in row.items() if k != "seconds"} for row in result["optimizer_step_records"]
        ],
    }


def _differences(reference, current, prefix=""):
    differences = []
    for key in sorted(reference.keys() | current.keys()):
        label = prefix + key
        if key not in reference or key not in current:
            differences.append(label)
        elif isinstance(reference[key], dict) and isinstance(current[key], dict):
            differences.extend(_differences(reference[key], current[key], label + "."))
        elif reference[key] != current[key]:
            before, after = reference[key], current[key]
            if isinstance(before, list) and isinstance(after, list):
                first = next(
                    (i for i, pair in enumerate(zip(before, after)) if pair[0] != pair[1]),
                    min(len(before), len(after)),
                )
                differences.append(f"{label}[{first}]")
            else:
                differences.append(label)
    return differences


def _preflight(args, parser, config, inputs):
    out = args.out.resolve()
    if any(p == out or p.is_relative_to(out) for p in inputs):
        parser.error("output directory must not contain an input")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error("ppo-update requires an empty output directory")
    if args.minibatch_tokens is not None and args.minibatch_tokens < 1:
        parser.error("--minibatch-tokens must be positive")
    if args.check_rollback and config["ppo"].get("step_acceptance") != "full_categorical_kl_rollback":
        parser.error("--check-rollback requires full_categorical_kl_rollback")

    protected = dict(zip(("config", "checkpoint", "rollout"), map(file_sha256, inputs)))
    cuda = torch.device(args.device).type == "cuda"
    configure(args.seed, args.device)
    conditions = {
        "input_sha256": protected,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(args.device) if cuda else None,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "seed": args.seed,
        "episode_limit": args.episode_limit,
        "minibatch_tokens": args.minibatch_tokens,
    }
    baseline = read_json(args.baseline_result) if args.baseline_result else None
    if baseline is not None:
        if baseline.get("schema") != 1 or baseline.get("conditions") != conditions:
            parser.error("baseline format or benchmark conditions differ")
        if baseline.get("differences") or not baseline.get("timing"):
            parser.error("baseline needs a successful PPO update benchmark")
        if args.check_rollback and "rollback" not in baseline["checks"]:
            parser.error("baseline needs completed timing and the requested checks")

    return out, protected, conditions, baseline


class _Trial:
    def __init__(self, args, parser, config, inputs):
        self.device = args.device
        self.on_cuda = torch.device(self.device).type == "cuda"
        self.saved = torch.load(inputs[1], map_location="cpu", weights_only=False)
        if "optimizer" not in self.saved:
            parser.error("ppo-update needs a PPO boundary checkpoint containing Adam")
        self.decisions = self.saved.get("state", {}).get("decisions", 0)
        self.episodes = load_rollout(inputs[2])[: args.episode_limit]
        if not self.episodes:
            parser.error("ppo-update needs a nonempty rollout")
        self.policy = load_policy(config, inputs[1], args.device).train()
        self.policy.requires_grad_(True)
        make_optimizer = grouped_optimizer if uses_grouped_adam(config) else optimizer_for
        self.optimizer = make_optimizer(self.policy, config, args.device)
        self.shuffle = random.Random(args.seed)
        self.rng = capture_rng(self.shuffle)
        # Match the live collector's encoded histories and reuse them between trials.
        for episode in self.episodes:
            episode["encoded"] = encode_episode(episode, self.policy.variant)
            episode["encoded_variant"] = self.policy.variant

    def sync(self):
        if self.on_cuda:
            torch.cuda.synchronize(self.device)

    def reset(self):
        self.policy.load_state_dict(self.saved["model"])
        self.optimizer.load_state_dict(copy.deepcopy(self.saved["optimizer"]))
        self.optimizer.zero_grad(set_to_none=True)
        restore_rng(self.rng, self.shuffle)
        self.sync()

    def run(self, config, **kwargs):
        return update(
            self.policy,
            self.optimizer,
            self.episodes,
            config,
            self.decisions,
            self.device,
            self.shuffle,
            **kwargs,
        )


def _parity(trial, config, *, rollback=False):
    trial.reset()
    values = {name: [] for name in ("train_logits", "replay_logits", "gradients", "adam_after_step")}

    def record_logits(logits, *, training):
        category = "train_logits" if training else "replay_logits"
        values[category].append(_fingerprint(logits))

    class TracedBackend(BatchedBackend):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, sequence_observer=record_logits, **kwargs)

        def backward_minibatch(self, *args, **kwargs):
            result = super().backward_minibatch(*args, **kwargs)
            values["gradients"].append(
                _fingerprint([parameter.grad for parameter in trial.policy.parameters()])
            )
            return result

    def record_adam(optimizer, *_):
        values["adam_after_step"].append(_fingerprint(optimizer.state_dict()))

    hook = trial.optimizer.register_step_post_hook(record_adam)
    try:
        result = trial.run(config, _backend=TracedBackend)
    finally:
        hook.remove()
    values.update(
        {
            "model": _fingerprint(trial.policy.state_dict()),
            "adam": _fingerprint(trial.optimizer.state_dict()),
            "rng": _fingerprint(capture_rng(trial.shuffle)),
            "metrics": _metrics(result),
        }
    )
    if rollback:
        expected_adam = copy.deepcopy(trial.saved["optimizer"])
        for group in expected_adam["param_groups"]:
            group["lr"] = result["lr"]
        values["restored"] = (
            result["rejected_optimizer_steps"] == 1
            and result["stop_reason"] == "rejected_full_kl"
            and values["model"] == _fingerprint(trial.saved["model"])
            and values["adam"] == _fingerprint(expected_adam)
        )
    return values


def _verify_trials(trial, config, args, baseline):
    decisions = sum(len(episode["chosen"]) for episode in trial.episodes)
    print(f"Warming {decisions} saved decisions", flush=True)
    for _ in range(args.warmup):
        trial.reset()
        trial.run(config)
    checks = {"update": _parity(trial, config)}
    problems = []
    if not checks["update"]["metrics"]["accepted_optimizer_steps"]:
        problems.append("update.no_accepted_steps")
    if args.check_rollback:
        rollback_config = copy.deepcopy(config)
        rollback_config["ppo"].update({"stop_kl": 0.0, "minibatch_kl_spike_threshold": -1.0})
        checks["rollback"] = _parity(trial, rollback_config, rollback=True)
        if not checks["rollback"]["restored"]:
            problems.append("rollback.restore")
    if baseline is not None:
        expected = {name: baseline["checks"][name] for name in checks}
        problems.extend(_differences(expected, checks))
    return checks, problems


def _measure_trials(trial, config, args, baseline, expected, problems):
    measurements = []
    for _ in range(args.repeats):
        trial.reset()
        if trial.on_cuda:
            torch.cuda.reset_peak_memory_stats(trial.device)
        with CompilationClock() as compilation:
            started = perf_counter()
            result = trial.run(config)
            trial.sync()
            seconds = perf_counter() - started
        if _metrics(result) != expected["metrics"]:
            problems.append("timed_update.metrics")
            break
        measurements.append(
            {
                "seconds": seconds,
                "compilations": len(compilation.intervals),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(trial.device)
                if trial.on_cuda
                else None,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(trial.device)
                if trial.on_cuda
                else None,
            }
        )
    if not problems:
        for name, state in (("model", trial.policy.state_dict()), ("adam", trial.optimizer.state_dict())):
            if _fingerprint(state) != expected[name]:
                problems.append(f"timed_update.{name}")
    steady = [row["seconds"] for row in measurements if not row["compilations"]]
    timing = {"trials": measurements, "median_seconds": statistics.median(steady) if steady else None}
    if not problems and baseline and timing["median_seconds"] and baseline["timing"]["median_seconds"]:
        timing["relative_to_baseline"] = timing["median_seconds"] / baseline["timing"]["median_seconds"]
    return timing


def run_benchmark(args, parser, config, inputs):
    out, protected, conditions, baseline = _preflight(args, parser, config, inputs)
    config = copy.deepcopy(config)
    config["ppo"]["epochs"] = 1
    if args.minibatch_tokens is not None:
        config["ppo"]["valid_tokens_per_minibatch"] = args.minibatch_tokens
    trial = _Trial(args, parser, config, inputs)
    checks, problems = _verify_trials(trial, config, args, baseline)
    timing = (
        _measure_trials(trial, config, args, baseline, checks["update"], problems) if not problems else None
    )

    if list(protected.values()) != list(map(file_sha256, inputs)):
        raise RuntimeError("PPO benchmark input changed")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    execution_diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD", "--", "civ5rl", "civ5sim", "civ5map"], cwd=ROOT
    )
    write_json(
        out / "benchmark.json",
        {
            "schema": 1,
            "conditions": conditions,
            "git_commit": commit,
            "working_diff_sha256": hashlib.sha256(execution_diff).hexdigest(),
            "baseline": str(args.baseline_result.resolve()) if baseline else None,
            "checks": checks,
            "differences": problems,
            "timing": timing,
        },
    )
    print({"differences": problems, "timing": timing, "result": str(out / "benchmark.json")})
    return not problems
