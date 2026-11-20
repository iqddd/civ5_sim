"""Compare discarded PPO updates against a reusable, exact baseline."""

import copy
import hashlib
import pickle
import random
import statistics
import subprocess
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch

from civ5rl.engine.ppo.adam import grouped_optimizer, uses_grouped_adam
from civ5rl.engine.ppo.backend import BatchedBackend
from civ5rl.engine.ppo.core import encode_episode, update
from civ5rl.experiments.imitation_helpers import CompilationClock, load_policy
from civ5rl.experiments.ppo_after_imitation import optimizer_for
from civ5rl.experiments.ppo_artifacts import load_rollout
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
        "policy_loss", "value_loss", "entropy", "normalized_entropy", "clip_fraction",
        "lr", "entropy_coefficient", "kl", "pre_update_full_kl",
        "pre_update_max_total_variation", "attempted_optimizer_steps",
        "accepted_optimizer_steps", "rejected_optimizer_steps",
        "final_accepted_kl", "objective_valid_tokens", "stop_reason",
    )
    return {
        **{key: result[key] for key in fields if key in result},
        "steps": [
            {k: v for k, v in row.items() if k != "seconds"}
            for row in result["optimizer_step_records"]
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


def run_benchmark(args, parser, config, inputs):
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

    config = copy.deepcopy(config)
    config["ppo"]["epochs"] = 1
    if args.minibatch_tokens is not None:
        config["ppo"]["valid_tokens_per_minibatch"] = args.minibatch_tokens
    saved = torch.load(inputs[1], map_location="cpu", weights_only=False)
    if "optimizer" not in saved:
        parser.error("ppo-update needs a PPO boundary checkpoint containing Adam")
    decisions = saved.get("state", {}).get("decisions", 0)
    episodes = load_rollout(inputs[2])[:args.episode_limit]
    if not episodes:
        parser.error("ppo-update needs a nonempty rollout")
    policy = load_policy(config, inputs[1], args.device).train()
    for parameter in policy.parameters():
        parameter.requires_grad_(True)
    make_optimizer = grouped_optimizer if uses_grouped_adam(config) else optimizer_for
    optimizer = make_optimizer(policy, config, args.device)
    shuffle = random.Random(args.seed)
    rng = capture_rng(shuffle)
    # Match the live collector's encoded histories and reuse them between trials.
    for episode in episodes:
        episode["encoded"] = encode_episode(episode, policy.variant)
        episode["encoded_variant"] = policy.variant

    def sync():
        if cuda:
            torch.cuda.synchronize(args.device)

    def reset():
        policy.load_state_dict(saved["model"])
        optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
        optimizer.zero_grad(set_to_none=True)
        restore_rng(rng, shuffle)
        sync()

    def run(trial_config, **kwargs):
        return update(policy, optimizer, episodes, trial_config, decisions, args.device, shuffle, **kwargs)

    def parity(trial_config, *, rollback=False):
        reset()
        values = {k: [] for k in ("train_logits", "replay_logits", "gradients", "adam_after_step")}
        forward = policy.forward_sequences

        def forward_with_logits(*args, **kwargs):
            output = forward(*args, **kwargs)
            if not kwargs.get("memory_only"):
                key = "train_logits" if torch.is_grad_enabled() else "replay_logits"
                values[key].append(_fingerprint(output[0]))
            return output

        class TracedBackend(BatchedBackend):
            def backward_minibatch(self, *args, **kwargs):
                output = super().backward_minibatch(*args, **kwargs)
                values["gradients"].append(_fingerprint([p.grad for p in policy.parameters()]))
                return output

        def record_adam(*_):
            values["adam_after_step"].append(_fingerprint(optimizer.state_dict()))

        with patch.object(policy, "forward_sequences", forward_with_logits):
            hook = optimizer.register_step_post_hook(record_adam)
            try:
                result = run(trial_config, _backend=TracedBackend)
            finally:
                hook.remove()
        values.update(
            model=_fingerprint(policy.state_dict()),
            adam=_fingerprint(optimizer.state_dict()),
            rng=_fingerprint(capture_rng(shuffle)),
            metrics=_metrics(result),
        )
        if rollback:
            expected_adam = copy.deepcopy(saved["optimizer"])
            for group in expected_adam["param_groups"]:
                group["lr"] = result["lr"]
            values["restored"] = (
                result["rejected_optimizer_steps"] == 1
                and result["stop_reason"] == "rejected_full_kl"
                and values["model"] == _fingerprint(saved["model"])
                and values["adam"] == _fingerprint(expected_adam)
            )
        return values

    print(f"Warming {sum(len(episode['chosen']) for episode in episodes)} saved decisions", flush=True)
    for _ in range(args.warmup):
        reset()
        run(config)
    checks = {"update": parity(config)}
    problems = []
    if not checks["update"]["metrics"]["accepted_optimizer_steps"]:
        problems.append("update.no_accepted_steps")
    if args.check_rollback:
        rollback_config = copy.deepcopy(config)
        rollback_config["ppo"].update(stop_kl=0.0, minibatch_kl_spike_threshold=-1.0)
        checks["rollback"] = parity(rollback_config, rollback=True)
        if not checks["rollback"]["restored"]:
            problems.append("rollback.restore")
    if baseline is not None:
        expected = {k: baseline["checks"][k] for k in checks}
        problems.extend(_differences(expected, checks))

    timing = None
    if not problems:
        trials = []
        for _ in range(args.repeats):
            reset()
            if cuda:
                torch.cuda.reset_peak_memory_stats(args.device)
            with CompilationClock() as compilation:
                start = perf_counter()
                result = run(config)
                sync()
                seconds = perf_counter() - start
            if _metrics(result) != checks["update"]["metrics"]:
                problems.append("timed_update.metrics")
                break
            trials.append({
                "seconds": seconds,
                "compilations": len(compilation.intervals),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device) if cuda else None,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(args.device) if cuda else None,
            })
        if not problems:
            for name, state in (("model", policy.state_dict()), ("adam", optimizer.state_dict())):
                if _fingerprint(state) != checks["update"][name]:
                    problems.append(f"timed_update.{name}")
        steady = [row["seconds"] for row in trials if not row["compilations"]]
        timing = {"trials": trials, "median_seconds": statistics.median(steady) if steady else None}
        if not problems and baseline and timing["median_seconds"] and baseline["timing"]["median_seconds"]:
            timing["relative_to_baseline"] = timing["median_seconds"] / baseline["timing"]["median_seconds"]

    if list(protected.values()) != list(map(file_sha256, inputs)):
        raise RuntimeError("PPO benchmark input changed")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    execution_diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD", "--", "civ5rl", "civ5sim", "civ5map"], cwd=ROOT
    )
    write_json(out / "benchmark.json", {
        "schema": 1, "conditions": conditions, "git_commit": commit,
        "working_diff_sha256": hashlib.sha256(execution_diff).hexdigest(),
        "baseline": str(args.baseline_result.resolve()) if baseline else None,
        "checks": checks, "differences": problems, "timing": timing,
    })
    print({"differences": problems, "timing": timing, "result": str(out / "benchmark.json")})
    return not problems
