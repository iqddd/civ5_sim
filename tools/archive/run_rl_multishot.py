"""Bounded D0-M execution, dispatched by the shared diagnostic CLI."""

import gzip
import json
import os
import random
import tempfile
import traceback
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from civ5rl.collector import Collector
from civ5rl.common import configure, read_json, write_json
from civ5rl.control import TrainingControl
from civ5rl.evaluation import save_training_replay
from civ5rl.model import Policy, batch_inputs
from civ5rl.observation import ObservationEncoder
from civ5rl.ppo import schedule, update
from civ5rl.resources import ResourceMonitor
from civ5rl.validation import validate_config

from civ5rl import checkpoint
from civ5rl.protocols.diagnostic_d1 import load_weights, model_hash
from civ5rl.protocols.multishot import (
    STAGE,
    InvalidDataset,
    control_episode,
    factory,
    prepare,
    sha,
    specification,
)
from civ5rl.protocols.multishot_metrics import comparison, gate, summarize, transition
from civ5rl.provenance import bound_cases, validate_cases


@contextmanager
def isolated_rng():
    saved = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )
    try:
        yield
    finally:
        random.setstate(saved[0])
        np.random.set_state(saved[1])
        torch.set_rng_state(saved[2])
        if saved[3]:
            torch.cuda.set_rng_state_all(saved[3])


def save_rollout(path, episodes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wb", compresslevel=1) as stream:
        torch.save(episodes, stream)
    os.replace(tmp, path)
    return sha(path)


def load_rollout(path, expected_hash):
    if sha(path) != expected_hash:
        raise ValueError("Saved multishot rollout hash mismatch")
    with gzip.open(path, "rb") as stream:
        return torch.load(stream, weights_only=False, map_location="cpu")


def persist_evaluation(directory, episodes, **metadata):
    directory = Path(directory)
    summary, rows = summarize(episodes)
    selected = {}
    for reward, label in ((1, "win"), (0, "draw"), (-1, "loss")):
        matching = [ep for ep in episodes if ep["result"]["reward"] == reward]
        selected[label] = (
            min(matching, key=lambda ep: ep["spec"]["episode_id"]) if matching else None
        )
    wins = [ep for ep in episodes if ep["result"]["reward"] == 1]
    selected["shortest_win"] = (
        min(wins, key=lambda ep: (len(ep["chosen"]), ep["spec"]["episode_id"]))
        if wins
        else None
    )
    selected["longest_win"] = (
        min(wins, key=lambda ep: (-len(ep["chosen"]), ep["spec"]["episode_id"]))
        if wins
        else None
    )
    for label, ep in selected.items():
        if ep is not None:
            save_training_replay(ep, directory / "replays" / label)
    summary.update(
        metadata,
        saved_episode_ids={
            k: v["spec"]["episode_id"] if v else None for k, v in selected.items()
        },
    )
    summary["episodes_sha256"] = save_rollout(directory / "episodes.pt.gz", episodes)
    write_json(directory / "outcomes.json", rows)
    write_json(directory / "summary.json", summary)  # completion marker last
    return summary, rows


def evaluate(policy, collector, dataset, directory, monitor, control=None):
    directory = Path(directory)
    manifest = dataset[STAGE]["dev"]
    identity = "control:" + control if control else model_hash(policy.state_dict())
    if (directory / "summary.json").exists():
        summary, rows = (
            read_json(directory / "summary.json"),
            read_json(directory / "outcomes.json"),
        )
        keys = {(s["layout_id"], s["learner_side"]) for s in manifest["cases"]}
        if (
            summary["manifest_hash"] != manifest["hash"]
            or summary["policy_hash"] != identity
            or len(rows) != 128
            or {(r["layout_id"], r["side"]) for r in rows} != keys
            or sha(directory / "episodes.pt.gz") != summary["episodes_sha256"]
        ):
            raise ValueError(
                "Cached multishot evaluation identity/completeness mismatch"
            )
        return summary, rows
    with isolated_rng():
        start = perf_counter()
        monitor.mark("multishot_evaluation")
        cases = bound_cases(manifest)
        validate_cases(cases, manifest)
        if control:
            episodes = [control_episode(s, passive=control == "passive") for s in cases]
        else:
            episodes, _, _ = collector.collect(
                policy, None, 0, fixed_specs=cases, trusted_manifest=manifest
            )
        if len(episodes) != 128:
            raise ValueError("Incomplete multishot evaluation")
        summary, rows = persist_evaluation(
            directory,
            episodes,
            seconds=0,
            manifest_hash=manifest["hash"],
            policy_hash=identity,
            kind=control or "policy",
        )
        summary["seconds"] = perf_counter() - start
        write_json(directory / "summary.json", summary)
        return summary, rows


def recover_metrics(directory, state):
    path = Path(directory) / "metrics.jsonl"
    rows = []
    lines = path.read_text().splitlines() if path.exists() else []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise ValueError("Corrupt multishot metrics before last line") from None
    if state.get("last_metrics"):
        rows = [r for r in rows if r["update"] != state["last_metrics"]["update"]]
        rows.append(state["last_metrics"])
    if len({r["update"] for r in rows}) != len(rows):
        raise ValueError("Duplicate multishot metric updates")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        "".join(json.dumps(r, allow_nan=False) + "\n" for r in rows), encoding="utf-8"
    )
    os.replace(tmp, path)


def warmup(policy, dataset, device):
    if not device.startswith("cuda"):
        return 0.0
    from civ5rl.core.context import context_from_spec
    from civ5rl.core.env import DecisionEnv

    start = perf_counter()
    with isolated_rng(), DecisionEnv(bound_cases(dataset[STAGE]["dev"])[0]) as env:
        state = env.request()
        encoder = ObservationEncoder(context_from_spec(env.spec), "C")
        encoded = encoder.encode(state["observation"], state["actions"])
        logits, values, _ = policy.forward_sequences(
            batch_inputs([encoded] * 8, device), [1] * 8
        )
        (logits.logsumexp(-1).mean() + values.mean()).backward()
        policy.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
    return perf_counter() - start


def run_seed(config, dataset, directory, device="cuda", resume=False):
    validate_config(config)
    directory = Path(directory)
    if (directory / "latest.json").exists() and not resume:
        raise ValueError("Multishot run exists; use --resume")
    configure(17, device)
    with TrainingControl(directory) as control:
        if (directory / "passport.json").exists():
            identity = read_json(directory / "passport.json")
            with tempfile.TemporaryDirectory() as tmp:
                current = checkpoint.passport(tmp, config, dataset)
            if not checkpoint.same_execution_identity(identity, current):
                raise ValueError("Multishot resume source/environment mismatch")
        else:
            identity = checkpoint.passport(directory, config, dataset)
        policy = Policy("C").to(device)
        initialization = load_weights(policy, config)
        policy.enable_amp(device.startswith("cuda"), spatial_batch_size=32)
        if device.startswith("cuda"):
            policy.enable_compile()
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=0.0003,
            betas=tuple(config["optimizer"]["betas"]),
            eps=config["optimizer"]["eps"],
            weight_decay=config["optimizer"]["weight_decay"],
            fused=device.startswith("cuda"),
        )
        shuffle = random.Random(18)
        state = {
            "update": 0,
            "decisions": 0,
            "multishot_decisions": 0,
            "next_episode": 0,
            "consecutive": 0,
            "previous_decisions": 49152,
            "training_seconds": 0.0,
            "evaluation_seconds": 0.0,
            "warmup_seconds": 0.0,
            "pause_seconds": 0.0,
            "status": "running",
            "pending_rollout": None,
            "pending_evaluation": None,
        }
        if (directory / "latest.json").exists():
            state = checkpoint.restore(
                directory
                / "checkpoints"
                / read_json(directory / "latest.json")["checkpoint"],
                policy,
                optimizer,
                shuffle,
                config,
                identity,
            )
            if state.get("paused_at_utc"):
                state["pause_seconds"] += max(
                    0,
                    (
                        datetime.now(UTC)
                        - datetime.fromisoformat(state.pop("paused_at_utc"))
                    ).total_seconds(),
                )
        else:
            initialization.update(
                new_optimizer_state_entries=len(optimizer.state),
                training_seed=17,
                counters_reset=True,
                episode_memory_initial_state="zero",
            )
            write_json(directory / "initialization-verification.json", initialization)

        def save(name):
            checkpoint.save(
                directory / "checkpoints" / name,
                policy,
                optimizer,
                deepcopy(state),
                shuffle,
                config,
                identity,
            )
            write_json(directory / "latest.json", {"checkpoint": name, **state})

        def timed_save(name):
            tick = perf_counter()
            save(name)
            state["training_seconds"] += perf_counter() - tick
            if name.startswith("update-") and state.get("last_metrics"):
                state["last_metrics"]["training_seconds"] = state["training_seconds"]
            save(name)

        if not (directory / "latest.json").exists():
            save("initial.pt")
        recover_metrics(directory, state)
        write_json(
            directory / "execution.json",
            {
                "kind": "diagnostic",
                "suite_directory": str(directory.parent.resolve()),
                "budget": None,
                "device": device,
                "variant": "C",
            },
        )
        try:
            with (
                ResourceMonitor(directory) as monitor,
                Collector(
                    config, factory(dataset), device, directory / "errors"
                ) as collector,
            ):
                if state["update"] == 0 and not state["pending_rollout"]:
                    state["warmup_seconds"] += warmup(policy, dataset, device)
                original = deepcopy(policy.state_dict())
                with isolated_rng():
                    load_weights(policy, config, random_baseline=True)
                    random_base = evaluate(
                        policy,
                        collector,
                        dataset,
                        directory / "evaluation/random",
                        monitor,
                    )
                    policy.load_state_dict(
                        torch.load(
                            directory / "checkpoints/initial.pt",
                            map_location=device,
                            weights_only=False,
                        )["model"]
                    )
                    transfer = evaluate(
                        policy,
                        collector,
                        dataset,
                        directory / "evaluation/transfer",
                        monitor,
                    )
                    controls = [
                        evaluate(
                            policy,
                            collector,
                            dataset,
                            directory / ("evaluation/" + label),
                            monitor,
                            control=label,
                        )
                        for label in ("reference", "passive")
                    ]
                policy.load_state_dict(original)
                del original
                state["baseline_evaluation_seconds"] = sum(
                    x[0]["seconds"] for x in [random_base, transfer, *controls]
                )
                if (
                    state["update"] == 0
                    and not state["pending_rollout"]
                    and gate(transfer[0])
                ):
                    state.update(
                        status="solved_at_transfer", reason="transfer_at_least_122_wins"
                    )
                    save("initial.pt")
                while state["status"] == "running":
                    if state["pending_evaluation"] is not None:
                        control.mark("evaluation", update=state["update"])
                        summary, rows = evaluate(
                            policy,
                            collector,
                            dataset,
                            directory / f"evaluation/update-{state['update']:06}",
                            monitor,
                        )
                        summary.update(
                            gate_passed=gate(summary),
                            comparison_to_random=comparison(rows, random_base[1]),
                            comparison_to_transfer=comparison(rows, transfer[1]),
                        )
                        write_json(
                            directory
                            / f"evaluation/update-{state['update']:06}/summary.json",
                            summary,
                        )
                        state["evaluation_seconds"] += summary["seconds"]
                        state = transition(state, summary["gate_passed"])
                        state["pending_evaluation"] = None
                        save(f"update-{state['update']:06}.pt")
                        print(
                            json.dumps(
                                {
                                    "update": state["update"],
                                    "wins": summary["wins"],
                                    "status": state["status"],
                                }
                            ),
                            flush=True,
                        )
                        continue
                    if state["pending_rollout"] is None:
                        if control.stopping():
                            break
                        if state["update"] >= 8:
                            state.update(
                                status="failed_update_limit",
                                reason="eight_updates_completed",
                            )
                        elif state["training_seconds"] >= 1800:
                            state.update(
                                status="failed_training_time_limit",
                                reason="training_time_limit",
                            )
                        if state["status"] != "running":
                            save(read_json(directory / "latest.json")["checkpoint"])
                            break
                        control.mark("collection", update=state["update"] + 1)
                        monitor.mark("collection", update=state["update"] + 1)
                        start = perf_counter()
                        snapshot = {
                            "decisions": state["decisions"],
                            "lr": schedule(
                                config["optimizer"]["lr_schedule"], state["decisions"]
                            ),
                            "entropy_coefficient": schedule(
                                config["ppo"]["entropy_schedule"], state["decisions"]
                            ),
                        }
                        episodes, next_episode, timings = collector.collect(
                            policy, STAGE, state["next_episode"], monitor
                        )
                        path = (
                            directory
                            / f"rollouts/rollout-{state['update'] + 1:06}.pt.gz"
                        )
                        rollout_hash = save_rollout(path, episodes)
                        count = sum(len(ep["chosen"]) for ep in episodes)
                        state.update(
                            decisions=state["decisions"] + count,
                            multishot_decisions=state["decisions"] + count,
                            next_episode=next_episode,
                            pending_rollout={
                                "path": str(path.relative_to(directory)),
                                "sha256": rollout_hash,
                                "schedule": snapshot,
                                "timings": timings,
                                "decisions": count,
                            },
                        )
                        state["training_seconds"] += perf_counter() - start
                        timed_save(f"collected-{state['update'] + 1:06}.pt")
                        del episodes
                    pending = state["pending_rollout"]
                    start = perf_counter()
                    episodes = load_rollout(
                        directory / pending["path"], pending["sha256"]
                    )
                    outcomes, _ = summarize(episodes)
                    before = deepcopy(policy.state_dict())
                    if state["update"] == 0 and outcomes["wins"] == 0:
                        from civ5rl.ppo_fast import BatchedBackend

                        backend = BatchedBackend(
                            policy, episodes, config, device, monitor
                        )
                        backend.precheck()
                        persist_evaluation(
                            directory / "initial-rollout",
                            episodes,
                            kind="training_rollout",
                            behavior_replay=backend.replay,
                        )
                        state.update(
                            status="no_positive_reward_in_initial_rollout",
                            reason="zero_wins_before_first_Adam_step",
                        )
                        state["training_seconds"] += perf_counter() - start
                        timed_save("collected-000001.pt")
                        break
                    control.mark("ppo", update=state["update"] + 1)
                    monitor.mark("ppo", update=state["update"] + 1)
                    stats = update(
                        policy,
                        optimizer,
                        episodes,
                        config,
                        pending["schedule"]["decisions"],
                        device,
                        shuffle,
                        monitor,
                    )
                    if any(
                        stats[k] != pending["schedule"][k]
                        for k in ("lr", "entropy_coefficient")
                    ):
                        raise ValueError("Multishot schedules changed within update")
                    delta = (
                        sum(
                            (v - before[k]).double().square().sum()
                            for k, v in policy.state_dict().items()
                        )
                        .sqrt()
                        .item()
                    )
                    training_dir = (
                        directory / f"training/update-{state['update'] + 1:06}"
                    )
                    persist_evaluation(training_dir, episodes, kind="training_rollout")
                    state["update"] += 1
                    state["training_seconds"] += perf_counter() - start
                    state["last_metrics"] = {
                        "update": state["update"],
                        "decisions": state["decisions"],
                        "rollout_decisions": pending["decisions"],
                        "overflow": pending["decisions"] - 8192,
                        "episodes": len(episodes),
                        "outcomes": outcomes,
                        "schedule": pending["schedule"],
                        "timings": pending["timings"],
                        "ppo": stats,
                        "weight_delta_l2": delta,
                        "training_seconds": state["training_seconds"],
                    }
                    state.update(
                        pending_rollout=None, pending_evaluation=state["update"]
                    )
                    timed_save(f"update-{state['update']:06}.pt")
                    recover_metrics(directory, state)
                    del episodes, before
                if state["status"] == "running":
                    state["paused_at_utc"] = datetime.now(UTC).isoformat()
                    save(read_json(directory / "latest.json")["checkpoint"])
                result = {
                    **state,
                    "status": "paused"
                    if state["status"] == "running"
                    else state["status"],
                    "seed": 17,
                }
                write_json(directory / "run-result.json", result)
                control.mark(
                    result["status"],
                    status=result["status"],
                    update=state["update"],
                    decisions=state["decisions"],
                )
                return result
        except BaseException:
            write_json(
                directory / "run-result.json",
                {
                    **state,
                    "status": "technical_error",
                    "reason": "exception",
                    "seed": 17,
                },
            )
            write_json(
                directory / "failure.json", {"traceback": traceback.format_exc()}
            )
            raise


def run_suite(config, directory, device="cuda", resume=False, prepare_only=False):
    directory = Path(directory)
    validate_config(config)
    try:
        specification(config, full_config=True)
        start = perf_counter()
        dataset = prepare(directory / "manifests", config)
        write_json(
            directory / "preparation.json",
            {"seconds_this_invocation": perf_counter() - start},
        )
        if prepare_only:
            return
        if resume:
            (directory / "seed-17/stop.request").unlink(missing_ok=True)
        result = run_seed(config, dataset, directory / "seed-17", device, resume)
        write_json(
            directory / "suite-result.json",
            {
                "status": result["status"],
                "reason": result.get("reason"),
                "seeds": [17],
                "closed_test_evaluated": False,
            },
        )
    except InvalidDataset as exc:
        write_json(
            directory / "suite-result.json",
            {
                "status": "invalid_dataset",
                "reason": str(exc),
                "closed_test_evaluated": False,
            },
        )
        raise
    except BaseException:
        write_json(
            directory / "suite-result.json",
            {
                "status": "technical_error",
                "reason": traceback.format_exc(),
                "closed_test_evaluated": False,
            },
        )
        raise
    finally:
        from tools.archive.report_rl_multishot import report

        report(directory)
