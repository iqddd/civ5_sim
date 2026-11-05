"""Two bounded rounds of CP advice on policy trajectories; no PPO or tuning."""

import gc
import hashlib
import json
import random
import traceback
from collections import Counter
from pathlib import Path
from time import perf_counter

import torch

from civ5rl.engine.imitation import DemonstrationCache, teacher_metrics
from civ5rl.engine.imitation_performance import (
    ResidentDemonstrations,
    backward_minibatch_async,
)
from civ5rl.engine.resources import ResourceMonitor
from civ5rl.engine.sequence import take_counted_span
from civ5rl.engine.training import FrozenParamsCheck, bc_optimizer_step
from civ5rl.experiments.evaluation import (
    compare_paired_directories,
    run_validation_cases,
    verify_replay,
)
from civ5rl.experiments.imitation_helpers import (
    assert_equal,
    collect_summary,
    isolated_rng,
)
from civ5rl.protocols import imitation_own_states as own
from civ5rl.protocols.imitation import (
    load_episode,
    prepare_manifest,
)
from civ5rl.protocols.imitation_invariants import (
    OWN_PARENT_FIELDS,
    OWN_ROOT,
    OWN_SETTINGS,
    require_values,
)
from civ5rl.provenance import checkpoint
from civ5rl.provenance.common import (
    ROOT,
    configure,
    file_sha256,
    read_json,
    seed_for,
    write_json,
)
from civ5rl.provenance.lineage import current_parent_lineage, verify_existing_run
from civ5sim.engine import digest


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def fingerprint(model):
    value = hashlib.sha256()
    for name, tensor in sorted(model.items()):
        value.update(name.encode())
        value.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def validate_config(config, parent_config):
    require_values(
        config,
        {field: parent_config[field] for field in OWN_PARENT_FIELDS},
        "own_states parent",
    )
    require_values(config["own_states"], OWN_SETTINGS, "own_states")
    require_values(config, OWN_ROOT, "own_states")


def verify_files(files):
    for name, expected in files.items():
        if file_sha256(ROOT / name) != expected:
            raise ValueError(f"Input hash mismatch: {name}")


def initialize(config, out):
    settings = config["own_states"]
    parent = ROOT / settings["parent_run"]
    parent_config = read_json(parent / "resolved-config.json")
    validate_config(config, parent_config)
    source = parent / "checkpoints/best.pt"
    if file_sha256(source) != settings["parent_checkpoint_sha256"]:
        raise ValueError("Parent checkpoint hash differs")
    parent_manifest = read_json(parent / "manifest.json")
    base = prepare_manifest(config)
    if base != parent_manifest["base"]:
        raise ValueError("Own-state world split or validation cases changed")
    for files in parent_manifest["inputs"].values():
        verify_files(files)
    plan = {
        "schema": 1,
        "kind": config["kind"],
        "base": base,
        "parent_manifest_sha256": file_sha256(parent / "manifest.json"),
        "parent_checkpoint_sha256": file_sha256(source),
        "config_hash": digest(config),
        "old_inputs": parent_manifest["inputs"],
    }
    allowed = {
        "civ5rl/core/model.py",
        "civ5rl/core/training_graphs.py",
        "civ5rl/engine/imitation.py",
        "civ5rl/engine/imitation_performance.py",
        "tools/run_cp_imitation.py",
        "civ5rl/engine/imitation_own_states.py",
        "civ5rl/protocols/imitation_own_states.py",
        "configs/rl-cp-imitation-own-states.json",
        "tests/test_rl_own_states.py",
    }
    current, changes = current_parent_lineage(
        parent, config, plan, allowed, same_corpus=False
    )
    if (out / "passport.json").exists():
        verify_existing_run(out, config, current)
        if read_json(out / "manifest.json") != plan:
            raise ValueError("Own-state manifest differs")
    else:
        checkpoint.passport(out, config, plan)
        write_json(out / "manifest.json", plan)
        write_json(
            out / "lineage.json",
            {
                "parent": relative(parent),
                "parent_Adam_steps": 1580,
                "parent_checkpoint_sha256": file_sha256(source),
                "changed_execution_sources": changes,
                "advisor_reset": settings["advisor_reset"],
                "GPU_target_policy": "For iterative GPU phases >=10 minutes, 60-second NVML core load target is 80%; investigate and explain lower utilization.",
            },
        )
        write_json(
            out / "state.json",
            {"steps_performed": 0, "training_seconds": 0.0, "cycles": {}, "best": None},
        )
    for path in sorted(out.glob("*/collection.json")):
        collection = read_json(path)
        if collection["manifest_sha256"] != file_sha256(path.parent / "inputs.json"):
            raise ValueError(f"Frozen collection manifest differs: {path}")
        verify_files(read_json(path.parent / "inputs.json")["files"])
    return parent, plan, read_json(out / "passport.json")


def eta_decision(waves, target, elapsed, settings, previous_streak):
    labels = sum(w["labels"] for w in waves)
    matches = sum(w["matches"] for w in waves)
    if labels >= target:
        return {"stop": False, "streak": 0, "quota_met": True}
    if matches < settings["eta_min_matches"] or elapsed < settings["eta_min_seconds"]:
        return {"stop": False, "streak": 0, "eligible": False}
    recent, count = [], 0
    for wave in reversed(waves):
        recent.append(wave)
        count += wave["matches"]
        if count >= settings["eta_recent_matches"]:
            break
    overall = labels / elapsed
    recent_rate = sum(w["labels"] for w in recent) / sum(w["seconds"] for w in recent)
    fastest, slowest = max(overall, recent_rate), min(overall, recent_rate)
    stable = slowest > 0 and fastest / slowest <= settings["eta_rate_ratio"]
    eta = (target - labels) / fastest if fastest else None
    exceeds = eta is None or elapsed + eta > settings["collection_seconds"]
    streak = previous_streak + 1 if stable and exceeds else 0
    return {
        "stop": streak >= settings["eta_streak"],
        "streak": streak,
        "eligible": True,
        "stable": stable,
        "overall_labels_per_second": overall,
        "recent_labels_per_second": recent_rate,
        "remaining_eta_seconds": eta,
        "predicted_finish_seconds": elapsed + eta if eta is not None else None,
    }


def collection_summary(rows, seconds, status, target, waves):
    labels = sum(r["labels"] for r in rows)
    slices = {}
    for field in ("side", "placement", "opponent", "policy_mode", "parent_world"):
        by_value = {}
        for value in sorted({r[field] for r in rows}):
            selected = [r for r in rows if r[field] == value]
            by_value[str(value)] = {
                "matches": len(selected),
                "labels": sum(r["labels"] for r in selected),
            }
        slices[field] = by_value
    return {
        "status": status,
        "target": target,
        "labels": labels,
        "overflow": max(0, labels - target),
        "matches": len(rows),
        "seconds": seconds,
        "waves": waves,
        "slices": slices,
        "forced": sum(r["forced"] for r in rows),
        "disagreements": sum(r["disagreements"] for r in rows),
        "advice_seconds": sum(r["advice_seconds"] for r in rows),
        "policy_seconds": sum(r["policy_seconds"] for r in rows),
        "outcomes": dict(
            Counter(
                "D" if r["winner"] is None else "W" if r["winner"] == r["side"] else "L"
                for r in rows
            )
        ),
    }


def collect_split(config, plan, out, split, behavior):
    from civ5rl.experiments.imitation_helpers import pool

    directory = out / split
    if (directory / "collection.json").exists():
        raise ValueError(f"Collection already recorded: {split}")
    settings = config["own_states"]
    target = (
        settings["dev_labels"] if split == "dev" else settings["new_labels_per_cycle"]
    )
    budget = (
        settings["dev_collection_seconds"]
        if split == "dev"
        else settings["collection_seconds"]
    )
    seen = set()
    for files in plan["old_inputs"].values():
        for name in files:
            case = read_json((ROOT / name).parent / "case.json")
            seen.add(own.own_state_physical_hash(case["scenario"]))
    for path in out.glob("*/inputs.json"):
        seen.update(c["physical_hash"] for c in read_json(path)["cases"])
    rows, waves, cases, files = [], [], [], {}
    labels = index = streak = 0
    started = perf_counter()
    with ResourceMonitor(directory / "telemetry") as monitor, pool(4) as executor:
        monitor.mark("CPU_own_state_collection", split=split)
        while labels < target and perf_counter() - started < budget:
            wave_start = perf_counter()
            jobs = []
            for case_id in range(index, index + 4):
                for attempt in range(100):
                    case = own.make_case(
                        config, plan["base"], split, case_id, attempt=attempt
                    )
                    if case["physical_hash"] not in seen:
                        seen.add(case["physical_hash"])
                        break
                else:
                    raise RuntimeError(
                        "invalid_dataset: 100 duplicate placement attempts"
                    )
                cases.append(case)
                jobs.append(
                    (
                        config,
                        plan["base"],
                        case,
                        str(ROOT / behavior),
                        str(directory / f"match-{case_id:06}"),
                    )
                )
            write_json(directory / "planned-cases.json", cases)
            completed = list(executor.map(own.match_task, jobs))
            for row in completed:
                path = (
                    directory / f"match-{row['case_id']:06}" / "demonstration.json.gz"
                )
                files[relative(path)] = row["demonstration_sha256"]
            rows.extend(completed)
            wave = {
                "matches": 4,
                "labels": sum(r["labels"] for r in completed),
                "seconds": perf_counter() - wave_start,
            }
            waves.append(wave)
            labels += wave["labels"]
            index += 4
            eta = eta_decision(
                waves, target, perf_counter() - started, settings, streak
            )
            streak = eta["streak"]
            wave["eta"] = eta
            write_json(
                directory / "progress.json",
                collection_summary(
                    rows, perf_counter() - started, "collecting", target, waves
                ),
            )
            print(
                json.dumps(
                    {
                        "split": split,
                        "matches": index,
                        "labels": labels,
                        "target": target,
                        "seconds": perf_counter() - started,
                        "eta": eta,
                    }
                ),
                flush=True,
            )
            if eta["stop"]:
                break
    status = (
        "complete"
        if labels >= target
        else "collection_eta_exceeds_budget"
        if streak >= settings["eta_streak"]
        else "collection_budget_exhausted"
    )
    frozen = {
        "split": split,
        "files": files,
        "cases": cases,
        "behavior": behavior,
        "behavior_sha256": file_sha256(ROOT / behavior),
        "plan_hash": digest(plan),
    }
    write_json(directory / "inputs.json", frozen)
    result = collection_summary(rows, perf_counter() - started, status, target, waves)
    result["manifest_sha256"] = file_sha256(directory / "inputs.json")
    write_json(directory / "collection.json", result)
    return result


class LabelStream:
    """Uniform shuffled passes, carrying the 512-label remainder across passes."""

    def __init__(self, chunks, masks, rng):
        self.chunks, self.masks, self.rng = chunks, masks, rng
        self.order, self.position, self.offset, self.passes = [], 0, None, 0

    def take(self, count):
        result, labels = [], 0
        while labels < count:
            if self.position == len(self.order):
                self.order = list(self.chunks)
                self.rng.shuffle(self.order)
                self.position, self.offset = 0, None
                self.passes += 1
            episode, start, stop = self.order[self.position]
            start = start if self.offset is None else self.offset
            end, piece_labels = take_counted_span(
                episode, start, stop, self.masks, count - labels
            )
            labels += piece_labels
            if piece_labels:
                result.append((episode, start, end))
            self.offset = end
            if end == stop:
                self.position += 1
                self.offset = None
        return result

    def state(self):
        return {
            "order": self.order,
            "position": self.position,
            "offset": self.offset,
            "passes": self.passes,
            "rng": self.rng.getstate(),
        }

    def restore(self, state):
        self.order, self.position, self.offset, self.passes = (
            state["order"],
            state["position"],
            state["offset"],
            state["passes"],
        )
        self.rng.setstate(state["rng"])


def load_episodes(files):
    verify_files(files)
    return [load_episode(ROOT / name) for name in files]


def dataset_files(out, split):
    info = read_json(out / split / "collection.json")
    if info["status"] != "complete":
        raise ValueError(f"Incomplete own-state dataset: {split}")
    path = out / split / "inputs.json"
    if info["manifest_sha256"] != file_sha256(path):
        raise ValueError("Collection manifest hash changed")
    return read_json(path)["files"]


def measure(policy, old_dev, own_dev):
    # Evaluation must not consume the training RNG stream.
    with isolated_rng(random.Random(0)):
        a = teacher_metrics(policy, old_dev, move_details=True)
        b = teacher_metrics(policy, own_dev, move_details=True)
    return {
        "old_holdout": a,
        "own_state_dev": b,
        "score": 0.5 * a["ce"] + 0.5 * b["ce"],
    }


def train_cycle(config, plan, identity, out, cycle):
    from civ5rl.experiments.imitation_helpers import adam, load_policy

    directory = out / f"cycle-{cycle:02}"
    if (directory / "training.json").exists():
        raise ValueError("Cycle training already recorded")
    state = read_json(out / "state.json")
    settings = config["own_states"]
    if state["training_seconds"] >= settings["training_seconds"]:
        return None
    configure(17, "cuda")
    preparation_start = perf_counter()
    old = load_episodes(plan["old_inputs"]["training"])
    old_count = sum(r["supervised"] for ep in old for r in ep["decisions"])
    if old_count != 100040:
        raise ValueError("Original training corpus is not 100040 free labels")
    new_files = {}
    for previous in range(1, cycle + 1):
        new_files.update(dataset_files(out, f"cycle-{previous:02}"))
    new = load_episodes(new_files)
    old_dev_episodes = load_episodes(plan["old_inputs"]["holdout"])
    if sum(r["supervised"] for ep in old_dev_episodes for r in ep["decisions"]) != 2485:
        raise ValueError("Old holdout changed")
    new_dev_episodes = load_episodes(dataset_files(out, "dev"))
    cache = DemonstrationCache(old + new, "cuda")
    old_dev = DemonstrationCache(old_dev_episodes, "cuda")
    new_dev = DemonstrationCache(new_dev_episodes, "cuda")
    base_path = ROOT / (
        state["best"]["path"]
        if state["best"]
        else config["own_states"]["parent_run"] + "/checkpoints/best.pt"
    )
    saved = torch.load(base_path, weights_only=False, map_location="cpu")
    policy = load_policy(config, base_path, "cuda")
    optimizer = adam(policy, config, "cuda")
    optimizer.load_state_dict(saved["optimizer"])
    assert_equal(policy.state_dict(), saved["model"])
    assert_equal(optimizer.state_dict(), saved["optimizer"])
    base_counter = int(saved["state"]["Adam_steps"])
    if cycle == 1 and base_counter != 1580:
        raise ValueError("Wrong initial Adam counter")
    write_json(
        directory / "initialization-check.json",
        {
            "model_equal": True,
            "Adam_equal": True,
            "base_Adam_counter": base_counter,
            "source": relative(base_path),
            "old_free_labels": old_count,
            "new_free_labels": sum(
                r["supervised"] for ep in new for r in ep["decisions"]
            ),
        },
    )
    chunks = cache.chunks()
    old_rng = random.Random(seed_for(17, 0, "old_mixture", namespace="own_states"))
    new_rng = random.Random(seed_for(17, 0, "new_mixture", namespace="own_states"))
    old_stream = LabelStream(
        [c for c in chunks if c[0] < len(old)], cache.masks, old_rng
    )
    new_stream = LabelStream(
        [c for c in chunks if c[0] >= len(old)], cache.masks, new_rng
    )
    if cycle == 2:
        previous_streams = torch.load(
            out / "sampler.pt", weights_only=False, map_location="cpu"
        )
        old_stream.restore(previous_streams["old"])
        # Pool version changes: begin a uniform pass over both cycles, retaining RNG.
        new_rng.setstate(previous_streams["new"]["rng"])
    streams = lambda: {"old": old_stream.state(), "new": new_stream.state()}

    def next_batch():
        left, right = old_stream.take(512), new_stream.take(512)
        assert sum(sum(cache.masks[e][a:b]) for e, a, b in left) == 512
        assert sum(sum(cache.masks[e][a:b]) for e, a, b in right) == 512
        return left + right

    snapshots = streams()
    with ResourceMonitor(directory / "training-telemetry") as monitor:
        monitor.mark("preparation")
        backend = "encoded_CPU_to_GPU"
        selected = cache
        try:
            selected = ResidentDemonstrations(cache, reserve_bytes=4 * 1024**3)
            selected.enable_compiled_gather()
            selected.enable_async_transfers()
            backend = "resident_compiled_gather_async_coalesced_GRU_graphs"
        except MemoryError as exc:
            write_json(directory / "resident-unavailable.json", {"reason": str(exc)})
        write_json(
            directory / "backend.json",
            {
                "selected": backend,
                "selection_before_first_update": True,
                "spatial_block": 32,
                "microbatch": 8,
                "labels": 1024,
                "GPU_load_gate": None,
            },
        )
        monitor.mark("warmup_no_Adam")
        warm_spans = next_batch()
        optimizer.zero_grad(set_to_none=True)
        backward_minibatch_async(policy, selected, warm_spans, 1024)
        torch.cuda.synchronize()
        optimizer.zero_grad(set_to_none=True)
        old_stream.restore(snapshots["old"])
        new_stream.restore(snapshots["new"])
        configure(17, "cuda")
        assert_equal(policy.state_dict(), saved["model"])
        assert_equal(optimizer.state_dict(), saved["optimizer"])
        frozen_value = FrozenParamsCheck(policy.value)
        monitor.mark("offline_metrics")
        evaluation_start = perf_counter()
        initial_metrics = measure(policy, old_dev, new_dev)
        initial_metrics.update(
            cycle_step=0,
            steps_performed=state["steps_performed"],
            Adam_counter=base_counter,
        )
        offline_seconds = perf_counter() - evaluation_start
        measurements = [initial_metrics]
        if state["best"] is None:
            state["best"] = {
                "path": relative(base_path),
                "weights_sha256": fingerprint(policy.state_dict()),
                "score": initial_metrics["score"],
                "cycle": 0,
                "step": 0,
                "Adam_counter": base_counter,
            }
            write_json(out / "initial-metrics.json", initial_metrics)
        preparation_seconds = perf_counter() - preparation_start - offline_seconds
        records, local_steps = [], 0
        reason = "step_limit"

        def save_candidate():
            nonlocal offline_seconds
            torch.cuda.synchronize()
            if {int(s["step"]) for s in optimizer.state.values()} != {
                base_counter + local_steps
            }:
                raise RuntimeError("Saved Adam counters differ from performed updates")
            monitor.mark("checkpoint_and_metrics")
            point = directory / f"checkpoints/step-{local_steps:04}.pt"
            checkpoint.save(
                point,
                policy,
                optimizer,
                {
                    "cycle": cycle,
                    "cycle_step": local_steps,
                    "Adam_steps": base_counter + local_steps,
                    "steps_performed": state["steps_performed"],
                    "training_seconds": state["training_seconds"],
                    "sampler": streams(),
                    "input_manifest_hashes": {
                        split: file_sha256(out / split / "inputs.json")
                        for split in [
                            "dev",
                            *[f"cycle-{c:02}" for c in range(1, cycle + 1)],
                        ]
                    },
                },
                old_rng,
                config,
                identity,
            )
            started = perf_counter()
            measured = measure(policy, old_dev, new_dev)
            offline_seconds += perf_counter() - started
            measured.update(
                cycle_step=local_steps,
                steps_performed=state["steps_performed"],
                Adam_counter=base_counter + local_steps,
                checkpoint=relative(point),
                weights_sha256=fingerprint(policy.state_dict()),
            )
            measurements.append(measured)
            if measured["score"] < state["best"]["score"]:
                state["best"] = {
                    "path": relative(point),
                    "weights_sha256": measured["weights_sha256"],
                    "score": measured["score"],
                    "cycle": cycle,
                    "step": local_steps,
                    "Adam_counter": base_counter + local_steps,
                }
            write_json(directory / "measurements.json", measurements)
            write_json(out / "state.json", state)
            torch.save(streams(), out / "sampler.pt")
            print(
                json.dumps(
                    {
                        "cycle": cycle,
                        "step": local_steps,
                        "measure": measured,
                        "best": state["best"],
                    }
                ),
                flush=True,
            )

        for _ in range(settings["steps_per_cycle"]):
            if state["training_seconds"] >= settings["training_seconds"]:
                reason = "optimizer_time_limit"
                break
            monitor.mark("iterative_training", cycle=cycle, step=local_steps + 1)
            torch.cuda.synchronize()
            started = perf_counter()
            spans = next_batch()
            ce, norm = bc_optimizer_step(
                policy,
                optimizer,
                backward_minibatch_async,
                selected,
                spans,
                1024,
                max_gradient_norm=config["max_gradient_norm"],
                asynchronous=True,
            )
            torch.cuda.synchronize()
            ce_value, norm_value = float(ce), float(norm)
            seconds = perf_counter() - started
            local_steps += 1
            state["steps_performed"] += 1
            state["training_seconds"] += seconds
            if local_steps == 1:
                if {int(s["step"]) for s in optimizer.state.values()} != {
                    base_counter + 1
                }:
                    raise RuntimeError("First Adam step counter differs")
                write_json(
                    directory / "first-step-check.json",
                    {
                        "before": base_counter,
                        "after": base_counter + 1,
                        "all_counters_equal": True,
                        "mixture": [512, 512],
                        "spans": spans,
                    },
                )
            row = {
                "cycle_step": local_steps,
                "step_performed": state["steps_performed"],
                "Adam_counter": base_counter + local_steps,
                "labels": 1024,
                "old_labels": 512,
                "new_labels": 512,
                "CE": ce_value,
                "gradient_norm_pre_clip": norm_value,
                "seconds": seconds,
                "spans_hash": digest(spans),
            }
            records.append(row)
            with (directory / "optimizer-steps.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(json.dumps(row) + "\n")
            if local_steps % settings["measure_every"] == 0:
                save_candidate()
            if local_steps % 10 == 0:
                print(
                    json.dumps(
                        {
                            "cycle": cycle,
                            "step": local_steps,
                            "CE": row["CE"],
                            "optimizer_seconds_total": state["training_seconds"],
                        }
                    ),
                    flush=True,
                )
        if local_steps and measurements[-1]["cycle_step"] != local_steps:
            save_candidate()
        frozen_value.verify()
        state["cycles"][str(cycle)] = {
            "steps": local_steps,
            "selected": dict(state["best"]),
            "reason": reason,
        }
        result = {
            "status": "complete",
            "stop_reason": reason,
            "cycle": cycle,
            "steps": local_steps,
            "steps_performed_total": state["steps_performed"],
            "training_seconds_total": state["training_seconds"],
            "preparation_seconds": preparation_seconds,
            "offline_seconds": offline_seconds,
            "base_Adam_counter": base_counter,
            "last_Adam_counter": base_counter + local_steps,
            "selected": state["best"],
            "measurements": measurements,
            "backend": backend,
            "value_head_bitwise_unchanged": True,
            "presentations": local_steps * 1024,
            "new_pool_version": cycle,
            "new_pool_boundary": "fresh_uniform_pass_continued_RNG",
        }
        write_json(directory / "training.json", result)
        write_json(out / "state.json", state)
        torch.save(streams(), out / "sampler.pt")
    selected = cache = old_dev = new_dev = policy = optimizer = old = new = None
    frozen_value = None
    gc.collect()
    torch.cuda.empty_cache()
    return result


def evaluate_cycle(config, plan, out, parent, cycle):

    directory = out / f"cycle-{cycle:02}"
    if (directory / "evaluation.json").exists():
        raise ValueError("Cycle evaluation already recorded")
    training = read_json(directory / "training.json")
    candidate = training["selected"]
    if (
        fingerprint(
            torch.load(
                ROOT / candidate["path"], weights_only=False, map_location="cpu"
            )["model"]
        )
        != candidate["weights_sha256"]
    ):
        raise ValueError("Selected policy changed before evaluation")
    cache_path = out / "evaluations.json"
    recorded = read_json(cache_path) if cache_path.exists() else {"runs": {}}
    parent_saved = torch.load(
        parent / "checkpoints/best.pt", weights_only=False, map_location="cpu"
    )
    parent_hash = fingerprint(parent_saved["model"])
    result = {"candidate": candidate, "runs": {}}
    for mode in ("sample", "argmax"):
        key = candidate["weights_sha256"] + ":" + mode
        if key in recorded["runs"]:
            run = {**recorded["runs"][key], "new_matches": 0, "reused": True}
        elif candidate["weights_sha256"] == parent_hash:
            inherited = read_json(parent / "evaluations.json")["runs"][mode]
            run = {**inherited, "new_matches": 0, "reused": True}
        else:
            directories = {
                enemy: directory / "evaluation" / mode / enemy
                for enemy in ("cp", "greedy")
            }
            started = perf_counter()

            def progress(matches, *, mode=mode, started=started):
                print(
                    json.dumps(
                        {
                            "evaluation_cycle": cycle,
                            "mode": mode,
                            "matches": matches,
                            "seconds": perf_counter() - started,
                        }
                    ),
                    flush=True,
                )

            new_matches = run_validation_cases(
                config,
                plan["base"],
                ROOT / candidate["path"],
                mode,
                directories,
                4,
                batch_size=4,
                progress=progress,
            )
            run = {
                "directories": {
                    enemy: relative(path) for enemy, path in directories.items()
                },
                "new_matches": new_matches,
                "reused": False,
                "seconds": perf_counter() - started,
                "opponents": {
                    enemy: collect_summary(path) for enemy, path in directories.items()
                },
            }
        for enemy, path in run["directories"].items():
            if collect_summary(ROOT / path) != run["opponents"][enemy]:
                raise ValueError("Cached evaluation metrics changed")
            actual = sorted((ROOT / path).glob("*/case.json"))
            if len(actual) != 64:
                raise ValueError("Evaluation case count differs")
            for case_file in actual:
                case = read_json(case_file)
                if case["dataset_split"] != "validation":
                    raise ValueError("Closed/non-validation cases in evaluation")
                from civ5rl.protocols.imitation import validate_case

                validate_case(case, config, plan["base"])
                for filename in ("replay.json", "result.json", "final.json"):
                    if not (case_file.parent / filename).is_file():
                        raise ValueError("Missing full evaluation replay")
        recorded["runs"][key] = run
        result["runs"][mode] = run
        write_json(cache_path, recorded)
    write_json(directory / "evaluation.json", result)
    return result


def compare_battles(config, before_path, after_path):
    settings = config["own_states"]
    result, slices, _, _ = compare_paired_directories(
        before_path,
        after_path,
        samples=settings["bootstrap_samples"],
        seed=settings["bootstrap_seed"],
    )
    return {
        **result,
        "slices": {
            name.replace("side0", "side-0").replace("side1", "side-1"): {
                key: value[key]
                for key in ("matches", "baseline_wins", "candidate_wins")
            }
            for name, value in slices.items()
        },
    }


def report(config, out, parent):
    state = read_json(out / "state.json")
    lines = [
        "# CP advice на собственных траекториях",
        "",
        "Sampling против CP — основной результат. Validation используется для разработки; test закрыт.",
        "",
        "## Сбор",
        "",
        "| Набор | Статус | Метки | Матчи | Секунды |",
        "|---|---|---:|---:|---:|",
    ]
    for split in ("dev", "cycle-01", "cycle-02"):
        path = out / split / "collection.json"
        if path.exists():
            r = read_json(path)
            lines.append(
                f"| {split} | {r['status']} | {r['labels']} | {r['matches']} | {r['seconds']:.1f} |"
            )
    lines += [
        "",
        "## Обучение и выбор",
        "",
        f"Фактически выполнено {state['steps_performed']} Adam steps, {state['training_seconds']:.2f} с работы updates. Шаг содержит 512 старых и 512 новых свободных меток. Value head frozen. Для итеративной GPU-фазы от 10 минут требуется средняя загрузка Core Load не ниже 80% за 60 секунд либо обоснование продолжения; короткие фазы не продлеваются.",
        "",
        "| Цикл / шаг | Old CE | Own-state dev CE | Composite | Adam counter |",
        "|---|---:|---:|---:|---:|",
    ]
    comparisons, selected_replays, all_measurements = {}, [], []
    for cycle in (1, 2):
        directory = out / f"cycle-{cycle:02}"
        if (directory / "training.json").exists():
            training = read_json(directory / "training.json")
            for m in training["measurements"]:
                all_measurements.append((cycle, m))
                lines.append(
                    f"| {cycle}/{m['cycle_step']} | {m['old_holdout']['ce']:.6f} | {m['own_state_dev']['ce']:.6f} | {m['score']:.6f} | {m['Adam_counter']} |"
                )
    if all_measurements:
        initial = read_json(out / "initial-metrics.json")
        chosen = (
            initial
            if state["best"]["cycle"] == 0
            else next(
                m
                for _, m in all_measurements
                if m.get("checkpoint") == state["best"]["path"]
            )
        )
        lines += [
            "",
            "### Accuracy выбранной модели",
            "",
            "| Dev | Тип | Initial accuracy | Selected accuracy |",
            "|---|---|---:|---:|",
        ]
        for dataset in ("old_holdout", "own_state_dev"):
            a, b = initial[dataset], chosen[dataset]
            lines.append(
                f"| {dataset} | all | {100 * a['accuracy']:.2f}% | {100 * b['accuracy']:.2f}% |"
            )
            for kind in sorted(a["by_kind"].keys() | b["by_kind"].keys()):
                aa, bb = (
                    a["by_kind"].get(kind, {}).get("accuracy"),
                    b["by_kind"].get(kind, {}).get("accuracy"),
                )
                lines.append(
                    f"| {dataset} | {kind} | {100 * aa:.2f}% | {100 * bb:.2f}% |"
                )
            for detail in ("type", "type_unit", "exact"):
                lines.append(
                    f"| {dataset} | move {detail} | {100 * a['move'][detail]:.2f}% | {100 * b['move'][detail]:.2f}% |"
                )
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        for dataset, label in (
            ("old_holdout", "Old holdout"),
            ("own_state_dev", "Own-state dev"),
        ):
            for c in (1, 2):
                points = [m for cycle, m in all_measurements if cycle == c]
                if points:
                    axes[0].plot(
                        [m["steps_performed"] for m in points],
                        [m[dataset]["ce"] for m in points],
                        marker="o",
                        label=f"{label}, cycle {c}",
                    )
                    axes[1].plot(
                        [m["steps_performed"] for m in points],
                        [100 * m[dataset]["accuracy"] for m in points],
                        marker="o",
                        label=f"{label}, cycle {c}",
                    )
        for ax, ylabel in zip(axes, ("CE", "Accuracy, %")):
            ax.set_xlabel("Actually performed Adam steps")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(out / "learning-curves.png", dpi=150)
        plt.close(figure)
        lines += ["", "![Learning curves](learning-curves.png)"]
    lines += [
        "",
        "## Боевая сила",
        "",
        "| Цикл | Режим | Противник | Baseline W/D/L | Candidate W/D/L | Прирост п.п. | 95% CI п.п. |",
        "|---|---|---|---|---|---:|---|",
    ]
    baseline = read_json(parent / "evaluations.json")["runs"]
    for cycle in (1, 2):
        path = out / f"cycle-{cycle:02}" / "evaluation.json"
        if not path.exists():
            continue
        evaluated = read_json(path)
        for mode, run in evaluated["runs"].items():
            for enemy, directory in run["directories"].items():
                previous = baseline[mode]
                comparison = compare_battles(
                    config, previous["directories"][enemy], directory
                )
                comparisons[f"cycle-{cycle}-{mode}-{enemy}"] = comparison
                a, b = previous["opponents"][enemy], run["opponents"][enemy]
                low, high = comparison["CI95"]
                lines.append(
                    f"| {cycle} | {mode} | {enemy} | {a['W']}/{a['D']}/{a['L']} | {b['W']}/{b['D']}/{b['L']} | {100 * comparison['delta']:.2f} | [{100 * low:.2f}; {100 * high:.2f}] |"
                )
                if not run.get("reused"):
                    metrics = [
                        (p, read_json(p))
                        for p in sorted((ROOT / directory).glob("*/metrics.json"))
                    ]
                    for outcome in ("W", "D", "L"):
                        item = next(
                            (
                                (p, r)
                                for p, r in metrics
                                if (
                                    "D"
                                    if r["winner"] is None
                                    else "W"
                                    if r["winner"] == r["side"]
                                    else "L"
                                )
                                == outcome
                            ),
                            None,
                        )
                        if item:
                            selected_replays.append(item[0].parent)
    write_json(out / "battle-comparisons.json", comparisons)
    for split in ("dev", "cycle-01", "cycle-02"):
        choices = sorted((out / split).glob("match-*/metrics.json"))
        if choices:
            selected_replays.append(choices[0].parent)
            disagreement = next(
                (p.parent for p in choices if read_json(p)["disagreements"]), None
            )
            if disagreement:
                selected_replays.append(disagreement)
    checks = [
        verify_replay(path, demonstration=(path / "demonstration.json.gz").exists())
        for path in dict.fromkeys(selected_replays)
    ]
    write_json(out / "replay-verification.json", checks)
    lines += [
        "",
        "## Итог",
        "",
        f"Выбранная модель: `{state['best']['path'] if state['best'] else 'обучение не начиналось'}`.",
        f"Статус выполнения: `{read_json(out / 'status.json')['status'] if (out / 'status.json').exists() else 'in_progress'}`.",
        "Положительный результат требует прироста sampling win rate против CP и нижней границы CI >0. Если CI пересекает ноль, результат неопределённый. Argmax и greedy — дополнительные результаты.",
        "Этот опыт объединяет собственные траектории и обучение против CP; их причинные вклады не изолированы. Seed 42, PPO, третий цикл и test не запускались.",
        "",
        f"Полностью воспроизведено {len(checks)} характерных боёв с проверкой результатов команд, публичных входов и хешей состояний. Детальные метрики типов/move, времён, расхождений и покрытия находятся в JSON соответствующих фаз.",
    ]
    lines += ["", "## Примеры сохранённых боёв", ""]
    for path in list(dict.fromkeys(selected_replays))[:12]:
        result = read_json(path / "result.json")
        case = read_json(path / "case.json")
        replay = read_json(path / "replay.json")
        side = case["learner_side"]
        active, actions = 0, []
        for step in replay["steps"]:
            a = step["action"]
            if active == side:
                actions.append(f"{a['kind']}({a['unit']},{a['target']})")
            if a["kind"] == "end_turn":
                active = 1 - active
        outcome = (
            "draw"
            if result["winner"] is None
            else "win"
            if result["winner"] == side
            else "loss"
        )
        lines.append(
            f"- `{relative(path)}`: {outcome}, {result['completed_rounds']} раундов, HP {result['remaining_hp']}. Первые команды learner: {' → '.join(actions[:10])}."
        )
        if (path / "demonstration.json.gz").exists():
            ep = load_episode(path / "demonstration.json.gz")
            differing = next(
                (
                    (i, r)
                    for i, r in enumerate(ep["decisions"])
                    if r["teacher_action"] != r["executed_action"]
                ),
                None,
            )
            if differing:
                i, r = differing
                lines.append(
                    f"  Решение {i}: CP советовал `{r['actions'][r['teacher_action']]}`, сеть исполнила `{r['actions'][r['executed_action']]}`. Это наблюдаемое расхождение, не доказательство причин исхода."
                )
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    final_best = (
        torch.load(ROOT / state["best"]["path"], weights_only=False, map_location="cpu")
        if state["best"]
        else None
    )
    if final_best and {
        int(s["step"]) for s in final_best["optimizer"]["state"].values()
    } != {state["best"]["Adam_counter"]}:
        raise ValueError("Selected Adam counters differ from recorded selection")
    write_json(
        out / "verification.json",
        {
            "replays": len(checks),
            "selected_Adam_counter_verified": final_best is not None,
            "source_hashes": read_json(out / "passport.json")["source_hashes"],
            "full_suite": False,
            "resume_tests": False,
        },
    )
    hashes = {
        relative(p): file_sha256(p)
        for p in sorted(out.rglob("*"))
        if p.is_file()
        and p.name != "artifact-hashes.json"
        and not p.name.endswith(".tmp")
    }
    write_json(out / "artifact-hashes.json", hashes)


def run(phase, config, out, workers, device, cycle=None):
    if workers != 4:
        raise ValueError("Own-state protocol requires four workers")
    parent, plan, identity = initialize(config, out)
    if phase in ("collect", "train", "evaluate") and cycle not in (1, 2):
        raise ValueError("Own-state phase requires --cycle 1 or 2")
    if phase not in ("prepare", "collect", "train", "evaluate", "report", "run"):
        raise ValueError("Unsupported own-state phase")
    if phase in ("train", "run") and device != "cuda":
        raise ValueError("Own-state training requires CUDA/BF16")
    started = perf_counter()
    current_phase = phase
    try:
        if phase == "report":
            report(config, out, parent)
            return
        phases = []
        if phase == "run":
            phases = [(p, c) for c in (1, 2) for p in ("collect", "train", "evaluate")]
        elif phase != "prepare":
            phases = [(phase, cycle)]
        for current_phase, c in phases:
            # Identity includes every source/config and every completed data phase.
            initialize(config, out)
            directory = out / f"cycle-{c:02}"
            marker = (
                directory
                / {
                    "collect": "collection.json",
                    "train": "training.json",
                    "evaluate": "evaluation.json",
                }[current_phase]
            )
            if marker.exists():
                if phase == "run":
                    if (
                        current_phase == "collect"
                        and read_json(marker)["status"] != "complete"
                    ):
                        break
                    continue
                raise ValueError(f"Phase already recorded: {marker}")
            if c == 2 and not (out / "cycle-01/evaluation.json").exists():
                raise ValueError("Cycle 1 must finish before cycle 2")
            state = read_json(out / "state.json")
            if current_phase == "collect":
                if (
                    state["training_seconds"]
                    >= config["own_states"]["training_seconds"]
                ):
                    write_json(
                        out / "status.json",
                        {"status": "optimizer_time_limit", "cycle": c},
                    )
                    break
                if not (out / "dev/collection.json").exists():
                    result = collect_split(
                        config,
                        plan,
                        out,
                        "dev",
                        relative(parent / "checkpoints/best.pt"),
                    )
                    if result["status"] != "complete":
                        write_json(
                            out / "status.json",
                            {"status": result["status"], "split": "dev"},
                        )
                        break
                dataset_files(out, "dev")
                behavior = (
                    relative(parent / "checkpoints/best.pt")
                    if c == 1
                    else state["cycles"]["1"]["selected"]["path"]
                )
                result = collect_split(config, plan, out, f"cycle-{c:02}", behavior)
                if result["status"] != "complete":
                    write_json(
                        out / "status.json", {"status": result["status"], "cycle": c}
                    )
                    break
            elif current_phase == "train":
                result = train_cycle(config, plan, identity, out, c)
                if result is None:
                    write_json(
                        out / "status.json",
                        {"status": "optimizer_time_limit", "cycle": c},
                    )
                    break
            else:
                evaluate_cycle(config, plan, out, parent, c)
            write_json(
                out / "status.json",
                {"status": "in_progress", "phase": current_phase, "cycle": c},
            )
        else:
            if phase == "run":
                write_json(out / "status.json", {"status": "complete", "cycles": 2})
        if phase != "prepare":
            report(config, out, parent)
        write_json(
            out / f"{phase}-execution.json",
            {"status": "completed", "wall_seconds": perf_counter() - started},
        )
    except BaseException:
        (out / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(
            out / "status.json",
            {"status": "technical_error", "phase": current_phase, "cycle": cycle},
        )
        try:
            report(config, out, parent)
        except Exception:  # noqa: BLE001 -- record report failure before propagating
            (out / "report-error.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
        raise
