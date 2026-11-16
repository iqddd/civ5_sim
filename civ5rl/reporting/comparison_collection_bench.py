"""Worker-count comparison for the shared-queue, two-stage comparison collector.

Fixed saved work: forks of a completed rollout-comparison run, replayed with
their original seeds through the production Scheduler (forecast narrowing off,
so the measured window is steady state), plus a barrier control in the former
per-fork mode and a separate source-match measurement on saved cases. Outputs
summary.json, report.md and recommended-runtime.json. No scientific run is
created or modified.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
from time import perf_counter, monotonic
import json
import math
import shutil
import statistics

from civ5rl.provenance.common import read_json, write_json, configure, seed_for

OLD_NAMESPACE_FUTURES = "c2p_rollout_comparison_v1_futures"


def timed_match(task):
    from civ5rl.experiments.comparison_match import match_task
    started = perf_counter()
    match_task(task)
    return perf_counter() - started


class FixedBudget:
    def __init__(self, limit):
        self.limit, self.started = limit, perf_counter()

    def elapsed(self):
        return perf_counter() - self.started

    def expired(self):
        return self.elapsed() >= self.limit


class SavedForkProtocol:
    """Saved forks only; old fork-id seeds reproduce the original continuations."""

    def __init__(self, source, scratch, out, records, saved_labels):
        self.source, self.scratch, self.out = source, scratch, out
        self.config = read_json(source / "match-config.json")
        self.parent = None
        self.cases = {}
        self.records = records
        self.saved_labels = saved_labels
        self.first_delta_matches = self.first_delta_checked = 0
        self.decided = []

    def case(self, episode):
        if episode not in self.cases:
            self.cases[episode] = read_json(self.scratch / "sources" / str(episode) / "case.json")
        return self.cases[episode]

    def has_sources(self, running):
        return False

    def next_source(self):
        return None

    def source_done(self, key, seconds):
        return []

    def folder(self, record, repetition, alternative):
        return self.out / f"{record['id']}" / f"{repetition}-{alternative}"

    def continuation_task(self, record, repetition, alternative):
        seed = seed_for(17, record["episode"], record["id"], repetition, namespace=OLD_NAMESPACE_FUTURES) or 1
        branch = dict(source=str(self.scratch / "sources" / str(record["episode"])), command=record["command"],
                      seed=seed, action=record["actions"][alternative])
        return (self.config, self.case(record["episode"]), str(self.folder(record, repetition, alternative)),
                self.parent, branch)

    def continuation_done(self, record, repetition, alternative, seconds):
        return read_json(self.folder(record, repetition, alternative) / "completed.json")["winner"]

    def decide(self, record, winners):
        from civ5rl.experiments.comparison_queue import two_stage_label
        label = two_stage_label(winners, record["side"])
        saved = self.saved_labels.get(record["id"])
        if saved is not None:
            self.first_delta_checked += 1
            self.first_delta_matches += int(saved["deltas"][0] == label["deltas"][0])
        self.decided.append(label)

    def reserve_seconds(self):
        return 0.

    def tick(self, stats):
        pass


def phase_resources(rows, phase, start=None, stop=None):
    """Select one measured interval; drain is not part of steady utilization."""
    samples = [r for r in rows if r.get("phase") == phase
               and (start is None or r['elapsed_seconds'] >= start)
               and (stop is None or r['elapsed_seconds'] < stop)]

    def values(name):
        return [r[name] for r in samples if r.get(name) is not None]

    def summary(name, reducer):
        found = values(name)
        return reducer(found) if found else None

    return dict(samples=len(samples),
                peak_ram_tree_rss_gib=(summary("ram_tree_rss_bytes", max) or 0) / 2**30 or None,
                peak_vram_gib=(summary("vram_used_bytes", max) or 0) / 2**30 or None,
                mean_gpu_core_percent=summary("gpu_core_load_percent", statistics.mean),
                mean_gpu_power_watts=summary("gpu_power_watts", statistics.mean),
                mean_cpu_tree_cores=(summary("cpu_tree_percent", statistics.mean) or 0) / 100 or None,
                mean_cpu_system_percent=summary("cpu_system_percent", statistics.mean))


def make_records(source, collection, count, offset):
    forks = [f for f in collection["forks"] if f["status"] == "completed"][offset:offset + count]
    return [dict(f, side=None, status="pending", winners={}) for f in forks]


def measure_workers(cfg, source, scratch, out, workers, records, warmup_records, seconds, saved_labels,
                    monitor, source_cases):
    from civ5rl.engine.evaluation import EvaluationRuntime, evaluation_precision
    from civ5rl.experiments.comparison_match import match_task
    from civ5rl.experiments.comparison_queue import CollectionExecutor, Scheduler

    matches = out / "matches" / f"W{workers}"
    runtime = EvaluationRuntime("cuda", workers, workers, 0.001)
    service_count = cfg.get('inference_services', 0)
    if service_count:
        from civ5rl.reporting.comparison_services import ProcessInferenceExecutor
        executor_context = ProcessInferenceExecutor(cfg['parents'], runtime, service_count,
                                                    prefix_states=cfg.get('prefix_states',False))
    else:
        executor_context = CollectionExecutor(cfg['parents'], runtime,prefix_states=cfg.get('prefix_states',False))
    monitor.mark(f"W{workers}-startup")
    with evaluation_precision(), executor_context as executor:
        warmup = SavedForkProtocol(source, scratch, matches / "warmup", warmup_records, {})
        warmup.parent = cfg["parents"][0]
        scheduler = Scheduler(executor, warmup, FixedBudget(10**9), workers=workers, source_share=0,
                              function=match_task)
        scheduler.forecast = False
        scheduler.add_selected([dict(r, winners={}) for r in warmup_records])
        monitor.mark(f"W{workers}-warmup")
        scheduler.run()
        protocol = SavedForkProtocol(source, scratch, matches / "measured", records, saved_labels)
        protocol.parent = cfg["parents"][0]
        budget = FixedBudget(seconds)
        scheduler = Scheduler(executor, protocol, budget, workers=workers, source_share=0, function=match_task)
        scheduler.forecast = False
        scheduler.add_selected([dict(r, winners={}) for r in records])
        before = executor.statistics()
        measured_start = monotonic() - monitor.started
        monitor.mark(f"W{workers}")
        stats = scheduler.run()
        monitor.mark(f"W{workers}-sources")
        after = executor.statistics()
        # Source matches on saved cases (proposal path), measured worker-side.
        source_seconds = []
        futures = []
        started = perf_counter()
        for episode in source_cases:
            case = read_json(scratch / "sources" / str(episode) / "case.json")
            task = (protocol.config, case, str(matches / "sources" / str(episode)), cfg["parents"][0], None)
            futures.append(executor.submit(timed_match, task, "source"))
        for future, client in futures:
            source_seconds.append(future.result())
            client.close()
        source_wall = perf_counter() - started
        monitor.mark(f"W{workers}-done")
    window = min(x for x in (seconds, stats["buffer_empty_at"]) if x is not None)
    in_window = [d for t, d in scheduler.continuation_finish_times if t <= window]
    decided = protocol.decided
    nonzero = sum(1 for d in decided if d["label"] != "first_group_zero")
    batches = after["inference_batches"] - before["inference_batches"]
    items = after["inference_items"] - before["inference_items"]
    return dict(
        workers=workers,
        inference_services=service_count,
        prefix_states=cfg.get('prefix_states',False),
        prefix_observations_skipped=after.get('prefix_observations_skipped',0)-before.get('prefix_observations_skipped',0),
        prefix_source_prefills=after.get('prefix_source_prefills',0)-before.get('prefix_source_prefills',0),
        window_seconds=window,
        measurement_started_elapsed_seconds=measured_start,
        continuations_in_window=len(in_window),
        continuations_per_second=len(in_window) / window,
        busy_workers_in_window=sum(in_window) / window,
        continuation_mean_seconds=statistics.mean(d for _, d in scheduler.continuation_finish_times),
        continuation_p90_seconds=stats["continuation_p90_seconds"],
        decided_forks=len(decided),
        forks_per_second_in_window=sum(1 for _ in decided) / max(stats["elapsed"], 1e-9),
        first_group_nonzero_fraction=nonzero / max(1, len(decided)),
        total_continuations=stats["completed_continuations"],
        total_seconds_including_drain=stats["elapsed"],
        drain_seconds_after_window=stats["elapsed"] - window,
        idle_worker_fraction_including_drain=stats["idle_worker_fraction"],
        mean_service_batch=items / max(1, batches),
        first_delta_reproduced=dict(checked=protocol.first_delta_checked, matching=protocol.first_delta_matches),
        source_matches=len(source_seconds),
        source_mean_seconds=statistics.mean(source_seconds),
        source_wall_seconds=source_wall,
        matches_directory=str(matches),
    )


def measure_barrier(cfg, source, scratch, out, workers, records, warmup_record, seconds, monitor):
    from civ5rl.engine.evaluation import EvaluationExecutor, EvaluationRuntime, evaluation_precision
    protocol = SavedForkProtocol(source, scratch, out / "matches" / f"barrier-W{workers}", records, {})
    protocol.parent = cfg["parents"][0]

    def tasks(record):
        return [protocol.continuation_task(record, j, a) for j in range(8) for a in (0, 1)]

    runtime = EvaluationRuntime("cuda", workers, 32, 0.001)
    monitor.mark(f"barrier-W{workers}-startup")
    with evaluation_precision(), EvaluationExecutor(cfg["parents"][0], runtime) as executor:
        list(executor.run(timed_match, tasks(warmup_record)))
        monitor.mark(f"barrier-W{workers}")
        started = perf_counter()
        completed = forks = 0
        for record in records:
            if perf_counter() - started >= seconds:
                break
            completed += len(list(executor.run(timed_match, tasks(record))))
            forks += 1
        elapsed = perf_counter() - started
        monitor.mark(f"barrier-W{workers}-done")
    return dict(workers=workers, seconds=elapsed, forks=forks, continuations=completed,
                continuations_per_second=completed / elapsed, forks_per_second=forks / elapsed)


def measure_packing(directory, out):
    from civ5rl.reporting.storage import pack_verified
    files = [p for p in Path(directory).rglob("*") if p.is_file()]
    size = sum(p.stat().st_size for p in files)
    packet = out / "packing-sample.packet"
    started = perf_counter()
    pack_verified(directory, packet, token="comparison-collection-bench", temporary_prefix="bench-pack-",
                  changed_message="Benchmark matches changed during packing")
    seconds = perf_counter() - started
    shutil.rmtree(packet)
    return dict(files=len(files), bytes=size, seconds=seconds, mib_per_second=size / 2**20 / seconds)


def forecast(row, saved, packing, budget_seconds=7200):
    """Little's law on measured busy workers; source cost amortized per fork."""
    p_second = saved["first_group_nonzero_fraction"]
    continuations_per_fork = 8 + 8 * p_second
    worker_seconds_per_fork = (continuations_per_fork * row["continuation_mean_seconds"]
                               + row["source_mean_seconds"] / saved["forks_per_source"])
    bytes_per_fork = saved["bytes_per_continuation"] * continuations_per_fork + \
        saved["bytes_per_source"] / saved["forks_per_source"]
    busy = min(row["workers"], row["busy_workers_in_window"])
    forks_per_second = busy / worker_seconds_per_fork
    # Solve forks*(1/rate + 1.25*bytes/pack_rate) = budget.
    seconds_per_fork = 1 / forks_per_second + 1.25 * bytes_per_fork / (packing["mib_per_second"] * 2**20)
    return dict(continuations_per_fork=continuations_per_fork, worker_seconds_per_fork=worker_seconds_per_fork,
                forks_per_hour=3600 * forks_per_second, forks_in_budget=math.floor(budget_seconds / seconds_per_fork),
                expected_accepted_pairs=math.floor(budget_seconds / seconds_per_fork) * saved["accepted_fraction"],
                assumptions="Busy workers from the continuation-only window; source matches measured separately "
                            "with the same pool; saved first-group nonzero fraction and forks per source.")


def run_benchmark(args, parser):
    import psutil
    import torch
    from civ5rl.deploy.archive import unpack
    from civ5rl.engine.resources import ResourceMonitor
    from civ5rl.experiments.rollout_comparison_collection import load_config
    from civ5rl.provenance.common import ROOT

    if args.run is None or args.config is None:
        parser.error("comparison-collection requires --run SAVED_COMPARISON and --config COLLECTION_CONFIG")
    cfg = load_config(args.config)
    cfg = dict(cfg, parents=[str(ROOT / p) for p in cfg["parents"]])
    cfg['inference_services'] = args.inference_services
    cfg['prefix_states'] = args.prefix_states
    source = args.run.resolve()
    from civ5rl.provenance.common import file_sha256
    protected = [Path(p) for p in cfg["parents"]]
    protected += [source / name for name in ("collection.json", "match-config.json", "continuations.storage/manifest.json")]
    protected += [p for name in ("sources.storage", "labels") for p in (source / name).rglob("*") if p.is_file()]
    original_hashes = {p: file_sha256(p) for p in protected}
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Use an empty output")
    if out.is_relative_to(source) or source.is_relative_to(out):
        parser.error("Output must be separate from the saved run")
    workers = [int(x) for x in args.worker_counts.split(",")]
    if (any(w < max(1,args.inference_services) for w in workers)
            or args.inference_services < 0 or args.forks < 1 or args.seconds <= 0):
        parser.error("Invalid worker counts, forks or seconds")
    out.mkdir(parents=True)
    scratch = out / "scratch"
    unpack(source / "sources.storage", scratch / "sources", token="comparison-sources", direction="download")
    collection = read_json(source / "collection.json")
    saved_labels = {}
    for path in (source / "labels").glob("*.json"):
        label = read_json(path)
        saved_labels[label["fork"]["id"]] = label
    completed = [f for f in collection["forks"] if f["status"] == "completed"]
    nonzero = sum(1 for f in completed if saved_labels[f["id"]]["deltas"][0] != 0)
    source_sizes = [sum(p.stat().st_size for p in (scratch / "sources" / str(e)).iterdir()) for e in collection["sources"]]
    manifest = read_json(source / "continuations.storage" / "manifest.json")["files"]
    saved = dict(first_group_nonzero_fraction=nonzero / len(completed),
                 accepted_fraction=sum(saved_labels[f["id"]]["accepted"] for f in completed) / len(completed),
                 forks_per_source=len(collection["forks"]) / len(collection["sources"]),
                 bytes_per_continuation=sum(v["size"] for v in manifest.values()) / (16 * len(completed)),
                 bytes_per_source=statistics.mean(source_sizes))
    configure(17, "cuda")
    warmup = make_records(source, collection, args.warmup_forks, 0)
    measured = make_records(source, collection, args.forks, args.warmup_forks)
    for record in warmup + measured:
        record["side"] = read_json(scratch / "sources" / str(record["episode"]) / "case.json")["learner_side"]
    source_cases = collection["sources"][-args.source_cases:]
    rows = []
    packing = None
    with ResourceMonitor(out / "telemetry", interval=0.5) as monitor:
        for count in workers:
            row = measure_workers(cfg, source, scratch, out, count, measured, warmup, args.seconds,
                                  saved_labels, monitor, source_cases)
            rows.append(row)
            print(json.dumps(row), flush=True)
            if packing is None:
                packing = measure_packing(Path(row["matches_directory"]) / "measured", out)
            shutil.rmtree(row["matches_directory"])
        barrier = None
        if args.barrier_workers:
            barrier = measure_barrier(cfg, source, scratch, out, args.barrier_workers, measured, warmup[0],
                                      args.seconds, monitor)
            shutil.rmtree(out / "matches" / f"barrier-W{args.barrier_workers}")
            print(json.dumps(barrier), flush=True)
    telemetry = [json.loads(line) for line in (out / "telemetry/resources.jsonl").read_text().splitlines() if line.strip()]
    total_ram = psutil.virtual_memory().total
    for row in rows:
        row["resources"] = phase_resources(telemetry, f"W{row['workers']}")
        start = row['measurement_started_elapsed_seconds']
        row['steady_resources'] = phase_resources(telemetry, f"W{row['workers']}",
                                                  start, start+row['window_seconds'])
        row['drain_resources'] = phase_resources(telemetry, f"W{row['workers']}",
                                                 start+row['window_seconds'])
        row["forecast"] = forecast(row, saved, packing)
    if barrier is not None:
        barrier["resources"] = phase_resources(telemetry, f"barrier-W{barrier['workers']}")
    feasible = [r for r in rows if (r["resources"]["peak_ram_tree_rss_gib"] or 0) * 2**30 < 0.85 * total_ram]
    best = max(r["forecast"]["forks_in_budget"] for r in feasible)
    chosen = min((r for r in feasible if r["forecast"]["forks_in_budget"] >= 0.95 * best), key=lambda r: r["workers"])
    recommended = dict(runtime=dict(workers=chosen["workers"], source_share=0.25,
                                    prefix_states=args.prefix_states,
                                    buffer_limit=max(32, chosen["workers"]),
                                    pack_mib_per_second=round(packing["mib_per_second"] * 0.8, 3)),
                       basis="largest forecast forks per 7200 s; smaller W within 5%; peak RSS < 85% RAM; "
                             "packing rate derated by 20%",
                       forecast=chosen["forecast"])
    hardware = dict(cpu=__import__("platform").processor(), logical_cpus=psutil.cpu_count(),
                    physical_cores=psutil.cpu_count(logical=False), ram_gib=total_ram / 2**30,
                    gpu=torch.cuda.get_device_name())
    summary = dict(hardware=hardware, saved_reference=saved, workers=rows, barrier=barrier, packing=packing,
                   recommended=recommended, seconds_per_mode=args.seconds, forks=args.forks,
                   warmup_forks=args.warmup_forks, source_cases=len(source_cases))
    if any(file_sha256(p) != sha for p, sha in original_hashes.items()):
        raise ValueError("Protected comparison benchmark input changed")
    summary.update(preliminary=False, protected_inputs_unchanged=True)
    write_json(out / "summary.json", summary)
    write_json(out / "recommended-runtime.json", recommended)
    shutil.rmtree(scratch)
    write_report(out, summary)
    print(json.dumps(recommended, indent=2), flush=True)
    return summary


def write_report(out, s):
    lines = ["# Comparison collection: worker-count measurement", "",
             f"Hardware: {s['hardware']}. Fixed work: {s['forks']} saved forks (original seeds), "
             f"two-stage shared queue, {s['seconds_per_mode']} s window per mode after warmup "
             f"({s['warmup_forks']} forks). Sources: {s['source_cases']} saved cases on the same pool.", "",
             "| W | cont/s | busy workers | mean cont s | p90 s | source mean s | batch | peak RSS GiB | VRAM GiB | GPU core % | CPU cores | forks/7200 s |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in s["workers"]:
        res = r["resources"]

        def f(v, d=2):
            return "n/a" if v is None else f"{v:.{d}f}"
        lines.append(f"| {r['workers']} | {f(r['continuations_per_second'])} | {f(r['busy_workers_in_window'], 1)} | "
                     f"{f(r['continuation_mean_seconds'])} | {f(r['continuation_p90_seconds'])} | {f(r['source_mean_seconds'])} | "
                     f"{f(r['mean_service_batch'])} | {f(res['peak_ram_tree_rss_gib'])} | {f(res['peak_vram_gib'])} | "
                     f"{f(res['mean_gpu_core_percent'], 0)} | {f(res['mean_cpu_tree_cores'], 1)} | {r['forecast']['forks_in_budget']} |")
    if s["barrier"]:
        b = s["barrier"]
        lines += ["", f"Barrier control (former per-fork mode, 16 continuations per fork, W={b['workers']}): "
                      f"{b['continuations_per_second']:.2f} cont/s, {b['forks_per_second'] * 3600:.0f} forks/h over {b['seconds']:.0f} s."]
    p = s["packing"]
    lines += ["", f"Verified PPMd packing: {p['bytes'] / 2**20:.0f} MiB in {p['seconds']:.1f} s = {p['mib_per_second']:.1f} MiB/s.",
              "", f"Saved reference: {s['saved_reference']}.", "",
              f"Recommended runtime: {s['recommended']['runtime']} ({s['recommended']['basis']}).", "",
              "Forecast = measured busy workers / worker-seconds per fork (expected continuations per fork × mean "
              "continuation duration + source duration / forks per source), minus a 1.25× packing reserve. It is an "
              "estimate: source matches were measured separately, and the run's mixed queue can differ."
              " First-group deltas are compared with the saved run as a reproducibility diagnostic only (different GPU)."]
    lines += ["", "First-group delta reproduction: " + ", ".join(
        f"W{r['workers']} {r['first_delta_reproduced']['matching']}/{r['first_delta_reproduced']['checked']}" for r in s["workers"])]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
