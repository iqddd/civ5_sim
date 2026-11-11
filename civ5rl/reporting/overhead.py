"""Additive active-time accounting; nested timers and process gaps stay separate."""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from civ5rl.provenance.common import write_json


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _account(rows, active_seconds):
    collection = sum(r["collection"]["collection_seconds"] for r in rows)
    operation = sum(r["ppo"]["operation_seconds"] for r in rows)
    callbacks = sum(r["ppo"]["runtime_compilation_seconds"] for r in rows)
    return {
        "decisions": sum(r["decisions"] for r in rows),
        "active_seconds": active_seconds,
        "collection_seconds": collection,
        "ppo_dynamo_callback_seconds": callbacks,
        "ppo_without_callbacks_seconds": operation - callbacks,
        "unattributed_active_seconds": active_seconds - collection - operation,
    }


ADDITIVE = (
    "collection_seconds", "ppo_dynamo_callback_seconds",
    "ppo_without_callbacks_seconds", "unattributed_active_seconds",
)


def summarize_overhead(directory, *, reference_start, reference_stop, through_update=None, deployments=()):
    directory = Path(directory)
    rows = _rows(directory / "updates.jsonl")
    if not rows or len({r["rollout"] for r in rows}) != len(rows):
        raise ValueError("Need nonempty, unique committed updates")
    state = _read(directory / "state.json")
    totals = _account(rows, state["active_seconds"])
    selected = [r for r in rows if reference_start <= r["rollout"] <= reference_stop]
    if len(selected) != reference_stop - reference_start + 1:
        raise ValueError("Reference must be a complete contiguous rollout range")
    first = rows.index(selected[0])
    start_active = rows[first - 1]["active_seconds"] if first else 0
    reference = _account(selected, selected[-1]["active_seconds"] - start_active)
    reference["rollouts"] = [r["rollout"] for r in selected]
    reference["decisions_per_second"] = reference["decisions"] / reference["active_seconds"]

    def excess(subset, active_seconds):
        observed = _account(subset, active_seconds)
        scale = observed["decisions"] / reference["decisions"]
        delta = {key: observed[key] - reference[key] * scale for key in ADDITIVE}
        return {
            "observed": observed,
            "reference_equivalent_seconds": reference["active_seconds"] * scale,
            "excess_seconds": sum(delta.values()),
            "excess_components_seconds": delta,
        }

    nested = {
        key: sum(r["ppo"].get(key, 0) or 0 for r in rows)
        for key in ("cache_seconds", "prefix_seconds", "backward_seconds", "optimizer_seconds", "kl_seconds", "transaction_seconds")
    }
    nested["checked_step_records"] = sum(
        bool(step.get("full_kl_checked", True))
        for r in rows for step in r["ppo"]["optimizer_step_records"]
    )
    nested["attempted_optimizer_steps"] = sum(r["ppo"]["attempted_optimizer_steps"] for r in rows)

    # Phase intervals can include failed attempts with a subsequently committed
    # rollout ID. Use events only to locate session starts, not to balance timers.
    starts, session_times = {}, {}
    for event in _rows(directory / "telemetry/phases.jsonl"):
        rollout = event.get("rollout", 0)
        if rollout > 0:
            starts.setdefault(event["session_id"], rollout)
            if event.get("timestamp_utc"):
                session_times.setdefault(event["session_id"], event["timestamp_utc"])
    start_ids = set(starts.values())
    startup_callbacks = sum(r["ppo"]["runtime_compilation_seconds"] for r in rows if r["rollout"] in start_ids)
    modes = defaultdict(lambda: {"updates": [], "decisions": 0, "operation_seconds": 0, "kl_seconds": 0})
    for row in rows:
        group = modes[row["ppo"].get("replay_mode", "unrecorded")]
        group["updates"].append(row["rollout"])
        group["decisions"] += row["decisions"]
        group["operation_seconds"] += row["ppo"]["operation_seconds"]
        group["kl_seconds"] += row["ppo"]["kl_seconds"]

    jobs = {}
    for deployment in deployments:
        tuning = Path(deployment) / "tuning/summary.json"
        cpu = _read(tuning).get("hardware", {}).get("CPU") if tuning.exists() else None
        for path in Path(deployment).glob("job-*.json"):
            job = _read(path)
            execution = job.get("remote_execution", {})
            start, stop = execution.get("started_at_utc"), execution.get("finished_at_utc")
            command = job.get("remote_command", "")
            if not start or not stop or "--mode tune" in command or job.get("remote_sync_dir", "").endswith("/tuning"):
                continue
            jobs[job["job_id"]] = {
                "job_id": job["job_id"], "metadata": str(path),
                "started_at_utc": start, "finished_at_utc": stop,
                "exit_code": job.get("execution", {}).get("exit_code"),
                "cpu": cpu,
                "seconds": (datetime.fromisoformat(stop) - datetime.fromisoformat(start)).total_seconds(),
            }
    ordered = sorted(jobs.values(), key=lambda j: j["started_at_utc"])
    gaps = [
        max(0, (datetime.fromisoformat(b["started_at_utc"]) - datetime.fromisoformat(a["finished_at_utc"])).total_seconds())
        for a, b in zip(ordered, ordered[1:])
    ]
    hosts = defaultdict(list)
    session_order = sorted(starts, key=lambda key: starts[key])
    for index, session in enumerate(session_order):
        first = starts[session]
        stop = starts[session_order[index + 1]] if index + 1 < len(session_order) else float("inf")
        when = datetime.fromisoformat(session_times[session]) if session in session_times else None
        job = next((j for j in ordered if when is not None and datetime.fromisoformat(j["started_at_utc"]) <= when <= datetime.fromisoformat(j["finished_at_utc"])), None)
        cpu = job.get("cpu") if job else None
        hosts[cpu or "unattributed"].extend(r for r in rows if first <= r["rollout"] < stop)
    host_summary = {}
    for cpu, group in hosts.items():
        collection = sum(r["collection"]["collection_seconds"] for r in group)
        decisions = sum(r["decisions"] for r in group)
        host_summary[cpu] = {"updates": len(group), "decisions": decisions,
                             "collection_seconds": collection,
                             "collection_decisions_per_second": decisions / collection if collection else None}
    prefix = [r for r in rows if through_update is not None and r["rollout"] <= through_update]
    record = _read(directory / "run.json") if (directory / "run.json").exists() else {}
    saved_plan = record.get("time_budget_check", {})
    return {
        "schema": 1, "run": str(directory), "status": _read(directory / "status.json"),
        "updates": len(rows), "additive_active_accounting": totals,
        "nested_ppo_timers_not_additive": nested, "reference": reference,
        "reference_comparison": excess(rows, state["active_seconds"]),
        "prefix_rollout": through_update,
        "prefix_reference_comparison": excess(prefix, prefix[-1]["active_seconds"]) if prefix else None,
        "sessions": {"first_rollouts": sorted(start_ids), "startup_callback_seconds": startup_callbacks,
                     "other_callback_seconds": totals["ppo_dynamo_callback_seconds"] - startup_callbacks},
        "replay_modes_observational": dict(modes),
        "host_collection_observational": host_summary,
        "saved_kl_plan_review": {
            "claimed_checks_upper_bound": saved_plan.get("planned_kl_checks_upper_bound"),
            "checked_post_step_records": nested["checked_step_records"],
            "prechecks_and_rollback_calls_not_in_step_records": True,
        },
        "processes": {"jobs": ordered, "execution_wall_seconds": sum(j["seconds"] for j in ordered),
                      "gaps_between_jobs_seconds": sum(gaps), "individual_gap_seconds": gaps},
        "limits": [
            "Additive accounting uses committed-update timers and saved ActiveClock only; residual is explicitly unattributed.",
            "KL/backward/prefix/cache timers are nested and may include compilation or asynchronous GPU work; do not add them to active components.",
            "Reference scaling is descriptive, not a controlled estimate of optimal speed or a causal host comparison.",
            "Dynamo callbacks include tracing/backend work; session-start callbacks are not wholly avoidable overhead.",
            "Telemetry phase totals may include failed/retried attempts and are not interchangeable with committed-update timers.",
            "Inter-process gaps include deliberate pauses, preparation, synchronization and observation delays; they are not GPU training time or a billing estimate.",
            "Replay-mode comparisons also differ in rollout size, KL-check count, shapes and process warmup; no causal speedup follows from these aggregates.",
        ],
    }


def write_overhead(directory, output, **kwargs):
    from civ5rl.reporting.operations import derived_output

    output = derived_output(directory, output)
    result = summarize_overhead(directory, **kwargs)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "overhead.json", result)
    lines = ["# PPO active-time accounting", "", "| Component | Seconds | Minutes |", "|---|---:|---:|"]
    for key in (*ADDITIVE, "active_seconds"):
        value = result["additive_active_accounting"][key]
        lines.append(f"| {key} | {value:.2f} | {value / 60:.2f} |")
    lines += ["", "## Reference comparison", "",
              f"Rollouts {result['reference']['rollouts']}: {result['reference']['decisions_per_second']:.2f} decisions/s.",
              "", "| Slice | Observed, min | Reference equivalent, min | Excess, min |", "|---|---:|---:|---:|"]
    for key in ("prefix_reference_comparison", "reference_comparison"):
        item = result[key]
        if item:
            lines.append(f"| {key} | {item['observed']['active_seconds']/60:.2f} | {item['reference_equivalent_seconds']/60:.2f} | {item['excess_seconds']/60:.2f} |")
    lines += ["", "Excess components relative to the reference, minutes (descriptive, not causal):", "",
              "| Component | Prefix | Whole run |", "|---|---:|---:|"]
    prefix = result["prefix_reference_comparison"]
    for key in ADDITIVE:
        a = f"{prefix['excess_components_seconds'][key]/60:.2f}" if prefix else "n/a"
        b = result["reference_comparison"]["excess_components_seconds"][key] / 60
        lines.append(f"| {key} | {a} | {b:.2f} |")
    sessions = result["sessions"]
    lines += ["", "## Process starts and nested PPO costs", "",
              f"First rollouts per telemetry session: {sessions['first_rollouts']}.",
              f"Callbacks on first rollouts: {sessions['startup_callback_seconds']/60:.2f} min; other callbacks: {sessions['other_callback_seconds']/60:.2f} min.",
              "", "| Nested timer (do not add to the active components) | Minutes |", "|---|---:|"]
    for key, value in result["nested_ppo_timers_not_additive"].items():
        if key.endswith("seconds"):
            lines.append(f"| {key} | {value/60:.2f} |")
    processes = result["processes"]
    lines += ["", f"Finished training/run/resume processes with supplied metadata: {len(processes['jobs'])}.",
              f"Their execution wall time: {processes['execution_wall_seconds']/60:.2f} min, including evaluations/startup.",
              f"Gaps between processes: {processes['gaps_between_jobs_seconds']/60:.2f} min, including deliberate pauses and host preparation; not all are failure downtime."]
    lines += ["", "## Host collection (observational)", "", "| CPU | Updates | Decisions/s during collection |", "|---|---:|---:|"]
    for cpu, item in result["host_collection_observational"].items():
        rate = item["collection_decisions_per_second"]
        lines.append(f"| {cpu} | {item['updates']} | {rate:.2f} |")
    plan = result["saved_kl_plan_review"]
    if plan["claimed_checks_upper_bound"] is not None:
        lines += ["", f"Saved claimed KL-check upper bound: {plan['claimed_checks_upper_bound']}; observed checked post-step records: {plan['checked_post_step_records']} (prechecks and rollback verification are additional)."]
        if plan["checked_post_step_records"] > plan["claimed_checks_upper_bound"]:
            lines.append("The saved claimed bound was exceeded. The historical single-spike estimate did not bound the sticky per-step safeguard; historical run.json is preserved.")
    lines += ["", "## Interpretation limits", "", *[f"- {s}" for s in result["limits"]], ""]
    (output / "overhead.md").write_text("\n".join(lines), encoding="utf-8")
    return result
