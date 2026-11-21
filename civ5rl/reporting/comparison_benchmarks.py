"""Render comparison benchmark and precheck reports from saved results."""

from __future__ import annotations

import json

from civ5rl.provenance.common import read_json


def format_measurement(aggregates, name, *, scale=1, suffix="", precision=1):
    value = aggregates.get(name, {}).get("mean")
    return "n/a" if value is None else f"{value / scale:.{precision}f}{suffix}"


def write_performance_report(out):
    result = read_json(out / "summary.json")
    hardware = read_json(out / "identity.json")["hardware"]
    factory_results = read_json(out / "factory.json")
    map_sizes = result["training"]["map_sizes"]
    rows = [
        "# CPU/GPU rollout-comparison benchmark",
        "",
        f"Local {hardware['gpu']} ({hardware['gpu_limit_watts']:g}W limit), CUDA deterministic FP32 inference / mixed BF16 spatial training. No scientific training or evaluation; all measured Adam steps are discarded. Startup/warmup excluded.",
        "",
        "| Phase | Wall | GPU Core | Memory Controller | Power | CPU logical cores |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in [
        (
            "source_collection",
            f"{result['source_collection']['matches']} source matches",
        ),
        ("continuations", f"{result['continuations']['matches']} continuations"),
        (
            "training",
            f"{result['training']['logical_pairs']} pairs+{result['training']['logical_anchors']} anchors per batch",
        ),
    ]:
        phase_measurement = result[key]
        resource_means = phase_measurement["nvml_cpu"]
        seconds = phase_measurement.get("seconds", phase_measurement.get("mean_seconds"))
        rows.append(
            f"| {label} | {seconds:.2f}s | {format_measurement(resource_means, 'gpu_core_load_percent', suffix='%')} | {format_measurement(resource_means, 'gpu_memory_controller_load_percent', suffix='%')} | {format_measurement(resource_means, 'gpu_power_watts', suffix='W')} | {format_measurement(resource_means, 'cpu_tree_percent', scale=100, precision=2)} |"
        )
    rows += [
        "",
        "Memory Controller Load measures activity, not occupied VRAM or percent of peak GB/s. CPU cores are process-tree CPU percent/100, not hardware occupancy. Collection CP worker scopes and inference service overlap and must not be added to phase wall time. CP search is nested inside act.",
        "",
        "Source inference service measurements: "
        + str(result["source_collection"]["service"])
        + ". Continuations: "
        + str(result["continuations"]["service"])
        + ". Worker model-act scopes include RPC waiting; worker process CPU time separates actual CPU consumption from waiting.",
        "",
        f"Training uses {result['training']['unique_targets']} distinct target positions across {len(map_sizes)} episodes; map sizes: {', '.join(str(size) for size in sorted(set(map_sizes)))}. {len(result['training']['step_samples'])} timed batches follow warmup; samples and NVML are in summary.json/telemetry. Timed training excludes reference preparation and full KL checks. Each batch starts from the same restored model/Adam.",
        "",
        "Kineto traces are short, separate profiled fragments and include profiler overhead: their GPU busy ratio is not the unprofiled phase utilization. Inspect top kernels, CPU launch gaps, DMA and synchronization times in trace-summary.json to attribute the remaining cost.",
        "",
        "Inspect cpu_system_percent before extrapolating to another machine. Assess each iterative GPU phase against the utilization policy using its measured duration.",
        "",
        "The measurement uses the recorded compiled runtime, compact attention and batched independent histories. Logical batch, training budget and precision are unchanged; the execution strategy is recorded in identity.json.",
        "",
        "CPU-only cold/warm scenario generation: "
        + str(factory_results)
        + ". This cost is separate from saved-case match timing.",
        "",
        "Raw match JSON and Kineto traces are stored in the sibling PPMd packet after verified packing; summary, telemetry and this report remain readable.",
    ]
    (out / "report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_collection_report(out):
    summary = read_json(out / "summary.json")
    lines = [
        "# Comparison collection: worker-count measurement",
        "",
        f"Hardware: {summary['hardware']}. Fixed work: {summary['forks']} saved forks (original seeds), "
        f"two-stage shared queue, {summary['seconds_per_mode']} s window per mode after warmup "
        f"({summary['warmup_forks']} forks). Sources: {summary['source_cases']} saved cases on the same pool.",
        "",
        f"| W | cont/s | busy workers | mean cont s | p90 s | source mean s | batch | peak RSS GiB | VRAM GiB | GPU core % | CPU cores | forks/{summary['budget_seconds']} s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for worker_trial in summary["workers"]:
        res = worker_trial["resources"]

        def format_value(value, digits=2):
            return "n/a" if value is None else f"{value:.{digits}f}"

        lines.append(
            f"| {worker_trial['workers']} | {format_value(worker_trial['continuations_per_second'])} | {format_value(worker_trial['busy_workers_in_window'], 1)} | "
            f"{format_value(worker_trial['continuation_mean_seconds'])} | {format_value(worker_trial['continuation_p90_seconds'])} | {format_value(worker_trial['source_mean_seconds'])} | "
            f"{format_value(worker_trial['mean_service_batch'])} | {format_value(res['peak_ram_tree_rss_gib'])} | {format_value(res['peak_vram_gib'])} | "
            f"{format_value(res['mean_gpu_core_percent'], 0)} | {format_value(res['mean_cpu_tree_cores'], 1)} | {worker_trial['forecast']['forks_in_budget']} |"
        )
    if summary["barrier"]:
        barrier = summary["barrier"]
        lines += [
            "",
            f"Barrier control (former per-fork mode, 16 continuations per fork, W={barrier['workers']}): "
            f"{barrier['continuations_per_second']:.2f} cont/s, {barrier['forks_per_second'] * 3600:.0f} forks/h over {barrier['seconds']:.0f} s.",
        ]
    packing = summary["packing"]
    lines += [
        "",
        f"Verified PPMd packing: {packing['bytes'] / 2**20:.0f} MiB in {packing['seconds']:.1f} s = {packing['mib_per_second']:.1f} MiB/s.",
        "",
        f"Saved reference: {summary['saved_reference']}.",
        "",
        f"Recommended runtime: {summary['recommended']['runtime']} ({summary['recommended']['basis']}).",
        "",
        "Forecast = measured busy workers / worker-seconds per fork (expected continuations per fork × mean "
        "continuation duration + source duration / forks per source), minus a 1.25× packing reserve. It is an "
        "estimate: source matches were measured separately, and the run's mixed queue can differ."
        " First-group deltas are compared with the saved run as a reproducibility diagnostic only (different GPU).",
    ]
    lines += [
        "",
        "First-group delta reproduction: "
        + ", ".join(
            f"W{worker_trial['workers']} {worker_trial['first_delta_reproduced']['matching']}/{worker_trial['first_delta_reproduced']['checked']}"
            for worker_trial in summary["workers"]
        ),
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_precheck_report(out):
    result = read_json(out / "precheck.json")
    forks = result["real_completed_forks"]
    collection = (
        f"The precheck reused {forks} completed forks from the frozen collection."
        if result["reused_frozen_collection"]
        else f"The precheck completed {forks} real forks and verified their continuation replays."
    )
    (out / "report.md").write_text(
        "# Rollout comparison preparation\n\n"
        + collection
        + " "
        + "The synthetic fit used a disposable copy.\n\n```json\n"
        + json.dumps(result, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
