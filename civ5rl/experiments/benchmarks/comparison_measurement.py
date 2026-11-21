"""Lightweight match workers for bounded comparison measurements."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from time import perf_counter, process_time

from civ5rl.experiments.comparison.match import match_task
from civ5rl.provenance.common import write_json


def measured_match(task):
    # The RPC worker owns both the simulator and its public encoder.
    _, case, match_output, _, _ = task
    samples = []
    wall_started, cpu_started = perf_counter(), process_time()
    result = match_task(task, metrics=samples.append)
    grouped = defaultdict(float)
    custom = defaultdict(float)
    custom_kinds = {"prefix_hook", "replay_verification", "public_encoding"}
    for sample in samples:
        kind = sample["kind"]
        seconds = sample["ns"] / 1e9
        if kind in custom_kinds:
            custom[kind] += seconds
        else:
            grouped[f"{kind}/side{sample['side']}"] += seconds
    write_json(
        Path(match_output) / "measurement.json",
        {
            "wall_seconds": perf_counter() - wall_started,
            "cpu_seconds": process_time() - cpu_started,
            "scopes": dict(grouped),
            "prefix_hook_seconds": custom["prefix_hook"],
            "learner_side": case["learner_side"],
            "replay_verification_seconds": custom["replay_verification"],
            "public_encoding_seconds": custom["public_encoding"],
        },
    )
    return result


def timed_match(task):
    started = perf_counter()
    match_task(task)
    return perf_counter() - started
