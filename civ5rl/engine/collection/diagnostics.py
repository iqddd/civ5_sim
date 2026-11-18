"""Opt-in collection clocks and bounded samples; never imports torch in workers.

All timestamps are system-wide perf_counter nanoseconds. CPU durations use the
calling thread's clock; neither wall minus CPU nor local queue time identifies
OS preemption. Nested scopes and concurrently running processes overlap.
"""

import math
import os
import random
import threading
from time import (
    get_clock_info,
    perf_counter_ns,
    process_time_ns,
    thread_time_ns,
    time_ns,
)


class Distribution:
    # Eight bins per octave, with a zero bin and a final overflow bin.
    SIZE = 384

    def __init__(self):
        self.count = self.total = self.maximum = 0
        self.minimum = None
        self.bins = [0] * self.SIZE

    def add(self, ns):
        ns = int(ns)
        if ns < 0:
            raise ValueError("Negative diagnostic interval")
        self.count += 1
        self.total += ns
        self.minimum = ns if self.minimum is None else min(self.minimum, ns)
        self.maximum = max(self.maximum, ns)
        index = min(self.SIZE - 1, int(math.log2(ns) * 8) + 1) if ns else 0
        self.bins[index] += 1

    def merge(self, other):
        if not other.count:
            return
        self.count += other.count
        self.total += other.total
        self.minimum = (
            other.minimum if self.minimum is None else min(self.minimum, other.minimum)
        )
        self.maximum = max(self.maximum, other.maximum)
        self.bins = [a + b for a, b in zip(self.bins, other.bins, strict=True)]

    def report(self):
        result = {
            "count": self.count,
            "sum_ns": self.total,
            "min_ns": self.minimum,
            "max_ns": self.maximum,
        }
        for percentile in (50, 95, 99):
            target = math.ceil(self.count * percentile / 100)
            accumulated = 0
            estimate = 0
            for index, count in enumerate(self.bins):
                accumulated += count
                if accumulated >= target:
                    estimate = (
                        min(self.maximum, math.ceil(2 ** (index / 8))) if index else 0
                    )
                    if index == self.SIZE - 1:
                        estimate = self.maximum
                    break
            result[f"p{percentile}_upper_ns"] = estimate
        result["histogram"] = {str(i): n for i, n in enumerate(self.bins) if n}
        return result


class Diagnostics:
    def __init__(self, limit, *, collection=None):
        self.limit = limit
        self.collection = collection
        self.pid = os.getpid()
        self.tid = threading.get_native_id()
        self.started_ns = perf_counter_ns()
        anchor_before = perf_counter_ns()
        anchor_utc = time_ns()
        anchor_after = perf_counter_ns()
        self.clock_anchor = {
            "perf_counter_ns": (anchor_before + anchor_after) // 2,
            "unix_time_ns": anchor_utc,
            "uncertainty_ns": anchor_after - anchor_before,
        }
        self.process_started_ns = process_time_ns()
        self.last_work_checkpoint = (self.started_ns, self.process_started_ns)
        self.metrics = {}
        self.samples = []
        self.seen = 0
        self.rng = random.Random(71983)  # Independent of policy/episode generators.
        self.gauges = {}
        self.requests = {}

    def add(self, name, ns):
        if name not in self.metrics:
            self.metrics[name] = Distribution()
        self.metrics[name].add(ns)

    def begin(self):
        return perf_counter_ns(), thread_time_ns()

    def end(self, name, start, *, event=None):
        # Stop clocks before updating histograms or sampling.
        end_ns, cpu_ns = perf_counter_ns(), thread_time_ns()
        self.add(name + ".wall", end_ns - start[0])
        self.add(name + ".cpu", cpu_ns - start[1])
        if event is not None:
            self.sample(
                dict(
                    kind=name,
                    start_ns=start[0],
                    end_ns=end_ns,
                    cpu_ns=cpu_ns - start[1],
                    **event,
                )
            )
        return end_ns

    def sample(self, event):
        self.seen += 1
        if len(self.samples) < self.limit:
            self.samples.append(event)
        else:
            index = self.rng.randrange(self.seen)
            if index < self.limit:
                self.samples[index] = event

    def gauge(self, name, value, at=None):
        at = perf_counter_ns() if at is None else at
        previous = self.gauges.get(name)
        if previous is None:
            previous = [at, 0, 0, 0]
            self.gauges[name] = previous
        previous[2] += previous[1] * (at - previous[0])
        previous[0], previous[1] = at, value
        previous[3] = max(previous[3], value)

    def checkpoint(self):
        self.last_work_checkpoint = (perf_counter_ns(), process_time_ns())

    def stop(self, at=None, *, cpu_checkpoint=None):
        self.stopped_ns = perf_counter_ns() if at is None else at
        self.cpu_sampled_at_ns, self.stopped_cpu_ns = (
            cpu_checkpoint
            if cpu_checkpoint is not None
            else (perf_counter_ns(), process_time_ns())
        )

    def report(self, *, ended_ns=None, compact=False, include_process_cpu=True):
        ended_ns = perf_counter_ns() if ended_ns is None else ended_ns
        duration = max(1, ended_ns - self.started_ns)
        gauges = {}
        for name, (at, value, area, maximum) in self.gauges.items():
            area += value * (ended_ns - at)
            gauges[name] = {
                "area_ns": area,
                "mean": area / duration,
                "max": maximum,
                "final": value,
            }
        metrics = {name: value.report() for name, value in self.metrics.items()}
        if compact:
            metrics = {
                name: {k: v for k, v in row.items() if k != "histogram"}
                for name, row in metrics.items()
            }
        report = {
            "schema": 1,
            "units": "ns",
            "pid": self.pid,
            "native_thread_id": self.tid,
            "collection": self.collection,
            "started_ns": self.started_ns,
            "ended_ns": ended_ns,
            "clock_anchor": self.clock_anchor,
            "thread_clock_resolution_ns": get_clock_info("thread_time").resolution
            * 1e9,
            "metrics": metrics,
            "gauges": gauges,
            "events": self.samples,
            "event_count": self.seen,
            "event_limit": self.limit,
        }
        if include_process_cpu:
            report["process_cpu_ns"] = (
                getattr(self, "stopped_cpu_ns", process_time_ns())
                - self.process_started_ns
            )
            report["process_cpu_sampled_at_ns"] = getattr(
                self, "cpu_sampled_at_ns", ended_ns
            )
        return report


def merge_workers(workers):
    """Aggregate exact distributions; retain compact per-worker clocks/samples."""
    merged = {}
    reports = []
    for diag in workers:
        for name, value in diag.metrics.items():
            if name not in merged:
                merged[name] = Distribution()
            merged[name].merge(value)
        reports.append(diag.report(compact=True, ended_ns=diag.stopped_ns))
    return {
        "metrics": {name: value.report() for name, value in merged.items()},
        "workers": reports,
    }
