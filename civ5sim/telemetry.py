"""Optional wall-clock samples, kept outside deterministic match artifacts."""

from contextlib import contextmanager
from time import perf_counter_ns


@contextmanager
def measure(collector, kind, **labels):
    if collector is None:
        yield
        return
    start = perf_counter_ns()
    try:
        yield
    finally:
        collector({"kind": kind, **labels, "ns": perf_counter_ns() - start})
