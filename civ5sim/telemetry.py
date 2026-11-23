"""Optional timing samples, kept outside deterministic match artifacts."""

from contextlib import contextmanager
from time import perf_counter_ns


@contextmanager
def measure(collector, kind, **labels):
    if collector is None:
        yield
        return
    clock = getattr(collector, "clock_ns", perf_counter_ns)
    start = clock()
    try:
        yield
    finally:
        collector({"kind": kind, **labels, "ns": clock() - start})
