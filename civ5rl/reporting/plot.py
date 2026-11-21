"""Headless plotting for report writers."""

from __future__ import annotations


def pyplot():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    return pyplot
