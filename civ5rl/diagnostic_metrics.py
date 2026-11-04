"""Compatibility alias for civ5rl.protocols.diagnostic_metrics."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("civ5rl.protocols.diagnostic_metrics")
