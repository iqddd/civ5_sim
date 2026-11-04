"""Compatibility alias for civ5rl.core.env."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("civ5rl.core.env")
