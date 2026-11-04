"""Compatibility alias for civ5rl.protocols.diagnostic."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("civ5rl.protocols.diagnostic")
