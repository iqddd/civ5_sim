"""RL-01 training; intentionally not imported by the simulator.

Legacy module names resolve lazily to the canonical layered modules.
"""

import importlib
import importlib.abc
import importlib.util
import sys

_ALIASES = {
    **{
        name: "core"
        for name in (
            "agent",
            "context",
            "contracts",
            "env",
            "model",
            "observation",
            "schema",
        )
    },
    **{
        name: "engine"
        for name in ("collector", "control", "ppo", "ppo_fast", "resources", "shared")
    },
    **{
        name: "provenance"
        for name in ("checkpoint", "common", "migration", "validation")
    },
    **{
        name: "protocols"
        for name in (
            "corpus",
            "diagnostic",
            "diagnostic_metrics",
            "diagnostic_one",
            "evaluation",
            "memory",
        )
    },
}


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, canonical):
        self.canonical = canonical

    def create_module(self, spec):
        module = importlib.import_module(self.canonical)
        # Import machinery assigns the alias spec to the returned module.
        self.metadata = (
            module.__name__,
            module.__loader__,
            module.__package__,
            module.__spec__,
        )
        return module

    def exec_module(self, module):
        module.__name__, module.__loader__, module.__package__, module.__spec__ = (
            self.metadata
        )


class _AliasFinder(importlib.abc.MetaPathFinder):
    _civ5rl_alias_finder = True

    def find_spec(self, fullname, path=None, target=None):
        prefix, dot, name = fullname.rpartition(".")
        if prefix == "civ5rl" and dot and name in _ALIASES:
            canonical = f"civ5rl.{_ALIASES[name]}.{name}"
            return importlib.util.spec_from_loader(fullname, _AliasLoader(canonical))
        return None


# Reloading the package must not accumulate obsolete finder instances.
sys.meta_path[:] = [
    finder
    for finder in sys.meta_path
    if not getattr(finder, "_civ5rl_alias_finder", False)
]
sys.meta_path.insert(0, _AliasFinder())
