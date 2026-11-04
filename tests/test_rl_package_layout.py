import importlib
import pickle
import subprocess
import sys
import unittest
from unittest.mock import patch

from civ5rl.provenance.checkpoint import artifact_sources

LAYERS = {
    "core": ("schema", "contracts", "context", "observation", "model", "agent", "env"),
    "engine": ("collector", "shared", "ppo", "ppo_fast", "control", "resources"),
    "provenance": ("common", "checkpoint", "migration", "validation"),
    "protocols": (
        "corpus",
        "evaluation",
        "memory",
        "diagnostic",
        "diagnostic_one",
        "diagnostic_metrics",
    ),
}


class PackageLayoutTests(unittest.TestCase):
    def test_aliases_are_the_same_module_including_private_names(self):
        for layer, names in LAYERS.items():
            for name in names:
                old = importlib.import_module(f"civ5rl.{name}")
                canonical = importlib.import_module(f"civ5rl.{layer}.{name}")
                self.assertIs(old, canonical)
        import civ5rl.collector as legacy
        import civ5rl.engine.collector as canonical

        with patch("civ5rl.collector._worker", object()) as sentinel:
            self.assertIs(canonical._worker, sentinel)
            self.assertIs(legacy._worker, sentinel)

    def test_worker_import_is_torch_free(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import civ5rl.engine.collector; import sys; assert 'torch' not in sys.modules",
            ],
            check=True,
        )

    def test_old_encoded_pickle_path_loads(self):
        import numpy as np

        from civ5rl.core.observation import Encoded

        arr = np.zeros(0)
        obj = Encoded(arr, arr, arr, arr, arr, arr, arr)
        # Produce the same global reference emitted by historical pickles.
        with patch.object(Encoded, "__module__", "civ5rl.observation"):
            data = pickle.dumps(obj)
        restored = pickle.loads(data)
        self.assertIs(type(restored), Encoded)

    def test_nested_modules_are_in_passport(self):
        from civ5rl.provenance.common import ROOT

        paths = {p.relative_to(ROOT).as_posix() for p in artifact_sources({})}
        self.assertIn("civ5rl/core/observation.py", paths)
        self.assertIn("civ5rl/engine/collector.py", paths)
        self.assertIn("civ5rl/provenance/__init__.py", paths)
        self.assertIn("civ5rl/protocols/diagnostic.py", paths)
