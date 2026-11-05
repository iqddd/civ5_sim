"""Workers stay CPU-only; real historical rollout pickles remain readable."""

import pickle
import subprocess
import sys
import unittest
from unittest.mock import patch


class ImportBoundaryTests(unittest.TestCase):
    def test_worker_import_is_torch_free(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import civ5rl.engine.collector, civ5rl.engine.stream_worker, civ5rl.core.env; import sys; assert 'torch' not in sys.modules",
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
