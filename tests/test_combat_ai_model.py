"""Run native branch checks against the public combat kernel ABI."""

import os
import subprocess
import unittest
from pathlib import Path


EXECUTABLE = Path(__file__).resolve().parents[1] / "build" / (
    "combat_ai_model_test.exe" if os.name == "nt" else "combat_ai_model_test"
)


@unittest.skipUnless(EXECUTABLE.exists(), "Build tools/build_combat_ai.py first")
class BranchModelTests(unittest.TestCase):
    def check_native(self, name):
        result = subprocess.run(
            [str(EXECUTABLE), name], capture_output=True, text=True, timeout=60
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_movement(self):
        self.check_native("movement")

    def test_terrain_and_zoc(self):
        self.check_native("terrain_and_zoc")

    def test_swaps(self):
        self.check_native("swaps")

    def test_combat(self):
        self.check_native("combat")

    def test_turn_and_healing(self):
        self.check_native("turn_and_healing")

    def test_contact_stops_command_and_allows_continuation(self):
        self.check_native("contact")

    def test_visibility_and_privileged_queries(self):
        self.check_native("visibility")

    def test_ownership_and_real_rng_isolation(self):
        self.check_native("ownership")

    def test_bounds_and_invalid(self):
        self.check_native("bounds_and_invalid")

    def test_random_sequences(self):
        self.check_native("random_sequences")


if __name__ == "__main__":
    unittest.main()
