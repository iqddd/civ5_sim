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
            [str(EXECUTABLE), name], capture_output=True, text=True, timeout=60, check=False
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_movement(self):
        self.check_native("movement")

    def test_delayed_visibility_and_mid_turn_roots(self):
        self.check_native("delayed_visibility")

    def test_extra_damage_weakens_defender_in_both_attack_modes(self):
        self.check_native("extra_defender_damage")

    def test_terrain_and_zoc(self):
        self.check_native("terrain_and_zoc")

    def test_swaps(self):
        self.check_native("swaps")

    def test_combat(self):
        self.check_native("combat")

    def test_turn_and_healing(self):
        self.check_native("turn_and_healing")

    def test_contact_memory_roundtrip_and_isolation(self):
        self.check_native("contact_memory_roundtrip")

    def test_contact_stops_command_and_allows_continuation(self):
        self.check_native("contact")

    def test_visibility_and_privileged_queries(self):
        self.check_native("visibility")

    def test_ownership_and_real_rng_isolation(self):
        self.check_native("ownership")

    def test_adapter_query_preconditions_and_accepted_visibility(self):
        self.check_native("adapter_query_contracts")

    def test_bounds_and_invalid(self):
        self.check_native("bounds_and_invalid")

    def test_random_sequences(self):
        self.check_native("random_sequences")

    def test_danger_damage(self):
        self.check_native("danger_damage")

    def test_danger_movement(self):
        self.check_native("danger_movement")

    def test_danger_heuristics(self):
        self.check_native("danger_heuristics")

    def test_danger_cache_and_contacts(self):
        self.check_native("danger_cache_and_contacts")

    def test_danger_bounds(self):
        self.check_native("danger_bounds")


if __name__ == "__main__":
    unittest.main()
