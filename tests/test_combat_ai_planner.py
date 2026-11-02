"""Behavioral checks for the CP-04 local planner."""
import os
import subprocess
import unittest
from pathlib import Path

EXECUTABLE = Path(__file__).resolve().parents[1] / "build" / (
    "combat_ai_planner_test.exe" if os.name == "nt" else "combat_ai_planner_test"
)


@unittest.skipUnless(EXECUTABLE.exists(), "Build tools/build_combat_ai.py first")
class LocalPlannerTests(unittest.TestCase):
    def check_native(self, name):
        result = subprocess.run(
            [str(EXECUTABLE), name], capture_output=True, text=True, timeout=60, check=False
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_focus_fire_and_finishing(self):
        self.check_native("attacks")

    def test_swaps_freeing_and_omitted_friendlies(self):
        self.check_native("swaps_and_freeing")

    def test_risk_budget_and_numeric_penalty(self):
        self.check_native("risk")

    def test_cover_and_movement_after_attack(self):
        self.check_native("cover_and_retreat")

    def test_budgets_dedup_and_determinism(self):
        self.check_native("budgets_and_determinism")

    def test_search_continues_after_contact(self):
        self.check_native("contact")

    def test_seeded_plans_replay_across_fog_parity_and_wrap(self):
        self.check_native("random_replay")

    def test_thirteen_selected_with_larger_armies_and_tile_255(self):
        self.check_native("scale_and_last_tile")

    def test_support_healing_and_retreat_use_real_rules(self):
        self.check_native("support_healing")

    def test_support_approach_detours_occupancy_and_last_tile(self):
        self.check_native("support_approach")

    def test_cp_area_destinations_visibility_and_preference_ties(self):
        self.check_native("cp_area_and_preferences")

    def test_cp_blocker_intentions_freeing_swaps_and_end_turn_cover(self):
        self.check_native("cp_blocker_combinations")

    def test_cp_pinned_forecast_parameters_and_root_danger_index(self):
        self.check_native("cp_forecast_and_danger")

    def test_cp_damage_history_predicate_and_admission_before_neighborhoods(self):
        self.check_native("cp_history_and_admission")

    def test_cp_area_capacity_admits_participant_center_before_neighbors(self):
        self.check_native("cp_area_capacity")

    def test_planner_target_ids_and_candidate_invariants(self):
        self.check_native("planner_invariants")

    def test_planner_rejected_evaluation_does_not_rewrite_assignments(self):
        self.check_native("planner_finalization")

    def test_planner_contact_source_and_atomic_swap_bookkeeping(self):
        self.check_native("planner_combo_contacts")
