"""A resumed match preserves physical history, RNG and greedy observations."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from test_cp_agent import SMALL
from test_maps import field

from civ5sim import Battle, MatchConfig, run_match
from civ5sim.combat_ai import LIBRARY_PATH
from civ5sim.engine import digest
from tools.resume_combat_replay import resume


@unittest.skipUnless(LIBRARY_PATH.exists(), "Build combat and AI libraries first")
class ResumeReplayTests(unittest.TestCase):
    def test_verified_prefix_boundary_rng_and_separate_output(self):
        m = field(8, 5)
        m["spawn_candidates"] = [[[0, 2]], [[7, 2]]]
        original = run_match(MatchConfig(m, [["warrior"], ["archer"]], 3, seed=17,
                                        fog=True, random_damage=True), ["greedy", "cp"])
        boundary = next(i for i, e in enumerate(original.replay["steps"], 1)
                        if e["action"]["kind"] == "end_turn")
        with tempfile.TemporaryDirectory() as directory:
            original.save(directory)
            source = Path(directory) / "replay.json"
            before = source.read_bytes()
            continued = resume(source, boundary, search_config=SMALL)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(continued.result["status"], "completed")
            self.assertEqual(continued.replay["steps"][:boundary], original.replay["steps"][:boundary])
            with Battle(original.replay["initial"]) as battle:
                for entry in original.replay["steps"][:boundary]:
                    self.assertTrue(battle.step(entry["action"]).ok)
                self.assertEqual(digest(battle.snapshot()),
                                 continued.result["continuation"]["checkpoint_sha256"])
            # A valid command with a changed expected result must fail before
            # any new decisions; an end_turn marker alone is not enough.
            damaged = deepcopy(original.replay)
            damaged["steps"][boundary-1]["state_sha256"] = "wrong RNG/state"
            source.write_text(json.dumps(damaged), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Source replay differs"):
                resume(source, boundary, search_config=SMALL)
            source.write_bytes(before)
            with self.assertRaisesRegex(ValueError, "end_turn"):
                resume(source, boundary-1, search_config=SMALL)


if __name__ == "__main__":
    unittest.main()
