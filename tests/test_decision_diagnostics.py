"""General diagnostics: observational parity, provenance, replay and counterfactual isolation."""

import gzip
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from test_cp_agent import SMALL, scenario, unit
from test_maps import field

from civ5sim import Battle, MatchConfig, run_match
from civ5sim.combat_ai import (
    ContactMemory,
    Planner,
    PlanningState,
    SearchRequest,
    SupportMode,
)
from civ5sim.cp_agent import CPAgent
from civ5sim.decision_diagnostics import (
    DecisionRecorder,
    load_package,
    query_with_trace,
    repeat_query,
)
from tools.diagnose_combat import alternative, explain, markdown, verified_states
from tools.resume_combat_replay import resume


class DecisionDiagnosticsTests(unittest.TestCase):
    def match(self, mode):
        board = field(8, 5)
        board["spawn_candidates"] = [[[1, 2], [1, 3]], [[6, 2], [6, 3]]]
        return run_match(MatchConfig(board, [["horseman", "archer"], ["spearman", "archer"]],
                                     3, seed=17, fog=True, random_damage=True),
                         [CPAgent(SMALL), "greedy"], diagnostics=mode)

    def test_recording_modes_do_not_change_match_or_rng(self):
        baseline = self.match("off")
        self.assertEqual(baseline.result["status"], "completed")
        self.assertIsNone(baseline.diagnostic_recorder)
        for mode in ("compact", "full"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                result = self.match(mode)
                self.assertEqual(result.result, baseline.result)
                self.assertEqual(result.replay, baseline.replay)
                self.assertEqual(result.final, baseline.final)
                result.save(directory)
                path = Path(directory)
                package = load_package(path / "diagnostics.json.gz", result.replay)
                self.assertTrue(package["complete"], package["errors"])
                self.assertTrue(package["queries"])
                origins = [c for c in package["calls"] if c.get("origin")]
                self.assertTrue(origins)
                number = origins[-1]["command"]
                report = explain(path / "replay.json", [number], repeat=True)
                self.assertIn("Запрос", markdown(report))
                for item in report["queries"].values():
                    self.assertTrue(item["experiment"]["matches_recorded_result"])
                # A fresh repeat must not mutate the loaded package.
                before = deepcopy(package)
                repeat_query(package, package["queries"][0])
                self.assertEqual(package, before)
                if mode == "full":
                    query = next(q for q in package["queries"] if q.get("trace", {}).get("artifact"))
                    trace = path / "diagnostic-traces" / query["trace"]["artifact"]
                    trace.write_bytes(trace.read_bytes() + b"corruption")
                    with self.assertRaisesRegex(ValueError, "Corrupt diagnostic trace"):
                        query_with_trace(package, query)

    def test_trace_isolation_reset_and_truncation_preserve_results(self):
        with Battle(scenario([unit("horseman", 0, 18, hp=40), unit("spearman", 1, 19)])) as battle:
            before = battle.snapshot()
            state = PlanningState.from_battle(battle, 0)
            request = SearchRequest(state, 0, 0, 19, (0,), ContactMemory().updated(battle.observe(0)))
            with Planner(SMALL) as traced, Planner(SMALL) as plain_planner:
                expected = plain_planner.search(request)
                traced.set_trace(True, max_bytes=64)
                self.assertEqual(traced.search(request), expected)
                self.assertTrue(traced.read_trace()["truncated"])
                self.assertEqual(plain_planner.read_trace()["events"], [])
                traced.set_trace(True)
                self.assertEqual(traced.search(request), expected)
                events = traced.read_trace()["events"]
                self.assertTrue(any(e["event"] == "search_result" for e in events))
                traced.support(request, SupportMode.RECOVERY)
                events = traced.read_trace()["events"]
                self.assertFalse(any(e["event"] == "search_result" for e in events))
                self.assertTrue(any(e["event"].startswith("recovery_") for e in events))
                traced.set_trace(False)
                self.assertEqual(traced.search(request), expected)
                self.assertEqual(traced.read_trace()["events"], [])
            self.assertEqual(battle.snapshot(), before)

    def test_package_mismatch_corruption_and_legacy_report(self):
        outcome = self.match("compact")
        with tempfile.TemporaryDirectory() as directory:
            outcome.save(directory)
            path = Path(directory)
            package = load_package(path / "diagnostics.json.gz", outcome.replay)
            different = deepcopy(outcome.replay)
            different["steps"] = different["steps"][:-1]
            with self.assertRaisesRegex(ValueError, "different replay"):
                load_package(path / "diagnostics.json.gz", different)
            first = next(iter(package["states"]))
            package["states"][first]["turn"] = -123
            with gzip.open(path / "diagnostics.json.gz", "wt", encoding="utf-8") as stream:
                json.dump(package, stream)
            with self.assertRaisesRegex(ValueError, "Corrupt"):
                load_package(path / "diagnostics.json.gz", outcome.replay)
            (path / "diagnostics.json.gz").unlink()
            report = explain(path / "replay.json", [1])
            self.assertTrue(report["commands"][0]["missing"])
            self.assertEqual(report["queries"], {})
            with self.assertRaisesRegex(ValueError, "1-based"):
                verified_states(outcome.replay, [0])

    def test_alternative_legality_and_no_source_mutation(self):
        outcome = self.match("off")
        before = deepcopy(outcome.replay)
        first = outcome.replay["steps"][0]["action"]
        result = alternative(outcome.replay, 1, [first])
        self.assertEqual(result["steps"][0], outcome.replay["steps"][0])
        with self.assertRaisesRegex(ValueError, "Illegal alternative"):
            alternative(outcome.replay, 1, [first, {"kind": "melee", "unit": 999, "target": 0}])
        response = alternative(outcome.replay, 1, [{"kind": "end_turn", "unit": -1, "target": -1}],
                               reply="greedy", match_result=outcome.result)
        self.assertTrue(response["steps"])
        self.assertEqual(outcome.replay, before)

    def test_continuation_records_only_new_calls(self):
        board = field(8, 5)
        board["spawn_candidates"] = [[[0, 2]], [[7, 2]]]
        original = run_match(MatchConfig(board, [["warrior"], ["archer"]], 2, seed=17), ["greedy", "cp"])
        boundary = next(i for i, step in enumerate(original.replay["steps"], 1)
                        if step["action"]["kind"] == "end_turn")
        with tempfile.TemporaryDirectory() as directory:
            original.save(directory)
            source = Path(directory) / "replay.json"
            off = resume(source, boundary, search_config=SMALL)
            full = resume(source, boundary, search_config=SMALL, diagnostics="full")
            self.assertEqual(off.replay, full.replay)
            self.assertEqual(off.final, full.final)
            self.assertTrue(all(c["command"] > boundary for c in full.diagnostic_recorder.calls))
            self.assertFalse(full.diagnostic_recorder.errors)

    def test_diagnostic_error_does_not_change_result(self):
        recorder = DecisionRecorder("compact")
        # Deliberately invalid call sequence is isolated and marked incomplete.
        recorder.event({"event": "test"}, None)
        self.assertTrue(recorder.errors)
        self.assertIsNone(recorder.current)


if __name__ == "__main__":
    unittest.main()
