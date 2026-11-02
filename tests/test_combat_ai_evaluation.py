"""CP-07 baseline, telemetry, corpus and independent-context regressions."""

import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from civ5sim import Battle, MatchConfig, run_match
from civ5sim.agents import load_agent
from civ5sim.combat_ai import (
    CallPlanningServices,
    ContactMemory,
    Planner,
    PlanningState,
    SearchConfig,
    SearchRequest,
    SupportMode,
    _Assignment,
)
from civ5sim.cp_agent import CPAgent
from civ5sim.evaluation import (
    cases,
    distribution,
    load_suite,
    reusable_check,
    timing_summary,
    verify_replay,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
SMALL = SearchConfig(position_capacity=180, max_completed=12)


class EvaluationTests(unittest.TestCase):
    def suite(self):
        return load_suite(ROOT / "scenarios/combat-ai-v1.json")

    def config(self, seed=17):
        m = self.suite()["fields"][0]["map"]
        m["spawn_candidates"] = [[[5, 7], [5, 8]], [[8, 7], [8, 8]]]
        return MatchConfig(m, [["warrior", "archer"], ["spearman", "archer"]],
                           6, seed, max_calls_per_turn=100)

    def test_corpus_pairs_preserve_placement_and_cover_modes(self):
        corpus = cases(self.suite())
        self.assertEqual(len(corpus), 148)
        self.assertEqual(len({c["id"] for c in corpus}), 148)
        for first, second in zip(corpus[:32:2], corpus[1:32:2]):
            self.assertEqual(first["config"], second["config"])
            self.assertEqual(first["agents"], list(reversed(second["agents"])))
        self.assertEqual({(c["config"]["fog"], c["config"]["random_damage"]) for c in corpus},
                         {(False, False), (False, True), (True, False), (True, True)})

    def test_greedy_targets_low_hp_and_only_uses_legal_commands(self):
        config = self.config()
        from civ5sim.match import create_scenario
        scenario = create_scenario(config)
        scenario["units"] = [
            {"type": "archer", "side": 0, "tile": 7*16+6},
            {"type": "warrior", "side": 1, "tile": 7*16+7, "hp": 90},
            {"type": "warrior", "side": 1, "tile": 8*16+7, "hp": 10},
        ]
        scenario["fog"] = False
        agent = load_agent("greedy")
        context = {"map": scenario["map"], "side": 0}
        agent.reset(context)
        with Battle(scenario) as battle:
            actions = tuple(battle.available_actions())
            selected = agent.act(battle.observe(), actions)
            self.assertEqual(selected.target, 8*16+7)
            self.assertIn(selected, actions)
            self.assertTrue(battle.step(selected).ok)
        first = run_match(config, ["greedy", "greedy"])
        self.assertEqual(first.result["status"], "completed")
        self.assertEqual(first, run_match(config, ["greedy", "greedy"]))
        verify_replay(first)

    def test_local_combat_preserves_ranged_shot_across_water_barrier(self):
        from civ5sim.evaluation import synthetic_map
        m = synthetic_map()
        for y in range(15):
            m["tiles"][y*16+2].update(terrain="coast", elevation="water")
        scenario = {"schema": 1, "kind": "civ5sim.scenario", "map": m,
                    "fog": False, "random_damage": False, "seed": 17,
                    "units": [{"type": "archer", "side": 0, "tile": 17},
                              {"type": "warrior", "side": 1, "tile": 19}]}
        agent = CPAgent(SMALL)
        try:
            with Battle(scenario) as battle:
                agent.reset({"map": m, "side": 0})
                self.assertGreater(agent._distance(17, 19), 6)
                service = CallPlanningServices(agent.planner, PlanningState.from_battle(battle, 0), 0, 0)
                try:
                    action = agent.act_with_planning(battle.observe(), tuple(battle.available_actions()), service)
                finally:
                    service.expire()
                self.assertEqual(action.kind, "ranged")
                self.assertEqual(action.target, 19)
        finally:
            agent.close()

    def test_telemetry_does_not_change_artifacts_and_covers_services(self):
        samples = []
        agents = [CPAgent(SMALL), CPAgent(SMALL)]
        try:
            measured = run_match(self.config(), agents, metrics=samples.append)
            plain = run_match(self.config(), agents)
            self.assertEqual(measured, plain)
            self.assertTrue({"act", "callback", "search", "support", "prepare_planning"}
                            <= {s["kind"] for s in samples})
            self.assertTrue(all(s["ns"] >= 0 for s in samples))
            summary = timing_summary(samples)
            self.assertEqual(summary["side0/decision_ms"]["count"], measured.result["agent_calls"][0])
            self.assertNotIn("metrics", measured.replay)
        finally:
            for a in agents:
                a.close()

    def test_reset_and_interleaved_instances_match_fresh_runs(self):
        agents = [CPAgent(SMALL), CPAgent(SMALL)]
        try:
            baseline = run_match(self.config(), agents)
            run_match(self.config(42), list(reversed(agents)))
            self.assertEqual(baseline, run_match(self.config(), agents))
            fresh = [CPAgent(SMALL), CPAgent(SMALL)]
            try:
                self.assertEqual(baseline, run_match(self.config(), fresh))
            finally:
                for a in fresh:
                    a.close()
            from civ5sim.match import create_scenario
            with Battle(create_scenario(self.config())) as battle:
                state = PlanningState.from_battle(battle, 0)
                service = CallPlanningServices(agents[0].planner, state, 0, state.turn)
                own = tuple(u.id for u in state.units if u.side == 0)
                target = next(u.tile for u in state.units if u.side == 1)
                before = service.search(target, own, contacts=ContactMemory())
                service2 = CallPlanningServices(agents[1].planner, state, 0, state.turn)
                service2.support(own[0], own, contacts=ContactMemory(), mode=SupportMode.APPROACH)
                self.assertEqual(before, service.search(target, own, contacts=ContactMemory()))
                service.expire()
                service2.expire()
        finally:
            for a in agents:
                a.close()

    def test_statistics_include_partial_turn_without_double_counting_search(self):
        samples = [{"side": 0, "turn": 0, "kind": k, "ns": n * 1_000_000}
                   for k, n in (("prepare_actions", 1), ("search", 3), ("act", 5),
                                ("callback", 2))]
        summary = timing_summary(samples)
        self.assertEqual(summary["side0/decision_ms"]["median"], 8)
        self.assertEqual(summary["side0/turn_ms"]["median"], 8)
        self.assertEqual(distribution([])["count"], 0)
        self.assertEqual(distribution([1, 3, 2, 4])["median"], 2.5)
        self.assertEqual(asdict(SMALL)["position_capacity"], 180)

    def test_output_buffer_avoids_second_search_and_grows_without_changing_plan(self):
        from civ5sim.match import create_scenario
        with Battle(create_scenario(self.config())) as battle, Planner(SMALL) as planner:
            state = PlanningState.from_battle(battle, 0)
            request = SearchRequest(state, 0, state.turn,
                                    next(u.tile for u in state.units if u.side == 1),
                                    (0, 1), ContactMemory())
            original = planner._lib.cai_search
            with patch.object(planner._lib, "cai_search", wraps=original) as invoke:
                expected = planner.search(request)
                self.assertTrue(expected.assignments)
                self.assertEqual(invoke.call_count, 1)

                planner._assignment_buffer = (_Assignment * 0)()
                invoke.reset_mock()
                self.assertEqual(expected, planner.search(request))
                self.assertEqual(invoke.call_count, 2)
                invoke.reset_mock()
                self.assertEqual(expected, planner.search(request))
                self.assertEqual(invoke.call_count, 1)

    def test_resume_rejects_stale_missing_or_overwritten_artifacts(self):
        from civ5sim.engine import digest
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            artifacts = {name: {"value": name} for name in ("result", "replay", "final")}
            for name, value in artifacts.items():
                write_json(directory / f"{name}.json", value)
            write_json(directory / "metrics.json", {})
            write_json(directory / "checked.json", {"run_signature": "old", "case": "one",
                       "replay_verified": True, "deterministic_sha256": digest(artifacts)})
            self.assertTrue(reusable_check(directory, "old", "one"))
            self.assertFalse(reusable_check(directory, "new", "one"))
            self.assertFalse(reusable_check(directory, "old", "two"))
            write_json(directory / "final.json", {"overwritten": True})
            self.assertFalse(reusable_check(directory, "old", "one"))
            (directory / "metrics.json").unlink()
            self.assertFalse(reusable_check(directory, "old", "one"))

    def test_timeout_stops_descendants_including_venv_redirectors(self):
        from tools.benchmark_combat_ai import run_bounded
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "escaped.txt"
            child = f"import time; from pathlib import Path; time.sleep(1); Path({str(marker)!r}).touch()"
            parent = ("import subprocess,sys,time; "
                      f"subprocess.Popen([sys.executable, '-c', {child!r}]); time.sleep(20)")
            with self.assertRaises(subprocess.TimeoutExpired):
                run_bounded([sys.executable, "-c", parent], .4)
            time.sleep(1.2)
            self.assertFalse(marker.exists(), "A worker descendant survived its timeout")



if __name__ == "__main__":
    unittest.main()
