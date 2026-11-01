"""Match lifecycle, deployment and the player information boundary."""

import json
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

from test_maps import field

from civ5sim import (
    Action,
    Battle,
    EndTurnAgent,
    MatchConfig,
    RandomAgent,
    create_scenario,
    run_match,
)
from civ5sim import _native as N
from civ5sim.engine import digest
from civ5sim.match import load_match_config

ROOT = Path(__file__).resolve().parents[1]
KERNEL = N.ROOT / "build/combat.dll"
if not KERNEL.exists():
    KERNEL = N.ROOT / "build/libcombat.so"


def config(**kwargs):
    m = field(8, 6)
    m["spawn_candidates"] = [
        [[0, y] for y in range(6)],
        [[7, y] for y in range(6)],
    ]
    return MatchConfig(
        m,
        [["warrior", "archer"], ["warrior", "archer"]],
        max_rounds=kwargs.pop("max_rounds", 3),
        **kwargs,
    )


class Recorder(EndTurnAgent):
    def reset(self, context):
        self.context = context
        self.observations = []

    def act(self, observation, actions):
        self.observations.append(deepcopy(observation))
        return super().act(observation, actions)


class DeploymentTests(unittest.TestCase):
    def test_repeatable_deployment_and_no_input_mutation(self):
        c = config()
        original = deepcopy(c)
        a, b = create_scenario(c), create_scenario(c)
        self.assertEqual(a, b)
        self.assertEqual(c, original)
        self.assertNotEqual(a["units"], create_scenario(replace(c, seed=1))["units"])
        self.assertEqual(len({u["tile"] for u in a["units"]}), 4)
        for u in a["units"]:
            xy = [u["tile"] % 8, u["tile"] // 8]
            self.assertIn(xy, c.map["spawn_candidates"][u["side"]])

    def test_bad_settings(self):
        c = config()
        for changes in (
            {"armies": [[], ["warrior"]]},
            {"armies": [["unknown"], ["warrior"]]},
            {"armies": [["warrior"]]},
            {"armies": [["warrior"] * 257] * 2},
            {"max_rounds": 0},
            {"max_rounds": True},
            {"seed": -1},
            {"fog": 1},
            {"random_damage": 0},
            {"max_calls_per_turn": 0},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                create_scenario(replace(c, **changes))

    def test_bad_spawn_zones(self):
        c = config()
        cases = []
        m = deepcopy(c.map)
        del m["spawn_candidates"]
        cases.append(m)
        for zone in ([], [[0, 0]], [[0, 0], [0, 0]], [[8, 0], [0, 1]], [[True, 0]]):
            m = deepcopy(c.map)
            m["spawn_candidates"][0] = zone
            cases.append(m)
        m = deepcopy(c.map)
        m["spawn_candidates"][1][0] = [0, 0]
        cases.append(m)
        m = deepcopy(c.map)
        m["tiles"][0]["elevation"] = "mountain"
        cases.append(m)
        m = deepcopy(c.map)
        for y in range(6):
            m["tiles"][y * 8 + 3]["elevation"] = "mountain"
        cases.append(m)
        for m in cases:
            with self.subTest(map=m), self.assertRaises(ValueError):
                create_scenario(replace(c, map=m))

    def test_load_config_relative_map_and_required_fields(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            (folder / "map.json").write_text(json.dumps(config().map), encoding="utf-8")
            value = {
                "map": "map.json",
                "armies": [["warrior"], ["archer"]],
                "max_rounds": 1,
                "agents": ["random", "end_turn"],
            }
            path = folder / "config.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            c, agents = load_match_config(path)
            self.assertEqual(c.map, folder / "map.json")
            self.assertEqual(agents, value["agents"])
            self.assertEqual(len(create_scenario(c)["units"]), 2)
            for key in ("armies", "max_rounds"):
                bad = value.copy()
                del bad[key]
                path.write_text(json.dumps(bad), encoding="utf-8")
                with self.assertRaises(TypeError):
                    load_match_config(path)


@unittest.skipUnless(KERNEL.exists(), "Build tools/build_combat.py first")
class MatchTests(unittest.TestCase):
    def verify_replay(self, match):
        with Battle(match.replay["initial"]) as b:
            for entry in match.replay["steps"]:
                self.assertEqual(asdict(b.step(entry["action"])), entry["result"])
                self.assertEqual(digest(b.snapshot()), entry["state_sha256"])
            self.assertEqual(b.snapshot(), match.final)
        self.assertEqual(digest(match.final), match.result["final_state_sha256"])

    def test_exact_round_limit_and_only_active_agent_called(self):
        for rounds in (1, 3):
            agents = [Recorder(), Recorder()]
            match = run_match(config(max_rounds=rounds), agents)
            self.assertEqual(match.result["outcome"], "draw")
            self.assertEqual(match.result["reason"], "turn_limit")
            self.assertIsNone(match.result["winner"])
            self.assertEqual(match.result["commands"], 2 * rounds)
            self.assertEqual(match.result["completed_turns"], 2 * rounds)
            self.assertEqual(match.result["agent_calls"], [rounds, rounds])
            for side, a in enumerate(agents):
                self.assertEqual(
                    [o["turn"] for o in a.observations],
                    list(range(side, 2 * rounds, 2)),
                )
                self.assertTrue(all(o["active_side"] == side for o in a.observations))
            self.verify_replay(match)

    def test_full_match_repeatable_in_all_modes(self):
        for fog in (True, False):
            for random_damage in (True, False):
                c = config(seed=42, fog=fog, random_damage=random_damage)
                a = run_match(c, ["random", "random"])
                b = run_match(c, ["random", "random"])
                self.assertEqual(a, b)
                self.verify_replay(a)
                self.assertEqual(len(set(a.result["seeds"].values())), 4)

    def test_player_inputs_and_agent_mutation_isolation(self):
        c = config()
        c.map["metadata"] = {"secret": "deployment seed"}
        c.map["private_extra"] = "hidden"
        c.map["tiles"][0]["metadata"] = {"hidden_units": [123], "rng": 456}
        c.map["boundary_rivers"] = [
            {"x": 0, "y": 5, "direction": 0, "flow": 2, "private_extra": "hidden"}
        ]

        class Mutator(Recorder):
            def reset(self, context):
                super().reset(context)
                context["map"]["tiles"][0]["terrain"] = "snow"
                context["armies"][1][0] = "horseman"

            def act(self, observation, actions):
                super().act(observation, actions)
                observation["units"].clear()
                return Action("end_turn")

        agents = [Mutator(), Recorder()]
        match = run_match(c, agents)
        other = agents[1]
        self.assertEqual(other.context["armies"], c.armies)
        self.assertEqual(other.context["map"]["tiles"][0]["terrain"], "grass")
        self.assertTrue(other.observations[0]["units"])
        for a in agents:
            self.assertNotIn("metadata", a.context["map"])
            self.assertNotIn("spawn_candidates", a.context["map"])
            self.assertNotIn("private_extra", a.context["map"])
            self.assertEqual(
                set(a.context["map"]["tiles"][0]),
                {"terrain", "elevation", "feature", "rivers"},
            )
            self.assertEqual(
                set(a.context["map"]["boundary_rivers"][0]),
                {"x", "y", "direction", "flow"},
            )
            self.assertNotIn(
                a.context["seed"],
                (match.result["seeds"]["placement"], match.result["seeds"]["battle"]),
            )
            self.assertEqual(
                {u["side"] for u in a.observations[0]["units"]}, {a.context["side"]}
            )
            self.assertNotIn("rng", a.observations[0])
        self.assertEqual(c.map["tiles"][0]["terrain"], "grass")
        self.verify_replay(match)

    def test_elimination_both_sides_stops_immediately(self):
        class Fighter(EndTurnAgent):
            def act(self, observation, actions):
                return next(
                    (a for a in actions if a.kind == "melee"), Action("end_turn")
                )

        # Adjacent fixed deployments, no moves; both outcomes use real kernel damage.
        m = field(4, 4)
        m["spawn_candidates"] = [[[1, 1]], [[2, 1]]]
        for winner in (0, 1):
            armies = [["warrior"], ["warrior"]]
            armies[winner] = ["horseman"]
            c = MatchConfig(m, armies, 100, fog=False, random_damage=False)
            match = run_match(c, [Fighter(), Fighter()])
            self.assertEqual(match.result["winner"], winner)
            self.assertEqual(match.result["reason"], "army_destroyed")
            self.assertEqual(match.result["outcome"], "win")
            self.assertEqual(sum(match.result["agent_calls"]), match.result["commands"])
            self.assertEqual(match.result["alive"][1 - winner], 0)
            self.verify_replay(match)

    def test_agent_failures_save_successful_prefix(self):
        class Broken(EndTurnAgent):
            def reset(self, context):
                self.calls = 0

            def act(self, observation, actions):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("test failure")
                return Action("end_turn")

        match = run_match(config(), [Broken(), "end_turn"])
        self.assertEqual(match.result["status"], "error")
        self.assertEqual(match.result["reason"], "agent_error")
        self.assertEqual(match.result["error"]["side"], 0)
        self.assertEqual(match.result["commands"], 2)
        self.assertIsNone(match.result["winner"])
        self.verify_replay(match)

    def test_invalid_response_and_factory_failure(self):
        for response in (None, {}, Action("move", 0, 9999), Action("end_turn", True)):

            class Invalid(EndTurnAgent):
                def act(self, observation, actions, response=response):
                    return response

            match = run_match(config(), [Invalid(), "end_turn"])
            self.assertEqual(match.result["reason"], "invalid_action")
            self.assertEqual(match.result["commands"], 0)
            self.verify_replay(match)
        match = run_match(config(), ["missing_module_for_match:Agent", "end_turn"])
        self.assertEqual(match.result["reason"], "agent_error")
        self.assertEqual(match.result["error"]["phase"], "reset")
        self.verify_replay(match)

    def test_action_limit_and_independent_instances(self):
        class Repeater(EndTurnAgent):
            def act(self, observation, actions):
                return next(a for a in actions if a.kind == "skip")

        match = run_match(config(max_calls_per_turn=2), [Repeater(), "end_turn"])
        self.assertEqual(match.result["reason"], "action_limit")
        self.assertEqual(match.result["commands"], 2)
        self.assertEqual(match.result["agent_calls"], [2, 0])
        self.verify_replay(match)
        a = RandomAgent()
        with self.assertRaises(ValueError):
            run_match(config(), [a, a])

    def test_module_factory_and_reset_exception(self):
        match = run_match(
            config(), ["civ5sim.agents:EndTurnAgent", "civ5sim.agents:EndTurnAgent"]
        )
        self.assertEqual(match.result["reason"], "turn_limit")
        self.verify_replay(match)

        class BrokenReset(EndTurnAgent):
            def reset(self, context):
                raise RuntimeError("reset failed")

        match = run_match(config(), ["end_turn", BrokenReset()])
        self.assertEqual(match.result["reason"], "agent_error")
        self.assertEqual(match.result["error"]["side"], 1)
        self.assertEqual(match.result["error"]["phase"], "reset")
        self.assertEqual(match.result["commands"], 0)
        self.assertEqual(match.result["agent_calls"], [0, 0])
        self.verify_replay(match)

    def test_cli_agent_error_still_saves_replay_and_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            value = {
                "map": config().map,
                "armies": config().armies,
                "max_rounds": 1,
                "agents": ["end_turn", "missing_module_for_match:Agent"],
            }
            path = folder / "config.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            run = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "civ5sim",
                    "match",
                    str(path),
                    "--out",
                    str(folder),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(run.returncode, 1, run.stdout + run.stderr)
            result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["reason"], "agent_error")
            self.assertIsNone(result["winner"])
            final = json.loads((folder / "final.json").read_text(encoding="utf-8"))
            replay = json.loads((folder / "replay.json").read_text(encoding="utf-8"))
            self.assertEqual(replay["steps"], [])
            self.assertEqual(replay["initial"], final)
            self.assertEqual(digest(final), result["final_state_sha256"])

    def test_cli_and_standard_replay_verifier(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            command = [sys.executable, "-m", "civ5sim"]
            run = subprocess.run(
                [
                    *command,
                    "match",
                    "scenarios/match-example.json",
                    "--out",
                    str(folder),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "completed")
            replay = subprocess.run(
                [
                    *command,
                    "replay",
                    str(folder / "replay.json"),
                    "--out",
                    str(folder / "verified.json"),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(replay.returncode, 0, replay.stdout + replay.stderr)
            final = json.loads((folder / "final.json").read_text(encoding="utf-8"))
            verified = json.loads(
                (folder / "verified.json").read_text(encoding="utf-8")
            )
            self.assertEqual(final, verified)
            self.assertEqual(digest(final), result["final_state_sha256"])


if __name__ == "__main__":
    unittest.main()
