"""Behavioral fixtures for the CP Only subset and the player information boundary."""

import ctypes as C
import json
import random
import tempfile
import unittest
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from test_maps import field

from civ5map.maps import adjacent, crop
from civ5sim import Action, Battle, load_scenario, step_batch
from civ5sim import _native as N
from civ5sim.engine import digest

KERNEL = N.ROOT / "build/combat.dll"
if not KERNEL.exists():
    KERNEL = N.ROOT / "build/libcombat.so"


def scenario(*units, fog=False, random_damage=False, terrain=None):
    return {
        "schema": 1,
        "kind": "civ5sim.scenario",
        "map": terrain or field(12, 8),
        "units": list(units),
        "fog": fog,
        "random_damage": random_damage,
        "seed": 42,
    }


def unit(kind="warrior", side=0, x=2, y=2, **state):
    return {"type": kind, "side": side, "tile": [x, y], **state}


@unittest.skipUnless(KERNEL.exists(), "Build tools/build_combat.py first")
class CombatTests(unittest.TestCase):
    def battle(self, *args, **kwargs):
        b = Battle(scenario(*args, **kwargs))
        self.addCleanup(b.close)
        return b

    def test_equal_strength_melee_and_no_rng_for_preview(self):
        b = self.battle(unit(), unit(side=1, x=3), random_damage=True)
        before = b.snapshot()
        p = b.preview_attack(Action("melee", 0, 27))
        self.assertEqual(
            (p.damage, p.retaliation, p.attacker_strength, p.defender_strength),
            (30, 30, 800, 800),
        )
        self.assertEqual(before, b.snapshot())
        self.assertTrue(b.step(Action("melee", 0, 27)).ok)
        self.assertNotEqual(before["state"]["rng"], b.snapshot()["state"]["rng"])

    def test_damage_against_original_cp_function(self):
        lib = N.library()
        oracle = lib.cp_reference_damage
        oracle.argtypes, oracle.restype = [C.c_int] * 3, C.c_int
        rng = random.Random(49)
        cases = [(800, 800, r) for r in (0, 599, 600, 1199)]
        cases += [
            (a, d, r)
            for a, d in ((1, 1200), (1650, 1200), (640, 1360), (800, 1))
            for r in (0, 600, 1199)
        ]
        cases += [
            (rng.randint(1, 2500), rng.randint(1, 2500), rng.randrange(1200))
            for _ in range(5000)
        ]
        for a, d, roll in cases:
            self.assertEqual(
                lib.sim_damage(a, d, roll), oracle(a, d, roll), (a, d, roll)
            )

    def test_fixed_equal_melee_and_ranged_defense(self):
        b = self.battle(unit(), unit(side=1, x=3))
        r = b.step(Action("melee", 0, 27))
        self.assertEqual((r.damage, r.retaliation), (30, 30))
        self.assertEqual([u["hp"] for u in b.snapshot()["units"]], [70, 70])
        b = self.battle(unit("archer"), unit("archer", side=1, x=4))
        p = b.preview_attack(Action("ranged", 0, 28))
        self.assertEqual(
            (p.attacker_strength, p.defender_strength, p.damage, p.retaliation),
            (700, 700, 30, 0),
        )

    def test_cp_wounded_strength_penalty(self):
        b = self.battle(unit(hp=50), unit(side=1, x=3))
        p = b.preview_attack(Action("melee", 0, 27))
        self.assertEqual(p.attacker_strength, 664)  # 50*34//100 = 17%
        b = self.battle(unit("archer"), unit("archer", side=1, x=4, hp=50))
        self.assertEqual(
            b.preview_attack(Action("ranged", 0, 28)).defender_strength, 581
        )

    def test_anti_mounted_and_chariot_combat_class(self):
        b = self.battle(unit("spearman"), unit("horseman", side=1, x=3))
        self.assertEqual(
            b.preview_attack(Action("melee", 0, 27)).attacker_strength, 1650
        )
        b = self.battle(unit("spearman"), unit("chariot_archer", side=1, x=3))
        self.assertEqual(
            b.preview_attack(Action("melee", 0, 27)).attacker_strength, 1100
        )

    def test_overkill_reduces_retaliation_and_advances(self):
        b = self.battle(unit("horseman"), unit(side=1, x=3, hp=1))
        p = b.preview_attack(Action("melee", 0, 27))
        self.assertEqual((p.damage, p.retaliation), (1, 0))
        self.assertTrue(b.step(Action("melee", 0, 27)).ok)
        self.assertEqual(b.snapshot()["units"][0]["tile"], 27)
        self.assertGreater(b.snapshot()["units"][0]["moves"], 0)
        self.assertEqual(b.army_status()["destroyed"], [False, True])
        self.assertTrue(b.step(Action("end_turn")).ok)  # no automatic terminal state

    def test_mutual_death_has_exactly_one_survivor(self):
        b = self.battle(unit(hp=1), unit(side=1, x=3, hp=1))
        b.step(Action("melee", 0, 27))
        self.assertEqual(sum(b.army_status()["alive"]), 1)

    def test_shot_cannot_advance_and_archer_cannot_melee(self):
        b = self.battle(unit("archer"), unit(side=1, x=3, hp=1))
        before = b.snapshot()
        self.assertEqual(b.step(Action("melee", 0, 27)).error, "no_attack")
        self.assertEqual(before, b.snapshot())
        b.step(Action("ranged", 0, 27))
        self.assertEqual(b.snapshot()["units"][0]["tile"], 26)

    def test_one_attack_horse_can_move_afterward(self):
        b = self.battle(unit("horseman"), unit(side=1, x=3))
        b.step(Action("melee", 0, 27))
        self.assertEqual(b.step(Action("melee", 0, 27)).error, "no_attack")
        self.assertTrue(any(a.kind == "move" for a in b.available_actions()))

    def test_one_sixtieth_allows_moves_into_open_rough_and_river_tiles(self):
        for kind in ("warrior", "archer", "chariot_archer", "spearman", "horseman"):
            for terrain in ("open", "rough", "river"):
                with self.subTest(kind=kind, terrain=terrain):
                    m = field(12, 8)
                    if terrain == "rough":
                        m["tiles"][27].update(elevation="hills", feature="forest")
                    elif terrain == "river":
                        m["tiles"][26]["rivers"][0] = 0
                    b = self.battle(unit(kind, moves=1), terrain=m)
                    command = Action("move", 0, 27)
                    self.assertIn(command, b.available_actions())
                    self.assertTrue(b.step(command).ok)
                    after = b.snapshot()["units"][0]
                    self.assertEqual((after["tile"], after["moves"]), (27, 0))
                    self.assertFalse(any(a.unit == 0 for a in b.available_actions()))

    def test_one_sixtieth_allows_melee_and_ranged_attacks(self):
        for kind in ("warrior", "archer", "chariot_archer", "spearman", "horseman"):
            with self.subTest(kind=kind):
                m = field(12, 8)
                m["tiles"][26]["rivers"][0] = 0
                m["tiles"][27].update(elevation="hills", feature="forest")
                b = self.battle(unit(kind, moves=1), unit(side=1, x=3), terrain=m)
                attack = "ranged" if kind in ("archer", "chariot_archer") else "melee"
                command = Action(attack, 0, 27)
                self.assertIn(command, b.available_actions())
                self.assertTrue(b.preview_attack(command).ok)
                result = b.step(command)
                self.assertTrue(result.ok)
                self.assertGreater(result.damage, 0)
                self.assertEqual(b.snapshot()["units"][0]["moves"], 0)
                self.assertTrue(b.snapshot()["units"][0]["attacked"])

    def test_zero_movement_blocks_moves_and_both_attack_types(self):
        for kind, attack in (("warrior", "melee"), ("archer", "ranged")):
            b = self.battle(unit(kind, moves=0), unit(side=1, x=3), random_damage=True)
            before = b.snapshot()
            for command in (Action("move", 0, 25), Action(attack, 0, 27)):
                self.assertNotIn(command, b.available_actions())
                self.assertEqual(b.step(command).error, "exhausted")
                self.assertEqual(before, b.snapshot())

    def test_last_fraction_does_not_restore_a_used_attack(self):
        b = self.battle(unit("horseman", moves=1, attacked=True), unit(side=1, x=3))
        self.assertEqual(b.step(Action("melee", 0, 27)).error, "no_attack")
        self.assertTrue(b.step(Action("move", 0, 25)).ok)

    def test_river_both_directions_and_terrain(self):
        m = field(12, 8)
        m["tiles"][26]["rivers"][0] = 0
        m["tiles"][27].update(elevation="hills", feature="forest")
        b = self.battle(unit(), unit(side=1, x=3), terrain=m)
        p = b.preview_attack(Action("melee", 0, 27))
        self.assertEqual((p.attacker_strength, p.defender_strength), (640, 1200))
        s = scenario(unit(x=3), unit(side=1), terrain=m)
        with Battle(s) as reverse:
            self.assertEqual(
                reverse.preview_attack(Action("melee", 0, 26)).attacker_strength, 640
            )
        b = self.battle(unit("horseman"), terrain=m)
        self.assertTrue(b.step(Action("move", 0, 27)).ok)
        self.assertEqual(b.snapshot()["units"][0]["moves"], 0)

    def test_mounted_ignores_positive_terrain_but_not_negative(self):
        m = field(12, 8)
        m["tiles"][27].update(elevation="hills", feature="forest")
        b = self.battle(unit(), unit("horseman", side=1, x=3), terrain=m)
        self.assertEqual(
            b.preview_attack(Action("melee", 0, 27)).defender_strength, 1200
        )
        m["tiles"][27].update(elevation="flat", feature="marsh")
        b = self.battle(unit(), unit("horseman", side=1, x=3), terrain=m)
        self.assertEqual(
            b.preview_attack(Action("melee", 0, 27)).defender_strength, 1020
        )

    def test_rough_chariot_and_hill_forest_cost(self):
        m = field(12, 8)
        m["tiles"][27].update(elevation="hills", feature="forest")
        b = self.battle(unit("chariot_archer"), terrain=m)
        b.step(Action("move", 0, 27))
        self.assertEqual(b.snapshot()["units"][0]["moves"], 0)
        b = self.battle(unit("horseman"), terrain=m)
        b.step(Action("move", 0, 27))
        self.assertEqual(b.snapshot()["units"][0]["moves"], 60)

    def test_friendly_transit_and_occupied_destination(self):
        b = self.battle(unit("horseman"), unit(x=3))
        self.assertEqual(b.step(Action("move", 0, 27)).error, "blocked")
        self.assertTrue(b.step(Action("move", 0, 28)).ok)
        state = b.snapshot()["units"]
        self.assertEqual([u["tile"] for u in state], [28, 27])

    def test_swap_all_unit_pairs_and_modes(self):
        types = ("warrior", "archer", "chariot_archer", "spearman", "horseman")
        for first in types:
            for second in types:
                for fog, damage in (
                    (False, False),
                    (False, True),
                    (True, False),
                    (True, True),
                ):
                    with (
                        self.subTest(
                            first=first, second=second, fog=fog, damage=damage
                        ),
                        Battle(
                            scenario(
                                unit(first),
                                unit(second, x=3),
                                fog=fog,
                                random_damage=damage,
                            )
                        ) as b,
                    ):
                        before = b.snapshot()
                        sight = b.observe()["visibility"]
                        self.assertIn(Action("swap", 0, 27), b.available_actions())
                        self.assertIn(Action("swap", 1, 26), b.available_actions())
                        result = b.step(Action("swap", 0, 27))
                        self.assertTrue(result.ok)
                        self.assertEqual(
                            (result.moved, result.damage, result.retaliation),
                            (2, 0, 0),
                        )
                        after = b.snapshot()
                        self.assertEqual([u["tile"] for u in after["units"]], [27, 26])
                        self.assertEqual(
                            [u["moves"] for u in after["units"]],
                            [u["moves"] - 60 for u in before["units"]],
                        )
                        self.assertTrue(all(u["acted"] for u in after["units"]))
                        self.assertEqual(sight, b.observe()["visibility"])
                        self.assertEqual(before["state"], after["state"])

    def test_swap_charges_directional_terrain_river_and_zoc_costs(self):
        m = field(12, 8)
        m["tiles"][26]["feature"] = "forest"
        b = self.battle(unit("horseman"), unit("chariot_archer", x=3), terrain=m)
        b.step(Action("swap", 0, 27))
        self.assertEqual([u["moves"] for u in b.snapshot()["units"]], [180, 0])
        m["tiles"][26]["feature"] = None
        m["tiles"][27].update(elevation="hills", feature="forest")
        b = self.battle(unit("horseman"), unit(x=3), terrain=m)
        b.step(Action("swap", 0, 27))
        self.assertEqual([u["moves"] for u in b.snapshot()["units"]], [60, 60])
        m["tiles"][26]["rivers"][0] = 0
        b = self.battle(unit("horseman"), unit(x=3), terrain=m)
        b.step(Action("swap", 0, 27))
        self.assertEqual([u["moves"] for u in b.snapshot()["units"]], [0, 0])
        north = adjacent(m, 2, 2, 0)
        b = self.battle(
            unit("horseman"),
            unit("horseman", x=3),
            unit(side=1, x=north[0], y=north[1]),
        )
        self.assertTrue(b.step(Action("swap", 0, 27)).ok)
        self.assertEqual([u["moves"] for u in b.snapshot()["units"][:2]], [0, 0])

    def test_swap_with_partial_movement_removes_fortification_and_prevents_heal(self):
        b = self.battle(
            unit(hp=50, fortified=True, moves=1),
            unit(x=3, hp=60, fortified=True, moves=30),
        )
        self.assertTrue(b.step(Action("swap", 0, 27)).ok)
        self.assertEqual([u["moves"] for u in b.snapshot()["units"]], [0, 0])
        b.step(Action("end_turn"))
        b.step(Action("end_turn"))
        units = b.snapshot()["units"]
        self.assertEqual([u["hp"] for u in units], [50, 60])
        self.assertTrue(all(not u["fortified"] for u in units))

    def test_swap_wakes_partner_and_preserves_attacks(self):
        b = self.battle(
            unit("horseman", attacked=True, acted=True),
            unit("horseman", x=3, attacked=True, acted=True),
        )
        b.step(Action("skip", 1))
        self.assertTrue(b.step(Action("swap", 0, 27)).ok)
        units = b.snapshot()["units"]
        self.assertTrue(all(u["attacked"] and not u["waiting"] for u in units))
        self.assertTrue(
            any(a.kind == "move" and a.unit == 1 for a in b.available_actions())
        )
        b = self.battle(unit(), unit(x=3, fortified=True))
        b.step(Action("fortify", 1))
        self.assertTrue(b.step(Action("swap", 0, 27)).ok)
        self.assertFalse(b.snapshot()["units"][1]["fortified"])

    def test_invalid_swaps_are_atomic_and_absent_from_actions(self):
        cases = [
            ([unit(), unit(x=3, moves=0)], Action("swap", 0, 27)),
            ([unit(moves=0), unit(x=3)], Action("swap", 0, 27)),
            ([unit(waiting=True), unit(x=3)], Action("swap", 0, 27)),
            ([unit(), unit(x=3, side=1)], Action("swap", 0, 27)),
            ([unit(), unit(x=3, side=1)], Action("swap", 1, 26)),
            ([unit(), unit(x=3, hp=0)], Action("swap", 0, 27)),
            ([unit(), unit(x=4)], Action("swap", 0, 28)),
            ([unit()], Action("swap", 0, 27)),
            ([unit()], Action("swap", 0, 26)),
            ([unit()], Action("swap", 0, -1)),
            ([unit()], Action("swap", 0, 99999)),
            ([unit()], Action("swap", 100, 27)),
        ]
        for army, command in cases:
            with self.subTest(command=command, army=army):
                b = self.battle(*army, fog=True, random_damage=True)
                before = b.snapshot()
                self.assertNotIn(command, b.available_actions())
                self.assertFalse(b.step(command).ok)
                self.assertEqual(before, b.snapshot())

    def test_swap_respects_parity_wrap_and_closed_boundaries(self):
        for parity in (0, 1):
            m = field(12, 8)
            m["row_parity"] = parity
            x, y = adjacent(m, 2, 2, 0)
            b = self.battle(unit(), unit(x=x, y=y), terrain=m)
            self.assertTrue(b.step(Action("swap", 0, y * 12 + x)).ok)
        m = field(12, 8)
        for wrap in (False, True):
            m["wrap_x"] = wrap
            b = self.battle(unit(x=0), unit(x=11), terrain=m)
            self.assertEqual(b.step(Action("swap", 0, 35)).ok, wrap)

    def test_swap_batch_snapshot_and_replay(self):
        with Battle(
            scenario(unit(), unit(x=3), random_damage=True, fog=True), log=True
        ) as a:
            with a.clone(log=True) as b:
                command = Action("swap", 0, 27)
                self.assertEqual([a.step(command)], step_batch([b], [command]))
                self.assertEqual(a.snapshot(), b.snapshot())
                self.assertEqual(a.replay(), b.replay())
            with Battle(a.snapshot()) as restored:
                self.assertEqual(
                    a.step(Action("swap", 1, 27)), restored.step(Action("swap", 1, 27))
                )
                self.assertEqual(a.snapshot(), restored.snapshot())
            replay = a.replay()
            with Battle(replay["initial"]) as check:
                for entry in replay["steps"]:
                    self.assertEqual(
                        asdict(check.step(entry["action"])), entry["result"]
                    )
                    self.assertEqual(digest(check.snapshot()), entry["state_sha256"])

    def test_zone_of_control_and_escape(self):
        # East along the two cells controlled by the enemy to the north.
        m = field(12, 8)
        north = adjacent(m, 2, 2, 0)
        b = self.battle(unit("horseman"), unit(side=1, x=north[0], y=north[1]))
        b.step(Action("move", 0, 27))
        self.assertEqual(b.snapshot()["units"][0]["moves"], 0)
        b = self.battle(unit("horseman"), unit(side=1, x=north[0], y=north[1]))
        b.step(Action("move", 0, 25))
        self.assertEqual(b.snapshot()["units"][0]["moves"], 180)

    def test_turn_reset_modes_apply_to_the_correct_side_and_survive_restore(self):
        # Wrong boundary silently changes attacks/healing; normal matches do not flag it.
        for mode in ("legacy", "end_turn"):
            with self.subTest(mode=mode):
                sc = scenario(unit(hp=50), unit(side=1, x=8, hp=60))
                sc["turn_reset_mode"] = mode
                with Battle(sc, log=True) as b:
                    b.step(Action("skip", 0))
                    b.step(Action("end_turn"))
                    own, enemy = b.snapshot()["units"]
                    self.assertEqual((own["hp"], own["fortified"]),
                                     (60, True) if mode == "end_turn" else (50, False))
                    self.assertEqual((enemy["hp"], enemy["fortified"]),
                                     (60, False) if mode == "end_turn" else (70, True))
                    snapshot = b.snapshot()
                    with b.clone() as clone, Battle(snapshot) as restored:
                        for other in (clone, restored):
                            self.assertEqual(other.turn_reset_mode, mode)
                            other.step(Action("end_turn"))
                        b.step(Action("end_turn"))
                        self.assertEqual(b.snapshot(), clone.snapshot())
                        self.assertEqual(b.snapshot(), restored.snapshot())
                    with Battle(b.replay()["initial"]) as replayed:
                        for step in b.replay()["steps"]:
                            replayed.step(step["action"])
                            self.assertEqual(digest(replayed.snapshot()), step["state_sha256"])
                    b.reset(scenario(unit()))
                    self.assertEqual(b.turn_reset_mode, "legacy")
                    self.assertNotIn("turn_reset_mode", b.snapshot()["state"])

    def test_end_turn_fortification_protects_against_the_immediate_response(self):
        for mode in ("legacy", "end_turn"):
            for action in ("skip", "fortify", "none", "move", "melee"):
                with self.subTest(mode=mode, action=action):
                    sc = scenario(unit(), unit(side=1, x=3))
                    sc["turn_reset_mode"] = mode
                    with Battle(sc) as b:
                        if action in ("skip", "fortify"):
                            b.step(Action(action, 0))
                        elif action == "move":
                            b.step(Action("move", 0, 25))
                        elif action == "melee":
                            b.step(Action("melee", 0, 27))
                        b.step(Action("end_turn"))
                        expected = mode == "end_turn" and action in ("skip", "fortify", "none")
                        self.assertEqual(bool(b.snapshot()["units"][0]["fortified"]), expected)
                        if action in ("skip", "fortify", "none"):
                            result = b.step(Action("melee", 1, 26))
                            self.assertTrue(result.ok)
                            self.assertEqual(result.defender_strength, 960 if expected else 800)

    def test_fortification_and_healing_at_owner_turn_reset(self):
        b = self.battle(unit(hp=50), unit(side=1, x=8))
        b.step(Action("fortify", 0))
        self.assertFalse(b.snapshot()["units"][0]["fortified"])
        b.step(Action("end_turn"))
        self.assertEqual(b.snapshot()["units"][0]["hp"], 50)
        b.step(Action("end_turn"))
        u = b.snapshot()["units"][0]
        self.assertEqual((u["hp"], u["fortified"]), (60, True))
        b.step(Action("move", 0, 27))
        b.step(Action("end_turn"))
        b.step(Action("end_turn"))
        u = b.snapshot()["units"][0]
        self.assertEqual((u["hp"], u["fortified"]), (60, False))

    def test_skip_does_not_prevent_healing_or_fortification(self):
        b = self.battle(unit(hp=50))
        b.step(Action("skip", 0))
        self.assertEqual(b.step(Action("move", 0, 27)).error, "exhausted")
        b.step(Action("end_turn"))
        b.step(Action("end_turn"))
        self.assertEqual(b.snapshot()["units"][0]["hp"], 60)
        self.assertTrue(b.snapshot()["units"][0]["fortified"])

    def test_mounted_cannot_fortify_but_can_heal(self):
        b = self.battle(unit("horseman", hp=50))
        self.assertEqual(b.step(Action("fortify", 0)).error, "no_fortify")
        b.step(Action("skip", 0))
        b.step(Action("end_turn"))
        b.step(Action("end_turn"))
        self.assertEqual(b.snapshot()["units"][0]["hp"], 60)
        self.assertFalse(b.snapshot()["units"][0]["fortified"])

    def test_fog_los_and_ranged_obstruction(self):
        m = field(12, 8)
        m["tiles"][27]["feature"] = "forest"
        b = self.battle(unit("archer"), unit(side=1, x=4), terrain=m, fog=True)
        self.assertEqual(len(b.observe()["units"]), 1)
        self.assertEqual(b.preview_attack(Action("ranged", 0, 28)).error, "bad_target")
        m["tiles"][26]["elevation"] = "hills"
        b = self.battle(unit("archer"), unit(side=1, x=4), terrain=m, fog=True)
        self.assertEqual(len(b.observe()["units"]), 2)
        self.assertTrue(b.preview_attack(Action("ranged", 0, 28)).ok)

    def test_long_move_stops_on_new_contact_without_losing_intermediate_sight(self):
        m = field(12, 8)
        for i, t in enumerate(m["tiles"]):
            if i // 12 != 2:
                t.update(terrain="coast", elevation="water")
        b = self.battle(unit("horseman", x=1), unit(side=1, x=6), terrain=m, fog=True)
        self.assertEqual(len(b.observe()["units"]), 1)
        self.assertIn(Action("move", 0, 29), b.available_actions())
        result = b.step(Action("move", 0, 29))
        self.assertTrue(result.ok)
        self.assertEqual(result.moved, 3)
        self.assertEqual(b.snapshot()["units"][0]["tile"], 28)
        self.assertEqual(len(b.observe()["units"]), 2)
        self.assertEqual(b.snapshot()["units"][0]["moves"], 60)

    def test_fortification_is_flat_cp_bonus_and_resets_on_attack(self):
        b = self.battle(unit(), unit(side=1, x=3, fortified=True))
        self.assertEqual(
            b.preview_attack(Action("melee", 0, 27)).defender_strength, 960
        )
        b.step(Action("end_turn"))
        b.step(Action("melee", 1, 26))
        self.assertFalse(b.snapshot()["units"][1]["fortified"])

    def test_hidden_death_and_private_enemy_movement_are_not_in_observation(self):
        a = self.battle(unit(), unit(side=1, x=8), fog=True)
        b = self.battle(unit(), unit(side=1, x=8, hp=0, moves=0), fog=True)
        self.assertEqual(a.observe(), b.observe())
        self.assertEqual(a.available_actions(), b.available_actions())
        a = self.battle(unit(), unit(side=1, x=3), fog=True)
        b = self.battle(
            unit(), unit(side=1, x=3, moves=0, attacked=True, acted=True), fog=True
        )
        self.assertEqual(a.observe(), b.observe())
        for battle in (a, b):
            packed = battle.observation_buffer()
            enemy = packed["units"][1]
            self.assertEqual(
                (enemy.moves, enemy.attacked, enemy.acted, enemy.waiting),
                (-1, -1, -1, -1),
            )

    def test_all_owned_river_edges_both_parities_and_directions(self):
        for parity in (0, 1):
            for d, flow in ((1, 0), (2, 1), (3, 2)):
                m = field(12, 8)
                m["row_parity"] = parity
                m["tiles"][26]["rivers"][d - 1] = flow
                x, y = adjacent(m, 2, 2, d)
                for source, target in (((2, 2), (x, y)), ((x, y), (2, 2))):
                    with self.subTest(parity=parity, direction=d, source=source):
                        b = self.battle(
                            unit("horseman", x=source[0], y=source[1]), terrain=m
                        )
                        self.assertTrue(
                            b.step(Action("move", 0, target[1] * 12 + target[0])).ok
                        )
                        self.assertEqual(b.snapshot()["units"][0]["moves"], 0)

    def test_existing_pangaea_crops_load_and_execute_in_all_modes(self):
        rng = random.Random(101)
        types = ("warrior", "archer", "chariot_archer", "spearman", "horseman")
        for path in sorted((N.ROOT / "artifacts/pangaea-42").glob("crop-*.json")):
            m = json.loads(path.read_text(encoding="utf-8"))
            armies = [
                {"type": types[i % 5], "side": side, "tile": xy}
                for side in (0, 1)
                for i, xy in enumerate(m["spawn_candidates"][side][:10])
            ]
            for fog in (False, True):
                for damage in (False, True):
                    with (
                        self.subTest(map=path.name, fog=fog, random_damage=damage),
                        Battle(
                            scenario(*armies, terrain=m, fog=fog, random_damage=damage)
                        ) as b,
                    ):
                        for _ in range(30):
                            self.assertTrue(
                                b.step(rng.choice(b.available_actions())).ok
                            )

    def test_observation_and_actions_do_not_disclose_hidden_positions(self):
        a = self.battle(unit("horseman"), unit(side=1, x=8), fog=True)
        b = self.battle(unit("horseman"), unit(side=1, x=9, hp=1), fog=True)
        self.assertEqual(a.observe(), b.observe())
        self.assertEqual(a.available_actions(), b.available_actions())
        for command in (Action("ranged", 0, 32), Action("melee", 0, 33)):
            self.assertEqual(a.preview_attack(command), b.preview_attack(command))

    def test_preview_does_not_disclose_hidden_flanking(self):
        m = field(12, 8)
        # Forest hides a second enemy from the attacker.
        m["tiles"][38]["feature"] = "forest"
        a = self.battle(
            unit(), unit(side=1, x=3), unit(side=1, x=8), terrain=m, fog=True
        )
        b = self.battle(
            unit(), unit(side=1, x=3), unit(side=1, x=9), terrain=m, fog=True
        )
        self.assertEqual(a.observe(), b.observe())
        self.assertEqual(
            a.preview_attack(Action("melee", 0, 27)),
            b.preview_attack(Action("melee", 0, 27)),
        )

    def test_cp_flanks_are_relative_on_attack_and_defense(self):
        b = self.battle(unit(), unit(side=1, x=3), unit(x=4))
        self.assertEqual(
            b.preview_attack(Action("melee", 0, 27)).attacker_strength, 880
        )
        b = self.battle(unit(), unit(side=1, x=3), unit(side=1, x=1))
        self.assertEqual(
            b.preview_attack(Action("melee", 0, 27)).defender_strength, 880
        )

    def test_reproducibility_snapshot_clone_and_all_mode_combinations(self):
        for fog in (False, True):
            for rand in (False, True):
                with self.subTest(fog=fog, rand=rand):
                    a = self.battle(
                        unit(), unit(side=1, x=3), fog=fog, random_damage=rand
                    )
                    b = a.clone()
                    self.addCleanup(b.close)
                    self.assertEqual(
                        a.step(Action("melee", 0, 27)), b.step(Action("melee", 0, 27))
                    )
                    restored = Battle(json.loads(json.dumps(a.snapshot())))
                    self.addCleanup(restored.close)
                    for command in (Action("end_turn"), Action("melee", 1, 26)):
                        self.assertEqual(a.step(command), restored.step(command))
                    self.assertEqual(a.snapshot(), restored.snapshot())

    def test_batch_matches_single_execution_and_rejects_aliasing(self):
        singles = [
            self.battle(unit(), unit(side=1, x=3), random_damage=True) for _ in range(4)
        ]
        batches = [s.clone() for s in singles]
        for b in batches:
            self.addCleanup(b.close)
        for commands in (
            [Action("melee", 0, 27)] * 4,
            [Action("end_turn")] * 4,
            [Action("melee", 1, 26)] * 4,
        ):
            expected = [b.step(a) for b, a in zip(singles, commands)]
            self.assertEqual(step_batch(batches, commands), expected)
            self.assertEqual(
                [b.snapshot() for b in singles], [b.snapshot() for b in batches]
            )
        with self.assertRaises(ValueError):
            step_batch([singles[0]] * 2, [Action("end_turn")] * 2)

    def test_invalid_commands_are_atomic(self):
        b = self.battle(unit(), unit(side=1, x=3), random_damage=True)
        before = b.snapshot()
        for command in (
            Action("move", 0, 10000),
            Action("melee", 999, 27),
            Action("melee", 1, 26),
            Action("ranged", 0, 27),
            Action("move", 0, -1),
        ):
            self.assertFalse(b.step(command).ok)
            self.assertEqual(before, b.snapshot())
        with self.assertRaises(ValueError):
            b.step(Action("move", 0, 2**40))
        self.assertEqual(before, b.snapshot())

    def test_map_parity_wrap_and_no_escape(self):
        for parity in (0, 1):
            m = field(12, 8)
            m["row_parity"] = parity
            b = self.battle(unit(), terrain=m)
            for d in range(6):
                tile = adjacent(m, 2, 2, d)
                self.assertIn(
                    Action("move", 0, tile[1] * 12 + tile[0]), b.available_actions()
                )
        m = field(12, 8)
        m["wrap_x"] = True
        b = self.battle(unit(x=0), terrain=m)
        self.assertTrue(b.step(Action("move", 0, 35)).ok)
        m["wrap_x"] = False
        b = self.battle(unit(x=0), terrain=m)
        self.assertEqual(b.step(Action("move", 0, 35)).error, "blocked")

    def test_crop_has_same_sight_and_attacks_at_both_parities(self):
        parent = field(24, 24)
        for y in (2, 3):
            m = crop(parent, 2, y, 16, 16)
            b = self.battle(
                unit("archer", x=4, y=4), unit(side=1, x=6, y=4), terrain=m, fog=True
            )
            self.assertTrue(b.preview_attack(Action("ranged", 0, 70)).ok)

    def test_impassable_and_map_input_is_not_mutated(self):
        m = field(12, 8)
        m["tiles"][27]["elevation"] = "mountain"
        before = deepcopy(m)
        b = self.battle(unit(), terrain=m)
        self.assertEqual(b.step(Action("move", 0, 27)).error, "bad_target")
        self.assertEqual(m, before)
        with self.assertRaises(ValueError):
            Battle(scenario(unit(x=3), terrain=m))

    def test_replay_reset_and_rules_compatibility(self):
        with Battle(
            scenario(unit(), unit(side=1, x=3), random_damage=True), log=True
        ) as b:
            initial = b.snapshot()
            for command in (
                Action("melee", 0, 27),
                Action("end_turn"),
                Action("melee", 1, 26),
            ):
                b.step(command)
            replay = b.replay()
            with Battle(replay["initial"]) as check:
                for entry in replay["steps"]:
                    self.assertEqual(
                        asdict(check.step(entry["action"])), entry["result"]
                    )
                    self.assertEqual(digest(check.snapshot()), entry["state_sha256"])
            b.reset()
            self.assertEqual(initial, b.snapshot())
            self.assertEqual(b.replay()["steps"], [])
            wrong = deepcopy(initial)
            wrong["rules"]["fingerprint"] = "wrong"
            with self.assertRaises(ValueError):
                b.reset(wrong)
            self.assertEqual(initial, b.snapshot())

    def test_loading_relative_map_and_close_lifetime(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            s = scenario(unit())
            (path / "map.json").write_text(json.dumps(s["map"]), encoding="utf-8")
            s["map"] = "map.json"
            (path / "battle.json").write_text(json.dumps(s), encoding="utf-8")
            b = Battle(load_scenario(path / "battle.json"))
            b.close()
            b.close()
            with self.assertRaises(RuntimeError):
                b.step(Action("end_turn"))

    def test_random_legal_play_keeps_occupancy_and_hp_consistent(self):
        rng = random.Random(5)
        b = self.battle(
            *[
                unit(t, side=s, x=2 + s * 6, y=1 + i)
                for s in (0, 1)
                for i, t in enumerate(
                    ("warrior", "archer", "chariot_archer", "spearman", "horseman")
                )
            ],
            random_damage=True,
            fog=True,
        )
        for _ in range(300):
            command = rng.choice(b.available_actions())
            self.assertTrue(b.step(command).ok, command)
            units = b.snapshot()["units"]
            living = [u for u in units if u["hp"] > 0]
            self.assertEqual(len(living), len({u["tile"] for u in living}))
            self.assertTrue(all(0 <= u["hp"] <= 100 and u["moves"] >= 0 for u in units))


if __name__ == "__main__":
    unittest.main()
