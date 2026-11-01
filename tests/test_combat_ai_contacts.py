"""Real observation history: time is counted in observer turns, not calls."""

import unittest

from civ5sim.combat_ai import Contact, ContactMemory, own_turn


def observation(turn, side=0, active=None, visible=True):
    return {
        "turn": turn,
        "side": side,
        "active_side": (side if active is None else active),
        "units": [{"id": 1, "side": 1 - side, "hp": 100}] if visible else [],
    }


class ContactTests(unittest.TestCase):
    def test_five_six_seven_and_repeated_calls(self):
        for side in (0, 1):
            # Own ordinal 5, then 6 and 7, for either starting side.
            first = observation(10, side)
            memory = ContactMemory().updated(first)
            self.assertEqual(memory.known_enemies, (Contact(1, 5),))
            self.assertEqual(memory.updated(first), memory)
            hidden = observation(12, side, visible=False)
            grace = memory.updated(hidden)
            self.assertEqual(grace.known_enemies, memory.known_enemies)
            self.assertEqual(grace.updated(hidden), grace)
            expired = grace.updated(observation(14, side, visible=False))
            self.assertEqual(expired.known_enemies, ())
            self.assertEqual(memory.updated(observation(14, side, visible=False)), expired)

    def test_enemy_half_turn_does_not_age_memory(self):
        for side in (0, 1):
            memory = ContactMemory().updated(observation(10, side))
            for turn, active in ((11, 1 - side), (12, side), (13, 1 - side)):
                memory = memory.updated(observation(turn, side, active, visible=False))
                self.assertEqual(len(memory.known_enemies), 1)
            self.assertFalse(memory.updated(observation(14, side, visible=False)).known_enemies)

    def test_rediscovery_refreshes_expired_contact(self):
        memory = ContactMemory().updated(observation(10))
        memory = memory.updated(observation(12))
        self.assertEqual(memory.known_enemies, (Contact(1, 6),))
        self.assertTrue(memory.updated(observation(14, visible=False)).known_enemies)
        self.assertFalse(memory.updated(observation(16, visible=False)).known_enemies)
        self.assertEqual(memory.updated(observation(18)).known_enemies, (Contact(1, 9),))

    def test_validation_and_no_cross_side_or_match_reuse(self):
        obs = observation(10)
        for memory in (
            ContactMemory(-1), ContactMemory(11, 0), ContactMemory(10, 1),
            ContactMemory(0, -1, (Contact(1, 0),)),
            ContactMemory(10, 0, (Contact(1, 6),)),
            ContactMemory(10, 0, (Contact(1, -1),)),
            ContactMemory(10, 0, (Contact(-1, 0),)),
            ContactMemory(10, 0, (Contact(1, 5), Contact(1, 5))),
            ContactMemory(10, 0, (Contact(True, 5),)),
        ):
            with self.subTest(memory=memory), self.assertRaises(ValueError):
                memory.updated(obs)
        with self.assertRaises(ValueError):
            ContactMemory().updated(obs).updated(observation(0))

    def test_no_input_mutation_and_stable_order(self):
        obs = observation(10)
        obs["units"].extend([{"id": 2, "side": 1, "hp": 100},
                             {"id": 0, "side": 0, "hp": 100}])
        first = ContactMemory().updated(obs)
        obs["units"].clear()
        self.assertEqual(first.known_enemies, (Contact(1, 5), Contact(2, 5)))
        self.assertEqual(first.updated(obs), first)

    def test_own_turn_bounds_and_arbitrary_initial_parity(self):
        self.assertEqual(own_turn(2**31 - 1, 0, 0), 2**30)
        for side in (0, 1):
            for active in (0, 1):
                for turn in (0, 1, 8, 9):
                    now = own_turn(turn, active, side)
                    next_turn = own_turn(turn + 1, 1 - active, side)
                    self.assertEqual(next_turn - now, int(active != side))
        for args in ((-1, 0, 0), (True, 0, 0), (0, 2, 0), (0, 0, -1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                own_turn(*args)


if __name__ == "__main__":
    unittest.main()
