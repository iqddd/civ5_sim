"""One learner command, then the frozen opponent up to the next learner call."""

from dataclasses import asdict
from time import perf_counter

from civ5sim.agents import load_agent
from civ5sim.engine import Battle

from .corpus import context_for
from .observation import ordered_actions


def terminal_result(battle, completed_turns, max_rounds):
    destroyed = battle.army_status()["destroyed"]
    if any(destroyed):
        return {
            "winner": None if all(destroyed) else (1 if destroyed[0] else 0),
            "reason": "mutual_destruction" if all(destroyed) else "army_destroyed",
        }
    if completed_turns >= 2 * max_rounds:
        return {"winner": None, "reason": "turn_limit"}
    return None


class DecisionEnv:
    def __init__(self, spec):
        self.spec = spec
        self.battle = Battle(spec["scenario"], log=False)
        self.side = spec["learner_side"]
        self.turns = self.calls = self.commands = 0
        self.history = []
        self.timings = {"simulation_seconds": 0, "opponent_seconds": 0}
        self.opponent = load_agent(spec["opponent"])
        self.opponent.reset(
            context_for(
                spec["scenario"],
                spec["armies"],
                1 - self.side,
                spec["max_rounds"],
                spec["agent_seeds"][1 - self.side],
            )
        )
        self.first_contact_turn = None

    def check(self):
        self.terminal = terminal_result(
            self.battle, self.turns, self.spec["max_rounds"]
        )
        return self.terminal

    def applied(self, action, result):
        self.history.append(asdict(action))
        if not result.ok:
            raise RuntimeError(f"Kernel rejected public legal action: {result.error}")
        self.commands += 1
        self.calls += 1
        if action.kind == "end_turn":
            self.turns += 1
            self.calls = 0

    def request(self):
        while not self.check() and self.turns % 2 != self.side:
            if self.calls >= self.spec["max_calls_per_turn"]:
                raise RuntimeError("Opponent action limit")
            obs = self.battle.observe(1 - self.side)
            actions = tuple(self.battle.available_actions())
            start = perf_counter()
            action = self.opponent.act(obs, actions)
            self.timings["opponent_seconds"] += perf_counter() - start
            if action not in actions:
                raise RuntimeError("Illegal opponent action")
            start = perf_counter()
            result = self.battle.step(action)
            self.timings["simulation_seconds"] += perf_counter() - start
            self.applied(action, result)
        if self.terminal:
            final = self.battle.snapshot()
            return {
                "done": True,
                "joint_commands": self.history,
                "result": {
                    **self.terminal,
                    "reward": 0
                    if self.terminal["winner"] is None
                    else (1 if self.terminal["winner"] == self.side else -1),
                    "commands": self.commands,
                    "rounds": self.turns / 2,
                    "timings": self.timings.copy(),
                    "first_contact_turn": self.first_contact_turn,
                    "remaining_hp": [
                        sum(u["hp"] for u in final["units"] if u["side"] == s)
                        for s in (0, 1)
                    ],
                },
            }
        if self.calls >= self.spec["max_calls_per_turn"]:
            raise RuntimeError("Learner action limit")
        observation = self.battle.observe(self.side)
        if self.first_contact_turn is None and any(
            u["side"] != self.side for u in observation["units"]
        ):
            self.first_contact_turn = self.turns
        return {
            "done": False,
            "observation": observation,
            "actions": ordered_actions(self.battle.available_actions()),
        }

    def reproduction(self):
        return {"spec": self.spec, "commands": self.history}

    def close(self):
        close = getattr(self.opponent, "close", None)
        if close is not None:
            close()
        self.battle.close()
