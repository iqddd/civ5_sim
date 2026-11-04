"""One learner command, then the frozen opponent up to the next learner call."""

from dataclasses import asdict
from time import perf_counter

from civ5rl.core.context import context_for
from civ5rl.core.contracts import ScenarioSpec, WorkerState
from civ5rl.core.observation import ordered_actions
from civ5sim.agents import load_agent
from civ5sim.engine import Battle


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
    def __init__(self, spec: ScenarioSpec):
        self.spec = spec
        self.learner_decisions = 0
        self.learner_decision_limit = spec.get("learner_decision_limit")
        if self.learner_decision_limit is not None:
            origin = spec.get("provenance", {})
            if (
                type(self.learner_decision_limit) is not int
                or self.learner_decision_limit != 1
                or spec.get("diagnostic_protocol") != "D0-one-decision"
                or origin.get("source") != "rl01.diagnostic.D0-one-decision"
                or origin.get("split") not in ("diagnostic_train", "diagnostic_dev")
            ):
                raise ValueError(
                    "Learner decision limit requires trusted D0-one-decision provenance"
                )
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
        self.damage = [0, 0]
        self._pending_attack = None

    def prepare(self, action):
        self._pending_attack = None
        if self.spec.get("diagnostics") and action.kind in ("melee", "ranged"):
            units = self.battle.snapshot()["units"]
            attacker = units[action.unit]
            defender = next(
                u for u in units if u["tile"] == action.target and u["hp"] > 0
            )
            self._pending_attack = (
                attacker["side"],
                attacker["hp"],
                defender["side"],
                defender["hp"],
            )

    def check(self):
        self.terminal = terminal_result(
            self.battle, self.turns, self.spec["max_rounds"]
        )
        if (
            self.terminal is None
            and self.learner_decision_limit is not None
            and self.learner_decisions >= self.learner_decision_limit
        ):
            self.terminal = {"winner": None, "reason": "diagnostic_decision_limit"}
        return self.terminal

    def applied(self, action, result):
        self.history.append(asdict(action))
        if not result.ok:
            raise RuntimeError(f"Kernel rejected public legal action: {result.error}")
        if self.turns % 2 == self.side:
            self.learner_decisions += 1
        if self._pending_attack is not None:
            side, hp, other, other_hp = self._pending_attack
            self.damage[side] += min(result.damage, other_hp)
            self.damage[other] += min(result.retaliation, hp)
            self._pending_attack = None
        self.commands += 1
        self.calls += 1
        if action.kind == "end_turn":
            self.turns += 1
            self.calls = 0

    def request(self) -> WorkerState:
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
            self.prepare(action)
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
                    "learner_decisions": self.learner_decisions,
                    "rounds": self.turns / 2,
                    "timings": self.timings.copy(),
                    "first_contact_turn": self.first_contact_turn,
                    "remaining_hp": [
                        sum(u["hp"] for u in final["units"] if u["side"] == s)
                        for s in (0, 1)
                    ],
                    "damage_dealt": self.damage.copy(),
                    "units_lost": [
                        sum(u["hp"] <= 0 for u in final["units"] if u["side"] == s)
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

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
