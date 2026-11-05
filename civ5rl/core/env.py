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
        if self.learner_decision_limit is not None and (
            type(self.learner_decision_limit) is not int
            or self.learner_decision_limit < 1
        ):
            raise ValueError("Learner decision limit must be a positive integer")
        self.battle = Battle(spec["scenario"], log=False)
        self.side = spec["learner_side"]
        self.turns = self.calls = self.commands = 0
        self.history = []
        self.timings = {"simulation_seconds": 0, "opponent_seconds": 0}
        self.reward_profile = spec.get("reward_profile", "symmetric")
        if self.reward_profile not in ("symmetric", "win_probability"):
            self.battle.close()
            raise ValueError("Unknown terminal reward profile")
        if spec["opponent"] == "cp" and "opponent_search" in spec:
            from civ5sim.combat_ai import Aggression, SearchConfig
            from civ5sim.cp_agent import CPAgent

            options = dict(spec["opponent_search"])
            options["aggression"] = Aggression(options["aggression"])
            self.opponent = CPAgent(SearchConfig(**options))
        else:
            self.opponent = load_agent(spec["opponent"])
        try:
            self.opponent.reset(
                context_for(
                    spec["scenario"],
                    spec["armies"],
                    1 - self.side,
                    spec["max_rounds"],
                    spec["agent_seeds"][1 - self.side],
                )
            )
        except BaseException:
            self.close()
            raise
        self.first_contact_turn = None
        self.damage = [0, 0]
        self._pending_attack = None
        self.diagnostic_events = []
        self._step_before = None

    def prepare(self, action):
        self._pending_attack = None
        if self.spec.get("record_step_events", False):
            self._step_before = self.battle.snapshot()["units"]
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
        if self._step_before is not None:
            after = self.battle.snapshot()["units"]
            self.diagnostic_events.append(
                {
                    "side": self.turns % 2,
                    "turn": self.turns,
                    "action": asdict(action),
                    "hp_before": [u["hp"] for u in self._step_before],
                    "hp_after": [u["hp"] for u in after],
                    "healing": [
                        max(0, b["hp"] - a["hp"])
                        for a, b in zip(self._step_before, after)
                    ],
                }
            )
            self._step_before = None
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
            if callable(getattr(self.opponent, "act_with_planning", None)):
                from civ5sim.combat_ai import CallPlanningServices, PlanningState

                planning = CallPlanningServices(
                    self.opponent.planner,
                    PlanningState.from_battle(self.battle, 1 - self.side),
                    1 - self.side,
                    obs["turn"],
                )
                try:
                    action = self.opponent.act_with_planning(obs, actions, planning)
                finally:
                    planning.expire()
            else:
                action = self.opponent.act(obs, actions)
            self._drain_opponent()
            self.timings["opponent_seconds"] += perf_counter() - start
            if action not in actions:
                raise RuntimeError("Illegal opponent action")
            start = perf_counter()
            self.prepare(action)
            result = self.battle.step(action)
            self.timings["simulation_seconds"] += perf_counter() - start
            self.applied(action, result)
            callback = getattr(self.opponent, "on_action_result", None)
            if callable(getattr(self.opponent, "act_with_planning", None)) and callback:
                start = perf_counter()
                callback(action, result, self.battle.observe(1 - self.side))
                self.timings["opponent_seconds"] += perf_counter() - start
                self._drain_opponent()
        if self.terminal:
            final = self.battle.snapshot()
            return {
                "done": True,
                "joint_commands": self.history,
                "result": {
                    **self.terminal,
                    "reward": 0
                    if self.terminal["winner"] is None
                    else (
                        1
                        if self.terminal["winner"] == self.side
                        else 0
                        if self.reward_profile == "win_probability"
                        else -1
                    ),
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
                    **(
                        {"diagnostic_events": self.diagnostic_events}
                        if self.spec.get("record_step_events", False)
                        else {}
                    ),
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

    def _drain_opponent(self):
        drain = getattr(self.opponent, "drain_diagnostics", None)
        if drain:
            drain()

    def close(self):
        close = getattr(self.opponent, "close", None)
        if close is not None:
            close()
        self.battle.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
