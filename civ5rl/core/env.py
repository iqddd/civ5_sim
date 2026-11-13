"""One learner command, then the frozen opponent up to the next learner call."""

from dataclasses import asdict
from time import perf_counter
from typing import Any, Literal, NamedTuple

from civ5rl.core.context import context_for
from civ5rl.core.contracts import ScenarioSpec, WorkerState
from civ5rl.core.observation import ordered_actions
from civ5sim.agents import load_agent
from civ5sim.engine import Action, Battle

ATTACK_KINDS = ("melee", "ranged")
END_TURN = "end_turn"
NUM_SIDES = 2


class PendingAttack(NamedTuple):
    attacker_side: int
    attacker_hp: int
    defender_side: int
    defender_hp: int


class PreparedStep(NamedTuple):
    action: Action
    acting_side: int
    units_before: list[dict[str, Any]] | None
    pending_attack: PendingAttack | None


def learner_outcome(winner: int | None, learner_side: int) -> Literal["W", "D", "L"]:
    """Return the learner's W/D/L match outcome independently of reward_profile."""
    if learner_side not in (0, 1) or winner not in (None, 0, 1):
        raise ValueError("Invalid side or winner")
    return "D" if winner is None else "W" if winner == learner_side else "L"


def _terminal_reward(winner: int | None, side: int, profile: str) -> int:
    if winner is None:
        return 0
    if winner == side:
        return 1
    return 0 if profile == "win_probability" else -1


def terminal_result(battle, completed_turns, max_rounds):
    destroyed = battle.army_status()["destroyed"]
    if any(destroyed):
        return {
            "winner": None if all(destroyed) else (1 if destroyed[0] else 0),
            "reason": "mutual_destruction" if all(destroyed) else "army_destroyed",
        }
    if completed_turns >= NUM_SIDES * max_rounds:
        return {"winner": None, "reason": "turn_limit"}
    return None


class DecisionEnv:
    def __init__(self, spec: ScenarioSpec):
        self.spec = spec
        self.attack_preview = False
        self.learner_decisions = 0
        self.learner_decision_limit = spec.get("learner_decision_limit")
        if self.learner_decision_limit is not None and (
            type(self.learner_decision_limit) is not int or self.learner_decision_limit < 1
        ):
            raise ValueError("Learner decision limit must be a positive integer")
        self.battle = Battle(spec["scenario"], log=False)
        self.side = spec["learner_side"]
        self.turns = 0
        self.calls = 0
        self.commands = 0
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
        self._prepared_step: PreparedStep | None = None
        self.diagnostic_events = []
        self.collection_diagnostics = None
        self.diagnostic_key = None

    def prepare_step(self, action) -> PreparedStep:
        acting_side = self.turns % NUM_SIDES
        record_events = self.spec.get("record_step_events", False)
        need_attack = bool(self.spec.get("diagnostics") and action.kind in ATTACK_KINDS)
        units = self.battle.snapshot()["units"] if record_events or need_attack else None
        units_before = units if record_events else None
        pending_attack = None
        if need_attack and units is not None and 0 <= action.unit < len(units):
            attacker = units[action.unit]
            defender = next((u for u in units if u["tile"] == action.target and u["hp"] > 0), None)
            if defender is not None:
                pending_attack = PendingAttack(
                    attacker["side"],
                    attacker["hp"],
                    defender["side"],
                    defender["hp"],
                )
        return PreparedStep(action, acting_side, units_before, pending_attack)

    def finish_step(self, prepared: PreparedStep, result):
        action = prepared.action
        self.history.append(asdict(action))
        if not result.ok:
            raise RuntimeError(f"Kernel rejected public legal action: {result.error}")
        if prepared.units_before is not None:
            after = self.battle.snapshot()["units"]
            self.diagnostic_events.append(
                {
                    "side": prepared.acting_side,
                    "turn": self.turns,
                    "action": asdict(action),
                    "hp_before": [u["hp"] for u in prepared.units_before],
                    "hp_after": [u["hp"] for u in after],
                    "healing": [
                        max(0, unit_after["hp"] - unit_before["hp"])
                        for unit_before, unit_after in zip(prepared.units_before, after)
                    ],
                }
            )
        if prepared.acting_side == self.side:
            self.learner_decisions += 1
        if prepared.pending_attack is not None:
            attack = prepared.pending_attack
            self.damage[attack.attacker_side] += min(result.damage, attack.defender_hp)
            self.damage[attack.defender_side] += min(result.retaliation, attack.attacker_hp)
        self.commands += 1
        self.calls += 1
        if action.kind == END_TURN:
            self.turns += 1
            self.calls = 0

    def step(self, action):
        prepared = self.prepare_step(action)
        result = self.battle.step(action)
        self.finish_step(prepared, result)
        return result

    def prepare(self, action) -> PreparedStep:
        self._prepared_step = self.prepare_step(action)
        return self._prepared_step

    def check(self):
        self.terminal = terminal_result(self.battle, self.turns, self.spec["max_rounds"])
        if (
            self.terminal is None
            and self.learner_decision_limit is not None
            and self.learner_decisions >= self.learner_decision_limit
        ):
            self.terminal = {"winner": None, "reason": "diagnostic_decision_limit"}
        return self.terminal

    def applied(self, action, result):
        prepared = self._prepared_step
        self._prepared_step = None
        if prepared is None or prepared.action != action:
            prepared = PreparedStep(action, self.turns % NUM_SIDES, None, None)
        self.finish_step(prepared, result)

    def request(self) -> WorkerState:
        state = self.request_slice()
        while state is None:
            state = self.request_slice()
        return state

    def request_slice(self) -> WorkerState | None:
        """Advance at most one opponent command; None yields this CPU slot."""
        if not self.check() and self.turns % NUM_SIDES != self.side:
            if self.calls >= self.spec["max_calls_per_turn"]:
                raise RuntimeError("Opponent action limit")
            obs = self.battle.observe(1 - self.side)
            actions = tuple(self.battle.available_actions())
            start = perf_counter()
            diag = self.collection_diagnostics
            cp = callable(getattr(self.opponent, "act_with_planning", None))
            clock = diag.begin() if diag else None
            if cp:
                from civ5sim.combat_ai import CallPlanningServices, PlanningState

                planning = CallPlanningServices(
                    self.opponent.planner,
                    PlanningState.from_battle(self.battle, 1 - self.side, observation=obs),
                    1 - self.side,
                    obs["turn"],
                )
                try:
                    action = self.opponent.act_with_planning(obs, actions, planning)
                finally:
                    planning.expire()
            else:
                action = self.opponent.act(obs, actions)
            if diag:
                diag.end(
                    "cp_decision" if cp else "opponent_decision",
                    clock,
                    event={"key": self.diagnostic_key},
                )
            self._drain_opponent()
            self.timings["opponent_seconds"] += perf_counter() - start
            if action not in actions:
                raise RuntimeError("Illegal opponent action")
            start = perf_counter()
            result = self.step(action)
            self.timings["simulation_seconds"] += perf_counter() - start
            callback = getattr(self.opponent, "on_action_result", None)
            if callable(getattr(self.opponent, "act_with_planning", None)) and callback:
                start = perf_counter()
                clock = diag.begin() if diag else None
                callback(action, result, self.battle.observe(1 - self.side))
                if diag:
                    diag.end("cp_callback", clock, event={"key": self.diagnostic_key})
                self.timings["opponent_seconds"] += perf_counter() - start
                self._drain_opponent()
            return None
        if self.terminal:
            final = self.battle.snapshot()
            return {
                "done": True,
                "joint_commands": self.history,
                "result": {
                    **self.terminal,
                    "reward": _terminal_reward(self.terminal["winner"], self.side, self.reward_profile),
                    "commands": self.commands,
                    "learner_decisions": self.learner_decisions,
                    "rounds": self.turns / NUM_SIDES,
                    "timings": self.timings.copy(),
                    "first_contact_turn": self.first_contact_turn,
                    "remaining_hp": [
                        sum(u["hp"] for u in final["units"] if u["side"] == s) for s in range(NUM_SIDES)
                    ],
                    "damage_dealt": self.damage.copy(),
                    "units_lost": [
                        sum(u["hp"] <= 0 for u in final["units"] if u["side"] == s) for s in range(NUM_SIDES)
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
        if self.first_contact_turn is None and any(u["side"] != self.side for u in observation["units"]):
            self.first_contact_turn = self.turns
        if self.attack_preview:
            actions, previews = self.battle.actions_with_previews()
            observation["attack_previews"] = previews
        else:
            actions = ordered_actions(self.battle.available_actions())
        return {
            "done": False,
            "observation": observation,
            "actions": actions,
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
