"""CP-05 local-task executor. Full-army coordination belongs to CP-06."""

from collections import deque
from copy import deepcopy
from dataclasses import asdict

from .combat_ai import AssignmentKind as K
from .combat_ai import ContactMemory, Planner, SearchConfig, SearchReason
from .engine import Action, digest


def hex_distance(map_info, a, b):
    """Kernel-compatible offset coordinates, including the horizontal seam."""
    w = map_info["width"]
    ay, by = a // w + map_info["row_parity"], b // w + map_info["row_parity"]
    aq, bq = a % w - ay // 2, b % w - by // 2
    dr = ay - by
    return min(max(abs(aq - bq + shift), abs(dr), abs(aq - bq + shift + dr))
               for shift in ((-w, 0, w) if map_info["wrap_x"] else (0,)))


class CPAgent:
    """One visible target and one group per turn; one simulator Action per call."""

    def __init__(self, config: SearchConfig | None = None):
        self.config = SearchConfig() if config is None else config
        self.planner = None

    def close(self):
        if self.planner is not None:
            self.planner.close()

    def reset(self, context):
        self.close()
        self.planner = Planner(self.config)
        self.side = context["side"]
        self.map = deepcopy(context["map"])
        self.memory = ContactMemory()
        self._events = []
        self._turn = None
        self._new_turn()

    def _new_turn(self):
        self._queue = deque()
        self._pending = None
        self._group = None
        self._finished = set()
        self._done = False
        self._replan = False
        self._searches = 0
        self._search_limit = 0
        self._visited = set()
        self._last_observation = None

    def _event(self, kind, **values):
        self._events.append({"event": kind, "turn": self._turn, **values})

    def drain_diagnostics(self):
        events, self._events = self._events, []
        return events

    def _cancel(self, reason):
        self._queue.clear()
        self._pending = None
        self._replan = True
        self._event("plan_cancelled", reason=reason)

    def _end(self, reason):
        self._done = True
        self._queue.clear()
        self._event("task_finished", reason=reason)
        return self._issue(Action("end_turn"))

    def _issue(self, action):
        self._event("action", action=asdict(action))
        return action

    def act_with_planning(self, observation, actions, planning):
        if observation["side"] != self.side or observation["active_side"] != self.side:
            raise ValueError("CP agent called for an inactive side")
        if self._turn != observation["turn"]:
            if self._queue or self._pending:
                self._cancel("turn_changed")
            self._new_turn()
            self._turn = observation["turn"]
        self.memory = self.memory.updated(observation)
        if self._pending is not None:
            self._cancel("missing_action_result")
        if self._last_observation is not None and observation != self._last_observation:
            self._cancel("state_changed")
        if self._done:
            return self._end("task_already_finished")
        own = {u["id"]: u for u in observation["units"] if u["side"] == self.side}
        enemies = [u for u in observation["units"] if u["side"] != self.side]
        if not enemies:
            return self._end("no_visible_enemy")
        available = [u for u in own.values() if u["moves"] > 0 and not u["waiting"]
                     and u["id"] not in self._finished]
        if self._group is not None:
            available = [u for u in available if u["id"] in self._group]
        if not available:
            return self._end("group_exhausted")
        searches_this_call = 0
        while True:
            if self._queue:
                a = self._queue.popleft()
                if a.kind in (K.INITIAL, K.BLOCKED, K.RESTART, K.FINISH_TEMP, K.SWAP_REVERSE):
                    continue
                u = own.get(a.unit)
                if a.kind == K.FINISH:
                    if u is None or u["moves"] <= 0 or u["waiting"]:
                        self._finished.add(a.unit)
                        continue
                    action = Action("skip", a.unit)
                else:
                    action = Action({K.MOVE: "move", K.MELEE: "melee", K.RANGED: "ranged",
                                     K.SWAP: "swap"}[a.kind], a.unit, a.command_target)
                if u is None or u["tile"] != a.from_tile or action not in actions:
                    self._cancel("precondition_failed")
                    continue
                reverse = None
                if a.kind == K.SWAP:
                    if not self._queue or self._queue[0].kind != K.SWAP_REVERSE:
                        self._cancel("missing_swap_partner")
                        continue
                    reverse = self._queue.popleft()
                    partner = own.get(reverse.unit)
                    if partner is None or partner["tile"] != reverse.from_tile:
                        self._cancel("precondition_failed")
                        continue
                self._pending = (action, a, reverse, deepcopy(observation))
                if a.kind == K.FINISH:
                    self._finished.add(a.unit)
                return self._issue(action)
            if self._searches and not getattr(self, "_replan", False):
                return self._end("plan_completed")
            if searches_this_call >= 2 or (self._group is not None and self._searches >= self._search_limit):
                return self._end("search_limit")
            candidates = [u for u in available if u["id"] not in self._finished]
            if not candidates:
                return self._end("group_exhausted")
            target = min(enemies, key=lambda e: (
                min(hex_distance(self.map, u["tile"], e["tile"]) for u in candidates), e["id"]
            ))["tile"]
            selected = tuple(u["id"] for u in sorted(candidates, key=lambda u: (
                hex_distance(self.map, u["tile"], target), u["id"]
            ))[:13])
            if self._group is None:
                self._group = selected
                self._search_limit = 2 * len(selected) + 1
            identity = digest({"observation": observation, "target": target, "selected": selected})
            if identity in self._visited:
                return self._end("no_progress")
            self._visited.add(identity)
            self._searches += 1
            searches_this_call += 1
            result = planning.search(target, selected, contacts=self.memory)
            self._event("search", target=target, selected_units=list(selected),
                        reason=result.reason.value, stats=asdict(result.stats))
            if result.reason in (SearchReason.INVALID_REQUEST, SearchReason.NOT_IMPLEMENTED):
                raise RuntimeError(f"CP search failed: {result.reason}")
            if result.reason == SearchReason.TURN_CHANGED:
                return self._end("turn_changed")
            if not result.assignments:
                return self._end(result.reason.value)
            self._replan = False
            self._queue.extend(result.assignments)

    def on_action_result(self, action, result, observation):
        """Validate immediately, including the action that ends a battle."""
        self.memory = self.memory.updated(observation)
        if self._pending is None:
            self._event("action_result", action=asdict(action), reason="turn_ended")
            self._last_observation = deepcopy(observation)
            return
        expected_action, a, reverse, before = self._pending
        self._pending = None
        old = {u["id"]: u for u in before["units"]}
        now = {u["id"]: u for u in observation["units"]}
        reason = None
        if action != expected_action or not result.ok:
            reason = "command_failed"
        elif observation["turn"] != before["turn"] or observation["active_side"] != self.side:
            reason = "turn_changed"
        elif any(u["side"] != self.side and u["id"] not in old for u in now.values()):
            reason = "new_contact"
        elif result.damage != a.damage or result.retaliation != a.self_damage:
            reason = "damage_changed"
        else:
            expected = {}
            u = old[a.unit]
            hp = max(0, u["hp"] - a.self_damage)
            if hp:
                tile = a.to_tile if a.kind in (K.MOVE, K.SWAP) else u["tile"]
                if a.kind == K.MELEE and a.killed_unit >= 0:
                    tile = a.to_tile
                expected[a.unit] = {**u, "hp": hp, "tile": tile, "moves": a.remaining_moves}
                if a.kind != K.FINISH:
                    expected[a.unit].update(acted=True, fortified=False)
                if a.kind in (K.MELEE, K.RANGED):
                    expected[a.unit]["attacked"] = True
            else:
                expected[a.unit] = None
            if a.kind == K.FINISH:
                expected[a.unit]["waiting"] = True
            if reverse is not None:
                expected[a.unit]["waiting"] = False
                expected[reverse.unit] = {**old[reverse.unit], "tile": reverse.to_tile,
                                          "moves": reverse.remaining_moves, "waiting": False,
                                          "acted": True, "fortified": False}
            if a.kind in (K.MELEE, K.RANGED):
                defender = next((u for u in old.values() if u["tile"] == a.command_target
                                 and u["side"] != self.side), None)
                if defender is None:
                    reason = "target_changed"
                else:
                    remaining = max(0, defender["hp"] - a.damage)
                    expected[defender["id"]] = {"hp": remaining} if remaining else None
            for uid, values in expected.items():
                if values is None:
                    matches = uid not in now
                else:
                    matches = uid in now and all(now[uid].get(k) == v for k, v in values.items())
                if not matches:
                    reason = "state_changed"
                    break
            if reason is None and a.kind == K.MOVE and a.to_tile != a.command_target:
                reason = "movement_interrupted"
        self._event("action_result", action=asdict(action), reason=reason or "matched")
        if reason is not None:
            self._cancel(reason)
        self._last_observation = deepcopy(observation)

    def __del__(self):
        self.close()
