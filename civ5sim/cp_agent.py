"""CP-06 full-battle coordinator and checked, one-command CP executor."""

from collections import deque
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction

from civ5map.maps import adjacent, passable

from .combat_ai import (
    Aggression,
    Assignment,
    ContactMemory,
    Planner,
    SearchConfig,
    SearchReason,
    SupportMode,
    own_turn,
)
from .combat_ai import AssignmentKind as K
from .engine import Action, digest


def _observation_snapshot(observation):
    """Own the public schema's lists/dicts; unit fields and headers are scalars."""
    return {
        **observation,
        "visibility": observation["visibility"].copy(),
        "units": [unit.copy() for unit in observation["units"]],
    }


def hex_distance(map_info, a, b):
    """Kernel-compatible offset coordinates, including the horizontal seam."""
    w = map_info["width"]
    ay, by = a // w + map_info["row_parity"], b // w + map_info["row_parity"]
    aq, bq = a % w - ay // 2, b % w - by // 2
    dr = ay - by
    return min(max(abs(aq - bq + shift), abs(dr), abs(aq - bq + shift + dr))
               for shift in ((-w, 0, w) if map_info["wrap_x"] else (0,)))


class CPAgent:
    """Sequential army tasks, wide exploration, and one simulator Action per call."""

    def __init__(self, config: SearchConfig | None = None):
        self.config = SearchConfig() if config is None else config
        self.planner = None

    def close(self):
        if self.planner is not None:
            self.planner.close()

    def reset(self, context):
        self._decision_recorder = None
        self.close()
        self.planner = Planner(self.config)
        self.side = context["side"]
        self.map = deepcopy(context["map"])
        self.memory = ContactMemory()
        self._last_seen = {}
        self._seen_at = [-1] * len(self.map["tiles"])
        self._objectives = {}
        self._blocked = {}
        self._combat_targets = {}
        self._target_blocked = {}
        self._routes = {}
        self._neighbors = {}
        for tile, value in enumerate(self.map["tiles"]):
            if not passable(value):
                continue
            neighbors = []
            for direction in range(6):
                xy = adjacent(self.map, tile % self.map["width"], tile // self.map["width"], direction)
                if xy is not None:
                    n = xy[1] * self.map["width"] + xy[0]
                    if passable(self.map["tiles"][n]):
                        neighbors.append(n)
            self._neighbors[tile] = tuple(neighbors)
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
        self._cancelled_plan = False
        self._searches = 0
        self._call_searches = 0
        self._call_combat_task = None
        self._probe_searches = 0
        self._search_cache = {}
        self._cache_observation = None
        self._search_limit = 0
        self._support_calls = 0
        self._support_limit = 0
        self._visited = set()
        self._last_observation = None
        self._task = None
        self._phase = 0
        self._phase_seen = {0: set(), 2: set(), 3: set()}
        self._combat_attempts = set()
        self._approach_attempts = set()
        self._reservations = []
        self._recovery_calls = 0
        self._recovery_limit = 0
        self._recovery_attempts = set()
        self._recovery_search_attempts = set()
        self._recovery_moved = set()
        self._recovery_units = set()
        self._retry_count = 0

    def _observe(self, observation):
        self.memory = self.memory.updated(observation)
        now = own_turn(observation["turn"], observation["active_side"], self.side)
        visible = observation["visibility"]
        enemies = {u["id"]: u for u in observation["units"] if u["side"] != self.side}
        for tile, seen in enumerate(visible):
            if seen:
                self._seen_at[tile] = now
        active = {c.id for c in self.memory.known_enemies}
        self._last_seen = {uid: tile for uid, tile in self._last_seen.items()
                           if uid in active and (not visible[tile] or
                              (uid in enemies and enemies[uid]["tile"] == tile))}
        self._last_seen.update({uid: u["tile"] for uid, u in enemies.items()})

    def _distances(self, target):
        if target not in self._routes:
            distances = {target: 0} if target in self._neighbors else {}
            queue = deque(distances)
            while queue:
                tile = queue.popleft()
                for n in self._neighbors[tile]:
                    if n not in distances:
                        distances[n] = distances[tile] + 1
                        queue.append(n)
            if len(self._routes) >= 64:
                self._routes.pop(next(iter(self._routes)))
            self._routes[target] = distances
        return self._routes[target]

    def _distance(self, start, target):
        return self._distances(start).get(target, 1_000_000)

    def _select(self, units, target, limit=13):
        return tuple(u["id"] for u in sorted(units, key=lambda u: (
            self._distance(u["tile"], target), u["id"]
        ))[:limit])

    def _explore_target(self, group, observation):
        key = tuple(sorted(u["id"] for u in group))
        now = own_turn(observation["turn"], observation["active_side"], self.side)
        previous = self._objectives.get(key)
        def distance(tile):
            return min(self._distance(u["tile"], tile) for u in group)
        if previous is not None:
            target, old_distance, stalled, last_turn = previous
            if last_turn != now:
                stalled = stalled + 1 if distance(target) >= old_distance else 0
            if stalled >= 2:
                self._blocked[key, target] = now + 1
            elif not observation["visibility"][target]:
                self._objectives[key] = (target, distance(target), stalled, now)
                self._reservations.append(target)
                return target
        candidates = [t for t in self._neighbors if distance(t) < 1_000_000
                      and self._blocked.get((key, t), -1) < now]
        if not candidates:
            return None
        target = min(candidates, key=lambda t: (
            self._seen_at[t],
            sum(max(0, 5 - hex_distance(self.map, t, other)) for other in self._reservations),
            distance(t), t,
        ))
        self._objectives[key] = (target, distance(target), 0, now)
        self._reservations.append(target)
        return target

    def _group_rank(self, task, own, actions):
        """Prefer the largest joint roster; geometry only breaks head-count ties."""
        group = [own[uid] for uid in task["selected"]]
        ready = {a.unit for a in actions if a.kind in ("melee", "ranged")
                 and a.target == task["target"] and a.unit in task["selected"]}
        target_distance = sum(self._distance(u["tile"], task["target"]) for u in group)
        rank = (len(group), len(ready), sum(u["hp"] for u in group),
                -Fraction(target_distance, len(group)), -task["enemy_id"])
        task["ranking"] = {"participants": len(group), "immediate_attackers": len(ready),
                           "hp": rank[2], "target_distance_sum": target_distance}
        return rank

    def _approach_group(self, units, enemies, observation):
        """Group reinforcements only around their own nearest allowed targets.

        These route lengths count passable edges, not movement costs or turns.
        Combat recruitment and physical destinations remain native decisions.
        """
        now = own_turn(observation["turn"], observation["active_side"], self.side)
        targets = sorted([(e["id"], e["tile"]) for e in enemies] if enemies
                         else list(self._last_seen.items()))
        groups = {eid: [] for eid, _ in targets}
        evidence = []
        for unit in sorted(units, key=lambda u: u["id"]):
            distances = []
            choices = []
            for eid, target in targets:
                distance = self._distance(unit["tile"], target)
                attempted = (unit["id"], target) in self._approach_attempts
                blocked = self._target_blocked.get((unit["id"], eid), -1) >= now
                distances.append({"enemy": eid, "target": target, "distance": distance,
                                  "already_attempted": attempted, "temporarily_blocked": blocked})
                if not attempted and not blocked and distance < 1_000_000:
                    choices.append((distance, eid))
            minimum = min((distance for distance, _ in choices), default=None)
            nearest = [eid for distance, eid in choices if distance == minimum]
            for eid in nearest:
                groups[eid].append(unit)
            evidence.append({"unit": unit["id"], "targets": distances,
                             "minimum_distance": minimum, "nearest_enemies": nearest})

        options = []
        own = {u["id"]: u for u in units}
        for eid, target in targets:
            if not groups[eid]:
                continue
            selected = self._select(groups[eid], target)
            options.append({"enemy": eid, "target": target, "selected_units": list(selected),
                            "distance_sum": sum(self._distance(own[uid]["tile"], target)
                                                for uid in selected)})
        options.sort(key=lambda option: (-len(option["selected_units"]),
                                        option["distance_sum"], option["enemy"]))
        winner = options[0] if options else None
        self._event("approach_target_selection", policy="nearest_per_unit",
                    source="visible" if enemies else "last_seen", units=evidence,
                    candidates=options, target=winner["target"] if winner else None,
                    reason="largest_nearest_group" if winner else "no_allowed_target")
        if winner is None:
            return None
        return winner["target"], tuple(winner["selected_units"])

    def _next_task(self, available, enemies, observation, eligibility=None, actions=()):
        available = [u for u in available if u["id"] not in self._recovery_units]
        while self._phase < 4:
            phase = self._phase
            if phase == 1:
                now = own_turn(observation["turn"], observation["active_side"], self.side)
                by_id = {e["id"]: e for e in enemies}
                for u in available:
                    previous = self._combat_targets.get(u["id"])
                    if previous is not None and previous[0] in by_id:
                        eid, old_distance, stalled, last_turn = previous
                        distance = self._distance(u["tile"], by_id[eid]["tile"])
                        if last_turn != now:
                            stalled = stalled + 1 if distance >= old_distance else 0
                        if stalled >= 2:
                            self._target_blocked[u["id"], eid] = now + 1
                            self._combat_targets.pop(u["id"], None)
                        else:
                            self._combat_targets[u["id"]] = (eid, distance, stalled, now)
                recorder = getattr(self, "_decision_recorder", None)
                if recorder is not None:
                    for enemy in enemies:
                        recorder.event({"event": "coordinator_eligibility", "enemy": enemy["id"],
                                        "target": enemy["tile"], "participants": [
                                            {"unit": u["id"],
                                             "already_attempted": (u["id"], enemy["id"]) in self._combat_attempts,
                                             "native_admitted": eligibility is None or u["id"] in eligibility.get(enemy["id"], ()),
                                             "temporarily_blocked": self._target_blocked.get((u["id"], enemy["id"]), -1) >= now}
                                            for u in available]}, self._task)
                if available:
                    options = []
                    own = {u["id"]: u for u in available}
                    for enemy_id in sorted(by_id):
                        candidate_enemy = by_id[enemy_id]
                        candidates = [u for u in available if (u["id"], enemy_id) not in self._combat_attempts
                                      and (eligibility is None or u["id"] in eligibility.get(enemy_id, ()))
                                      and self._target_blocked.get((u["id"], enemy_id), -1) < now]
                        if not candidates:
                            continue
                        options.append({"kind": "combat", "target": candidate_enemy["tile"],
                                        "enemy_id": enemy_id,
                                        "selected": self._select(candidates, candidate_enemy["tile"])})
                    if not options:
                        self._phase += 1
                        continue
                    options.sort(key=lambda t: self._group_rank(t, own, actions), reverse=True)
                    primary = options[0]
                    primary["ranked_candidates"] = [
                        {"target": t["target"], "selected_units": list(t["selected"]),
                         "ranking": t["ranking"]} for t in options]
                    primary_roster = set(primary["selected"])
                    alternative = next((option for option in options[1:]
                                        if set(option["selected"]) == primary_roster), None)
                    if alternative is not None:
                        primary["alternative"] = alternative
                    return primary
            else:
                candidates = [u for u in available if u["id"] not in self._phase_seen[phase]]
                if candidates:
                    if phase == 3:
                        if enemies or self._last_seen:
                            group = self._approach_group(candidates, enemies, observation)
                            if group is None:
                                self._phase_seen[phase].update(u["id"] for u in candidates)
                                continue
                            target, selected = group
                        else:
                            anchor = min(candidates, key=lambda u: u["id"])
                            selected = self._select(candidates, anchor["tile"], 4)
                            group = [u for u in candidates if u["id"] in selected]
                            target = self._explore_target(group, observation)
                        kind = "approach"
                    else:
                        selected = tuple(sorted(u["id"] for u in candidates)[:13])
                        target = next(u["tile"] for u in candidates if u["id"] == selected[0])
                        kind = "heal_before" if phase == 0 else "heal_after"
                    self._phase_seen[phase].update(selected)
                    if target is not None:
                        if phase == 3:
                            self._approach_attempts.update((uid, target) for uid in selected)
                        return {"kind": kind, "target": target, "selected": selected}
                    continue
            self._phase += 1
        return None

    def _finish_task(self, reason, *, retry_approach=False):
        if self._task is not None:
            if retry_approach and self._task["kind"] == "approach":
                self._phase_seen[3].difference_update(self._task["selected"])
            if self._task["kind"] == "combat":
                self._phase = min(self._phase, 1)
            self._event("task_finished", task_kind=self._task["kind"],
                        target=self._task["target"], reason=reason)
        self._task = None
        self._group = None
        self._replan = False

    def diagnostic_state(self):
        """Read-only dispatch evidence; never used as an input to decisions."""
        return {"phase": self._phase, "finished": sorted(self._finished),
                "task": deepcopy(self._task), "queued_assignments": [asdict(a) for a in self._queue],
                "searches": self._searches, "search_limit": self._search_limit,
                "support_calls": self._support_calls, "support_limit": self._support_limit,
                "recovery_calls": self._recovery_calls, "recovery_limit": self._recovery_limit,
                "combat_attempts": sorted(self._combat_attempts),
                "recovery_units": sorted(self._recovery_units),
                "recovery_moved": sorted(self._recovery_moved), "replan": self._replan}

    def _event(self, kind, **values):
        event = {"event": kind, "turn": self._turn, **values}
        self._events.append(event)
        recorder = getattr(self, "_decision_recorder", None)
        if recorder is not None:
            recorder.event(event, self._task)

    def drain_diagnostics(self):
        events, self._events = self._events, []
        return events

    def _cancel(self, reason):
        self._queue.clear()
        self._pending = None
        self._replan = True
        self._cancelled_plan = True
        self._event("plan_cancelled", reason=reason)
        if self._task is not None and self._task["kind"] == "recovery":
            # Recovery is re-evaluated from real state, never dispatched as an
            # ordinary healing/approach task after random damage or contact.
            self._finish_task(reason)
            self._phase = min(self._phase, 1)

    def _end(self, reason):
        self._done = True
        self._queue.clear()
        self._event("task_finished", reason=reason)
        return self._issue(Action("end_turn"))

    def _issue(self, action):
        recorder = getattr(self, "_decision_recorder", None)
        if recorder is not None:
            recorder.action(action, self._pending[1] if self._pending else None)
        self._event("action", action=asdict(action))
        return action

    def _wait(self, unit, observation, reason):
        action = Action("skip", unit["id"])
        a = Assignment(K.FINISH, unit["id"], unit["tile"], unit["tile"],
                       unit["moves"], 0, 0, 0, 0, -1)
        self._pending = (action, a, None, _observation_snapshot(observation))
        self._finished.add(unit["id"])
        self._event("waiting", unit=unit["id"], reason=reason)
        return self._issue(action)

    def _recover(self, available, observation, planning, *, allow_wait=True):
        """One bounded tactical decision, outside normal search/support budgets."""
        state = digest(observation)
        for unit in sorted(available, key=lambda value: value["id"]):
            # CP PerformRangedOpportunityAttack(true): a normal LOW search,
            # before fleeing with fast ranged units or after an actual retreat.
            # Keep it in the ordinary search ceiling; support never nests searches.
            identity = (state, unit["id"])
            ranged = unit["type"] in ("archer", "chariot_archer")
            moving_shot = (unit["type"] == "chariot_archer" and unit["moves"] > 60
                           or unit["id"] in self._recovery_moved)
            if (ranged and moving_shot and not unit["attacked"] and self._searches < self._search_limit
                    and self._call_searches < 2
                    and identity not in self._recovery_search_attempts):
                self._recovery_search_attempts.add(identity)
                self._searches += 1
                self._call_searches += 1
                result = planning.search(unit["tile"], (unit["id"],), contacts=self.memory,
                                         aggression=Aggression.LOW)
                self._event("search", task_kind="recovery", target=unit["tile"],
                            selected_units=[unit["id"]], probe=False,
                            reason=result.reason.value, stats=asdict(result.stats),
                            assignments=len(result.assignments), replanning=self._cancelled_plan,
                            diagnostics=asdict(result.diagnostics))
                if result.reason in (SearchReason.INVALID_REQUEST, SearchReason.NOT_IMPLEMENTED,
                                     SearchReason.TURN_CHANGED):
                    raise RuntimeError(f"CP opportunity search failed: {result.reason}")
                if any(a.kind in (K.MOVE, K.MELEE, K.RANGED, K.SWAP) for a in result.assignments):
                    self._task = {"kind": "recovery", "target": unit["tile"],
                                  "selected": (unit["id"],)}
                    self._group = self._task["selected"]
                    self._replan = False
                    self._queue.extend(result.assignments)
                    return True
            if self._recovery_calls >= self._recovery_limit:
                return False
            if identity in self._recovery_attempts:
                continue
            self._recovery_attempts.add(identity)
            self._recovery_calls += 1
            result = planning.support(unit["tile"], (unit["id"],), contacts=self.memory,
                                      mode=SupportMode.RECOVERY)
            self._event("recovery", target=unit["tile"], selected_units=[unit["id"]],
                        reason=result.reason.value, stats=asdict(result.stats),
                        assignments=len(result.assignments), diagnostics=asdict(result.diagnostics))
            if result.reason in (SearchReason.INVALID_REQUEST, SearchReason.NOT_IMPLEMENTED,
                                 SearchReason.TURN_CHANGED):
                raise RuntimeError(f"CP recovery failed: {result.reason}")
            if not allow_wait and result.diagnostics.recovery_kind == "wait":
                continue
            if result.assignments:
                self._task = {"kind": "recovery", "target": unit["tile"], "selected": (unit["id"],)}
                self._group = self._task["selected"]
                self._replan = False
                self._queue.extend(result.assignments)
                return True
        return False

    def _enter_recovery(self, task, available, reason):
        """Transfer the whole remaining roster; membership lasts for this turn."""
        eligible = {u["id"] for u in available}
        selected = tuple(uid for uid in task["selected"] if uid in eligible)
        self._recovery_units.update(selected)
        self._event("group_recovery", target=task["target"], selected_units=list(selected),
                    original_units=list(task["selected"]), attempts=task.get("attempts", 0),
                    reason=reason)
        self._finish_task(reason)

    @staticmethod
    def _prefix_value(result, observation):
        """Compare physical HP changes, never scores from different CP groups.

        RESTART follows the command that exposed an enemy. Later hypothetical
        commands cannot be relied upon by the real coordinator.
        """
        hp = {u["id"]: u["hp"] for u in observation["units"]}
        occupants = {u["tile"]: u["id"] for u in observation["units"]}
        enemy_damage = own_damage = enemy_kills = own_kills = 0
        commands = 0
        for a in result.assignments:
            if a.kind == K.RESTART:
                break
            if a.kind not in (K.MOVE, K.MELEE, K.RANGED, K.SWAP):
                continue
            commands += 1
            before = hp[a.unit]
            lost = min(before, a.self_damage)
            hp[a.unit] -= lost
            own_damage += lost
            own_kills += before > 0 and hp[a.unit] == 0
            if a.kind in (K.MELEE, K.RANGED):
                victim = occupants[a.command_target]
                before = hp[victim]
                lost = min(before, a.damage)
                hp[victim] -= lost
                enemy_damage += lost
                enemy_kills += before > 0 and hp[victim] == 0
        return {"net_hp": enemy_damage - own_damage, "net_kills": enemy_kills - own_kills,
                "own_damage": own_damage, "enemy_damage": enemy_damage,
                "commands": commands}

    def _combat_search(self, task, observation, own, enemies, planning, *, probe=False):
        target, selected = task["target"], task["selected"]
        local = [e for e in enemies if hex_distance(self.map, target, e["tile"]) <= 5
                 or any(hex_distance(self.map, own[uid]["tile"], e["tile"]) <= 4 for uid in selected)]
        aggression = task.get("aggression")
        if aggression is None:
            aggression = self.config.aggression
            if sum(own[uid]["hp"] for uid in selected) < sum(e["hp"] for e in local):
                aggression = min(aggression, Aggression.LOW)
            task["aggression"] = aggression
        identity = digest({"observation": observation, "target": target, "selected": selected,
                           "aggression": int(aggression), "config": asdict(self.config),
                           "contacts": asdict(self.memory)})
        reused = identity in self._search_cache
        if reused:
            result = self._search_cache[identity]
        else:
            if probe:
                self._probe_searches += 1
            else:
                self._searches += 1
                self._call_searches += 1
                self._call_combat_task = task
            self._event("posture", aggression=Aggression(aggression).name.lower(), target=target)
            result = planning.search(target, selected, contacts=self.memory, aggression=Aggression(aggression))
            self._search_cache[identity] = result
        task["attempts"] = task.get("attempts", 0) + 1
        if task.get("enemy_id") is not None:
            self._combat_attempts.update((uid, task["enemy_id"]) for uid in selected)
        self._event("search_reused" if reused else "search", target=target,
                    task_kind="combat", selected_units=list(selected), probe=probe,
                    reason=result.reason.value, stats=asdict(result.stats),
                    assignments=len(result.assignments), replanning=self._cancelled_plan,
                    diagnostics=asdict(result.diagnostics))
        if result.reason in (SearchReason.INVALID_REQUEST, SearchReason.NOT_IMPLEMENTED):
            raise RuntimeError(f"CP search failed: {result.reason}")
        return result, reused

    def _choose_combat_plan(self, task, observation, own, enemies, planning):
        # A fallback can change the target, never the roster. A useful result
        # must contain a physical command before its first contact restart.
        alternative = task.pop("alternative", None)
        if alternative is not None and set(alternative["selected"]) != set(task["selected"]):
            alternative = None
        ranked_candidates = task.pop("ranked_candidates", None)
        primary_result, primary_reused = self._combat_search(task, observation, own, enemies, planning)
        choices = [(task, primary_result, primary_reused)]
        if (not self._has_commands(primary_result) and alternative is not None
                and primary_result.reason != SearchReason.TURN_CHANGED
                and self._probe_searches < self._searches):
            result, reused = self._combat_search(alternative, observation, own, enemies, planning, probe=True)
            choices.append((alternative, result, reused))
        winner = 0
        rows = []
        for index, (option, result, reused) in enumerate(choices):
            value = self._prefix_value(result, observation)
            if self._has_commands(result):
                winner = index
            rows.append({"target": option["target"], "selected_units": list(option["selected"]),
                         "ranking": option.get("ranking"), "prefix": value,
                         "reason": result.reason.value, "stats": asdict(result.stats),
                         "assignments": len(result.assignments), "reused": reused})
        chosen, result, _ = choices[winner]
        if len(choices) > 1:
            chosen["attempts"] = sum(option["attempts"] for option, _, _ in choices)
        # Empty alternatives retain the primary diagnostics for investigation.
        self._task = chosen
        self._group = chosen["selected"]
        for uid in chosen["selected"]:
            if chosen.get("enemy_id") is not None and self._combat_targets.get(uid, (None,))[0] != chosen["enemy_id"]:
                self._combat_targets[uid] = (chosen["enemy_id"], self._distance(own[uid]["tile"], chosen["target"]),
                                             0, own_turn(observation["turn"], observation["active_side"], self.side))
        self._event("target_comparison", candidates=rows, chosen_target=chosen["target"],
                    ranked_candidates=ranked_candidates, policy="max_participants",
                    chosen_index=winner, reason="fallback_after_empty" if winner else
                    "max_participants" if self._has_commands(result) else "all_empty")
        if winner:
            self._event("task_assigned", task_kind="combat", target=chosen["target"],
                        selected_units=list(chosen["selected"]))
        return result

    @staticmethod
    def _has_commands(result):
        for assignment in result.assignments:
            if assignment.kind == K.RESTART:
                break
            if assignment.kind in (K.MOVE, K.MELEE, K.RANGED, K.SWAP, K.FINISH):
                return True
        return False

    def act_with_planning(self, observation, actions, planning):
        if observation["side"] != self.side or observation["active_side"] != self.side:
            raise ValueError("CP agent called for an inactive side")
        if self._turn != observation["turn"]:
            if self._queue or self._pending:
                self._cancel("turn_changed")
            self._new_turn()
            self._turn = observation["turn"]
        self._observe(observation)
        cache_observation = digest(observation)
        if self._cache_observation != cache_observation:
            self._search_cache.clear()
            self._cache_observation = cache_observation
        if self._pending is not None:
            self._cancel("missing_action_result")
        if self._last_observation is not None and observation != self._last_observation:
            self._cancel("state_changed")
        if self._done:
            return self._end("task_already_finished")
        own = {u["id"]: u for u in observation["units"] if u["side"] == self.side}
        enemies = [u for u in observation["units"] if u["side"] != self.side]
        available = [u for u in own.values() if u["moves"] > 0 and not u["waiting"]
                     and u["id"] not in self._finished]
        if not available:
            return self._end("army_exhausted")
        if not self._search_limit:
            n = len(available)
            self._search_limit = 2*n + (n+12)//13
            self._support_limit = 3*n + (n+12)//13
            # Speculative waits are deferred so APPROACH remains possible.
            # They may be rechecked after an ally frees a tile. There are at
            # most 5N physical commands and N distinct unit probes per state.
            # This is a structural ceiling, not a reason to terminate a unit.
            self._recovery_limit = n * (5*n + 1)
        self._call_searches = 0
        self._call_combat_task = None
        eligibility = None
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
                self._pending = (action, a, reverse, _observation_snapshot(observation))
                if a.kind == K.FINISH:
                    self._finished.add(a.unit)
                return self._issue(action)
            available = [u for u in available if u["id"] not in self._finished]
            if not available:
                return self._end("army_exhausted")
            if self._task is not None and not self._replan:
                self._finish_task("plan_completed", retry_approach=bool(enemies or self._last_seen))
            if self._task is None:
                recovering = [u for u in available if u["id"] in self._recovery_units]
                if recovering:
                    if self._recover(recovering, observation, planning):
                        continue
                    # A failed/metadata-only continuation must still yield a
                    # physical command for THIS group, never a bystander.
                    return self._wait(min(recovering, key=lambda u: u["id"]), observation,
                                      "recovery_no_continuation")
            if self._task is None:
                if eligibility is None:
                    eligibility = {}
                    roster = tuple(u["id"] for u in available if u["id"] not in self._recovery_units)
                    if roster:
                        for enemy in enemies:
                            recruited = planning.attack_candidates(enemy["tile"], roster, contacts=self.memory)
                            eligibility[enemy["id"]] = frozenset(recruited)
                            self._event("attack_candidates", target=enemy["tile"],
                                        available_units=list(roster), selected_units=list(recruited))
                if (self._phase >= 4 and self._call_searches < 2 and self._searches < self._search_limit
                        and any((unit["id"], enemy["id"]) not in self._combat_attempts
                                and unit["id"] in eligibility.get(enemy["id"], ())
                                for unit in available for enemy in enemies)):
                    self._phase = 1
                self._task = self._next_task(available, enemies, observation, eligibility, actions)
                if self._task is None:
                    self._enter_recovery({"target": available[0]["tile"],
                                          "selected": tuple(u["id"] for u in available)},
                                         available, "ordinary_tasks_exhausted")
                    continue
                self._group = self._task["selected"]
                self._retry_count = 0
                self._event("task_assigned", task_kind=self._task["kind"],
                            target=self._task["target"], selected_units=list(self._group))
            task = self._task
            selected = tuple(uid for uid in task["selected"]
                             if any(u["id"] == uid for u in available))
            if not selected:
                self._finish_task("group_exhausted")
                continue
            if task["kind"] == "combat" and task.get("attempts", 0) >= 4:
                self._enter_recovery(task, available, "task_attempt_limit")
                continue
            if task["kind"] == "combat" and self._call_searches >= 2 and self._searches < self._search_limit:
                # No command has been emitted yet (e.g. stale predictions).
                # Recover the last ATTEMPTED roster. The untried task remains
                # eligible on the next callback, with a fresh per-call budget.
                self._enter_recovery(self._call_combat_task, available, "call_search_limit")
                continue
            if task["kind"] == "combat" and self._searches >= self._search_limit:
                self._enter_recovery(task, available, "search_budget_exhausted")
                continue
            if task["kind"] != "combat" and self._support_calls >= self._support_limit:
                self._finish_task("support_limit")
                self._phase = 4
                continue
            target = task["target"]
            identity = digest({"observation": observation, "target": target,
                               "selected": selected, "kind": task["kind"]})
            if identity in self._visited:
                if task["kind"] == "combat":
                    self._enter_recovery(task, available, "no_progress")
                else:
                    self._finish_task("no_progress", retry_approach=bool(enemies or self._last_seen))
                continue
            self._visited.add(identity)
            if task["kind"] == "combat":
                task["selected"] = selected
                result = self._choose_combat_plan(task, observation, own, enemies, planning)
                task = self._task
                target, selected = task["target"], task["selected"]
            else:
                self._support_calls += 1
                mode = {"heal_before": SupportMode.HEAL_BEFORE,
                        "heal_after": SupportMode.HEAL_AFTER,
                        "approach": SupportMode.APPROACH}[task["kind"]]
                result = planning.support(target, selected, contacts=self.memory, mode=mode)
            if task["kind"] != "combat":
                self._event("support", target=target, task_kind=task["kind"], selected_units=list(selected),
                            reason=result.reason.value, stats=asdict(result.stats),
                            assignments=len(result.assignments), replanning=self._cancelled_plan,
                            diagnostics=asdict(result.diagnostics))
            self._cancelled_plan = False
            if result.reason in (SearchReason.INVALID_REQUEST, SearchReason.NOT_IMPLEMENTED):
                raise RuntimeError(f"CP search failed: {result.reason}")
            if result.reason == SearchReason.TURN_CHANGED:
                return self._end("turn_changed")
            if not self._has_commands(result):
                if task["kind"] == "combat":
                    reason = "no_executable_plan" if result.assignments else result.reason.value
                    self._enter_recovery(task, available, reason)
                else:
                    self._finish_task(result.reason.value, retry_approach=bool(enemies or self._last_seen))
                continue
            self._replan = False
            self._queue.extend(result.assignments)

    def on_action_result(self, action, result, observation):
        """Validate immediately, including the action that ends a battle."""
        self._observe(observation)
        self._search_cache.clear()  # Even a failed command expires the detached planning snapshot.
        if self._pending is None:
            self._event("action_result", action=asdict(action), reason="turn_ended")
            self._last_observation = _observation_snapshot(observation)
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
        if result.ok and a.kind == K.MOVE and self._task is not None and self._task["kind"] == "recovery":
            self._recovery_moved.add(a.unit)
        if result.ok and result.damage > 0 and self._task is not None:
            victim = next((u for u in old.values() if u["side"] != self.side and u["tile"] == action.target), None)
            if victim is not None:
                self._combat_targets[action.unit] = (victim["id"], 0, 0,
                    own_turn(observation["turn"], observation["active_side"], self.side))
        living_enemies = {u["id"] for u in now.values() if u["side"] != self.side}
        self._combat_targets = {uid: value for uid, value in self._combat_targets.items()
                                if value[0] in living_enemies or value[0] in self._last_seen}
        if result.ok and action.kind not in ("skip", "fortify", "end_turn"):
            # Attempts refer to a physical position, never the rest of the turn.
            self._combat_attempts.clear()
            self._approach_attempts.clear()
            if self._task is not None:
                self._phase_seen[3].difference_update(self._task["selected"])
            if reason is None and self._task is not None and self._task["kind"] == "approach" and living_enemies:
                reason = "combat_reconsideration"
        if reason is not None:
            if not result.ok:
                self._finished.discard(a.unit)
            self._cancel(reason)
            if reason == "new_contact" and self._task is not None and self._task["kind"] == "combat":
                # CP FindAndExecuteBestUnitAssignments retries the same target
                # and input roster. Availability is filtered on the next call.
                self._task.pop("alternative", None)
                self._event("task_restarted", target=self._task["target"],
                            selected_units=list(self._task["selected"]),
                            attempts=self._task.get("attempts", 0))
            elif reason in ("new_contact", "combat_reconsideration"):
                self._finish_task(reason, retry_approach=True)
                self._phase = min(self._phase, 1)
                self._combat_attempts.clear()
        self._last_observation = _observation_snapshot(observation)

    def __del__(self):
        self.close()
