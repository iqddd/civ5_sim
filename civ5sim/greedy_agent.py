"""Observation-only baseline: attack, move to attack, then approach/explore."""

from collections import deque
from copy import deepcopy

from civ5map.maps import adjacent, passable

from .engine import Action, Battle


class GreedyAgent:
    def reset(self, context):
        self.side = context["side"]
        self.map = deepcopy(context["map"])
        self.seen = [-1] * len(self.map["tiles"])
        self.neighbors = {}
        for tile, value in enumerate(self.map["tiles"]):
            if not passable(value):
                continue
            ns = []
            for direction in range(6):
                xy = adjacent(self.map, tile % self.map["width"],
                              tile // self.map["width"], direction)
                if xy is not None:
                    n = xy[1] * self.map["width"] + xy[0]
                    if passable(self.map["tiles"][n]):
                        ns.append(n)
            self.neighbors[tile] = ns
        self.routes = {}
        self.turn = None
        self.moved = set()

    def _attack_approach(self, observation, actions, own, enemies):
        """Check one move plus one attack using only publicly observed units.

        This is a legality probe, not a damage or trade evaluation. Unknown
        enemies are absent, so new contact can still interrupt the real move;
        the next call rechecks the new observation rather than forcing a plan.
        """
        moves = [a for a in actions if a.kind == "move" and
                 not own[a.unit]["attacked"] and own[a.unit]["moves"] > 60]
        if not moves or not enemies:
            return None
        # Battle assigns dense IDs. Keep a separate mapping to the public IDs;
        # missing enemy slots convey no information about hidden/dead units.
        observed = sorted(observation["units"], key=lambda u: u["id"])
        indices = {u["id"]: i for i, u in enumerate(observed)}
        units = [{k: v for k, v in u.items() if k != "id"} for u in observed]
        scenario = {"schema": 1, "kind": "civ5sim.scenario", "map": self.map,
                    "units": units, "active_side": self.side, "fog": False,
                    "random_damage": False, "seed": 0}
        candidates = []
        with Battle(scenario) as root:
            for move in moves:
                local_id = indices[move.unit]
                with root.clone() as branch:
                    result = branch.step(Action("move", local_id, move.target))
                    if not result.ok:
                        continue
                    attacks = [a for a in branch.available_actions() if a.unit == local_id
                               and a.kind in ("melee", "ranged") and a.target in enemies]
                    if not attacks:
                        continue
                    attack = min(attacks, key=lambda a: (
                        enemies[a.target]["hp"], enemies[a.target]["id"], a.kind))
                    remaining = branch.observe(self.side)["units"]
                    remaining = next(u["moves"] for u in remaining if u["id"] == local_id)
                    candidates.append(((enemies[attack.target]["hp"], enemies[attack.target]["id"],
                                        move.unit, attack.kind, -remaining, move.target), move))
        return min(candidates, key=lambda item: item[0])[1] if candidates else None

    def distances(self, target):
        if target not in self.routes:
            distances = {target: 0}
            queue = deque([target])
            while queue:
                tile = queue.popleft()
                for n in self.neighbors.get(tile, ()):
                    if n not in distances:
                        distances[n] = distances[tile] + 1
                        queue.append(n)
            self.routes[target] = distances
        return self.routes[target]

    def act(self, observation, actions):
        if observation["turn"] != self.turn:
            self.turn = observation["turn"]
            self.moved.clear()
        for tile, visible in enumerate(observation["visibility"]):
            if visible:
                self.seen[tile] = self.turn
        own = {u["id"]: u for u in observation["units"] if u["side"] == self.side}
        enemies = {u["tile"]: u for u in observation["units"] if u["side"] != self.side}
        attacks = [a for a in actions if a.kind in ("melee", "ranged")]
        if attacks:
            return min(attacks, key=lambda a: (enemies[a.target]["hp"],
                       enemies[a.target]["id"], a.unit, a.kind))
        approach = self._attack_approach(observation, actions, own, enemies)
        if approach is not None:
            self.moved.add(approach.unit)
            return approach
        candidates = []
        for uid, unit in sorted(own.items()):
            if uid in self.moved:
                continue
            reachable = self.distances(unit["tile"])
            if enemies:
                target = min(enemies, key=lambda t: (reachable.get(t, 10**9), t))
            else:
                target = min(reachable, key=lambda t: (self.seen[t], reachable[t], t))
            distances = self.distances(target)
            start = distances.get(unit["tile"], 10**9)
            for action in actions:
                if action.kind == "move" and action.unit == uid:
                    distance = distances.get(action.target, 10**9)
                    if distance < start:
                        candidates.append((distance, uid, action.target, action))
        if candidates:
            action = min(candidates, key=lambda c: c[:3])[3]
            self.moved.add(action.unit)
            return action
        return Action("end_turn")
