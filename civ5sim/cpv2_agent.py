"""CP coordinator variant for concentrated engagements and defensive deployment."""

from .combat_ai import Aggression, SearchConfig
from .cp_agent import CPAgent


class CPv2Agent(CPAgent):
    def __init__(self):
        super().__init__(SearchConfig(aggression=Aggression.HIGH, risk_budget_base=2))

    def reset(self, context):
        super().reset(context)
        self._opening = True
        self._opening_turn = None
        self._opening_positions = None
        self._idle_opening = False
        self._recovered = False
        self._wounded_in_fog = set()

    def _observe(self, observation):
        super()._observe(observation)
        if self._recovered:
            return
        units = observation["units"]
        if any(u["side"] != self.side for u in units):
            self._opening = False
            self._wounded_in_fog.clear()
            return
        if (
            self._opening
            and observation["active_side"] == self.side
            and observation["turn"] != self._opening_turn
        ):
            positions = tuple((u["id"], u["tile"]) for u in units)
            self._idle_opening = positions == self._opening_positions
            self._opening_positions = positions
            self._opening_turn = observation["turn"]
        wounded = {u["id"] for u in units if u["hp"] < 100}
        if not wounded:
            # A surviving unit healed. Losing the last wounded unit is not recovery.
            self._recovered = any(u["id"] in self._wounded_in_fog for u in units)
        self._wounded_in_fog = wounded

    def _new_turn(self):
        super()._new_turn()
        # Find attacks before assigning any injured attackers to healing.
        self._phase = 1

    def _combat_search(self, task, observation, own, enemies, planning, *, probe=False):
        task["aggression"] = Aggression.HIGH
        return super()._combat_search(
            task, observation, own, enemies, planning, probe=probe
        )

    def _group_rank(self, task, own, actions):
        rank = super()._group_rank(task, own, actions)
        ready = task["ranking"]["immediate_attackers"]
        return (min(ready, 3), *rank)

    def _rally_target(self, army):
        """Choose a common location for deployment or recovery."""

        def score(unit):
            tile = unit["tile"]
            terrain = self.map["tiles"][tile]
            return (
                sum(min(5, self._distance(tile, u["tile"])) for u in army)
                - 2 * (terrain["elevation"] == "hills"),
                tile,
            )

        return min(army, key=score)["tile"]

    def _opening_patrol(self, army, available, observation, actions):
        if not self._opening or not self._idle_opening:
            return None
        self._idle_opening = False
        group = {u["id"]: u for u in available}
        objective = self._explore_target(available, observation)
        if objective is None:
            return None
        routes = self._distances(objective)
        candidates = []
        for action in actions:
            unit = group.get(action.unit)
            if action.kind != "move" or unit is None:
                continue
            tile = action.target
            if tile not in self._neighbors[unit["tile"]]:
                continue
            if routes.get(tile, 1000000) >= routes.get(unit["tile"], 1000000):
                continue
            cover = sum(
                ally["id"] != action.unit and self._distance(tile, ally["tile"]) <= 2
                for ally in army
            )
            if len(army) > 1 and not cover:
                continue
            candidates.append((routes[tile], -cover, -unit["hp"], action.unit, tile))
        if candidates:
            _, _, _, uid, target = min(candidates)
            return {"kind": "approach", "target": target, "selected": (uid,)}
        return None

    def _next_task(self, available, enemies, observation, eligibility=None, actions=()):
        army = [u for u in observation["units"] if u["side"] == self.side]
        # Use the existing assault search to clear isolated remnants late in battle.
        if self._support_calls >= self._support_limit:
            return None
        ready_to_scout = (
            self._recovered and not enemies and all(u["hp"] == 100 for u in army)
        )
        if ready_to_scout or (observation["turn"] >= 48 and len(enemies) <= 1):
            return super()._next_task(
                available, enemies, observation, eligibility, actions
            )
        if enemies:
            target = min(
                enemies,
                key=lambda e: (
                    e["hp"]
                    + 9 * min(self._distance(e["tile"], u["tile"]) for u in army),
                    e["id"],
                ),
            )["tile"]
        else:
            target = self._rally_target(army)
            patrol = self._opening_patrol(army, available, observation, actions)
            if patrol is not None:
                return patrol
        return {
            "kind": "formation",
            "target": target,
            "selected": self._select(available, target),
        }
