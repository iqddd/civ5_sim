"""Reachable dev histories; model-free construction and exact raw-input checks."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from civ5rl.core.agent import RLAgent
from civ5rl.core.context import context_for
from civ5rl.core.model import batch_inputs
from civ5rl.core.observation import ordered_actions
from civ5rl.protocols.evaluation import require_split
from civ5rl.provenance.common import seed_for, write_json
from civ5sim.agents import load_agent
from civ5sim.engine import Action, Battle
from civ5sim.match import MatchConfig, run_match

FAMILIES = (
    "lost_contact_fork",
    "hidden_horse_archer",
    "cover_flank",
    "scout_before_army",
)

MEMORY_SCOPE = "history_sensitivity_only_not_gru_isolation"


def flat_map(size=16, parity=0):
    return {
        "schema": 1,
        "width": size,
        "height": size,
        "row_parity": parity,
        "wrap_x": False,
        "tiles": [
            {
                "terrain": "grass",
                "elevation": "flat",
                "feature": None,
                "rivers": [-1, -1, -1],
            }
            for _ in range(size * size)
        ],
    }


def _history(geometry, history_index, side, branch, shared_last_contact=False):
    # Distinct initial scout positions reveal the flank. Every command is executed
    # through the kernel; no observation, visibility mask or state is fabricated.
    parity, row = geometry % 2, 5 + geometry // 2
    m = flat_map(parity=parity)
    if geometry // 2 == 2:
        for x in (6, 7, 9, 10):
            m["tiles"][(row + 3) * 16 + x]["feature"] = "forest"
    left = branch == 0
    tile = lambda x, y: y * 16 + x
    scout_x = 3 if left else 12
    horse_x = 1 if left else 14
    units = [
        {"type": "horseman", "side": side, "tile": tile(scout_x, row)},
        {"type": "archer", "side": side, "tile": tile(8, row + 1)},
        {"type": "warrior", "side": side, "tile": tile(8, row + 2)},
        {"type": "horseman", "side": 1 - side, "tile": tile(horse_x, row)},
        {"type": "archer", "side": 1 - side, "tile": tile(horse_x, row + 2)},
        {"type": "warrior", "side": 1 - side, "tile": tile(horse_x, row + 3)},
    ]
    # Historical name only: these branches do NOT produce equal current Encoded B.
    # They are history-sensitivity cases, never a test isolating GRU over B.
    if shared_last_contact:
        horse_x = 2
        units[0]["tile"] = tile(4, row - 2 if left else row + 2)
        units[3]["tile"] = tile(2, row - 2 if left else row + 2)
        units[4]["tile"] = tile(0, 0)
        units[5]["tile"] = tile(0, 1)
    scenario = {
        "schema": 1,
        "kind": "civ5sim.scenario",
        "map": m,
        "units": units,
        "fog": True,
        "random_damage": True,
        "active_side": 0,
        "seed": seed_for(
            20261107, geometry * 4 + history_index, "battle", namespace="memory_dev"
        ),
    }
    armies = [[u["type"] for u in units if u["side"] == s] for s in (0, 1)]
    prefix, observations = [], []
    prefix_sides = []
    with Battle(scenario) as battle:

        def apply(action):
            if action not in battle.available_actions():
                raise ValueError(
                    f"Unreachable memory prefix: {action}, geometry={geometry}"
                )
            active = battle.observe()["side"]
            prefix_sides.append(active)
            if active == side:
                observations.append(
                    {
                        "observation": battle.observe(side),
                        "actions": ordered_actions(battle.available_actions()),
                        "action": action,
                    }
                )
            result = battle.step(action)
            if not result.ok:
                raise ValueError("Memory prefix rejected")
            prefix.append(asdict(action))

        if side == 1:
            apply(Action("end_turn"))
        if shared_last_contact:
            # Scout reaches a common observation point; enemy follows to its common
            # last-seen tile on the intervening opponent turn.
            apply(Action("move", 0, tile(4, row)))
            apply(Action("end_turn"))
            apply(Action("move", 3, tile(2, row)))
            apply(Action("end_turn"))
        else:
            # Two own turns are needed for the farther starting position. First
            # converge at x=7, then normalize own state by waiting a full turn.
            intermediate = 7 if left else 8
            apply(Action("move", 0, tile(intermediate, row)))
            apply(Action("end_turn"))
            apply(Action("end_turn"))
        if battle.observe(side)["units"][0]["tile"] != tile(8, row):
            apply(Action("move", 0, tile(8, row)))
        apply(Action("end_turn"))
        if shared_last_contact:
            apply(Action("move", 3, tile(0, row - 2 if left else row + 2)))
        apply(Action("end_turn"))
        current, actions = (
            battle.observe(side),
            ordered_actions(battle.available_actions()),
        )
        # The common scout must have lost the enemy before free choice.
        if any(u["side"] != side for u in current["units"]):
            raise ValueError("Memory decision still sees an enemy")
        return {
            "geometry": geometry,
            "history_index": history_index,
            "family": FAMILIES[(geometry // 2) % 4],
            "branch": branch,
            "side": side,
            "shared_last_contact": shared_last_contact,
            "scenario": scenario,
            "armies": armies,
            "prefix": prefix,
            "history": observations,
            "prefix_sides": prefix_sides,
            "current": current,
            "actions": actions,
        }


def build_dev():
    cases = []
    for geometry in range(8):
        for history_index in range(4):
            for side in (0, 1):
                pair = [
                    _history(
                        geometry,
                        history_index,
                        side,
                        branch,
                        shared_last_contact=history_index >= 2,
                    )
                    for branch in (0, 1)
                ]
                if (
                    pair[0]["current"] != pair[1]["current"]
                    or pair[0]["actions"] != pair[1]["actions"]
                ):
                    raise ValueError("Memory pair differs in current raw inputs")
                cases.extend(pair)
    return cases


class PrefixAgent(RLAgent):
    def __init__(self, policy, device, case):
        super().__init__(policy, device)
        self.case = case

    def reset(self, context):
        super().reset(context)
        for item in self.case["history"]:
            self.observe_step(item["observation"], item["actions"], item["action"])


def evaluate_dev(policy, cases, device, directory, config):
    require_split(config, "memory_dev")
    directory = Path(directory)
    rows, outcomes = [], []
    for case_index, case in enumerate(cases):
        agent = PrefixAgent(policy, device, case)
        ctx = context_for(
            case["scenario"],
            case["armies"],
            case["side"],
            60,
            seed_for(
                20261107,
                case["geometry"] * 4 + case["history_index"],
                "learner_sampling",
                case["side"],
                "memory_dev",
            ),
        )
        agent.reset(ctx)
        encoded = agent.encoder.encode(case["current"], case["actions"])
        with torch.no_grad():
            logits, value, _ = policy(batch_inputs([encoded], device), agent.hidden)
        probabilities = logits.softmax(-1)[0].cpu().numpy()
        rows.append(
            {
                "geometry": case["geometry"],
                "history_index": case["history_index"],
                "side": case["side"],
                "branch": case["branch"],
                "family": case["family"],
                "probabilities": probabilities.tolist(),
                "value": value.item(),
                "argmax": asdict(case["actions"][int(probabilities.argmax())]),
            }
        )
        agents = []
        for side in (0, 1):
            forced = [
                Action(**a)
                for a, s in zip(case["prefix"], case["prefix_sides"])
                if s == side
            ]
            agents.append(
                ForcedPrefix(policy, device, forced, learner=side == case["side"])
            )
        match_config = MatchConfig(
            case["scenario"]["map"], case["armies"], 60, max_calls_per_turn=2000
        )
        result = run_match(
            match_config,
            agents,
            scenario=case["scenario"],
            agent_seeds=[
                seed_for(
                    20261107,
                    case["geometry"] * 4 + case["history_index"],
                    "learner_sampling",
                    s,
                    "memory_dev",
                )
                for s in (0, 1)
            ],
        )
        if result.result["status"] != "completed":
            raise RuntimeError(result.result["error"])
        reward = (
            0
            if result.result["winner"] is None
            else (1 if result.result["winner"] == case["side"] else -1)
        )
        outcomes.append(
            {
                "case": case_index,
                "geometry": case["geometry"],
                "side": case["side"],
                "branch": case["branch"],
                "reward": reward,
            }
        )
        if case_index < 8:
            result.save(directory / "replays" / f"case-{case_index:03}")
    distances = [
        float(
            np.abs(np.array(a["probabilities"]) - np.array(b["probabilities"])).sum()
            / 2
        )
        for a, b in zip(rows[::2], rows[1::2])
    ]
    summary = {
        "scope": MEMORY_SCOPE,
        "gru_isolation_valid": False,
        "cases": len(cases),
        "pairs": len(distances),
        "mean_policy_total_variation": float(np.mean(distances)),
        "different_argmax_pairs": sum(
            a["argmax"] != b["argmax"] for a, b in zip(rows[::2], rows[1::2])
        ),
        "interpretation": "Policy sensitivity and dev continuation outcomes; causal memory advantage requires independent A/B/C comparisons.",
    }
    summary.update(
        score=float(np.mean([(o["reward"] + 1) / 2 for o in outcomes])),
        wins=sum(o["reward"] == 1 for o in outcomes),
        draws=sum(o["reward"] == 0 for o in outcomes),
        losses=sum(o["reward"] == -1 for o in outcomes),
    )
    write_json(directory / "decisions.json", rows)
    write_json(directory / "outcomes.json", outcomes)
    write_json(directory / "summary.json", summary)
    return summary


class ForcedPrefix:
    """Execute a reachable prefix; no loss, sampling or rewards on imposed actions."""

    def __init__(self, policy, device, actions, learner):
        self.policy, self.device, self.actions, self.learner = (
            policy,
            device,
            actions,
            learner,
        )

    def reset(self, context):
        self.agent = (
            RLAgent(self.policy, self.device) if self.learner else load_agent("greedy")
        )
        self.agent.reset(context)
        self.index = 0

    def act(self, observation, actions):
        if self.index >= len(self.actions):
            return self.agent.act(observation, actions)
        chosen = self.actions[self.index]
        if chosen not in actions:
            raise ValueError("Forced action is not legal")
        if self.learner:
            self.agent.observe_step(observation, actions, chosen)
        self.index += 1
        return chosen
