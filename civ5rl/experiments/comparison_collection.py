"""Public histories and paired counterfactual matches with exact CP prefix replay."""

from __future__ import annotations

import random
from time import perf_counter

import torch

from civ5rl.core.agent import RLAgent
from civ5rl.core.model import batch_inputs
from civ5rl.core.observation import ordered_actions
from civ5rl.engine.evaluation.executor import BatchedInference, EvaluationExecutor
from civ5rl.experiments.comparison_match import _verify as _verify  # noqa: PLC0414 -- historical re-export
from civ5rl.experiments.comparison_match import match_task as match_task  # noqa: PLC0414 -- historical re-export
from civ5rl.provenance.common import seed_for


def select_forks(history, episode_id, namespace, maximum=4):
    rounds = {}
    for i, row in enumerate(history):
        if row["chosen"] != row["alternative"]:
            rounds.setdefault(row["observation"]["turn"], []).append(i)
    rng = random.Random(seed_for(17, episode_id, "selection", namespace=namespace))
    chosen = rng.sample(sorted(rounds), min(maximum, len(rounds)))
    return sorted(rng.choice(rounds[r]) for r in chosen)


def preference(winners, side):
    # Two alternatives, eight paired futures. Rewards remain exactly 1/0/0.
    wins = [[int(w == side) for w in row] for row in winners]
    deltas = [sum(wins[1][i : i + 4]) - sum(wins[0][i : i + 4]) for i in (0, 4)]
    d1, d2 = deltas
    reason = (
        "both_zero"
        if d1 == d2 == 0
        else "one_zero"
        if not d1 or not d2
        else "opposite_signs"
        if d1 * d2 < 0
        else "accepted"
    )
    return {
        "deltas": deltas,
        "accepted": reason == "accepted",
        "reason": reason,
        "better": 1 if d1 > 0 else 0,
        "weight": abs(d1 + d2) / 8 if reason == "accepted" else 0.0,
        "wdl": [
            {k: sum(("D" if w is None else "W" if w == side else "L") == k for w in row) for k in "WDL"}
            for row in winners
        ],
    }


class ProposalInference(BatchedInference):
    def __init__(self, policies, device, *, hidden_capacity=32):
        super().__init__(policies[0], device, hidden_capacity=hidden_capacity)
        self.policies = policies

    def combine_encoded_logits(self, logits_all, requests):
        choices = torch.stack([z.argmax(-1) for z in logits_all]).cpu().tolist()
        return [
            None
            if command == "observe"
            else {
                "index": choices[0][i],
                "events": [{"kind": "proposal", "alternative": choices[1][i]}],
            }
            for i, (_, command, _) in enumerate(requests)
        ]

    @torch.inference_mode()
    def infer(self, requests):
        start = perf_counter()
        legal = [ordered_actions(r[3]) for r in requests]
        choices = []
        for member, policy in enumerate(self.policies):
            agents = [r[0].members[member] for r in requests]
            encoded = [a.encoder.encode(r[2], acts) for a, r, acts in zip(agents, requests, legal)]
            hidden = self.gather_hidden(policy, agents, member)
            logits, _, hidden = policy(
                batch_inputs(
                    encoded,
                    self.device,
                    compact_units=policy.attention_backend == "compact_fp32",
                ),
                hidden,
            )
            choices.append(logits.argmax(-1))
            self.store_hidden(agents, hidden, member)
        choices = torch.stack(choices).cpu().tolist()
        outputs = []
        for i, (agent, command, obs, _, forced) in enumerate(requests):
            action = forced if command == "observe" else legal[i][choices[0][i]]
            if action not in legal[i]:
                raise ValueError("Illegal recorded command")
            for a in agent.members:
                a.encoder.remember_action(action, obs)
            outputs.append(
                None
                if command == "observe"
                else (action, [{"kind": "proposal", "alternative": choices[1][i]}])
            )
        self.forward_seconds += perf_counter() - start
        self.batches += 1
        self.items += len(requests)
        return outputs


class ProposalExecutor(EvaluationExecutor):
    def __init__(self, checkpoints, runtime):
        super().__init__(str(checkpoints[0]), runtime)
        from civ5rl.engine.evaluation.executor import prepare_evaluation_policy

        second = prepare_evaluation_policy(
            RLAgent.from_checkpoint(checkpoints[1], "cpu", "argmax").policy,
            self.runtime.device,
        )
        self.inference = ProposalInference(
            [self.inference.policy, second],
            self.runtime.device,
            hidden_capacity=self.runtime.workers,
        )
