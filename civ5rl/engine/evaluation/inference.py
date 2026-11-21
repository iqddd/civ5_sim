"""Batched evaluation over shared public inputs and independent recurrent states."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch

from civ5rl.core.model import Policy, batch_inputs
from civ5rl.core.observation import ordered_actions
from civ5rl.engine.evaluation.state import RecurrentStates


@dataclass
class AgentMember:
    hidden: torch.Tensor | None = None


@dataclass
class AgentSession:
    mode: str
    temperature: float
    collect_selection_metrics: bool
    members: list[AgentMember]


class BatchedInference:
    """Batch public inputs while retaining independent GRU state per policy."""

    def __init__(self, policy, device, *, hidden_capacity=32):
        self.policies: list[Policy] = [policy]
        self.device = device
        self.hidden_capacity = hidden_capacity
        self.hidden_banks = {}
        self.batches = self.items = 0
        self.forward_seconds = self.encoding_seconds = 0.0

    @property
    def policy(self):
        return self.policies[0]

    def gather_hidden(self, policy, agents, member=0):
        if not policy.is_recurrent:
            return None
        if torch.device(self.device).type == "cuda":
            bank = self.hidden_banks.get(member)
            if bank is None:
                bank = self.hidden_banks[member] = RecurrentStates(
                    policy.hidden_size, self.device, self.hidden_capacity
                )
            return bank.gather(agents)
        return torch.cat(
            [
                a.hidden
                if a.hidden is not None
                else torch.zeros(1, 1, policy.hidden_size, device=self.device)
                for a in agents
            ],
            dim=1,
        )

    def store_hidden(self, agents, hidden, member=0):
        if hidden is None:
            return
        bank = self.hidden_banks.get(member)
        if bank is not None:
            bank.scatter(agents, hidden)
        else:
            for agent, row in zip(agents, hidden.split(1, dim=1)):
                agent.hidden = row.clone()

    def release_agent(self, agent):
        if agent is not None:
            for member in getattr(agent, "members", [agent]):
                RecurrentStates.release(member)

    def combine_encoded_logits(self, logits_all, requests):
        logits = logits_all[0]
        choices = logits.argmax(-1).cpu().tolist()
        detailed = any(
            command != "observe" and (agent.mode != "argmax" or agent.collect_selection_metrics)
            for agent, command, _ in requests
        )
        host = logits.cpu().numpy() if detailed else None
        return [
            None
            if command == "observe"
            else {"index": choices[i]}
            if agent.mode == "argmax" and not agent.collect_selection_metrics
            else {"logits": host[i, : len(encoded.sources)].copy()}
            for i, (agent, command, encoded) in enumerate(requests)
        ]

    def _forward_encoded(self, requests):
        start = perf_counter()
        policies = self.policies
        if any(p.variant != policies[0].variant for p in policies):
            raise ValueError("Shared public encoding requires identical policy variants")
        encoded = [r[2] for r in requests]
        batch = batch_inputs(
            encoded,
            self.device,
            non_blocking=True,
            compact_units=self.policy.attention_backend == "compact_fp32",
        )
        self.encoding_seconds += perf_counter() - start
        start = perf_counter()
        logits_all = []
        for member, policy in enumerate(policies):
            agents = [r[0].members[member] for r in requests]
            hidden = self.gather_hidden(policy, agents, member)
            logits, _, last = policy(batch, hidden, compute_value=False)
            logits_all.append(logits)
            self.store_hidden(agents, last, member)
        self.batches += 1
        self.items += len(requests)
        return logits_all, start

    @torch.inference_mode()
    def infer_encoded(self, requests):
        """One public encoding shared by policies; each policy keeps its own GRU."""
        logits_all, start = self._forward_encoded(requests)
        results = (
            self.combine_encoded_logits(logits_all, requests)
            if any(r[1] != "observe" for r in requests)
            else [None] * len(requests)
        )
        self.forward_seconds += perf_counter() - start
        return results

    @torch.inference_mode()
    def forward_encoded_argmax(self, requests) -> torch.Tensor:
        """Enqueue public-encoded forward + argmax on device without host synchronization."""
        logits, start = self._forward_encoded(requests)
        choices = logits[0].argmax(-1)
        self.forward_seconds += perf_counter() - start
        return choices

    @torch.inference_mode()
    def infer(self, requests):
        # requests are (agent, command, observation, legal_actions, forced_action).
        started = perf_counter()
        actions = [ordered_actions(r[3]) for r in requests]
        encoded = [r[0].encoder.encode(r[2], a) for r, a in zip(requests, actions)]
        batch = batch_inputs(
            encoded,
            self.device,
            compact_units=self.policy.attention_backend == "compact_fp32",
        )
        agents = [r[0] for r in requests]
        hidden = self.gather_hidden(self.policy, agents)
        self.encoding_seconds += perf_counter() - started
        started = perf_counter()
        logits, _, next_hidden = self.policy(batch, hidden, compute_value=False)
        fast = [
            r[1] != "observe" and r[0].mode == "argmax" and not r[0].collect_selection_metrics
            for r in requests
        ]
        need_logits = any(r[1] != "observe" and not greedy for r, greedy in zip(requests, fast))
        choices = logits.argmax(-1).cpu().tolist() if any(fast) else None
        host_logits = logits.cpu() if need_logits else None
        self.forward_seconds += perf_counter() - started
        self.store_hidden(agents, next_hidden)
        outputs = []
        for index, (agent, command, observation, _, forced) in enumerate(requests):
            if command == "observe":
                if forced not in actions[index]:
                    raise ValueError("Forced action is not legal")
                agent.encoder.remember_action(forced, observation)
                outputs.append(None)
            else:
                choice = (
                    agent.action_from_index(choices[index], observation, actions[index])
                    if fast[index]
                    else agent.action_from_logits(
                        host_logits[index : index + 1, : len(actions[index])],
                        observation,
                        actions[index],
                    )
                )
                outputs.append((choice, agent.drain_diagnostics()))
        self.batches += 1
        self.items += len(requests)
        return outputs

    def statistics(self):
        return {
            "inference_batches": self.batches,
            "inference_items": self.items,
            "mean_batch": self.items / max(1, self.batches),
            "forward_seconds": self.forward_seconds,
            "encoding_seconds": self.encoding_seconds,
        }
