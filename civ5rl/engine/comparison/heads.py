"""Frozen-history FP32 inputs for action-head-only preference fitting.

Cache packets keep cell embeddings and one hidden vector per position, rather
than expanding either across legal actions. Packed action tensors retain every
legal command. CPU mmap packets bound RAM; only a microbatch is copied to CUDA.
Packets are derived data: their source/runtime identity and SHA-256 are checked
before reuse. No output of the trainable preview projection is cached.
"""

from __future__ import annotations

from collections import defaultdict
from time import perf_counter

import torch
from torch.nn import functional as F

from civ5rl.core.distributions import masked_kl_terms
from civ5rl.core.schema import ACTION_DIM, ATTACK_PREVIEW_DIM
from civ5rl.engine.comparison import PREFERENCE_TEMPERATURE
from civ5rl.engine.comparison.frozen_features import FEATURE_CACHE_VERSION as FEATURE_CACHE_VERSION
from civ5rl.engine.comparison.frozen_features import FrozenFeatureCache, feature_positions
from civ5rl.engine.comparison.policy import configure_heads as configure_heads
from civ5rl.engine.comparison.policy import step_comparison_policy
from civ5rl.engine.comparison.reference import target_logits

MODE = "frozen_heads_step_kl"


def learning_metrics(pairs, original, baseline, current, anchors):
    """Summarize fixed-position post-update logits; ties are not preferences.

    Each entry is (argmax, pair_margin, weighted_softplus). Pair margins/losses are
    stored per pair, not per position, so repeated targets retain multiplicity.
    Anchor changes describe spread, not correctness on those positions.
    """
    groups = {}
    for label, indices in (
        ("all", list(range(len(pairs)))),
        (
            "alternative",
            [i for i, p in enumerate(pairs) if p["better"] != original[(p["episode"], p["index"])]],
        ),
        (
            "parent_confirmation",
            [i for i, p in enumerate(pairs) if p["better"] == original[(p["episode"], p["index"])]],
        ),
    ):
        rows = [(baseline["pairs"][i], current["pairs"][i]) for i in indices]
        n = len(rows)
        groups[label] = {
            "count": n,
            "pair_loss": sum(b[2] for _, b in rows) / n if n else 0.0,
            "preferred": sum(b[1] > 0 for _, b in rows),
            "ties": sum(b[1] == 0 for _, b in rows),
            "better_argmax": sum(b[0] == pairs[i]["better"] for i, (_, b) in zip(indices, rows)),
            "preference_gained": sum(a[1] <= 0 < b[1] for a, b in rows),
            "preference_lost": sum(a[1] > 0 >= b[1] for a, b in rows),
            "argmax_gained": sum(
                a[0] != pairs[i]["better"] and b[0] == pairs[i]["better"] for i, (a, b) in zip(indices, rows)
            ),
            "argmax_lost": sum(
                a[0] == pairs[i]["better"] and b[0] != pairs[i]["better"] for i, (a, b) in zip(indices, rows)
            ),
        }
    pair_keys = {(p["episode"], p["index"]) for p in pairs}
    changes = {}
    for label, keys in (
        ("all", list(map(tuple, anchors))),
        ("outside_pairs", [tuple(k) for k in anchors if tuple(k) not in pair_keys]),
    ):
        count = sum(baseline["choices"][k] != current["choices"][k] for k in keys)
        changes[label] = {
            "count": len(keys),
            "changed": count,
            "fraction": count / len(keys) if keys else 0.0,
        }
    return {"pairs": groups, "anchors": changes}


class HeadFeatures:
    def __init__(self, policy, histories, dataset):
        self.policy, self.histories, self.dataset = policy, histories, dataset
        self.device = next(policy.parameters()).device
        self.original = {}
        self.keys = feature_positions(dataset)
        self.statistics = {
            "positions": len(self.keys),
            "cache_hits": 0,
            "cache_builds": 0,
            "bytes": 0,
            "precision": "deterministic_fp32_no_tf32",
        }
        self.cache = FrozenFeatureCache(histories, self.device, policy.attention_backend, self.statistics)
        self.directory = self.cache.directory
        self.records = self.cache.records
        self.locations = self.cache.locations
        self.baseline = None

    @torch.no_grad()
    def prepare(self, checkpoint, budget=None):
        started = perf_counter()
        self.cache.prepare(checkpoint, self.keys, lambda: self.policy, budget=budget)
        for key, (episode, position) in self.locations.items():
            self.original[key] = self.records[episode]["chosen"][position]
        self.statistics["preparation_seconds"] = perf_counter() - started
        self.references = {}
        for keys, logits in self.logits(self.keys):
            logp = (logits / PREFERENCE_TEMPERATURE).log_softmax(-1).cpu()
            for row, key in enumerate(keys):
                count = self.count(key)
                self.references[key] = logp[row, :count].clone()
        _, self.baseline = self.measure(raw=True)
        self.parity = self.check_parity()
        print(
            f"Frozen FP32 features: {len(self.keys)} positions, {self.statistics['bytes'] / 1024**3:.2f} GiB",
            flush=True,
        )

    def count(self, key):
        episode, row = self.locations[tuple(key)]
        offsets = self.records[episode]["offsets"]
        return offsets[row + 1] - offsets[row]

    def groups(self, keys):
        groups = defaultdict(list)
        for key in dict.fromkeys(map(tuple, keys)):
            episode, row = self.locations[key]
            groups[self.records[episode]["x"].shape[1]].append(key)
        for keys in groups.values():
            batch = []
            width = 0
            for key in keys:
                count = self.count(key)
                if batch and (len(batch) >= 32 or max(width, count) * (len(batch) + 1) > 8192):
                    yield batch
                    batch = []
                    width = 0
                batch.append(key)
                width = max(width, count)
            if batch:
                yield batch

    def inputs(self, keys):
        width = max(self.count(key) for key in keys)
        b = len(keys)
        batch = {
            "sources": torch.full((b, width), -1, dtype=torch.long),
            "targets": torch.full((b, width), -1, dtype=torch.long),
            "action_features": torch.zeros(b, width, ACTION_DIM),
            "attack_preview": torch.zeros(b, width, ATTACK_PREVIEW_DIM),
            "action_mask": torch.zeros(b, width, dtype=torch.bool),
        }
        xs = []
        hs = []
        for i, key in enumerate(keys):
            episode, row = self.locations[key]
            record = self.records[episode]
            start, stop = record["offsets"][row : row + 2]
            count = stop - start
            for name in record["actions"]:
                batch[name][i, :count].copy_(record["actions"][name][start:stop])
            batch["action_mask"][i, :count] = True
            xs.append(record["x"][row])
            hs.append(record["h"][row])
        return (
            {k: v.to(self.device) for k, v in batch.items()},
            torch.stack(xs).to(self.device),
            torch.stack(hs).to(self.device),
        )

    def logits(self, keys):
        for group in self.groups(keys):
            batch, x, h = self.inputs(group)
            logits, _ = self.policy.score_heads(batch, x, h)
            yield group, logits

    def parent_batch(self, keys, width):
        result = torch.zeros(len(keys), width, device=self.device)
        for row, key in enumerate(keys):
            ref = self.references[key]
            result[row, : len(ref)].copy_(ref)
        return result

    @torch.no_grad()
    def check_parity(self):
        """Check independent ordinary FP32 history execution, including forced states."""
        selected = []
        for key in self.keys:
            if not selected or self.count(key) == 1 or len(selected) < 3 and key[0] != selected[-1][0]:
                selected.append(key)
            if len(selected) >= 4:
                break
        maximum = 0.0
        argmax_changes = 0
        for key in selected:
            history, _ = self.histories.get(key[0])
            ordinary = target_logits(self.policy, history, key[1])
            cached = next(self.logits([key]))[1][0, : len(ordinary)]
            maximum = max(maximum, float((cached - ordinary).abs().max()))
            argmax_changes += int(cached.argmax() != ordinary.argmax())
            torch.testing.assert_close(cached, ordinary, atol=1e-4, rtol=1e-4)
        if argmax_changes:
            raise ValueError("Cached FP32 heads changed parity argmax")
        return {
            "positions": len(selected),
            "max_logit_absolute_error": maximum,
            "argmax_changes": argmax_changes,
        }

    @torch.no_grad()
    def measure(self, raw=False):
        total = torch.zeros((), dtype=torch.float64, device=self.device)
        anchors = set(map(tuple, self.dataset["anchors"]))
        pair_lookup = defaultdict(list)
        for i, p in enumerate(self.dataset["pairs"]):
            pair_lookup[(p["episode"], p["index"])].append((i, p))
        current = {"choices": {}, "pairs": [None] * len(self.dataset["pairs"])}
        for keys, logits in self.logits(self.keys):
            parent = self.parent_batch(keys, logits.shape[1])
            logq = (logits / PREFERENCE_TEMPERATURE).log_softmax(-1)
            values = (
                masked_kl_terms(parent.double(), logq.double(), ~torch.isneginf(logits))
                .masked_fill(torch.isneginf(logits), 0.0)
                .sum(-1)
            )
            weights = torch.tensor([int(key in anchors) for key in keys], device=self.device)
            total += (values * weights).sum()
            # One compact readback per batch, not one synchronization per position.
            choices = logits.argmax(-1).cpu().tolist()
            positions = []
            better = []
            worse = []
            indices = []
            pair_weights = []
            for row, key in enumerate(keys):
                current["choices"][key] = choices[row]
                for i, p in pair_lookup[key]:
                    positions.append(row)
                    better.append(p["better"])
                    worse.append(p["worse"])
                    indices.append(i)
                    pair_weights.append(p["weight"])
            if positions:
                rows = torch.tensor(positions, device=self.device)
                margins = (
                    logits[rows, torch.tensor(better, device=self.device)]
                    - logits[rows, torch.tensor(worse, device=self.device)]
                )
                losses = F.softplus(-margins / PREFERENCE_TEMPERATURE) * torch.tensor(
                    pair_weights, device=self.device
                )
                measured = torch.stack((margins, losses), -1).cpu().tolist()
                for i, row, (margin, loss) in zip(indices, positions, measured):
                    current["pairs"][i] = (choices[row], margin, loss)
        kl = float(total / len(anchors))
        if raw:
            return kl, current
        return kl, learning_metrics(
            self.dataset["pairs"],
            self.original,
            self.baseline,
            current,
            self.dataset["anchors"],
        )

    def update(self, optimizer, pairs, anchor_keys):
        optimizer.zero_grad(set_to_none=True)
        pair_lookup = defaultdict(list)
        for p in pairs:
            pair_lookup[(p["episode"], p["index"])].append(p)
        anchor_set = set(map(tuple, anchor_keys))
        keys = list(dict.fromkeys([*pair_lookup, *map(tuple, anchor_keys)]))
        pair_total = torch.zeros((), device=self.device)
        kl_total = torch.zeros_like(pair_total)
        for keys, logits in self.logits(keys):
            pair = logits.new_zeros(())
            kl = logits.new_zeros(())
            # Batched index gathers keep the loss free of per-pair kernel launches.
            positions = []
            better = []
            worse = []
            weights = []
            for row, key in enumerate(keys):
                for p in pair_lookup[key]:
                    positions.append(row)
                    better.append(p["better"])
                    worse.append(p["worse"])
                    weights.append(p["weight"] / len(pairs))
            if positions:
                rows = torch.tensor(positions, device=self.device)
                diff = (
                    logits[rows, torch.tensor(worse, device=self.device)]
                    - logits[rows, torch.tensor(better, device=self.device)]
                )
                pair = (
                    F.softplus(diff / PREFERENCE_TEMPERATURE) * torch.tensor(weights, device=self.device)
                ).sum()
            parent = self.parent_batch(keys, logits.shape[1])
            logq = (logits / PREFERENCE_TEMPERATURE).log_softmax(-1)
            terms = masked_kl_terms(parent, logq, ~torch.isneginf(logits))
            factors = torch.tensor(
                [int(k in anchor_set) / len(anchor_keys) for k in keys],
                device=self.device,
            )
            kl = (terms * factors[:, None]).sum()
            (pair + kl).backward()
            pair_total += pair.detach()
            kl_total += kl.detach()
        return step_comparison_policy(
            self.policy, optimizer, pair_total, kl_total, gradient_error="Non-finite head gradient"
        )
