"""Reference-relative whole-turn preference gradients and bounded KL control.

Each starting position builds one independent graph over the deduplicated
observations of its nonzero comparisons, routing spatial features and 128-step
context windows via dense selection matrices (GEMM backward, avoiding sorted
indexing_backward under deterministic CUDA). Compressed prefix tries share
source prefixes and identical initial branch steps during no-grad passes.
Every scored decision retains its own detach boundary; GRU windows are not
joined. Only frozen inputs/references survive an optimizer update, never
candidate GRU states or activations. Zero-outcome comparisons remain in
normalization without running a model forward or graph.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field

import torch
from torch.nn import functional as F

from civ5rl.core.distributions import masked_kl_rows
from civ5rl.core.model import batch_inputs
from civ5rl.engine.comparison import PREFERENCE_TEMPERATURE
from civ5rl.engine.comparison import training as shared
from civ5rl.engine.comparison.inputs import History
from civ5rl.engine.comparison.reference import reference_distributions as collect_reference_distributions
from civ5rl.engine.runtime.transactions import clip_and_step_checked
from civ5sim.engine import digest

NO_GRAD_HISTORY_BATCH = 64
NO_GRAD_HEAD_BATCH = 128
BPTT_WINDOW_LENGTH = 128

HistoryKey = tuple[int, int]
BranchSteps = dict[int, list[tuple[str, int]]]


@dataclass
class _PackedWindows:
    spans: list[tuple[int, int, int]]
    observation_count: int
    target_indices: torch.Tensor
    window_indices: torch.Tensor
    ends: torch.Tensor | None


@dataclass
class PositionWork:
    windows: _PackedWindows
    boundary_keys: list[HistoryKey | None]
    action_batch: dict[str, torch.Tensor]
    key_target_slots: torch.Tensor
    chosen: torch.Tensor
    key_slots: torch.Tensor
    indices: torch.Tensor
    mask: torch.Tensor
    signs: torch.Tensor
    weights: torch.Tensor


@dataclass
class TurnBatch:
    keys: list[HistoryKey]
    indices: torch.Tensor
    mask: torch.Tensor
    signs: torch.Tensor
    weights: torch.Tensor
    positions: list[PositionWork] = field(default_factory=list)
    branch_steps: BranchSteps | None = None


@dataclass(frozen=True)
class TurnPreference:
    a: list[tuple[int, int]]
    b: list[tuple[int, int]]
    sign: int
    repeats: int


@dataclass
class PreparedPreferences:
    rows: list[list[TurnPreference]]
    training_observations: set[str]
    branch_steps: BranchSteps = field(default_factory=dict)


@dataclass
class AnchorSubset:
    """Fixed membership with one device mask per stable evaluation batch."""

    outside: frozenset[tuple[int, int]]
    masks: dict[tuple[torch.device, tuple[tuple[int, int], ...]], torch.Tensor] = field(default_factory=dict)

    @property
    def outside_count(self):
        return len(self.outside)

    def mask(self, keys, device):
        batch = (device, tuple(keys))
        if batch not in self.masks:
            self.masks[batch] = torch.tensor([key in self.outside for key in keys], device=device)
        return self.masks[batch]


def observation_key(row):
    return digest({"observation": row["observation"], "actions": row["actions"]})


def prepare_anchor_subset(histories, anchors, training_observations, *, budget=None):
    """Classify each anchor once, reading each source history once per pass."""
    by_episode = defaultdict(list)
    for episode, index in anchors:
        by_episode[episode].append(index)
    outside = set()
    for episode, indices in by_episode.items():
        if budget is not None and budget.expired():
            raise TimeoutError("Whole-turn anchor preparation budget exhausted")
        rows = histories.storage.source_rows(episode)
        outside.update(
            (episode, index) for index in indices if observation_key(rows[index]) not in training_observations
        )
    return AnchorSubset(frozenset(outside))


def preference_loss(current, parent, indices, mask, signs, weights):
    """Means include each pair's fixed 4/8 repeats and every batch position.

    signs=0 has exactly zero value and derivative, not a softplus(0) constant.
    Empty identical suffixes also have zero derivative without dividing by zero.
    """
    scores = ((current - parent)[indices] * mask).sum(-1) / mask.sum(-1).clamp_min(1)
    return (weights * (signs != 0) * F.softplus(-signs * (scores[:, 0] - scores[:, 1]))).sum()


def prepare_preferences(histories, positions, *, budget=None):
    """Validate recorded comparisons and retain only targets and diagnostic identities."""
    prepared, observations, branch_steps = [], set(), {}
    scored = set()
    for position in positions:
        if budget is not None and budget.expired():
            raise TimeoutError("Whole-turn preference preparation budget exhausted")
        records, preferences, branch_max = {}, [], {}
        for c in position["comparisons"]:
            if not c["sign"]:
                continue
            # Branches belong to one position. Release raw rows before the next one.
            for branch in (c["a"], c["b"]):
                if branch not in records:
                    records[branch] = histories.storage.turn_record(branch)
            ra, win_a = records[c["a"]]
            rb, win_b = records[c["b"]]
            if win_a - win_b != c["sign"]:
                raise ValueError("Training outcome differs from verified continuation")
            a, b = histories.comparison_targets(c["a"], c["b"], ra, rb)
            if not a and not b:
                continue
            for episode, index in (*a, *b):
                branch_max[episode] = max(branch_max.get(episode, -1), index)
            preferences.append(TurnPreference(a, b, c["sign"], c["repeats"]))
        for episode, max_idx in branch_max.items():
            pref = histories.branches[episode]["prefix"]
            rows = records[episode][0]
            branch_steps[episode] = [
                (observation_key(row), row["chosen"]) for row in rows[: max_idx - pref + 1]
            ]
        for episode, index in sorted({k for p in preferences for k in (*p.a, *p.b)}):
            if (episode, index) not in scored:
                offset = index - histories.branches[episode]["prefix"]
                observations.add(branch_steps[episode][offset][0])
                scored.add((episode, index))
        prepared.append(preferences)
    return PreparedPreferences(prepared, observations, branch_steps)


@dataclass
class _CompressedSpan:
    episode: int
    start: int
    stop: int
    parent_span: int
    level: int


@dataclass
class _BranchTrie:
    parent: list[int]
    representatives: list[HistoryKey]
    depth: list[int]
    key_to_node: dict[HistoryKey, int]
    spans: list[_CompressedSpan]
    node_to_span_offset: dict[int, tuple[int, int]]
    has_children: list[bool]


def _build_trie(histories, keys, *, branch_steps=None):
    """Share source prefixes and matching branch continuations."""
    last_index = {}
    for episode, index in map(tuple, keys):
        if index > last_index.get(episode, -1):
            last_index[episode] = index
    branches = getattr(histories, "branches", None) or {}
    steps_by_branch = branch_steps or {}
    parents = []
    representatives = []
    depths = []
    children = []
    source_nodes, tail_nodes, key_to_node = {}, {}, {}

    def add_node(previous, episode, index):
        node = len(parents)
        parents.append(previous)
        representatives.append((episode, index))
        depths.append(index)
        children.append([])
        if previous >= 0:
            children[previous].append(node)
        return node

    for episode, last in last_index.items():
        branch = branches.get(episode) if episode in steps_by_branch else None
        source = branch["episode"] if branch is not None else episode
        boundary = branch["prefix"] if branch is not None else last + 1
        previous = -1
        for index in range(min(last + 1, boundary)):
            key = (source, index)
            node = source_nodes.get(key)
            if node is None:
                node = source_nodes[key] = add_node(previous, source, index)
            key_to_node[episode, index] = node
            previous = node
        if branch is None or last < boundary:
            continue

        transition = ("prefix", source, boundary)
        steps = steps_by_branch[episode]
        for index in range(boundary, last + 1):
            observation, chosen = steps[index - boundary]
            token = (transition, observation)
            node = tail_nodes.get(token)
            if node is None:
                node = tail_nodes[token] = add_node(previous, episode, index)
            key_to_node[episode, index] = node
            previous = node
            transition = (node, chosen)

    spans, node_to_span_offset, has_children = [], {}, []
    queue = deque((node, -1, 0) for node, parent in enumerate(parents) if parent == -1)
    while queue:
        start_node, parent_span, level = queue.popleft()
        episode, start = representatives[start_node]
        nodes = [start_node]
        node = start_node
        while len(children[node]) == 1:
            child = children[node][0]
            if representatives[child] != (episode, representatives[node][1] + 1):
                break
            nodes.append(child)
            node = child
        span_id = len(spans)
        spans.append(_CompressedSpan(episode, start, representatives[node][1] + 1, parent_span, level))
        has_children.append(bool(children[node]))
        for offset, target in enumerate(nodes):
            node_to_span_offset[target] = (span_id, offset)
        for child in children[node]:
            queue.append((child, span_id, level + 1))
    return _BranchTrie(
        parents, representatives, depths, key_to_node, spans, node_to_span_offset, has_children
    )


def _cell_count(histories, episode):
    storage = getattr(histories, "storage", None)
    if storage is not None:
        if episode in storage.records:
            return int(storage.records[episode]["spatial"]["cells"].shape[1])
        branches = getattr(histories, "branches", None) or {}
        if episode in branches and branches[episode]["episode"] in storage.records:
            return int(storage.records[branches[episode]["episode"]]["spatial"]["cells"].shape[1])
    history, _ = histories.get(episode)
    return len(history[0].cells)


def _history_span(histories, episode, start, stop):
    history, _ = histories.get(episode)
    return history[start:stop]


def _spatial_inputs(histories, sequences, device, attention_backend):
    if all(isinstance(seq, History) and seq.owner is histories for seq in sequences):
        inputs = histories.spatial_inputs(sequences, device)
        if attention_backend != "compact_fp32":
            inputs.pop("unit_indices", None)
        return inputs
    items = [item for sequence in sequences for item in sequence]
    return batch_inputs(
        items,
        device,
        spatial_only=True,
        compact_units=attention_backend == "compact_fp32",
    )


@torch.no_grad()
def _encode_histories_nograd(policy, histories, sequences, *, target_offsets=None, chunk=32):
    device = next(policy.parameters()).device
    inputs = _spatial_inputs(histories, sequences, device, policy.attention_backend)
    total = len(inputs["cells"])
    contexts = []
    xs = [] if target_offsets is not None else None
    cursor = 0
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        row_x, row_ctx = policy.encode_spatial({k: v[start:stop] for k, v in inputs.items()})
        contexts.append(row_ctx)
        if xs is not None:
            end = cursor
            while end < len(target_offsets) and target_offsets[end] < stop:
                end += 1
            if end > cursor:
                local = torch.tensor(
                    [target_offsets[i] - start for i in range(cursor, end)],
                    device=device,
                    dtype=torch.long,
                )
                xs.append(row_x.index_select(0, local))
                cursor = end
    x_out = torch.cat(xs, 0) if xs else None
    return x_out, torch.cat(contexts, 0)


def _pack_windows(row_keys, windows, device):
    """Map independent BPTT windows to one sorted set of encoded spatial rows."""
    spans = []
    for episode, index in row_keys:
        if spans and spans[-1][0] == episode and spans[-1][2] == index:
            spans[-1] = (episode, spans[-1][1], index + 1)
        else:
            spans.append((episode, index, index + 1))

    lengths = [len(window) for window in windows]
    width = max(lengths)
    indices = torch.full((width, len(windows)), -1, dtype=torch.long)
    for column, window in enumerate(windows):
        for row, flat_index in enumerate(window):
            indices[row, column] = flat_index
    ends = (
        None
        if all(length == width for length in lengths)
        else torch.tensor([length - 1 for length in lengths], device=device, dtype=torch.long).view(
            1, len(windows), 1
        )
    )
    return _PackedWindows(
        spans=spans,
        observation_count=len(row_keys),
        target_indices=torch.tensor([window[-1] for window in windows], device=device, dtype=torch.long),
        window_indices=indices.to(device),
        ends=ends,
    )


def _pack_position_work(preferences, global_slots, step_positions, histories, branch_steps, device):
    position_keys = sorted({key for preference in preferences for key in (*preference.a, *preference.b)})
    trie = _build_trie(histories, position_keys, branch_steps=branch_steps)
    target_nodes = sorted({trie.key_to_node[key] for key in position_keys})
    target_slots = {node: index for index, node in enumerate(target_nodes)}
    windows, boundary_keys = [], []
    covered = set()
    for target_node in target_nodes:
        node = target_node
        window = []
        for _ in range(min(trie.depth[target_node] + 1, BPTT_WINDOW_LENGTH)):
            window.append(node)
            node = trie.parent[node]
        window.reverse()
        windows.append(window)
        covered.update(window)
        boundary_keys.append(trie.representatives[node] if node >= 0 else None)

    ordered_nodes = sorted(covered, key=trie.representatives.__getitem__)
    flat_slots = {node: index for index, node in enumerate(ordered_nodes)}
    packed_windows = _pack_windows(
        [trie.representatives[node] for node in ordered_nodes],
        [[flat_slots[node] for node in window] for window in windows],
        device,
    )
    action_batch = batch_inputs(
        [histories.item(*trie.representatives[node]) for node in target_nodes],
        device,
        actions_only=True,
    )
    local_slots = {key: i for i, key in enumerate(position_keys)}
    width = max(max(len(c.a), len(c.b)) for c in preferences)
    indices = torch.zeros((len(preferences), 2, width), dtype=torch.long)
    mask = torch.zeros_like(indices, dtype=torch.float32)
    for i, c in enumerate(preferences):
        for side, sequence in enumerate((c.a, c.b)):
            if sequence:
                indices[i, side, : len(sequence)] = torch.tensor([local_slots[k] for k in sequence])
                mask[i, side, : len(sequence)] = 1

    return PositionWork(
        windows=packed_windows,
        boundary_keys=boundary_keys,
        action_batch=action_batch,
        key_target_slots=torch.tensor(
            [target_slots[trie.key_to_node[k]] for k in position_keys], device=device, dtype=torch.long
        ),
        chosen=torch.tensor([histories.targets[k] for k in position_keys], device=device, dtype=torch.long),
        key_slots=torch.tensor([global_slots[k] for k in position_keys], device=device, dtype=torch.long),
        indices=indices.to(device),
        mask=mask.to(device),
        signs=torch.tensor([c.sign for c in preferences], device=device, dtype=torch.float32),
        weights=torch.tensor([1 / (step_positions * 6 * c.repeats) for c in preferences], device=device),
    )


def pack_batch(positions, device, *, histories=None, branch_steps=None):
    """Pack scored suffixes without revisiting raw histories or changing their weights."""
    comparisons = [comparison for position in positions for comparison in position]
    keys = sorted({key for c in comparisons for key in (*c.a, *c.b)})
    slots = {key: i for i, key in enumerate(keys)}
    width = max((max(len(c.a), len(c.b)) for c in comparisons), default=1)
    indices = torch.zeros((len(comparisons), 2, width), dtype=torch.long)
    mask = torch.zeros_like(indices, dtype=torch.float32)
    for i, c in enumerate(comparisons):
        for side, sequence in enumerate((c.a, c.b)):
            if sequence:
                indices[i, side, : len(sequence)] = torch.tensor([slots[k] for k in sequence])
                mask[i, side, : len(sequence)] = 1
    pos_works = []
    if histories is not None:
        for pos in positions:
            nonzero = [c for c in pos if c.sign]
            if nonzero:
                pos_works.append(
                    _pack_position_work(nonzero, slots, len(positions), histories, branch_steps, device)
                )
    return TurnBatch(
        keys,
        indices.to(device),
        mask.to(device),
        torch.tensor([c.sign for c in comparisons], device=device, dtype=torch.float32),
        torch.tensor([1 / (len(positions) * 6 * c.repeats) for c in comparisons], device=device),
        positions=pos_works,
        branch_steps=branch_steps,
    )


def _trie_target_batches(policy, histories, trie, keys, *, include_spatial=False):
    span_targets = defaultdict(list)
    for node in {trie.key_to_node[key] for key in keys}:
        span_id, offset = trie.node_to_span_offset[node]
        span_targets[span_id].append((offset, node))
    by_level = defaultdict(lambda: defaultdict(list))
    for span_id, span in enumerate(trie.spans):
        by_level[span.level][_cell_count(histories, span.episode)].append(span_id)

    parent_hidden = {}
    for level in sorted(by_level):
        child_hidden = {}
        for span_ids in by_level[level].values():
            for start in range(0, len(span_ids), NO_GRAD_HISTORY_BATCH):
                batch_ids = span_ids[start : start + NO_GRAD_HISTORY_BATCH]
                spans = [trie.spans[span_id] for span_id in batch_ids]
                sequences = [_history_span(histories, span.episode, span.start, span.stop) for span in spans]
                lengths = [len(sequence) for sequence in sequences]
                target_offsets, target_nodes = [], []
                offset = 0
                for span_id, length in zip(batch_ids, lengths, strict=True):
                    for position, node in sorted(span_targets.get(span_id, ())):
                        target_offsets.append(offset + position)
                        target_nodes.append(node)
                    offset += length
                spatial_targets, context = _encode_histories_nograd(
                    policy,
                    histories,
                    sequences,
                    target_offsets=target_offsets if include_spatial and target_offsets else None,
                )
                initial_hidden = (
                    torch.cat([parent_hidden[span.parent_span] for span in spans], 1) if level else None
                )
                hidden, last = policy.sequence_memory(context, lengths, initial_hidden)
                for index, span_id in enumerate(batch_ids):
                    if trie.has_children[span_id]:
                        child_hidden[span_id] = last[:, index : index + 1].detach().clone()
                if target_nodes:
                    yield target_nodes, target_offsets, hidden, spatial_targets
        parent_hidden = child_hidden


@torch.no_grad()
def hidden_states_at(policy, histories, keys, *, branch_steps=None):
    """Compute detached GRU states at requested history positions."""
    unique_keys = list(dict.fromkeys(map(tuple, keys)))
    if not unique_keys:
        return {}
    trie = _build_trie(histories, unique_keys, branch_steps=branch_steps)
    by_node = {}
    for nodes, offsets, hidden, _ in _trie_target_batches(policy, histories, trie, unique_keys):
        for node, offset in zip(nodes, offsets, strict=True):
            by_node[node] = hidden[offset].detach().clone()
    return {key: by_node[trie.key_to_node[key]] for key in unique_keys}


@torch.no_grad()
def current_chosen(policy, histories, keys, microbatch=2, *, branch_steps=None):
    device = next(policy.parameters()).device
    if not keys:
        return torch.empty(0, device=device)
    key_tuples = list(map(tuple, keys))
    unique_keys = list(dict.fromkeys(key_tuples))
    trie = _build_trie(histories, unique_keys, branch_steps=branch_steps)
    by_node = {}
    for nodes, offsets, hidden, spatial_targets in _trie_target_batches(
        policy, histories, trie, unique_keys, include_spatial=True
    ):
        hidden_targets = hidden.index_select(0, torch.tensor(offsets, device=device, dtype=torch.long))
        for start in range(0, len(nodes), NO_GRAD_HEAD_BATCH):
            part = nodes[start : start + NO_GRAD_HEAD_BATCH]
            items = [histories.item(*trie.representatives[node]) for node in part]
            stop = start + len(part)
            logits, _ = policy.score_heads(
                batch_inputs(items, device, actions_only=True),
                spatial_targets[start:stop],
                hidden_targets[start:stop],
            )
            log_probs = (logits / PREFERENCE_TEMPERATURE).log_softmax(-1)
            for row, item, node in zip(log_probs, items, part, strict=True):
                by_node[node] = row[: len(item.sources)].detach()
    return torch.stack([by_node[trie.key_to_node[key]][histories.targets[key]] for key in key_tuples])


def _windowed_target_logits(policy, histories, windows, initial_hidden, action_batch, device):
    """Encode shared rows once and select independent gradient windows by matrix multiply."""
    sequences = [_history_span(histories, episode, start, stop) for episode, start, stop in windows.spans]
    inputs = _spatial_inputs(histories, sequences, device, policy.attention_backend)
    spatial, context = policy.encode_spatial(inputs)
    row_count = windows.observation_count
    target_count = len(windows.target_indices)
    spatial_selection = F.one_hot(windows.target_indices, row_count).to(spatial.dtype)
    target_spatial = (spatial_selection @ spatial.reshape(row_count, -1)).reshape(
        target_count, *spatial.shape[1:]
    )
    width = windows.window_indices.shape[0]
    valid = windows.window_indices >= 0
    context_selection = (
        F.one_hot(windows.window_indices.clamp_min(0), row_count).to(context.dtype) * valid.unsqueeze(-1)
    ).reshape(width * target_count, row_count)
    padded_context = (context_selection @ context).reshape(width, target_count, -1)
    with torch.autocast(padded_context.device.type, enabled=False):
        runner = policy._memory_runner or policy.memory
        output, _ = runner(padded_context.float(), None if initial_hidden is None else initial_hidden.float())
    target_hidden = (
        output[-1]
        if windows.ends is None
        else output.gather(0, windows.ends.expand(1, target_count, policy.hidden_size)).squeeze(0)
    )
    logits, _ = policy.score_heads(action_batch, target_spatial, target_hidden)
    return logits


def _anchor_microbatches(histories, unique_anchors, microbatch):
    obs_budget = max(BPTT_WINDOW_LENGTH, microbatch * BPTT_WINDOW_LENGTH)
    by_cells = defaultdict(list)
    for ep, idx in unique_anchors:
        by_cells[_cell_count(histories, ep)].append((ep, idx))
    batches = []
    for anchors in by_cells.values():
        anchors.sort()
        current, covered = [], set()
        for ep, idx in anchors:
            boundary = max(0, idx + 1 - BPTT_WINDOW_LENGTH)
            window = {(ep, t) for t in range(boundary, idx + 1)}
            if current and len(covered | window) > obs_budget:
                batches.append((current, covered))
                current, covered = [], set()
            current.append((ep, idx))
            covered.update(window)
        if current:
            batches.append((current, covered))
    return batches


def logical_step(
    policy,
    optimizer,
    histories,
    batch,
    anchors,
    parent_chosen,
    references,
    *,
    microbatch=2,
    anchor_prefixes=None,
    branch_prefixes=None,
):
    if policy.amp_enabled:
        raise ValueError("Whole-turn fit requires FP32")
    device = next(policy.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    pair_total = torch.zeros((), device=device)

    if batch.positions:
        if branch_prefixes is None:
            needed_bp = [bk for pos in batch.positions for bk in pos.boundary_keys if bk is not None]
            branch_prefixes = hidden_states_at(policy, histories, needed_bp, branch_steps=batch.branch_steps)
        zero_1d = torch.zeros(policy.hidden_size, device=device)
        for pos in batch.positions:
            h_init = (
                torch.stack(
                    [branch_prefixes[bk] if bk is not None else zero_1d for bk in pos.boundary_keys]
                ).unsqueeze(0)
                if any(bk is not None for bk in pos.boundary_keys)
                else None
            )
            logits = _windowed_target_logits(
                policy,
                histories,
                pos.windows,
                h_init,
                pos.action_batch,
                device,
            )
            log_probs = (
                (logits / PREFERENCE_TEMPERATURE)
                .log_softmax(-1)
                .masked_fill(~pos.action_batch["action_mask"], 0.0)
            )
            m_key = F.one_hot(pos.key_target_slots, len(pos.action_batch["sources"])).to(log_probs.dtype)
            current = (m_key @ log_probs).gather(1, pos.chosen[:, None]).squeeze(1)
            loss = preference_loss(
                current, parent_chosen[pos.key_slots], pos.indices, pos.mask, pos.signs, pos.weights
            )
            loss.backward()
            pair_total += loss.detach()
    elif batch.keys:
        # Fallback for callers that construct TurnBatch without histories.
        measured = (
            current_chosen(policy, histories, batch.keys, microbatch, branch_steps=batch.branch_steps)
            .detach()
            .requires_grad_(True)
        )
        pair = preference_loss(measured, parent_chosen, batch.indices, batch.mask, batch.signs, batch.weights)
        (coefficients,) = torch.autograd.grad(pair, measured)
        coefficients = coefficients.detach()
        slots = {key: i for i, key in enumerate(batch.keys)}
        bp_keys = [
            (k[0], max(0, k[1] + 1 - BPTT_WINDOW_LENGTH) - 1)
            for k in batch.keys
            if k[1] >= BPTT_WINDOW_LENGTH
        ]
        bp_h = hidden_states_at(policy, histories, bp_keys, branch_steps=batch.branch_steps)
        prefixes = {
            k: bp_h[(k[0], k[1] - BPTT_WINDOW_LENGTH)][None, None]
            for k in batch.keys
            if k[1] >= BPTT_WINDOW_LENGTH
        }
        zero = torch.zeros(1, 1, policy.hidden_size, device=device)
        for k in batch.keys:
            prefixes.setdefault(k, zero)
        for entries in shared._history_groups(histories, batch.keys, size=microbatch):
            logits = shared.batched_target_logits(
                policy, entries, gradient=True, padded=True, prefixes=prefixes
            )
            indices = torch.tensor([histories.targets[key] for key, _ in entries], device=logits.device)
            values = (logits / PREFERENCE_TEMPERATURE).log_softmax(-1).gather(1, indices[:, None]).squeeze(1)
            ids = torch.tensor([slots[key] for key, _ in entries], device=device)
            (values * coefficients[ids]).sum().backward()
        pair_total = pair.detach()

    kl_sum = torch.zeros((), device=device)
    anchor_keys = list(map(tuple, anchors))
    if anchor_keys:
        anchor_counts = Counter(anchor_keys)
        unique_anchors = list(anchor_counts)
        if anchor_prefixes is None:
            needed_bp = [
                (ep, idx - BPTT_WINDOW_LENGTH) for ep, idx in unique_anchors if idx >= BPTT_WINDOW_LENGTH
            ]
            bp_h = hidden_states_at(policy, histories, needed_bp)
            anchor_prefixes = {
                (ep, idx): bp_h[(ep, idx - BPTT_WINDOW_LENGTH)]
                for ep, idx in unique_anchors
                if idx >= BPTT_WINDOW_LENGTH
            }
        zero_1d = torch.zeros(policy.hidden_size, device=device)
        for mb_anchors, covered in _anchor_microbatches(histories, unique_anchors, microbatch):
            row_keys = sorted(covered)
            flat_slots = {key: index for index, key in enumerate(row_keys)}
            windows = _pack_windows(
                row_keys,
                [
                    [
                        flat_slots[episode, index]
                        for index in range(max(0, end + 1 - BPTT_WINDOW_LENGTH), end + 1)
                    ]
                    for episode, end in mb_anchors
                ],
                device,
            )
            h_init = (
                torch.stack([anchor_prefixes.get(k, zero_1d) for k in mb_anchors]).unsqueeze(0)
                if any(idx >= BPTT_WINDOW_LENGTH for _, idx in mb_anchors)
                else None
            )
            action_batch = batch_inputs(
                [histories.item(ep, idx) for ep, idx in mb_anchors], device, actions_only=True
            )
            logits = _windowed_target_logits(
                policy,
                histories,
                windows,
                h_init,
                action_batch,
                device,
            )
            parent, _ = references.gather(mb_anchors, logits.shape[1])
            weights = torch.tensor(
                [anchor_counts[k] for k in mb_anchors], device=device, dtype=logits.dtype
            ) / len(anchor_keys)
            loss = (
                masked_kl_rows(
                    parent, (logits / PREFERENCE_TEMPERATURE).log_softmax(-1), ~torch.isneginf(logits)
                )
                * weights
            ).sum()
            loss.backward()
            kl_sum += loss.detach()

    step = clip_and_step_checked(
        policy,
        optimizer,
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        asynchronous=device.type == "cuda",
        gradient_error="Non-finite whole-turn gradient",
        extra_metrics=(pair_total, kl_sum),
    )
    finite = torch.ones((), device=device, dtype=torch.bool)
    for flag in step.finite_flags.values():
        finite &= flag.to(device)
    values = torch.stack((*step.metrics, finite.to(step.gradient_norm.dtype))).detach().cpu().tolist()
    if not values[-1] or not all(math.isfinite(v) for v in values[:-1]):
        raise FloatingPointError("Non-finite whole-turn optimizer attempt")
    return {
        "pair_loss": values[0],
        "anchor_kl": values[1],
        "gradient_norm": values[2],
        "scored_decisions": len(batch.keys),
        "nonzero_comparisons": len(batch.signs),
    }


@torch.no_grad()
def reference_logits_batches(policy, histories, anchors, *, collect_prefixes=None, collected_prefixes=None):
    """Batched no-grad pass over complete anchor episodes, keeping x only at targets."""
    grouped = {}
    for episode, index in anchors:
        grouped.setdefault(episode, set()).add(index)
    collect_by_ep = defaultdict(list)
    if collect_prefixes is not None and collected_prefixes is not None:
        for ep, idx in collect_prefixes:
            ep_idx = (int(ep), int(idx))
            if ep_idx[1] >= BPTT_WINDOW_LENGTH:
                collect_by_ep[ep_idx[0]].append(ep_idx)
                grouped.setdefault(ep_idx[0], set())
    by_cells = defaultdict(list)
    for episode, indices in grouped.items():
        max_idx = max(indices) if indices else 0
        for ep_idx in collect_by_ep.get(episode, ()):
            max_idx = max(max_idx, ep_idx[1] - BPTT_WINDOW_LENGTH)
        by_cells[_cell_count(histories, episode)].append((episode, max_idx))

    device = next(policy.parameters()).device
    for episodes in by_cells.values():
        episodes.sort()
        for start in range(0, len(episodes), NO_GRAD_HISTORY_BATCH):
            part_eps = episodes[start : start + NO_GRAD_HISTORY_BATCH]
            sequences = [_history_span(histories, ep, 0, max_idx + 1) for ep, max_idx in part_eps]
            lengths = [len(seq) for seq in sequences]
            target_offsets, targets = [], []
            offset = 0
            for (ep, _), length in zip(part_eps, lengths):
                for idx in sorted(grouped[ep]):
                    target_offsets.append(offset + idx)
                    targets.append(((ep, idx), histories.item(ep, idx)))
                offset += length
            x_targets, context = _encode_histories_nograd(
                policy,
                histories,
                sequences,
                target_offsets=target_offsets if target_offsets else None,
            )
            h, _ = policy.sequence_memory(context, lengths)
            if collected_prefixes is not None:
                offset = 0
                for (ep, _), length in zip(part_eps, lengths):
                    for key in collect_by_ep.get(ep, ()):
                        collected_prefixes[key] = h[offset + key[1] - BPTT_WINDOW_LENGTH].detach().clone()
                    offset += length
            if target_offsets:
                h_targets = h.index_select(0, torch.tensor(target_offsets, device=device, dtype=torch.long))
                for head_start in range(0, len(targets), NO_GRAD_HEAD_BATCH):
                    part = targets[head_start : head_start + NO_GRAD_HEAD_BATCH]
                    logits, _ = policy.score_heads(
                        batch_inputs([item for _, item in part], device, actions_only=True),
                        x_targets[head_start : head_start + len(part)],
                        h_targets[head_start : head_start + len(part)],
                    )
                    yield [(key, len(item.sources)) for key, item in part], logits


def reference_distributions(policy, histories, anchors):
    return collect_reference_distributions(
        policy, histories, anchors, logits_batches=reference_logits_batches
    )


@torch.no_grad()
def control_kl(
    policy,
    histories,
    anchors,
    references,
    *,
    anchor_subset,
    collect_prefixes=None,
    collected_prefixes=None,
):
    """One complete-history pass; FP64 KL reduction and argmax-change diagnostics."""
    if policy.amp_enabled:
        raise ValueError("Whole-turn control requires FP32")
    device = next(policy.parameters()).device
    total = torch.zeros((), device=device, dtype=torch.float64)
    changed = torch.zeros((), device=device, dtype=torch.long)
    outside_changed = torch.zeros_like(changed)
    outside_count = anchor_subset.outside_count
    for part, logits in reference_logits_batches(
        policy,
        histories,
        anchors,
        collect_prefixes=collect_prefixes,
        collected_prefixes=collected_prefixes,
    ):
        keys = [key for key, _ in part]
        parent, probabilities = references.gather(keys, logits.shape[1])
        total += shared.full_kl_rows(logits, parent, probabilities).sum()
        differs = logits.argmax(-1) != parent.masked_fill(probabilities == 0, -torch.inf).argmax(-1)
        changed += differs.sum()
        outside_changed += (differs & anchor_subset.mask(keys, device)).sum()
    values = torch.stack((total / len(anchors), changed.double(), outside_changed.double())).cpu().tolist()
    return values[0], {
        "anchor_argmax_changed": int(values[1]),
        "anchor_argmax_changed_fraction": values[1] / len(anchors),
        "outside_training_anchors": outside_count,
        "outside_argmax_changed": int(values[2]),
        "outside_argmax_changed_fraction": values[2] / outside_count if outside_count else None,
    }
