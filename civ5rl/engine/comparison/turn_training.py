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

import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field

import torch
from torch.nn import functional as F

from civ5rl.core.distributions import masked_kl_rows
from civ5rl.core.model import batch_inputs
from civ5rl.engine.comparison import training as shared
from civ5rl.engine.comparison.inputs import History
from civ5rl.engine.runtime.transactions import finite_state_flags
from civ5sim.engine import digest

NO_GRAD_HISTORY_BATCH = 64
NO_GRAD_HEAD_BATCH = 128


@dataclass
class PositionWork:
    spans: list[tuple[int, int, int]]
    u_count: int
    target_flat_indices: torch.Tensor
    window_flat_indices: torch.Tensor
    ends: torch.Tensor | None
    boundary_keys: list[tuple[int, int] | None]
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
    keys: list
    indices: torch.Tensor
    mask: torch.Tensor
    signs: torch.Tensor
    weights: torch.Tensor
    positions: list[PositionWork] = field(default_factory=list)
    branch_steps: dict[int, list[tuple[str, int]]] | None = None


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
    branch_steps: dict[int, list[tuple[str, int]]] = field(default_factory=dict)


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
    return digest(dict(observation=row['observation'], actions=row['actions']))


def prepare_anchor_subset(histories, anchors, training_observations, *, budget=None):
    """Classify each anchor once, reading each source history once per pass."""
    by_episode = defaultdict(list)
    for episode, index in anchors:
        by_episode[episode].append(index)
    outside = set()
    for episode, indices in by_episode.items():
        if budget is not None and budget.expired():
            raise TimeoutError('Whole-turn anchor preparation budget exhausted')
        rows = histories.storage.source_rows(episode)
        outside.update((episode, index) for index in indices
                       if observation_key(rows[index]) not in training_observations)
    return AnchorSubset(frozenset(outside))


def preference_loss(current, parent, indices, mask, signs, weights):
    """Means include each pair's fixed 4/8 repeats and every batch position.

    signs=0 has exactly zero value and derivative, not a softplus(0) constant.
    Empty identical suffixes also have zero derivative without dividing by zero.
    """
    scores = ((current-parent)[indices] * mask).sum(-1) / mask.sum(-1).clamp_min(1)
    return (weights * (signs != 0) * F.softplus(-signs * (scores[:,0]-scores[:,1]))).sum()


def prepare_preferences(histories, positions, *, budget=None):
    """Validate recorded comparisons and retain only targets and diagnostic identities."""
    prepared, observations, branch_steps = [], set(), {}
    scored = set()
    for position in positions:
        if budget is not None and budget.expired():
            raise TimeoutError('Whole-turn preference preparation budget exhausted')
        records, preferences, branch_max = {}, [], {}
        for c in position['comparisons']:
            if not c['sign']:
                continue
            # Branches belong to one position. Release raw rows before the next one.
            for branch in (c['a'], c['b']):
                if branch not in records:
                    records[branch] = histories.storage.turn_record(branch)
            ra, win_a = records[c['a']]
            rb, win_b = records[c['b']]
            if win_a - win_b != c['sign']:
                raise ValueError('Training outcome differs from verified continuation')
            a, b = histories.comparison_targets(c['a'], c['b'], ra, rb)
            if not a and not b:
                continue
            for episode, index in (*a, *b):
                branch_max[episode] = max(branch_max.get(episode, -1), index)
            preferences.append(TurnPreference(a, b, c['sign'], c['repeats']))
        for episode, max_idx in branch_max.items():
            pref = histories.branches[episode]['prefix']
            rows = records[episode][0]
            branch_steps[episode] = [
                (observation_key(row), row['chosen']) for row in rows[: max_idx - pref + 1]
            ]
        for episode, index in sorted({k for p in preferences for k in (*p.a, *p.b)}):
            if (episode, index) not in scored:
                offset = index - histories.branches[episode]['prefix']
                observations.add(branch_steps[episode][offset][0])
                scored.add((episode, index))
        prepared.append(preferences)
    return PreparedPreferences(prepared, observations, branch_steps)


@dataclass
class _CompressedSpan:
    ep: int
    start: int
    stop: int
    parent_span: int
    level: int
    nodes: list[int]


@dataclass
class _BranchTrie:
    parent: list[int]
    rep: list[tuple[int, int]]
    depth: list[int]
    key_to_node: dict[tuple[int, int], int]
    spans: list[_CompressedSpan]
    node_to_span_offset: dict[int, tuple[int, int]]
    has_children: list[bool]


def _build_trie(histories, keys, *, branch_steps=None):
    """Build a compressed public-history prefix tree sharing source prefixes and identical branch tails."""
    max_by_ep = {}
    for ep, idx in map(tuple, keys):
        if idx > max_by_ep.get(ep, -1):
            max_by_ep[ep] = idx
    branches = getattr(histories, 'branches', None) or {}
    steps_map = branch_steps or {}
    parent, rep, depth, children = [], [], [], []
    src_nodes, tail_nodes, key_to_node = {}, {}, {}
    for ep, max_idx in max_by_ep.items():
        if ep in branches and ep in steps_map:
            b = branches[ep]
            src, pref = b['episode'], b['prefix']
            prev = -1
            for t in range(min(max_idx + 1, pref)):
                token = (src, t)
                node = src_nodes.get(token)
                if node is None:
                    node = len(parent)
                    parent.append(prev)
                    rep.append((src, t))
                    depth.append(t)
                    children.append([])
                    src_nodes[token] = node
                    if prev >= 0:
                        children[prev].append(node)
                key_to_node[(ep, t)] = node
                prev = node
            if max_idx >= pref:
                trans = ('prefix', src, pref)
                b_steps = steps_map[ep]
                for t in range(pref, max_idx + 1):
                    obs_key, chosen = b_steps[t - pref]
                    token = (trans, obs_key)
                    node = tail_nodes.get(token)
                    if node is None:
                        node = len(parent)
                        parent.append(prev)
                        rep.append((ep, t))
                        depth.append(t)
                        children.append([])
                        tail_nodes[token] = node
                        if prev >= 0:
                            children[prev].append(node)
                    key_to_node[(ep, t)] = node
                    prev = node
                    trans = (node, chosen)
        else:
            prev = -1
            for t in range(max_idx + 1):
                token = (ep, t)
                node = src_nodes.get(token)
                if node is None:
                    node = len(parent)
                    parent.append(prev)
                    rep.append((ep, t))
                    depth.append(t)
                    children.append([])
                    src_nodes[token] = node
                    if prev >= 0:
                        children[prev].append(node)
                key_to_node[(ep, t)] = node
                prev = node

    spans, node_to_span_offset, has_children = [], {}, []
    queue = deque((u, -1, 0) for u in range(len(parent)) if parent[u] == -1)
    while queue:
        start_node, parent_span, level = queue.popleft()
        span_ep, span_start = rep[start_node]
        nodes = [start_node]
        cur = start_node
        while len(children[cur]) == 1:
            nxt = children[cur][0]
            if rep[nxt] != (span_ep, rep[cur][1] + 1):
                break
            nodes.append(nxt)
            cur = nxt
        span_id = len(spans)
        spans.append(_CompressedSpan(span_ep, span_start, rep[cur][1] + 1, parent_span, level, nodes))
        has_children.append(bool(children[cur]))
        for offset, node in enumerate(nodes):
            node_to_span_offset[node] = (span_id, offset)
        for child in children[cur]:
            queue.append((child, span_id, level + 1))
    return _BranchTrie(parent, rep, depth, key_to_node, spans, node_to_span_offset, has_children)


def _cell_count(histories, episode):
    storage = getattr(histories, 'storage', None)
    if storage is not None:
        if episode in storage.records:
            return int(storage.records[episode]['spatial']['cells'].shape[1])
        branches = getattr(histories, 'branches', None) or {}
        if episode in branches and branches[episode]['episode'] in storage.records:
            return int(storage.records[branches[episode]['episode']]['spatial']['cells'].shape[1])
    history, _ = histories.get(episode)
    return len(history[0].cells)


def _history_span(histories, episode, start, stop):
    history, _ = histories.get(episode)
    return history[start:stop]


def _spatial_inputs(histories, sequences, device, attention_backend):
    if all(isinstance(seq, History) and seq.owner is histories for seq in sequences):
        inputs = histories.spatial_inputs(sequences, device)
        if attention_backend != 'compact_fp32':
            inputs.pop('unit_indices', None)
        return inputs
    items = [item for sequence in sequences for item in sequence]
    return batch_inputs(
        items,
        device,
        spatial_only=True,
        compact_units=attention_backend == 'compact_fp32',
    )


@torch.no_grad()
def _encode_histories_nograd(policy, histories, sequences, *, target_offsets=None, chunk=32):
    device = next(policy.parameters()).device
    inputs = _spatial_inputs(histories, sequences, device, policy.attention_backend)
    total = len(inputs['cells'])
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


def _pack_position_work(pos_prefs, global_slots, step_positions, histories, branch_steps, device):
    pos_keys = sorted({key for c in pos_prefs for key in (*c.a, *c.b)})
    trie = _build_trie(histories, pos_keys, branch_steps=branch_steps)
    target_nodes = sorted({trie.key_to_node[k] for k in pos_keys})
    target_slot = {u: i for i, u in enumerate(target_nodes)}
    windows, boundary_keys = [], []
    u_set = set()
    for u in target_nodes:
        depth_u = trie.depth[u]
        boundary = max(0, depth_u + 1 - 128)
        cur = u
        path = []
        for _ in range(depth_u + 1 - boundary):
            path.append(cur)
            cur = trie.parent[cur]
        path.reverse()
        windows.append(path)
        u_set.update(path)
        boundary_keys.append(trie.rep[cur] if cur >= 0 else None)

    u_nodes = sorted(u_set, key=lambda u: trie.rep[u])
    spans, u_to_flat = [], {}
    for flat_idx, u in enumerate(u_nodes):
        ep, idx = trie.rep[u]
        u_to_flat[u] = flat_idx
        if spans and spans[-1][0] == ep and spans[-1][2] == idx:
            spans[-1] = (ep, spans[-1][1], idx + 1)
        else:
            spans.append((ep, idx, idx + 1))

    k_unique = len(target_nodes)
    t_max = max(len(w) for w in windows)
    window_flat = torch.full((t_max, k_unique), -1, dtype=torch.long)
    for k, w in enumerate(windows):
        for t, u in enumerate(w):
            window_flat[t, k] = u_to_flat[u]
    lengths = [len(w) for w in windows]
    ends = (
        None
        if all(l == t_max for l in lengths)
        else torch.tensor([l - 1 for l in lengths], device=device, dtype=torch.long).view(1, k_unique, 1)
    )
    action_batch = batch_inputs(
        [histories.item(*trie.rep[u]) for u in target_nodes], device, actions_only=True
    )
    local_slots = {key: i for i, key in enumerate(pos_keys)}
    width = max(max(len(c.a), len(c.b)) for c in pos_prefs)
    indices = torch.zeros((len(pos_prefs), 2, width), dtype=torch.long)
    mask = torch.zeros_like(indices, dtype=torch.float32)
    for i, c in enumerate(pos_prefs):
        for side, sequence in enumerate((c.a, c.b)):
            if sequence:
                indices[i, side, :len(sequence)] = torch.tensor([local_slots[k] for k in sequence])
                mask[i, side, :len(sequence)] = 1

    return PositionWork(
        spans=spans,
        u_count=len(u_nodes),
        target_flat_indices=torch.tensor([u_to_flat[u] for u in target_nodes], device=device, dtype=torch.long),
        window_flat_indices=window_flat.to(device),
        ends=ends,
        boundary_keys=boundary_keys,
        action_batch=action_batch,
        key_target_slots=torch.tensor([target_slot[trie.key_to_node[k]] for k in pos_keys], device=device, dtype=torch.long),
        chosen=torch.tensor([histories.targets[k] for k in pos_keys], device=device, dtype=torch.long),
        key_slots=torch.tensor([global_slots[k] for k in pos_keys], device=device, dtype=torch.long),
        indices=indices.to(device),
        mask=mask.to(device),
        signs=torch.tensor([c.sign for c in pos_prefs], device=device, dtype=torch.float32),
        weights=torch.tensor([1 / (step_positions * 6 * c.repeats) for c in pos_prefs], device=device),
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
                indices[i, side, :len(sequence)] = torch.tensor([slots[k] for k in sequence])
                mask[i, side, :len(sequence)] = 1
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


@torch.no_grad()
def hidden_states_at(policy, histories, keys, *, branch_steps=None):
    """Compute detached GRU hidden states at requested (episode, index) positions without retaining x."""
    unique_keys = list(dict.fromkeys(map(tuple, keys)))
    if not unique_keys:
        return {}
    device = next(policy.parameters()).device
    trie = _build_trie(histories, unique_keys, branch_steps=branch_steps)
    span_targets = defaultdict(list)
    for node in {trie.key_to_node[k] for k in unique_keys}:
        sid, off = trie.node_to_span_offset[node]
        span_targets[sid].append((off, node))
    by_level = defaultdict(lambda: defaultdict(list))
    for sid, span in enumerate(trie.spans):
        by_level[span.level][_cell_count(histories, span.ep)].append(sid)

    node_h, last_h = {}, {}
    for level in sorted(by_level):
        next_last_h = {}
        for sids in by_level[level].values():
            for start in range(0, len(sids), NO_GRAD_HISTORY_BATCH):
                batch_sids = sids[start : start + NO_GRAD_HISTORY_BATCH]
                sequences = [
                    _history_span(histories, trie.spans[sid].ep, trie.spans[sid].start, trie.spans[sid].stop)
                    for sid in batch_sids
                ]
                _, context = _encode_histories_nograd(policy, histories, sequences, target_offsets=None)
                lengths = [len(seq) for seq in sequences]
                init_h = (
                    None
                    if level == 0
                    else torch.cat([last_h[trie.spans[sid].parent_span] for sid in batch_sids], 1)
                )
                h, last = policy.sequence_memory(context, lengths, init_h)
                offset = 0
                for i, (sid, length) in enumerate(zip(batch_sids, lengths)):
                    if trie.has_children[sid]:
                        next_last_h[sid] = last[:, i : i + 1].detach().clone()
                    for off, node in span_targets.get(sid, ()):
                        node_h[node] = h[offset + off].detach().clone()
                    offset += length
        last_h = next_last_h
    return {k: node_h[trie.key_to_node[k]] for k in unique_keys}


@torch.no_grad()
def current_chosen(policy, histories, keys, microbatch=2, *, branch_steps=None):
    if not keys:
        return torch.empty(0, device=next(policy.parameters()).device)
    device = next(policy.parameters()).device
    key_tuples = [tuple(k) for k in keys]
    unique_keys = list(dict.fromkeys(key_tuples))
    trie = _build_trie(histories, unique_keys, branch_steps=branch_steps)
    span_targets = defaultdict(list)
    for node in sorted({trie.key_to_node[k] for k in unique_keys}):
        sid, off = trie.node_to_span_offset[node]
        span_targets[sid].append((off, node))
    by_level = defaultdict(lambda: defaultdict(list))
    for sid, span in enumerate(trie.spans):
        by_level[span.level][_cell_count(histories, span.ep)].append(sid)

    node_log_probs, last_h = {}, {}
    for level in sorted(by_level):
        next_last_h = {}
        for sids in by_level[level].values():
            for start in range(0, len(sids), NO_GRAD_HISTORY_BATCH):
                batch_sids = sids[start : start + NO_GRAD_HISTORY_BATCH]
                sequences = [
                    _history_span(histories, trie.spans[sid].ep, trie.spans[sid].start, trie.spans[sid].stop)
                    for sid in batch_sids
                ]
                lengths = [len(seq) for seq in sequences]
                target_offsets, target_nodes = [], []
                offset = 0
                for sid, length in zip(batch_sids, lengths):
                    for off, node in sorted(span_targets.get(sid, ())):
                        target_offsets.append(offset + off)
                        target_nodes.append(node)
                    offset += length
                x_targets, context = _encode_histories_nograd(
                    policy,
                    histories,
                    sequences,
                    target_offsets=target_offsets if target_offsets else None,
                )
                init_h = (
                    None
                    if level == 0
                    else torch.cat([last_h[trie.spans[sid].parent_span] for sid in batch_sids], 1)
                )
                h, last = policy.sequence_memory(context, lengths, init_h)
                for i, sid in enumerate(batch_sids):
                    if trie.has_children[sid]:
                        next_last_h[sid] = last[:, i : i + 1].detach().clone()
                if target_offsets:
                    h_targets = h.index_select(0, torch.tensor(target_offsets, device=device, dtype=torch.long))
                    for head_start in range(0, len(target_nodes), NO_GRAD_HEAD_BATCH):
                        part_nodes = target_nodes[head_start : head_start + NO_GRAD_HEAD_BATCH]
                        items = [histories.item(*trie.rep[node]) for node in part_nodes]
                        logits, _ = policy.score_heads(
                            batch_inputs(items, device, actions_only=True),
                            x_targets[head_start : head_start + len(part_nodes)],
                            h_targets[head_start : head_start + len(part_nodes)],
                        )
                        logp = (logits / 0.5).log_softmax(-1)
                        for row, item, node in zip(logp, items, part_nodes):
                            node_log_probs[node] = row[: len(item.sources)].detach()
        last_h = next_last_h
    return torch.stack(
        [node_log_probs[trie.key_to_node[k]][histories.targets[k]] for k in key_tuples]
    )


def _windowed_target_logits(
    policy,
    histories,
    spans,
    u_count,
    target_flat_indices,
    window_flat_indices,
    ends,
    h_init,
    action_batch,
    device,
):
    """Encode unique observations once and construct BPTT windows with selection matrices."""
    sequences = [_history_span(histories, ep, start, stop) for ep, start, stop in spans]
    inputs = _spatial_inputs(histories, sequences, device, policy.attention_backend)
    x_u, ctx_u = policy.encode_spatial(inputs)
    k_unique = len(target_flat_indices)
    m_x = F.one_hot(target_flat_indices, u_count).to(x_u.dtype)
    x_targets = (m_x @ x_u.reshape(u_count, -1)).reshape(k_unique, *x_u.shape[1:])
    t_max = window_flat_indices.shape[0]
    valid_w = window_flat_indices >= 0
    m_ctx = (
        F.one_hot(window_flat_indices.clamp_min(0), u_count).to(ctx_u.dtype) * valid_w.unsqueeze(-1)
    ).reshape(t_max * k_unique, u_count)
    padded_ctx = (m_ctx @ ctx_u).reshape(t_max, k_unique, -1)
    with torch.autocast(padded_ctx.device.type, enabled=False):
        runner = policy._memory_runner or policy.memory
        output, _ = runner(padded_ctx.float(), None if h_init is None else h_init.float())
    h_targets = (
        output[-1]
        if ends is None
        else output.gather(0, ends.expand(1, k_unique, policy.hidden_size)).squeeze(0)
    )
    logits, _ = policy.score_heads(action_batch, x_targets, h_targets)
    return logits


def _anchor_microbatches(histories, unique_anchors, microbatch):
    obs_budget = max(128, microbatch * 128)
    by_cells = defaultdict(list)
    for ep, idx in unique_anchors:
        by_cells[_cell_count(histories, ep)].append((ep, idx))
    batches = []
    for anchors in by_cells.values():
        anchors.sort()
        current, covered = [], set()
        for ep, idx in anchors:
            boundary = max(0, idx + 1 - 128)
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
        raise ValueError('Whole-turn fit requires FP32')
    device = next(policy.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    pair_total = torch.zeros((), device=device)

    if batch.positions:
        if branch_prefixes is None:
            needed_bp = [
                bk for pos in batch.positions for bk in pos.boundary_keys if bk is not None
            ]
            branch_prefixes = hidden_states_at(
                policy, histories, needed_bp, branch_steps=batch.branch_steps
            )
        zero_1d = torch.zeros(policy.hidden_size, device=device)
        for pos in batch.positions:
            h_init = (
                torch.stack([branch_prefixes[bk] if bk is not None else zero_1d for bk in pos.boundary_keys]).unsqueeze(0)
                if any(bk is not None for bk in pos.boundary_keys)
                else None
            )
            logits = _windowed_target_logits(
                policy,
                histories,
                pos.spans,
                pos.u_count,
                pos.target_flat_indices,
                pos.window_flat_indices,
                pos.ends,
                h_init,
                pos.action_batch,
                device,
            )
            log_probs = (logits / 0.5).log_softmax(-1).masked_fill(~pos.action_batch['action_mask'], 0.0)
            m_key = F.one_hot(pos.key_target_slots, len(pos.action_batch['sources'])).to(log_probs.dtype)
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
        coefficients, = torch.autograd.grad(pair, measured)
        coefficients = coefficients.detach()
        slots = {key: i for i, key in enumerate(batch.keys)}
        bp_keys = [(k[0], max(0, k[1] + 1 - 128) - 1) for k in batch.keys if k[1] >= 128]
        bp_h = hidden_states_at(policy, histories, bp_keys, branch_steps=batch.branch_steps)
        prefixes = {k: bp_h[(k[0], k[1] - 128)][None, None] for k in batch.keys if k[1] >= 128}
        zero = torch.zeros(1, 1, policy.hidden_size, device=device)
        for k in batch.keys:
            prefixes.setdefault(k, zero)
        for entries in shared._history_groups(histories, batch.keys, size=microbatch):
            logits = shared.batched_target_logits(policy, entries, gradient=True, padded=True, prefixes=prefixes)
            indices = torch.tensor([histories.targets[key] for key, _ in entries], device=logits.device)
            values = (logits / 0.5).log_softmax(-1).gather(1, indices[:, None]).squeeze(1)
            ids = torch.tensor([slots[key] for key, _ in entries], device=device)
            (values * coefficients[ids]).sum().backward()
        pair_total = pair.detach()

    kl_sum = torch.zeros((), device=device)
    anchor_keys = list(map(tuple, anchors))
    if anchor_keys:
        anchor_counts = Counter(anchor_keys)
        unique_anchors = list(anchor_counts)
        if anchor_prefixes is None:
            needed_bp = [(ep, idx - 128) for ep, idx in unique_anchors if idx >= 128]
            bp_h = hidden_states_at(policy, histories, needed_bp)
            anchor_prefixes = {(ep, idx): bp_h[(ep, idx - 128)] for ep, idx in unique_anchors if idx >= 128}
        zero_1d = torch.zeros(policy.hidden_size, device=device)
        for mb_anchors, covered in _anchor_microbatches(histories, unique_anchors, microbatch):
            u_nodes = sorted(covered)
            u_to_flat = {node: i for i, node in enumerate(u_nodes)}
            spans = []
            for ep, idx in u_nodes:
                if spans and spans[-1][0] == ep and spans[-1][2] == idx:
                    spans[-1] = (ep, spans[-1][1], idx + 1)
                else:
                    spans.append((ep, idx, idx + 1))
            k_count = len(mb_anchors)
            windows = [
                [(ep, t) for t in range(max(0, idx + 1 - 128), idx + 1)]
                for ep, idx in mb_anchors
            ]
            t_max = max(len(w) for w in windows)
            window_flat = torch.full((t_max, k_count), -1, dtype=torch.long)
            for k, w in enumerate(windows):
                for t, node in enumerate(w):
                    window_flat[t, k] = u_to_flat[node]
            lengths = [len(w) for w in windows]
            ends = (
                None
                if all(l == t_max for l in lengths)
                else torch.tensor([l - 1 for l in lengths], device=device, dtype=torch.long).view(1, k_count, 1)
            )
            h_init = (
                torch.stack([anchor_prefixes.get(k, zero_1d) for k in mb_anchors]).unsqueeze(0)
                if any(idx >= 128 for _, idx in mb_anchors)
                else None
            )
            action_batch = batch_inputs([histories.item(ep, idx) for ep, idx in mb_anchors], device, actions_only=True)
            logits = _windowed_target_logits(
                policy,
                histories,
                spans,
                len(u_nodes),
                torch.tensor([u_to_flat[k] for k in mb_anchors], device=device, dtype=torch.long),
                window_flat.to(device),
                ends,
                h_init,
                action_batch,
                device,
            )
            parent, _ = references.gather(mb_anchors, logits.shape[1])
            weights = torch.tensor(
                [anchor_counts[k] for k in mb_anchors], device=device, dtype=logits.dtype
            ) / len(anchor_keys)
            loss = (masked_kl_rows(parent, (logits / 0.5).log_softmax(-1), ~torch.isneginf(logits)) * weights).sum()
            loss.backward()
            kl_sum += loss.detach()

    norm = torch.nn.utils.clip_grad_norm_(
        [p for p in policy.parameters() if p.requires_grad],
        0.5,
        error_if_nonfinite=device.type != 'cuda',
    )
    if device.type == 'cuda':
        torch._assert_async(torch.isfinite(norm), 'Non-finite whole-turn gradient')
    optimizer.step()
    finite = torch.ones((), device=device, dtype=torch.bool)
    for flag in finite_state_flags(policy, optimizer).values():
        finite &= flag.to(device)
    values = torch.stack((pair_total, kl_sum, norm, finite.to(norm.dtype))).detach().cpu().tolist()
    if not values[-1] or not all(math.isfinite(v) for v in values[:-1]):
        raise FloatingPointError('Non-finite whole-turn optimizer attempt')
    return dict(
        pair_loss=values[0],
        anchor_kl=values[1],
        gradient_norm=values[2],
        scored_decisions=len(batch.keys),
        nonzero_comparisons=len(batch.signs),
    )


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
            if ep_idx[1] >= 128:
                collect_by_ep[ep_idx[0]].append(ep_idx)
                grouped.setdefault(ep_idx[0], set())
    by_cells = defaultdict(list)
    for episode, indices in grouped.items():
        max_idx = max(indices) if indices else 0
        for ep_idx in collect_by_ep.get(episode, ()):
            max_idx = max(max_idx, ep_idx[1] - 128)
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
                        collected_prefixes[key] = h[offset + key[1] - 128].detach().clone()
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


@torch.no_grad()
def reference_distributions(policy, histories, anchors):
    """Copy reference batches into pinned CPU storage with one final stream wait."""
    result = {}
    pending = []
    for part, logits in reference_logits_batches(policy, histories, anchors):
        logp = (logits / 0.5).log_softmax(-1)
        if logp.device.type == 'cuda':
            host = torch.empty(logp.shape, dtype=logp.dtype, device='cpu', pin_memory=True)
            host.copy_(logp, non_blocking=True)
        else:
            host = logp.cpu()
        pending.append((part, host))
    if pending and next(policy.parameters()).is_cuda:
        torch.cuda.current_stream(next(policy.parameters()).device).synchronize()
    for part, host in pending:
        for row, (key, count) in zip(host, part):
            result[key] = row[:count]
    return result


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
        raise ValueError('Whole-turn control requires FP32')
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
    return values[0], dict(
        anchor_argmax_changed=int(values[1]),
        anchor_argmax_changed_fraction=values[1] / len(anchors),
        outside_training_anchors=outside_count,
        outside_argmax_changed=int(values[2]),
        outside_argmax_changed_fraction=values[2] / outside_count if outside_count else None,
    )
