"""Public-history encoding and legal-action reference distributions."""

from __future__ import annotations

import hashlib
from time import perf_counter

import torch

from civ5rl.core.model import batch_inputs
from civ5rl.engine.comparison import PREFERENCE_TEMPERATURE
from civ5rl.engine.comparison.inputs import History
from civ5rl.engine.evaluation.runtime import evaluation_precision
from civ5rl.engine.runtime.transactions import atomic_torch_save

HISTORY_BATCH = 8


class DeviceReferences(dict):
    """One immutable packed upload, with views for each legal-action distribution.

    Control references cache FP64 log-probabilities and probabilities on device;
    training references retain their original dtype. Repeated anchor keys are
    weighted by the caller rather than duplicated in this storage.
    """

    def __init__(self, references, device, *, control=False):
        keys = list(references)
        self.flat = (
            torch.cat([references[key] for key in keys]).to(device) if keys else torch.zeros(1, device=device)
        )
        if control:
            self.flat = self.flat.double()
        self.probabilities = self.flat.exp() if control else None
        self.ranges = {}
        offset = 0
        for key in keys:
            count = len(references[key])
            self[key] = self.flat[offset : offset + count]
            self.ranges[key] = (offset, count)
            offset += count

    def gather(self, keys, width, *, allow_missing=False):
        if not allow_missing and any(key not in self.ranges for key in keys):
            raise ValueError("Missing parent reference distribution")
        metadata = torch.tensor([self.ranges.get(key, (0, 0)) for key in keys], device=self.flat.device)
        columns = torch.arange(width, device=self.flat.device)[None, :]
        mask = columns < metadata[:, 1:2]
        indices = (metadata[:, :1] + columns).clamp_max(self.flat.numel() - 1)
        logp = self.flat[indices].masked_fill(~mask, 0.0)
        probabilities = (
            self.probabilities[indices].masked_fill(~mask, 0.0) if self.probabilities is not None else None
        )
        return logp, probabilities


def runtime_references(saved, device):
    return {
        **saved,
        "mixed": DeviceReferences(saved["mixed"], device),
        "fp32": DeviceReferences(saved["fp32"], device, control=True),
    }


def target_logits(policy, history, index, *, gradient=False, block=32):
    """Whole current-weight prefix; only final <=128 decisions retain a graph."""
    device = next(policy.parameters()).device
    boundary = max(0, index + 1 - 128) if gradient else index + 1
    hidden = None
    with torch.no_grad():
        for start in range(0, boundary, block):
            part = history[start : min(boundary, start + block)]
            batch = batch_inputs(part, device, compact_units=policy.attention_backend == "compact_fp32")
            x, context = policy.encode_spatial(batch)
            h, hidden = policy.sequence_memory(context, [len(part)], hidden)
        if not gradient:
            target = {k: v[-1:] for k, v in batch.items()}
            logits, _ = policy.score_heads(target, x[-1:], h[-1:])
            return logits[0, : len(history[index].sources)]
    hidden = None if hidden is None else hidden.detach()
    # Spatial execution partitions internally; retain one complete GRU BPTT window.
    part = history[boundary : index + 1]
    batch = batch_inputs(part, device, compact_units=policy.attention_backend == "compact_fp32")
    x, context = policy.encode_spatial(batch)
    h, hidden = policy.sequence_memory(context, [len(part)], hidden)
    target = {k: v[-1:] for k, v in batch.items()}
    logits, _ = policy.score_heads(target, x[-1:], h[-1:])
    return logits[0, : len(history[index].sources)]


def history_groups(histories, keys, size=HISTORY_BATCH):
    # Maps with different cell counts need not pad one another to the largest map.
    grouped = {}
    for key in sorted(dict.fromkeys(map(tuple, keys))):
        history, _ = histories.get(key[0])
        cells = len(history[0].cells)
        entries = grouped.setdefault(cells, [])
        entries.append((key, history))
        if len(entries) == size:
            yield entries
            grouped[cells] = []
    for entries in grouped.values():
        if entries:
            yield entries


def encode_histories(policy, sequences):
    device = next(policy.parameters()).device
    if all(isinstance(seq, History) and seq.owner is sequences[0].owner for seq in sequences):
        inputs = sequences[0].owner.spatial_inputs(sequences, device)
        if policy.attention_backend != "compact_fp32":
            inputs.pop("unit_indices")
    else:
        items = [item for sequence in sequences for item in sequence]
        inputs = batch_inputs(
            items,
            device,
            spatial_only=True,
            compact_units=policy.attention_backend == "compact_fp32",
        )
    if torch.is_grad_enabled() and policy.spatial_batch_size:
        # Use the executor so cell padding and precision match the ordinary path.
        return policy.encode_spatial(inputs, coalesced=True)
    # FP32 full-KL has no mixed-padding setting. Keep its spatial activations
    # bounded too: an entire group of histories is not one attention matrix.
    encoded = [
        policy.encode_spatial({k: v[start : start + 32] for k, v in inputs.items()})
        for start in range(0, len(inputs["cells"]), 32)
    ]
    return tuple(torch.cat([row[i] for row in encoded], 0) for i in (0, 1))


def reference_logits_batches(policy, histories, anchors):
    grouped = {}
    for episode, index in anchors:
        grouped.setdefault(episode, set()).add(index)
    with torch.no_grad():
        for entries in history_groups(histories, [(e, max(indices)) for e, indices in grouped.items()]):
            sequences = [history[: key[1] + 1] for key, history in entries]
            x, context = encode_histories(policy, sequences)
            lengths = [len(s) for s in sequences]
            h, _ = policy.sequence_memory(context, lengths)
            targets = []
            offset = 0
            for (key, history), length in zip(entries, lengths):
                targets.extend(((key[0], i), history[i], offset + i) for i in sorted(grouped[key[0]]))
                offset += length
            for start in range(0, len(targets), 32):
                part = targets[start : start + 32]
                selected = torch.tensor([p[2] for p in part], device=x.device)
                logits, _ = policy.score_heads(
                    batch_inputs([p[1] for p in part], x.device, actions_only=True),
                    x[selected],
                    h[selected],
                )
                yield [(key, len(item.sources)) for key, item, _ in part], logits


def reference_distributions(policy, histories, anchors, *, logits_batches=reference_logits_batches):
    """Copy reference batches into pinned CPU storage with one final stream wait."""
    result = {}
    pending = []
    with torch.no_grad():
        for part, logits in logits_batches(policy, histories, anchors):
            logp = (logits / PREFERENCE_TEMPERATURE).log_softmax(-1)
            if logp.device.type == "cuda":
                host = torch.empty(logp.shape, dtype=logp.dtype, device="cpu", pin_memory=True)
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


def prepare_references(parent, histories, anchors, path, *, budget=None):
    started = perf_counter()
    histories.prepare(anchors, device=next(parent.parameters()).device, budget=budget)
    model_hash = hashlib.sha256()
    for name, value in parent.state_dict().items():
        model_hash.update(name.encode())
        model_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
    sources = {str(e): histories.source_identity(e) for e in sorted({k[0] for k in anchors})}
    identity = {
        "parent": model_hash.hexdigest(),
        "anchors": [list(k) for k in anchors],
        "histories": sources,
        "implementation_version": 1,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name() if next(parent.parameters()).is_cuda else None,
        "modes": ["deterministic_fp32_no_tf32", "mixed_bf16_spatial_fp32_gru"],
        "temperature": PREFERENCE_TEMPERATURE,
        "block": 32,
        "attention_backend": parent.attention_backend,
        "history_batch": HISTORY_BATCH,
    }
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        saved_identity = saved.get("identity", {})
        # Older reference caches used source-file hashes. Source movement does
        # not change their tensor semantics; all scientific inputs still match.
        comparable = dict(saved_identity)
        old_implementation = comparable.pop("implementation", None)
        if old_implementation is not None:
            comparable["implementation_version"] = 1
        distributions = saved.get("distributions", {})
        expected_keys = set(map(tuple, anchors))
        counts = {key: histories.action_count(*key) for key in expected_keys}
        valid = comparable == identity and all(
            isinstance(distributions.get(mode), dict)
            and set(distributions[mode]) == expected_keys
            and all(
                torch.is_tensor(row)
                and row.ndim == 1
                and row.numel() == counts[key]
                and torch.isfinite(row).all()
                for key, row in distributions[mode].items()
            )
            for mode in ("mixed", "fp32")
        )
        if valid:
            result = distributions
            result["_timing"] = {
                "cache_hit": True,
                "load_seconds": perf_counter() - started,
                "cold_seconds": None,
            }
            return runtime_references(result, next(parent.parameters()).device)
    result = {}
    with evaluation_precision():
        for mixed in (False, True):
            parent.enable_amp(mixed, spatial_batch_size=32 if mixed else 0)
            result["mixed" if mixed else "fp32"] = reference_distributions(parent, histories, anchors)
    seconds = perf_counter() - started
    atomic_torch_save({"identity": identity, "distributions": result}, path)
    result["_timing"] = {"cache_hit": False, "load_seconds": None, "cold_seconds": seconds}
    return runtime_references(result, next(parent.parameters()).device)
