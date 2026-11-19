"""Bounded pairwise policy fitting on complete recorded public histories.

Targets are grouped by map size, at most HISTORY_BATCH per microbatch; repeated
pair/anchor targets share logits while their loss multiplicities are retained.
Each Adam step recomputes one no-grad prefix per episode through its latest BPTT
boundary. Hidden states are selected at each target's own boundary. Overlapping
spatial windows share one encoding within a microbatch, but GRU windows remain
independent and detach at their own boundaries. No activations survive a step.

Mixed spatial blocks use size 32 and CoalescedSpatial/foreach accumulation.
Activation checkpointing is disabled for the bounded eight-window working set.
Shared spatial VJPs and compilation can change BF16 rounding; historical
bitwise gradient equality is not a contract. Execution choices are recorded in
training_runtime for new runs.
"""

import json
import math
import random
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from civ5rl.core.agent import RLAgent
from civ5rl.core.distributions import masked_kl_rows
from civ5rl.core.model import batch_inputs
from civ5rl.engine.comparison.inputs import Histories, History
from civ5rl.engine.evaluation.executor import evaluation_precision
from civ5rl.engine.runtime.transactions import atomic_torch_save, finite_state_flags
from civ5rl.provenance.common import read_json, write_json
from civ5rl.provenance.rng import capture_rng, restore_rng

HISTORY_BATCH = 8
ATTENTION_BACKEND = "compact_fp32"


def training_runtime(cfg=None):
    if cfg and cfg.get("training", {}).get("mode") == "frozen_heads_step_kl":
        return dict(
            trainable=["action", "attack_preview"],
            trainable_parameters=68609,
            precision="deterministic_fp32_no_amp_no_tf32",
            feature_cache="frozen_public_histories_fp32",
            full_kl_backend="fp32_logsoftmax_fp64_cuda_reduction",
            compile_mode="default",
            cuda_graphs=False,
        )
    return dict(
        attention_backend=ATTENTION_BACKEND,
        history_batch=HISTORY_BATCH,
        spatial_batch=32,
        bptt=128,
        compile_mode="default",
        cuda_graphs=False,
        coalesced_spatial_gradients=True,
        activation_checkpointing=False,
        prefix_reuse="per_step_episode",
        spatial_window_reuse="per_microbatch_union",
        input_storage="verified_mmap_with_bounded_spatial_residency",
        loss_backend="batched_compiled",
        full_kl_backend="fp32_logsoftmax_fp64_cuda_reduction",
        precision="mixed_bf16_spatial_fp32_gru_heads_loss",
    )


def pair_loss(logits, better, worse, weight):
    return weight * F.softplus((logits[worse] - logits[better]) / 0.5)


def categorical_kl(reference, logits, *, double=False):
    # Both inputs contain only the target's real legal actions (no padding).
    current = (logits / 0.5).log_softmax(-1)
    if double:
        reference, current = reference.double(), current.double()
    return masked_kl_rows(reference, current)


class DeviceReferences(dict):
    """One immutable packed upload, with views for each legal-action distribution.

    Control references cache FP64 log-probabilities and probabilities on device;
    training references retain their original dtype. Repeated anchor keys are
    weighted by the caller rather than duplicated in this storage.
    """

    def __init__(self, references, device, *, control=False):
        keys = list(references)
        self.flat = (
            torch.cat([references[key] for key in keys]).to(device)
            if keys
            else torch.zeros(1, device=device)
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
        metadata = torch.tensor(
            [self.ranges.get(key, (0, 0)) for key in keys], device=self.flat.device
        )
        columns = torch.arange(width, device=self.flat.device)[None, :]
        mask = columns < metadata[:, 1:2]
        indices = (metadata[:, :1] + columns).clamp_max(self.flat.numel() - 1)
        logp = self.flat[indices].masked_fill(~mask, 0.0)
        probabilities = (
            self.probabilities[indices].masked_fill(~mask, 0.0)
            if self.probabilities is not None
            else None
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
            batch = batch_inputs(
                part, device, compact_units=policy.attention_backend == "compact_fp32"
            )
            x, context = policy.encode_spatial(batch)
            h, hidden = policy.sequence_memory(context, [len(part)], hidden)
        if not gradient:
            target = {k: v[-1:] for k, v in batch.items()}
            logits, _ = policy.score_heads(target, x[-1:], h[-1:])
            return logits[0, : len(history[index].sources)]
    hidden = None if hidden is None else hidden.detach()
    # Spatial execution partitions internally; retain one complete GRU BPTT window.
    part = history[boundary : index + 1]
    batch = batch_inputs(
        part, device, compact_units=policy.attention_backend == "compact_fp32"
    )
    x, context = policy.encode_spatial(batch)
    h, hidden = policy.sequence_memory(context, [len(part)], hidden)
    target = {k: v[-1:] for k, v in batch.items()}
    logits, _ = policy.score_heads(target, x[-1:], h[-1:])
    return logits[0, : len(history[index].sources)]


def configure_policy(checkpoint, device="cuda", mixed=True):
    policy = (
        RLAgent.from_checkpoint(checkpoint, "cpu", "argmax").policy.to(device).train()
    )
    policy.enable_amp(mixed, spatial_batch_size=32 if mixed else 0)
    policy.enable_inverse_routing()
    policy.enable_action_routing()
    policy.enable_maximum_routing()
    policy.configure_execution(
        activation_checkpointing=False, attention_backend=ATTENTION_BACKEND
    )
    if str(device).startswith("cuda"):
        # Fuse eager kernels without introducing private CUDA Graph pools.
        policy.enable_compile(dynamic=True, mode="default", spatial_mode="default")
    for parameter in policy.value.parameters():
        parameter.requires_grad_(False)
    return policy


def _history_groups(histories, keys, size=HISTORY_BATCH):
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


def _encode_histories(policy, sequences):
    device = next(policy.parameters()).device
    if all(
        isinstance(seq, History) and seq.owner is sequences[0].owner
        for seq in sequences
    ):
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


@torch.no_grad()
def prefix_states(policy, entries):
    """One current-weight no-grad GRU prefix per episode, scoped to one Adam step."""
    episodes = {}
    result = {}
    device = next(policy.parameters()).device
    zero = torch.zeros(1, 1, policy.hidden_size, device=device)
    for key, history in entries:
        boundary = max(0, key[1] + 1 - 128)
        if not boundary:
            result[key] = zero
            continue
        item = episodes.setdefault(key[0], dict(history=history, targets=[]))
        item["targets"].append((key, boundary))
    groups = {}
    for episode, item in episodes.items():
        item["episode"] = episode
        groups.setdefault(len(item["history"][0].cells), []).append(item)
    for group in groups.values():
        for first in range(0, len(group), HISTORY_BATCH):
            part = group[first : first + HISTORY_BATCH]
            sequences = [p["history"][: max(b for _, b in p["targets"])] for p in part]
            _, context = _encode_histories(policy, sequences)
            lengths = [len(seq) for seq in sequences]
            h, _ = policy.sequence_memory(context, lengths)
            keys = []
            indices = []
            offset = 0
            for item, length in zip(part, lengths):
                for key, boundary in item["targets"]:
                    keys.append(key)
                    indices.append(offset + boundary - 1)
                offset += length
            selected = h.index_select(0, torch.tensor(indices, device=device))
            result.update(
                (key, selected[i : i + 1].unsqueeze(0)) for i, key in enumerate(keys)
            )
    return result


def _window_features(policy, entries, boundaries):
    """Encode the union of overlapping spatial windows once, without sharing GRU graphs."""
    grouped = {}
    for (key, history), start in zip(entries, boundaries):
        item = grouped.setdefault(key[0], dict(history=history, intervals=[]))
        item["intervals"].append((start, key[1] + 1))
    segments = []
    lookup = {}
    for episode, item in grouped.items():
        merged = []
        for start, stop in sorted(item["intervals"]):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(stop, merged[-1][1]))
            else:
                merged.append((start, stop))
        lookup[episode] = []
        for start, stop in merged:
            lookup[episode].append((start, stop, len(segments)))
            segments.append(item["history"][start:stop])
    x, context = _encode_histories(policy, segments)
    lengths = [len(seq) for seq in segments]
    xs = x.split(lengths)
    contexts = context.split(lengths)
    windows = []
    targets = []
    for (key, _), boundary in zip(entries, boundaries):
        start, _, slot = next(
            row for row in lookup[key[0]] if row[0] <= boundary and key[1] < row[1]
        )
        windows.append(contexts[slot][boundary - start : key[1] + 1 - start])
        targets.append(xs[slot][key[1] - start])
    return torch.stack(targets), torch.cat(windows, 0)


def batched_target_logits(
    policy, entries, *, gradient=True, padded=False, prefixes=None
):
    """Independent current-weight histories; detach each one's own BPTT boundary."""
    keys, histories = zip(*entries)
    indices = [key[1] for key in keys]
    boundaries = [max(0, i + 1 - 128) if gradient else 0 for i in indices]
    hidden = None
    if gradient:
        if prefixes is None:
            prefixes = prefix_states(policy, entries)
        hidden = torch.cat([prefixes[key] for key in keys], 1)
    with torch.set_grad_enabled(gradient):
        x, context = _window_features(policy, entries, boundaries)
        lengths = [i + 1 - b for i, b in zip(indices, boundaries)]
        h, _ = policy.sequence_memory(context, lengths, hidden)
        ends = torch.tensor(
            [sum(lengths[: i + 1]) - 1 for i in range(len(lengths))], device=x.device
        )
        targets = [history[index] for history, index in zip(histories, indices)]
        batch = batch_inputs(targets, x.device, actions_only=True)
        logits, _ = policy.score_heads(batch, x, h[ends])
        if padded:
            return logits
        return [row[: len(target.sources)] for row, target in zip(logits, targets)]


def reference_logits_batches(policy, histories, anchors):
    grouped = {}
    for episode, index in anchors:
        grouped.setdefault(episode, set()).add(index)
    with torch.no_grad():
        for entries in _history_groups(
            histories, [(e, max(indices)) for e, indices in grouped.items()]
        ):
            sequences = [history[: key[1] + 1] for key, history in entries]
            x, context = _encode_histories(policy, sequences)
            lengths = [len(s) for s in sequences]
            h, _ = policy.sequence_memory(context, lengths)
            targets = []
            offset = 0
            for (key, history), length in zip(entries, lengths):
                targets.extend(
                    ((key[0], i), history[i], offset + i)
                    for i in sorted(grouped[key[0]])
                )
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


def reference_distributions(policy, histories, anchors):
    """Copy reference batches into pinned CPU storage with one final stream wait."""
    result = {}
    pending = []
    with torch.no_grad():
        for part, logits in reference_logits_batches(policy, histories, anchors):
            logp = (logits / 0.5).log_softmax(-1)
            if logp.device.type == "cuda":
                host = torch.empty(
                    logp.shape, dtype=logp.dtype, device="cpu", pin_memory=True
                )
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


def full_kl(policy, histories, anchors, references):
    """Reduce FP32 log-softmax differences in FP64 on device; read one scalar.

    Episode histories are encoded once per pass, with bounded spatial blocks.
    Duplicate anchors retain their multiplicity in both numerator and denominator.
    """
    device = next(policy.parameters()).device
    if (
        not isinstance(references, DeviceReferences)
        or references.probabilities is None
        or references.flat.device != device
    ):
        references = DeviceReferences(references, device, control=True)
    multiplicity = {}
    for key in map(tuple, anchors):
        multiplicity[key] = multiplicity.get(key, 0) + 1
    previous_amp, previous_batch = policy.amp_enabled, policy.spatial_batch_size
    policy.enable_amp(False)
    try:
        total = torch.zeros((), device=device, dtype=torch.float64)
        with evaluation_precision():
            for part, logits in reference_logits_batches(policy, histories, anchors):
                keys = [key for key, _ in part]
                parent, probabilities = references.gather(keys, logits.shape[1])
                values = control_runner(logits.is_cuda)(logits, parent, probabilities)
                counts = torch.tensor(
                    [multiplicity[key] for key in keys],
                    device=device,
                    dtype=torch.float64,
                )
                total += (values * counts).sum()
        return float(total / len(anchors))
    finally:
        policy.enable_amp(previous_amp, spatial_batch_size=previous_batch)


def prepare_references(parent, histories, anchors, path, *, budget=None):
    import hashlib

    started = perf_counter()
    histories.prepare(anchors, device=next(parent.parameters()).device, budget=budget)
    model_hash = hashlib.sha256()
    for name, value in parent.state_dict().items():
        model_hash.update(name.encode())
        model_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
    sources = {
        str(e): histories.source_identity(e) for e in sorted({k[0] for k in anchors})
    }
    identity = dict(
        parent=model_hash.hexdigest(),
        anchors=[list(k) for k in anchors],
        histories=sources,
        implementation_version=1,
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name() if next(parent.parameters()).is_cuda else None,
        modes=["deterministic_fp32_no_tf32", "mixed_bf16_spatial_fp32_gru"],
        temperature=0.5,
        block=32,
        attention_backend=parent.attention_backend,
        history_batch=HISTORY_BATCH,
    )
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
            result["_timing"] = dict(
                cache_hit=True, load_seconds=perf_counter() - started, cold_seconds=None
            )
            return runtime_references(result, next(parent.parameters()).device)
    result = {}
    with evaluation_precision():
        for mixed in (False, True):
            parent.enable_amp(mixed, spatial_batch_size=32 if mixed else 0)
            result["mixed" if mixed else "fp32"] = reference_distributions(
                parent, histories, anchors
            )
    seconds = perf_counter() - started
    atomic_torch_save(dict(identity=identity, distributions=result), path)
    result["_timing"] = dict(cache_hit=False, load_seconds=None, cold_seconds=seconds)
    return runtime_references(result, next(parent.parameters()).device)


def logical_step(policy, optimizer, histories, pairs, anchor_keys, references):
    """Accumulate batched preference/KL losses, then clip and apply one Adam step.

    CUDA gradient finiteness is asserted before Adam. Model/Adam finiteness and
    loss metrics share the post-step host readback; no finite check is skipped.
    """
    optimizer.zero_grad(set_to_none=True)
    pair_total = torch.zeros((), device=next(policy.parameters()).device)
    kl_total = torch.zeros_like(pair_total)
    pair_rows = {}
    anchor_counts = {}
    for row in pairs:
        pair_rows.setdefault((row["episode"], row["index"]), []).append(row)
    for key in map(tuple, anchor_keys):
        anchor_counts[key] = anchor_counts.get(key, 0) + 1
    keys = list(dict.fromkeys([*pair_rows, *anchor_counts]))
    prefixes = prefix_states(policy, [(key, histories.get(key[0])[0]) for key in keys])
    if not isinstance(references, DeviceReferences):
        references = DeviceReferences(references, pair_total.device)
    if any(key not in references for key in anchor_counts):
        raise ValueError("Missing anchor reference")
    for entries in _history_groups(histories, keys):
        logits = batched_target_logits(policy, entries, padded=True, prefixes=prefixes)
        pair_indices = []
        pair_weights = []
        weights = []
        for index, (key, _) in enumerate(entries):
            for row in pair_rows.get(key, ()):
                pair_indices.append((index, row["better"], row["worse"]))
                pair_weights.append(row["weight"] / len(pairs))
            weights.append(
                anchor_counts.get(key, 0) / len(anchor_keys) if anchor_keys else 0.0
            )
        parent, _ = references.gather(
            [key for key, _ in entries], logits.shape[1], allow_missing=True
        )
        indices = torch.tensor(
            pair_indices, device=logits.device, dtype=torch.long
        ).reshape(-1, 3)
        factors = torch.tensor(pair_weights, device=logits.device, dtype=logits.dtype)
        anchor_weights = torch.tensor(weights, device=logits.device, dtype=logits.dtype)
        pair, kl = loss_runner(logits.is_cuda)(
            logits, indices, factors, parent, anchor_weights
        )
        pair_total += pair.detach()
        kl_total += kl.detach()
        (pair + kl).backward()
    cuda = pair_total.is_cuda
    norm = torch.nn.utils.clip_grad_norm_(
        [p for p in policy.parameters() if p.requires_grad],
        0.5,
        error_if_nonfinite=not cuda,
    )
    if cuda:
        torch._assert_async(torch.isfinite(norm), "Non-finite gradient before Adam")
    optimizer.step()
    finite = torch.ones((), device=pair_total.device, dtype=torch.bool)
    for device, flag in finite_state_flags(policy, optimizer).items():
        if device == finite.device:
            finite &= flag
        elif not flag.item():
            raise FloatingPointError("Non-finite model/Adam state after attempt")
    measured = (
        torch.stack((pair_total, kl_total, norm, finite.to(norm.dtype)))
        .detach()
        .cpu()
        .tolist()
    )
    if not measured[-1] or not all(math.isfinite(v) for v in measured[:-1]):
        raise FloatingPointError("Non-finite loss/gradient/model/Adam after attempt")
    return dict(zip(("pair_loss", "anchor_kl", "gradient_norm"), measured[:-1]))


def fit(cfg, out, dataset, identity, budget=None):
    if budget is not None:
        return fit_with_storage(cfg, out, dataset, identity, budget)
    from civ5rl.engine.runtime.budget import PhaseBudget

    with PhaseBudget(out, "training", cfg["training_seconds"]) as budget:
        return fit_with_storage(cfg, out, dataset, identity, budget)


def fit_with_storage(cfg, out, dataset, identity, budget):
    from civ5rl.reporting.comparison_storage import archive_tree

    out = Path(out)
    state = read_json(out / "training.json") if (out / "training.json").exists() else {}
    if state.get("status") in ("complete", "no_accepted_pairs"):
        return _fit(cfg, out, dataset, identity, budget)
    result = _fit(cfg, out, dataset, identity, budget)
    if (out / "sources").exists():
        archive_tree(out, "sources")
    state = read_json(out / "training.json")
    state.update(
        seconds=budget.elapsed(),
        overshoot_seconds=max(0, budget.elapsed() - cfg["training_seconds"]),
    )
    write_json(out / "training.json", state)
    return result


def publish_candidate(cfg, out, state, identity):
    from civ5rl.provenance.common import file_sha256

    original = torch.load(cfg["parents"][0], map_location="cpu", weights_only=False)
    changed = any(
        not torch.equal(value.cpu(), original["model"][name])
        for name, value in state["model"].items()
    )
    atomic_torch_save(
        dict(
            variant="C2P",
            model=state["model"],
            optimizer=state["optimizer"],
            state={k: v for k, v in state.items() if k not in ("model", "optimizer")},
            config=original["config"],
            identity=identity,
        ),
        out / "candidate.pt",
    )
    report = dict(
        status="complete",
        training_steps=state["steps"],
        epoch=state["epoch"],
        cursor=state["cursor"],
        seconds=state["seconds"],
        overshoot_seconds=max(0, state["seconds"] - cfg["training_seconds"]),
        stop_reason=state["stop_reason"],
        changed=changed,
        attempted_final_kl=state.get("attempted_final_kl", state.get("last_full_kl")),
        input_cache=state.get("input_cache"),
        candidate_sha256=file_sha256(out / "candidate.pt"),
    )
    if state.get("mode") in ("full_corpus_step_kl", "frozen_heads_step_kl"):
        report.update(
            {
                k: state[k]
                for k in (
                    "mode",
                    "attempts",
                    "rejected_attempts",
                    "lr",
                    "anchor_cursor",
                    "last_full_kl",
                )
            }
        )
        for key in (
            "learning",
            "baseline_learning",
            "feature_parity",
            "inherited_steps",
            "inherited_seconds",
        ):
            if key in state:
                report[key] = state[key]
    write_json(out / "training.json", report)
    return out / "candidate.pt" if changed else None


def fit_identity(passport, dataset):
    from civ5sim.engine import digest

    return dict(passport=passport, dataset_hash=digest(dataset))


def prepare_fit(cfg, out, state, identity, *, empty_dataset=None):
    """Validate saved state and publish or verify a completed candidate."""
    from civ5rl.provenance.common import file_sha256
    from civ5rl.reporting.comparison_storage import restore_tree

    if state.get("identity", identity) != identity:
        raise ValueError("Training dataset/identity changed")
    if state.get("stop_reason"):
        report_path, candidate = out / "training.json", out / "candidate.pt"
        report = read_json(report_path) if report_path.exists() else {}
        if candidate.exists() and report.get("status") == "complete":
            if file_sha256(candidate) != report["candidate_sha256"]:
                raise ValueError("Candidate checkpoint changed")
            return True, candidate if report["changed"] else None
        return True, publish_candidate(cfg, out, state, identity)
    if empty_dataset is not None and not empty_dataset["pairs"]:
        write_json(
            out / "training.json",
            dict(
                status="no_accepted_pairs",
                training_steps=0,
                **{
                    k: empty_dataset.get(k)
                    for k in (
                        "selected_forks",
                        "completed_forks",
                        "accepted_pairs",
                        "rejections",
                    )
                },
            ),
        )
        return True, None
    restore_tree(out, "sources")
    return False, None


def prepare_histories(out, dataset, policy, budget, *, cache_dir=None):
    pairs, anchors = dataset["pairs"], dataset["anchors"]
    keys = [(p["episode"], p["index"]) for p in pairs] + [tuple(k) for k in anchors]
    limits = {}
    for episode, index in keys:
        limits[episode] = max(limits.get(episode, 0), index + 1)
    histories = Histories(
        out,
        source_hashes=dataset.get("source_hashes"),
        limits=limits,
        cache_dir=cache_dir,
    )
    histories.prepare(keys, device=next(policy.parameters()).device, budget=budget)
    return histories


def _fit(cfg, out, dataset, identity, budget):
    if cfg.get("training", {}).get("mode") in (
        "full_corpus_step_kl",
        "frozen_heads_step_kl",
    ):
        from civ5rl.engine.comparison.control import fit_full_corpus

        return fit_full_corpus(cfg, out, dataset, identity, budget)
    out = Path(out)
    state_path = out / "training.pt"
    from civ5rl.engine.comparison.control import ComparisonTrainingState

    state = (
        ComparisonTrainingState.from_checkpoint_dict(
            torch.load(state_path, map_location="cpu", weights_only=False)
        )
        if state_path.exists()
        else ComparisonTrainingState(order=[])
    )
    pairs = dataset["pairs"]
    anchors = dataset["anchors"]
    identity = fit_identity(identity, dataset)
    done, result = prepare_fit(
        cfg, out, state.to_checkpoint_dict(), identity, empty_dataset=dataset
    )
    if done:
        return result
    policy = configure_policy(cfg["parents"][0])
    optimizer = torch.optim.Adam(
        [p for p in policy.parameters() if p.requires_grad],
        lr=1e-5,
        betas=(0.9, 0.999),
        eps=1e-5,
        weight_decay=0,
    )
    shuffle = random.Random(cfg["seed"] + 101)
    histories = prepare_histories(out, dataset, policy, budget)
    # Policy still has parent weights. Reuse it for frozen references before
    # installing the student's resumed state; a second CUDA model is unnecessary.
    references = prepare_references(
        policy, histories, anchors, out / "references.pt", budget=budget
    )
    if "model" in state.metadata:
        policy.load_state_dict(state.metadata["model"])
        optimizer.load_state_dict(state.metadata["optimizer"])
        shuffle.setstate(state.metadata["shuffle"])
    if "rng" in state.metadata:
        restore_rng(state.metadata["rng"])

    def save(path, *copies):
        state.metadata.update(
            model=policy.state_dict(),
            optimizer=optimizer.state_dict(),
            shuffle=shuffle.getstate(),
            rng=capture_rng(),
            identity=identity,
            input_cache=histories.statistics(),
        )
        state.seconds = budget.elapsed()
        frozen = atomic_torch_save(state.to_checkpoint_dict(), path, *copies)
        write_json(
            out / "training.json",
            {
                k: frozen[k]
                for k in ("epoch", "cursor", "steps", "seconds", "input_cache")
            },
        )
        return frozen

    if not (out / "verified.pt").exists():
        save(out / "verified.pt")
    stop = False
    while state.epoch < 2 and not stop:
        if state.cursor >= len(pairs):
            kl = full_kl(policy, histories, anchors, references["fp32"])
            if kl > 0.015:
                stop = True
                break
            state.epoch += 1
            state.cursor = 0
            state.order = []
            state.last_kl = kl
            save(out / "verified.pt", state_path)
            continue
        if budget.expired():
            break
        if not state.order:
            state.order = list(range(len(pairs)))
            shuffle.shuffle(state.order)
        selected = state.order[state.cursor : state.cursor + 32]
        anchor_batch = []
        for _ in range(2 * len(selected)):
            if state.anchor_cursor >= len(state.anchor_order):
                state.anchor_order = list(range(len(anchors)))
                shuffle.shuffle(state.anchor_order)
                state.anchor_cursor = 0
            anchor_batch.append(anchors[state.anchor_order[state.anchor_cursor]])
            state.anchor_cursor += 1
        step_started = perf_counter()
        metrics = logical_step(
            policy,
            optimizer,
            histories,
            [pairs[i] for i in selected],
            anchor_batch,
            references["mixed"],
        )
        state.record_epoch_step(len(selected))
        save(state_path)
        with (out / "steps.jsonl").open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    dict(
                        step=state.steps,
                        seconds=state.seconds,
                        step_seconds=perf_counter() - step_started,
                        **metrics,
                    )
                )
                + "\n"
            )
    if not stop:
        kl = (
            state.last_kl
            if state.cursor == 0 and state.last_kl is not None
            else full_kl(policy, histories, anchors, references["fp32"])
        )
    if stop or kl > 0.015:
        state = ComparisonTrainingState.from_checkpoint_dict(
            torch.load(out / "verified.pt", map_location="cpu", weights_only=False)
        )
        policy.load_state_dict(state.metadata["model"])
        optimizer.load_state_dict(state.metadata["optimizer"])
        shuffle.setstate(state.metadata["shuffle"])
        restore_rng(state.metadata["rng"])
        state.stop_reason = "kl_limit"
    else:
        state.stop_reason = "epochs_completed" if state.epoch == 2 else "time_limit"
    state.attempted_final_kl = kl
    frozen = save(state_path)
    return publish_candidate(cfg, out, frozen, identity)


def preference_and_kl(logits, pair_indices, pair_weights, reference, anchor_weights):
    # Rows with no pair are omitted; duplicates retain every original weight.
    pair_logits = logits.index_select(0, pair_indices[:, 0])
    better = pair_logits.gather(1, pair_indices[:, 1:2]).squeeze(1)
    worse = pair_logits.gather(1, pair_indices[:, 2:3]).squeeze(1)
    pair = (pair_weights * F.softplus((worse - better) / 0.5)).sum()
    logq = (logits / 0.5).log_softmax(-1)
    kl = (
        masked_kl_rows(reference, logq, ~torch.isneginf(logits)) * anchor_weights
    ).sum()
    return pair, kl


def full_kl_rows(logits, reference, probabilities):
    # Retain FP32 log-softmax, followed by the established FP64 subtraction/exp.
    # Padding is -inf in logits, but must contribute zero without 0*inf NaNs.
    logq = (logits / 0.5).log_softmax(-1).double()
    return masked_kl_rows(
        reference, logq, ~torch.isneginf(logits), probabilities=probabilities
    )


@lru_cache(maxsize=2)
def loss_runner(cuda):
    return (
        torch.compile(preference_and_kl, dynamic=True, fullgraph=True, mode="default")
        if cuda
        else preference_and_kl
    )


@lru_cache(maxsize=2)
def control_runner(cuda):
    return (
        torch.compile(full_kl_rows, dynamic=True, fullgraph=True, mode="default")
        if cuda
        else full_kl_rows
    )
