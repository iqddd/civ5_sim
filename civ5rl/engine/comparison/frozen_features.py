"""Store verified FP32 public-history features in the shared mmap cache."""

from __future__ import annotations

from collections import defaultdict

import torch

from civ5rl.engine.comparison import PREFERENCE_TEMPERATURE
from civ5rl.engine.comparison.reference import encode_histories
from civ5rl.engine.runtime.transactions import atomic_torch_save
from civ5rl.provenance.common import file_sha256, read_json, write_json
from civ5sim.engine import digest

FEATURE_CACHE_VERSION = 1
ACTION_FIELDS = ("sources", "targets", "action_features", "attack_preview")


def feature_positions(dataset):
    return sorted(
        {(pair["episode"], pair["index"]) for pair in dataset["pairs"]} | set(map(tuple, dataset["anchors"]))
    )


class FrozenFeatureCache:
    """One packet per episode with legal-action tensors and unexpanded positions."""

    def __init__(self, histories, device, attention_backend, statistics):
        self.histories = histories
        self.device = torch.device(device)
        self.attention_backend = attention_backend
        self.statistics = statistics
        self.directory = histories.directory.parent / "head-features-fp32"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.records = {}
        self.locations = {}

    @torch.no_grad()
    def prepare(self, checkpoint, keys, get_policy, *, verify_record=None, budget=None):
        grouped = defaultdict(list)
        for episode, index in keys:
            grouped[episode].append(index)
        common = {
            "parent": file_sha256(checkpoint),
            "implementation": FEATURE_CACHE_VERSION,
            "encoder": self.histories.encoder_identity,
            "torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu",
            "attention_backend": self.attention_backend,
            "precision": "fp32_no_amp_no_tf32",
            "temperature": PREFERENCE_TEMPERATURE,
            "spatial_block": 32,
            "history_batch": 1,
        }
        for episode, indices in grouped.items():
            identity = {
                "common": common,
                "source": self.histories.source_identity(episode),
                "indices": indices,
            }
            path = self.directory / f"{episode}-{digest(identity)}.pt"
            receipt = path.with_suffix(".json")
            if path.exists() and receipt.exists():
                saved = read_json(receipt)
                if saved["identity"] != identity or saved["sha256"] != file_sha256(path):
                    raise ValueError("Frozen feature cache changed")
                self.statistics["cache_hits"] += 1
            else:
                policy = get_policy()
                history, _ = self.histories.get(episode)
                record = atomic_torch_save(self._record(policy, history, episode, indices, identity), path)
                if verify_record is not None:
                    verify_record(policy, history, indices[0], record)
                write_json(receipt, {"identity": identity, "sha256": file_sha256(path)})
                self.statistics["cache_builds"] += 1
            record = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            if record["identity"] != identity:
                raise ValueError("Frozen feature cache identity mismatch")
            self.records[episode] = record
            self.statistics["bytes"] += path.stat().st_size
            for row, index in enumerate(indices):
                self.locations[episode, index] = (episode, row)
            if budget is not None and budget.expired():
                raise TimeoutError("Training budget expired during frozen-head feature preparation")

    def _record(self, policy, history, episode, indices, identity):
        sequence = history[: max(indices) + 1]
        x, context = encode_histories(policy, [sequence])
        h, _ = policy.sequence_memory(context, [len(sequence)])
        selected = torch.tensor(indices, device=self.device)
        items = [history[index] for index in indices]
        offsets = [0]
        for item in items:
            offsets.append(offsets[-1] + len(item.sources))
        raw = self.histories.raw(episode)
        return {
            "identity": identity,
            "x": x[selected],
            "h": h[selected],
            "actions": {
                name: torch.cat([torch.from_numpy(getattr(item, name)) for item in items])
                for name in ACTION_FIELDS
            },
            "offsets": offsets,
            "chosen": [raw[index]["chosen"] for index in indices],
        }
