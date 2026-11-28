"""Shared encoded prefixes plus immutable branch tails; no duplicated battle tensors."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy

import torch

from civ5rl.core.context import context_from_spec
from civ5rl.core.observation import ObservationEncoder
from civ5rl.engine.comparison.inputs import Histories, History
from civ5rl.provenance.common import file_sha256, read_json
from civ5sim import Action
from civ5sim.engine import digest


def _encode_and_remember(encoder, row):
    observation = row["observation"]
    legal = [Action(**action) for action in row["actions"]]
    encoded = encoder.encode(observation, legal)
    encoder.remember_action(legal[row["chosen"]], observation)
    return encoded


class TurnStorage(Histories):
    def __init__(self, out, sources, roots, branches, anchors, device):
        limits = {}
        for episode, index in anchors:
            limits[episode] = max(limits.get(episode, 0), index + 1)
        for branch in branches.values():
            source_episode = branch["episode"]
            limits[source_episode] = max(limits.get(source_episode, 0), branch["prefix"])
        super().__init__(out, limits=limits, device=device)
        self.source_root, self.roots, self.branches = sources, roots, branches
        self.prefix_encoder = None
        self.prefix_key = None
        self.history_rows = OrderedDict()

    def source_folder(self, episode):
        if episode in self.branches:
            branch = self.branches[episode]
            return self.roots[branch["packet"]] / branch["folder"]
        return self.source_root / str(episode)

    def source_identity(self, episode):
        if episode not in self.branches:
            return super().source_identity(episode)
        if episode not in self.sources:
            branch = self.branches[episode]
            self.sources[episode] = {
                "prefix": super().source_identity(branch["episode"]),
                "prefix_length": branch["prefix"],
                "tail": file_sha256(self.source_folder(episode) / "turn-history.json"),
                "encoding_version": 2,
            }
        return self.sources[episode]

    def source_rows(self, episode):
        if episode not in self.history_rows:
            self.history_rows[episode] = read_json(self.source_root / str(episode) / "history.json")
            if len(self.history_rows) > 2:
                self.history_rows.popitem(last=False)
        self.history_rows.move_to_end(episode)
        return self.history_rows[episode]

    def turn_record(self, episode):
        """Load and validate the public turn and its recorded outcome together."""
        expected = self.branches[episode]
        folder = self.source_folder(episode)
        rows = read_json(folder / "turn-history.json")
        completed = read_json(folder / "completed.json")
        branch = read_json(folder / "branch.json")
        if (
            branch["position_key"] != expected["key"]
            or branch["command"] != expected["command"]
            or branch["turn_variant"] != expected["variant"]
            or branch["repetition"] != expected["repetition"]
        ):
            raise ValueError("Branch does not match fixed training manifest")
        if not completed["prefix_verified"] or completed["position_key"] != expected["key"]:
            raise ValueError("Unverified branch prefix")
        original = self.source_rows(expected["episode"])[expected["prefix"]]
        if original["command"] != expected["command"] or rows[0]["command"] != expected["command"]:
            raise ValueError("Own-observation prefix boundary differs")
        if digest(rows[0]["observation"]) != digest(original["observation"]):
            raise ValueError("Branch has a different public initial observation")
        return rows, int(completed["winner"] == completed["side"])

    def tail_rows(self, episode):
        return self.turn_record(episode)[0]

    def encoded_rows(self, episode):
        if episode not in self.branches:
            return super().encoded_rows(episode)
        branch = self.branches[episode]
        source_episode, prefix_length = branch["episode"], branch["prefix"]
        key = (source_episode, prefix_length)
        if key != self.prefix_key:
            case = read_json(self.source_root / str(source_episode) / "case.json")
            seed = case["agent_seeds"][case["learner_side"]]
            encoder = ObservationEncoder(context_from_spec(case, seed=seed), "C2P")
            for row in self.source_rows(source_episode)[:prefix_length]:
                _encode_and_remember(encoder, row)
            self.prefix_key, self.prefix_encoder = key, encoder

        encoder = deepcopy(self.prefix_encoder)
        return [_encode_and_remember(encoder, row) for row in self.tail_rows(episode)]


class TurnHistories:
    """Expose complete branch histories while sharing stored source-prefix tensors."""

    def __init__(self, out, sources, roots, branches, anchors, device="cuda"):
        self.storage = TurnStorage(out, sources, roots, branches, anchors, device)
        self.branches, self.records, self.targets = branches, {}, {}

    def _make_record(self, episode):
        self.storage.get(episode)
        record = self.storage.records[episode]
        branch = self.branches.get(episode)
        if branch is None:
            return record
        prefix_length = branch["prefix"]
        fields = ("counts", "units")
        if not prefix_length:
            return {key: record[key] for key in fields}
        source_episode = branch["episode"]
        self.storage.get(source_episode)
        prefix = self.storage.records[source_episode]
        return {key: torch.cat((prefix[key][:prefix_length], record[key])) for key in fields}

    def get(self, episode):
        if episode not in self.records:
            self.records[episode] = self._make_record(episode)
        return History(self, episode, 0, len(self.records[episode]["counts"])), None

    def item(self, episode, index):
        if episode in self.branches:
            branch = self.branches[episode]
            if index < branch["prefix"]:
                return self.storage.item(branch["episode"], index)
            index -= branch["prefix"]
        return self.storage.item(episode, index)

    def action_count(self, episode, index):
        self.get(episode)
        return int(self.records[episode]["counts"][index])

    def source_identity(self, episode):
        return self.storage.source_identity(episode)

    def spatial_inputs(self, sequences, device):
        spans = []
        for sequence in sequences:
            branch = self.branches.get(sequence.episode)
            if branch is None:
                spans.append(History(self.storage, sequence.episode, sequence.start, sequence.stop))
                continue
            boundary = branch["prefix"]
            if sequence.start < boundary:
                spans.append(
                    History(self.storage, branch["episode"], sequence.start, min(sequence.stop, boundary))
                )
            if sequence.stop > boundary:
                spans.append(
                    History(
                        self.storage,
                        sequence.episode,
                        max(0, sequence.start - boundary),
                        sequence.stop - boundary,
                    )
                )
        return self.storage.spatial_inputs(spans, device)

    def prepare(self, keys, *, device=None, budget=None):
        for episode in dict.fromkeys(episode for episode, _ in keys):
            if budget is not None and budget.expired():
                raise TimeoutError("Whole-turn preparation budget exhausted")
            self.get(episode)
            self.storage._place(episode)
            branch = self.branches.get(episode)
            if branch is not None and branch["prefix"]:
                self.storage._place(branch["episode"])

    def statistics(self):
        return self.storage.statistics()

    def comparison_targets(self, first_episode, second_episode, first_rows, second_rows):
        """Score only free actions after the shared command prefix."""
        common = 0
        for first, second in zip(first_rows, second_rows):
            if first["actions"][first["chosen"]] != second["actions"][second["chosen"]]:
                break
            if digest(first["observation"]) != digest(second["observation"]):
                raise ValueError("Equal prefix commands have different public observations")
            common += 1

        result = []
        for episode, rows in ((first_episode, first_rows), (second_episode, second_rows)):
            keys = []
            for index, row in enumerate(rows[common:], start=common):
                if len(row["actions"]) > 1:
                    key = (episode, self.branches[episode]["prefix"] + index)
                    self.targets[key] = row["chosen"]
                    keys.append(key)
            result.append(keys)
        return result
