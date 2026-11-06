"""Fixed target-scale sampling: public scenarios, no outcome selection."""

import itertools
import random
from collections import Counter
from copy import deepcopy

from civ5map.maps import components, load
from civ5rl.core.observation import distance
from civ5rl.protocols.corpus import ScenarioFactory, map_path
from civ5rl.protocols.common import (
    CoverageTracker,
    own_state_physical_hash,
    self_digest,
)
from civ5rl.config import load_config as resolve_config
from civ5rl.provenance.common import ROOT, read_json, seed_for
from civ5sim import Battle
from civ5sim.engine import digest

KIND = "rl01.ppo_target_scale"
SIZES = (10, 12, 16)
BINS = ((5, 7), (8, 10), (11, 13))
SEPARATED_VERSION = "native_visibility_regions_v2"


def separated_candidates(m, group, n, rng, conflicts):
    """Preferred zones first, then new compact shapes throughout the component.

    None is a bounded candidate rejected by capacity. All returned positions
    still require the final full-army Battle observation check.
    """
    width = m["width"]
    zones = [[y * width + x for x, y in zone] for zone in m["spawn_candidates"]]
    used = 0
    if all(len(z) >= n for z in zones):
        for _ in range(64):
            used += 1
            yield [rng.sample(z, n) for z in zones], "preferred"
    anchors = [sorted(group, key=lambda t: (t not in zones[s], t)) for s in (0, 1)]
    for attempt in range(10000 - used):
        first_side = attempt % 2
        visit = attempt // 2
        anchor = anchors[first_side][visit % len(group)]
        ranked = sorted(group, key=lambda t: (distance(m, anchor, t), t))
        cycle = visit // len(group)
        if cycle:
            pool = ranked[: min(len(group), n + cycle * 2)]
            first = rng.sample(pool, n)
        else:
            first = ranked[:n]
        if len(first) < n:
            yield None, "capacity"
            continue
        forbidden = set().union(*(conflicts[t] for t in first))
        available = [t for t in group if t not in forbidden]
        if len(available) < n:
            yield None, "compatible_capacity"
            continue
        # The other region follows available terrain; it need not be circular.
        other = max(
            available,
            key=lambda t: (distance(m, anchor, t), t in zones[1 - first_side], -t),
        )
        second_ranked = sorted(available, key=lambda t: (distance(m, other, t), t))
        second = (
            rng.sample(second_ranked[: min(len(available), n + cycle * 2)], n)
            if cycle
            else second_ranked[:n]
        )
        positions = [None, None]
        positions[first_side], positions[1 - first_side] = first, second
        for row in positions:
            rng.shuffle(row)
        yield positions, "revised"


class InvalidDataset(RuntimeError):
    def __init__(self, details):
        self.details = details
        super().__init__(
            f"invalid_dataset: {details['map']}, n={details['army_size']}, "
            f"contact={details['contact']}; {details['rejections']}"
        )


def load_config(path):
    config = resolve_config(path)
    validate_profile(config)
    return config


def validate_profile(config):
    expected = read_json(ROOT / "configs/rl-ppo-target-scale.json")
    reference = resolve_config(ROOT / expected["base_config"])
    for section in (
        "observation",
        "environment",
        "model",
        "reward",
        "optimizer",
        "numerics",
        "teacher_search",
        "seed",
        "workers",
    ):
        if config[section] != reference[section]:
            raise ValueError(f"Unsupported target-scale {section}")
    if config["kind"] != KIND or config["output"] != expected["output"]:
        raise ValueError("Unsupported target-scale identity")
    for key, value in expected["experiment"].items():
        if config["experiment"][key] != value:
            raise ValueError(f"Unsupported target-scale experiment.{key}")
    collection = deepcopy(reference["collection"])
    envs = config["collection"]["num_envs"]
    workers = config["collection"]["cpu_workers"]
    if (
        envs not in (4, 8, 16, 24, 32, 64)
        or workers not in (4, 6, 8, 16, 32)
        or envs % workers
    ):
        raise ValueError("Unsupported number of target-scale environments")
    collection.update(
        num_envs=envs, cpu_workers=workers, envs_per_worker=envs // workers
    )
    if config["collection"] != collection:
        raise ValueError("Unsupported target-scale collection")
    ppo = deepcopy(reference["ppo"])
    micro = config["ppo"]["sequences_per_microbatch"]
    if micro not in (1, 2, 4, 8):
        raise ValueError("Unsupported target-scale microbatch")
    ppo["sequences_per_microbatch"] = micro
    if config["ppo"] != ppo:
        raise ValueError("Unsupported target-scale PPO")
    performance = {**reference["performance"], **expected["performance"]}
    performance["worker_reply_size"] = min(4, envs // workers)
    performance["kl_sequences_per_batch"] = micro
    if config["performance"] != performance:
        raise ValueError("Unsupported target-scale performance")


class TargetFactory:
    def __init__(self, config, base):
        self.config, self.base = config, base
        self.maps = {}
        self.visibility_conflicts = {}
        self.issued = set()
        self._coverage_tracker = CoverageTracker()
        self.physical = self._coverage_tracker.physical
        self.conditions = self._coverage_tracker.conditions
        self.worlds = self._coverage_tracker.worlds
        self.crops = self._coverage_tracker.crops

    @property
    def duplicates(self):
        return self._coverage_tracker.duplicates

    def _get_visibility_conflicts(self, key, m, group, battle_seed):
        if key not in self.visibility_conflicts:
            visibility = {}
            for tile in group:
                with Battle(
                    {
                        "schema": 1,
                        "kind": "civ5sim.scenario",
                        "map": m,
                        "fog": True,
                        "random_damage": True,
                        "seed": battle_seed,
                        "units": [
                            {"type": "warrior", "side": 0, "tile": tile, "hp": 100}
                        ],
                    }
                ) as battle:
                    visibility[tile] = battle.observe(0)["visibility"]
            self.visibility_conflicts[key] = {
                a: {b for b in group if a == b or visibility[a][b] or visibility[b][a]}
                for a in group
            }
        return self.visibility_conflicts[key]

    def case(
        self,
        index,
        world,
        size,
        bin_index,
        contact,
        side,
        namespace,
        *,
        split="train",
        crop=None,
        n=None,
    ):
        if (
            split not in ("train", "validation")
            or world
            not in self.base["worlds"]["training" if split == "train" else "validation"]
        ):
            raise ValueError("Closed or untrusted target-scale world")
        entries = sorted(
            (
                e
                for e in self.base["maps"]
                if e["split"] == split
                and e["parent_seed"] == world
                and e["size"] == size
            ),
            key=lambda e: e["path"],
        )
        if not entries:
            raise ValueError(f"Missing crop: {world}/{size}")
        entry = entries[(index // 57 if crop is None else crop) % len(entries)]
        if entry["path"] not in self.maps:
            self.maps[entry["path"]] = load(map_path(entry["path"]))
        m = self.maps[entry["path"]]
        group = sorted(components(m)[0])
        streams = {
            key: seed_for(17, index, key, namespace=namespace)
            for key in ("roster", "placement", "battle", "opponent")
        }
        roster = ScenarioFactory._sample_roster(
            random.Random(streams["roster"]), BINS[bin_index] if n is None else (n, n)
        )
        n = len(roster)
        rng = random.Random(streams["placement"])
        if not contact:
            key = entry["path"]
            conflicts = self._get_visibility_conflicts(key, m, group, streams["battle"])
            separated = separated_candidates(m, group, n, rng, conflicts)
        reasons = Counter()
        scenario = None
        for attempt in range(10000):
            radius = 2 + min(attempt // 500, 3)
            if contact:
                positions = ScenarioFactory.sample_contact_positions(
                    m, group, n, rng, radius=radius, reasons=reasons
                )
                if positions is None:
                    continue
            else:
                positions, placement_method = next(separated)
                if positions is None:
                    reasons[placement_method] += 1
                    continue
            scenario = {
                "schema": 1,
                "kind": "civ5sim.scenario",
                "map": m,
                "fog": True,
                "random_damage": True,
                "seed": streams["battle"],
                "active_side": 0,
                "units": [
                    {"type": kind, "side": s, "tile": tile, "hp": 100}
                    for s in (0, 1)
                    for kind, tile in zip(roster, positions[s])
                ],
            }
            with Battle(scenario) as battle:
                seen = [
                    any(u["side"] != s for u in battle.observe(s)["units"])
                    for s in (0, 1)
                ]
            if seen != [bool(contact), bool(contact)]:
                reasons["visibility"] += 1
                continue
            spec = {
                "episode_id": index,
                "scenario": scenario,
                "armies": [roster, roster.copy()],
                "learner_side": side,
                "max_rounds": 30,
                "max_calls_per_turn": self.config["environment"]["max_calls_per_turn"],
                "opponent": "cp",
                "opponent_search": deepcopy(self.config["teacher_search"]),
                "reward_profile": "win_probability",
                "agent_seeds": [
                    seed_for(17, index, "learner_sampling", s, namespace)
                    for s in (0, 1)
                ],
                "streams": streams,
                "parent_world": world,
                "placement": "contact" if contact else "separated",
                "map_size": size,
                "army_size": n,
                "army_bin": bin_index,
                "map_entry": entry,
                "placement_attempts": attempt + 1,
                "placement_radius": radius,
                "rejections": dict(reasons),
                "placement_algorithm": "contact_v1" if contact else SEPARATED_VERSION,
                "placement_method": "contact" if contact else placement_method,
                "spawn_zones_revised": not contact and placement_method == "revised",
                "source": KIND,
                "split": split,
                "namespace": namespace,
                "manifest_hash": self.base["hash"],
            }
            spec["scenario_hash"] = digest(scenario)
            return spec
        raise InvalidDataset(
            {
                "map": entry["path"],
                "parent_world": world,
                "army_size": n,
                "roster": roster,
                "contact": bool(contact),
                "attempts": 10000,
                "rejections": dict(reasons),
                "namespace": namespace,
                "index": index,
                "streams": streams,
                "placement_algorithm": "contact_v1" if contact else SEPARATED_VERSION,
                "last_scenario": scenario,
            }
        )

    def make(self, index, unused=None):
        relative = index - self.config["experiment"]["start_episode"]
        strata = list(itertools.product(SIZES, range(3), (False, True), (0, 1))) * 4
        namespace = self.config["experiment"]["training_namespace"]
        random.Random(
            seed_for(17, relative // 144, "balance", namespace=namespace)
        ).shuffle(strata)
        size, category, contact, side = strata[relative % 144]
        worlds = self.base["worlds"]["training"]
        spec = self.case(
            index,
            worlds[relative % len(worlds)],
            size,
            category,
            contact,
            side,
            namespace,
            crop=relative // len(worlds),
        )
        self.issued.add(digest(spec))
        h = own_state_physical_hash(spec["scenario"])
        self._coverage_tracker.record(
            h,
            f"{size}/{category}/{int(contact)}/{side}",
            str(spec["parent_world"]),
            spec["map_entry"]["crop_hash"],
        )
        return spec

    def validate_specs(self, specs):
        if any(
            digest(spec) not in self.issued or spec["split"] != "train"
            for spec in specs
        ):
            raise ValueError("Untrusted target-scale training case")

    def summary(self):
        return self._coverage_tracker.summary()


def evaluation_cases(config, base):
    factory, used = TargetFactory(config, base), set()
    result = {}
    # Every world has equal contact/separated counts. Nine size/bin cells rotate
    # across worlds, so marginal cell counts differ by at most one physical layout.
    for purpose, per_world, shift in (("development", 4, 0), ("confirmation", 8, 4)):
        cases = []
        for wi, world in enumerate(base["worlds"]["validation"]):
            for j in range(per_world):
                cell = (wi * per_world + j + shift) % 9
                size, category = SIZES[cell // 3], cell % 3
                index = wi * per_world + j
                spec = factory.case(
                    index,
                    world,
                    size,
                    category,
                    j % 2,
                    0,
                    f"ppo_target_scale_{purpose}_v1",
                    split="validation",
                    crop=(j + shift) % 4,
                )
                h = own_state_physical_hash(spec["scenario"])
                if h in used:
                    raise ValueError("Duplicate target validation physical layout")
                used.add(h)
                for side in (0, 1):
                    case = {
                        **deepcopy(spec),
                        "case_id": index,
                        "layout_id": index,
                        "learner_side": side,
                        "physical_hash": h,
                    }
                    case["case_hash"] = self_digest(case)
                    cases.append(case)
        result[purpose] = cases
    return result
