"""Parent-world splits, exact scenarios and explicit RNG stream manifests."""

import random
from pathlib import Path

from civ5map.maps import components, content_hash, load, save, select_crops
from civ5rl.core.context import context_for  # noqa: F401 -- historical public import
from civ5rl.core.contracts import ScenarioSpec
from civ5rl.core.observation import distance
from civ5rl.provenance.common import ROOT, seed_for, write_json
from civ5sim.engine import UNIT_TYPES, Battle, digest


def map_path(value):
    """Manifest paths are workspace-relative; accept archived corpus locations."""
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    legacy = ROOT / "artifacts/rl/corpus-v1"
    if not path.exists() and path.is_relative_to(legacy):
        path = ROOT / "artifacts/rl/corpus" / path.relative_to(legacy)
    return path


def manifest_path(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def prepare_world(task):
    game, directory, split, seed = task
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    world_path = directory / "world.json"
    try:
        if world_path.exists():
            world = load(world_path)
        else:
            from civ5map.runtime import generate

            world = generate(Path(game), seed, 3, (2, 2, 2, 2, 2))
            save(world, world_path)
        entries = []
        for size in (10, 12, 16):
            crops = select_crops(world, count=4, seed=seed, size=size)
            for i, crop in enumerate(crops):
                path = directory / f"crop-{size}-{i}.json"
                save(crop, path)
                entries.append(
                    {
                        "path": manifest_path(path),
                        "size": size,
                        "parent_seed": seed,
                        "parent_hash": content_hash(world),
                        "crop_hash": digest(crop),
                        "split": split,
                    }
                )
        return {"accepted": True, "split": split, "seed": seed, "maps": entries}
    except (ValueError, AttributeError) as exc:
        # Some original-script seeds fail in the existing Lua/Python adapter.
        # Preserve these as explicit generation failures, never as valid worlds.
        import traceback

        return {
            "accepted": False,
            "split": split,
            "seed": seed,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
        }


def prepare_corpus(config, directory, game, workers=4):
    from concurrent.futures import ProcessPoolExecutor

    directory = Path(directory).resolve()
    entries, rejections = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for split, spec in config["corpus"]["parent_worlds"].items():
            needed, next_seed = spec["count"], spec["first_seed"]
            ceiling = {"train": 20000, "validation": 30000, "test": 2**32}[split]
            accepted = 0
            while accepted < needed:
                seeds = list(
                    range(next_seed, next_seed + min(workers, needed - accepted))
                )
                if seeds[-1] >= ceiling:
                    raise RuntimeError("Parent-world reserve exhausted")
                tasks = [
                    (str(game), str(directory / split / str(s)), split, s)
                    for s in seeds
                ]
                for result in pool.map(prepare_world, tasks):
                    if result["accepted"]:
                        entries.extend(result["maps"])
                        accepted += 1
                    else:
                        rejections.append(result)
                    print(
                        f"corpus {split} {accepted}/{needed}: seed {result['seed']} accepted={result['accepted']}",
                        flush=True,
                    )
                next_seed = seeds[-1] + 1
    manifest = {
        "schema": 1,
        "kind": "rl01.corpus",
        "generation": config["corpus"],
        "maps": entries,
        "rejections": rejections,
    }
    validate_corpus(manifest)
    write_json(directory / "manifest.json", manifest)
    return manifest


def validate_corpus(manifest):
    owners = {}
    for entry in manifest["maps"]:
        for key in ("parent_seed", "parent_hash", "crop_hash"):
            identity = (key, entry[key])
            if identity in owners and owners[identity] != entry["split"]:
                raise ValueError(f"Corpus split leakage: {identity}")
            owners[identity] = entry["split"]
        m = load(map_path(entry["path"]))
        if (
            digest(m) != entry["crop_hash"]
            or m["wrap_x"]
            or m["width"] != entry["size"]
        ):
            raise ValueError("Corpus hash or dimensions changed")
        group = components(m)[0]
        if len(group) < 0.65 * entry["size"] ** 2:
            raise ValueError("Disconnected crop")
        zones = [{y * m["width"] + x for x, y in z} for z in m["spawn_candidates"]]
        if any(len(z) < 10 or not z <= group for z in zones):
            raise ValueError("Invalid spawn component")


class ScenarioFactory:
    def __init__(self, config, manifest):
        self.config, self.manifest = config, manifest
        self.maps = {}
        self._issued = set()
        self.manifest_hash = digest(manifest)

    def validate_specs(self, specs):
        for spec in specs:
            if digest(spec) not in self._issued:
                raise ValueError("Training case was not issued by this trusted factory")

    def get_map(self, entry):
        path = entry["path"]
        if path not in self.maps:
            self.maps[path] = load(map_path(path))
        return self.maps[path]

    @staticmethod
    def _sample_roster(rng, unit_bounds):
        n = rng.randint(*unit_bounds)
        while True:
            roster = [rng.choice(UNIT_TYPES) for _ in range(n)]
            if any(k in ("archer", "chariot_archer") for k in roster) and any(
                k in ("warrior", "spearman", "horseman") for k in roster
            ):
                return roster

    @staticmethod
    def _sample_positions(m, group, n, contact, rng, size):
        if not contact:
            return [
                rng.sample([y * size + x for x, y in zone], n)
                for zone in m["spawn_candidates"]
            ]
        a = rng.choice(group)
        anchors = [t for t in group if distance(m, a, t) == 2]
        if not anchors:
            return None
        b = rng.choice(anchors)
        pools = [
            [t for t in group if distance(m, anchor, t) <= 2 and t not in (a, b)]
            for anchor in (a, b)
        ]
        if min(map(len, pools)) < n - 1:
            return None
        pos0 = [a, *rng.sample(pools[0], n - 1)]
        pool1 = [t for t in pools[1] if t not in pos0]
        if len(pool1) < n - 1:
            return None
        pos1 = [b, *rng.sample(pool1, n - 1)]
        rng.shuffle(pos0)
        rng.shuffle(pos1)
        return [pos0, pos1]

    @staticmethod
    def _select_opponent(weights, rng):
        pick = rng.random()
        opponent = next(reversed(weights))
        for name, weight in weights.items():
            pick -= weight
            if pick < 0:
                opponent = name
                break
        return opponent

    def make(
        self,
        episode_id,
        stage,
        *,
        master=None,
        namespace="train",
        split="train",
        case_index=None,
    ) -> ScenarioSpec:
        if split not in ("train", "validation"):
            raise ValueError(f"Closed corpus split: {split}")
        master = self.config["seed"] if master is None else master
        streams = {
            s: seed_for(master, episode_id, s, namespace=namespace)
            for s in ("map", "roster", "placement", "battle", "opponent")
        }
        map_rng, roster_rng, placement_rng = [
            random.Random(streams[s]) for s in ("map", "roster", "placement")
        ]
        size = stage["map_size"][0]
        candidates = [
            m
            for m in self.manifest["maps"]
            if m["split"] == split and m["size"] == size
        ]
        if not candidates:
            raise ValueError(f"No {split} maps of size {size}")
        # Evaluation evenly covers worlds, then crops, independent of model seed.
        if case_index is None:
            first = map_rng.randrange(len(candidates))
        else:
            worlds = sorted({m["parent_seed"] for m in candidates})
            world = worlds[case_index % len(worlds)]
            options = [m for m in candidates if m["parent_seed"] == world]
            first = candidates.index(
                options[(case_index // len(worlds)) % len(options)]
            )
        roster = self._sample_roster(roster_rng, stage["units_per_side"])
        n = len(roster)
        contact = placement_rng.random() < stage.get("contact_probability", 0)
        rejected = []
        for offset in range(len(candidates)):
            entry = candidates[(first + offset) % len(candidates)]
            m = self.get_map(entry)
            group = sorted(components(m)[0])
            for attempt in range(self.config["corpus"]["placement_rejection_limit"]):
                positions = self._sample_positions(
                    m, group, n, contact, placement_rng, size
                )
                if positions is None:
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
                        {"type": k, "side": s, "tile": t}
                        for s in (0, 1)
                        for k, t in zip(roster, positions[s])
                    ],
                }
                with Battle(scenario) as battle:
                    contacts = [
                        any(u["side"] != s for u in battle.observe(s)["units"])
                        for s in (0, 1)
                    ]
                if (contact and all(contacts)) or (not contact and not any(contacts)):
                    opponents = stage.get("opponents", {"greedy": 1})
                    opponent = self._select_opponent(
                        opponents, random.Random(streams["opponent"])
                    )
                    spec = {
                        "episode_id": episode_id,
                        "scenario": scenario,
                        "armies": [roster, roster.copy()],
                        "max_rounds": stage["max_rounds"],
                        "max_calls_per_turn": self.config["environment"][
                            "max_calls_per_turn"
                        ],
                        "learner_side": random.Random(
                            seed_for(
                                master, episode_id, "learner_side", namespace=namespace
                            )
                        ).randrange(2),
                        "opponent": opponent,
                        "streams": streams,
                        "parent_world": entry["parent_seed"],
                        "placement": "contact" if contact else "separated",
                        "rejected": rejected,
                        "agent_seeds": [
                            seed_for(
                                master, episode_id, "learner_sampling", s, namespace
                            )
                            for s in (0, 1)
                        ],
                        "split": split,
                        "namespace": namespace,
                        "source": "rl01.corpus",
                        "scenario_hash": digest(scenario),
                        "manifest_hash": self.manifest_hash,
                    }
                    if split not in ("train", "validation"):
                        raise ValueError(f"Closed corpus split: {split}")
                    self._issued.add(digest(spec))
                    return spec
            rejected.append(
                {"map": entry["path"], "reason": "placement_rejection_limit"}
            )
            if case_index is not None:
                raise ValueError(
                    f"Evaluation placement failed for assigned world/crop: {entry['path']}"
                )
        raise RuntimeError("No constructible scenario in corpus")
