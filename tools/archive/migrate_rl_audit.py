"""Preserve model/Adam/RNG with a checked, explicit audit or pipeline revision."""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import torch

    from civ5rl import checkpoint
    from civ5rl.config import load_config
    from civ5rl.common import read_json, write_json
    from civ5rl.control import is_active
    from civ5rl.migration import validate_audit_migration, validate_pipeline_migration

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="configs/rl-v1.json")
    parser.add_argument("--kind", choices=("audit", "pipeline"), default="audit")
    args = parser.parse_args()
    source, directory = Path(args.checkpoint).resolve(), Path(args.out).resolve()
    parent = source.parent.parent
    if is_active(parent) or directory.exists():
        raise ValueError("Source must be stopped; destination must be a new directory")
    value = torch.load(source, map_location="cpu", weights_only=False)
    config, manifest = (
        load_config(args.config),
        read_json(parent / "corpus-manifest.json"),
    )
    with tempfile.TemporaryDirectory() as temporary:
        identity = checkpoint.passport(temporary, config, manifest)
    validate = (
        validate_pipeline_migration
        if args.kind == "pipeline"
        else validate_audit_migration
    )
    changes = validate(value["config"], config, value["identity"], identity)
    checkpoint.passport(directory, config, manifest)
    target = directory / "checkpoints"
    target.mkdir()
    lineage = {
        "kind": f"rl01.{args.kind}_migration",
        "created_utc": datetime.now(UTC).isoformat(),
        "parent_checkpoint": str(source),
        "parent_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "parent_identity": value["identity"],
        **changes,
        "unchanged": [
            "model",
            "optimizer",
            "all_rng",
            "state",
            "model_objective_precision_corpus_rules",
        ],
        "collection_conditions_changed": args.kind == "pipeline",
        "trajectory_identical_resume_claimed": False
        if args.kind == "pipeline"
        else None,
    }
    names = {source.name, "initial.pt", value["state"]["best_checkpoint"]}
    for name in names - {None}:
        if Path(name).name != name:
            raise ValueError("Checkpoint name must be a basename")
        saved = torch.load(source.parent / name, map_location="cpu", weights_only=False)
        if saved["identity"] != value["identity"] or saved["config"] != value["config"]:
            raise ValueError("Historical checkpoint identity differs")
        saved.update(config=config, identity=identity, migration=lineage)
        temporary = target / (name + ".tmp")
        torch.save(saved, temporary)
        os.replace(temporary, target / name)
    write_json(directory / "migration.json", lineage)
    write_json(directory / "latest.json", {"checkpoint": source.name, **value["state"]})
    shutil.copy2(parent / "execution.json", directory / "execution.json")
    for name in ("validation-manifest.json",):
        if (parent / name).exists():
            shutil.copy2(parent / name, directory / name)
    if (parent / "evaluation").exists():
        shutil.copytree(parent / "evaluation", directory / "evaluation")
    if (parent / "metrics.jsonl").exists():
        rows = [
            json.loads(line)
            for line in (parent / "metrics.jsonl").read_text().splitlines()
        ]
        with (directory / "metrics.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                if row["update"] <= value["state"]["update"]:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
    write_json(
        directory / "run-status.json", {"status": "ready_to_resume", **value["state"]}
    )
    # Reload the serialized result: preservation is checked, not inferred from
    # assigning the same Python references before torch.save.
    import numpy as np

    def same(a, b):
        if isinstance(a, torch.Tensor):
            return torch.equal(a, b)
        if isinstance(a, np.ndarray):
            return np.array_equal(a, b)
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
        if isinstance(a, (tuple, list)):
            return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
        return a == b

    restored = torch.load(target / source.name, map_location="cpu", weights_only=False)
    verified = {
        key: same(value[key], restored[key])
        for key in (
            "model",
            "optimizer",
            "python_rng",
            "numpy_rng",
            "torch_rng",
            "cuda_rng",
            "shuffle_rng",
            "state",
            "variant",
        )
    }
    if not all(verified.values()):
        raise RuntimeError(f"Migrated checkpoint state differs: {verified}")
    write_json(directory / "migration-verification.json", verified)
    print(json.dumps({"directory": str(directory), **changes}, indent=2))


if __name__ == "__main__":
    main()
