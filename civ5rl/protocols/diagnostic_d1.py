"""D1 transfer experiment: frozen original cases, fresh optimizer and schedules."""

import hashlib
from copy import deepcopy
from pathlib import Path

from civ5rl.provenance import bound_cases, validate_cases
from civ5rl.provenance.common import ROOT, read_json, write_json
from civ5sim.engine import digest


def is_d1(config):
    return config.get("diagnostic", {}).get("mode") == "D1-transfer"


def model_hash(model):
    value = hashlib.sha256()
    for name, tensor in sorted(model.items()):
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def load_weights(policy, config, *, random_baseline=False):
    import torch

    field = "random_checkpoint" if random_baseline else "initial_checkpoint"
    path = ROOT / config["diagnostic"][field]
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value["variant"] != "C" or value["state"]["update"] != (
        0 if random_baseline else 6
    ):
        raise ValueError("Unexpected D1 source checkpoint")
    policy.load_state_dict(value["model"])
    if not all(
        torch.equal(v.cpu(), value["model"][k]) for k, v in policy.state_dict().items()
    ):
        raise ValueError("D1 initialization parameters differ")
    return {
        "source_checkpoint": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "model_tensors_sha256": model_hash(value["model"]),
        "all_model_parameters_bitwise_equal": True,
        "source_optimizer_loaded": False,
        "source_training_rng_loaded": False,
        "previous_decisions": value["state"]["decisions"],
    }


def prepare_d1(directory, config):
    directory = Path(directory)
    source_path = ROOT / config["diagnostic"]["source_dataset"]
    source = read_json(source_path)
    if source["hash"] != digest({k: v for k, v in source.items() if k != "hash"}):
        raise ValueError("Original D1 dataset hash mismatch")
    dataset = {
        "schema": 1,
        "kind": "rl01.D1-transfer",
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "preparation": deepcopy(config["diagnostic"]),
        "D1": deepcopy(source["D1"]),
    }
    for split in ("train", "dev"):
        manifest = dataset["D1"][split]
        validate_cases(bound_cases(manifest), manifest)
        if manifest["split"] != "diagnostic_" + split:
            raise ValueError("D1 source split differs")
        if any("learner_decision_limit" in s for s in manifest["cases"]):
            raise ValueError("D1 cannot have a diagnostic decision limit")
    train, dev = (dataset["D1"][s]["cases"] for s in ("train", "dev"))
    from civ5rl.protocols.diagnostic import physical_hash

    for spec in [*train, *dev]:
        if physical_hash(spec["scenario"]) != spec["physical_hash"]:
            raise ValueError("D1 physical hash differs")
    if {s["physical_hash"] for s in train} & {s["physical_hash"] for s in dev}:
        raise ValueError("D1 train/dev overlap")
    if (
        len(dev) != 128
        or len({(s["layout_id"], s["learner_side"]) for s in dev}) != 128
    ):
        raise ValueError("D1 dev must contain 64 complete paired layouts")
    if len({s["physical_hash"] for s in dev}) != 64:
        raise ValueError("D1 dev layout count differs")
    dataset["hash"] = digest(dataset)
    path = directory / "manifest.json"
    if path.exists() and read_json(path) != dataset:
        raise ValueError("D1 dataset identity mismatch")
    write_json(path, dataset)
    write_json(
        directory / "construction-verification.json",
        {
            "source_sha256": dataset["source_sha256"],
            "train_cases": len(train),
            "dev_cases": len(dev),
            "original_manifests_exactly_equal": True,
            "train_dev_disjoint": True,
            "no_one_decision_limit": True,
        },
    )
    return dataset


def next_state_d1(state, passed):
    value = deepcopy(state)
    value["consecutive"] = value.get("consecutive", 0) + 1 if passed else 0
    if value["consecutive"] >= 2:
        value.update(status="passed", reason="D1_two_successive_evaluations")
    elif value["stage_updates"] >= 8:
        value.update(status="failed", reason="D1_update_limit")
    return value
