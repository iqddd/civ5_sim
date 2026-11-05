"""D1 transfer experiment: frozen original cases, fresh optimizer and schedules."""

from copy import deepcopy
from pathlib import Path

from civ5rl.protocols.common import advance_stage_state
from civ5rl.provenance import bound_cases, validate_cases
from civ5rl.provenance.common import (
    ROOT,
    file_sha256,
    load_verified_checkpoint,
    read_json,
    verify_self_digest,
    write_json,
)
from civ5sim.engine import digest


def is_d1(config):
    return config.get("diagnostic", {}).get("mode") == "D1-transfer"


def load_weights(policy, config, *, random_baseline=False):
    field = "random_checkpoint" if random_baseline else "initial_checkpoint"
    path = ROOT / config["diagnostic"][field]
    value, metadata = load_verified_checkpoint(
        policy,
        path,
        expected_update=0 if random_baseline else 6,
        source_error="Unexpected D1 source checkpoint",
        equality_error="D1 initialization parameters differ",
    )
    metadata["previous_decisions"] = value["state"]["decisions"]
    return metadata


def prepare_d1(directory, config):
    directory = Path(directory)
    source_path = ROOT / config["diagnostic"]["source_dataset"]
    source = read_json(source_path)
    verify_self_digest(source, error_msg="Original D1 dataset hash mismatch")
    dataset = {
        "schema": 1,
        "kind": "rl01.D1-transfer",
        "source_sha256": file_sha256(source_path),
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
    return advance_stage_state(
        state,
        passed,
        update_field="stage_updates",
        update_limit=8,
        success={"status": "passed", "reason": "D1_two_successive_evaluations"},
        failure={"status": "failed", "reason": "D1_update_limit"},
    )
