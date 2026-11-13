"""Read-only checks of saved PPO and BC ledgers and checkpoints."""

import hashlib
import io
import json
import math
import random
from pathlib import Path

import numpy as np

from civ5rl.provenance.common import read_json, write_json


def verify_ppo_ledger(directory, *, offsets=None):
    """Check committed updates against state. Offsets describe earlier history."""
    directory = Path(directory)
    state = read_json(directory / "state.json")
    config_path = directory / "run.json"
    if not config_path.exists():
        config_path = directory / "resolved-config.json"
    run = read_json(config_path)
    config = run.get("config", run)
    threshold = config.get("ppo", {}).get("stop_kl", 0.015)
    if (
        not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or threshold < 0
    ):
        raise ValueError("Invalid KL threshold")
    offsets = {
        "updates": 0,
        "decisions": 0,
        "accepted": 0,
        "attempted": 0,
        "rejected": 0,
        **(offsets or {}),
    }
    path = directory / "updates.jsonl"
    rows = (
        [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if path.exists()
        else []
    )
    totals = dict(offsets)
    adam_origin = run.get("parent_Adam_counter", config.get("experiment", {}).get("parent_optimizer_steps", 0))
    previous_ids = {
        "update": offsets["updates"],
        "rollout": offsets.get("rollouts", offsets["updates"]),
    }
    for row in rows:
        for key in ("update", "rollout"):
            if key in row:
                value = row[key]
                if (
                    type(value) is not int
                    or value < 1
                    or value != previous_ids[key] + 1
                ):
                    raise ValueError(f"Out-of-sequence {key} identifier")
                previous_ids[key] = value
        ppo = row["ppo"]
        records = ppo.get("optimizer_step_records")
        skipped = ppo.get("PPO_skipped") is True
        if (
            records is None
            and skipped
            and all(
                ppo.get(k) == 0
                for k in (
                    "optimizer_steps",
                    "attempted_optimizer_steps",
                    "rejected_optimizer_steps",
                )
            )
            and ppo.get("accepted_optimizer_steps", 0) == 0
        ):
            records = []
        if records is None:
            raise ValueError(
                "Optimizer step records unavailable; KL cannot be verified"
            )
        if skipped and records:
            raise ValueError("Skipped PPO update has optimizer step records")
        if any("accepted" not in r or "attempted_kl" not in r for r in records):
            raise ValueError(
                "Legacy non-transactional step records cannot verify KL acceptance"
            )
        accepted = sum(bool(r["accepted"]) for r in records)
        attempted = len(records)
        rejected = attempted - accepted
        for record in records:
            if record["accepted"] and (
                not math.isfinite(record["attempted_kl"])
                or record["attempted_kl"] > threshold
            ):
                raise ValueError("Accepted step exceeds KL limit or is non-finite")
        for key, actual in (
            ("accepted_optimizer_steps", accepted),
            ("attempted_optimizer_steps", attempted),
            ("rejected_optimizer_steps", rejected),
            ("optimizer_steps", accepted),
        ):
            if key in ppo and ppo[key] != actual:
                raise ValueError(f"PPO {key} differs from step records")
        totals["updates"] += not skipped
        totals["decisions"] += row["decisions"]
        totals["accepted"] += accepted
        totals["attempted"] += attempted
        totals["rejected"] += rejected
        for key, name in (
            ("total_decisions", "decisions"),
            ("total_Adam_steps", "accepted"),
            ("total_attempted_steps", "attempted"),
            ("total_rejected_steps", "rejected"),
        ):
            if key in row and row[key] != totals[name]:
                raise ValueError(f"Ledger {key} inconsistent with totals")
    pending = state.get("phase") in ("ppo", "critic")
    if "Adam_counter_total" in state and state["Adam_counter_total"] != adam_origin + state["optimizer_steps"]:
        raise ValueError("Total Adam counter differs from inherited + new accepted steps")
    for name, key in (
        ("updates", "updates"),
        ("accepted", "optimizer_steps"),
        ("attempted", "attempted_steps"),
        ("rejected", "rejected_steps"),
        ("decisions", "decisions"),
    ):
        if key in state and state[key] != totals[name]:
            if (
                name == "decisions"
                and pending
                and state[key] > totals[name]
                and state.get("pending_rollout")
            ):
                continue
            raise ValueError(f"State {key} differs from committed ledger")
    result = {
        "status": "pending" if pending else "verified",
        "totals": totals,
        "offsets": offsets,
        "stop_kl": threshold,
        "pending_decisions": state.get("decisions", 0) - totals["decisions"],
    }
    from civ5rl.engine.ppo_adam import uses_grouped_adam
    if uses_grouped_adam(config):
        from civ5rl.engine.ppo_adam import expected_counters
        expected = expected_counters(state)
        if state.get("Adam_counters") != expected:
            raise ValueError("Grouped Adam counters differ from preparation + PPO")
        result["adam"] = dict(inherited=state.get("origin_Adam_counters", adam_origin), new_accepted=totals["accepted"],
                              preparation=state.get("critic_progress", {}).get("steps", 0),
                              counters=expected)
    elif adam_origin:
        result["adam"] = {"inherited": adam_origin, "new_accepted": totals["accepted"],
                          "total": adam_origin + totals["accepted"]}
    return result


def verify_bc_ledger(directory):
    """Check BC collection, steps.jsonl, state.json, and training.json accounting."""
    directory = Path(directory)
    state = read_json(directory / "state.json")
    collection = read_json(directory / "collection.json")
    if collection.get("status") != "complete":
        raise ValueError("BC collection is not complete")
    steps_path = directory / "steps.jsonl"
    rows = (
        [
            json.loads(line)
            for line in steps_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if steps_path.exists()
        else []
    )
    for index, row in enumerate(rows, start=1):
        if row.get("step") != index:
            raise ValueError(f"Out-of-sequence BC step {row.get('step')} at line {index}")
        if (
            not math.isfinite(float(row.get("ce", float("nan"))))
            or not math.isfinite(float(row.get("gradient_norm", float("nan"))))
            or int(row.get("labels", 0)) <= 0
        ):
            raise ValueError(f"Invalid BC step record at step {index}")
    total_steps = len(rows)
    total_labels = sum(int(r["labels"]) for r in rows)
    if state.get("Adam_steps") != total_steps:
        raise ValueError("State Adam_steps differs from steps.jsonl count")
    if state.get("free_labels_processed") != total_labels:
        raise ValueError("State free_labels_processed differs from steps.jsonl sum")
    if (directory / "training.json").exists():
        training = read_json(directory / "training.json")
        if training.get("Adam_steps") != total_steps:
            raise ValueError("training.json Adam_steps differs from steps.jsonl")
        if training.get("free_labels_processed") != total_labels:
            raise ValueError(
                "training.json free_labels_processed differs from steps.jsonl"
            )
    return {
        "status": "verified",
        "Adam_steps": total_steps,
        "free_labels_processed": total_labels,
        "corpus_labels": collection.get("labels"),
    }


def _equal(a, b):
    import torch

    if isinstance(a, torch.Tensor):
        return (
            isinstance(b, torch.Tensor)
            and a.dtype == b.dtype
            and a.shape == b.shape
            and torch.equal(a, b)
        )
    if isinstance(a, np.ndarray):
        return (
            isinstance(b, np.ndarray)
            and a.dtype == b.dtype
            and a.shape == b.shape
            and np.array_equal(a, b)
        )
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and a.keys() == b.keys()
            and all(_equal(a[k], b[k]) for k in a)
        )
    if isinstance(a, (list, tuple)):
        return (
            type(a) is type(b)
            and len(a) == len(b)
            and all(_equal(x, y) for x, y in zip(a, b, strict=True))
        )
    return a == b


def verify_checkpoint_chain(directory, *, adam_offset=0, parent_checkpoint=None):
    """Check initial parent equality and CPU checkpoint serialization/RNG streams."""
    import torch

    from civ5rl.core.model import Policy

    directory = Path(directory)
    points = sorted(
        p
        for p in (directory / "checkpoints").glob("*.pt")
        if not p.name.startswith("reference-")
    )
    if not points:
        raise ValueError("No checkpoints")
    checked = []
    for path in points:
        value = torch.load(path, map_location="cpu", weights_only=False)
        for key in (
            "model",
            "optimizer",
            "state",
            "python_rng",
            "numpy_rng",
            "torch_rng",
            "cuda_rng",
            "shuffle_rng",
        ):
            if key not in value:
                raise ValueError(f"Missing {key} in {path}")
        buffer = io.BytesIO()
        torch.save(value, buffer)
        buffer.seek(0)
        restored = torch.load(buffer, map_location="cpu", weights_only=False)
        if not _equal(value, restored):
            raise ValueError(f"Checkpoint roundtrip differs: {path}")
        random.Random().setstate(value["python_rng"])
        random.Random().setstate(value["shuffle_rng"])
        np.random.RandomState().set_state(value["numpy_rng"])
        torch.Generator(device="cpu").set_state(value["torch_rng"].cpu())
        if not isinstance(value["cuda_rng"], list) or any(
            not isinstance(x, torch.Tensor)
            or x.dtype != torch.uint8
            or x.ndim != 1
            or x.numel() == 0
            for x in value["cuda_rng"]
        ):
            raise ValueError(f"Invalid CUDA RNG states: {path}")
        caller_rng = torch.get_rng_state()
        try:
            policy = Policy(value["variant"])
        finally:
            torch.set_rng_state(caller_rng)
        policy.load_state_dict(value["model"], strict=True)
        parameters = list(policy.parameters())
        non_value_ids = [
            i
            for i, (name, _) in enumerate(policy.named_parameters())
            if not name.startswith("value.")
        ]
        groups = value["optimizer"]["param_groups"]
        ids = [index for group in groups for index in group["params"]]
        from civ5rl.engine.ppo_adam import uses_grouped_adam
        adapting = uses_grouped_adam(value.get("config", {}))
        if adapting:
            named = dict(policy.named_parameters())
            if [g.get("name") for g in groups] != ["shared_actor", "value"]:
                raise ValueError(f"Expected shared_actor/value Adam groups: {path}")
            lookup = {}
            for group in groups:
                expected_names = [n for n in named if n.startswith("value.") == (group["name"] == "value")]
                if group.get("parameter_names") != expected_names:
                    raise ValueError(f"Adam parameter names differ: {path}")
                lookup.update(zip(group["params"], [named[n] for n in expected_names], strict=True))
            if len(ids) != len(set(ids)) or len(lookup) != len(named):
                raise ValueError(f"Incomplete or repeated Adam parameters: {path}")
        elif len(groups) != 1 or ids not in (
            list(range(len(parameters))),
            non_value_ids,
        ):
            raise ValueError(f"Expected one sequential Adam parameter group: {path}")
        if adapting:
            pass
        elif ids == non_value_ids:
            lookup = dict(zip(range(len(non_value_ids)), [parameters[i] for i in non_value_ids], strict=True))
        else:
            lookup = dict(zip(ids, parameters, strict=True))
        cpu_groups = [
            {"params": [lookup[index] for index in group["params"]]}
            if all(index in lookup for index in group["params"])
            else {"params": [parameters[index] for index in group["params"]]}
            for group in groups
        ]
        optimizer = torch.optim.Adam(cpu_groups, fused=False)
        optimizer.load_state_dict(value["optimizer"])
        if not _equal(policy.state_dict(), value["model"]) or not _equal(
            optimizer.state_dict(), value["optimizer"]
        ):
            raise ValueError(f"Model/Adam CPU load differs: {path}")
        for parameter, item in optimizer.state.items():
            for name in ("exp_avg", "exp_avg_sq"):
                moment = item.get(name)
                if (
                    not isinstance(moment, torch.Tensor)
                    or moment.shape != parameter.shape
                    or moment.dtype != parameter.dtype
                ):
                    raise ValueError(
                        f"Adam {name} shape or dtype differs from parameter: {path}"
                    )
            if "max_exp_avg_sq" in item:
                moment = item["max_exp_avg_sq"]
                if (
                    not isinstance(moment, torch.Tensor)
                    or moment.shape != parameter.shape
                    or moment.dtype != parameter.dtype
                ):
                    raise ValueError(
                        f"Adam max_exp_avg_sq shape or dtype differs from parameter: {path}"
                    )
        adam_states = value["optimizer"]["state"]
        if any("step" not in item for item in adam_states.values()):
            raise ValueError(f"Adam state missing step: {path}")
        steps = set()
        for item in adam_states.values():
            step = item["step"]
            if isinstance(step, torch.Tensor):
                if step.numel() != 1:
                    raise ValueError(f"Invalid Adam step: {path}")
                step = step.item()
            if (
                isinstance(step, bool)
                or not isinstance(step, (int, float))
                or not math.isfinite(step)
                or step < 0
                or step != int(step)
            ):
                raise ValueError(f"Invalid Adam step: {path}")
            steps.add(int(step))
        state = value["state"]
        step_key = (
            "optimizer_steps"
            if "optimizer_steps" in state
            else "Adam_steps"
            if "Adam_steps" in state
            else None
        )
        if adapting:
            from civ5rl.engine.ppo_adam import check_counters, expected_counters
            check_counters(optimizer, state)
            if state.get("Adam_counters") != expected_counters(state):
                raise ValueError(f"Grouped Adam state counters differ: {path}")
        elif (
            step_key is not None
            and (steps or adam_offset + state[step_key])
            and steps != {adam_offset + state[step_key]}
        ):
            raise ValueError(f"Adam step counter mismatch: {path}: {steps}")
        checked.append(
            {
                "checkpoint": path.name,
                "adam_steps": sorted(steps),
                "phase": state.get("phase"),
            }
        )
    run_state_path = directory / "state.json"
    if run_state_path.exists():
        run_state = read_json(run_state_path)
        latest = run_state.get("latest")
        if latest:
            latest_path = (directory / latest).resolve()
            if (
                not latest_path.is_relative_to(directory.resolve())
                or not latest_path.is_file()
            ):
                raise ValueError(
                    "Latest checkpoint is missing or outside run directory"
                )
            saved_state = torch.load(
                latest_path, map_location="cpu", weights_only=False
            )["state"]
            for key in (
                "updates",
                "decisions",
                "optimizer_steps",
                "Adam_steps",
                "epoch",
                "epoch_completed",
                "free_labels_processed",
                "attempted_steps",
                "rejected_steps",
                "phase",
            ):
                if (
                    key in run_state
                    and key in saved_state
                    and run_state[key] != saved_state[key]
                ):
                    raise ValueError(f"Latest checkpoint state differs for {key}")
    if parent_checkpoint:
        initial = torch.load(
            directory / "checkpoints" / "initial.pt",
            map_location="cpu",
            weights_only=False,
        )
        parent = torch.load(parent_checkpoint, map_location="cpu", weights_only=False)
        if not _equal(initial["model"], parent["model"]):
            raise ValueError("Initial model differs from parent")
        if initial["config"].get("experiment", {}).get("parent_optimizer") == "preserve":
            from civ5rl.provenance.common import file_sha256
            if file_sha256(parent_checkpoint) != initial["config"]["experiment"]["parent_sha256"]:
                raise ValueError("Parent checkpoint SHA256 differs")
            if initial["config"]["experiment"].get("initial_value") == "adapt_parent_first_rollout":
                parent_ids = [i for g in parent["optimizer"]["param_groups"] for i in g["params"]]
                parent_named = dict(zip(dict(policy.named_parameters()), parent_ids, strict=True))
                for group in initial["optimizer"]["param_groups"]:
                    for name, index in zip(group["parameter_names"], group["params"], strict=True):
                        item = initial["optimizer"]["state"].get(index, {})
                        if name.startswith("value."):
                            if item:
                                raise ValueError("Initial value Adam was not reset")
                        elif not _equal(item, parent["optimizer"]["state"][parent_named[name]]):
                            raise ValueError("Initial shared/actor Adam differs from parent")
            elif not _equal(initial["optimizer"], parent["optimizer"]):
                raise ValueError("Initial Adam differs from parent")
    return {
        "status": "verified",
        "adam_offset": adam_offset,
        "parent_checked": bool(parent_checkpoint),
        "rng_validation": {
            "python": "isolated",
            "numpy": "isolated",
            "torch_cpu": "isolated",
            "shuffle": "isolated",
            "cuda": "structural_only",
        },
        "checkpoints": checked,
    }


def write_artifact_hashes(directory, output):
    """Hash regular files under directory, excluding transient files and symlinks."""
    root = Path(directory).resolve()
    output = Path(output).resolve()
    hashes = {}
    for path in sorted(root.rglob("*")):
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(root)
        ):
            continue
        relative = path.relative_to(root)
        if (
            path.resolve() == output
            or "__pycache__" in relative.parts
            or "prepared-inputs" in relative.parts
            or path.name in ("run.lock", "artifact-hashes.json")
            or path.suffix in (".tmp", ".log")
        ):
            continue
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        hashes[relative.as_posix()] = h.hexdigest()
    write_json(output, hashes)
    return hashes
