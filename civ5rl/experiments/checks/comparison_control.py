"""Discarded real-history smoke for the full-corpus control path.

Only two public files from one source battle are extracted into disposable
storage, with their scientific SHA-256 checked before use. This is a small
correctness check, not the 330+660 timing measurement or an experiment.
"""

from __future__ import annotations

import gc
import json
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path

import torch

from civ5rl.deploy.archive import seven_zip
from civ5rl.engine.comparison.continuation import continuation_seconds, validate_continuation
from civ5rl.engine.comparison.heads import configure_heads
from civ5rl.engine.comparison.training import fit_full_corpus
from civ5rl.engine.evaluation.executor import EvaluationRuntime, evaluation_key, evaluation_precision
from civ5rl.engine.runtime.budget import PhaseBudget
from civ5rl.provenance.checkpoint import verify_frozen_passport
from civ5rl.provenance.common import ROOT, configure, file_sha256, read_json, write_json


def continuation_smoke(cfg, out):
    """Verify real inherited state and evaluator compatibility without fitting."""

    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use an empty precheck output")
    state = validate_continuation(cfg)
    source = ROOT / cfg["fit_continuation"]["run"]
    configure(cfg["seed"], "cuda")
    with evaluation_precision():
        _, key = evaluation_key(
            read_json(source / "match-config.json"),
            read_json(source / "manifest.json"),
            source,
            "checkpoints/parent.pt",
            "development_512",
            EvaluationRuntime("cuda", 8, 32, 0.001).resolve().identity(),
        )
    if key not in read_json(source / "evaluations.json"):
        raise ValueError("Parent Development evaluator/runtime/GPU is incompatible")
    result = {
        "status": "verified",
        "inherited_steps": state["steps"],
        "next_adam_step": state["steps"] + 1,
        "inherited_seconds": continuation_seconds(cfg),
        "remaining_seconds": cfg["training_seconds"] - continuation_seconds(cfg),
        "anchor_cursor": state["anchor_cursor"],
        "head_adam_exact": True,
        "parent_evaluation_compatible": True,
    }
    write_json(out / "precheck.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def smoke(cfg, out):
    out = Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use an empty smoke output")
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT / cfg["collection_input"]["run"]
    for name, sha in cfg["collection_input"]["sha256"].items():
        if file_sha256(source / name) != sha:
            raise ValueError(f"Collection input changed: {name}")
    verify_frozen_passport(source)
    dataset = read_json(source / "dataset.json")
    example = dataset["pairs"][0]
    episode = example["episode"]
    anchors = [key for key in dataset["anchors"] if key[0] == episode][:4]
    if len(anchors) < 4:
        raise ValueError("Smoke requires four public target positions")
    configure(cfg["seed"], "cuda")
    with tempfile.TemporaryDirectory(prefix="comparison-control-", dir=ROOT / ".tmp") as tmp:
        scratch = Path(tmp)
        files = [f"{episode}/{name}" for name in ("case.json", "history.json")]
        subprocess.run(
            [
                seven_zip(),
                "x",
                str(source / "sources.storage/json.7z"),
                "-o" + str(scratch / "sources"),
                "-y",
                *files,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        for name, sha in dataset["source_hashes"][str(episode)].items():
            if file_sha256(scratch / "sources" / str(episode) / name) != sha:
                raise ValueError("Smoke public source hash differs")
        tiny = {
            "pairs": [example],
            "anchors": anchors,
            "source_hashes": {str(episode): dataset["source_hashes"][str(episode)]},
        }
        tiny_cfg = deepcopy(cfg)
        tiny_cfg["training"].update(max_accepted_steps=2, anchors_per_step=2)
        with PhaseBudget(scratch, "training", 1800) as budget:
            fit_full_corpus(tiny_cfg, scratch, tiny, {"discarded_smoke": True}, budget)
        result = read_json(scratch / "training.json")
        before = file_sha256(scratch / "candidate.pt")
        with PhaseBudget(scratch, "training", 1800) as budget:
            fit_full_corpus(tiny_cfg, scratch, tiny, {"discarded_smoke": True}, budget)
        if before != file_sha256(scratch / "candidate.pt"):
            raise AssertionError("Terminal smoke resume changed the candidate")
        candidate = torch.load(scratch / "candidate.pt", map_location="cpu", weights_only=False)
        parent = torch.load(ROOT / cfg["parents"][0], map_location="cpu", weights_only=False)
        if any(
            not torch.equal(v, parent["model"][k])
            for k, v in candidate["model"].items()
            if k.startswith("value.")
        ):
            raise AssertionError("Critic changed during comparison smoke")
        if cfg["training"]["mode"] == "frozen_heads_step_kl":
            if any(
                not torch.equal(v, parent["model"][k])
                for k, v in candidate["model"].items()
                if not k.startswith(("action.", "attack_preview."))
            ):
                raise AssertionError("Frozen representation changed")

            model = configure_heads(cfg["parents"][0], device="cpu")
            named_parameters = [
                (name, param) for name, param in model.named_parameters() if param.requires_grad
            ]
            if not named_parameters or any(
                not name.startswith(("action.", "attack_preview.")) for name, _ in named_parameters
            ):
                raise AssertionError("Wrong trainable head set")
            parameters = [parameter for _, parameter in named_parameters]
            if len(candidate["optimizer"]["param_groups"][0]["params"]) != len(parameters):
                raise AssertionError("Frozen parameters entered head Adam")
            result.update(
                frozen_weights_exact=True,
                trainable_parameters=sum(parameter.numel() for parameter in parameters),
            )
        accepted = result["training_steps"]
        if any(int(v["step"]) != accepted for v in candidate["optimizer"]["state"].values()):
            raise AssertionError("Smoke Adam counters differ from accepted steps")
        if result["last_full_kl"] > cfg["training"]["kl_limit"]:
            raise AssertionError("Smoke published an unverified model")
        result.update(
            discarded_smoke=True,
            source_pairs=1,
            training_anchors_per_step=2,
            control_anchors=4,
            terminal_resume_exact=True,
            critic_unchanged=True,
            parents_unchanged=all(
                file_sha256(ROOT / p) == sha for p, sha in zip(cfg["parents"], cfg["parent_sha256"])
            ),
        )

        gc.collect()
    write_json(out / "precheck.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result
