"""Frozen probability ensembles, using the ordinary match and RPC evaluator."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
from time import perf_counter

import torch

from civ5rl.core.agent import RLAgent
from civ5rl.core.model import batch_inputs
from civ5rl.core.observation import ordered_actions
from civ5rl.engine.evaluation.executor import (
    BatchedInference,
    EvaluationExecutor,
    EvaluationRuntime,
    evaluation_precision,
    prepare_evaluation_policy,
)
from civ5rl.engine.evaluation.storage import archive_evaluations, readable_evaluations, restore_evaluations
from civ5rl.engine.runtime.control import RunLock
from civ5rl.experiments.evaluation.ensemble_match import checked_match
from civ5rl.provenance.common import configure, file_sha256, read_json, write_json
from civ5rl.reporting.ensemble_report import report
from civ5rl.reporting.evaluation_audit import clustered_paired_bootstrap, eval_wdl_summary, load_eval_rows
from civ5sim.combat_ai import LIBRARY_PATH
from civ5sim.engine import digest


def mean_probabilities(logits):
    return torch.stack([(value / 0.5).softmax(-1) for value in logits]).mean(0)


class EnsembleInference(BatchedInference):
    def __init__(self, policies, device, *, hidden_capacity=32):
        super().__init__(policies[0], device, hidden_capacity=hidden_capacity)
        self.policies = policies

    def combine_encoded_logits(self, logits_all, requests):
        choices = mean_probabilities(logits_all).argmax(-1).cpu().tolist()
        return [
            None if command == "observe" else {"index": choices[i]}
            for i, (_, command, _) in enumerate(requests)
        ]

    @torch.inference_mode()
    def infer(self, requests):
        started = perf_counter()
        actions = [ordered_actions(r[3]) for r in requests]
        logits_all = []
        for member, policy in enumerate(self.policies):
            agents = [r[0].members[member] for r in requests]
            encoded = [a.encoder.encode(r[2], legal) for a, r, legal in zip(agents, requests, actions)]
            batch = batch_inputs(
                encoded, self.device, compact_units=policy.attention_backend == "compact_fp32"
            )
            hidden = self.gather_hidden(policy, agents, member)
            logits, _, hidden = policy(batch, hidden)
            logits_all.append(logits)
            self.store_hidden(agents, hidden, member)
        choices = mean_probabilities(logits_all).argmax(-1).cpu().tolist()
        outputs = []
        for i, (agent, command, observation, _, forced) in enumerate(requests):
            chosen = forced if command == "observe" else actions[i][choices[i]]
            if chosen not in actions[i]:
                raise ValueError("Forced action is not legal")
            for member in agent.members:
                member.encoder.remember_action(chosen, observation)
            outputs.append(None if command == "observe" else (chosen, []))
        self.forward_seconds += perf_counter() - started
        self.batches += 1
        self.items += len(requests)
        return outputs


class EnsembleExecutor(EvaluationExecutor):
    def __init__(self, checkpoints, runtime):
        super().__init__(str(checkpoints[0]), runtime)
        policies = [self.inference.policy]
        policies += [
            prepare_evaluation_policy(RLAgent.from_checkpoint(p, "cpu", "argmax").policy, self.runtime.device)
            for p in checkpoints[1:]
        ]
        if any(p.variant != policies[0].variant for p in policies):
            raise ValueError("Ensemble variants differ")
        self.inference = EnsembleInference(
            policies, self.runtime.device, hidden_capacity=self.runtime.workers
        )


def _evaluate_side(config, cases, output, runtime, purpose, label, paths):
    folder = output / "evaluation" / purpose / label
    pending = []
    for case in cases:
        if case["max_rounds"] != 60 or case["scenario"].get("turn_reset_mode") != "end_turn":
            raise ValueError("Wrong match physics")
        directory = folder / f"case-{case['case_id']:03}-side-{case['learner_side']}"
        if (directory / "checked.json").exists():
            if read_json(directory / "case.json") != case:
                raise ValueError("Changed case")
            continue
        pending.append((config, case, str(paths[0]), str(directory)))
    if pending:
        started = perf_counter()
        executor = (
            EnsembleExecutor(paths, runtime)
            if len(paths) == 2
            else EvaluationExecutor(str(paths[0]), runtime)
        )
        for count, _ in enumerate(executor.run(checked_match, pending), 1):
            if count % 8 == 0 or count == len(pending):
                print(f"{purpose}/{label}: {count}/{len(pending)}", flush=True)
        write_json(
            output / f"{purpose}-{label}-timing.json",
            {"seconds": perf_counter() - started, "inference": executor.statistics()},
        )
    return load_eval_rows(folder, plan=cases)


def _historical_parent(run, cases):
    with RunLock(run), readable_evaluations(run):
        baseline = next(
            record
            for record in read_json(run / "evaluations.json").values()
            if record["checkpoint"] == "checkpoints/initial.pt" and record["purpose"] == "development_512"
        )
        return load_eval_rows(run / baseline["runs"]["argmax"]["cp"]["directory"], plan=cases)


def evaluate_ensemble(run, checkpoints, out):
    if len(checkpoints) != 2:
        raise ValueError("A 50/50 ensemble requires exactly two checkpoints")
    run, out = Path(run).resolve(), Path(out).resolve()
    checkpoints = [Path(path).resolve() for path in checkpoints]
    config = read_json(run / "run.json")["config"]
    plan = read_json(run / "manifest.json")
    if plan.get("turn_reset_mode") != "end_turn":
        raise ValueError("Requires end_turn source run")
    configure(17, "cuda")
    runtime = EvaluationRuntime("cuda", 8, 32, 0.001).resolve()

    def evaluate(purpose, label, paths):
        return _evaluate_side(config, plan[purpose], out, runtime, purpose, label, paths)

    if (out / "evaluation.storage").exists() and not (out / "evaluation").exists():
        restore_evaluations(out)
    with evaluation_precision(), RunLock(out):
        identity = {
            "schema": 1,
            "checkpoints": [{"path": str(path), "sha256": file_sha256(path)} for path in checkpoints],
            "weights": [0.5, 0.5],
            "temperature": 0.5,
            "runtime": runtime.identity(),
            "planner": file_sha256(LIBRARY_PATH),
            "manifest": file_sha256(run / "manifest.json"),
            "source_config": digest(config),
        }
        if (out / "identity.json").exists() and read_json(out / "identity.json") != identity:
            raise ValueError("Evaluation identity changed")
        write_json(out / "identity.json", identity)
        ensemble = evaluate("development_512", "ensemble", checkpoints)
        parent = _historical_parent(run, plan["development_512"])
        result = {
            "development": eval_wdl_summary(ensemble),
            "comparison": clustered_paired_bootstrap(parent, ensemble),
            "historical_gpu_comparison": True,
        }
        write_json(out / "summary.json", result)
        if result["development"]["W"] > eval_wdl_summary(parent)["W"]:
            confirmation_parent = evaluate("confirmation", "parent", checkpoints[:1])
            confirmation_ensemble = evaluate("confirmation", "ensemble", checkpoints)
            result["additional"] = {
                "parent": eval_wdl_summary(confirmation_parent),
                "ensemble": eval_wdl_summary(confirmation_ensemble),
                "comparison": clustered_paired_bootstrap(confirmation_parent, confirmation_ensemble),
            }
        else:
            result["additional"] = {"status": "not_run_no_development_gain"}
        for item, checkpoint in zip(identity["checkpoints"], checkpoints):
            if file_sha256(checkpoint) != item["sha256"]:
                raise ValueError("Checkpoint changed")
        write_json(out / "summary.json", result)
        report(out)
    archive_evaluations(out)
    return result
