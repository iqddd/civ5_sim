"""A bounded continuation experiment with one run record, not environment certification."""

import gc
import importlib.metadata
import json
import os
import platform
import random
import traceback
from collections import Counter
from pathlib import Path
from time import perf_counter

import torch

from civ5rl.engine.collector import Collector
from civ5rl.engine.control import RunLock
from civ5rl.engine.ppo import update
from civ5rl.engine.resources import ResourceMonitor, analyze_telemetry
from civ5rl.experiments.evaluation import paired_world_bootstrap, verify_replay
from civ5rl.experiments.imitation_helpers import (
    CompilationClock, assert_equal, isolated_rng, load_policy, pool,
)
from civ5rl.experiments.ppo_after_imitation import (
    match_summary, optimizer_for, play_case, policy_hash, summarize_rollout,
)
from civ5rl.experiments.ppo_artifacts import prune_checkpoints, save_rollout
from civ5rl.protocols.corpus import validate_corpus
from civ5rl.protocols.ppo_scale_up import KIND, ScaleUpFactory
from civ5rl.provenance import checkpoint
from civ5rl.provenance.common import ROOT, configure, file_sha256, read_json, seed_for, write_json
from civ5sim.engine import digest


def make_policy(config, device):
    policy = load_policy(config, ROOT / config["experiment"]["parent"], device)
    for parameter in policy.value.parameters():
        parameter.requires_grad_(True)
    return policy


def load_parent(config, device="cpu"):
    saved = torch.load(ROOT / config["experiment"]["parent"], map_location="cpu", weights_only=False)
    policy = make_policy(config, device)
    optimizer = optimizer_for(policy, config, device)
    optimizer.load_state_dict(saved["optimizer"])
    assert_equal(policy.state_dict(), saved["model"])
    assert_equal(optimizer.state_dict(), saved["optimizer"])
    counters = {int(v["step"]) for v in optimizer.state.values()}
    if counters != {config["experiment"]["parent_optimizer_steps"]}:
        raise ValueError(f"Parent Adam counters differ: {counters}")
    if policy_hash(policy.state_dict()) != config["experiment"]["parent_policy_hash"]:
        raise ValueError("Parent policy differs")
    return policy, optimizer


def prepare(config, out):
    e = config["experiment"]
    for name, expected in ((e["parent"], e["parent_sha256"]),
                           (e["parent_manifest"], e["parent_manifest_sha256"])):
        if file_sha256(ROOT / name) != expected:
            raise ValueError(f"Key input differs: {name}")
    previous = read_json(ROOT / e["parent_manifest"])
    base = previous["base"]
    worlds = base["worlds"]
    if len(worlds["training"]) != 57 or set(worlds["training"]) & (set(worlds["validation"]) | set(worlds["holdout"])):
        raise ValueError("Training/validation worlds overlap")
    maps = [m for m in base["maps"] if m["split"] == "train" and m["parent_seed"] in worlds["training"]]
    if len(maps) != 228:
        raise ValueError("Expected 228 training crops")
    validate_corpus({"maps": maps})
    plan = {"schema": 1, "kind": KIND, "base": {**base, "maps": maps},
            "development": previous["development"], "confirmation": previous["confirmation"]}
    if (len(plan["development"]), len(plan["confirmation"])) != (64, 256):
        raise ValueError("Unexpected evaluation cases")
    for purpose in ("development", "confirmation"):
        for case in plan[purpose]:
            if case["split"] != "validation" or case["parent_world"] not in worlds["validation"]:
                raise ValueError("Closed/untrusted evaluation case")
            if digest({k: v for k, v in case.items() if k != "case_hash"}) != case["case_hash"]:
                raise ValueError("Evaluation case changed")
    plan["hash"] = digest(plan)
    configure(17, "cpu")
    policy, optimizer = load_parent(config)
    run_record = {
        "schema": 1, "kind": KIND, "config": config,
        "source_snapshot": os.environ.get("CIV5RL_SOURCE_SNAPSHOT", "local-working-tree"),
        "parent_sha256": e["parent_sha256"], "parent_policy_hash": e["parent_policy_hash"],
        "input_manifest_sha256": e["parent_manifest_sha256"],
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "platform": platform.platform(), "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "CPU": platform.processor(), "seed": 17, "training_namespace": e["training_namespace"],
        "shuffle_seed": seed_for(17, 0, "shuffle", namespace=e["shuffle_namespace"]),
        "initial_model_Adam_bitwise_equal": True, "critic_preserved": True,
        "parent_Adam_counter": e["parent_optimizer_steps"],
        "native_build": read_json(ROOT / "build/remote-build.json") if (ROOT / "build/remote-build.json").exists() else None,
    }
    run_record["run_id"] = digest({k: run_record[k] for k in ("source_snapshot", "config", "parent_sha256", "input_manifest_sha256")})
    state = {"updates": 0, "accepted_updates": 0, "attempted_steps": 0, "rejected_steps": 0,
             "optimizer_steps": 0, "origin_Adam_counter": e["parent_optimizer_steps"],
             "decisions": 0, "next_episode": e["start_episode"], "zero_accepted_streak": 0,
             "active_seconds": 0., "evaluation_seconds": 0., "preparation_seconds": 0.,
             "evaluated": {}, "latest": "checkpoints/initial.pt", "pending_evaluation": True,
             "best": {"path": "checkpoints/initial.pt", "wins": -1, "update": 0}, "stop_reason": None}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "manifest.json", plan)
    write_json(out / "run.json", run_record)
    write_json(out / "packages.json", {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()})
    shuffle = random.Random(run_record["shuffle_seed"])
    checkpoint.save(out / state["latest"], policy, optimizer, state, shuffle, config, {"run_id": run_record["run_id"]})
    write_json(out / "state.json", state)
    write_json(out / "status.json", {"status": "prepared"})
    print(f"Prepared: model/Adam equal to parent; Adam={e['parent_optimizer_steps']}; "
          f"new decisions=0; episode={e['start_episode']}", flush=True)
    return plan


def evaluation_task(task):
    config, case, path, mode, enemy, directory = task
    if case["split"] != "validation":
        raise ValueError("Evaluation requires open validation")
    return play_case(config, case, path, mode, enemy, directory)


def evaluate_checkpoint(config, plan, out, path, purpose="development"):
    fingerprint = policy_hash(torch.load(out / path, map_location="cpu", weights_only=False)["model"])
    cache_path = out / "evaluations.json"
    cache = read_json(cache_path) if cache_path.exists() else {}
    key = purpose + ":" + fingerprint
    if key in cache:
        return cache[key]
    started = perf_counter()
    runs = {}
    with pool(4) as executor:
        for mode in (("argmax", "sample") if purpose == "development" else ("argmax",)):
            runs[mode] = {}
            for enemy in (("cp", "greedy") if purpose == "development" else ("cp",)):
                relative = f"evaluation/{purpose}/{fingerprint[:16]}/{mode}/{enemy}"
                directory = out / relative
                tasks = []
                for case in plan[purpose]:
                    target = directory / f"case-{case['case_id']:03}-side-{case['learner_side']}"
                    if not (target / "metrics.json").exists():
                        tasks.append((config, case, str(out / path), mode, enemy, str(target)))
                for first in range(0, len(tasks), 4):
                    list(executor.map(evaluation_task, tasks[first:first+4]))
                runs[mode][enemy] = {"directory": relative, "summary": match_summary(directory, len(plan[purpose]))}
                print(f"Evaluation {purpose}/{mode}/{enemy}: {runs[mode][enemy]['summary']}", flush=True)
    result = {"policy_hash": fingerprint, "checkpoint": path, "purpose": purpose,
              "runs": runs, "seconds": perf_counter() - started}
    cache[key] = result
    write_json(cache_path, cache)
    return result


def development(config, plan, out, state, shuffle):
    started = perf_counter()
    with isolated_rng(shuffle):
        result = evaluate_checkpoint(config, plan, out, state["latest"])
    state["evaluation_seconds"] += perf_counter() - started
    state["evaluated"][str(state["updates"])] = {**result, "decisions": state["decisions"]}
    wins = result["runs"]["argmax"]["cp"]["summary"]["W"]
    if wins > state["best"]["wins"]:
        state["best"] = {"path": state["latest"], "wins": wins, "update": state["updates"],
                         "policy_hash": result["policy_hash"]}
    state["pending_evaluation"] = False
    write_json(out / "state.json", state)
    with isolated_rng(shuffle):
        report(config, out)


def train(config, plan, out, device, *, factory_type=ScaleUpFactory, development_fn=development):
    state = read_json(out / "state.json")
    if state["stop_reason"]:
        return
    configure(17, device)
    record = read_json(out / "run.json")
    identity = {"run_id": record["run_id"]}
    policy = make_policy(config, device)
    optimizer = optimizer_for(policy, config, device)
    shuffle = random.Random(record["shuffle_seed"])
    saved = checkpoint.restore(out / state["latest"], policy, optimizer, shuffle, config, identity)
    if saved["decisions"] != state["decisions"] or saved["optimizer_steps"] != state["optimizer_steps"]:
        raise ValueError("Checkpoint and progress disagree")
    if config.get('kind') == 'rl01.ppo_target_scale':
        loaded = torch.load(out/state['latest'],map_location='cpu',weights_only=False)
        assert_equal(policy.state_dict(),loaded['model'])
        assert_equal(optimizer.state_dict(),loaded['optimizer'])
        assert_equal(random.getstate(),loaded['python_rng'])
        import numpy as np
        assert_equal(np.random.get_state(),loaded['numpy_rng'])
        assert_equal(torch.get_rng_state(),loaded['torch_rng'])
        assert_equal(shuffle.getstate(),loaded['shuffle_rng'])
        if device.startswith('cuda') and loaded['cuda_rng']:
            assert_equal(torch.cuda.get_rng_state_all(),loaded['cuda_rng'])
        write_json(out/'resume-verification.json',{'checkpoint':state['latest'],
            'model_Adam_RNG_equal':True,'Adam_counters':sorted({int(s['step']) for s in optimizer.state.values()}),
            'decisions':state['decisions'],'next_episode':state['next_episode'],'device':device})
        del loaded
    if state["pending_evaluation"]:
        development_fn(config, plan, out, state, shuffle)
    factory = factory_type(config, plan["base"])
    with ResourceMonitor(out / "telemetry") as monitor, Collector(config, factory, device, out / "errors") as collector:
        monitor.mark("initial_warmup")
        # Existing public rollout fixture: no new episode or optimizer work.
        prep = perf_counter()
        with isolated_rng(shuffle), torch.no_grad():
            from civ5rl.core.model import batch_inputs
            from civ5rl.engine.ppo import encode_episode
            from civ5rl.experiments.ppo_artifacts import load_rollout
            fixture = load_rollout(ROOT / "artifacts/rl/ppo-controlled-updates/rollouts/first.pt")[0]
            encoded = encode_episode(fixture, "C")[:20]
            policy.forward_sequences(batch_inputs(encoded, device), [len(encoded)])
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        state["preparation_seconds"] += perf_counter() - prep

        def inspect_first_step(step):
            if config.get('kind')=='rl01.ppo_target_scale' and step['accepted'] and not (out/'first-resumed-step.json').exists():
                counters=sorted({int(s['step']) for s in optimizer.state.values()})
                expected=config['experiment']['parent_optimizer_steps']+state['optimizer_steps']+step['accepted_step']
                if counters!=[expected]:
                    raise ValueError(f'Resumed Adam counter differs: {counters} != {expected}')
                write_json(out/'first-resumed-step.json',{'Adam_counters':counters,'checkpoint':saved['latest'],
                    'restored_optimizer_steps':saved['optimizer_steps'],'step':step})
            if step["accepted"] and not (out / "first-accepted-step.json").exists():
                counters = sorted({int(s["step"]) for s in optimizer.state.values()})
                expected = config["experiment"]["parent_optimizer_steps"] + 1
                if counters != [expected]:
                    raise ValueError(f"First accepted Adam counter is {counters}, expected {expected}")
                write_json(out / "first-accepted-step.json", {"Adam_counters": counters, "step": step})

        while not state["stop_reason"]:
            if (out/'pause.request').exists():
                write_json(out/'status.json',{'status':'paused','checkpoint':state['latest']})
                return
            e = config["experiment"]
            if state["active_seconds"] >= e["active_seconds"]:
                state["stop_reason"] = "active_time_limit"
                break
            if state["decisions"] >= e["decisions"]:
                state["stop_reason"] = "decision_limit"
                break
            started = perf_counter()
            deadline = started + e["active_seconds"] - state["active_seconds"]
            number = state["updates"] + 1
            monitor.mark("collection", update=number)
            episodes, following, timings = collector.collect(policy, None, state["next_episode"], monitor,
                deadline=deadline, max_decisions=e["decisions"] - state["decisions"])
            if not episodes:
                state["active_seconds"] += perf_counter() - started
                state["stop_reason"] = "active_time_limit"
                break
            summary = summarize_rollout(episodes)
            before_decisions = state["decisions"]
            state["decisions"] += summary["decisions"]
            state["next_episode"] = following
            monitor.mark("rollout_save")
            artifact = save_rollout(episodes, out / "rollouts", number)
            metrics = {"optimizer_steps": 0, "attempted_optimizer_steps": 0, "rejected_optimizer_steps": 0,
                       "stop_reason": "active_time_limit", "update_complete": False}
            delta = 0.
            if perf_counter() < deadline:
                monitor.mark("ppo", update=number)
                before = [p.detach().clone() for p in policy.parameters()]
                with CompilationClock() as compiler:
                    compiler.monitor = monitor
                    metrics = update(policy, optimizer, episodes, config, before_decisions, device, shuffle,
                                     deadline=deadline, on_step=inspect_first_step)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                metrics["runtime_compilation_seconds"] = sum(max(0, b-a) for a, b in compiler.intervals)
                delta = float(torch.sqrt(sum((p.detach()-b).float().square().sum() for p, b in zip(policy.parameters(), before))))
                del before
            del episodes
            gc.collect()
            state["active_seconds"] += perf_counter() - started
            state["updates"] = number
            state["optimizer_steps"] += metrics["optimizer_steps"]
            state["attempted_steps"] += metrics["attempted_optimizer_steps"]
            state["rejected_steps"] += metrics["rejected_optimizer_steps"]
            state["accepted_updates"] += bool(metrics["optimizer_steps"])
            state["zero_accepted_streak"] = 0 if metrics["optimizer_steps"] else state["zero_accepted_streak"] + 1
            state["latest"] = f"checkpoints/update-{number:04}.pt"
            if state["active_seconds"] >= e["active_seconds"] or metrics["stop_reason"] == "active_time_limit":
                state["stop_reason"] = "active_time_limit"
            elif state["decisions"] >= e["decisions"]:
                state["stop_reason"] = "decision_limit"
            elif state["zero_accepted_streak"] >= 3:
                state["stop_reason"] = "no_accepted_updates"
            due = any(before_decisions < m <= state["decisions"] for m in e["milestones"])
            state["pending_evaluation"] = bool(due or state["stop_reason"])
            servicing = perf_counter()
            monitor.mark("checkpoint")
            checkpoint.save(out / state["latest"], policy, optimizer, state, shuffle, config, identity)
            prune_checkpoints(out, state)
            write_json(out / "coverage.json", factory.summary())
            state["active_seconds"] += perf_counter() - servicing
            if state["active_seconds"] >= e["active_seconds"]:
                state["stop_reason"] = "active_time_limit"
                state["pending_evaluation"] = True
            write_json(out / "state.json", state)
            row = {"update": number, **summary, "total_decisions": state["decisions"],
                   "total_Adam_steps": state["optimizer_steps"],
                   "Adam_counter": e["parent_optimizer_steps"] + state["optimizer_steps"],
                   "active_seconds": state["active_seconds"], "weight_delta_l2": delta,
                   "decision_overshoot": max(0, state["decisions"] - e["decisions"]),
                   "active_time_overshoot_seconds": max(0, state["active_seconds"] - e["active_seconds"]),
                   "collection": timings, "rollout_artifact": artifact, "ppo": metrics}
            with (out / "updates.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            print(f"Rollout {number}: decisions={state['decisions']}; W/D/L={summary['W']}/{summary['D']}/{summary['L']}; "
                  f"new Adam={state['optimizer_steps']}; active={state['active_seconds']:.1f}s; {metrics['stop_reason']}", flush=True)
            if number == 3 or due:
                eta = {"decisions": state["decisions"], "active_seconds": state["active_seconds"],
                       "estimated_remaining_active_seconds": max(0., min(e["active_seconds"] - state["active_seconds"],
                           (e["decisions"] - state["decisions"]) * state["active_seconds"] / state["decisions"]))}
                write_json(out / "eta.json", eta)
                print("ETA " + json.dumps(eta), flush=True)
            if (out/'pause.request').exists():
                # Persist all counters/RNG after update servicing, before any
                # pending evaluation; resume evaluates before collecting again.
                checkpoint.save(out/state['latest'],policy,optimizer,state,shuffle,config,identity)
                write_json(out/'state.json',state)
                write_json(out/'status.json',{'status':'paused','checkpoint':state['latest']})
                return
            if state["pending_evaluation"]:
                monitor.mark("evaluation")
                development_fn(config, plan, out, state, shuffle)
        monitor.mark("stopped")
        if policy_hash(policy.state_dict()) not in {v["policy_hash"] for v in state["evaluated"].values()}:
            state["pending_evaluation"] = True
        if state["pending_evaluation"]:
            development_fn(config, plan, out, state, shuffle)
    write_json(out / "state.json", state)
    write_json(out / "status.json", {"status": "training_complete", "stop_reason": state["stop_reason"]})


def confirmation(config, plan, out):
    state = read_json(out / "state.json")
    if not state["stop_reason"] or state["pending_evaluation"]:
        raise ValueError("Training/selection is incomplete")
    write_json(out / "selected.json", state["best"])
    baseline = evaluate_checkpoint(config, plan, out, "checkpoints/initial.pt", "confirmation")
    candidate = evaluate_checkpoint(config, plan, out, state["best"]["path"], "confirmation")
    same = baseline["policy_hash"] == candidate["policy_hash"]
    def rows(evaluation):
        directory = out / evaluation["runs"]["argmax"]["cp"]["directory"]
        result, cases, outcomes = {}, {}, {}
        for path in sorted(directory.glob("*/metrics.json")):
            m, c = read_json(path), read_json(path.parent / "case.json")
            key = (c["parent_world"], c["case_id"], c["learner_side"])
            if key in result:
                raise ValueError("Duplicate validation case")
            result[key] = int(m["winner"] == m["side"])
            cases[key] = c
            outcomes[key] = "D" if m["winner"] is None else "W" if m["winner"] == m["side"] else "L"
        return result, cases, outcomes
    before, cases, before_outcomes = rows(baseline)
    after, paired, after_outcomes = rows(candidate)
    if cases != paired:
        raise ValueError("Unpaired additional validation")
    ci = None if same else paired_world_bootstrap(before, after, samples=10000, seed=20261103, cases=cases, cases_per_world=16)
    slices = {}
    for name, predicate in (("contact", lambda c: c["placement"] == "contact"),
                            ("separated", lambda c: c["placement"] == "separated"),
                            ("side0", lambda c: c["learner_side"] == 0),
                            ("side1", lambda c: c["learner_side"] == 1)):
        keys = [k for k, c in cases.items() if predicate(c)]
        slices[name] = {"matches": len(keys), "baseline_wins": sum(before[k] for k in keys),
                        "candidate_wins": sum(after[k] for k in keys),
                        "baseline_WDL": dict(Counter(before_outcomes[k] for k in keys)),
                        "candidate_WDL": dict(Counter(after_outcomes[k] for k in keys))}
    write_json(out / "confirmation.json", {"baseline": baseline, "candidate": candidate, "comparison": ci,
               "slices": slices, "baseline_selected": same, "selected_before_confirmation": True})


def report(config, out):
    state = read_json(out / "state.json")
    origin = config["experiment"]["parent_optimizer_steps"]
    parent = config["experiment"].get("parent_label", Path(config["experiment"]["parent"]).stem)
    rows = [json.loads(s) for s in (out / "updates.jsonl").read_text().splitlines()] if (out / "updates.jsonl").exists() else []
    final = read_json(out / "confirmation.json") if (out / "confirmation.json").exists() else None
    lines = ["# PPO scale-up на RTX 4090", "", f"Статус: {read_json(out / 'status.json')['status']}; причина: {state['stop_reason']}.",
             f"Новые решения: {state['decisions']}; rollouts: {state['updates']}; attempted/accepted/rejected: "
             f"{state['attempted_steps']}/{state['optimizer_steps']}/{state['rejected_steps']}; "
             f"Adam counter: {origin + state['optimizer_steps']}.",
             f"Active: {state['active_seconds']:.2f} с; development evaluation: {state['evaluation_seconds']:.2f} с; warmup: {state['preparation_seconds']:.2f} с.",
             "", f"Родитель — {parent} с обученным critic и Adam {origin}. Основная пара оценивается одним Linux evaluator.",
             "", "| Новые решения | Rollout | Режим | Соперник | W/D/L | Win rate |", "|---:|---:|---|---|---|---:|"]
    for number, point in state["evaluated"].items():
        for mode, enemies in point["runs"].items():
            for enemy, result in enemies.items():
                m = result["summary"]
                lines.append(f"| {point['decisions']} | {number} | {mode} | {enemy} | {m['W']}/{m['D']}/{m['L']} | {m['win_rate']:.2%} |")
    lines += ["", f"Выбранный checkpoint: {state['best']}."]
    if final:
        lines += ["", "## Дополнительный validation, 256 argmax-матчей против CP", "",
                  f"Linux baseline {parent}: {final['baseline']['runs']['argmax']['cp']['summary']}",
                  f"Кандидат: {final['candidate']['runs']['argmax']['cp']['summary']}",
                  f"Парный прирост/95% CI: {final['comparison']}", f"Срезы: {final['slices']}",
                  "Продолжение не прошло development-отбор; одинаковые веса не дают проверки эквивалентности." if final["baseline_selected"] else
                  "Bootstrap: 10 000 выборок, seed 20261103, кластер — parent-world. Дополнительная оценка не участвовала в выборе.",
                  f"Additional evaluation: {final['baseline']['seconds'] + (0 if final['baseline_selected'] else final['candidate']['seconds']):.2f} с."]
    if rows:
        lines += ["", "## Стоимость обучения", "",
                  f"Collection: {sum(r['collection']['collection_seconds'] for r in rows):.2f} с; "
                  f"PPO including KL: {sum(r['ppo'].get('ppo_seconds', 0) for r in rows):.2f} с; "
                  f"KL: {sum(r['ppo'].get('kl_seconds', 0) for r in rows):.2f} с; "
                  f"runtime compilation: {sum(r['ppo'].get('runtime_compilation_seconds', 0) for r in rows):.2f} с (входит в бюджет).",
                  f"Последний rollout: {rows[-1]['decisions']} решений; превышения: {rows[-1]['decision_overshoot']} решений, {rows[-1]['active_time_overshoot_seconds']:.3f} с.",
                  "", "| Rollout | Decisions | W/D/L | Attempt/accept/reject | Full KL max/final | Policy/value loss | Critic MSE/EV | Gradient mean/max |", "|---:|---:|---|---|---|---|---|---|"]
        for r in rows:
            p = r["ppo"]
            lines.append(f"| {r['update']} | {r['decisions']} | {r['W']}/{r['D']}/{r['L']} | {p['attempted_optimizer_steps']}/{p['optimizer_steps']}/{p['rejected_optimizer_steps']} | {p.get('max_attempted_kl')}/{p.get('final_accepted_kl')} | {p.get('policy_loss')}/{p.get('value_loss')} | {p.get('critic_mse')}/{p.get('critic_explained_variance')} | {p.get('gradient_norm_mean')}/{p.get('gradient_norm_max')} |")
    if (out / "coverage.json").exists():
        coverage = read_json(out / "coverage.json")
        lines += ["", f"Покрытие: {coverage['episodes']} эпизодов; {len(coverage['worlds'])} миров, {len(coverage['crops'])} различных карт; "
                  f"уникальных физических стартов {coverage['physical_unique']}, повторов {coverage['physical_repeats']}. Историческое физическое пересечение не измерялось."]
    if (out / "telemetry/resources.jsonl").exists():
        gpu = analyze_telemetry(out / "telemetry/resources.jsonl", active_phase="ppo")
        write_json(out / "gpu-analysis.json", {k: v for k, v in gpu.items() if k != "active"})
        lines += ["", f"NVML (PPO, без CPU collection/evaluation и compilation): {gpu['metrics']}; 60s windows: {len(gpu['windows'])}.",
                  "Если длительная GPU-фаза недогружена: проверенные ранее ограничения — CPU/autograd/launch overhead и низкая occupancy небольших spatial GEMM; условия ради загрузки не менялись."]
    lines += ["", "Оба validation-набора уже используются для разработки; это не закрытый test. Другие seeds, BC, подбор параметров и автоматическое продление не выполнялись."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if state["evaluated"]:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        points = list(state["evaluated"].values())
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for mode in ("argmax", "sample"):
            ax.plot([p["decisions"] for p in points], [p["runs"][mode]["cp"]["summary"]["win_rate"] for p in points], marker="o", label=mode + " vs CP")
        ax.set_xlabel("New learner decisions")
        ax.set_ylabel("Development win rate")
        ax.legend()
        ax.grid(alpha=.25)
        fig.savefig(out / "learning-curves.png", dpi=150)
        plt.close(fig)


def verify_results(config, out):
    state = read_json(out / "state.json")
    saved = torch.load(out / state["best"]["path"], map_location="cpu", weights_only=False)
    counters = sorted({int(v["step"]) for v in saved["optimizer"]["state"].values()})
    expected = config["experiment"]["parent_optimizer_steps"] + saved["state"]["optimizer_steps"]
    if counters != [expected]:
        raise ValueError("Selected Adam counters differ")
    games = 0
    for evaluation in read_json(out / "evaluations.json").values():
        count = 64 if evaluation["purpose"] == "development" else 256
        for enemies in evaluation["runs"].values():
            for entry in enemies.values():
                directory = out / entry["directory"]
                match_summary(directory, count)
                for path in directory.glob("*/metrics.json"):
                    if any(not (path.parent / n).is_file() for n in ("case.json", "replay.json", "result.json", "final.json")):
                        raise ValueError(f"Missing battle artifact: {path.parent}")
                games += count
    final = read_json(out / "confirmation.json")
    directory = out / final["candidate"]["runs"]["argmax"]["cp"]["directory"]
    representatives = {}
    for path in sorted(directory.glob("*/metrics.json")):
        m = read_json(path)
        outcome = "D" if m["winner"] is None else "W" if m["winner"] == m["side"] else "L"
        representatives.setdefault(outcome, path.parent)
    checks = {k: verify_replay(path) for k, path in representatives.items()}
    write_json(out / "verification.json", {"selected_Adam_counter": expected, "evaluation_matches": games, "replays": checks})


def run(phase, config, out, device):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with RunLock(out):
        try:
            if not (out / "run.json").exists():
                if phase not in ("prepare", "run"):
                    raise ValueError("Prepare the experiment first")
                prepare(config, out)
            elif read_json(out / "run.json")["config"] != config:
                raise ValueError("This run uses a different training configuration")
            plan = read_json(out / "manifest.json")
            if phase in ("train", "run"):
                write_json(out / "status.json", {"status": "training"})
                train(config, plan, out, device)
            if phase in ("evaluate", "run"):
                confirmation(config, plan, out)
                verify_results(config, out)
                write_json(out / "status.json", {"status": "complete", "stop_reason": read_json(out / "state.json")["stop_reason"]})
            report(config, out)
        except BaseException:
            (out / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
            write_json(out / "status.json", {"status": "technical_error", "phase": phase})
            if (out / "state.json").exists():
                report(config, out)
            raise
