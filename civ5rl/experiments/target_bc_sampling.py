"""One CPU-only, paired evaluation of the frozen target BC policy."""

import platform
import random
import sys
import traceback
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from civ5rl.config import load_config
from civ5rl.core.agent import RLAgent, action_probabilities
from civ5rl.core.context import context_from_spec
from civ5rl.core.observation import ordered_actions
from civ5rl.engine.control import RunLock
from civ5rl.engine.pool import pool
from civ5rl.experiments.evaluation import paired_world_bootstrap
from civ5rl.experiments.ppo_after_imitation import play_case
from civ5rl.experiments.ppo_artifacts import load_rollout
from civ5rl.provenance.common import (
    ROOT,
    choose,
    configure,
    file_sha256,
    read_json,
    write_json,
)
from civ5sim.engine import digest

PARENT_SHA256 = "4677fa13b2ac7a84214c545a6b311fd68bc05b31b82fe00fbe63c8856fc6e985"
MANIFEST_SHA256 = "5b60a2e2f272c66bede9f65867da0854d54d6d5d4a4e1ef23785a3454020465c"
MODES = [
    {"label": "argmax", "mode": "argmax", "temperature": 1.0},
    {"label": "sample-t1", "mode": "sample", "temperature": 1.0},
    {"label": "sample-t0.5", "mode": "sample", "temperature": 0.5},
]


def execution_identity():
    names = [
        p
        for package in ("civ5rl", "civ5sim", "civ5map")
        for p in (ROOT / package).rglob("*.py")
    ]
    names.append(ROOT / "tools/archive/evaluate_target_bc_sampling.py")
    suffixes = (
        ("combat.dll", "combat_ai.dll")
        if sys.platform == "win32"
        else ("libcombat.so", "libcombat_ai.so")
    )
    names += [
        ROOT / "build" / name
        for name in (
            *suffixes,
            "combat_rules.h",
            "combat_rules.json",
            "cp_geometry.inc",
        )
    ]
    return {
        "files": {
            p.relative_to(ROOT).as_posix(): file_sha256(p) for p in sorted(names)
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "workers": 4,
        "torch_threads_per_worker": 1,
    }


def mode_key(identity, setting):
    return digest(
        {
            "experiment": identity,
            "mode": setting["mode"],
            "temperature": setting["temperature"],
        }
    )


def inputs(settings):
    if (
        settings["workers"] != 4
        or settings["modes"] != MODES
        or settings["bootstrap_samples"] != 10000
        or settings["bootstrap_seed"] != 20261103
    ):
        raise ValueError(
            "This experiment is fixed to three modes, four workers and the agreed bootstrap"
        )
    parent = ROOT / settings["checkpoint"]
    if (
        settings["checkpoint_sha256"] != PARENT_SHA256
        or file_sha256(parent) != PARENT_SHA256
    ):
        raise ValueError("Expected original BC epoch 10, before critic preparation/PPO")
    manifest = ROOT / settings["manifest"]
    if file_sha256(manifest) != MANIFEST_SHA256:
        raise ValueError("Target BC manifest changed")
    plan = read_json(manifest)
    if digest({k: v for k, v in plan.items() if k != "hash"}) != plan["hash"]:
        raise ValueError("Invalid target manifest digest")
    cases = plan["development"]
    config = load_config(ROOT / settings["evaluator_config"])
    if len(cases) != 128 or len({c["case_hash"] for c in cases}) != 128:
        raise ValueError("Expected exactly 128 unique development cases")
    worlds = Counter(c["parent_world"] for c in cases)
    if len(worlds) != 16 or set(worlds.values()) != {8}:
        raise ValueError("Expected 16 parent worlds, eight paired cases per world")
    for case in cases:
        if (
            case["split"] != "validation"
            or case["max_rounds"] != 30
            or case["opponent"] != "cp"
            or case["max_calls_per_turn"] != config["environment"]["max_calls_per_turn"]
            or case["opponent_search"] != config["teacher_search"]
            or digest({k: v for k, v in case.items() if k != "case_hash"})
            != case["case_hash"]
        ):
            raise ValueError("Development conditions changed")
    identity = {
        "checkpoint_sha256": PARENT_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "ordered_case_hashes": [c["case_hash"] for c in cases],
        "evaluator_config": config,
        "execution": execution_identity(),
    }
    return config, cases, identity


def preflight(settings, out):
    """Real BC logits on saved public states; no new battle or model training."""
    configure(17, "cpu")
    episode = load_rollout(
        ROOT / "artifacts/rl/ppo-after-target-bc/rollouts/latest.pt"
    )[0]
    agent = RLAgent.from_checkpoint(ROOT / settings["checkpoint"], "cpu")
    agent.reset(context_from_spec(episode["spec"]))
    tensors, checks = [], []
    with torch.no_grad():
        for index in range(3):
            observation = episode["observations"][index]
            actions = ordered_actions(episode["actions"][index])
            logits = agent._infer(observation, actions)
            old = logits.softmax(-1)[0].cpu().numpy()
            new = action_probabilities(logits, 1)[0].cpu().numpy()
            np.testing.assert_array_equal(old, new)
            before, after = random.Random(20261103), random.Random(20261103)
            assert [choose(old, before) for _ in range(64)] == [
                choose(new, after) for _ in range(64)
            ]
            assert before.getstate() == after.getstate()
            colder = action_probabilities(logits, 0.5)
            assert int(colder[0].argmax()) == int(logits[0].argmax())
            assert torch.all(colder[torch.isneginf(logits)] == 0)
            tensors.append(logits.cpu())
            checks.append(
                {
                    "saved_observation": index,
                    "legal_actions": len(actions),
                    "T1_probabilities_and_seeded_choices_exact": True,
                    "argmax_and_mask_preserved": True,
                }
            )
            agent.encoder.remember_action(
                episode["actions"][index][episode["chosen"][index]], observation
            )
    torch.save(tensors, out / "preflight-logits.pt")
    write_json(
        out / "preflight.json",
        {
            "checkpoint_sha256": PARENT_SHA256,
            "checks": checks,
            "new_matches": 0,
            "device": "cpu",
        },
    )


def completed_case(directory, case, key):
    names = (
        "metrics.json",
        "case.json",
        "result.json",
        "replay.json",
        "final.json",
        "selection-metrics.json",
    )
    if not all((directory / name).is_file() for name in names):
        return False
    metric = read_json(directory / "metrics.json")
    return (
        metric.get("status") == "completed"
        and metric.get("evaluation_key") == key
        and metric.get("case_hash") == case["case_hash"]
        and read_json(directory / "case.json") == case
        and read_json(directory / "result.json").get("status") == "completed"
    )


def match_task(task):
    config, case, parent, setting, directory, key = task
    directory = Path(directory)
    result = play_case(
        config,
        case,
        parent,
        setting["mode"],
        "cp",
        directory,
        temperature=setting["temperature"],
        collect_selection_metrics=True,
    )
    result.update(
        evaluation_key=key, mode=setting["mode"], temperature=setting["temperature"]
    )
    write_json(directory / "metrics.json", result)
    return result


def case_directory(out, setting, case):
    return (
        out
        / setting["label"]
        / f"case-{case['case_id']:03}-side-{case['learner_side']}"
    )


def wdl(rows):
    counts = Counter(
        "D" if r["winner"] is None else "W" if r["winner"] == r["side"] else "L"
        for r in rows
    )
    return {
        "matches": len(rows),
        **{k: counts[k] for k in ("W", "D", "L")},
        "win_rate": counts["W"] / len(rows),
        "mean_rounds": float(np.mean([r["rounds"] for r in rows])),
        "median_rounds": float(np.median([r["rounds"] for r in rows])),
        "turn_limit_draw_rate": sum(
            r["winner"] is None and r["reason"] == "turn_limit" for r in rows
        )
        / len(rows),
    }


def paired(cases, rows):
    return {
        (c["parent_world"], c["case_id"], c["learner_side"]): int(
            r["winner"] == r["side"]
        )
        for c, r in zip(cases, rows, strict=True)
    }


def report(settings, cases, identity, out):
    modes, outcomes = {}, {}
    wall_times = read_json(out / "run.json").get("mode_wall_seconds", {})
    for setting in settings["modes"]:
        key = mode_key(identity, setting)
        rows, decisions = [], []
        for case in cases:
            directory = case_directory(out, setting, case)
            if not completed_case(directory, case, key):
                continue
            row = read_json(directory / "metrics.json")
            # Independently verify W/D/L against the match completion record.
            raw = read_json(directory / "result.json")
            if row["winner"] != raw["winner"] or row["reason"] != raw["reason"]:
                raise ValueError("Metrics differ from saved match result")
            rows.append(row)
            decisions.extend(read_json(directory / "selection-metrics.json"))
        if len(rows) != 128:
            write_json(
                out / "summary.json",
                {
                    "status": "incomplete",
                    "mode": setting["label"],
                    "completed": len(rows),
                },
            )
            (out / "report.md").write_text(
                f"# Неполная оценка\n\n{setting['label']}: {len(rows)}/128; итоговое сравнение не готово.\n",
                encoding="utf-8",
            )
            return False
        slices = {}
        for field in ("map_size", "army_bin", "placement", "learner_side"):
            slices[field] = {
                str(value): wdl(
                    [r for c, r in zip(cases, rows, strict=True) if c[field] == value]
                )
                for value in sorted({c[field] for c in cases}, key=str)
            }
        behavior = {"free_decisions": len(decisions)}
        for field in (
            "p_max",
            "argmax_selected",
            "normalized_entropy",
            "legal_actions",
        ):
            values = [r[field] for r in decisions]
            behavior[field] = (
                {"mean": float(np.mean(values)), "median": float(np.median(values))}
                if values
                else {"mean": None, "median": None}
            )
        modes[setting["label"]] = {
            **wdl(rows),
            "match_seconds_sum": sum(r["seconds"] for r in rows),
            "wall_seconds": wall_times.get(setting["label"]),
            "behavior": behavior,
            "slices": slices,
            "evaluation_key": key,
            "mode": setting["mode"],
            "temperature": setting["temperature"],
        }
        outcomes[setting["label"]] = paired(cases, rows)
    metadata = {(c["parent_world"], c["case_id"], c["learner_side"]): c for c in cases}
    comparison = paired_world_bootstrap(
        outcomes["sample-t1"],
        outcomes["sample-t0.5"],
        samples=10000,
        seed=20261103,
        clusters=16,
        cases=metadata,
        cases_per_world=8,
    )
    result = {
        "status": "complete",
        "modes": modes,
        "T05_minus_T1": comparison,
        "versus_argmax": {
            label: modes[label]["win_rate"] - modes["argmax"]["win_rate"]
            for label in ("sample-t1", "sample-t0.5")
        },
        "T05_at_least_13_wins": modes["sample-t0.5"]["W"] >= 13,
        "positive_lower_CI": comparison["CI95"][0] > 0,
        "paired_case_hashes_verified": True,
        "clusters": 16,
        "cases_per_world": 8,
    }
    write_json(out / "summary.json", result)
    lines = [
        "# Target BC: режим выбора действий",
        "",
        "CPU / 4 workers / один поток Torch; исходный BC epoch 10; ordinary stateful CP; 128 development случаев на режим.",
        "",
        "| Режим | W/D/L | Победы | Round mean/median | Ничьи по лимиту | Wall, с |",
        "|---|---|---:|---|---:|---:|",
    ]
    for label, row in modes.items():
        lines.append(
            f"| {label} | {row['W']}/{row['D']}/{row['L']} | {row['win_rate']:.2%} | {row['mean_rounds']:.2f}/{row['median_rounds']:.1f} | {row['turn_limit_draw_rate']:.2%} | {row['wall_seconds']:.1f} |"
        )
    lines += [
        "",
        f"T=0.5 − T=1: {comparison['delta']:+.2%}; 95% CI [{comparison['CI95'][0]:+.2%}, {comparison['CI95'][1]:+.2%}]. Bootstrap 10000, seed 20261103, 16 parent-world × 8 матчей.",
        "",
        "| Режим | Free decisions | p_max mean/median | Argmax share | Entropy/log(N) mean/median | Legal N mean/median |",
        "|---|---:|---|---:|---|---|",
    ]
    for label, row in modes.items():
        b = row["behavior"]

        def pair(k, b=b):
            return f"{b[k]['mean']:.4f}/{b[k]['median']:.4f}"

        lines.append(
            f"| {label} | {b['free_decisions']} | {pair('p_max')} | {b['argmax_selected']['mean']:.2%} | {pair('normalized_entropy')} | {pair('legal_actions')} |"
        )
    lines += [
        "",
        "Forced исключены из метрик уверенности. Для argmax распределение метрик — T=1. Посещённые состояния расходятся; эти метрики не являются парной оценкой качества действий.",
        "",
        f"T=0.5 ≥13 побед: {result['T05_at_least_13_wins']}; положительная нижняя граница CI: {result['positive_lower_CI']}.",
        "Ориентир 13 побед — минимальный сигнал, не гарантия полезности PPO. Выбор сделан на development; дополнительного подтверждения обобщения нет. Новый PPO, температуры и seeds автоматически не запускаются.",
        "",
        "Полные срезы: [summary.json](summary.json). Парные случаи и seeds: [cases.json](cases.json). Условия и fingerprint: [run.json](run.json).",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def run(settings, phase="run"):
    config, cases, identity = inputs(settings)
    out = ROOT / settings["output"]
    out.mkdir(parents=True, exist_ok=True)
    with RunLock(out):
        if phase == "check":
            preflight(settings, out)
            print(
                "Preflight passed: three saved BC logits, exact T=1 choices; no matches run.",
                flush=True,
            )
            return True
        if (out / "run.json").exists():
            record = read_json(out / "run.json")
            if record["identity"] != identity or record["settings"] != settings:
                raise ValueError(
                    "Existing evaluation has different inputs or evaluator; use a new output"
                )
        else:
            record = {
                "identity": identity,
                "settings": settings,
                "created_utc": datetime.now(UTC).isoformat(),
                "case_seeds": [
                    {
                        "case_hash": c["case_hash"],
                        "battle": c["scenario"]["seed"],
                        "agent_seeds": c["agent_seeds"],
                    }
                    for c in cases
                ],
            }
            write_json(out / "run.json", record)
            write_json(out / "cases.json", cases)
        if phase == "report":
            return report(settings, cases, identity, out)
        try:
            preflight(settings, out)
            write_json(out / "status.json", {"status": "running"})
            started = perf_counter()
            new_matches = 0
            pending = {}
            for setting in settings["modes"]:
                key = mode_key(identity, setting)
                pending[setting["label"]] = [
                    (
                        config,
                        c,
                        str(ROOT / settings["checkpoint"]),
                        setting,
                        str(case_directory(out, setting, c)),
                        key,
                    )
                    for c in cases
                    if not completed_case(case_directory(out, setting, c), c, key)
                ]
            remaining = sum(map(len, pending.values()))
            with pool(4) as executor:
                for setting in settings["modes"]:
                    tasks = pending[setting["label"]]
                    tick = perf_counter()
                    for first in range(0, len(tasks), 4):
                        part = list(executor.map(match_task, tasks[first : first + 4]))
                        new_matches += len(part)
                        remaining -= len(part)
                        elapsed = perf_counter() - started
                        print(
                            f"{setting['label']}: {min(first + 4, len(tasks))}/{len(tasks)} new matches; elapsed={elapsed:.1f}s; ETA~{elapsed / new_matches * remaining:.0f}s",
                            flush=True,
                        )
                    if tasks:
                        record.setdefault("mode_wall_seconds", {})[setting["label"]] = (
                            perf_counter() - tick
                        )
                    write_json(out / "run.json", record)
            if (
                execution_identity() != identity["execution"]
                or file_sha256(ROOT / settings["checkpoint"]) != PARENT_SHA256
            ):
                raise ValueError("Evaluator or parent changed during evaluation")
            if not report(settings, cases, identity, out):
                raise ValueError("Incomplete comparison")
            record["finished_utc"] = datetime.now(UTC).isoformat()
            write_json(out / "run.json", record)
            write_json(
                out / "status.json", {"status": "complete", "new_matches": new_matches}
            )
            print((out / "report.md").read_text(encoding="utf-8"), flush=True)
            return True
        except BaseException:
            (out / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
            write_json(out / "status.json", {"status": "technical_error"})
            raise
