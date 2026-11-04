"""Separate one-decision D0 protocol; original data/results remain immutable."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from civ5sim import Action
from civ5sim.engine import digest

from .common import ROOT, read_json, write_json
from .corpus import context_for
from .diagnostic import DecisionEnvContext
from .observation import ObservationEncoder
from .provenance import bound_cases, case_manifest


def is_one(config):
    return config.get("diagnostic", {}).get("mode") == "D0-one-decision"


def prepare_one(directory, config):
    from civ5sim import _native as native

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source_path = ROOT / config["diagnostic"]["source_dataset"]
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    path = directory / "manifest.json"
    if path.exists():
        dataset = read_json(path)
        if (
            dataset["hash"] != digest({k: v for k, v in dataset.items() if k != "hash"})
            or dataset["source_sha256"] != source_hash
            or dataset["preparation"] != config["diagnostic"]
        ):
            raise ValueError("One-decision dataset identity mismatch")
        for split in ("train", "dev"):
            bound_cases(dataset["D0"][split])
        return dataset
    source = read_json(source_path)
    if source["hash"] != digest({k: v for k, v in source.items() if k != "hash"}):
        raise ValueError("Original diagnostic dataset hash mismatch")
    dataset = {
        "schema": 1,
        "kind": "rl01.D0-one-decision",
        "source_sha256": source_hash,
        "preparation": config["diagnostic"],
        "D0": {},
    }
    for split in ("train", "dev"):
        cases = deepcopy(source["D0"][split]["cases"])
        for spec in cases:
            spec.update(diagnostic_protocol="D0-one-decision", learner_decision_limit=1)
        dataset["D0"][split] = case_manifest(
            cases,
            source="rl01.diagnostic.D0-one-decision",
            split="diagnostic_" + split,
            namespace="D0_one_decision_" + split,
        )
    proof = []
    for split in ("train", "dev"):
        original = source["D0"][split]["cases"]
        current = bound_cases(dataset["D0"][split])
        for old, new in zip(original, current):
            assert {
                k: v
                for k, v in new.items()
                if k
                not in ("provenance", "diagnostic_protocol", "learner_decision_limit")
            } == old
            with DecisionEnvContext(old) as a, DecisionEnvContext(new) as b:
                x, y = a.request(), b.request()
                if x != y:
                    raise ValueError(
                        "One-decision protocol changed the first player input"
                    )
                for variant in ("C",):
                    enc = []
                    for spec, state in ((old, x), (new, y)):
                        encoder = ObservationEncoder(
                            context_for(
                                spec["scenario"],
                                spec["armies"],
                                spec["learner_side"],
                                3,
                                0,
                            ),
                            variant,
                        )
                        enc.append(
                            encoder.encode(state["observation"], state["actions"])
                        )
                    for key in vars(enc[0]):
                        np.testing.assert_array_equal(
                            getattr(enc[0], key), getattr(enc[1], key)
                        )
                shots = [act for act in y["actions"] if act.kind == "ranged"]
                if not shots:
                    raise ValueError("One-decision D0 has no initial ranged attack")
                minima = []
                for shot in shots:
                    hp = next(
                        u["hp"]
                        for u in y["observation"]["units"]
                        if u["tile"] == shot.target
                    )
                    preview = b.battle.preview_attack(shot)
                    if (
                        native.library().sim_damage(
                            preview.attacker_strength, preview.defender_strength, 1200
                        )
                        != -1
                    ):
                        raise ValueError("Native random roll range changed")
                    minimum = min(
                        native.library().sim_damage(
                            preview.attacker_strength, preview.defender_strength, roll
                        )
                        for roll in range(1200)
                    )
                    if minimum < hp:
                        raise ValueError(
                            "Initial ranged attack is not guaranteed lethal"
                        )
                    minima.append(
                        {"minimum_damage": minimum, "target_hp": hp, "rolls": 1200}
                    )
                b.prepare(shots[0])
                b.applied(shots[0], b.battle.step(shots[0]))
                win = b.request()
                if not win["done"] or win["result"]["reward"] != 1:
                    raise ValueError("Immediate shot control did not win")
            with DecisionEnvContext(new) as env:
                env.request()
                action = Action("end_turn")
                env.prepare(action)
                env.applied(action, env.battle.step(action))
                commands = env.commands
                done = env.request()
                if (
                    not done["done"]
                    or done["result"]["reward"] != 0
                    or done["result"]["reason"] != "diagnostic_decision_limit"
                    or env.commands != commands
                ):
                    raise ValueError(
                        "Delayed shot control did not end at the first wait"
                    )
            proof.append(
                {
                    "split": split,
                    "layout_id": new["layout_id"],
                    "side": new["learner_side"],
                    "raw_encoded_actions_equal": True,
                    "immediate_win": True,
                    "first_wait_draw": True,
                    "shots": minima,
                }
            )
    if {s["physical_hash"] for s in dataset["D0"]["train"]["cases"]} & {
        s["physical_hash"] for s in dataset["D0"]["dev"]["cases"]
    }:
        raise ValueError("One-decision train/dev overlap")
    dataset["hash"] = digest(dataset)
    write_json(
        directory / "construction-verification.json",
        {"source_sha256": source_hash, "cases": proof, "verified": len(proof)},
    )
    write_json(path, dataset)
    return dataset


def load_initial_weights(policy, config):
    import torch

    path = ROOT / config["diagnostic"]["initial_checkpoint"]
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["variant"] != "C" or saved["state"]["update"] != 0:
        raise ValueError("One-decision initialization must be the original untrained C")
    policy.load_state_dict(saved["model"])
    if not all(
        torch.equal(v.detach().cpu(), saved["model"][k])
        for k, v in policy.state_dict().items()
    ):
        raise ValueError("Initial model parameters differ")
    tensors = hashlib.sha256()
    for k, v in sorted(saved["model"].items()):
        tensors.update(k.encode())
        tensors.update(v.contiguous().numpy().tobytes())
    return {
        "source_checkpoint": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "model_tensors_sha256": tensors.hexdigest(),
        "all_model_parameters_bitwise_equal": True,
        "source_optimizer_loaded": False,
        "source_training_rng_loaded": False,
    }


def gate_one(summary):
    if summary.get("games") != 128 or summary.get("wins") is None:
        raise ValueError("One-decision gate requires a complete 128-case evaluation")
    return {"passed": summary["wins"] >= 122, "minimum_wins": 122, "games": 128}


def next_state_one(state, passed):
    value = deepcopy(state)
    value["consecutive"] = value.get("consecutive", 0) + 1 if passed else 0
    if value["consecutive"] >= 2:
        value.update(
            status="passed", reason="D0_one_decision_two_successive_evaluations"
        )
    elif value["stage_updates"] >= 8:
        value.update(status="failed", reason="D0_one_decision_update_limit")
    return value


def report_one(directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    run = directory / "seed-17"
    if not (run / "run-result.json").exists():
        return
    result = read_json(run / "run-result.json")
    metrics = (
        [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
        if (run / "metrics.jsonl").exists()
        else []
    )
    text = [
        "# D0-one-decision",
        "",
        "Отдельный опыт с горизонтом одна команда learner. Исходный D0 и его формальный статус не изменены.",
        "",
        "Начальные веса побитово совпадают с прежним initial.pt; Adam, RNG обучения и счётчики созданы заново. Награда +1 за естественную победу, 0 при завершении без уничтожения. Native-ошибки остаются ошибками.",
        "",
        f"Статус: **{result['status']}**, причина `{result.get('reason', 'pause')}`. Updates: {result['update']}; решений: {result['decisions']}; время обучения: {result['training_seconds']:.2f} с; post-update оценки: {result['evaluation_seconds']:.2f} с.",
        "",
        "Критерий: не менее 122 побед из 128 в двух последовательных оценках после update. Максимум 8 updates и 1800 секунд обучения. D1, seed 42 и test-наборы не запускались.",
        "",
        "| Оценка | W/D/L | Победы | P выстрела | Gate |",
        "|---|---:|---:|---:|---|",
    ]
    points = []
    for path in sorted((run / "evaluation").glob("*/summary.json")):
        s = read_json(path)
        text.append(
            f"| {path.parent.name} | {s['wins']}/{s['draws']}/{s['losses']} | {s['win_rate']:.3%} | {s['first_winning_attack_probability']:.3%} | {s.get('gate', {}).get('passed', 'baseline')} |"
        )
        if path.parent.name.startswith("update-"):
            points.append((int(path.parent.name.split("-")[1]), s))
    text += [
        "",
        "| Update | Решений/эпизодов | PPO epochs | Adam steps | KL | Норма градиента max | Изменение весов L2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for m in metrics:
        p = m["ppo"]
        text.append(
            f"| {m['update']} | {m['rollout_decisions']}/{m['episodes']} | {p['epochs']} | {p['optimizer_steps']} | {p['kl']:.5f} | {p['gradient_norm_max']:.4f} | {m['weight_delta_l2']:.4f} |"
        )
    fourth = next((s for n, s in points if n == 4), None)
    text += [
        "",
        f"**Update 4:** {fourth['wins']}/128 побед ({fourth['win_rate']:.3%})."
        if fourth
        else "**Update 4:** не достигнут; опыт завершился раньше.",
        "",
        "Последовательности имеют длину 1; minibatch 8 последовательностей содержит 8 решений. При 8192 решениях одна полная эпоха даёт 1024 optimizer steps. Это не равный объём градиентной работы с четырьмя updates прежнего D0; LR, KL-stop и размер minibatch не подбирались.",
        "",
        "Raw outcomes, вероятность/частота пропуска атак, loss, telemetry и реплеи находятся в seed-17. Telemetry — CPU/RAM и NVML VRAM/Core/Memory Controller/Power % default TDP. Данные подтверждают только одношаговую диагностику; перенос и полезность памяти не проверялись.",
    ]
    resources = run / "resources-summary.json"
    if resources.exists():
        values = read_json(resources)["metrics"]
        text += [
            "",
            "| Ресурс всего запуска, включая оценки/прогрев | Среднее | Максимум |",
            "|---|---:|---:|",
        ]
        for key in (
            "cpu_tree_capacity_percent",
            "ram_tree_rss_bytes",
            "vram_used_bytes",
            "gpu_core_load_percent",
            "gpu_memory_controller_load_percent",
            "gpu_power_percent_tdp",
        ):
            if key in values:
                scale = 2**30 if key.endswith("bytes") else 1
                text += [
                    f"| {key} {'(GiB)' if scale > 1 else '(%)'} | {values[key]['mean'] / scale:.2f} | {values[key]['max'] / scale:.2f} |"
                ]
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    initial = read_json(run / "evaluation/initial-D0/summary.json")
    x = [0] + [n for n, _ in points]
    wins = [initial["win_rate"]] + [s["win_rate"] for _, s in points]
    probs = [initial["first_winning_attack_probability"]] + [
        s["first_winning_attack_probability"] for _, s in points
    ]
    ax.plot(x, wins, "o-", label="Dev win rate")
    ax.plot(x, probs, "o-", label="First shot probability")
    ax.axhline(0.95, color="gray", linestyle="--", label="Win threshold")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("PPO update")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(directory / "progress.png", dpi=140)
    plt.close(fig)
    text += ["", "![Прогресс](progress.png)", ""]
    (directory / "report.md").write_text("\n".join(text), encoding="utf-8")
