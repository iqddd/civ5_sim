"""Saved PPO-after-target-BC report."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from civ5rl.engine.ppo.adam import expected_counters, uses_grouped_adam
from civ5rl.engine.runtime.telemetry_analysis import summarize_performance
from civ5rl.provenance.common import read_json, write_json
from civ5rl.provenance.jsonl import read_jsonl
from civ5rl.reporting.markdown import MarkdownReport, format_wdl
from civ5rl.reporting.telemetry import performance_markdown


def _development_rows(state: dict[str, Any], *, dual: bool) -> Iterator[tuple[object, ...]]:
    baseline_wins = state.get("baseline_wins") if dual else None
    for point in state["evaluated"].values():
        summary = point["runs"]["argmax"]["cp"]["summary"]
        if not dual:
            yield point["decisions"], point["checkpoint"], format_wdl(summary)
            continue
        h2h_summary = point.get("h2h", {}).get("summary")
        cp_delta = int(summary["W"]) - int(baseline_wins) if baseline_wins is not None else None
        if h2h_summary is not None:
            h2h_delta = int(h2h_summary["W"]) - int(h2h_summary["L"])
            h2h_wdl = format_wdl(h2h_summary)
            joint_score = cp_delta + h2h_delta if cp_delta is not None else None
        else:
            h2h_delta = None
            h2h_wdl = "—"
            joint_score = None
        yield (
            point["decisions"],
            point["checkpoint"],
            format_wdl(summary),
            f"{cp_delta:+d}" if cp_delta is not None else "—",
            h2h_wdl,
            f"{h2h_delta:+d}" if h2h_delta is not None else "—",
            f"{joint_score:+d}" if joint_score is not None else "—",
        )


def _rollout_report_rows(rows: list[dict[str, Any]]) -> Iterator[tuple[object, ...]]:
    for row in rows:
        metrics = row["ppo"]
        yield (
            row["rollout"],
            format_wdl(row),
            metrics.get("raw_advantage_std"),
            f"{metrics.get('critic_mse')}/{metrics.get('critic_explained_variance')}/{metrics.get('critic_mean')}",
            metrics.get("entropy"),
            metrics.get("final_accepted_kl"),
            f"{metrics['optimizer_steps']}/{metrics['rejected_optimizer_steps']}",
            metrics["PPO_skipped"],
        )


def _write_performance_reports(out: Path) -> None:
    for name, last in (("performance", None), ("performance-last10", 10)):
        performance = summarize_performance(out, last=last)
        write_json(out / f"{name}.json", performance)
        (out / f"{name}.md").write_text(performance_markdown(performance), encoding="utf-8")


def _report_three(out, state):
    from civ5rl.protocols.three_opponents import results

    lines = MarkdownReport(["# C2PA против CPv2, CP и u0", "",
                            f"{state['decisions']:,} решений learner, {state['rollouts']} rollout, {state['optimizer_steps']} принятых шагов Adam.",
                            f"Активное время {state['active_seconds']:.0f} с. Статус `{state['stop_reason'] or state['phase']}`.", ""])
    points = state.get("three_opponent_points", {})
    if points:
        lines.table(("Веха", "Checkpoint", "CPv2 W/D/L", "CP W/D/L", "H2H W/D/L", "J"),
                    ("right", "left", "right", "right", "right", "right"),
                    ((point, row["checkpoint"], format_wdl(row["cpv2"]["runs"]["argmax"]["cpv2"]["summary"]),
                      format_wdl(row["runs"]["argmax"]["cp"]["summary"]), format_wdl(row["h2h"]["summary"]),
                      2 * (v2 - 384) + (cp - 384) + balance / 2)
                     for point, row in points.items() for v2, cp, balance in (results(row),)))
        lines += ["", f"Выбран `{state['best']['path']}`."]
    for filename in ("confirmation-first.json", "confirmation.json"):
        if not (out / filename).exists():
            continue
        record = read_json(out / filename)
        lines += ["", f"## Подтверждение {record['attempt']}", ""]
        if record.get("status"):
            lines.append("Допустимый новый кандидат отсутствует.")
            continue
        for label in ("additional", "reserve"):
            if label in record:
                v2, cp, balance = results(record[label]["candidate"])
                lines.append(f"{label.capitalize()} — CPv2 {v2}/256, CP {cp}/256, очный баланс {balance:+d}.")
        lines.append(f"Продвижение `{record['promotion']['promoted']}`. Двойной порог `{record['promotion']['dual_target_confirmed']}`.")
    if (out / "updates.jsonl").exists():
        _write_performance_reports(out)
        lines += ["", "[Профиль времени и GPU](performance.md) · [Метрики rollout](updates.jsonl)"]
    lines.write(out / "report.md")


def report(config: dict[str, Any], out: Path) -> None:
    if config["experiment"].get("three_opponent_evaluation"):
        _report_three(out, read_json(out / "state.json"))
        return
    state = read_json(out / "state.json")
    parent_label = config["experiment"].get("parent_label", "target BC epoch 10")
    eval_purpose = config.get("ppo", {}).get("evaluation_purpose", "development")
    max_rounds = config.get("environment", {}).get("max_rounds", 30)
    gamma = config["ppo"].get("gamma", 1)
    gae_lambda = config["ppo"].get("gae_lambda", 1)
    sp = config.get("self_play", {})
    dual = bool(sp.get("enabled"))
    opponent_desc = (
        f"смешанный пул (CP {sp['cp_share']:.0%} / current {sp['current_share']:.0%} / "
        f"u0 {sp['parent_share']:.0%} / archive {sp['archive_share']:.0%}, "
        f"T_opp={sp.get('opponent_temperature', 0.5)}, архив={len(state.get('self_play_archive', []))})"
        if dual
        else "обычный CP"
    )
    lines = MarkdownReport(
        [
            f"# PPO после {parent_label}",
            "",
            f"Статус: {state['stop_reason'] or state['phase']}; решения {state['decisions']}; active {state['active_seconds']:.2f} с.",
            f"Лимит active: {state.get('active_seconds_limit', config['experiment']['active_seconds']):.0f} с; исходный бюджет {config['experiment']['active_seconds']} с; продления в active-budget.json.",
            f"Sampling rollouts {state['rollouts']}; PPO updates {state['updates']}; accepted/rejected Adam {state['optimizer_steps']}/{state['rejected_steps']}.",
            (
                f"Adam groups: {state.get('Adam_counters', expected_counters(state))}; critic preparation {state.get('critic_progress', {}).get('steps', 0)}; new PPO accepted {state['optimizer_steps']}."
                if uses_grouped_adam(config)
                else f"Adam: inherited {state.get('origin_Adam_counter', 0)}; new accepted {state['optimizer_steps']}; total {state.get('origin_Adam_counter', 0) + state['optimizer_steps']}."
            ),
            f"GPU {read_json(out / 'run.json')['GPU']}; T={config.get('policy_temperature', 1)}; max_rounds={max_rounds}; gamma={gamma}; gae_lambda={gae_lambda}; microbatch {config['ppo']['sequences_per_microbatch']}; {opponent_desc}; reward 1/0/0.",
            "",
        ]
    )
    if dual:
        lines.table(
            (
                "Решения",
                "Checkpoint",
                f"W/D/L {eval_purpose} (CP)",
                "ΔW_cp",
                "W/D/L H2H vs u0",
                "ΔW_h2h",
                "J",
            ),
            ("right", "left", "left", "right", "left", "right", "right"),
            _development_rows(state, dual=True),
        )
    else:
        lines.table(
            ("Решения", "Checkpoint", f"W/D/L {eval_purpose}"),
            ("right", "left", "left"),
            _development_rows(state, dual=False),
        )
    lines += ["", f"Выбор: {state['best']}."]
    if (out / "runtime.json").exists():
        lines += [
            "",
            "Runtime: " + json.dumps(read_json(out / "runtime.json"), ensure_ascii=False),
        ]
    if (out / "updates.jsonl").exists():
        _write_performance_reports(out)
        lines += [
            "",
            "Performance: [all cycles](performance.md), [last 10](performance-last10.md).",
        ]
        rows = read_jsonl(out / "updates.jsonl", skip_blank=False, encoding=None)
        lines.append("")
        lines.table(
            (
                "Rollout",
                "Sampling W/D/L",
                "Raw advantage std",
                "Critic MSE / EV / mean",
                "Entropy",
                "Full KL",
                "Adam accept/reject",
                "PPO skipped",
            ),
            ("right", "left", "right", "left", "right", "right", "left", "left"),
            _rollout_report_rows(rows),
        )
        lines += [
            "",
            f"Rollout без побед: {sum(row['W'] == 0 for row in rows)}/{len(rows)}.",
        ]
    for point in state["evaluated"].values():
        lines += [
            "",
            f"Срезы development {point['checkpoint']}",
            "",
            json.dumps(point.get("slices", {}), ensure_ascii=False),
        ]
    if state["stop_reason"] == "no_development_gain":
        lines += ["", "Прироста в установленном раннем бюджете не получено."]
    lines += [
        "",
        (
            f"Ранний порог: {state.get('early_gain_required_wins', 'по конфигу')} побед после {config['experiment']['early_gain_decisions']} новых решений; полезность оценивается по парному additional-приросту и CI."
            if config["experiment"].get("early_gain_enabled", True)
            else (
                "Ранний порог продолжения отключён; двухлинейный выбор по Development-512 (CP) и H2H-512 (ΔW_cp ≥ 0, ΔW_h2h ≥ 0, max J), проверка нового кандидата на Additional-256 (CP + H2H)."
                if dual
                else "Ранний порог продолжения отключён; выбор по Development, проверка нового кандидата на парном Additional."
            )
        ),
    ]
    for name in ("critic-preparation.json", "confirmation.json"):
        if (out / name).exists():
            lines += [
                "",
                name,
                "",
                json.dumps(read_json(out / name), ensure_ascii=False),
            ]
    lines.write(out / "report.md")
