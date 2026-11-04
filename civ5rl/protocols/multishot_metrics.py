"""Public-observation firing sequences; privileged outcome data stays diagnostic."""

from collections import Counter

import numpy as np

from civ5rl.protocols.diagnostic_metrics import summarize_decisions


def ratio(numerator, denominator):
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
    }


def moments(values):
    if not len(values):
        return None
    x = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def sequence_metrics(ep):
    observations, actions, chosen = ep["observations"], ep["actions"], ep["chosen"]
    shots = [
        i
        for i, (acts, index) in enumerate(zip(actions, chosen))
        if acts[index].kind == "ranged"
    ]
    output = {
        "shots": len(shots),
        "decisions": len(chosen),
        "first_shot_index": shots[0] if shots else None,
        "repeat_shot": len(shots) >= 2,
        "next_own_turn_reached": False,
        "contact_opportunity": False,
        "contact_used": False,
        "contact_probability": None,
        "contact_absence_reasons": [],
        "after_first_shot": {},
    }
    if not shots:
        return output
    first = shots[0]
    shot = actions[first][chosen[first]]
    own = {u["id"]: u for u in observations[first]["units"]}
    attacker = own[shot.unit]
    target = next(
        u
        for u in own.values()
        if u["tile"] == shot.target and u["side"] != attacker["side"]
    )
    ids = (attacker["id"], target["id"])
    positions = (attacker["tile"], target["tile"])
    reasons = set()
    output["after_first_shot"] = dict(
        Counter(actions[j][chosen[j]].kind for j in range(first + 1, len(chosen)))
    )
    for j in range(first + 1, len(chosen)):
        obs = observations[j]
        units = {u["id"]: u for u in obs["units"]}
        if ids[0] not in units or any(
            u.get("hp", 1) <= 0 for i, u in units.items() if i in ids
        ):
            reasons.add("unit_destroyed")
        if ids[1] not in units:
            reasons.add("visibility_lost")
        if any(
            i in units and units[i]["tile"] != pos for i, pos in zip(ids, positions)
        ):
            reasons.add("movement")
        if obs["turn"] == observations[first]["turn"]:
            continue
        output["next_own_turn_reached"] = True
        matching = [
            k
            for k, a in enumerate(actions[j])
            if a.kind == "ranged"
            and a.unit == ids[0]
            and ids[1] in units
            and a.target == units[ids[1]]["tile"]
        ]
        if not matching:
            reasons.add("no_legal_attack")
        if not reasons:
            output["contact_opportunity"] = True
            output["contact_used"] = chosen[j] in matching
            output["contact_probability"] = float(
                np.exp(ep["old_log_probs"][j][matching]).sum()
            )
        break
    else:
        reasons.add("episode_ended")
        if ep.get("result", {}).get("reason") == "army_destroyed":
            reasons.add("unit_destroyed")
    output["contact_absence_reasons"] = sorted(reasons)
    return output


def summarize(episodes):
    rows = []
    for ep in episodes:
        r, s = ep["result"], ep["spec"]
        seq = sequence_metrics(ep)
        enemy = [
            i
            for i, u in enumerate(s["scenario"]["units"])
            if u["side"] != s["learner_side"]
        ]
        healing = sum(
            event["healing"][i]
            for event in r.get("diagnostic_events", [])
            for i in enemy
        )
        rows.append(
            {
                "episode_id": s["episode_id"],
                "layout_id": s["layout_id"],
                "side": s["learner_side"],
                "win": r["reward"] == 1,
                "reward": r["reward"],
                "reason": r["reason"],
                "rounds": r["rounds"],
                "damage_dealt": r["damage_dealt"][s["learner_side"]],
                "enemy_healing": healing,
                "remaining_hp": r["remaining_hp"],
                **seq,
            }
        )
    n = len(rows)
    counts = Counter(r["reward"] for r in rows)
    shot_rows = [r for r in rows if r["shots"]]
    opportunities = [r for r in rows if r["contact_opportunity"]]
    wins = [r for r in rows if r["win"]]
    returns = [ep["result"]["reward"] for ep in episodes for _ in ep["chosen"]]
    values = [v for ep in episodes for v in ep.get("old_value", [])]
    raw_adv = (
        np.asarray(returns) - np.asarray(values) if len(values) == len(returns) else []
    )
    normalized = (
        (raw_adv - np.mean(raw_adv)) / (np.std(raw_adv) + 1e-8) if len(raw_adv) else []
    )
    result = {
        "games": n,
        "wins": counts[1],
        "draws": counts[0],
        "losses": counts[-1],
        "win_rate": counts[1] / n,
        "actions": summarize_decisions(episodes),
        "shots": dict(Counter(r["shots"] for r in rows)),
        "draw_reasons": dict(Counter(r["reason"] for r in rows if r["reward"] == 0)),
        "decisions": moments([r["decisions"] for r in rows]),
        "rounds": moments([r["rounds"] for r in rows]),
        "damage_dealt": moments([r["damage_dealt"] for r in rows]),
        "enemy_healing": moments([r["enemy_healing"] for r in rows]),
        "wins_with_two_or_more_shots": ratio(
            sum(r["shots"] >= 2 for r in wins), len(wins)
        ),
        "repeat_shot_any": ratio(
            sum(r["repeat_shot"] for r in shot_rows), len(shot_rows)
        ),
        "next_own_turn_after_shot": ratio(
            sum(r["next_own_turn_reached"] for r in shot_rows), len(shot_rows)
        ),
        "contact_opportunity_reached": ratio(len(opportunities), len(shot_rows)),
        "contact_immediate_use": ratio(
            sum(r["contact_used"] for r in opportunities), len(opportunities)
        ),
        "contact_probability": ratio(
            sum(r["contact_probability"] for r in opportunities), len(opportunities)
        ),
        "contact_absence_reasons": dict(
            Counter(
                reason for r in shot_rows for reason in r["contact_absence_reasons"]
            )
        ),
        "after_first_shot": dict(
            sum((Counter(r["after_first_shot"]) for r in rows), Counter())
        ),
        "episode_return": moments([r["reward"] for r in rows]),
        "token_return": moments(returns),
        "old_value": moments(values),
        "raw_advantage": moments(raw_adv),
        "normalized_advantage": moments(normalized),
        "explained_variance": None
        if not len(raw_adv) or np.var(returns) == 0
        else float(1 - np.var(raw_adv) / np.var(returns)),
    }
    return result, rows


def comparison(rows, baseline):
    def paired(values):
        result = {(r["layout_id"], r["side"]): int(r["win"]) for r in values}
        if len(values) != 128 or len(result) != 128:
            raise ValueError("Bootstrap requires 128 unique paired games")
        return result

    a, b = paired(rows), paired(baseline)
    if a.keys() != b.keys():
        raise ValueError("Bootstrap cases differ")
    layouts = sorted({k[0] for k in a})
    if len(layouts) != 64 or any(
        (i, side) not in a for i in layouts for side in (0, 1)
    ):
        raise ValueError("Bootstrap requires 64 complete layouts")
    delta = np.asarray(
        [sum(a[i, side] - b[i, side] for side in (0, 1)) / 2 for i in layouts]
    )
    rng = np.random.default_rng(20261103)
    ci = np.quantile(delta[rng.integers(64, size=(10000, 64))].mean(1), [0.025, 0.975])
    return {"delta": float(delta.mean()), "ci95": ci.tolist(), "descriptive_only": True}


def gate(summary):
    if summary["games"] != 128:
        raise ValueError("Incomplete multishot evaluation")
    return summary["wins"] >= 122


def transition(state, passed):
    value = dict(state)
    value["consecutive"] = state["consecutive"] + 1 if passed else 0
    if value["consecutive"] >= 2:
        value.update(status="passed", reason="two_successive_evaluations")
    elif value["update"] >= 8:
        value.update(status="failed_update_limit", reason="eight_updates_completed")
    elif value["training_seconds"] >= 1800:
        value.update(status="failed_training_time_limit", reason="training_time_limit")
    return value
