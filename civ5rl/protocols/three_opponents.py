"""Selection and stopping rules for the CPv2 / CP / u0 continuation."""

DEV_TARGET = 384
ADDITIONAL_TARGET = 192
EARLY_TARGET = 392
EARLY_GOAL = "dual_target_achieved"


def results(entry):
    cp = entry["runs"]["argmax"]["cp"]["summary"]["W"]
    cpv2 = entry["cpv2"]["runs"]["argmax"]["cpv2"]["summary"]["W"]
    h2h = entry["h2h"]["summary"]
    return cpv2, cp, h2h["W"] - h2h["L"]


def meets(entry, minimum):
    cpv2, cp, balance = results(entry)
    return cpv2 >= minimum and cp >= minimum and balance >= 0


def select(points, *, continuation=None):
    baseline = points["0"]
    baseline_v2, baseline_cp, _ = results(baseline)
    floor = DEV_TARGET if any(results(item)[1] >= DEV_TARGET for item in points.values()) else baseline_cp
    old_policies = {item["policy_hash"] for point, item in points.items()
                    if continuation and int(point) <= continuation["milestone"]}
    candidates = []
    for point, item in points.items():
        cpv2, cp, balance = results(item)
        if not int(point) or balance < 0:
            continue
        if continuation:
            if int(point) <= continuation["milestone"] or item["policy_hash"] in old_policies or not meets(item, DEV_TARGET):
                continue
        elif cpv2 <= baseline_v2 or cp < floor:
            continue
        minimum_slice = min(series["slices"]["placement"][placement]["W"]
                            for series in (item, item["cpv2"]) for placement in ("contact", "separated"))
        twice_j = 4 * (cpv2 - DEV_TARGET) + 2 * (cp - DEV_TARGET) + balance
        rank = (meets(item, DEV_TARGET), twice_j, minimum_slice, int(point))
        candidates.append((rank, item, cpv2, cp, balance))
    if not candidates:
        return {"path": baseline["checkpoint"], "wins": baseline_cp, "update": 0,
                "policy_hash": baseline["policy_hash"], "selection_rule": "three_opponents", "milestone": 0}
    rank, item, cpv2, cp, balance = max(candidates, key=lambda candidate: candidate[0])
    return {"path": item["checkpoint"], "wins": cp, "update": item["update"],
            "policy_hash": item["policy_hash"], "selection_rule": "three_opponents", "milestone": rank[-1],
            "J": rank[1] / 2, "cpv2_wins": cpv2, "cp_wins": cp, "h2h_balance": balance,
            "cpv2_gain": cpv2 - baseline_v2, "cp_gain": cp - baseline_cp}


def stop_reason(points, milestones, *, gain, gain_boundary, continuation=None):
    if continuation:
        return None
    early = [str(point) for point in milestones if point <= gain_boundary]
    if early and all(point in points for point in early) and max(results(points[point])[0] for point in early) < results(points["0"])[0] + gain:
        return "no_development_gain"
    for previous, current in zip((0, *milestones), milestones):
        if all(str(point) in points and meets(points[str(point)], EARLY_TARGET) for point in (previous, current)):
            return EARLY_GOAL
    return None


def initial_promotion(cpv2_comparison, cp_comparison, entry):
    _, _, balance = results(entry)
    promoted = cpv2_comparison["delta"] > 0 and cpv2_comparison["CI95"][0] > 0 and cp_comparison["delta"] >= 0 and balance >= 0
    return {"promoted": promoted, "dual_target_confirmed": promoted and meets(entry, ADDITIONAL_TARGET)}


def allows_continuation(entry):
    cpv2, cp, balance = results(entry)
    return 188 <= min(cpv2, cp) < ADDITIONAL_TARGET and balance >= 0
