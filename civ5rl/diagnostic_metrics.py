"""Action diagnostics computed exclusively from legal commands and policy output."""

from collections import Counter

import numpy as np


def decision_metrics(actions, chosen, probabilities):
    result = {"chosen_kind": actions[chosen].kind}
    for label, kinds in (
        ("attack", ("melee", "ranged")),
        ("ranged", ("ranged",)),
        ("melee", ("melee",)),
    ):
        mask = np.asarray([a.kind in kinds for a in actions])
        result[label + "_available"] = bool(mask.any())
        result[label + "_chosen"] = bool(mask[chosen])
        result[label + "_probability"] = float(np.asarray(probabilities)[mask].sum())
    result["move_available"] = any(a.kind == "move" for a in actions)
    return result


def summarize_decisions(episodes):
    rows = [r for ep in episodes for r in ep["decision_metrics"]]
    output = {
        "decisions": len(rows),
        "actions_by_kind": dict(Counter(r["chosen_kind"] for r in rows)),
    }
    for label in ("attack", "ranged", "melee"):
        available = [r for r in rows if r[label + "_available"]]
        output[label] = {
            "available_decisions": len(available),
            "chosen": sum(r[label + "_chosen"] for r in rows),
            "chosen_when_available": sum(r[label + "_chosen"] for r in available)
            / len(available)
            if available
            else None,
            "probability_when_available": float(
                np.mean([r[label + "_probability"] for r in available])
            )
            if available
            else None,
        }
    for label in ("attack", "move"):
        available = [r for r in rows if r[label + "_available"]]
        output["end_turn_when_" + label + "_available"] = (
            sum(r["chosen_kind"] == "end_turn" for r in available) / len(available)
            if available
            else None
        )
    return output
