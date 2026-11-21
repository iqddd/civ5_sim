"""Summarize collected rollouts for PPO experiments."""

from collections import Counter


def summarize_rollout(episodes):
    count = Counter(
        "D"
        if episode["result"]["winner"] is None
        else "W"
        if episode["result"]["winner"] == episode["spec"]["learner_side"]
        else "L"
        for episode in episodes
    )
    free = sum(sum(len(actions) >= 2 for actions in episode["actions"]) for episode in episodes)
    total = sum(len(episode["chosen"]) for episode in episodes)
    summary = {
        "episodes": len(episodes),
        "W": count["W"],
        "D": count["D"],
        "L": count["L"],
        "win_rate": count["W"] / len(episodes),
        "decisions": total,
        "free_decisions": free,
        "forced_decisions": total - free,
        "positive_reward_episodes": count["W"],
    }
    if any(episode["spec"].get("opponent") == "policy" for episode in episodes):
        buckets = {}
        for episode in episodes:
            spec = episode["spec"]
            if spec.get("opponent") != "policy":
                bucket = "cp"
            else:
                opponent_id = spec.get("opponent_id") or "current"
                bucket = "archive" if opponent_id.startswith("archive:") else opponent_id
            row = buckets.setdefault(bucket, Counter())
            outcome = (
                "D"
                if episode["result"]["winner"] is None
                else "W"
                if episode["result"]["winner"] == spec["learner_side"]
                else "L"
            )
            row[outcome] += 1
            row["episodes"] += 1
            row["decisions"] += len(episode["chosen"])
        summary["by_opponent"] = {
            name: {
                "episodes": int(counts["episodes"]),
                "W": int(counts["W"]),
                "D": int(counts["D"]),
                "L": int(counts["L"]),
                "win_rate": counts["W"] / max(1, counts["episodes"]),
                "decisions": int(counts["decisions"]),
            }
            for name, counts in sorted(buckets.items())
        }
    return summary
