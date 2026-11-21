"""Outcome summaries and match artifact selection shared by experiment runners."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, TypedDict

from civ5rl.core.env import learner_outcome
from civ5rl.provenance.common import read_json

Outcome = Literal["W", "D", "L"]
CaseKey = tuple[int | str, int | str, int]
SideSource = Literal["case", "metrics"]
MatchRow = Mapping[str, Any]
DEFAULT_MATCH_FILES = (
    "case.json",
    "metrics.json",
    "result.json",
    "final.json",
    "replay.json",
)
OUTCOME_ORDER: tuple[Outcome, ...] = ("W", "D", "L")


class _WDLCounts(TypedDict):
    matches: int
    W: int
    D: int
    L: int
    win_rate: float


class WDLSummary(_WDLCounts, total=False):
    actions: dict[str, int]
    seconds: float
    mean_rounds: float
    draw_reasons: dict[str, int]


def classify_outcome(winner: int | None, learner_side: int) -> Outcome:
    """Classify the saved winner from the learner's side."""
    return learner_outcome(winner, learner_side)


def summarize_wdl(rows: Iterable[MatchRow], *, extra_metrics: bool = False) -> WDLSummary:
    """Count metric outcomes, with optional timing, actions, and draw reasons.

    Each row supplies winner and side. The optional timing and action aggregates
    follow the fields present in the metric schema. Empty inputs have a zero
    win rate. The caller checks any expected count and completion status.
    """
    counts: Counter[Outcome] = Counter()
    actions: Counter[str] = Counter()
    draw_reasons: Counter[str] = Counter()
    seconds: list[float] = []
    rounds: list[int] = []
    has_actions = False
    for row in rows:
        outcome = classify_outcome(row["winner"], row["side"])
        counts[outcome] += 1
        if extra_metrics:
            if "seconds" in row:
                seconds.append(row["seconds"])
            if "rounds" in row:
                rounds.append(row["rounds"])
            if "actions_by_kind" in row:
                has_actions = True
                actions.update(row["actions_by_kind"])
            if outcome == "D":
                draw_reasons[row["reason"]] += 1

    n = sum(counts.values())
    summary: WDLSummary = {
        "matches": n,
        "W": counts["W"],
        "D": counts["D"],
        "L": counts["L"],
        "win_rate": counts["W"] / n if n else 0.0,
    }
    if extra_metrics:
        summary["draw_reasons"] = dict(draw_reasons)
        if has_actions:
            summary["actions"] = dict(actions)
        if seconds:
            summary["seconds"] = sum(seconds)
        if rounds:
            summary["mean_rounds"] = sum(rounds) / len(rounds)
    return summary


def verify_match_artifacts(case_dir: str | Path, required_files: Iterable[str] = DEFAULT_MATCH_FILES) -> bool:
    """Check file presence for one match without loading or replaying artifacts."""
    directory = Path(case_dir)
    return all((directory / name).is_file() for name in required_files)


def collect_paired_outcomes(
    eval_dir: str | Path,
    *,
    side_source: SideSource = "case",
    check_duplicates: bool = False,
    sort: bool = True,
) -> tuple[dict[CaseKey, int], dict[CaseKey, dict[str, Any]], dict[CaseKey, Outcome]]:
    """Collect wins, case metadata, and W/D/L from saved metrics in matching order.

    World and case identifiers come from case.json and keep their types. The
    requested side source controls the key only. Outcomes use metrics winner and
    side. Missing metrics are skipped, and missing case metadata is an error.
    With duplicate checks disabled, the last matching directory wins.
    """
    if side_source not in ("case", "metrics"):
        raise ValueError(f"Unknown learner side source: {side_source}")
    paths: Iterable[Path] = Path(eval_dir).glob("*/metrics.json")
    if sort:
        paths = sorted(paths)
    wins: dict[CaseKey, int] = {}
    cases: dict[CaseKey, dict[str, Any]] = {}
    outcomes: dict[CaseKey, Outcome] = {}
    for path in paths:
        metric = read_json(path)
        case = read_json(path.parent / "case.json")
        key = (
            case["parent_world"],
            case["case_id"],
            case["learner_side"] if side_source == "case" else metric["side"],
        )
        if check_duplicates and key in wins:
            raise ValueError(f"Duplicate evaluation case {key}: {path.parent}")
        outcome = classify_outcome(metric["winner"], metric["side"])
        wins[key] = int(outcome == "W")
        cases[key] = case
        outcomes[key] = outcome
    return wins, cases, outcomes


def select_outcome_representatives(
    source: str | Path | Iterable[Path], *, wdl_order: bool = False
) -> dict[Outcome, Path]:
    """Choose the first match directory for each observed outcome.

    An evaluation directory is traversed by sorted match filename. An iterable
    of match directories preserves its incoming order.
    """
    directories: Iterable[Path]
    if isinstance(source, (str, Path)):
        directories = (p.parent for p in sorted(Path(source).glob("*/metrics.json")))
    else:
        directories = source
    selected: dict[Outcome, Path] = {}
    for directory in directories:
        row = read_json(directory / "metrics.json")
        selected.setdefault(classify_outcome(row["winner"], row["side"]), directory)
        if wdl_order and len(selected) == len(OUTCOME_ORDER):
            break
    if wdl_order:
        return {outcome: selected[outcome] for outcome in OUTCOME_ORDER if outcome in selected}
    return selected
