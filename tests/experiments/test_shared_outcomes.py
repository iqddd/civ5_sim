from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from civ5rl.experiments.evaluation.paired import (
    evaluate_paired_dev512,
    run_chunked_pool,
)
from civ5rl.reporting.evaluation_audit import (
    MatchRow,
    classify_outcome,
    collect_paired_outcomes,
    select_outcome_representatives,
    summarize_wdl,
    verify_match_artifacts,
)
from civ5rl.provenance.common import file_sha256, write_json


def test_empty_outcomes(tmp_path: Path) -> None:
    assert summarize_wdl([]) == {"matches": 0, "W": 0, "D": 0, "L": 0, "win_rate": 0.0}
    assert collect_paired_outcomes(tmp_path) == ({}, {}, {})
    assert select_outcome_representatives(tmp_path) == {}


def test_summary_keeps_win_rate_distinct_from_draws() -> None:
    rows: list[MatchRow] = [
        {
            "winner": 1,
            "side": 1,
            "seconds": 4.0,
            "rounds": 2,
            "actions_by_kind": {"move": 2},
        },
        {
            "winner": None,
            "side": 0,
            "seconds": 2.0,
            "rounds": 4,
            "actions_by_kind": {"move": 1, "skip": 3},
            "reason": "turn_limit",
        },
        {"winner": 1, "side": 0, "seconds": 3.0, "rounds": 3},
    ]
    assert summarize_wdl(rows, extra_metrics=True) == {
        "matches": 3,
        "W": 1,
        "D": 1,
        "L": 1,
        "win_rate": 1 / 3,
        "actions": {"move": 3, "skip": 3},
        "seconds": 9.0,
        "mean_rounds": 3.0,
        "draw_reasons": {"turn_limit": 1},
    }
    with pytest.raises(ValueError, match="Invalid side or winner"):
        classify_outcome(2, 1)


def test_pair_key_source_is_independent_of_outcome(tmp_path: Path) -> None:
    # The two records share the case-side key but carry different metric sides.
    case = {"parent_world": 9, "case_id": "01", "learner_side": 0}
    winning = tmp_path / "first"
    drawing = tmp_path / "second"
    write_json(winning / "metrics.json", {"winner": 0, "side": 0})
    write_json(drawing / "metrics.json", {"winner": None, "side": 1})
    for directory in (drawing, winning):
        write_json(directory / "case.json", case)
    (tmp_path / "incomplete").mkdir()

    with pytest.raises(ValueError, match="Duplicate evaluation case"):
        collect_paired_outcomes(tmp_path, check_duplicates=True)
    assert collect_paired_outcomes(tmp_path) == (
        {(9, "01", 0): 0},
        {(9, "01", 0): case},
        {(9, "01", 0): "D"},
    )
    assert collect_paired_outcomes(
        tmp_path, side_source="metrics", check_duplicates=True
    ) == (
        {(9, "01", 0): 1, (9, "01", 1): 0},
        {(9, "01", 0): case, (9, "01", 1): case},
        {(9, "01", 0): "W", (9, "01", 1): "D"},
    )
    assert select_outcome_representatives(tmp_path) == {"W": winning, "D": drawing}
    assert not verify_match_artifacts(winning)
    assert verify_match_artifacts(winning, ("case.json", "metrics.json"))


def test_chunk_failure_stops_progress_and_future_work() -> None:
    executed: list[int] = []
    progress: list[int] = []

    def work(value: int) -> None:
        executed.append(value)
        if value == 3:
            raise RuntimeError("failed")

    with ThreadPoolExecutor(max_workers=1) as executor:
        run_chunked_pool(executor, work, [], 2, progress.append)
        with pytest.raises(RuntimeError, match="failed"):
            run_chunked_pool(executor, work, list(range(6)), 2, progress.append)
    assert executed == [0, 1, 2, 3]
    assert progress == [2]


def test_parent_reuse_failure_prohibits_fallback_evaluation(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(b"parent checkpoint")
    out = tmp_path / "run"
    out.mkdir()

    def reject_reuse(*_args: Any) -> None:
        raise ValueError("incompatible cache")

    def evaluate(*_args: Any) -> dict[str, Any]:
        pytest.fail("Incompatible parent must prevent evaluation")

    config = {"parents": [str(source)], "parent_sha256": [file_sha256(source)]}
    with pytest.raises(ValueError, match="incompatible cache"):
        evaluate_paired_dev512(
            config, out, {}, {}, evaluate, reuse_parent_fn=reject_reuse
        )
