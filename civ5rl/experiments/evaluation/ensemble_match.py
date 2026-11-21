"""Torch-free worker for ordinary matches with ensemble inference."""

from __future__ import annotations

from pathlib import Path

from civ5rl.engine.evaluation.replay import verify_replay
from civ5rl.experiments.evaluation.matches import target_scale_evaluation_task
from civ5rl.provenance.common import write_json


def checked_match(task):
    row = target_scale_evaluation_task(task)
    directory = Path(task[3])
    verify_replay(directory, project_relative=False)
    write_json(directory / "checked.json", {"replay_verified": True})
    return row
