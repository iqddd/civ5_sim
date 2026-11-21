"""Shared evaluator with verified recurrent prefixes before the changed turn.

Restore the state after the previous own observation, not the boundary
observation. The first changed decision therefore processes its observation
exactly once, while the worker reconstructs public encoder and CP history.
"""

from __future__ import annotations

from pathlib import Path

from civ5rl.engine.evaluation.executor import EvaluationExecutor
from civ5rl.experiments.comparison.prefix import SourcePrefixes
from civ5rl.provenance.common import read_json


class TurnExecutor(EvaluationExecutor):
    def __init__(self, checkpoint, runtime):
        super().__init__(checkpoint, runtime)
        self.prefixes = SourcePrefixes(self.inference.policy, checkpoint, self.runtime)
        self.tasks = {}
        self.previous = {}

    def configure_task(self, connection, task):
        self.tasks[connection] = task[-1]

    def configure_agent(self, connection, agent, info):
        branch = self.tasks[connection]
        source = branch["source"]
        key = (source, branch["command"])
        if key not in self.previous:
            history = read_json(Path(source) / "history.json")
            before = [row["command"] for row in history if row["command"] < branch["command"]]
            self.previous[key] = before[-1] if before else None
        command = self.previous[key]
        if command is not None:
            hidden, count = self.prefixes.restore(source, command)
            agent.members[0].hidden = hidden
            info["prefix_skip"] = count

    def release_task(self, connection):
        self.tasks.pop(connection, None)
