"""Exercise shared workflow boundaries with synthetic CPU-only inputs."""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from civ5rl.engine.runtime.control import is_active
from civ5rl.experiments._shared.critic_ops import (
    behavior_digest,
    critic_statistics,
    refresh_episode_old_values,
    value_only_grad_mode,
    verify_critic_invariance,
)
from civ5rl.experiments._shared.lifecycle import (
    ActiveSessionClock,
    PersistedActiveSessionClock,
    experiment_run_guard,
)
from civ5rl.experiments._shared.markdown import MarkdownReport, format_wdl, render_table
from civ5rl.experiments.ppo_after_imitation import policy_hash
from civ5rl.provenance.common import read_json, write_json

if TYPE_CHECKING:
    from collections.abc import Iterator

    from civ5rl.core.contracts import Episode


class _Policy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = torch.nn.Parameter(torch.zeros(1))
        self.value = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.value.weight.fill_(2)


def _episode(length: int) -> Episode:
    return cast(
        "Episode",
        {
            "chosen": [0] * length,
            "old_logp": [-0.5] * length,
            "old_log_probs": [np.array([-0.5, -1.0], dtype=np.float32)] * length,
            "old_value": [0.0] * length,
        },
    )


class SharedWorkflowTests(unittest.TestCase):
    def test_run_guard_records_only_protected_failures_while_holding_lock(self) -> None:
        with TemporaryDirectory() as directory:
            out = Path(directory)
            events = []

            @contextmanager
            def session() -> Iterator[None]:
                self.assertTrue(is_active(out))
                try:
                    yield
                finally:
                    events.append("session exited")

            def reject_admission() -> None:
                self.assertTrue(is_active(out))
                raise ValueError("closed run")

            def report_error() -> None:
                self.assertTrue(is_active(out))
                self.assertEqual(
                    read_json(out / "status.json"),
                    {
                        "status": "technical_error",
                        "phase": "training",
                    },
                )
                events.append("reported")

            with (
                self.assertRaisesRegex(ValueError, "closed run"),
                experiment_run_guard(out, phase="training", preflight=reject_admission),
            ):
                self.fail("admission blocked")
            self.assertFalse((out / "status.json").exists())
            with (
                self.assertRaisesRegex(RuntimeError, "prepare failed"),
                experiment_run_guard(
                    out, phase="training", session=session, on_error_report=report_error
                ),
            ):
                raise RuntimeError("prepare failed")
            self.assertEqual(events, ["reported", "session exited"])
            self.assertIn(
                "RuntimeError: prepare failed", (out / "traceback.txt").read_text()
            )
            self.assertFalse(is_active(out))

    def test_offset_clock_counts_exclusions_after_context_exit(self) -> None:
        now = 10.0
        clock = ActiveSessionClock(7.0, timer=lambda: now)
        now = 12.0
        self.assertEqual(clock.seconds(), 9.0)
        with self.assertRaisesRegex(ValueError, "stop"), clock.pause():
            now = 19.0
            self.assertEqual(clock.seconds(), 16.0)
            raise ValueError("stop")
        now = 20.0
        self.assertEqual(clock.seconds(), 10.0)

    def test_persisted_clock_resumes_and_uses_operational_budget(self) -> None:
        with TemporaryDirectory() as directory:
            out = Path(directory)
            write_json(
                out / "run.json",
                {
                    "kind": "rl01.ppo_after_target_bc",
                    "run_id": "test",
                    "config": {"experiment": {"active_seconds": 20}},
                },
            )
            write_json(
                out / "active-budget.json",
                {
                    "schema": 1,
                    "run_id": "test",
                    "base_active_seconds": 20,
                    "additional_active_seconds": 10,
                    "requests": [{}],
                },
            )
            now = 10.0
            state = {"active_seconds": 3.0}
            clock = PersistedActiveSessionClock(state, out, timer=lambda: now)
            now = 13.0
            clock.pause()
            now = 30.0
            self.assertEqual(clock.deadline(), 54.0)
            clock.resume()
            now = 33.0
            clock.progress()
            self.assertEqual(
                read_json(out / "state.json"),
                {
                    "active_seconds": 9.0,
                    "active_seconds_limit": 30.0,
                    "budget_revision_applied": 1,
                },
            )

    def test_value_only_update_preserves_policy_and_recorded_behavior(self) -> None:
        policy = _Policy()
        policy.value.weight.requires_grad_(False)
        episodes = [_episode(1)]
        policy_before = policy_hash(policy.state_dict())
        behavior_before = behavior_digest(episodes)
        with (
            self.assertRaisesRegex(ValueError, "restore"),
            value_only_grad_mode(policy),
        ):
            self.assertFalse(policy.actor.requires_grad)
            self.assertTrue(policy.value.weight.requires_grad)
            with torch.no_grad():
                policy.value.weight.add_(1)
            raise ValueError("restore")
        self.assertTrue(policy.actor.requires_grad)
        self.assertFalse(policy.value.weight.requires_grad)
        episodes[0]["old_value"][0] = 11.0
        verify_critic_invariance(policy, episodes, policy_before, behavior_before)
        episodes[0]["old_logp"][0] += 1
        with self.assertRaisesRegex(
            RuntimeError, "preparation changed recorded behavior"
        ):
            verify_critic_invariance(policy, episodes, policy_before, behavior_before)

    def test_value_refresh_keeps_each_assembly_and_progress_mode(self) -> None:
        policy = _Policy()
        features = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        calls = []
        episodes = [_episode(3), _episode(1)]
        self.assertEqual(
            refresh_episode_old_values(
                policy,
                episodes,
                features,
                "cpu",
                batch_size=2,
                progress=lambda: calls.append("cached"),
                expected_positions=4,
            ),
            [2.0, 4.0, 6.0, 8.0],
        )
        self.assertEqual(calls, ["cached"] * 3)
        episodes = [_episode(3), _episode(1)]
        blocks = iter([(1, 0, features[3:]), (0, 0, features[:3])])
        self.assertEqual(
            refresh_episode_old_values(
                policy,
                episodes,
                blocks,
                "cpu",
                expected_positions=4,
            ),
            [8.0, 2.0, 4.0, 6.0],
        )
        self.assertEqual([ep["old_value"] for ep in episodes], [[2.0, 4.0, 6.0], [8.0]])
        calls.clear()
        episodes = [_episode(3), _episode(1)]
        values = refresh_episode_old_values(
            policy,
            episodes,
            features,
            "cpu",
            batch_size=2,
            bulk_transfer=True,
            progress=lambda: calls.append("bulk"),
        )
        np.testing.assert_array_equal(values, np.array([2, 4, 6, 8], dtype=np.float32))
        self.assertEqual(values.dtype, np.float32)
        self.assertEqual([ep["old_value"] for ep in episodes], [[2.0, 4.0, 6.0], [8.0]])
        self.assertEqual(calls, ["bulk"])
        self.assertIsNone(critic_statistics(values, [1, 1, 1, 1])["explained_variance"])

    def test_markdown_preserves_table_dividers_and_trailing_newline(self) -> None:
        headers = ("Соперник", "Итог")
        rows = [("CP", format_wdl({"W": 1, "D": 2, "L": 3}))]
        expected = "| Соперник | Итог |\n|---|---:|\n| CP | 1/2/3 |"
        self.assertEqual(
            "\n".join(render_table(headers, ("left", "right"), rows)), expected
        )
        report = MarkdownReport(["# Результат", ""])
        report.table(headers, ("---", "---:"), rows)
        self.assertEqual(report.render(), f"# Результат\n\n{expected}\n")
        self.assertEqual(
            report.render(trailing_newline=False), f"# Результат\n\n{expected}"
        )
