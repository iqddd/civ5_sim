"""Run bounded PPO from BC or an inherited PPO checkpoint."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import torch

from civ5rl.engine.evaluation.executor import evaluation_session, runtime_for
from civ5rl.engine.runtime.control import RunLock, assert_not_closed
from civ5rl.experiments._shared.lifecycle import experiment_run_guard
from civ5rl.experiments.ppo.after_target_bc.evaluation import finish
from civ5rl.experiments.ppo.after_target_bc.initialization import initialize
from civ5rl.experiments.ppo.after_target_bc.training import train
from civ5rl.protocols.ppo_after_target_bc import validate_profile
from civ5rl.protocols.turn_reset import validate_case
from civ5rl.provenance.common import read_json, write_json
from civ5rl.reporting.ppo_after_target_bc import report


def run(
    phase: str,
    config: dict[str, Any],
    out: Path | str,
    device: str | torch.device,
    runtime_path: Path | str | None = None,
    max_updates: int | None = None,
) -> None:
    out = Path(out)
    validate_profile(config)
    if max_updates is not None and (type(max_updates) is not int or max_updates < 1):
        raise ValueError("max_updates must be a positive operational limit")
    if phase == "report":
        if not (out / "run.json").is_file() or not (out / "state.json").is_file():
            raise ValueError("A prepared run with saved state is required for reporting")
        with RunLock(out):
            if read_json(out / "run.json")["config"] != config:
                raise ValueError("Run configuration changed")
            report(config, out)
        return
    out.mkdir(parents=True, exist_ok=True)
    if phase == "pause":
        (out / "pause.request").write_text("pause requested\n", encoding="utf-8")
        return
    trains = phase in ("train", "run", "resume")
    with experiment_run_guard(
        out,
        phase=phase,
        session=evaluation_session,
        preflight=partial(assert_not_closed, out) if trains else None,
    ):
        if phase == "resume":
            (out / "pause.request").unlink(missing_ok=True)
        if not (out / "run.json").exists():
            initialize(config, out)
        if read_json(out / "run.json")["config"] != config:
            raise ValueError("Run configuration changed")
        if config.get("evaluation_runtime"):
            runtime_for(out, config)
        plan = read_json(out / "manifest.json")
        for purpose in ("development", "development_512", "confirmation"):
            for case in plan.get(purpose, []):
                validate_case(case, config)
        if trains:
            write_json(out / "status.json", {"status": "training"})
            train(
                config,
                plan,
                out,
                device,
                runtime_path=runtime_path,
                max_updates=max_updates,
            )
            if read_json(out / "status.json").get("status") == "paused":
                report(config, out)
                return
        if phase in ("evaluate", "run", "resume"):
            if not read_json(out / "state.json")["stop_reason"]:
                raise ValueError("Training has not stopped")
            finish(config, plan, out)
            write_json(out / "status.json", {"status": "complete"})
        report(config, out)
