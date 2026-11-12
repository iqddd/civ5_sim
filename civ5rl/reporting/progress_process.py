"""Process lifecycle recording and local/SSH viewing of normalized progress."""

import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from datetime import UTC, datetime
from uuid import uuid4

from civ5rl.reporting.progress import estimate, format_progress
from civ5rl.reporting.progress_adapters import adapter_for
from civ5rl.reporting.progress_source import ArtifactSource


def collect_progress(source, *, alpha=0.1, exclude_compilation=False):
    observation = adapter_for(source.config.get("kind", "")).observe(source)
    result = estimate(observation, alpha=alpha, exclude_compilation=exclude_compilation)
    result["kind"] = source.config.get("kind")
    extension = source.read("active-budget.json", {})
    result["budget_control"] = {
        "requested_revision": len(extension.get("requests", [])),
        "applied": source.read("budget-applied.json"),
    }
    latest = source.read("derived/progress/latest.json", {})
    process = latest.get("process", {})
    if process.get("id") == source.session.get("id") and process:
        result["process"] = process
        age = (datetime.now(UTC) - datetime.fromisoformat(latest["observed_at"])).total_seconds()
        stale = process.get("running") and age > max(90, 3 * process["interval_seconds"])
        result["process_heartbeat_age_seconds"] = age
        result["process_state"] = "stale" if stale else "running" if process.get("running") else "stopped"
        if process.get("error") and not process.get("running"):
            result["details"]["artifact_status"] = result["status"]
            result["status"] = "interrupted" if process["error"] == "KeyboardInterrupt" else "technical_error"
        if stale or not process.get("running"):
            if result["status"] != "complete":
                result["estimated_finish_at"] = None
                result["eta_assumes_resume"] = True
            if stale:
                result["warnings"].append("Process heartbeat is stale; a fresh artifact read does not confirm that training is alive.")
    return result


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class ProcessProgress:
    """Out-of-band observer; failures never stop or mutate scientific execution.

    One session per process invocation resets rates and evaluation calibration.
    Estimates are replaceable derived artifacts, never resume state.
    """

    def __init__(self, directory, config, *, interval=30):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("Progress interval must be positive and finite")
        self.source = ArtifactSource(directory, config)
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = None
        self.session = None
        self.published = False
        self.last_error = None

    def __enter__(self):
        try:
            self._start()
        except Exception as exc:
            print(f"Progress observer disabled: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return self

    def _start(self):
        # Baselines are captured synchronously, before the runner can append.
        updates = self.source.rows("updates.jsonl")
        steps = self.source.rows("steps.jsonl")
        state = self.source.read("state.json", {})
        logged_steps = max((r.get("step", 0) for r in steps), default=0)
        self.session = dict(id=uuid4().hex, pid=os.getpid(), interval_seconds=self.interval,
                            started_at=datetime.now(UTC).isoformat(),
                            active_seconds=state.get("active_seconds", 0),
                            # A resumed pending rollout has already been partly
                            # processed on another process. Skip that mixed cycle.
                            ppo_rollout=max(state.get("rollouts", 0),
                                            max((r.get("rollout", r.get("update", 0)) for r in updates), default=0)),
                            bc_step=min(logged_steps, state.get("Adam_steps", logged_steps)),
                            evaluation_keys=list(self.source.read("evaluations.json", {})))
        self.thread = threading.Thread(target=self._loop, name="run-progress", daemon=True)
        self.thread.start()

    def _publish(self, running, error=None):
        if not (self.source.directory / "run.json").exists():
            return  # Do not create output before the runner initializes it.
        directory = self.source.directory / "derived/progress"
        if not self.published:
            _atomic_json(directory / "session.json", self.session)
            self.published = True
        result = collect_progress(self.source)
        result["process"] = {**self.session, "running": running, "error": error}
        result["process_heartbeat_age_seconds"] = 0.0
        result["process_state"] = "running" if running else "stopped"
        if running:
            result["eta_assumes_resume"] = result["status"] in {"paused", "technical_error", "closed_by_user"}
        elif result["status"] != "complete":
            result["estimated_finish_at"] = None
            result["eta_assumes_resume"] = True
        if error:
            result["status"] = "interrupted" if error == "KeyboardInterrupt" else "technical_error"
            result["eta_assumes_resume"] = True
            result["estimated_finish_at"] = None
        _atomic_json(directory / "latest.json", result)

    def _safe_publish(self, running, error=None):
        try:
            self._publish(running, error)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            if message != self.last_error:
                print(f"Progress observer: {message}", file=sys.stderr, flush=True)
                self.last_error = message

    def _loop(self):
        while not self.stop_event.is_set():
            self._safe_publish(True)
            self.stop_event.wait(self.interval)

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
            self._safe_publish(False, exc_type.__name__ if exc_type else None)


def remote_progress(deployment_dir, *, alpha=0.1, exclude_compilation=False):
    from civ5rl.deploy.remote import _connection, _read, _remote_path, _ssh_json

    deployment = _read(Path(deployment_dir) / "deployment.json")
    if deployment.get("instance_deleted"):
        raise ValueError("Instance was deleted; inspect the pulled local run with --run")
    root = _remote_path(deployment["remote_project"])
    # Compute on the host: only the small normalized result crosses SSH, without
    # pulling ledgers, checkpoints, or starting models/simulations.
    code = """
import json, sys
from pathlib import Path
p = json.load(sys.stdin)
sys.path.insert(0, p['root'])
from civ5rl.reporting.progress_source import ArtifactSource
from civ5rl.reporting.progress_process import collect_progress
source = ArtifactSource(Path(p['root']) / p['output'])
print(json.dumps(collect_progress(source, alpha=p['alpha'], exclude_compilation=p['exclude_compilation']), allow_nan=False))
"""
    return _ssh_json(_connection(deployment["instance"]), code,
                     dict(root=root, output=deployment["output"], alpha=alpha, exclude_compilation=exclude_compilation))


def watch_progress(*, run=None, deployment=None, watch=False, interval=900, alpha=0.1,
                   exclude_compilation=False, json_output=False):
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("Interval must be positive and finite")
    if not math.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("Require 0 < alpha <= 1")
    source = ArtifactSource(run) if run is not None else None
    result = None
    try:
        while True:
            try:
                result = collect_progress(source, alpha=alpha, exclude_compilation=exclude_compilation) if source else remote_progress(
                    deployment, alpha=alpha, exclude_compilation=exclude_compilation)
            except Exception as error:
                if not watch:
                    raise
                print(f"Progress unavailable: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
                time.sleep(interval)
                continue
            print(json.dumps(result, ensure_ascii=False, allow_nan=False) if json_output else format_progress(result), flush=True)
            if not json_output:
                for warning in result["warnings"]:
                    print(f"  {warning}", flush=True)
            if not watch or result["status"] in {"complete", "paused", "technical_error", "closed_by_user", "interrupted"}:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        pass  # Interrupt viewing only; never signal the training process.
    return result
