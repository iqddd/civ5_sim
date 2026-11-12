"""Single-writer training runs and graceful stops at completed PPO updates."""

import os
import signal
import threading
from datetime import UTC, datetime
from pathlib import Path

from civ5rl.provenance.common import write_json


class RunLock:
    def __init__(self, directory, filename="run.lock"):
        self.path = Path(directory) / filename

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        self.stream.seek(0, 2)
        if self.stream.tell() == 0:
            self.stream.write(b"0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close()
            raise RuntimeError(
                f"{'Training run already active' if self.path.name == 'run.lock' else 'Operational control busy'}: {self.path.parent}"
            ) from exc
        return self

    def __exit__(self, *exc):
        self.stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self.stream, fcntl.LOCK_UN)
        self.stream.close()


def is_active(directory):
    try:
        with RunLock(directory):
            return False
    except RuntimeError:
        return True


class TrainingControl:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.stop_file = self.directory / "stop.request"
        self.requested = False
        self.handlers = {}
        self.state = {"status": "running", "phase": "startup", "pid": os.getpid()}

    def request_stop(self, *_):
        # Signal handlers only set a flag; no I/O or partially updated checkpoints.
        self.requested = True

    def stopping(self):
        return self.requested or self.stop_file.exists()

    def mark(self, phase, **state):
        self.state.update(phase=phase, **state)
        write_json(
            self.directory / "run-status.json",
            {**self.state, "updated_utc": datetime.now(UTC).isoformat()},
        )

    def __enter__(self):
        self.lock = RunLock(self.directory)
        self.lock.__enter__()
        if threading.current_thread() is threading.main_thread():
            for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
                sig = getattr(signal, name, None)
                if sig is not None:
                    self.handlers[sig] = signal.getsignal(sig)
                    signal.signal(sig, self.request_stop)
        self.mark("startup")
        return self

    def __exit__(self, kind, value, traceback):
        if kind is not None:
            self.mark("failed", status="failed", error=str(value))
        for sig, handler in self.handlers.items():
            signal.signal(sig, handler)
        self.lock.__exit__(kind, value, traceback)


def request_pause(directory, *, after_update=None, after_epoch=None):
    """Request a supported runner to stop at a saved update or epoch boundary."""
    import json

    directory = Path(directory)
    record = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    kind = record.get("kind")
    requested = [("update", after_update), ("epoch", after_epoch)]
    requested = [(name, value) for name, value in requested if value is not None]
    if len(requested) != 1:
        raise ValueError("Specify exactly one boundary")
    name, value = requested[0]
    if type(value) is not int or value < 1:
        raise ValueError("Boundary must be a positive integer")
    supported = {
        "rl01.ppo_after_target_bc": "update",
        "rl01.cp_imitation_target_scale": "epoch",
        "rl01.cp_imitation_target_more_data": "epoch",
    }
    if supported.get(kind) != name:
        raise ValueError(f"Run kind {kind!r} does not support after-{name}")
    request = {"schema": 1, "boundary": name, "at_least": value}
    write_json(directory / "pause.request", request)
    return request


def pause_due(directory, boundary, completed, *, at_boundary=True):
    """Interpret JSON threshold or the legacy immediate request at a safe boundary."""
    import json

    path = Path(directory) / "pause.request"
    if not at_boundary or not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    try:
        request = json.loads(text)
    except json.JSONDecodeError:
        if text.lstrip().startswith("{"):
            raise ValueError("Malformed pause request") from None
        return True
    if (
        not isinstance(request, dict)
        or request.get("schema") != 1
        or request.get("boundary") != boundary
        or type(request.get("at_least")) is not int
        or request["at_least"] < 1
    ):
        raise ValueError("Invalid or unsupported pause request")
    return completed >= request["at_least"]


def assert_not_closed(directory):
    if (Path(directory) / "closure.json").exists():
        raise RuntimeError("Run is closed; an explicit reopening decision is required")


def target_bc_evaluation_due(config, state, *, current_policy_hash=None):
    """Identify unfinished baseline, reached milestones, or terminal evaluation.

    Without a model hash, a terminal checkpoint is considered evaluated only
    when its path appears in the saved development records. This conservative
    form is for read-only close checks, not a decision to run an evaluation.
    """
    try:
        milestones = config["experiment"]["milestones"]
        decisions = state["decisions"]
        points = state["development_points"]
        evaluated = state["evaluated"]
        latest = state["latest"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Cannot infer pending target-BC evaluation from saved state/config"
        ) from exc
    if (
        not isinstance(milestones, list)
        or not isinstance(points, dict)
        or not isinstance(evaluated, dict)
    ):
        raise TypeError(
            "Cannot infer pending target-BC evaluation from saved state/config"
        )
    if any(not isinstance(point, (int, float)) for point in milestones):
        raise TypeError("Invalid target-BC development milestones")
    if any(value not in evaluated for value in points.values()):
        raise ValueError("Development milestone refers to a missing evaluation")
    if not evaluated or any(
        str(point) not in points and point <= decisions for point in milestones
    ):
        return True
    if not state.get("stop_reason"):
        return False
    if current_policy_hash is not None:
        return current_policy_hash not in evaluated
    return not any(
        isinstance(row, dict) and row.get("checkpoint") == latest
        for row in evaluated.values()
    )


def close_run(directory, reason):
    """Close a saved boundary under the run lock without modifying its state."""
    import json

    from civ5rl.provenance.common import file_sha256

    directory = Path(directory)
    if not reason.strip():
        raise ValueError("A closure reason is required")
    lock_path = directory / "run.lock"
    if not lock_path.is_file() or not lock_path.stat().st_size:
        raise ValueError("Existing run lock file is required")
    with RunLock(directory):
        assert_not_closed(directory)
        state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
        status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
        run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        if state.get("phase") not in (None, "collect"):
            raise ValueError("Run has a pending PPO phase or rollout")
        if state.get("pending_evaluation"):
            raise ValueError("Run has a pending evaluation")
        if run.get("kind") == "rl01.ppo_after_target_bc" and target_bc_evaluation_due(
            run.get("config", {}), state
        ):
            raise ValueError(
                "Run has a pending or unverifiable target-BC development evaluation"
            )
        if "epoch_completed" in state and not state["epoch_completed"]:
            raise ValueError("Cannot close an incomplete epoch")
        if not (
            "epoch_completed" in state or "rollouts" in state or "updates" in state
        ):
            raise ValueError("Unrecognized saved boundary")
        if (
            "rollouts" in state
            and "updates" in state
            and state["updates"] > state["rollouts"]
        ):
            raise ValueError("Saved update count exceeds rollout count")
        latest = state.get("latest")
        if not latest:
            raise ValueError("No saved boundary checkpoint")
        if Path(latest).name in ("pending.pt", "critic-ready.pt"):
            raise ValueError("Run checkpoint is an incomplete PPO boundary")
        path = (directory / latest).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file():
            raise ValueError(
                "Saved boundary checkpoint is missing or outside the run"
            )
        if state.get("rollouts") or state.get("updates"):
            rows = [
                json.loads(line)
                for line in (directory / "updates.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            expected = state.get("rollouts", state.get("updates"))
            if Path(latest).name != f"update-{expected:04}.pt":
                raise ValueError("Checkpoint name does not match saved update boundary")
            if not rows or rows[-1].get("rollout", rows[-1].get("update")) != expected:
                raise ValueError("Saved update ledger and boundary disagree")
            if state.get("decisions") != rows[-1].get("total_decisions"):
                raise ValueError("Saved decision count and update ledger disagree")
            for state_key, metric in (
                ("optimizer_steps", "optimizer_steps"),
                ("attempted_steps", "attempted_optimizer_steps"),
                ("rejected_steps", "rejected_optimizer_steps"),
            ):
                if state_key in state:
                    values = [row.get("ppo", {}).get(metric) for row in rows]
                    if (
                        any(value is None for value in values)
                        or sum(values) != state[state_key]
                    ):
                        raise ValueError(
                            f"Saved {state_key} and update ledger disagree"
                        )
        closure = {
            "status": "closed_by_user",
            "reason": reason,
            "previous_status": status.get("status"),
            "boundary": {
                k: state[k]
                for k in (
                    "phase",
                    "rollouts",
                    "updates",
                    "epoch",
                    "epoch_completed",
                    "decisions",
                    "optimizer_steps",
                    "Adam_steps",
                    "latest",
                )
                if k in state
            },
            "checkpoint_sha256": file_sha256(path),
            "closed_utc": datetime.now(UTC).isoformat(),
        }
        write_json(directory / "closure.json", closure)
        write_json(
            directory / "status.json",
            {"status": "closed_by_user", "closure": "closure.json"},
        )
        return closure
