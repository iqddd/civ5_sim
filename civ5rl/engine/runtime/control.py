"""Single-writer training runs and graceful stops at completed PPO updates."""

from __future__ import annotations

import os
import signal
import threading
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from civ5rl.engine.runtime import is_int_at_least
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
    if not is_int_at_least(value):
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
        or not is_int_at_least(request.get("at_least"))
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
            raise ValueError("Saved boundary checkpoint is missing or outside the run")
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


_TARGET_BC_PROTOCOLS = frozenset(
    {
        "rl01.ppo_after_target_bc",
        "ppo_after_target_bc",
    }
)

_CONTROLLED_PPO_PROTOCOLS = frozenset(
    {
        "rl01.ppo_controlled_updates",
        "ppo_controlled_updates",
        "rl01.ppo_scale_up",
        "ppo_scale_up",
        "rl01.ppo_target_scale",
        "ppo_target_scale",
    }
)

_ORDINARY_PPO_PROTOCOLS = frozenset(
    {
        "rl01.ppo_after_imitation",
        "ppo_after_imitation",
    }
)

_BC_PROTOCOLS = frozenset(
    {
        "bc",
        "imitation_base",
        "rl01.cp_imitation",
        "cp_imitation",
        "rl01.cp_imitation_own_states",
        "cp_imitation_own_states",
        "imitation_own_states",
        "rl01.cp_imitation_target_scale",
        "cp_imitation_target_scale",
        "imitation_target_scale",
        "rl01.cp_imitation_target_more_data",
        "cp_imitation_target_more_data",
        "imitation_target_more_data",
        "rl01.cp_imitation_target_c2",
        "cp_imitation_target_c2",
        "imitation_target_c2",
    }
)


def _normalize_protocol(
    protocol: str | None, data: Mapping[str, Any] | None = None
) -> str | None:
    raw = protocol
    if raw is None and data is not None:
        kind = data.get("kind")
        if isinstance(kind, str):
            raw = kind
    if raw is None:
        return None
    if raw in _TARGET_BC_PROTOCOLS:
        return "ppo_after_target_bc"
    if raw in _CONTROLLED_PPO_PROTOCOLS:
        return "ppo_controlled"
    if raw in _ORDINARY_PPO_PROTOCOLS:
        return "ppo_after_imitation"
    if raw in _BC_PROTOCOLS or raw.startswith(
        ("rl01.cp_imitation", "cp_imitation", "imitation")
    ):
        return "bc"
    raise ValueError(f"Unsupported counter protocol: {protocol!r}")


def _first_present(
    data: Mapping[str, Any], keys: tuple[str, ...], default: Any = None
) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


_ATTEMPTED_STEP_KEYS = (
    "adam_steps_attempted",
    "attempted_steps",
    "total_attempted_steps",
    "attempted_optimizer_steps",
)

_PPO_ACCEPTED_STEP_KEYS = (
    "adam_steps_accepted",
    "optimizer_steps",
    "accepted_optimizer_steps",
    "total_Adam_steps",
    "Adam_steps",
    "steps_performed",
    "Adam_steps_total",
    "optimizer_steps_total",
)

_TARGET_BC_ROLLOUT_KEYS = (
    "rollouts_collected",
    "rollouts",
    "rollout_number",
    "rollout",
    "updates",
    "update",
)

_CONTROLLED_PPO_ROLLOUT_KEYS = (
    "rollouts_collected",
    "rollouts",
    "updates",
    "rollout_index",
    "rollout_number",
    "rollout",
    "update",
)


def _resolve_accepted_rollouts(data: Mapping[str, Any]) -> Any:
    rwa = _first_present(
        data, ("rollouts_with_accepted_steps", "accepted_updates"), default=None
    )
    if rwa is not None:
        return rwa
    if (
        "optimizer_steps" in data
        and data["optimizer_steps"] is not None
        and int(data["optimizer_steps"]) == 0
    ):
        return 0
    return _first_present(data, ("updates", "update"), default=0)


def _resolve_attempted_steps(data: Mapping[str, Any], accepted_steps: Any) -> Any:
    att = _first_present(data, _ATTEMPTED_STEP_KEYS, default=None)
    if att is not None:
        return att
    if "rejected_steps" in data and data["rejected_steps"] is not None:
        return int(accepted_steps or 0) + int(data["rejected_steps"])
    return accepted_steps


@dataclass(frozen=True)
class TrainingCounters:
    rollouts_collected: int = 0
    rollouts_with_accepted_steps: int = 0
    adam_steps_attempted: int = 0
    adam_steps_accepted: int = 0

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | None,
        *,
        protocol: str | None = None,
    ) -> TrainingCounters:
        """Translate canonical target names or protocol-specific legacy JSON keys into TrainingCounters."""
        if not data:
            return cls()

        mode = _normalize_protocol(protocol, data)

        if mode == "bc":
            asa = _first_present(
                data,
                (
                    "adam_steps_accepted",
                    "Adam_steps",
                    "steps_performed",
                    "optimizer_steps",
                    "total_Adam_steps",
                    "Adam_steps_total",
                    "accepted_optimizer_steps",
                    "optimizer_steps_total",
                    "adam_steps",
                ),
                default=0,
            )
            att = _first_present(data, _ATTEMPTED_STEP_KEYS, default=asa)
            return cls(
                rollouts_collected=0,
                rollouts_with_accepted_steps=0,
                adam_steps_attempted=int(att) if att is not None else 0,
                adam_steps_accepted=int(asa) if asa is not None else 0,
            )

        if mode in ("ppo_after_target_bc", "ppo_controlled"):
            rollout_keys = (
                _TARGET_BC_ROLLOUT_KEYS
                if mode == "ppo_after_target_bc"
                else _CONTROLLED_PPO_ROLLOUT_KEYS
            )
            rc = _first_present(data, rollout_keys, default=0)
            asa = _first_present(data, _PPO_ACCEPTED_STEP_KEYS, default=0)
            att = _resolve_attempted_steps(data, asa)
            rwa = _resolve_accepted_rollouts(data)
            return cls(
                rollouts_collected=int(rc) if rc is not None else 0,
                rollouts_with_accepted_steps=int(rwa) if rwa is not None else 0,
                adam_steps_attempted=int(att) if att is not None else 0,
                adam_steps_accepted=int(asa) if asa is not None else 0,
            )

        if mode == "ppo_after_imitation":
            rc = _first_present(data, _TARGET_BC_ROLLOUT_KEYS, default=0)
            rwa = _first_present(
                data,
                (
                    "rollouts_with_accepted_steps",
                    "accepted_updates",
                    "updates",
                    "update",
                ),
                default=0,
            )
            asa = _first_present(data, _PPO_ACCEPTED_STEP_KEYS, default=0)
            att = _first_present(data, _ATTEMPTED_STEP_KEYS, default=asa)
            return cls(
                rollouts_collected=int(rc) if rc is not None else 0,
                rollouts_with_accepted_steps=int(rwa) if rwa is not None else 0,
                adam_steps_attempted=int(att) if att is not None else 0,
                adam_steps_accepted=int(asa) if asa is not None else 0,
            )

        # Fallback when protocol is None and not specified in data["kind"]
        rc = _first_present(
            data,
            (
                "rollouts_collected",
                "rollouts",
                "rollout_index",
                "rollout_number",
                "rollout",
                "updates",
                "update",
            ),
            default=0,
        )
        rwa = _resolve_accepted_rollouts(data)
        asa = _first_present(
            data,
            (
                "adam_steps_accepted",
                "optimizer_steps",
                "Adam_steps",
                "total_Adam_steps",
                "steps_performed",
                "accepted_optimizer_steps",
                "Adam_steps_total",
                "optimizer_steps_total",
                "adam_steps",
            ),
            default=0,
        )
        att = _resolve_attempted_steps(data, asa)

        return cls(
            rollouts_collected=int(rc) if rc is not None else 0,
            rollouts_with_accepted_steps=int(rwa) if rwa is not None else 0,
            adam_steps_attempted=int(att) if att is not None else 0,
            adam_steps_accepted=int(asa) if asa is not None else 0,
        )

    def to_state_dict(
        self,
        *,
        protocol: str | None = None,
        include_legacy_bc: bool = False,
    ) -> dict[str, int]:
        """Dual-write canonical counter fields and protocol-appropriate legacy keys."""
        mode = _normalize_protocol(protocol)
        base: dict[str, int] = {
            "rollouts_collected": self.rollouts_collected,
            "rollouts_with_accepted_steps": self.rollouts_with_accepted_steps,
            "adam_steps_attempted": self.adam_steps_attempted,
            "adam_steps_accepted": self.adam_steps_accepted,
        }
        rejected = max(0, self.adam_steps_attempted - self.adam_steps_accepted)

        if mode == "bc" or (mode is None and include_legacy_bc):
            return {
                **base,
                "Adam_steps": self.adam_steps_accepted,
                "steps_performed": self.adam_steps_accepted,
                "optimizer_steps": self.adam_steps_accepted,
            }

        if mode == "ppo_after_target_bc":
            result = {
                **base,
                "rollouts": self.rollouts_collected,
                "accepted_updates": self.rollouts_with_accepted_steps,
                "attempted_steps": self.adam_steps_attempted,
                "optimizer_steps": self.adam_steps_accepted,
                "rejected_steps": rejected,
            }
            if include_legacy_bc:
                result["Adam_steps"] = self.adam_steps_accepted
                result["steps_performed"] = self.adam_steps_accepted
            return result

        if mode == "ppo_controlled":
            result = {
                **base,
                "rollouts": self.rollouts_collected,
                "updates": self.rollouts_collected,
                "accepted_updates": self.rollouts_with_accepted_steps,
                "attempted_steps": self.adam_steps_attempted,
                "optimizer_steps": self.adam_steps_accepted,
                "rejected_steps": rejected,
            }
            if include_legacy_bc:
                result["Adam_steps"] = self.adam_steps_accepted
                result["steps_performed"] = self.adam_steps_accepted
            return result

        if mode == "ppo_after_imitation":
            result = {
                **base,
                "updates": self.rollouts_with_accepted_steps,
                "optimizer_steps": self.adam_steps_accepted,
            }
            if include_legacy_bc:
                result["Adam_steps"] = self.adam_steps_accepted
                result["steps_performed"] = self.adam_steps_accepted
            return result

        return {
            **base,
            "rollouts": self.rollouts_collected,
            "updates": self.rollouts_with_accepted_steps,
            "accepted_updates": self.rollouts_with_accepted_steps,
            "attempted_steps": self.adam_steps_attempted,
            "optimizer_steps": self.adam_steps_accepted,
            "Adam_steps": self.adam_steps_accepted,
            "rejected_steps": rejected,
            "steps_performed": self.adam_steps_accepted,
        }

    def replace(self, **kwargs: Any) -> TrainingCounters:
        """Return a copy with specified fields updated."""
        return replace(self, **kwargs)
