"""Single-writer training runs and graceful stops at completed PPO updates."""

import os
import signal
import threading
from datetime import UTC, datetime
from pathlib import Path

from civ5rl.provenance.common import write_json


class RunLock:
    def __init__(self, directory):
        self.path = Path(directory) / "run.lock"

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
                f"Training run already active: {self.path.parent}"
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
