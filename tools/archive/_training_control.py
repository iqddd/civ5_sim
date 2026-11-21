"""Lifecycle helper for the archived diagnostic training entry points."""

from __future__ import annotations

import os
import signal
import threading
from datetime import UTC, datetime
from pathlib import Path

from civ5rl.engine.runtime.control import RunLock
from civ5rl.provenance.common import write_json


class TrainingControl:
    """Signal-based lifecycle used by the archived training entry points."""

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
