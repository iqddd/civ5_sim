"""Bounded subprocess execution and verified tree cleanup for CLI tools."""

import os
import signal
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class WorkerCleanupError(Exception):
    """Fatal cleanup failure: subsequent benchmark timings cannot be trusted."""


def stop_tree(process, *, grace_seconds=5, timeout=5, isolated_group=None):
    """Stop an owned subprocess and descendants, with bounded waits.

    POSIX process groups are signalled only when the child is their leader.
    Other children use the captured process tree; unrelated groups are untouched.
    """
    own_group = os.name != "nt" and isolated_group is True
    if os.name != "nt" and isolated_group is None and process.poll() is None:
        try:
            own_group = os.getpgid(process.pid) == process.pid
        except ProcessLookupError:
            pass
    if own_group:
        try:
            if grace_seconds and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=grace_seconds)
                except subprocess.TimeoutExpired:
                    pass
            # The group may outlive its leader. A known isolated group is still
            # ours after poll() reports exit; kill it before collecting pipes.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=timeout)
            return
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise WorkerCleanupError(f"Could not stop owned process group: {exc}") from exc

    import psutil

    try:
        root = psutil.Process(process.pid)
        tracked = [*root.children(recursive=True), root]
    except psutil.NoSuchProcess:
        tracked = []
    try:
        if os.name == "nt" and process.poll() is None:
            killed = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True, timeout=timeout, check=False,
            )
            if killed.returncode and process.poll() is None:
                raise OSError(f"taskkill failed: {killed.stderr!r}")
        for child in tracked:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(tracked, timeout=timeout)
        alive = [p for p in alive if p.status() != psutil.STATUS_ZOMBIE]
        process.wait(timeout=timeout)
        if alive:
            raise OSError(f"Surviving child processes: {[p.pid for p in alive]}")
    except (OSError, psutil.Error, subprocess.TimeoutExpired) as exc:
        # Best effort after a failed cleanup; never silently continue timing.
        for child in tracked:
            try:
                child.kill()
            except psutil.Error:
                pass
        raise WorkerCleanupError(f"Could not confirm worker-tree termination: {exc}") from exc


def run_bounded(command, timeout, *, cwd=ROOT):
    """Capture a command and clean its entire tree on timeout or interruption."""
    process = subprocess.Popen(
        command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=os.name != "nt",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException:
        stop_tree(process, grace_seconds=0, isolated_group=os.name != "nt")
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired as exc:
            raise WorkerCleanupError("Worker output pipes survived tree cleanup") from exc
        raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
