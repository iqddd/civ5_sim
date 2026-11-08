"""Candidate construction and ranking for collection CLI tools."""

import math
import statistics
from dataclasses import dataclass, field

TIE_FRACTION = 0.03
DEFAULT_ENV_CAPACITY = 80


@dataclass(frozen=True, order=True)
class Candidate:
    envs: int
    workers: int
    reply: int = 1
    scheduler: str = "ready"
    inference_min_batch: int = 0
    inference_wait_seconds: float = 0.0

    def __post_init__(self):
        if (
            min(self.envs, self.workers, self.reply) < 1
            or self.workers > self.envs
            or self.reply > math.ceil(self.envs / self.workers)
        ):
            raise ValueError(
                "Require workers <= envs and a reply size that fits worker slots"
            )

    def __str__(self):
        label = f"{self.envs}/{self.workers}/{self.reply}"
        if self.inference_min_batch or self.inference_wait_seconds:
            label += f"@{self.inference_min_batch}/{self.inference_wait_seconds:g}"
        return label

    def runtime(self):
        result = {
            "num_envs": self.envs,
            "cpu_workers": self.workers,
            "worker_reply_size": self.reply,
        }
        result.update(
            collection_scheduler=self.scheduler,
            inference_min_batch=self.inference_min_batch,
            inference_wait_seconds=self.inference_wait_seconds,
        )
        return result


@dataclass
class Measurements:
    candidate: Candidate
    trials: list = field(default_factory=list)
    confirmations: list = field(default_factory=list)
    sessions: list = field(default_factory=list)
    status: str = "pending"

    @property
    def median(self):
        return (
            statistics.median(t["seconds"] for t in self.trials)
            if self.trials
            else None
        )


def streaming_candidates(budget, count):
    """Three probes; retain a large env pool independently of the CPU quota."""
    workers = max(1, math.ceil(budget))
    envs = min(count, max(DEFAULT_ENV_CAPACITY, 2 * workers))
    return [
        Candidate(envs, value, scheduler="streaming")
        for value in sorted({min(envs, workers), min(envs, 2 * workers), envs})
    ]


def initial_candidates(budget, max_envs):
    """CPU replay search used by benchmark_rl_machine, not the streaming tuner."""
    base = max(1, math.floor(budget))
    counts = sorted(
        {max(1, math.floor(value + 0.5)) for value in (base / 2, base, 1.5 * base)}
    )
    counts = [value for value in counts if value <= max_envs]
    # Small fixed datasets still need an executable starting point.
    return [Candidate(value, value) for value in counts] or [
        Candidate(max_envs, max_envs)
    ]


def preferred(rows):
    valid = [r for r in rows if r.status == "ok" and r.median is not None]
    if not valid:
        return None
    fastest = min(r.median for r in valid)
    close = [r for r in valid if r.median <= fastest * (1 + TIE_FRACTION)]
    return min(
        close,
        key=lambda r: (
            r.candidate.workers,
            r.candidate.envs,
            r.candidate.inference_wait_seconds,
            r.candidate.reply,
            r.candidate.inference_min_batch,
        ),
    )
