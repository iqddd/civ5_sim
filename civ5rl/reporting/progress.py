"""Run-independent work, rate and budget estimates. No training dependencies."""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import math
from typing import Protocol


@dataclass
class Sample:
    work: float
    seconds: float
    compilation_seconds: float = 0.0


@dataclass
class Stage:
    name: str
    count: int
    seconds_each: float | None
    conditional: bool = False
    budget: str | None = None


@dataclass
class Observation:
    status: str
    phase: str
    unit: str
    completed: float
    target: float | None
    samples: list[Sample] = field(default_factory=list)
    stages: list[Stage] = field(default_factory=list)
    # Each budget has independent semantics; e.g. evaluation pauses active time.
    budgets: dict = field(default_factory=dict)
    work_budget: str | None = None
    work_stopped: bool = False
    details: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class Adapter(Protocol):
    def observe(self, source) -> Observation: ...


def estimate(observation, *, alpha=0.1, exclude_compilation=False):
    if not math.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("Require 0 < alpha <= 1")
    rate = None
    used = 0
    for sample in observation.samples:
        seconds = sample.seconds - (sample.compilation_seconds if exclude_compilation else 0)
        if not all(math.isfinite(v) for v in (seconds, sample.work)):
            raise ValueError("Non-finite progress sample")
        if seconds > 0 and sample.work > 0:
            current = sample.work / seconds
            rate = current if rate is None else alpha * current + (1 - alpha) * rate
            used += 1
    remaining = max(0, observation.target - observation.completed) if observation.target is not None else None
    work_eta = 0.0 if observation.work_stopped or remaining == 0 else remaining / rate if remaining is not None and rate else None
    stages = [dict(name=s.name, count=s.count, conditional=s.conditional, budget=s.budget,
                   eta_seconds=0.0 if s.count == 0 else s.count * s.seconds_each if s.seconds_each is not None else None)
              for s in observation.stages]

    def total(values):
        return sum(values) if all(v is not None for v in values) else None

    other = total([s["eta_seconds"] for s in stages if not s["conditional"]])
    conditional = total([s["eta_seconds"] for s in stages if s["conditional"]])
    total_eta = total([work_eta, other])
    budgets = {}
    for name, budget in observation.budgets.items():
        left = max(0.0, budget["limit_seconds"] - budget["spent_seconds"])
        required = total(([work_eta] if name == observation.work_budget else []) +
                         [s["eta_seconds"] for s in stages if not s["conditional"] and s["budget"] == name])
        if not budget.get("work_estimate_complete", True):
            required = None
        budgets[name] = {**budget, "remaining_seconds": left, "eta_required_seconds": required,
                         "fits": required <= left if required is not None else None}
    now = datetime.now(UTC)
    suspended = observation.status in {"paused", "technical_error", "closed_by_user", "interrupted"}
    return dict(schema=1, observed_at=now.isoformat(), status=observation.status, phase=observation.phase,
                unit=observation.unit, completed=observation.completed, target=observation.target,
                remaining=remaining, ema_alpha=alpha, ema_per_second=rate, rate_samples=used,
                exclude_compilation=exclude_compilation, eta_work_seconds=work_eta,
                eta_other_seconds=other, eta_conditional_seconds=conditional,
                eta_total_seconds=total_eta, eta_assumes_resume=suspended,
                estimated_finish_at=(now + timedelta(seconds=total_eta)).isoformat()
                if total_eta is not None and not suspended and observation.status not in {"unknown", "complete"} else None,
                stages=stages, budgets=budgets, details=observation.details, warnings=observation.warnings)


def duration(seconds):
    return "?" if seconds is None else str(timedelta(seconds=round(seconds)))


def format_progress(result):
    rate = result["ema_per_second"]
    target = "?" if result["target"] is None else f"{result['target']:,.0f}"
    budgets = " ".join(f"{k} {duration(v['remaining_seconds'])} fits={v['fits']}" for k, v in result["budgets"].items())
    rate_text = f"{rate:.1f}" if rate is not None else "?"
    stages = ", ".join(f"{s['name']}:{s['count']}" for s in result["stages"] if not s["conditional"] and s["count"])
    return (f"{result['observed_at']} | {result['status']}/{result['phase']} | "
            f"{result['completed']:,.0f}/{target} {result['unit']} | EMA {rate_text}"
            f" {result['unit']}/s | ETA work {duration(result['eta_work_seconds'])} | "
            f"other {duration(result['eta_other_seconds'])} ({stages or 'none'}) | total {duration(result['eta_total_seconds'])} | "
            f"conditional +{duration(result['eta_conditional_seconds'])} | {budgets}" +
            (f" | finish ~{result['estimated_finish_at']}" if result["estimated_finish_at"] else "") +
            (" | assumes resume" if result["eta_assumes_resume"] else ""))
