"""Device selection and numerical settings for evaluation."""

from __future__ import annotations

import math
import os
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil
import torch

from civ5rl.engine.runtime import is_int_at_least
from civ5rl.engine.runtime.hardware import cgroup_limits, effective_cpu_budget
from civ5rl.provenance.common import read_json, write_json

_options = ContextVar("evaluation_options", default={})


@contextmanager
def evaluation_precision():
    """Evaluate in deterministic FP32, then restore the caller's settings."""
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    precision = torch.get_float32_matmul_precision()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn
        torch.set_float32_matmul_precision(precision)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


@contextmanager
def evaluation_options(**options):
    values = {key: value for key, value in options.items() if value is not None}
    if values:
        EvaluationRuntime(**values).resolve()
    token = _options.set(values)
    try:
        yield
    finally:
        _options.reset(token)


def runtime_for(directory, config, *, cp_only=False):
    path = Path(directory) / "evaluation-runtime.json"
    required = config.get("evaluation_runtime")
    if path.exists():
        defaults = read_json(path)["runtime"]
    else:
        record = read_json(Path(directory) / "run.json") if (Path(directory) / "run.json").exists() else None
        historical = (record is not None and record.get("evaluation_runtime_default") != "auto") or (
            record is None and (Path(directory) / "evaluations.json").exists()
        )
        # The run marker distinguishes a genuinely new experiment from an old
        # partially unpacked run whose evaluations.json is still archived.
        default_workers = int(config.get("workers", 4) or 4)
        if (
            not required
            and not historical
            and "workers" not in _options.get()
            and config.get("self_play", {}).get("enabled")
        ):
            default_workers = _self_play_evaluation_workers()
        defaults = required or {
            "device": "cpu" if historical else "auto",
            "workers": default_workers,
        }
    values = {**defaults, **_options.get()}
    if required is not None and values != required:
        raise ValueError(
            "Evaluation runtime is fixed by this experiment; backend/batching changes are not allowed"
        )
    runtime = EvaluationRuntime(**values).resolve()
    if not path.exists():
        write_json(path, {"schema": 1, "runtime": asdict(runtime)})
    return (
        EvaluationRuntime("cpu", runtime.workers, runtime.max_batch, runtime.wait_seconds)
        if cp_only
        else runtime
    )


def _self_play_evaluation_workers():
    process = psutil.Process()
    affinity = process.cpu_affinity() if hasattr(process, "cpu_affinity") else range(os.cpu_count() or 1)
    available = len(affinity)
    if sys.platform == "linux":
        limits = cgroup_limits()
        available = math.floor(effective_cpu_budget(affinity, limits["cpu_quota"], limits["cpuset"]))
    return max(1, min(24, available - 1))


@dataclass(frozen=True)
class EvaluationRuntime:
    device: str = "auto"
    workers: int = 4
    max_batch: int = 32
    wait_seconds: float = 0.001

    def resolve(self):
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Evaluation device must be auto, cpu or cuda")
        if not is_int_at_least(self.workers) or not is_int_at_least(self.max_batch):
            raise ValueError("Evaluation workers and max_batch must be positive integers")
        if not math.isfinite(self.wait_seconds) or self.wait_seconds < 0:
            raise ValueError("Evaluation wait_seconds must be finite and nonnegative")
        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")
        return EvaluationRuntime(device, self.workers, self.max_batch, self.wait_seconds)

    def identity(self):
        value = {
            **asdict(self),
            "precision": "float32",
            "evaluator_version": 1,
            "compiled_spatial_heads": self.device == "cuda",
            "compile_mode": "default" if self.device == "cuda" else None,
            "attention_backend": "compact_fp32" if self.device == "cuda" else "dense_fp32",
            "public_inputs": "worker_encoder_shared_slots_v1",
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        }
        if self.device == "cuda":
            value.update(
                gpu=torch.cuda.get_device_name(),
                capability=list(torch.cuda.get_device_capability()),
                cuda=torch.version.cuda,
            )
        return value
