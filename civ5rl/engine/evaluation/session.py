"""Sequential evaluation sessions, checkpoint identities and simulator pools."""

from __future__ import annotations

import platform
import sys
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import torch

from civ5rl.core.agent import RLAgent
from civ5rl.core.model import Policy
from civ5rl.engine.runtime.shared import pool
from civ5rl.provenance.common import ROOT, file_sha256
from civ5rl.provenance.identity import policy_hash
from civ5sim import combat_ai
from civ5sim.engine import digest

_session = ContextVar("evaluation_session", default=None)


class _EvaluationSession:
    def __init__(self):
        self.policies = OrderedDict()
        self.checkpoint_hashes = {}
        self.workers = None
        self.executor = None

    def close_pool(self):
        if self.executor is not None:
            executor, self.executor = self.executor, None
            executor.shutdown(wait=True, cancel_futures=True)


@contextmanager
def evaluation_session():
    """Keep compiled checkpoints and simulator processes for one sequential run."""
    session = _EvaluationSession()
    token = _session.set(session)
    try:
        yield
    finally:
        _session.reset(token)
        session.close_pool()


@contextmanager
def evaluation_worker_pool(workers):
    session = _session.get()
    if session is None:
        with pool(workers) as executor:
            yield executor
        return
    if session.workers != workers:
        session.close_pool()
        session.workers = workers
    if session.executor is None:
        session.executor = pool(workers)
    try:
        yield session.executor
    except BaseException:
        session.close_pool()
        raise


def started_pool_workers(executor):
    """Count children started by the lazy spawn pool for warmup diagnostics."""
    return len(executor._processes or {}) if executor is not None else 0


def prepare_evaluation_policy(policy, device):
    """FP32 inference with compiled spatial/heads; no training AMP or graph pools."""
    policy = policy.to(device).eval()
    if str(device).startswith("cuda"):
        policy.configure_execution(attention_backend="compact_fp32")
        policy.enable_compile(dynamic=True, mode="default", spatial_mode="default")
    return policy


def _checkpoint_identity(checkpoint):
    path = Path(checkpoint).resolve()
    stat = path.stat()
    return path, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def evaluation_policy_hash(checkpoint):
    session = _session.get()
    key = _checkpoint_identity(checkpoint) if session is not None else None
    if session is not None and key in session.checkpoint_hashes:
        return session.checkpoint_hashes[key]
    fingerprint = policy_hash(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    if session is not None:
        session.checkpoint_hashes[key] = fingerprint
    return fingerprint


def evaluation_key(config, plan, out, path, purpose, runtime_identity, *, opponent="cp"):
    """Return the canonical cache key without running matches."""
    fingerprint = (
        "cp_mirror_" + digest(config["teacher_search"])[:32]
        if path == "cp"
        else evaluation_policy_hash(out / path)
    )

    # Version behavior explicitly; source formatting and module names are metadata.
    evaluator = digest(
        {
            "version": 1,
            "platform": platform.system(),
            "torch": str(torch.__version__),
            "planner": file_sha256(combat_ai.LIBRARY_PATH),
            "kernel": file_sha256(
                ROOT / "build" / ("combat.dll" if sys.platform == "win32" else "libcombat.so")
            ),
        }
    )
    key = digest(
        {
            "purpose": purpose,
            "policy": fingerprint,
            "mode": "argmax",
            "temperature": config.get("policy_temperature", 1.0),
            "cases": [c["case_hash"] for c in plan[purpose]],
            "evaluator": evaluator,
            "runtime": runtime_identity,
            "search": config["teacher_search"],
            **({"opponent": opponent, "agent": file_sha256(ROOT / "civ5sim/cpv2_agent.py")} if opponent == "cpv2" else {}),
        }
    )
    return fingerprint, key


def load_evaluation_policy(checkpoint, device) -> Policy:
    """Reuse the two most recent CUDA checkpoints without changing their weights."""
    session = _session.get()
    key = None
    if session is not None and str(device).startswith("cuda"):
        key = (str(device), *_checkpoint_identity(checkpoint))
        cached = session.policies.get(key)
        if cached is not None:
            session.policies.move_to_end(key)
            return cached
    agent = RLAgent.from_checkpoint(checkpoint, "cpu", "argmax")
    policy = prepare_evaluation_policy(agent.policy, device)
    if key is not None:
        session.policies[key] = policy
        if len(session.policies) > 2:
            session.policies.popitem(last=False)
    return policy
