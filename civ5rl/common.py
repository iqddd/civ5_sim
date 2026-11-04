"""Versioned configuration, independent random streams and artifact I/O."""

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def seed_for(master, episode, stream, side=-1, namespace="train"):
    value = {
        "schema": 1,
        "namespace": namespace,
        "master_seed": master,
        "episode_id": episode,
        "stream": stream,
        "side": side,
    }
    return int(
        hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16],
        16,
    )


def configure(seed, device="cpu"):
    os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")
    import torch

    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.startswith("cuda"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.manual_seed_all(seed)


def choose(probabilities, rng):
    # One independent CPU draw per decision: unaffected by inference batching.
    cumulative = np.cumsum(np.asarray(probabilities, dtype=np.float64))
    if not np.isfinite(cumulative).all() or cumulative[-1] <= 0:
        raise FloatingPointError("Invalid action probabilities")
    return min(
        int(np.searchsorted(cumulative, rng.random() * cumulative[-1], side="right")),
        len(cumulative) - 1,
    )
