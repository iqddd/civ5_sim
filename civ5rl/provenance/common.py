"""Versioned configuration, independent random streams and artifact I/O."""

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


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


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def verify_self_digest(
    value, *, error_msg="Self digest mismatch", exception_type=ValueError
):
    from civ5sim.engine import digest

    if value.get("hash") != digest({k: v for k, v in value.items() if k != "hash"}):
        raise exception_type(error_msg)
    return value


def model_tensor_sha256(model):
    value = hashlib.sha256()
    for name, tensor in sorted(model.items()):
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def load_verified_checkpoint(
    policy,
    path,
    *,
    expected_update,
    expected_variant="C",
    source_error="Unexpected source checkpoint",
    equality_error="Model parameters differ",
):
    import torch

    value = torch.load(path, map_location="cpu", weights_only=False)
    if (
        value["variant"] != expected_variant
        or value["state"]["update"] != expected_update
    ):
        raise ValueError(source_error)
    policy.load_state_dict(value["model"], strict=True)
    if not all(
        torch.equal(v.detach().cpu(), value["model"][k])
        for k, v in policy.state_dict().items()
    ):
        raise ValueError(equality_error)
    return value, {
        "source_checkpoint": str(path),
        "source_sha256": file_sha256(path),
        "model_tensors_sha256": model_tensor_sha256(value["model"]),
        "all_model_parameters_bitwise_equal": True,
        "source_optimizer_loaded": False,
        "source_training_rng_loaded": False,
    }
