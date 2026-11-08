"""Versioned configuration, independent random streams and artifact I/O."""

import hashlib
import importlib.util
import json
import linecache
import os
import random
import subprocess
import sys
from pathlib import Path, PurePosixPath

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_module_from_git(module_name, rel_path, rev="HEAD"):
    """Load one trusted Git source blob in memory under an unused module name.

    Dependencies remain the current installed/workspace modules: this is a
    single-module baseline, not a historical environment. __file__ is virtual;
    code requiring resources relative to its source needs a separate checkout.
    A failed import removes its temporary registration. Existing modules are
    never replaced, which prevents an ABBA baseline from poisoning live imports.
    """
    if not module_name or any(not p.isidentifier() for p in module_name.split(".")):
        raise ValueError("Expected a valid dotted module name")
    if module_name in sys.modules:
        raise ValueError(f"Module already loaded: {module_name}")
    name = str(rel_path)
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or ":" in name
        or path.is_absolute()
        or any(p in ("", ".", "..") for p in name.split("/"))
        or path.suffix != ".py"
    ):
        raise ValueError("Expected a repository-relative Python source path")
    if not isinstance(rev, str) or not rev or rev.startswith("-"):
        raise ValueError("Expected a Git revision")
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{rev}^{{commit}}"], cwd=ROOT, text=True
    ).strip()
    raw = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
    source = importlib.util.decode_source(raw)
    filename = f"git:{commit}:{name}"
    spec = importlib.util.spec_from_loader(module_name, loader=None, origin=filename)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = filename
    module.__git_baseline__ = {
        "requested_rev": rev,
        "commit": commit,
        "path": name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "dependency_scope": "current workspace; selected module only",
    }
    # Functions remain inspectable and tracebacks show the actual Git source.
    linecache.cache[filename] = (
        len(raw),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    sys.modules[module_name] = module
    try:
        exec(compile(source, filename, "exec"), module.__dict__)  # noqa: S102 -- explicitly requested trusted Git code
    except BaseException:
        sys.modules.pop(module_name, None)
        linecache.cache.pop(filename, None)
        raise
    return module


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
