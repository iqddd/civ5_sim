"""Boundary-only CUDA diagnostics; no kernels, synchronization or allocation history.

Compare driver-free, allocated, reserved and private-pool counters at equivalent
update boundaries. Reserved bytes alone do not establish a leak; stable allocated
bytes do not rule out graph-pool growth. These categories overlap and must not be
added as independent memory volumes. empty_cache cannot release live graph pools.
"""

from __future__ import annotations

import gzip
import json
import sys
from os import PathLike

import torch


def sample_cuda_memory(policy: torch.nn.Module) -> dict[str, object] | None:
    """Read host allocator/graph metadata without changing CUDA execution.

    Never add synchronize(), tensor.item(), empty_cache(), CUDA kernels, memory
    history or graph resets here: boundary telemetry must not drain GPU queues,
    change allocator state or perturb the memory-retention problem being measured.
    Inspect existing graph managers only; do not create pools or advance counters.

    Allocator/retry/OOM counters are process-cumulative, not per-update. Private
    pools are correlated with GRU cache entries and optional version-specific
    Inductor metadata; the latter covers the device/current thread, not one policy.
    Unsupported metadata is unavailable rather than an inferred zero. Full layout
    snapshots belong in dump_oom_snapshot after a failure, outside the hot path.
    """
    device = next(policy.parameters()).device
    if device.type != "cuda" or not torch.cuda.is_initialized():
        return None
    stats = torch.cuda.memory_stats_as_nested_dict(device)
    pools = stats.get("reserved_bytes_by_private_pools")
    row: dict[str, object] = {
        "device": str(device),
        "allocator_backend": torch.cuda.get_allocator_backend(),
        "allocator": {
            key: stats.get(key)
            for key in (
                "allocated_bytes",
                "reserved_bytes",
                "active_bytes",
                "inactive_split_bytes",
                "requested_bytes",
                "segment",
                "num_alloc_retries",
                "num_ooms",
                "num_sync_all_streams",
                "num_device_alloc",
                "num_device_free",
            )
        },
        "private_pools": None
        if pools is None
        else [{"pool_id": list(key), "reserved_bytes": value} for key, value in pools.items()],
        "inductor": {"status": "not_loaded"},
        "gru": None,
    }
    runner = getattr(policy, "_memory_runner", None)
    if runner is not None and hasattr(runner, "entries"):
        row["gru"] = {
            "entries": len(runner.entries),
            "max_shapes": runner.max_shapes,
            "captures": runner.captures,
            "seen_shapes": len(runner._seen),
            "shapes": [
                {
                    "x": list(key[0]),
                    "hidden": list(key[1]),
                    "active": entry.active,
                    "forward_pool_id": list(entry.forward_graph.pool()),
                    "backward_pool_id": list(entry.backward_graph.pool()),
                }
                for key, entry in runner.entries.items()
            ],
        }
    # Optional version-specific metadata, from an existing manager only. Reading
    # itertools.count via __reduce__ does not advance it or traverse graph trees.
    module = sys.modules.get("torch._inductor.cudagraph_trees")
    if module is not None:
        try:
            manager = module.get_manager(device.index, create_if_none_exists=False)
            row["inductor"] = (
                {"status": "no_manager"}
                if manager is None
                else {
                    "status": "available",
                    "scope": "device_current_thread_not_one_policy",
                    "graphs_recorded": manager.graph_counter.__reduce__()[1][0],
                    "root_functions": len(manager.roots),
                    "pool_id": list(manager.cuda_graphs_thread_pool),
                }
            )
        except Exception as error:  # noqa: BLE001 -- metadata must not determine training success
            row["inductor"] = {"status": "unavailable", "error": repr(error)}
    return row


def dump_oom_snapshot(path: str | PathLike[str]) -> None:
    """Full allocator layout only after failure; allocation history stays disabled."""
    snapshot = torch.cuda.memory_snapshot()
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(snapshot, stream)
