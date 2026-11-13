"""Boundary-only CUDA diagnostics; no kernels, synchronization or allocation history."""

import gzip
import json
import sys

import torch


def sample_cuda_memory(policy):
    device = next(policy.parameters()).device
    if device.type != "cuda" or not torch.cuda.is_initialized():
        return None
    stats = torch.cuda.memory_stats_as_nested_dict(device)
    pools = stats.get("reserved_bytes_by_private_pools")
    row = {
        "device": str(device),
        "allocator_backend": torch.cuda.get_allocator_backend(),
        "allocator": {
            key: stats.get(key)
            for key in (
                "allocated_bytes", "reserved_bytes", "active_bytes", "inactive_split_bytes",
                "requested_bytes", "segment", "num_alloc_retries", "num_ooms",
                "num_sync_all_streams", "num_device_alloc", "num_device_free",
            )
        },
        "private_pools": None if pools is None else [
            {"pool_id": list(key), "reserved_bytes": value}
            for key, value in pools.items()
        ],
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
                    "x": list(key[0]), "hidden": list(key[1]), "active": entry.active,
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
            row["inductor"] = {"status": "no_manager"} if manager is None else {
                "status": "available",
                "scope": "device_current_thread_not_one_policy",
                "graphs_recorded": manager.graph_counter.__reduce__()[1][0],
                "root_functions": len(manager.roots),
                "pool_id": list(manager.cuda_graphs_thread_pool),
            }
        except Exception as error:  # metadata must not determine training success
            row["inductor"] = {"status": "unavailable", "error": repr(error)}
    return row


def dump_oom_snapshot(path):
    """Full allocator layout only after failure; allocation history stays disabled."""
    snapshot = torch.cuda.memory_snapshot()
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(snapshot, stream)
