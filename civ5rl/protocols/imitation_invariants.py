"""Fixed experimental protocol constraints."""

from civ5rl.protocols.common import require_config

CONTINUATION_CHANGES = {"kind", "output", "epochs", "checkpoint_rule", "continuation"}
MORE_DATA_CHANGES = {
    "kind",
    "output",
    "epochs",
    "training_labels",
    "continuation",
    "more_data",
}
CONTINUATION_BUDGET = {"epochs": 100, "training_seconds": 1800}
MORE_DATA_BUDGET = {"epochs": 10, "training_labels": 100000, "training_seconds": 1800}
OWN_ROOT = {"workers": 4}
OWN_PARENT_FIELDS = (
    "seed",
    "model",
    "numerics",
    "observation",
    "optimizer",
    "teacher_search",
    "labels_per_minibatch",
    "sequence_length",
    "microbatch_sequences",
    "max_gradient_norm",
    "max_calls_per_turn",
)
OWN_SETTINGS = {
    "cycles": 2,
    "new_labels_per_cycle": 25000,
    "dev_labels": 2000,
    "collection_seconds": 1800,
    "dev_collection_seconds": 1800,
    "steps_per_cycle": 500,
    "training_seconds": 1200,
    "old_labels_per_step": 512,
    "new_labels_per_step": 512,
    "measure_every": 100,
    "eta_min_matches": 64,
    "eta_min_seconds": 60,
    "eta_recent_matches": 32,
    "eta_rate_ratio": 1.25,
    "eta_streak": 3,
    "bootstrap_samples": 10000,
    "bootstrap_seed": 20261103,
    "gpu_core_target": None,
    "advisor_reset": "all_tactics_each_query_public_memory_only",
}


def require_values(actual, expected, label):
    require_config(actual, expected, prefix=f"Unsupported {label}.")


def require_changes(actual, parent, expected):
    differences = {
        k for k in actual.keys() | parent.keys() if actual.get(k) != parent.get(k)
    }
    if differences != expected:
        raise ValueError(f"Unexpected config changes: {differences}")
