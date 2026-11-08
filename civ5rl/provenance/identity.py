"""Normalize execution identities with explicit source comparison policy."""

from pathlib import Path


def is_readonly_source(path):
    """One source-role policy for discovery and old/new passport comparisons.

    Mixed orchestration/evaluator modules remain execution code. New tooling
    belongs in the dedicated reporting/deploy namespaces or non-execution CLI.
    """
    name = str(path).replace("\\", "/")
    if name.startswith(("civ5rl/reporting/", "civ5rl/deploy/", "tests/")):
        return True
    if name in (
        "civ5rl/provenance/audit.py",
        "civ5rl/evaluation/audit.py",
        "civ5rl/experiments/report_cp_imitation.py",
    ):
        return True
    if name.startswith("tools/"):
        # CLI that starts training/data generation/evaluation or builds the
        # simulator can affect the experiment and must remain pinned.
        return not (
            name.startswith(
                (
                    "tools/run_",
                    "tools/train_",
                    "tools/prepare_",
                    "tools/evaluate_",
                    "tools/build_",
                )
            )
            or name in (
                "tools/tactical_corpus.py",
                "tools/_build_common.py",
                "tools/_imitation_benchmark.py",
            )
        )
    return False


def execution_identity(
    identity, *, include_markdown: bool, include_corpus: bool = True
):
    # Config equality is checked by execution callers, and its saved hash by
    # frozen-passport verification. Do not compare parent and child config
    # hashes here: allowed config differences are validated by lineage callers.
    fields = {
        key: value
        for key, value in identity.items()
        if key not in ("git_commit", "config_hash")
    }
    if not include_corpus:
        fields.pop("corpus_hash", None)
    if "source_hashes" in fields:
        fields["source_hashes"] = {
            path: digest
            for path, digest in fields["source_hashes"].items()
            if not is_readonly_source(path)
            and (include_markdown or Path(path).suffix.lower() != ".md")
        }
    return fields
