"""Normalize execution identities with explicit source comparison policy."""

from pathlib import Path


def execution_identity(
    identity, *, include_markdown: bool, include_corpus: bool = True
):
    fields = {key: value for key, value in identity.items() if key != "git_commit"}
    if not include_corpus:
        fields.pop("corpus_hash", None)
    if "source_hashes" in fields:
        fields["source_hashes"] = {
            path: digest
            for path, digest in fields["source_hashes"].items()
            if include_markdown or Path(path).suffix.lower() != ".md"
        }
    return fields
