"""Explicit trusted case manifests, checked before any worker receives a case."""

from copy import deepcopy

from civ5rl.provenance.common import verify_self_digest
from civ5sim.engine import digest

ALLOWED = {"train", "validation", "diagnostic_train", "diagnostic_dev"}


def manifest_entry_digest(spec):
    return digest({k: v for k, v in spec.items() if k != "provenance"})


case_digest = manifest_entry_digest


def case_manifest(specs, *, source, split, namespace):
    if split not in ALLOWED:
        raise ValueError(f"Closed or unsupported split: {split}")
    cases = [{k: deepcopy(v) for k, v in spec.items() if k != "provenance"} for spec in specs]
    for spec in cases:
        _validate_protocol_case(spec, source, split)
    manifest = {
        "schema": 1,
        "kind": "rl01.cases",
        "source": source,
        "split": split,
        "namespace": namespace,
        "cases": cases,
    }
    manifest["hash"] = digest(manifest)
    return manifest


def bound_cases(manifest):
    validate_manifest(manifest)
    return [
        {
            **deepcopy(spec),
            "provenance": {
                "source": manifest["source"],
                "split": manifest["split"],
                "namespace": manifest["namespace"],
                "case_id": manifest_index,
                "case_hash": manifest_entry_digest(spec),
                "scenario_hash": digest(spec["scenario"]),
                "manifest_hash": manifest["hash"],
            },
        }
        for manifest_index, spec in enumerate(manifest["cases"])
    ]


def _validate_protocol_case(spec, source, split):
    # Protocol validation remains with the protocol; provenance invokes it for
    # both newly constructed and previously serialized manifests.
    from civ5rl.protocols.case_validation import validate_one_case

    validate_one_case(spec, source=source, split=split)


def validate_manifest(manifest):
    if manifest.get("kind") != "rl01.cases" or manifest.get("split") not in ALLOWED:
        raise ValueError("Untrusted/closed case manifest split")
    verify_self_digest(manifest, error_msg="Case manifest hash mismatch")
    for spec in manifest["cases"]:
        _validate_protocol_case(spec, manifest["source"], manifest["split"])


def validate_cases(specs, trusted_manifest):
    if trusted_manifest is None:
        raise ValueError("Fixed cases require an explicit trusted manifest")
    validate_manifest(trusted_manifest)
    for spec in specs:
        provenance = spec.get("provenance", {})
        manifest_index = provenance.get("case_id")
        if type(manifest_index) is not int or not 0 <= manifest_index < len(trusted_manifest["cases"]):
            raise ValueError("Case provenance identifier is missing or invalid")
        expected = trusted_manifest["cases"][manifest_index]
        fields = {
            "source": trusted_manifest["source"],
            "split": trusted_manifest["split"],
            "namespace": trusted_manifest["namespace"],
            "manifest_hash": trusted_manifest["hash"],
            "case_id": manifest_index,
            "case_hash": manifest_entry_digest(expected),
            "scenario_hash": digest(expected["scenario"]),
        }
        if provenance != fields or manifest_entry_digest(spec) != fields["case_hash"]:
            raise ValueError("Case differs from trusted manifest/provenance")
        if spec.get("split", provenance["split"]) != provenance["split"]:
            raise ValueError("Case split differs from trusted provenance")
