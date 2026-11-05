"""Validation of trusted protocol-specific case contracts."""


def validate_one_case(spec, *, source, split):
    is_one = (
        source == "rl01.diagnostic.D0-one-decision"
        or spec.get("diagnostic_protocol") == "D0-one-decision"
        or (source.startswith("rl01.diagnostic.") and "learner_decision_limit" in spec)
    )
    if not is_one:
        return
    if (
        source != "rl01.diagnostic.D0-one-decision"
        or split not in ("diagnostic_train", "diagnostic_dev")
        or spec.get("diagnostic_protocol") != "D0-one-decision"
        or type(spec.get("learner_decision_limit")) is not int
        or spec["learner_decision_limit"] != 1
    ):
        raise ValueError("Invalid one-decision protocol case")
