"""RL runtime modules."""


def is_int_at_least(value, minimum=1):
    """Validate integer options without accepting booleans."""
    return type(value) is int and value >= minimum
