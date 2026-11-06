"""Stable RL-01 feature layouts and model dimensions (no torch dependency)."""


def _layout(fields):
    result = {}
    offset = 0
    for name, width in fields:
        result[name] = offset if width == 1 else slice(offset, offset + width)
        offset += width
    return result, offset


TILE, TILE_DIM = _layout(
    (
        ("terrain", 7),
        ("elevation", 4),
        ("feature", 9),
        ("rivers", 6),
        ("neighbors", 6),
        ("position_passable", 3),
        ("visible", 1),
        ("visibility_age", 1),
        ("seen", 1),
        ("own_type", 5),
        ("own_status", 8),
        ("enemy_type", 5),
        ("enemy_status", 2),
        ("contact_count", 5),
        ("contact_age", 5),
        ("contact_hp", 5),
    )
)
GLOBAL, GLOBAL_DIM = _layout(
    (
        ("own_roster", 5),
        ("enemy_roster", 5),
        ("own_alive", 5),
        ("enemy_visible", 5),
        ("contacts", 5),
        ("turn", 1),
        ("elapsed", 1),
        ("new_turn", 1),
        ("width", 1),
        ("height", 1),
        ("first_side", 1),
        ("random_damage", 1),
        ("max_rounds", 1),
    )
)
PREVIOUS, PREV_ACTION_DIM = _layout(
    (
        ("kind", 8),
        ("source_type", 6),
        ("source_xy", 2),
        ("target_xy", 2),
        ("has_source", 1),
        ("has_target", 1),
    )
)
ACTION, ACTION_DIM = _layout(
    (
        ("kind", 7),
        ("source_type", 5),
        ("displacement", 3),
        ("has_source", 1),
        ("has_target", 1),
    )
)

AGE_CAP = 32
ROSTER_SCALE = 8
MAX_ROSTER_UNITS = 13
MAX_MAP_AXIS = 16
MAX_MAP_CELLS = MAX_MAP_AXIS * MAX_MAP_AXIS
HEX_DIRECTIONS = 6
HEX_DIM = 128
HEX_BLOCKS = 4
HEX_MLP_DIM = 256
HIDDEN_SIZE = 256
MEMORY_MLP_DIM = 768
CONTEXT_INPUT_DIM = 2 * HEX_DIM + GLOBAL_DIM + PREV_ACTION_DIM
ACTION_INPUT_DIM = HIDDEN_SIZE + 2 * HEX_DIM + ACTION_DIM
