"""Stable RL-01 feature layouts and model dimensions (no torch dependency)."""

from civ5sim.engine import ELEVATIONS, FEATURES, KINDS, TERRAINS, UNIT_TYPES


# Feature widths are part of the checkpoint format. Fail explicitly if the
# engine changes its categorical vocabulary.
if tuple(map(tuple, (TERRAINS, ELEVATIONS, FEATURES, UNIT_TYPES, KINDS))) != (
    ("grass", "plains", "desert", "tundra", "snow", "coast", "ocean"),
    ("flat", "hills", "mountain", "water"),
    (
        None,
        "forest",
        "jungle",
        "marsh",
        "oasis",
        "ice",
        "flood_plains",
        "fallout",
        "atoll",
    ),
    ("warrior", "archer", "chariot_archer", "spearman", "horseman"),
    ("move", "melee", "ranged", "fortify", "skip", "end_turn", "swap"),
):
    raise ValueError("Engine categories do not match the RL-01 feature schema")

HEX_DIRECTIONS = 6
OPPOSITE_HEX_OFFSET = HEX_DIRECTIONS // 2
SELF_AND_HEX_DIRECTIONS = HEX_DIRECTIONS + 1


def opposite_direction(direction: int) -> int:
    return (direction + OPPOSITE_HEX_OFFSET) % HEX_DIRECTIONS


def _layout(fields):
    result = {}
    offset = 0
    for name, width in fields:
        result[name] = offset if width == 1 else slice(offset, offset + width)
        offset += width
    return result, offset


TILE, TILE_DIM = _layout(
    (
        ("terrain", len(TERRAINS)),
        ("elevation", len(ELEVATIONS)),
        ("feature", len(FEATURES)),
        ("rivers", HEX_DIRECTIONS),
        ("neighbors", HEX_DIRECTIONS),
        ("position_passable", 3),
        ("visible", 1),
        ("visibility_age", 1),
        ("seen", 1),
        ("own_type", len(UNIT_TYPES)),
        ("own_status", 8),
        ("enemy_type", len(UNIT_TYPES)),
        ("enemy_status", 2),
        ("contact_count", len(UNIT_TYPES)),
        ("contact_age", len(UNIT_TYPES)),
        ("contact_hp", len(UNIT_TYPES)),
    )
)
GLOBAL, GLOBAL_DIM = _layout(
    (
        ("own_roster", len(UNIT_TYPES)),
        ("enemy_roster", len(UNIT_TYPES)),
        ("own_alive", len(UNIT_TYPES)),
        ("enemy_visible", len(UNIT_TYPES)),
        ("contacts", len(UNIT_TYPES)),
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
        ("kind", len(KINDS) + 1),
        ("source_type", len(UNIT_TYPES) + 1),
        ("source_xy", 2),
        ("target_xy", 2),
        ("has_source", 1),
        ("has_target", 1),
    )
)
ACTION, ACTION_DIM = _layout(
    (
        ("kind", len(KINDS)),
        ("source_type", len(UNIT_TYPES)),
        ("displacement", 3),
        ("has_source", 1),
        ("has_target", 1),
    )
)

PREVIOUS_START_KIND = PREVIOUS["kind"].start + len(KINDS)
PREVIOUS_START_SOURCE_TYPE = PREVIOUS["source_type"].start + len(UNIT_TYPES)

AGE_CAP = 32
ROSTER_SCALE = 8
MAX_ROSTER_UNITS = 13
MAX_MAP_AXIS = 16
MAX_MAP_CELLS = MAX_MAP_AXIS * MAX_MAP_AXIS
HEX_DIM = 128
HEX_BLOCKS = 4
HEX_MLP_DIM = 256
HIDDEN_SIZE = 256
MEMORY_MLP_DIM = 768
CONTEXT_INPUT_DIM = 2 * HEX_DIM + GLOBAL_DIM + PREV_ACTION_DIM
ACTION_INPUT_DIM = HIDDEN_SIZE + 2 * HEX_DIM + ACTION_DIM
