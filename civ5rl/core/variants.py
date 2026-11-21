from dataclasses import dataclass, replace
from enum import StrEnum

from civ5rl.core.schema import C2_HEX_BLOCKS, HEX_BLOCKS


class PolicyVariant(StrEnum):
    CURRENT_ONLY_MLP = "A"
    CONTACTS_MLP = "B"
    CONTACTS_GRU = "C"
    CONTACTS_GRU_C2 = "C2"
    CONTACTS_GRU_C2_PREVIEW = "C2P"
    CONTACTS_GRU_C2_PREVIEW_ACTIONS = "C2PA"


@dataclass(frozen=True)
class VariantSpec:
    variant: PolicyVariant
    uses_contact_history: bool
    is_recurrent: bool
    hex_blocks: int
    uses_unit_attention: bool
    bounded_time_features: bool
    attack_preview: bool = False
    action_context: bool = False


VARIANT_SPECS: dict[PolicyVariant, VariantSpec] = {
    PolicyVariant.CONTACTS_GRU_C2_PREVIEW: VariantSpec(
        variant=PolicyVariant.CONTACTS_GRU_C2_PREVIEW, uses_contact_history=True,
        is_recurrent=True, hex_blocks=C2_HEX_BLOCKS, uses_unit_attention=True,
        bounded_time_features=True, attack_preview=True,
    ),
    PolicyVariant.CURRENT_ONLY_MLP: VariantSpec(
        variant=PolicyVariant.CURRENT_ONLY_MLP,
        uses_contact_history=False,
        is_recurrent=False,
        hex_blocks=HEX_BLOCKS,
        uses_unit_attention=False,
        bounded_time_features=False,
    ),
    PolicyVariant.CONTACTS_MLP: VariantSpec(
        variant=PolicyVariant.CONTACTS_MLP,
        uses_contact_history=True,
        is_recurrent=False,
        hex_blocks=HEX_BLOCKS,
        uses_unit_attention=False,
        bounded_time_features=False,
    ),
    PolicyVariant.CONTACTS_GRU: VariantSpec(
        variant=PolicyVariant.CONTACTS_GRU,
        uses_contact_history=True,
        is_recurrent=True,
        hex_blocks=HEX_BLOCKS,
        uses_unit_attention=False,
        bounded_time_features=False,
    ),
    PolicyVariant.CONTACTS_GRU_C2: VariantSpec(
        variant=PolicyVariant.CONTACTS_GRU_C2,
        uses_contact_history=True,
        is_recurrent=True,
        hex_blocks=C2_HEX_BLOCKS,
        uses_unit_attention=True,
        bounded_time_features=True,
    ),
}

VARIANT_SPECS[PolicyVariant.CONTACTS_GRU_C2_PREVIEW_ACTIONS] = replace(
    VARIANT_SPECS[PolicyVariant.CONTACTS_GRU_C2_PREVIEW],
    variant=PolicyVariant.CONTACTS_GRU_C2_PREVIEW_ACTIONS,
    action_context=True,
)
RECURRENT_VARIANTS = tuple(variant for variant in PolicyVariant if VARIANT_SPECS[variant].is_recurrent)


def variant_spec(variant: PolicyVariant | str) -> VariantSpec:
    return VARIANT_SPECS[PolicyVariant(variant)]
