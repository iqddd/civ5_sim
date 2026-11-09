from enum import StrEnum


class PolicyVariant(StrEnum):
    CURRENT_ONLY_MLP = "A"
    CONTACTS_MLP = "B"
    CONTACTS_GRU = "C"
    CONTACTS_GRU_C2 = "C2"


RECURRENT_VARIANTS = (PolicyVariant.CONTACTS_GRU, PolicyVariant.CONTACTS_GRU_C2)

