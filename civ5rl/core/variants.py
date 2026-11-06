from enum import StrEnum


class PolicyVariant(StrEnum):
    CURRENT_ONLY_MLP = "A"
    CONTACTS_MLP = "B"
    CONTACTS_GRU = "C"
