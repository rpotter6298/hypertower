from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Any

import pandas as pd


@dataclass(frozen=True)
class SlotDescriptor:
    """
    Metadata for a generic batch slot key (e.g., image_1, matrix_1).
    """

    key: str
    kind: str
    description: str
    required: bool = True
    shape_hint: str | None = None


class DatasetProfile(Protocol):
    """
    Dataset-specific wiring that stays outside the generic V2 engine.
    """

    name: str
    patient_col: str
    label_col: str

    def slot_descriptors(self) -> dict[str, SlotDescriptor]:
        ...

    def semantic_aliases(self) -> dict[str, str]:
        ...

    def build_samples(self, *, df: pd.DataFrame, clinical: Any) -> list[dict[str, Any]]:
        ...


@dataclass(frozen=True)
class SimpleDatasetProfile:
    name: str
    patient_col: str
    label_col: str
    slots: dict[str, SlotDescriptor]
    aliases: dict[str, str]

    def slot_descriptors(self) -> dict[str, SlotDescriptor]:
        return dict(self.slots)

    def semantic_aliases(self) -> dict[str, str]:
        return dict(self.aliases)
