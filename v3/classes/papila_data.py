from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd

from v3.classes.data_bundle import DataBundle
from v3.classes.papila_builders import build_papila_data


@dataclass
class PapilaData:
    """
    V2-friendly wrapper around the DataBundle pipeline.

    Keeps all formatting/normalization behavior from build_papila_clinical,
    but exposes a minimal surface area for the V2 engine.
    """

    clinical: DataBundle
    patient_col: str = "Patient ID"

    @property
    def df(self) -> pd.DataFrame:
        return self.clinical.df

    @property
    def label_col(self) -> str:
        return self.clinical.label_col

    @property
    def feature_dim(self) -> int:
        return self.clinical.feature_dim

    def get_image_path(self, row: pd.Series):
        return self.clinical.get_image_path(row)

    def vectorize_row(self, row: pd.Series):
        return self.clinical.vectorize_row(row)

    @classmethod
    def from_dirs(
        cls,
        *,
        image_dir: str,
        clinical_dir: str,
        label_col: str,
        cat_cols: Iterable[str],
        n_splits: int = 5,
        random_seed: int = 42,
    ) -> "PapilaData":
        clinical = build_papila_data(
            image_dir=image_dir,
            clinical_dir=clinical_dir,
            label_col=label_col,
            cat_cols=list(cat_cols),
            n_splits=n_splits,
            random_seed=random_seed,
        )
        return cls(clinical=clinical)
