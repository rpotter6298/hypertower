from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from torch.utils.data import DataLoader

from .network_manager import LoaderBundle, PatientSplit
from .slot_dataset import SlotDataset, slot_collate
from .profiles.base import SlotDescriptor, SimpleDatasetProfile


def _default_slot_descriptors(patient_col: str, label_col: str) -> dict[str, SlotDescriptor]:
    return {
        "id_1": SlotDescriptor(
            key="id_1",
            kind="id",
            description=f"Patient identifier column ({patient_col})",
            required=True,
            shape_hint="scalar",
        ),
        "label_1": SlotDescriptor(
            key="label_1",
            kind="label",
            description=f"Label column ({label_col})",
            required=True,
            shape_hint="scalar",
        ),
        "image_1": SlotDescriptor(
            key="image_1",
            kind="image",
            description="Primary image slot",
            required=False,
            shape_hint="HWC or CHW",
        ),
        "matrix_1": SlotDescriptor(
            key="matrix_1",
            kind="matrix",
            description="Primary matrix slot",
            required=False,
            shape_hint="[feature_dim]",
        ),
    }


def _row_to_sample(
    row: Any,
    *,
    clinical: Any,
    patient_col: str,
    label_col: str,
) -> dict[str, Any]:
    return {
        "id_1": row[patient_col],
        "label_1": row[label_col],
        "image_1": clinical.get_image_path(row) if hasattr(clinical, "get_image_path") else None,
        "matrix_1": clinical.vectorize_row(row) if hasattr(clinical, "vectorize_row") else None,
    }


@dataclass
class SlotLoaderFactory:
    """
    Generic loader factory that emits dict batches keyed by slot names.
    """

    image_transform: Optional[Callable] = None
    matrix_transform: Optional[Callable] = None
    num_workers: int = 0

    def build(
        self,
        *,
        clinical: Any,
        split: PatientSplit,
        args: Any,
        fold: int,
        profile: Optional[Any] = None,
    ) -> LoaderBundle:
        batch_size = int(getattr(args, "batch_size", 8))
        slot_desc = self._resolve_slot_descriptors(clinical=clinical, profile=profile)

        train_samples = self._build_samples(split.train, clinical, profile, slot_desc)
        val_samples = self._build_samples(split.val, clinical, profile, slot_desc)
        holdout_samples = (
            self._build_samples(split.holdout, clinical, profile, slot_desc)
            if split.holdout is not None
            else None
        )

        train_loader = DataLoader(
            SlotDataset(
                train_samples,
                slot_desc,
                image_transform=self.image_transform,
                matrix_transform=self.matrix_transform,
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=slot_collate,
        )
        val_loader = DataLoader(
            SlotDataset(
                val_samples,
                slot_desc,
                image_transform=self.image_transform,
                matrix_transform=self.matrix_transform,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=slot_collate,
        )
        holdout_loader = None
        if holdout_samples is not None:
            holdout_loader = DataLoader(
                SlotDataset(
                    holdout_samples,
                    slot_desc,
                    image_transform=self.image_transform,
                    matrix_transform=self.matrix_transform,
                ),
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=slot_collate,
            )
        return LoaderBundle(train=train_loader, val=val_loader, holdout=holdout_loader)

    @staticmethod
    def _resolve_slot_descriptors(
        *,
        clinical: Any,
        profile: Optional[Any],
    ) -> dict[str, SlotDescriptor]:
        if profile is not None and hasattr(profile, "slot_descriptors"):
            return profile.slot_descriptors()
        patient_col = getattr(clinical, "patient_col", "Patient ID")
        label_col = getattr(clinical, "label_col", "Diagnosis")
        return _default_slot_descriptors(patient_col, label_col)

    @staticmethod
    def _build_samples(
        df,
        clinical: Any,
        profile: Optional[Any],
        slot_desc: dict[str, SlotDescriptor],
    ) -> list[dict[str, Any]]:
        if df is None or df.empty:
            return []
        if profile is not None and hasattr(profile, "build_samples"):
            return profile.build_samples(df=df, clinical=clinical)

        patient_col = getattr(profile, "patient_col", None) if profile is not None else None
        label_col = getattr(profile, "label_col", None) if profile is not None else None
        pcol = patient_col or "Patient ID"
        lcol = label_col or getattr(clinical, "label_col", "Diagnosis")
        samples = []
        for _, row in df.iterrows():
            sample = _row_to_sample(row, clinical=clinical, patient_col=pcol, label_col=lcol)
            for key in slot_desc.keys():
                sample.setdefault(key, None)
            samples.append(sample)
        return samples
