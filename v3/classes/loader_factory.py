from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

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
        "eye_id_1": SlotDescriptor(
            key="eye_id_1",
            kind="id",
            description="Eye side identifier (OD/OS)",
            required=False,
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
        "eye_id_1": str(row.get("eyeID", "")),
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


# ---------------------------------------------------------------------------
# V2 filter / loader helpers (used by V2HyperTower._run_fold)
# ---------------------------------------------------------------------------

def filter_eye_samples(samples: list[dict]) -> list[dict]:
    """Keep any single-eye sample with a valid image, matrix, and label."""
    return [
        s for s in samples
        if s.get("image_1") is not None
        and s.get("matrix_1") is not None
        and s.get("label_1") is not None
    ]


def filter_bilateral_samples(samples: list[dict]) -> list[dict]:
    """Keep only patient-level samples where both eyes are fully present."""
    return [
        s for s in samples
        if s.get("image_1") is not None
        and s.get("matrix_1") is not None
        and s.get("image_2") is not None
        and s.get("matrix_2") is not None
        and s.get("label_1") is not None
    ]


def make_loader(
    samples: list[dict],
    slots: dict,
    *,
    image_transform,
    image_preprocessor=None,
    image_cache=None,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    sampler: Optional[WeightedRandomSampler] = None,
    persistent_workers: bool = False,
) -> DataLoader:
    ds = SlotDataset(
        samples,
        slots,
        image_transform=image_transform,
        image_preprocessor=image_preprocessor,
        image_cache=image_cache,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(shuffle if sampler is None else False),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=slot_collate,
        persistent_workers=(persistent_workers and num_workers > 0),
    )


def build_balanced_sampler(samples: list[dict], label_key: str = "label_1") -> WeightedRandomSampler:
    """Return a WeightedRandomSampler that equalises class frequency for training."""
    from collections import Counter
    labels = [s[label_key] for s in samples]
    counts = Counter(labels)
    weights = [1.0 / counts[lbl] for lbl in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def to_label_tensor(labels, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(labels):
        return labels.to(device=device, dtype=torch.long)
    return torch.as_tensor(labels, dtype=torch.long, device=device)
