"""PredictionStore — unified per-epoch prediction tensor across all folds.

Tensor shape: (n_folds, n_epochs, n_samples, n_heads, n_classes)

The meaning of "sample" depends on tower_mode:
  single   — each eye is a sample; sample_ids like "5OD", "14OS"
  ensemble — each patient is a sample; sample_ids like "5", "14"
  fused    — same as ensemble
  bilateral— same as ensemble

Head names by mode:
  single    : ["fused", "img", "md"]
  ensemble  : ["od_fused", "od_img", "od_md", "os_fused", "os_img", "os_md"]
  fused     : ["od_fused", "od_img", "od_md", "os_fused", "os_img", "os_md", "bilat_fused"]
  bilateral : ["fused", "img_joint", "md_joint"]
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def head_names_for_mode(tower_mode: str, *, fused_head: bool = False) -> list[str]:
    """Return canonical head name list for a given tower_mode."""
    if tower_mode in ("single", "classic"):
        return ["fused", "img", "md"]
    if tower_mode == "ensemble":
        names = ["od_fused", "od_img", "od_md", "os_fused", "os_img", "os_md"]
        return names + ["bilat_fused"] if fused_head else names
    if tower_mode == "bilateral":
        return ["fused", "img_joint", "md_joint"]
    raise ValueError(f"Unknown tower_mode: {tower_mode!r}")


class PredictionStore:
    """
    Stores per-epoch predictions for every sample, head, and fold in one tensor.

    Usage
    -----
    # Build once before the fold loop:
    store = PredictionStore(
        sample_ids=all_eye_or_patient_ids,
        y_true=all_labels,
        head_names=head_names_for_mode(tower_mode, fused_head=args.fused_head),
        n_folds=n_folds,
        n_epochs=total_epochs,
        n_classes=num_classes,
    )

    # Inside each epoch, after collecting probs:
    store.record(fold, epoch, patient_ids_batch, "od_fused", probs_od)
    store.set_split(fold, train_ids, "train")
    store.set_split(fold, val_ids,   "val")

    # After all folds:
    store.save(run_dir / "predictions.npz")

    # Load and query:
    store = PredictionStore.load("predictions.npz")
    store.query("5", "od_fused", fold=0)   # → (n_epochs, n_classes)
    store.query("5", "od_fused")            # → (n_folds, n_epochs, n_classes)
    store.get_split("5", fold=0)            # → "train"
    """

    def __init__(
        self,
        sample_ids: Sequence[str],
        y_true: Sequence[int],
        head_names: Sequence[str],
        n_folds: int,
        n_epochs: int,
        n_classes: int,
    ):
        self.sample_ids = np.array(sample_ids, dtype=object)
        self.y_true     = np.array(y_true,     dtype=np.int64)
        self.head_names = np.array(head_names, dtype=object)
        self.n_folds    = n_folds
        self.n_epochs   = n_epochs
        self.n_classes  = n_classes

        n_samples = len(self.sample_ids)
        n_heads   = len(self.head_names)

        self.probs = np.full(
            (n_folds, n_epochs, n_samples, n_heads, n_classes),
            fill_value=np.nan,
            dtype=np.float32,
        )
        self.split = np.full((n_folds, n_samples), fill_value="", dtype=object)

        self._sid_index: dict[str, int]  = {str(s): i for i, s in enumerate(self.sample_ids)}
        self._head_index: dict[str, int] = {str(h): i for i, h in enumerate(self.head_names)}

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def record(
        self,
        fold: int,
        epoch: int,
        sample_ids: Sequence[str],
        head_name: str,
        probs: np.ndarray,
    ) -> None:
        """Record a batch of predictions for one head.

        Args:
            fold:       0-indexed fold number
            epoch:      0-indexed epoch number
            sample_ids: sequence of sample ID strings (length B)
            head_name:  which head — must be in self.head_names
            probs:      (B, n_classes) probability array
        """
        head_idx = self._head_index.get(head_name)
        if head_idx is None:
            return  # head not active in this mode — skip silently
        for i, sid in enumerate(sample_ids):
            s_idx = self._sid_index.get(str(sid))
            if s_idx is not None:
                self.probs[fold, epoch, s_idx, head_idx, :] = probs[i]

    def set_split(
        self,
        fold: int,
        sample_ids: Sequence[str],
        label: str,
    ) -> None:
        """Label a group of samples as 'train', 'val', or 'holdout' for a fold."""
        for sid in sample_ids:
            s_idx = self._sid_index.get(str(sid))
            if s_idx is not None:
                self.split[fold, s_idx] = label

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        sample_id: str,
        head_name: str,
        fold: int | None = None,
    ) -> np.ndarray:
        """Return epoch-level predictions for one sample + head.

        Returns:
            fold=None → (n_folds, n_epochs, n_classes)
            fold=int  → (n_epochs, n_classes)
        """
        s_idx    = self._sid_index[str(sample_id)]
        head_idx = self._head_index[str(head_name)]
        if fold is None:
            return self.probs[:, :, s_idx, head_idx, :]
        return self.probs[fold, :, s_idx, head_idx, :]

    def get_split(self, sample_id: str, fold: int) -> str:
        """Return the split label ('train'/'val'/'holdout') for a sample in a fold."""
        s_idx = self._sid_index[str(sample_id)]
        return str(self.split[fold, s_idx])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            probs=self.probs,
            split=self.split,
            sample_ids=self.sample_ids,
            y_true=self.y_true,
            head_names=self.head_names,
        )

    @classmethod
    def load(cls, path: str | Path) -> "PredictionStore":
        data   = np.load(path, allow_pickle=True)
        probs  = data["probs"]
        n_folds, n_epochs, _, _, n_classes = probs.shape
        store  = cls(
            sample_ids=data["sample_ids"].tolist(),
            y_true=data["y_true"],
            head_names=data["head_names"].tolist(),
            n_folds=n_folds,
            n_epochs=n_epochs,
            n_classes=n_classes,
        )
        store.probs = probs
        store.split = data["split"]
        return store
