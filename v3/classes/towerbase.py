"""towerbase — unified TowerBase ABC, backbone factory, and modular training utilities.

This module is the structural backbone of the HyperTower v3 architecture.
It provides the abstract tower interface plus the training/eval helpers that
operate on any list of TowerBase instances.

Design principles
-----------------
* No concrete tower classes are defined here (ImageEncoder, ClinicalEncoder, etc.
  live in their respective tower files).
* No imports from any tower file — this module is self-contained with respect to
  the tower layer.  Tower files import from here; this file does not import from them.
* Removing any tower file leaves this module fully intact.
* Duck-typing via optional TowerBase methods (e.g. cd_warmup_embedding) replaces
  isinstance checks so new tower types never require changes here.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from random import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from v3.classes.backbones import BACKBONES, list_names, load_backbone_weights
from v3.classes.bridges import Bridge
from random import random


# ---------------------------------------------------------------------------
# Cross-tower communication context
# ---------------------------------------------------------------------------


@dataclass
class EarlyPassContext:
    eye_train: list[dict]
    bilat_train: list[dict]
    bilat_val: list[dict]
    bilat_test: list[dict]
    image_preprocessor: object
    image_cache: object
    device: torch.device
    store: dict = field(default_factory=dict)  # cross-tower key-value store


# ---------------------------------------------------------------------------
# Backbone factory
# ---------------------------------------------------------------------------


def build_backbone(name: str, freeze_ratio: float = 0.0, augment: bool = True):
    """
    Operational builder:
      - instantiate with DEFAULT weights
      - strip classifier → features
      - apply ratio-based freezing over coarse blocks
      - return (model, out_dim, transform)
    """
    key = (name or "").lower()
    if key not in BACKBONES:
        raise ValueError(
            f"Unsupported backbone '{name}'. Valid options: {list_names()}"
        )

    spec = BACKBONES[key]
    m = spec.ctor(weights=spec.weights_default)
    out_dim, m = spec.strip(m)
    load_backbone_weights(key, m)

    mean = getattr(spec.weights_default, "meta", {}).get("mean", (0.485, 0.456, 0.406))
    std = getattr(spec.weights_default, "meta", {}).get("std", (0.229, 0.224, 0.225))
    crop = 299 if key == "inception_v3" else 224

    if augment:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(crop),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(crop),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    fr = max(0.0, min(1.0, float(freeze_ratio)))
    blocks = spec.blocks(m)
    n = len(blocks)
    freeze_n = int(math.floor(n * fr))
    for b in blocks[:freeze_n]:
        for p in b.parameters():
            p.requires_grad = False

    return m, out_dim, transform


# ---------------------------------------------------------------------------
# TowerBase ABC
# ---------------------------------------------------------------------------


class TowerBase(ABC):
    """Abstract base class for a HyperTower tower.

    Concrete sub-classes must implement ``embed_dims``, ``embed_batch``, and
    ``prepare_fold``.  Everything else has a sensible default no-op.
    """

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def embed_dims(self) -> list[int]:
        """Ordered list of embedding dimensionalities contributed to the bridge.

        Most towers contribute one embedding (e.g. GeometryTower → [geom_hidden]).
        ImageClinicalTower contributes two (image + clinical → [img_dim, cd_dim]).
        """

    @abstractmethod
    def embed_batch(
        self,
        batch: dict,
        *,
        device: torch.device,
        slot: int = 1,
    ) -> list[torch.Tensor]:
        """Return a list of embeddings for one eye slot in *batch*.

        Parameters
        ----------
        batch  : dict  — batch produced by a SlotDataset loader
        device : torch.device
        slot   : 1 (OD / image_1 / matrix_1) or 2 (OS / image_2 / matrix_2)

        Returns
        -------
        list of Tensor — same length and order as ``embed_dims``
        """

    @abstractmethod
    def prepare_fold(
        self,
        *,
        eye_train: list,
        bilat_train: list,
        bilat_val: list,
        bilat_test: list,
        image_preprocessor,
        image_cache,
        device: torch.device,
        args,
    ) -> None:
        """Called once per fold before the main epoch loop."""

    def early_pass(self, context: EarlyPassContext) -> None:
        """Optional: called once per fold before loaders are built."""
        pass

    def get_sample(self, entry) -> "torch.Tensor | dict":  # noqa: ARG002
        """Optional: called by HTDataset to retrieve one sample for this tower.

        entry : ShellEntry — carries entity_id, label, side, paired.

        Single mode (entry.paired=False): return one Tensor.
        Paired mode (entry.paired=True):  return {"a": Tensor, "b": Tensor}.

        Default raises NotImplementedError.  Towers that participate in the
        v4 HTDataset pipeline must implement this.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.get_sample() is not implemented. "
            "Implement it to use this tower with HTDataset."
        )

    # ------------------------------------------------------------------
    # Optional interface
    # ------------------------------------------------------------------

    def augment_samples(self, samples: list) -> list:
        """Optional: add modality-specific keys to sample dicts before loaders are built.

        Called by the orchestrator on each sample list (eye_train, bilat_train,
        bilat_val, bilat_test) *after* ``prepare_fold`` completes.

        The default implementation is a no-op.  GeometryTower overrides this to
        inject ``seg_map_1`` / ``seg_map_2`` numpy arrays so the loader can deliver
        them as tensors alongside the image and clinical slots.
        """
        return samples

    def cd_warmup_embedding(
        self,
        batch: dict,
        *,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Return the clinical embedding for cd_warmup phase, or None if not applicable.

        ClinicalDataTower overrides this to return its encoder output.
        All other towers return None (the default).

        This replaces isinstance(t, ClinicalDataTower) checks in train_towers_epoch,
        so new tower types never require changes to towerbase.py.
        """
        return None

    def set_phase(self, phase: str) -> None:
        """Control requires_grad on this tower's parameters for *phase*.

        Phases: ``cd_warmup``, ``tower_warmup``, ``fused_warmup``, ``main``.
        Default: no-op (tower parameters always trainable unless overridden).
        """

    @property
    def total_epochs(self) -> int:
        """How many epochs this tower participates in the main loop."""
        return 0

    @property
    def train_loader(self) -> Optional[DataLoader]:
        """Single-eye training loader, or None if not applicable."""
        return None

    @property
    def cd_only_loader(self) -> Optional[DataLoader]:
        """Clinical-data-only loader for cd_warmup phase, or None."""
        return None

    def finalize_fold(
        self,
        *,
        bridge,
        bilat_train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        device: torch.device,
        args,
    ) -> None:
        """Optional post-epoch-loop operations (e.g. fused head training)."""


# ---------------------------------------------------------------------------
# Shared training utilities
# ---------------------------------------------------------------------------


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for p in module.parameters():
        p.requires_grad = enabled


def _to_label_tensor(labels, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(labels):
        return labels.to(device=device, dtype=torch.long)
    return torch.as_tensor(labels, dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Modular tower training / evaluation (multi-tower interface)
# ---------------------------------------------------------------------------


def train_towers_epoch(
    towers: "list[TowerBase]",
    bridge: Bridge,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    *,
    phase: str,
    bcd_prob: float = 0.5,
    tower_loss_mode: str = "bcd",
) -> tuple[float, float]:
    """Train one epoch using the modular tower interface.

    All towers and the bridge are set to the given phase via their
    ``set_phase`` methods.  BCD / fused loss semantics mirror the
    existing ``train_single_epoch`` logic:

      cd_warmup    — clinical encoder aux head only (slot index 1)
      tower_warmup — img + cd aux heads equally
      fused_warmup — fused bridge output only
      main         — BCD (randomly img or cd aux) vs fused, per tower_loss_mode

    Duck typing: towers that implement ``cd_warmup_embedding`` participate in
    cd_warmup; all others are skipped for that phase.  No isinstance checks.
    """
    for t in towers:
        t.set_phase(phase)
    bridge.set_phase(phase)

    total_loss = total_correct = total_n = 0

    for batch in loader:
        y = batch.get("label_1")
        if y is None:
            continue

        # ---- cd_warmup: train clinical encoder via its aux head ----
        if phase == "cd_warmup":
            z_cd = None
            for t in towers:
                z = t.cd_warmup_embedding(batch, device=device)
                if z is not None:
                    z_cd = z
                    break
            if z_cd is None:
                continue
            logits = bridge.aux_heads[1](z_cd)
            y_t = _to_label_tensor(y, device)
            loss = F.cross_entropy(logits, y_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bs = y_t.shape[0]
            total_loss += float(loss.item()) * bs
            total_correct += int((logits.argmax(1) == y_t).sum())
            total_n += bs
            continue

        # ---- collect all slot-1 embeddings from every tower ----
        all_embs = []
        for t in towers:
            all_embs.extend(t.embed_batch(batch, device=device, slot=1))

        if any(e is None for e in all_embs):
            continue

        y_t = _to_label_tensor(y, device)

        if phase == "tower_warmup":
            # Average loss across all available aux heads
            aux_logits = [bridge.aux_heads[i](emb) for i, emb in enumerate(all_embs)]
            loss = sum(F.cross_entropy(l, y_t) for l in aux_logits) / len(aux_logits)
            # Softmax average for metrics
            logits = sum(F.softmax(l, dim=1) for l in aux_logits) / len(aux_logits)

        elif phase == "fused_warmup":
            logits_fused, _ = bridge.fuse(all_embs)
            loss = F.cross_entropy(logits_fused, y_t)
            logits = logits_fused

        else:  # main
            if tower_loss_mode == "all":
                logits_fused, aux = bridge.fuse(all_embs)
                loss = F.cross_entropy(logits_fused, y_t)
                for aux_l in aux:
                    loss = loss + F.cross_entropy(aux_l, y_t)
                logits = logits_fused
            elif random() < bcd_prob:
                # Randomly pick ONE tower to train (Generalized BCD)
                idx = int(random() * len(all_embs))
                logits = bridge.aux_heads[idx](all_embs[idx])
                loss = F.cross_entropy(logits, y_t)
            else:
                logits_fused, _ = bridge.fuse(all_embs)
                loss = F.cross_entropy(logits_fused, y_t)
                logits = logits_fused

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bs = y_t.shape[0]
        total_loss += float(loss.item()) * bs
        total_correct += int((logits.argmax(1) == y_t).sum())
        total_n += bs

    return (
        total_loss / total_n if total_n else float("nan"),
        total_correct / total_n if total_n else float("nan"),
    )


def collect_probs_towers(
    towers: "list[TowerBase]",
    bridge: Bridge,
    loader: DataLoader,
    device: torch.device,
    *,
    tower_mode: str = "ensemble",
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate using the modular tower interface.

    Returns ``(y_true, probs)`` with patient-level probabilities
    (OD + OS averaged for ensemble mode, OD-only for single mode).
    """
    for t in towers:
        if isinstance(t, nn.Module):
            t.eval()
    bridge.eval()

    y_chunks, p_chunks = [], []
    with torch.no_grad():
        for batch in loader:
            y = batch.get("label_1")
            if y is None:
                continue
            if not (
                torch.is_tensor(batch.get("image_1"))
                and torch.is_tensor(batch.get("image_2"))
            ):
                continue

            all_embs_od = []
            all_embs_os = []
            for t in towers:
                all_embs_od.extend(t.embed_batch(batch, device=device, slot=1))
                all_embs_os.extend(t.embed_batch(batch, device=device, slot=2))

            if any(e is None for e in all_embs_od + all_embs_os):
                continue

            logits_od, _ = bridge.fuse(all_embs_od)
            logits_os, _ = bridge.fuse(all_embs_os)

            if tower_mode == "ensemble":
                probs = 0.5 * (
                    F.softmax(logits_od, dim=1) + F.softmax(logits_os, dim=1)
                )
            else:
                probs = F.softmax(logits_od, dim=1)

            y_chunks.append(_to_label_tensor(y, device).cpu().numpy())
            p_chunks.append(probs.cpu().numpy())

    for t in towers:
        if isinstance(t, nn.Module):
            t.train()
    bridge.train()

    if not y_chunks:
        return np.array([], dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(y_chunks), np.concatenate(p_chunks, axis=0)
