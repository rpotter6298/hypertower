"""hypertower_models — HyperTower vehicle classes and their training/eval helpers.

A "vehicle" wires one or more tower encoders together with a Bridge to form
a complete trainable model.  Vehicles can in principle be run standalone;
the v3_hypertower orchestrator drives them through the full fold/epoch loop.

Contents
--------
SingleEyeHT              — ImageEncoder + ClinicalEncoder + Bridge (eye-level)
BilateralHT              — shared eye towers + joint fusion layers + Bridge
SiameseHT                — SiameseImageTower + Bridge (image-only bilateral)
FusedEnsembleHT          — SingleEyeHT base + per-eye attention scorer
LogitMLPEnsembleHT       — MLP head over concatenated per-eye logits
EmbeddingMLPEnsembleHT   — MLP head over concatenated per-eye z_fused embeddings
NTowerHT                 — N named encoders + Bridge (general, key-mapped)
NLateralHT               — N same-type inputs through shared encoder + joint MLP
MonoTowerHT              — single tower + direct classifier (no bridge)

Training helpers  : train_single_epoch, train_bilateral_epoch,
                    train_siamese_epoch, train_fusion_epoch,
                    train_ntower_epoch, train_mono_epoch
Inference helpers : collect_probs_classic, collect_probs_ensemble,
                    collect_probs_ensemble_pereye, collect_probs_bilateral,
                    collect_probs_bilateral_components, collect_probs_siamese,
                    collect_probs_fused, collect_probs_single_components,
                    collect_probs_eye_level, collect_probs_ntower,
                    collect_probs_mono
"""
from __future__ import annotations

from random import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from v3.classes.towerbase import _set_requires_grad, _to_label_tensor
from v3.classes.image_towers import ImageEncoder, SiameseImageTower
from v3.classes.clinical_towers import ClinicalEncoder
from v3.classes.bridges import Bridge, HTClassifier, HyperBridge


# ---------------------------------------------------------------------------
# Vehicle classes
# ---------------------------------------------------------------------------

class SingleEyeHT(nn.Module):
    """ImageEncoder + ClinicalEncoder + Bridge, trained on eye-level samples."""

    def __init__(
        self,
        *,
        backbone: str,
        freeze_ratio: float,
        augment: bool,
        clinical_data,
        num_classes: int,
        cd_hidden_dim: int = 128,
        fusion_dim: int = 256,
        bridge_mode: str = "fused",
        bridge_dropout: float = 0.5,
        cd_dropout: float = 0.1,
        se_img_tower: bool = False,
        se_cd_tower: bool = False,
        se_bridge: bool = False,
    ):
        super().__init__()
        self.img_tower = ImageEncoder(
            backbone=backbone, freeze_ratio=freeze_ratio,
            augment=augment, use_se=se_img_tower,
        )
        self.cd_tower = ClinicalEncoder(
            clinical_data=clinical_data, hidden_dim=cd_hidden_dim,
            dropout=cd_dropout, use_se=se_cd_tower,
        )
        self.bridge = Bridge(
            tower_dims=[self.img_tower.out_dim, self.cd_tower.out_dim],
            num_classes=num_classes, fusion_dim=fusion_dim,
            mode=bridge_mode, dropout=bridge_dropout, use_se=se_bridge,
        )

    @property
    def transform(self):
        return self.img_tower.transform

    def encode(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        """Return z_fused (fusion_dim) without the classifier head."""
        return self.bridge.encode([self.img_tower(x), self.cd_tower(meta)])

    def forward(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        img_feats = None if self.bridge.mode == "clinical_only" else self.img_tower(x)
        md_feats  = None if self.bridge.mode == "image_only"    else self.cd_tower(meta)
        out_f, _ = self.bridge.fuse([img_feats, md_feats])
        return out_f


class NTowerHT(nn.Module):
    """General N-tower vehicle: any named encoder modules fused through a Bridge.

    Parameters
    ----------
    towers      : ordered dict of {name: encoder_module}.  Each module must
                  expose ``.out_dim``.  Bridge slot order follows dict order.
    num_classes : number of output classes
    fusion_dim  : projection dimensionality inside the bridge
    dropout     : dropout before the fused output head
    use_se      : SE gate on the fused vector

    Forward contract
    ----------------
    ``forward(embeddings)`` takes a ``dict[str, Tensor]`` of pre-computed
    per-tower embeddings (keyed by tower name) and returns
    ``(logits_fused, aux_dict)`` where ``aux_dict`` maps each tower name to
    its auxiliary head logits.

    The vehicle does not know how to extract embeddings from raw data —
    that is the driver's responsibility.  Use ``embed_batch`` (TowerBase API)
    or custom extraction logic in the training loop, then pass the result here.

    Type-aware helpers
    ------------------
    ``transform`` — returns the ``.transform`` of the first tower that has one
                    (typically the image tower), for use by data loaders.
    """

    def __init__(
        self,
        towers: dict[str, nn.Module],
        num_classes: int,
        fusion_dim: int = 256,
        dropout: float = 0.5,
        use_se: bool = False,
    ):
        super().__init__()
        self.towers = nn.ModuleDict(towers)
        self.bridge = Bridge(
            tower_dims=[t.out_dim for t in self.towers.values()],
            num_classes=num_classes,
            fusion_dim=fusion_dim,
            dropout=dropout,
            use_se=use_se,
        )

    @property
    def transform(self):
        """Image transform from the first tower that exposes one, or None."""
        for t in self.towers.values():
            if hasattr(t, "transform"):
                return t.transform
        return None

    def encode(self, embeddings: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return z_fused (pre-classifier) from a dict of per-tower embeddings."""
        return self.bridge.encode([embeddings[name] for name in self.towers])

    def forward(
        self,
        embeddings: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Fuse pre-computed embeddings through the bridge.

        Returns
        -------
        logits_fused : Tensor [B, num_classes]
        aux_logits   : dict mapping tower name → Tensor [B, num_classes]
        """
        ordered = [embeddings[name] for name in self.towers]
        logits_fused, aux = self.bridge.fuse(ordered)
        return logits_fused, {name: aux[i] for i, name in enumerate(self.towers)}


class NLateralHT(nn.Module):
    """N same-type inputs through a shared encoder, jointly compressed, then classified.

    Intended for encoding multiple instances of the same modality together —
    e.g. left + right fundus images, or OD + OS clinical vectors.
    All inputs share the same encoder weights (one forward pass per input).

    Output matches NTowerHT's contract: ``(logits, {name: aux_logits})``.
    Aux logits are per-input classifications taken before the joint MLP,
    useful for BCD-style training.

    Parameters
    ----------
    encoder     : shared encoder module with ``.out_dim``
    input_names : ordered names for each input slot (e.g. ``["od", "os"]``)
    num_classes : output classes
    fusion_dim  : hidden dim of the joint MLP
    dropout     : dropout in joint MLP and classifier head
    """

    def __init__(
        self,
        encoder: nn.Module,
        input_names: list[str],
        num_classes: int,
        fusion_dim: int = 256,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.encoder     = encoder
        self.input_names = list(input_names)
        n                = len(input_names)
        in_dim: int      = encoder.out_dim  # type: ignore[assignment]

        # Joint MLP: [z0 ‖ z1 ‖ ... ‖ z_{n-1}] → fusion_dim → in_dim
        self.joint = nn.Sequential(
            nn.Linear(n * in_dim, fusion_dim), nn.LayerNorm(fusion_dim),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(fusion_dim, in_dim),
        )
        # Per-input aux heads — classify each input before joining
        self.aux_heads = nn.ModuleList([
            nn.Linear(in_dim, num_classes) for _ in range(n)
        ])
        # Main classifier on the joint representation
        self.head = nn.Sequential(
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(in_dim, num_classes),
        )

    @property
    def transform(self):
        return getattr(self.encoder, "transform", None)

    def encode(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return joint embedding (post-MLP, pre-classifier)."""
        zs = [self.encoder(inputs[name]) for name in self.input_names]
        return self.joint(torch.cat(zs, dim=1))

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Encode inputs jointly and classify.

        Parameters
        ----------
        inputs : ``{name: raw_tensor}`` for each input slot

        Returns
        -------
        logits   : [B, num_classes]
        aux_dict : ``{name: [B, num_classes]}`` — per-input pre-join logits
        """
        zs = [self.encoder(inputs[name]) for name in self.input_names]
        z_joint = self.joint(torch.cat(zs, dim=1))
        logits   = self.head(z_joint)
        aux      = {name: head(z)
                    for name, head, z in zip(self.input_names, self.aux_heads, zs)}
        return logits, aux


class BilateralHT(nn.Module):
    """Bilateral vehicle: shared eye-level towers + joint fusion layers + Bridge."""

    def __init__(
        self,
        *,
        backbone: str,
        freeze_ratio: float,
        augment: bool,
        clinical_data,
        num_classes: int,
        cd_hidden_dim: int = 128,
        fusion_dim: int = 256,
    ):
        super().__init__()
        self.eye_img_tower = ImageEncoder(
            backbone=backbone, freeze_ratio=freeze_ratio, augment=augment, use_se=False,
        )
        self.eye_cd_tower = ClinicalEncoder(
            clinical_data=clinical_data, hidden_dim=cd_hidden_dim, use_se=False,
        )
        img_dim = self.eye_img_tower.out_dim
        md_dim  = self.eye_cd_tower.out_dim
        self.joint_img = nn.Sequential(
            nn.Linear(2 * img_dim, fusion_dim), nn.LayerNorm(fusion_dim),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(fusion_dim, img_dim),
        )
        self.joint_md = nn.Sequential(
            nn.Linear(2 * md_dim, fusion_dim), nn.LayerNorm(fusion_dim),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(fusion_dim, md_dim),
        )
        self.bridge  = Bridge(tower_dims=[img_dim, md_dim], num_classes=num_classes,
                              fusion_dim=fusion_dim, mode="fused", use_se=False)
        self.aux_img = nn.Linear(img_dim, num_classes)
        self.aux_md  = nn.Linear(md_dim,  num_classes)

    @property
    def transform(self):
        return self.eye_img_tower.transform

    def encode_joint(
        self,
        x_od: torch.Tensor, meta_od: torch.Tensor,
        x_os: torch.Tensor, meta_os: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img_od = self.eye_img_tower(x_od);  md_od = self.eye_cd_tower(meta_od)
        img_os = self.eye_img_tower(x_os);  md_os = self.eye_cd_tower(meta_os)
        return (self.joint_img(torch.cat([img_od, img_os], dim=1)),
                self.joint_md(torch.cat([md_od,  md_os],  dim=1)))

    def forward(
        self,
        x_od: torch.Tensor, meta_od: torch.Tensor,
        x_os: torch.Tensor, meta_os: torch.Tensor,
    ) -> torch.Tensor:
        joint_img, joint_md = self.encode_joint(x_od, meta_od, x_os, meta_os)
        out_f, _ = self.bridge.fuse([joint_img, joint_md])
        return out_f


class SiameseHT(nn.Module):
    """Bilateral vehicle using a shared-weight SiameseImageTower (mean+delta)."""

    def __init__(
        self,
        *,
        backbone: str,
        freeze_ratio: float,
        augment: bool,
        num_classes: int,
        fusion_dim: int = 256,
    ):
        super().__init__()
        self.img_tower = SiameseImageTower(
            backbone=backbone, freeze_ratio=freeze_ratio, augment=augment, use_se=False,
        )
        img_dim    = self.img_tower.out_dim
        self.bridge    = Bridge(tower_dims=[img_dim], num_classes=num_classes,
                                fusion_dim=fusion_dim, use_se=False)
        self.aux_img   = nn.Linear(img_dim, num_classes)

    @property
    def transform(self):
        return self.img_tower.transform

    def encode(self, x_od: torch.Tensor, x_os: torch.Tensor) -> torch.Tensor:
        return self.img_tower(x_od, x_os)

    def forward(self, x_od: torch.Tensor, x_os: torch.Tensor) -> torch.Tensor:
        out_f, _ = self.bridge.fuse([self.encode(x_od, x_os)])
        return out_f


class FusedEnsembleHT(nn.Module):
    """SingleEyeHT base with a per-eye attention scorer for bilateral fusion."""

    def __init__(self, base: SingleEyeHT, num_classes: int):
        super().__init__()
        self.base       = base
        self.eye_scorer = nn.Linear(num_classes, 1, bias=True)

    @property
    def head(self) -> nn.Module:
        return self.eye_scorer

    def forward(
        self,
        x_od: torch.Tensor, meta_od: torch.Tensor,
        x_os: torch.Tensor, meta_os: torch.Tensor,
    ) -> torch.Tensor:
        logit_od = self.base(x_od, meta_od)
        logit_os = self.base(x_os, meta_os)
        scores   = torch.cat([self.eye_scorer(logit_od), self.eye_scorer(logit_os)], dim=1)
        alpha    = torch.softmax(scores, dim=1)
        return alpha[:, 0:1] * logit_od + alpha[:, 1:2] * logit_os


class LogitMLPEnsembleHT(nn.Module):
    """MLP head trained on concatenated per-eye logits."""

    def __init__(self, base: SingleEyeHT, num_classes: int, hidden: int = 64):
        super().__init__()
        self.base = base
        self.head = nn.Sequential(
            nn.Linear(2 * num_classes, hidden), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(hidden, num_classes),
        )

    def forward(
        self,
        x_od: torch.Tensor, meta_od: torch.Tensor,
        x_os: torch.Tensor, meta_os: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(torch.cat([self.base(x_od, meta_od), self.base(x_os, meta_os)], dim=1))


class EmbeddingMLPEnsembleHT(nn.Module):
    """MLP head trained on concatenated per-eye z_fused embeddings."""

    def __init__(self, base: SingleEyeHT, num_classes: int, hidden: int = 256):
        super().__init__()
        self.base = base
        fusion_dim = base.bridge.W[0].out_features
        self.head = nn.Sequential(
            nn.Linear(2 * fusion_dim, hidden), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(hidden, num_classes),
        )

    def forward(
        self,
        x_od: torch.Tensor, meta_od: torch.Tensor,
        x_os: torch.Tensor, meta_os: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(torch.cat([self.base.encode(x_od, meta_od),
                                    self.base.encode(x_os, meta_os)], dim=1))


# ---------------------------------------------------------------------------
# MonoTowerHT — single tower + Bridge(N=1)
# ---------------------------------------------------------------------------

class MonoTowerHT(nn.Module):
    """Single-tower vehicle: tower output fed directly into a classifier head.

    No bridge projection — the tower's embedding goes straight to
    ``ReLU → Dropout → Linear(out_dim → num_classes)``.  This is the
    minimal architecture: just the tower's learned representation with a
    classification head attached.

    Contrast with ``NTowerHT(N=1)``, which still projects through the bridge's
    shared ``fusion_dim`` space.  These are distinct architectures and may
    yield different results.

    Parameters
    ----------
    tower       : any nn.Module with ``.out_dim``  (ImageEncoder, ClinicalEncoder, …)
    num_classes : number of output classes
    dropout     : dropout before the output linear layer
    """

    def __init__(
        self,
        tower: nn.Module,
        num_classes: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.tower = tower
        out_dim: int = tower.out_dim  # type: ignore[assignment]
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes),
        )

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        """Classify a pre-computed embedding."""
        return self.head(z)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Pass inputs through tower then classifier head."""
        return self.head(self.tower(*args, **kwargs))


# ---------------------------------------------------------------------------
# Phase control
# ---------------------------------------------------------------------------

def _set_single_phase(model: SingleEyeHT, phase: str) -> None:
    bridge_mode = model.bridge.mode
    if bridge_mode in ("image_only", "clinical_only") and phase == "fused_warmup":
        phase = "tower_warmup"
    if phase == "cd_warmup":
        _set_requires_grad(model.img_tower, False)
        _set_requires_grad(model.cd_tower, True)
        _set_requires_grad(model.bridge.aux_heads[0], False)
        _set_requires_grad(model.bridge.aux_heads[1], True)
        _set_requires_grad(model.bridge.W[0], False)
        _set_requires_grad(model.bridge.W[1], False)
        _set_requires_grad(model.bridge.classifier_fused, False)
        return
    if phase == "tower_warmup":
        _set_requires_grad(model.img_tower, bridge_mode != "clinical_only")
        _set_requires_grad(model.cd_tower,  bridge_mode != "image_only")
        _set_requires_grad(model.bridge.aux_heads[0], bridge_mode != "clinical_only")
        _set_requires_grad(model.bridge.aux_heads[1], bridge_mode != "image_only")
        _set_requires_grad(model.bridge.W[0], False)
        _set_requires_grad(model.bridge.W[1], False)
        _set_requires_grad(model.bridge.classifier_fused, False)
        return
    if phase == "fused_warmup":
        _set_requires_grad(model.img_tower, False)
        _set_requires_grad(model.cd_tower,  False)
        _set_requires_grad(model.bridge.aux_heads[0], False)
        _set_requires_grad(model.bridge.aux_heads[1], False)
        _set_requires_grad(model.bridge.W[0], True)
        _set_requires_grad(model.bridge.W[1], True)
        _set_requires_grad(model.bridge.classifier_fused, True)
        return
    _set_requires_grad(model, True)


def _set_bilateral_phase(model: BilateralHT, phase: str) -> None:
    if phase == "tower_warmup":
        for m in (model.eye_img_tower, model.eye_cd_tower,
                  model.joint_img, model.joint_md, model.aux_img, model.aux_md):
            _set_requires_grad(m, True)
        _set_requires_grad(model.bridge, False)
        return
    if phase == "fused_warmup":
        for m in (model.eye_img_tower, model.eye_cd_tower,
                  model.joint_img, model.joint_md, model.aux_img, model.aux_md):
            _set_requires_grad(m, False)
        _set_requires_grad(model.bridge, True)
        return
    _set_requires_grad(model, True)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_single_epoch(
    model: SingleEyeHT,
    loader: DataLoader,
    opt,
    device: torch.device,
    *,
    phase: str,
    bcd_prob: float = 0.5,
    tower_loss_mode: str = "bcd",
) -> tuple[float, float]:
    model.train()
    _set_single_phase(model, phase)
    total_loss = total_correct = total_n = 0
    for batch in loader:
        x = batch.get("image_1");  m = batch.get("matrix_1");  y = batch.get("label_1")
        if phase == "cd_warmup":
            if not torch.is_tensor(m):
                continue
            y   = _to_label_tensor(y, device)
            logits = model.bridge.aux_heads[1](model.cd_tower(m.to(device)))
            loss   = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            bs = y.shape[0]
            total_loss += float(loss.item()) * bs
            total_correct += int((logits.argmax(1) == y).sum())
            total_n += bs
            continue
        if not (torch.is_tensor(x) and torch.is_tensor(m)):
            continue
        x = x.to(device); m = m.to(device); y = _to_label_tensor(y, device)
        bridge_mode = model.bridge.mode
        img_feats = None if bridge_mode == "clinical_only" else model.img_tower(x)
        md_feats  = None if bridge_mode == "image_only"    else model.cd_tower(m)

        if phase == "tower_warmup":
            if bridge_mode == "clinical_only":
                logits = model.bridge.aux_heads[1](md_feats);   loss = F.cross_entropy(logits, y)
            elif bridge_mode == "image_only":
                logits = model.bridge.aux_heads[0](img_feats);  loss = F.cross_entropy(logits, y)
            else:
                li = model.bridge.aux_heads[0](img_feats); lm = model.bridge.aux_heads[1](md_feats)
                loss = 0.5 * (F.cross_entropy(li, y) + F.cross_entropy(lm, y))
                logits = 0.5 * (F.softmax(li, dim=1) + F.softmax(lm, dim=1))
        elif phase == "fused_warmup":
            logits, _ = model.bridge.fuse([img_feats, md_feats]); loss = F.cross_entropy(logits, y)
        else:
            if bridge_mode == "clinical_only":
                logits = model.bridge.aux_heads[1](md_feats);   loss = F.cross_entropy(logits, y)
            elif bridge_mode == "image_only":
                logits = model.bridge.aux_heads[0](img_feats);  loss = F.cross_entropy(logits, y)
            elif tower_loss_mode == "all":
                li = model.bridge.aux_heads[0](img_feats); lm = model.bridge.aux_heads[1](md_feats)
                logits, _ = model.bridge.fuse([img_feats, md_feats])
                loss = F.cross_entropy(logits, y) + F.cross_entropy(li, y) + F.cross_entropy(lm, y)
            elif random() < bcd_prob:
                logits = (model.bridge.aux_heads[0](img_feats) if random() < 0.5
                          else model.bridge.aux_heads[1](md_feats))
                loss = F.cross_entropy(logits, y)
            else:
                logits, _ = model.bridge.fuse([img_feats, md_feats]); loss = F.cross_entropy(logits, y)

        opt.zero_grad(); loss.backward(); opt.step()
        bs = y.shape[0]
        total_loss    += float(loss.item()) * bs
        total_correct += int((logits.argmax(1) == y).sum())
        total_n       += bs
    return (total_loss / total_n if total_n else float("nan"),
            total_correct / total_n if total_n else float("nan"))


def train_bilateral_epoch(
    model: BilateralHT,
    loader: DataLoader,
    opt,
    device: torch.device,
    *,
    phase: str,
    bcd_prob: float = 0.5,
    tower_loss_mode: str = "bcd",
) -> tuple[float, float]:
    model.train()
    _set_bilateral_phase(model, phase)
    total_loss = total_correct = total_n = 0
    for batch in loader:
        x1 = batch.get("image_1"); m1 = batch.get("matrix_1")
        x2 = batch.get("image_2"); m2 = batch.get("matrix_2"); y = batch.get("label_1")
        if not (torch.is_tensor(x1) and torch.is_tensor(m1) and
                torch.is_tensor(x2) and torch.is_tensor(m2)):
            continue
        x1 = x1.to(device); m1 = m1.to(device)
        x2 = x2.to(device); m2 = m2.to(device); y = _to_label_tensor(y, device)
        joint_img, joint_md = model.encode_joint(x1, m1, x2, m2)

        if phase == "tower_warmup":
            li = model.aux_img(joint_img); lm = model.aux_md(joint_md)
            loss = 0.5 * (F.cross_entropy(li, y) + F.cross_entropy(lm, y))
            logits = 0.5 * (F.softmax(li, dim=1) + F.softmax(lm, dim=1))
        elif phase == "fused_warmup":
            logits, _ = model.bridge.fuse([joint_img, joint_md]); loss = F.cross_entropy(logits, y)
        else:
            if tower_loss_mode == "all":
                li = model.aux_img(joint_img); lm = model.aux_md(joint_md)
                logits, _ = model.bridge.fuse([joint_img, joint_md])
                loss = F.cross_entropy(logits, y) + F.cross_entropy(li, y) + F.cross_entropy(lm, y)
            elif random() < bcd_prob:
                logits = model.aux_img(joint_img) if random() < 0.5 else model.aux_md(joint_md)
                loss = F.cross_entropy(logits, y)
            else:
                logits, _ = model.bridge.fuse([joint_img, joint_md]); loss = F.cross_entropy(logits, y)

        opt.zero_grad(); loss.backward(); opt.step()
        bs = y.shape[0]
        total_loss    += float(loss.item()) * bs
        total_correct += int((logits.argmax(1) == y).sum())
        total_n       += bs
    return (total_loss / total_n if total_n else float("nan"),
            total_correct / total_n if total_n else float("nan"))


def train_siamese_epoch(
    model: SiameseHT,
    loader: DataLoader,
    opt,
    device: torch.device,
    *,
    bcd_prob: float = 0.5,
    tower_loss_mode: str = "bcd",
) -> tuple[float, float]:
    model.train()
    total_loss = total_correct = total_n = 0
    for batch in loader:
        x1 = batch.get("image_1"); x2 = batch.get("image_2"); y = batch.get("label_1")
        if not (torch.is_tensor(x1) and torch.is_tensor(x2)):
            continue
        x1 = x1.to(device); x2 = x2.to(device); y = _to_label_tensor(y, device)
        feats = model.encode(x1, x2)

        if tower_loss_mode == "all":
            out_f, _ = model.bridge.fuse([feats, None])
            loss = F.cross_entropy(out_f, y) + F.cross_entropy(model.aux_img(feats), y)
            logits = out_f
        elif random() < bcd_prob:
            logits = model.aux_img(feats); loss = F.cross_entropy(logits, y)
        else:
            logits, _ = model.bridge.fuse([feats, None]); loss = F.cross_entropy(logits, y)

        opt.zero_grad(); loss.backward(); opt.step()
        bs = y.shape[0]
        total_loss    += float(loss.item()) * bs
        total_correct += int((logits.argmax(1) == y).sum())
        total_n       += bs
    return (total_loss / total_n if total_n else float("nan"),
            total_correct / total_n if total_n else float("nan"))


def train_fusion_epoch(
    model,  # FusedEnsembleHT | LogitMLPEnsembleHT | EmbeddingMLPEnsembleHT
    loader: DataLoader,
    opt,
    device: torch.device,
) -> tuple[float, float]:
    """Train only the fusion head; base SingleEyeHT is frozen in eval mode."""
    model.base.eval(); model.head.train()
    total_loss = total_correct = total_n = 0
    for batch in loader:
        x1 = batch.get("image_1"); m1 = batch.get("matrix_1")
        x2 = batch.get("image_2"); m2 = batch.get("matrix_2"); y = batch.get("label_1")
        if not (torch.is_tensor(x1) and torch.is_tensor(m1) and
                torch.is_tensor(x2) and torch.is_tensor(m2)):
            continue
        y_t  = _to_label_tensor(y, device)
        out  = model(x1.to(device), m1.to(device), x2.to(device), m2.to(device))
        loss = F.cross_entropy(out, y_t)
        opt.zero_grad(); loss.backward(); opt.step()
        bs = y_t.shape[0]
        total_loss    += float(loss.item()) * bs
        total_correct += int((out.argmax(1) == y_t).sum())
        total_n       += bs
    return (total_loss / total_n if total_n else float("nan"),
            total_correct / total_n if total_n else float("nan"))


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def collect_probs_classic(
    model: SingleEyeHT, loader: DataLoader, device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_c, p_c = [], []
    with torch.no_grad():
        for batch in loader:
            x1 = batch.get("image_1"); m1 = batch.get("matrix_1")
            x2 = batch.get("image_2"); m2 = batch.get("matrix_2"); y = batch.get("label_1")
            if not (torch.is_tensor(x1) and torch.is_tensor(m1) and
                    torch.is_tensor(x2) and torch.is_tensor(m2)):
                continue
            y_t = _to_label_tensor(y, device)
            p_od = F.softmax(model(x1.to(device), m1.to(device)), dim=1)
            p_os = F.softmax(model(x2.to(device), m2.to(device)), dim=1)
            y_np = y_t.cpu().numpy()
            y_c += [y_np, y_np];  p_c += [p_od.cpu().numpy(), p_os.cpu().numpy()]
    if not y_c:
        return np.array([], dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(y_c), np.concatenate(p_c, axis=0)


def collect_probs_ensemble(
    model: SingleEyeHT, loader: DataLoader, device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_c, p_c = [], []
    with torch.no_grad():
        for batch in loader:
            x1 = batch.get("image_1"); m1 = batch.get("matrix_1")
            x2 = batch.get("image_2"); m2 = batch.get("matrix_2"); y = batch.get("label_1")
            if not (torch.is_tensor(x1) and torch.is_tensor(m1) and
                    torch.is_tensor(x2) and torch.is_tensor(m2)):
                continue
            y_t = _to_label_tensor(y, device)
            p_od = F.softmax(model(x1.to(device), m1.to(device)), dim=1)
            p_os = F.softmax(model(x2.to(device), m2.to(device)), dim=1)
            y_c.append(y_t.cpu().numpy());  p_c.append((0.5 * (p_od + p_os)).cpu().numpy())
    if not y_c:
        return np.array([], dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(y_c), np.concatenate(p_c, axis=0)


def collect_probs_ensemble_pereye(
    model: SingleEyeHT, loader: DataLoader, device: torch.device, *, return_ids: bool = False,
):
    model.eval()
    y_c = []; pf_od_c, pi_od_c, pm_od_c = [], [], []; pf_os_c, pi_os_c, pm_os_c = [], [], []
    id_c: list = []
    with torch.no_grad():
        for batch in loader:
            x1 = batch.get("image_1"); m1 = batch.get("matrix_1")
            x2 = batch.get("image_2"); m2 = batch.get("matrix_2"); y = batch.get("label_1")
            if not (torch.is_tensor(x1) and torch.is_tensor(m1) and
                    torch.is_tensor(x2) and torch.is_tensor(m2)):
                continue
            y_t = _to_label_tensor(y, device)

            def _fwd(x, m):
                img_f = None if model.bridge.mode == "clinical_only" else model.img_tower(x.to(device))
                md_f  = None if model.bridge.mode == "image_only"    else model.cd_tower(m.to(device))
                out_f, aux = model.bridge.fuse([img_f, md_f])
                out_i, out_m = aux[0], aux[1]
                pf = F.softmax(out_f, dim=1)
                pi = F.softmax(out_i, dim=1) if out_i is not None else pf
                pm = F.softmax(out_m, dim=1) if out_m is not None else pf
                return pf, pi, pm

            pf_od, pi_od, pm_od = _fwd(x1, m1);  pf_os, pi_os, pm_os = _fwd(x2, m2)
            y_c.append(y_t.cpu().numpy())
            for lst, t in ((pf_od_c, pf_od), (pi_od_c, pi_od), (pm_od_c, pm_od),
                           (pf_os_c, pf_os), (pi_os_c, pi_os), (pm_os_c, pm_os)):
                lst.append(t.cpu().numpy())
            if return_ids:
                ids = batch.get("id_1", [""] * len(y_t))
                id_c.extend([str(i) for i in (ids.tolist() if torch.is_tensor(ids) else ids)])

    if not y_c:
        z = np.zeros((0, 0), dtype=np.float32); ei = np.array([], dtype=np.int64)
        base = (ei, z, z, z, z, z, z)
        return base + (np.array([], dtype=object),) if return_ids else base

    y = np.concatenate(y_c)
    pf_od, pi_od, pm_od = (np.concatenate(c, axis=0) for c in (pf_od_c, pi_od_c, pm_od_c))
    pf_os, pi_os, pm_os = (np.concatenate(c, axis=0) for c in (pf_os_c, pi_os_c, pm_os_c))
    if return_ids:
        return y, pf_od, pi_od, pm_od, pf_os, pi_os, pm_os, np.array(id_c, dtype=object)
    return y, pf_od, pi_od, pm_od, pf_os, pi_os, pm_os


def collect_probs_bilateral(
    model: BilateralHT, loader: DataLoader, device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_c, p_c = [], []
    with torch.no_grad():
        for batch in loader:
            x1 = batch.get("image_1"); m1 = batch.get("matrix_1")
            x2 = batch.get("image_2"); m2 = batch.get("matrix_2"); y = batch.get("label_1")
            if not (torch.is_tensor(x1) and torch.is_tensor(m1) and
                    torch.is_tensor(x2) and torch.is_tensor(m2)):
                continue
            y_t = _to_label_tensor(y, device)
            p   = F.softmax(model(x1.to(device), m1.to(device),
                                  x2.to(device), m2.to(device)), dim=1)
            y_c.append(y_t.cpu().numpy());  p_c.append(p.cpu().numpy())
    if not y_c:
        return np.array([], dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(y_c), np.concatenate(p_c, axis=0)


def collect_probs_bilateral_components(
    model: BilateralHT, loader: DataLoader, device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_c = []; pf_c, pi_c, pm_c = [], [], []
    with torch.no_grad():
        for batch in loader:
            x1 = batch.get("image_1"); m1 = batch.get("matrix_1")
            x2 = batch.get("image_2"); m2 = batch.get("matrix_2"); y = batch.get("label_1")
            if not (torch.is_tensor(x1) and torch.is_tensor(m1) and
                    torch.is_tensor(x2) and torch.is_tensor(m2)):
                continue
            y_t = _to_label_tensor(y, device)
            ji, jm = model.encode_joint(x1.to(device), m1.to(device),
                                        x2.to(device), m2.to(device))
            out_f, _ = model.bridge.fuse([ji, jm])
            y_c.append(y_t.cpu().numpy())
            pf_c.append(F.softmax(out_f, dim=1).cpu().numpy())
            pi_c.append(F.softmax(model.aux_img(ji), dim=1).cpu().numpy())
            pm_c.append(F.softmax(model.aux_md(jm),  dim=1).cpu().numpy())
    if not y_c:
        z = np.zeros((0, 0), dtype=np.float32)
        return np.array([], dtype=np.int64), z, z, z
    return (np.concatenate(y_c), np.concatenate(pf_c, axis=0),
            np.concatenate(pi_c, axis=0), np.concatenate(pm_c, axis=0))


def collect_probs_siamese(
    model: SiameseHT, loader: DataLoader, device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_c, p_c = [], []
    with torch.no_grad():
        for batch in loader:
            x1 = batch.get("image_1"); x2 = batch.get("image_2"); y = batch.get("label_1")
            if not (torch.is_tensor(x1) and torch.is_tensor(x2)):
                continue
            feats = model.encode(x1.to(device), x2.to(device))
            logits, _ = model.bridge.fuse([feats])
            y_c.append(np.array(y) if not torch.is_tensor(y) else y.cpu().numpy())
            p_c.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(y_c, axis=0), np.concatenate(p_c, axis=0)


def collect_probs_fused(
    model: FusedEnsembleHT, loader: DataLoader, device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_c, p_c = [], []
    with torch.no_grad():
        for batch in loader:
            x1 = batch.get("image_1"); m1 = batch.get("matrix_1")
            x2 = batch.get("image_2"); m2 = batch.get("matrix_2"); y = batch.get("label_1")
            if not (torch.is_tensor(x1) and torch.is_tensor(m1) and
                    torch.is_tensor(x2) and torch.is_tensor(m2)):
                continue
            y_t = _to_label_tensor(y, device)
            p   = F.softmax(model(x1.to(device), m1.to(device),
                                  x2.to(device), m2.to(device)), dim=1)
            y_c.append(y_t.cpu().numpy());  p_c.append(p.cpu().numpy())
    if not y_c:
        return np.array([], dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(y_c), np.concatenate(p_c, axis=0)


def collect_probs_single_components(
    model: SingleEyeHT, loader: DataLoader, device: torch.device,
    *, aggregate_patient: bool, return_logits: bool = False,
):
    model.eval()
    y_c = []; pf_c, pi_c, pm_c = [], [], []; lf_c, li_c, lm_c = [], [], []
    with torch.no_grad():
        for batch in loader:
            x1 = batch.get("image_1"); m1 = batch.get("matrix_1")
            x2 = batch.get("image_2"); m2 = batch.get("matrix_2"); y = batch.get("label_1")
            if not (torch.is_tensor(x1) and torch.is_tensor(m1) and
                    torch.is_tensor(x2) and torch.is_tensor(m2)):
                continue
            y_t = _to_label_tensor(y, device)

            def _per_eye(x, m):
                img_f = None if model.bridge.mode == "clinical_only" else model.img_tower(x.to(device))
                md_f  = None if model.bridge.mode == "image_only"    else model.cd_tower(m.to(device))
                out_f, aux = model.bridge.fuse([img_f, md_f])
                out_i, out_m = aux[0], aux[1]
                pf = F.softmax(out_f, dim=1)
                pi = F.softmax(out_i, dim=1) if out_i is not None else pf
                pm = F.softmax(out_m, dim=1) if out_m is not None else pf
                return pf, pi, pm, out_f, (out_i if out_i is not None else out_f), (out_m if out_m is not None else out_f)

            pf_od, pi_od, pm_od, lf_od, li_od, lm_od = _per_eye(x1, m1)
            pf_os, pi_os, pm_os, lf_os, li_os, lm_os = _per_eye(x2, m2)

            if aggregate_patient:
                y_c.append(y_t.cpu().numpy())
                pf_c.append((0.5*(pf_od+pf_os)).cpu().numpy())
                pi_c.append((0.5*(pi_od+pi_os)).cpu().numpy())
                pm_c.append((0.5*(pm_od+pm_os)).cpu().numpy())
                lf_c.append((0.5*(lf_od+lf_os)).cpu().numpy())
                li_c.append((0.5*(li_od+li_os)).cpu().numpy())
                lm_c.append((0.5*(lm_od+lm_os)).cpu().numpy())
            else:
                y_np = y_t.cpu().numpy(); y_c += [y_np, y_np]
                pf_c += [pf_od.cpu().numpy(), pf_os.cpu().numpy()]
                pi_c += [pi_od.cpu().numpy(), pi_os.cpu().numpy()]
                pm_c += [pm_od.cpu().numpy(), pm_os.cpu().numpy()]
                lf_c += [lf_od.cpu().numpy(), lf_os.cpu().numpy()]
                li_c += [li_od.cpu().numpy(), li_os.cpu().numpy()]
                lm_c += [lm_od.cpu().numpy(), lm_os.cpu().numpy()]

    if not y_c:
        z = np.zeros((0, 0), dtype=np.float32)
        if return_logits:
            return np.array([], dtype=np.int64), z, z, z, z, z, z
        return np.array([], dtype=np.int64), z, z, z

    y  = np.concatenate(y_c)
    pf, pi, pm = (np.concatenate(c, axis=0) for c in (pf_c, pi_c, pm_c))
    if return_logits:
        lf, li, lm = (np.concatenate(c, axis=0) for c in (lf_c, li_c, lm_c))
        return y, pf, pi, pm, lf, li, lm
    return y, pf, pi, pm


def collect_probs_eye_level(
    model: SingleEyeHT, loader: DataLoader, device: torch.device, *, return_ids: bool = False,
):
    model.eval()
    y_c, pf_c, pi_c, pm_c, id_c = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch.get("image_1"); m = batch.get("matrix_1"); y = batch.get("label_1")
            if not (torch.is_tensor(x) and torch.is_tensor(m)):
                continue
            y_t = _to_label_tensor(y, device)
            img_f = None if model.bridge.mode == "clinical_only" else model.img_tower(x.to(device))
            md_f  = None if model.bridge.mode == "image_only"    else model.cd_tower(m.to(device))
            out_f, aux = model.bridge.fuse([img_f, md_f])
            out_i, out_m = aux[0], aux[1]
            pf = F.softmax(out_f, dim=1)
            pi = F.softmax(out_i, dim=1) if out_i is not None else pf
            pm = F.softmax(out_m, dim=1) if out_m is not None else pf
            y_c.append(y_t.cpu().numpy())
            pf_c.append(pf.cpu().numpy()); pi_c.append(pi.cpu().numpy()); pm_c.append(pm.cpu().numpy())
            if return_ids:
                ids  = batch.get("id_1",     [""] * len(y_t))
                eyes = batch.get("eye_id_1", [""] * len(y_t))
                if torch.is_tensor(ids):  ids  = ids.tolist()
                if torch.is_tensor(eyes): eyes = eyes.tolist()
                id_c.extend([f"{p}{e}" for p, e in zip(ids, eyes)])

    if not y_c:
        z = np.zeros((0, 0), dtype=np.float32)
        if return_ids:
            return np.array([], dtype=np.int64), z, z, z, np.array([], dtype=object)
        return np.array([], dtype=np.int64), z, z, z

    y  = np.concatenate(y_c)
    pf, pi, pm = (np.concatenate(c, axis=0) for c in (pf_c, pi_c, pm_c))
    if return_ids:
        return y, pf, pi, pm, np.array(id_c, dtype=object)
    return y, pf, pi, pm


# ---------------------------------------------------------------------------
# MonoTowerHT training / inference helpers
# ---------------------------------------------------------------------------

def train_mono_epoch(
    model: MonoTowerHT,
    loader: DataLoader,
    opt,
    device: torch.device,
) -> tuple[float, float]:
    """One training epoch for MonoTowerHT.

    Uses the tower's ``embed_batch`` (TowerBase API) to extract the embedding,
    then classifies directly with the head.  Loss is cross-entropy on the
    head output only — no aux head.

    Returns
    -------
    (mean_loss, accuracy)
    """
    model.train()
    total_loss = total_correct = total_n = 0
    for batch in loader:
        y = _to_label_tensor(batch.get("label_1"), device)
        if y.numel() == 0:
            continue
        [z] = model.tower.embed_batch(batch, device=device)
        logits = model.classify(z)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss    += loss.item() * len(y)
        total_correct += int((logits.argmax(1) == y).sum())
        total_n       += len(y)
    mean_loss = total_loss / total_n if total_n else float("nan")
    accuracy  = total_correct / total_n if total_n else float("nan")
    return mean_loss, accuracy


def collect_probs_mono(
    model: MonoTowerHT,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect predictions from a MonoTowerHT.

    Returns
    -------
    y_true : int64 array [N]
    probs  : float32 array [N, num_classes]  — softmax of fused head
    """
    model.eval()
    y_all, p_all = [], []
    with torch.no_grad():
        for batch in loader:
            y = _to_label_tensor(batch.get("label_1"), device)
            if y.numel() == 0:
                continue
            [z] = model.tower.embed_batch(batch, device=device)
            logits = model.classify(z)
            y_all.append(y.cpu().numpy())
            p_all.append(F.softmax(logits, dim=1).cpu().numpy())
    if not y_all:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(y_all), np.concatenate(p_all, axis=0)


# ---------------------------------------------------------------------------
# NTowerHT training / inference helpers
# ---------------------------------------------------------------------------

def train_ntower_epoch(
    model: NTowerHT,
    loader: DataLoader,
    opt,
    device: torch.device,
    *,
    batch_key_map: dict[str, str],
    phase: str,
    bcd_prob: float = 0.5,
    tower_loss_mode: str = "bcd",
) -> tuple[float, float]:
    """One training epoch for NTowerHT.

    Parameters
    ----------
    batch_key_map : {tower_name: batch_key} — maps each tower slot to the key
                   in the DataLoader batch dict that carries its input tensor.
                   Example: {'od': 'image_1', 'os': 'image_2', 'cd': 'matrix_1'}
    phase         : 'tower_warmup' | 'fused_warmup' | 'main'
    bcd_prob      : probability of using a random aux head instead of the fused
                   head in 'main' phase (BCD training).
    tower_loss_mode : 'bcd' | 'all' — 'all' adds all aux + fused losses each step.
    """
    from random import choice
    model.train()
    model.bridge.set_phase(phase)
    # Mirror _set_single_phase tower freezing:
    # tower_warmup → towers trainable; fused_warmup → towers frozen; main → trainable
    towers_trainable = phase != "fused_warmup"
    for t in model.towers.values():
        for p in t.parameters():
            p.requires_grad_(towers_trainable)
    tower_names = list(model.towers.keys())
    total_loss = total_correct = total_n = 0

    for batch in loader:
        tensors = {name: batch.get(key) for name, key in batch_key_map.items()}
        if not all(torch.is_tensor(t) for t in tensors.values()):
            continue
        y = _to_label_tensor(batch.get("label_1"), device)
        if y.numel() == 0:
            continue
        embeddings = {name: model.towers[name](t.to(device)) for name, t in tensors.items()}

        if phase == "tower_warmup":
            aux_logits = [model.bridge.aux_heads[i](embeddings[name])
                          for i, name in enumerate(tower_names)]
            loss = sum(F.cross_entropy(l, y) for l in aux_logits) / len(aux_logits)
            avg_probs = sum(F.softmax(l, dim=1) for l in aux_logits) / len(aux_logits)
            logits = avg_probs  # for accuracy tracking
        elif phase == "fused_warmup":
            logits, _ = model(embeddings)
            loss = F.cross_entropy(logits, y)
        else:  # main
            if tower_loss_mode == "all":
                logits, aux_dict = model(embeddings)
                loss = F.cross_entropy(logits, y) + sum(
                    F.cross_entropy(a, y) for a in aux_dict.values()
                )
            elif random() < bcd_prob:
                # BCD: pick one random aux head
                name = choice(tower_names)
                idx = tower_names.index(name)
                logits = model.bridge.aux_heads[idx](embeddings[name])
                loss = F.cross_entropy(logits, y)
            else:
                logits, _ = model(embeddings)
                loss = F.cross_entropy(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()
        if logits.ndim == 2:
            total_correct += int((logits.argmax(1) == y).sum())
        total_loss += loss.item() * len(y)
        total_n    += len(y)

    return (
        total_loss / total_n if total_n else float("nan"),
        total_correct / total_n if total_n else float("nan"),
    )


def collect_probs_ntower(
    model: NTowerHT,
    loader: DataLoader,
    device: torch.device,
    *,
    batch_key_map: dict[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    """Collect fused-head predictions from NTowerHT.

    Parameters
    ----------
    batch_key_map : same mapping used during training.

    Returns
    -------
    y_true : int64 array [N]
    probs  : float32 array [N, num_classes] — softmax of fused head
    """
    model.eval()
    y_all, p_all = [], []
    with torch.no_grad():
        for batch in loader:
            tensors = {name: batch.get(key) for name, key in batch_key_map.items()}
            if not all(torch.is_tensor(t) for t in tensors.values()):
                continue
            y = _to_label_tensor(batch.get("label_1"), device)
            if y.numel() == 0:
                continue
            embeddings = {name: model.towers[name](t.to(device)) for name, t in tensors.items()}
            logits, _ = model(embeddings)
            y_all.append(y.cpu().numpy())
            p_all.append(F.softmax(logits, dim=1).cpu().numpy())
    if not y_all:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(y_all), np.concatenate(p_all, axis=0)


# ---------------------------------------------------------------------------
# V2ModeComparisonOps — backward-compat namespace
# ---------------------------------------------------------------------------

class V2ModeComparisonOps:
    """Namespace kept for backward-compatibility imports."""
    _set_requires_grad    = staticmethod(_set_requires_grad)
    _set_single_phase     = staticmethod(_set_single_phase)
    _set_bilateral_phase  = staticmethod(_set_bilateral_phase)
    train_single_epoch    = staticmethod(train_single_epoch)
    train_bilateral_epoch = staticmethod(train_bilateral_epoch)
    collect_probs_classic    = staticmethod(collect_probs_classic)
    collect_probs_ensemble   = staticmethod(collect_probs_ensemble)
    collect_probs_bilateral  = staticmethod(collect_probs_bilateral)

    @staticmethod
    def _to_label_tensor(labels, device: torch.device) -> torch.Tensor:
        return _to_label_tensor(labels, device)
