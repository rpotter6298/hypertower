"""V2 model classes and training/inference helpers."""
from __future__ import annotations

from random import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from classes.v2.bridges import Bridge
from classes.v2.towers import ImageTower, MDTower


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------

class SingleEyeHT(nn.Module):
    """
    ImageTower + MDTower + Bridge, trained on eye-level samples.
    Supports both Classic (eye-level) and Ensemble (patient-level averaging) eval.
    """

    def __init__(
        self,
        *,
        backbone: str,
        freeze_ratio: float,
        augment: bool,
        clinical_data,
        num_classes: int,
        md_hidden_dim: int = 128,
        fusion_dim: int = 256,
    ):
        super().__init__()
        self.img_tower = ImageTower(
            backbone=backbone,
            freeze_ratio=freeze_ratio,
            augment=augment,
            use_se=False,
        )
        self.md_tower = MDTower(
            clinical_data=clinical_data,
            hidden_dim=md_hidden_dim,
            use_se=False,
        )
        self.bridge = Bridge(
            img_dim=self.img_tower.out_dim,
            meta_dim=self.md_tower.out_dim,
            num_classes=num_classes,
            fusion_dim=fusion_dim,
            mode="fused",
            use_se=False,
        )

    @property
    def transform(self):
        return self.img_tower.transform

    def forward(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        img_feats = self.img_tower(x)
        md_feats  = self.md_tower(meta)
        out_f, _, _ = self.bridge(img_feats, md_feats)
        return out_f


class BilateralHT(nn.Module):
    """
    Bilateral mode with joint towers:
    - shared eye-level towers encode OD/OS independently
    - joint image and metadata towers combine OD/OS embeddings
    - standard Bridge fuses joint image + joint metadata embeddings
    """

    def __init__(
        self,
        *,
        backbone: str,
        freeze_ratio: float,
        augment: bool,
        clinical_data,
        num_classes: int,
        md_hidden_dim: int = 128,
        fusion_dim: int = 256,
    ):
        super().__init__()
        self.eye_img_tower = ImageTower(
            backbone=backbone,
            freeze_ratio=freeze_ratio,
            augment=augment,
            use_se=False,
        )
        self.eye_md_tower = MDTower(
            clinical_data=clinical_data,
            hidden_dim=md_hidden_dim,
            use_se=False,
        )
        img_dim = self.eye_img_tower.out_dim
        md_dim = self.eye_md_tower.out_dim
        self.joint_img = nn.Sequential(
            nn.Linear(2 * img_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, img_dim),
        )
        self.joint_md = nn.Sequential(
            nn.Linear(2 * md_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, md_dim),
        )
        self.bridge = Bridge(
            img_dim=img_dim,
            meta_dim=md_dim,
            num_classes=num_classes,
            fusion_dim=fusion_dim,
            mode="fused",
            use_se=False,
        )
        # Auxiliary heads for tower warmup / BCD tower steps.
        self.aux_img = nn.Linear(img_dim, num_classes)
        self.aux_md = nn.Linear(md_dim, num_classes)

    @property
    def transform(self):
        return self.eye_img_tower.transform

    def encode_joint(
        self,
        x_od: torch.Tensor,
        meta_od: torch.Tensor,
        x_os: torch.Tensor,
        meta_os: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img_od = self.eye_img_tower(x_od)
        md_od  = self.eye_md_tower(meta_od)
        img_os = self.eye_img_tower(x_os)
        md_os  = self.eye_md_tower(meta_os)
        joint_img = self.joint_img(torch.cat([img_od, img_os], dim=1))
        joint_md = self.joint_md(torch.cat([md_od, md_os], dim=1))
        return joint_img, joint_md

    def forward(
        self,
        x_od:    torch.Tensor,
        meta_od: torch.Tensor,
        x_os:    torch.Tensor,
        meta_os: torch.Tensor,
    ) -> torch.Tensor:
        joint_img, joint_md = self.encode_joint(x_od, meta_od, x_os, meta_os)
        out_f, _, _ = self.bridge(joint_img, joint_md)
        return out_f


class FusedEnsembleHT(nn.Module):
    """
    SingleEyeHT base with a per-eye attention scorer for bilateral fusion.

    The base model is trained eye-level (identical to ensemble mode).
    After base training completes, the base is frozen and only the
    eye_scorer is trained on bilateral (patient-level) samples.

    At inference, eye_scorer is applied independently to each eye's logit
    vector to produce a scalar attention score.  Softmax over the two scores
    gives attention weights; the final logit is a weighted sum:

        score_od = eye_scorer(logit_od)          # [B, 1]
        score_os = eye_scorer(logit_os)          # [B, 1]
        alpha    = softmax([score_od, score_os]) # [B, 2], sums to 1
        out      = alpha[:,0:1]*logit_od + alpha[:,1:2]*logit_os

    Because eye_scorer is applied to each eye with the same weights, the
    mechanism is permutation-equivariant — there is no left/right positional
    bias.  Through training on bilateral labels the scorer learns to give high
    scores to logits that point strongly toward the GC class, creating the
    desired asymmetry: a confidently GC eye dominates the patient prediction
    more than a comparably confident healthy eye would.
    """

    def __init__(self, base: SingleEyeHT, num_classes: int):
        super().__init__()
        self.base = base
        # Applied independently to each eye's logit → scalar attention score.
        # Learns the GC-direction in logit space from bilateral labels.
        self.eye_scorer = nn.Linear(num_classes, 1, bias=True)

    def forward(
        self,
        x_od:    torch.Tensor,
        meta_od: torch.Tensor,
        x_os:    torch.Tensor,
        meta_os: torch.Tensor,
    ) -> torch.Tensor:
        logit_od = self.base(x_od, meta_od)                                    # [B, C]
        logit_os = self.base(x_os, meta_os)                                    # [B, C]
        scores   = torch.cat([self.eye_scorer(logit_od),
                               self.eye_scorer(logit_os)], dim=1)              # [B, 2]
        alpha    = torch.softmax(scores, dim=1)                                # [B, 2]
        return alpha[:, 0:1] * logit_od + alpha[:, 1:2] * logit_os            # [B, C]


# ---------------------------------------------------------------------------
# Phase control
# ---------------------------------------------------------------------------

def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for p in module.parameters():
        p.requires_grad = enabled


def _set_single_phase(model: SingleEyeHT, phase: str) -> None:
    if phase == "tower_warmup":
        _set_requires_grad(model.img_tower, True)
        _set_requires_grad(model.md_tower, True)
        _set_requires_grad(model.bridge.classifier_img, True)
        _set_requires_grad(model.bridge.classifier_md, True)
        _set_requires_grad(model.bridge.W_img, False)
        _set_requires_grad(model.bridge.W_md, False)
        _set_requires_grad(model.bridge.classifier_fused, False)
        return
    if phase == "fused_warmup":
        _set_requires_grad(model.img_tower, False)
        _set_requires_grad(model.md_tower, False)
        _set_requires_grad(model.bridge.classifier_img, False)
        _set_requires_grad(model.bridge.classifier_md, False)
        _set_requires_grad(model.bridge.W_img, True)
        _set_requires_grad(model.bridge.W_md, True)
        _set_requires_grad(model.bridge.classifier_fused, True)
        return
    _set_requires_grad(model, True)


def _set_bilateral_phase(model: BilateralHT, phase: str) -> None:
    if phase == "tower_warmup":
        _set_requires_grad(model.eye_img_tower, True)
        _set_requires_grad(model.eye_md_tower, True)
        _set_requires_grad(model.joint_img, True)
        _set_requires_grad(model.joint_md, True)
        _set_requires_grad(model.aux_img, True)
        _set_requires_grad(model.aux_md, True)
        _set_requires_grad(model.bridge, False)
        return
    if phase == "fused_warmup":
        _set_requires_grad(model.eye_img_tower, False)
        _set_requires_grad(model.eye_md_tower, False)
        _set_requires_grad(model.joint_img, False)
        _set_requires_grad(model.joint_md, False)
        _set_requires_grad(model.aux_img, False)
        _set_requires_grad(model.aux_md, False)
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
) -> tuple[float, float]:
    model.train()
    _set_single_phase(model, phase)
    total_loss = total_correct = total_n = 0
    for batch in loader:
        x = batch.get("image_1")
        m = batch.get("matrix_1")
        y = batch.get("label_1")
        if not torch.is_tensor(x) or not torch.is_tensor(m):
            continue
        x = x.to(device)
        m = m.to(device)
        y = _to_label_tensor(y, device)
        img_feats = model.img_tower(x)
        md_feats = model.md_tower(m)

        if phase == "tower_warmup":
            logits_i = model.bridge.classifier_img(img_feats)
            logits_m = model.bridge.classifier_md(md_feats)
            loss = 0.5 * (F.cross_entropy(logits_i, y) + F.cross_entropy(logits_m, y))
            logits = 0.5 * (F.softmax(logits_i, dim=1) + F.softmax(logits_m, dim=1))
        elif phase == "fused_warmup":
            logits, _, _ = model.bridge(img_feats, md_feats)
            loss = F.cross_entropy(logits, y)
        else:
            if random() < bcd_prob:
                if random() < 0.5:
                    logits = model.bridge.classifier_img(img_feats)
                else:
                    logits = model.bridge.classifier_md(md_feats)
                loss = F.cross_entropy(logits, y)
            else:
                logits, _, _ = model.bridge(img_feats, md_feats)
                loss = F.cross_entropy(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()
        bs = y.shape[0]
        total_loss += float(loss.item()) * bs
        total_correct += int((logits.argmax(1) == y).sum())
        total_n += bs
    return (
        total_loss / total_n if total_n else float("nan"),
        total_correct / total_n if total_n else float("nan"),
    )


def train_bilateral_epoch(
    model: BilateralHT,
    loader: DataLoader,
    opt,
    device: torch.device,
    *,
    phase: str,
    bcd_prob: float = 0.5,
) -> tuple[float, float]:
    model.train()
    _set_bilateral_phase(model, phase)
    total_loss = total_correct = total_n = 0
    for batch in loader:
        x1 = batch.get("image_1")
        m1 = batch.get("matrix_1")
        x2 = batch.get("image_2")
        m2 = batch.get("matrix_2")
        y = batch.get("label_1")
        if not (torch.is_tensor(x1) and torch.is_tensor(m1) and torch.is_tensor(x2) and torch.is_tensor(m2)):
            continue
        x1 = x1.to(device); m1 = m1.to(device)
        x2 = x2.to(device); m2 = m2.to(device)
        y = _to_label_tensor(y, device)
        joint_img, joint_md = model.encode_joint(x1, m1, x2, m2)

        if phase == "tower_warmup":
            logits_i = model.aux_img(joint_img)
            logits_m = model.aux_md(joint_md)
            loss = 0.5 * (F.cross_entropy(logits_i, y) + F.cross_entropy(logits_m, y))
            logits = 0.5 * (F.softmax(logits_i, dim=1) + F.softmax(logits_m, dim=1))
        elif phase == "fused_warmup":
            logits, _, _ = model.bridge(joint_img, joint_md)
            loss = F.cross_entropy(logits, y)
        else:
            if random() < bcd_prob:
                if random() < 0.5:
                    logits = model.aux_img(joint_img)
                else:
                    logits = model.aux_md(joint_md)
                loss = F.cross_entropy(logits, y)
            else:
                logits, _, _ = model.bridge(joint_img, joint_md)
                loss = F.cross_entropy(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()
        bs = y.shape[0]
        total_loss += float(loss.item()) * bs
        total_correct += int((logits.argmax(1) == y).sum())
        total_n += bs
    return (
        total_loss / total_n if total_n else float("nan"),
        total_correct / total_n if total_n else float("nan"),
    )


def train_fusion_epoch(
    model: FusedEnsembleHT,
    loader: DataLoader,
    opt,
    device: torch.device,
) -> tuple[float, float]:
    """Train only the fusion head; the base SingleEyeHT is frozen in eval mode."""
    model.base.eval()
    model.eye_scorer.train()
    total_loss = total_correct = total_n = 0
    for batch in loader:
        x1 = batch.get("image_1"); m1 = batch.get("matrix_1")
        x2 = batch.get("image_2"); m2 = batch.get("matrix_2")
        y  = batch.get("label_1")
        if not (torch.is_tensor(x1) and torch.is_tensor(m1) and
                torch.is_tensor(x2) and torch.is_tensor(m2)):
            continue
        y_t   = _to_label_tensor(y, device)
        out   = model(x1.to(device), m1.to(device), x2.to(device), m2.to(device))
        loss  = F.cross_entropy(out, y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        bs = y_t.shape[0]
        total_loss    += float(loss.item()) * bs
        total_correct += int((out.argmax(1) == y_t).sum())
        total_n       += bs
    return (
        total_loss / total_n if total_n else float("nan"),
        total_correct / total_n if total_n else float("nan"),
    )


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _to_label_tensor(labels, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(labels):
        return labels.to(device=device, dtype=torch.long)
    return torch.as_tensor(labels, dtype=torch.long, device=device)


def collect_probs_classic(
    model: SingleEyeHT,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Classic eye-level eval using the bilateral val loader.
    OD and OS are treated as independent samples (both contribute to the
    arrays with the same patient label). Returns (y_true [2N], probs [2N, C]).
    """
    model.eval()
    y_chunks, p_chunks = [], []
    with torch.no_grad():
        for batch in loader:
            x1 = batch.get("image_1"); m1 = batch.get("matrix_1")
            x2 = batch.get("image_2"); m2 = batch.get("matrix_2")
            y = batch.get("label_1")
            if not (torch.is_tensor(x1) and torch.is_tensor(m1) and torch.is_tensor(x2) and torch.is_tensor(m2)):
                continue
            y_t = _to_label_tensor(y, device)
            p_od = F.softmax(model(x1.to(device), m1.to(device)), dim=1)
            p_os = F.softmax(model(x2.to(device), m2.to(device)), dim=1)
            y_np = y_t.cpu().numpy()
            y_chunks += [y_np, y_np]
            p_chunks += [p_od.cpu().numpy(), p_os.cpu().numpy()]
    if not y_chunks:
        return np.array([], dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(y_chunks), np.concatenate(p_chunks, axis=0)


def collect_probs_ensemble(
    model: SingleEyeHT,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Patient-level ensemble eval: average OD and OS softmax probabilities.
    Returns (y_true [N], probs [N, C]).
    """
    model.eval()
    y_chunks, p_chunks = [], []
    with torch.no_grad():
        for batch in loader:
            x1 = batch.get("image_1"); m1 = batch.get("matrix_1")
            x2 = batch.get("image_2"); m2 = batch.get("matrix_2")
            y = batch.get("label_1")
            if not (torch.is_tensor(x1) and torch.is_tensor(m1) and torch.is_tensor(x2) and torch.is_tensor(m2)):
                continue
            y_t = _to_label_tensor(y, device)
            p_od = F.softmax(model(x1.to(device), m1.to(device)), dim=1)
            p_os = F.softmax(model(x2.to(device), m2.to(device)), dim=1)
            p = 0.5 * (p_od + p_os)
            y_chunks.append(y_t.cpu().numpy())
            p_chunks.append(p.cpu().numpy())
    if not y_chunks:
        return np.array([], dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(y_chunks), np.concatenate(p_chunks, axis=0)


def collect_probs_bilateral(
    model: BilateralHT,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Patient-level bilateral eval. Returns (y_true [N], probs [N, C])."""
    model.eval()
    y_chunks, p_chunks = [], []
    with torch.no_grad():
        for batch in loader:
            x1 = batch.get("image_1"); m1 = batch.get("matrix_1")
            x2 = batch.get("image_2"); m2 = batch.get("matrix_2")
            y = batch.get("label_1")
            if not (torch.is_tensor(x1) and torch.is_tensor(m1) and torch.is_tensor(x2) and torch.is_tensor(m2)):
                continue
            y_t = _to_label_tensor(y, device)
            p = F.softmax(model(x1.to(device), m1.to(device), x2.to(device), m2.to(device)), dim=1)
            y_chunks.append(y_t.cpu().numpy())
            p_chunks.append(p.cpu().numpy())
    if not y_chunks:
        return np.array([], dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(y_chunks), np.concatenate(p_chunks, axis=0)


def collect_probs_fused(
    model: FusedEnsembleHT,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Patient-level fused-head eval. Returns (y_true [N], probs [N, C])."""
    model.eval()
    y_chunks, p_chunks = [], []
    with torch.no_grad():
        for batch in loader:
            x1 = batch.get("image_1"); m1 = batch.get("matrix_1")
            x2 = batch.get("image_2"); m2 = batch.get("matrix_2")
            y  = batch.get("label_1")
            if not (torch.is_tensor(x1) and torch.is_tensor(m1) and
                    torch.is_tensor(x2) and torch.is_tensor(m2)):
                continue
            y_t = _to_label_tensor(y, device)
            p   = F.softmax(model(x1.to(device), m1.to(device),
                                  x2.to(device), m2.to(device)), dim=1)
            y_chunks.append(y_t.cpu().numpy())
            p_chunks.append(p.cpu().numpy())
    if not y_chunks:
        return np.array([], dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(y_chunks), np.concatenate(p_chunks, axis=0)


def collect_probs_single_components(
    model: SingleEyeHT,
    loader: DataLoader,
    device: torch.device,
    *,
    aggregate_patient: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect fused/img/md probabilities for SingleEyeHT.
    - aggregate_patient=False: eye-level (OD/OS as independent samples)
    - aggregate_patient=True : patient-level (average OD/OS per head)
    """
    model.eval()
    y_chunks = []
    pf_chunks, pi_chunks, pm_chunks = [], [], []
    with torch.no_grad():
        for batch in loader:
            x1 = batch.get("image_1"); m1 = batch.get("matrix_1")
            x2 = batch.get("image_2"); m2 = batch.get("matrix_2")
            y = batch.get("label_1")
            if not (torch.is_tensor(x1) and torch.is_tensor(m1) and torch.is_tensor(x2) and torch.is_tensor(m2)):
                continue
            y_t = _to_label_tensor(y, device)

            def _per_eye_probs(x, m):
                img_feats = model.img_tower(x.to(device))
                md_feats = model.md_tower(m.to(device))
                out_f, out_i, out_m = model.bridge(img_feats, md_feats)
                return (
                    F.softmax(out_f, dim=1),
                    F.softmax(out_i, dim=1),
                    F.softmax(out_m, dim=1),
                )

            pf_od, pi_od, pm_od = _per_eye_probs(x1, m1)
            pf_os, pi_os, pm_os = _per_eye_probs(x2, m2)

            if aggregate_patient:
                y_chunks.append(y_t.cpu().numpy())
                pf_chunks.append((0.5 * (pf_od + pf_os)).cpu().numpy())
                pi_chunks.append((0.5 * (pi_od + pi_os)).cpu().numpy())
                pm_chunks.append((0.5 * (pm_od + pm_os)).cpu().numpy())
            else:
                y_np = y_t.cpu().numpy()
                y_chunks += [y_np, y_np]
                pf_chunks += [pf_od.cpu().numpy(), pf_os.cpu().numpy()]
                pi_chunks += [pi_od.cpu().numpy(), pi_os.cpu().numpy()]
                pm_chunks += [pm_od.cpu().numpy(), pm_os.cpu().numpy()]

    if not y_chunks:
        z = np.zeros((0, 0), dtype=np.float32)
        return np.array([], dtype=np.int64), z, z, z
    return (
        np.concatenate(y_chunks),
        np.concatenate(pf_chunks, axis=0),
        np.concatenate(pi_chunks, axis=0),
        np.concatenate(pm_chunks, axis=0),
    )


def collect_probs_bilateral_components(
    model: BilateralHT,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect fused/img/md probabilities for bilateral joint-tower model."""
    model.eval()
    y_chunks = []
    pf_chunks, pi_chunks, pm_chunks = [], [], []
    with torch.no_grad():
        for batch in loader:
            x1 = batch.get("image_1"); m1 = batch.get("matrix_1")
            x2 = batch.get("image_2"); m2 = batch.get("matrix_2")
            y = batch.get("label_1")
            if not (torch.is_tensor(x1) and torch.is_tensor(m1) and torch.is_tensor(x2) and torch.is_tensor(m2)):
                continue
            y_t = _to_label_tensor(y, device)
            joint_img, joint_md = model.encode_joint(
                x1.to(device), m1.to(device), x2.to(device), m2.to(device)
            )
            out_f, _, _ = model.bridge(joint_img, joint_md)
            out_i = model.aux_img(joint_img)
            out_m = model.aux_md(joint_md)
            y_chunks.append(y_t.cpu().numpy())
            pf_chunks.append(F.softmax(out_f, dim=1).cpu().numpy())
            pi_chunks.append(F.softmax(out_i, dim=1).cpu().numpy())
            pm_chunks.append(F.softmax(out_m, dim=1).cpu().numpy())
    if not y_chunks:
        z = np.zeros((0, 0), dtype=np.float32)
        return np.array([], dtype=np.int64), z, z, z
    return (
        np.concatenate(y_chunks),
        np.concatenate(pf_chunks, axis=0),
        np.concatenate(pi_chunks, axis=0),
        np.concatenate(pm_chunks, axis=0),
    )


# ---------------------------------------------------------------------------
# V2ModeComparisonOps — thin class wrapper kept for external import compat
# ---------------------------------------------------------------------------

class V2ModeComparisonOps:
    """Namespace wrapper kept for backward-compatibility imports."""

    _set_requires_grad   = staticmethod(_set_requires_grad)
    _set_single_phase    = staticmethod(_set_single_phase)
    _set_bilateral_phase = staticmethod(_set_bilateral_phase)
    train_single_epoch   = staticmethod(train_single_epoch)
    train_bilateral_epoch = staticmethod(train_bilateral_epoch)
    collect_probs_classic   = staticmethod(collect_probs_classic)
    collect_probs_ensemble  = staticmethod(collect_probs_ensemble)
    collect_probs_bilateral = staticmethod(collect_probs_bilateral)

    @staticmethod
    def _to_label_tensor(labels, device: torch.device) -> torch.Tensor:
        return _to_label_tensor(labels, device)
