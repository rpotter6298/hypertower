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
        bridge_mode: str = "fused",
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
            mode=bridge_mode,
            use_se=False,
        )

    @property
    def transform(self):
        return self.img_tower.transform

    def forward(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        img_feats = None if self.bridge.mode == "metadata_only" else self.img_tower(x)
        md_feats  = None if self.bridge.mode == "image_only"    else self.md_tower(meta)
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
    bridge_mode = model.bridge.mode
    # Ablation modes have no fusion bridge; fused_warmup is meaningless — treat as tower_warmup
    if bridge_mode in ("image_only", "metadata_only") and phase == "fused_warmup":
        phase = "tower_warmup"
    if phase == "md_warmup":
        _set_requires_grad(model.img_tower, False)
        _set_requires_grad(model.md_tower, True)
        _set_requires_grad(model.bridge.classifier_img, False)
        _set_requires_grad(model.bridge.classifier_md, True)
        _set_requires_grad(model.bridge.W_img, False)
        _set_requires_grad(model.bridge.W_md, False)
        _set_requires_grad(model.bridge.classifier_fused, False)
        return
    if phase == "tower_warmup":
        _set_requires_grad(model.img_tower, bridge_mode != "metadata_only")
        _set_requires_grad(model.md_tower, bridge_mode != "image_only")
        _set_requires_grad(model.bridge.classifier_img, bridge_mode != "metadata_only")
        _set_requires_grad(model.bridge.classifier_md, bridge_mode != "image_only")
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
    tower_loss_mode: str = "bcd",
) -> tuple[float, float]:
    model.train()
    _set_single_phase(model, phase)
    total_loss = total_correct = total_n = 0
    for batch in loader:
        x = batch.get("image_1")
        m = batch.get("matrix_1")
        y = batch.get("label_1")
        if phase == "md_warmup":
            if not torch.is_tensor(m):
                continue
            m = m.to(device)
            y = _to_label_tensor(y, device)
            md_feats = model.md_tower(m)
            logits = model.bridge.classifier_md(md_feats)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            bs = y.shape[0]
            total_loss += float(loss.item()) * bs
            total_correct += int((logits.argmax(1) == y).sum())
            total_n += bs
            continue
        if not torch.is_tensor(x) or not torch.is_tensor(m):
            continue
        x = x.to(device)
        m = m.to(device)
        y = _to_label_tensor(y, device)
        bridge_mode = model.bridge.mode

        img_feats = None if bridge_mode == "metadata_only" else model.img_tower(x)
        md_feats  = None if bridge_mode == "image_only"    else model.md_tower(m)

        if phase == "tower_warmup":
            if bridge_mode == "metadata_only":
                logits = model.bridge.classifier_md(md_feats)
                loss = F.cross_entropy(logits, y)
            elif bridge_mode == "image_only":
                logits = model.bridge.classifier_img(img_feats)
                loss = F.cross_entropy(logits, y)
            else:
                logits_i = model.bridge.classifier_img(img_feats)
                logits_m = model.bridge.classifier_md(md_feats)
                loss = 0.5 * (F.cross_entropy(logits_i, y) + F.cross_entropy(logits_m, y))
                logits = 0.5 * (F.softmax(logits_i, dim=1) + F.softmax(logits_m, dim=1))
        elif phase == "fused_warmup":
            logits, _, _ = model.bridge(img_feats, md_feats)
            loss = F.cross_entropy(logits, y)
        else:
            if bridge_mode == "metadata_only":
                logits = model.bridge.classifier_md(md_feats)
                loss = F.cross_entropy(logits, y)
            elif bridge_mode == "image_only":
                logits = model.bridge.classifier_img(img_feats)
                loss = F.cross_entropy(logits, y)
            elif tower_loss_mode == "all":
                loss_i = F.cross_entropy(model.bridge.classifier_img(img_feats), y)
                loss_m = F.cross_entropy(model.bridge.classifier_md(md_feats), y)
                logits, _, _ = model.bridge(img_feats, md_feats)
                loss = F.cross_entropy(logits, y) + loss_i + loss_m
            elif random() < bcd_prob:
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
    tower_loss_mode: str = "bcd",
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
            if tower_loss_mode == "all":
                loss_i = F.cross_entropy(model.aux_img(joint_img), y)
                loss_m = F.cross_entropy(model.aux_md(joint_md), y)
                logits, _, _ = model.bridge(joint_img, joint_md)
                loss = F.cross_entropy(logits, y) + loss_i + loss_m
            elif random() < bcd_prob:
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


def collect_probs_ensemble_pereye(
    model: "SingleEyeHT",
    loader: DataLoader,
    device: torch.device,
    *,
    return_ids: bool = False,
):
    """
    Per-patient, per-eye probs for all 3 heads from a bilateral loader (ensemble mode).

    OD corresponds to image_1/matrix_1; OS to image_2/matrix_2.
    Arrays are in patient order (not interleaved at sample level).

    Returns:
        (y, pf_od, pi_od, pm_od, pf_os, pi_os, pm_os)
        or, when return_ids=True:
        (y, pf_od, pi_od, pm_od, pf_os, pi_os, pm_os, patient_ids)

    Patient-level averaged ensemble probs can be recovered as:
        p_en = 0.5 * (pf_od + pf_os)
    """
    model.eval()
    y_chunks: list = []
    pf_od_c, pi_od_c, pm_od_c = [], [], []
    pf_os_c, pi_os_c, pm_os_c = [], [], []
    id_chunks: list[str] = []

    with torch.no_grad():
        for batch in loader:
            x1 = batch.get("image_1"); m1 = batch.get("matrix_1")
            x2 = batch.get("image_2"); m2 = batch.get("matrix_2")
            y  = batch.get("label_1")
            if not (torch.is_tensor(x1) and torch.is_tensor(m1) and
                    torch.is_tensor(x2) and torch.is_tensor(m2)):
                continue
            y_t = _to_label_tensor(y, device)

            def _fwd(x, m):
                img_feats = None if model.bridge.mode == "metadata_only" else model.img_tower(x.to(device))
                md_feats  = None if model.bridge.mode == "image_only"    else model.md_tower(m.to(device))
                out_f, out_i, out_m = model.bridge(img_feats, md_feats)
                pf = F.softmax(out_f, dim=1)
                pi = F.softmax(out_i, dim=1) if out_i is not None else pf
                pm = F.softmax(out_m, dim=1) if out_m is not None else pf
                return pf, pi, pm

            pf_od, pi_od, pm_od = _fwd(x1, m1)
            pf_os, pi_os, pm_os = _fwd(x2, m2)

            y_chunks.append(y_t.cpu().numpy())
            pf_od_c.append(pf_od.cpu().numpy()); pi_od_c.append(pi_od.cpu().numpy()); pm_od_c.append(pm_od.cpu().numpy())
            pf_os_c.append(pf_os.cpu().numpy()); pi_os_c.append(pi_os.cpu().numpy()); pm_os_c.append(pm_os.cpu().numpy())

            if return_ids:
                ids = batch.get("id_1", [""] * len(y_t))
                if torch.is_tensor(ids):
                    ids = ids.tolist()
                id_chunks.extend([str(i) for i in ids])

    if not y_chunks:
        z = np.zeros((0, 0), dtype=np.float32)
        empty_i = np.array([], dtype=np.int64)
        base = (empty_i, z, z, z, z, z, z)
        return base + (np.array([], dtype=object),) if return_ids else base

    y     = np.concatenate(y_chunks)
    pf_od = np.concatenate(pf_od_c, axis=0); pi_od = np.concatenate(pi_od_c, axis=0); pm_od = np.concatenate(pm_od_c, axis=0)
    pf_os = np.concatenate(pf_os_c, axis=0); pi_os = np.concatenate(pi_os_c, axis=0); pm_os = np.concatenate(pm_os_c, axis=0)
    if return_ids:
        return y, pf_od, pi_od, pm_od, pf_os, pi_os, pm_os, np.array(id_chunks, dtype=object)
    return y, pf_od, pi_od, pm_od, pf_os, pi_os, pm_os


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
    return_logits: bool = False,
):
    """
    Collect fused/img/md probabilities (and optionally raw logits) for SingleEyeHT.
    - aggregate_patient=False: eye-level (OD/OS as independent samples)
    - aggregate_patient=True : patient-level (average OD/OS per head)
    - return_logits=False: returns (y, probs_f, probs_i, probs_m)
    - return_logits=True:  returns (y, probs_f, probs_i, probs_m,
                                       logits_f, logits_i, logits_m)
      Note: logits are averaged across eyes when aggregate_patient=True,
      which is equivalent to averaging in logit space (before softmax).
    """
    model.eval()
    y_chunks = []
    pf_chunks, pi_chunks, pm_chunks = [], [], []
    lf_chunks, li_chunks, lm_chunks = [], [], []
    with torch.no_grad():
        for batch in loader:
            x1 = batch.get("image_1"); m1 = batch.get("matrix_1")
            x2 = batch.get("image_2"); m2 = batch.get("matrix_2")
            y = batch.get("label_1")
            if not (torch.is_tensor(x1) and torch.is_tensor(m1) and torch.is_tensor(x2) and torch.is_tensor(m2)):
                continue
            y_t = _to_label_tensor(y, device)

            def _per_eye(x, m):
                img_feats = None if model.bridge.mode == "metadata_only" else model.img_tower(x.to(device))
                md_feats  = None if model.bridge.mode == "image_only"    else model.md_tower(m.to(device))
                out_f, out_i, out_m = model.bridge(img_feats, md_feats)
                pf = F.softmax(out_f, dim=1)
                pi = F.softmax(out_i, dim=1) if out_i is not None else pf
                pm = F.softmax(out_m, dim=1) if out_m is not None else pf
                lf = out_f
                li = out_i if out_i is not None else out_f
                lm = out_m if out_m is not None else out_f
                return pf, pi, pm, lf, li, lm

            pf_od, pi_od, pm_od, lf_od, li_od, lm_od = _per_eye(x1, m1)
            pf_os, pi_os, pm_os, lf_os, li_os, lm_os = _per_eye(x2, m2)

            if aggregate_patient:
                y_chunks.append(y_t.cpu().numpy())
                pf_chunks.append((0.5 * (pf_od + pf_os)).cpu().numpy())
                pi_chunks.append((0.5 * (pi_od + pi_os)).cpu().numpy())
                pm_chunks.append((0.5 * (pm_od + pm_os)).cpu().numpy())
                lf_chunks.append((0.5 * (lf_od + lf_os)).cpu().numpy())
                li_chunks.append((0.5 * (li_od + li_os)).cpu().numpy())
                lm_chunks.append((0.5 * (lm_od + lm_os)).cpu().numpy())
            else:
                y_np = y_t.cpu().numpy()
                y_chunks += [y_np, y_np]
                pf_chunks += [pf_od.cpu().numpy(), pf_os.cpu().numpy()]
                pi_chunks += [pi_od.cpu().numpy(), pi_os.cpu().numpy()]
                pm_chunks += [pm_od.cpu().numpy(), pm_os.cpu().numpy()]
                lf_chunks += [lf_od.cpu().numpy(), lf_os.cpu().numpy()]
                li_chunks += [li_od.cpu().numpy(), li_os.cpu().numpy()]
                lm_chunks += [lm_od.cpu().numpy(), lm_os.cpu().numpy()]

    if not y_chunks:
        z = np.zeros((0, 0), dtype=np.float32)
        if return_logits:
            return np.array([], dtype=np.int64), z, z, z, z, z, z
        return np.array([], dtype=np.int64), z, z, z

    y   = np.concatenate(y_chunks)
    pf  = np.concatenate(pf_chunks, axis=0)
    pi  = np.concatenate(pi_chunks, axis=0)
    pm  = np.concatenate(pm_chunks, axis=0)
    if return_logits:
        lf = np.concatenate(lf_chunks, axis=0)
        li = np.concatenate(li_chunks, axis=0)
        lm = np.concatenate(lm_chunks, axis=0)
        return y, pf, pi, pm, lf, li, lm
    return y, pf, pi, pm


def collect_probs_eye_level(
    model: "SingleEyeHT",
    loader: DataLoader,
    device: torch.device,
    *,
    return_ids: bool = False,
):
    """
    Collect fused/img/md probabilities from a single-eye loader (image_1/matrix_1 only).
    Used for eval-mode passes over the training set.

    Returns (y, probs_f, probs_i, probs_m) or, when return_ids=True,
    (y, probs_f, probs_i, probs_m, sample_ids) where sample_ids is an
    array of strings like "2OD", "4OS".
    """
    model.eval()
    y_chunks, pf_chunks, pi_chunks, pm_chunks, id_chunks = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch.get("image_1")
            m = batch.get("matrix_1")
            y = batch.get("label_1")
            if not (torch.is_tensor(x) and torch.is_tensor(m)):
                continue
            y_t = _to_label_tensor(y, device)
            img_feats = None if model.bridge.mode == "metadata_only" else model.img_tower(x.to(device))
            md_feats  = None if model.bridge.mode == "image_only"    else model.md_tower(m.to(device))
            out_f, out_i, out_m = model.bridge(img_feats, md_feats)
            pf = F.softmax(out_f, dim=1)
            pi = F.softmax(out_i, dim=1) if out_i is not None else pf
            pm = F.softmax(out_m, dim=1) if out_m is not None else pf
            y_chunks.append(y_t.cpu().numpy())
            pf_chunks.append(pf.cpu().numpy())
            pi_chunks.append(pi.cpu().numpy())
            pm_chunks.append(pm.cpu().numpy())
            if return_ids:
                ids   = batch.get("id_1", [""] * len(y_t))
                eyes  = batch.get("eye_id_1", [""] * len(y_t))
                # ids/eyes may be tensors (int) or lists of strings
                if torch.is_tensor(ids):
                    ids = ids.tolist()
                if torch.is_tensor(eyes):
                    eyes = eyes.tolist()
                id_chunks.extend(
                    [f"{pid}{eye}" for pid, eye in zip(ids, eyes)]
                )

    if not y_chunks:
        z = np.zeros((0, 0), dtype=np.float32)
        empty_ids = np.array([], dtype=object)
        if return_ids:
            return np.array([], dtype=np.int64), z, z, z, empty_ids
        return np.array([], dtype=np.int64), z, z, z

    y  = np.concatenate(y_chunks)
    pf = np.concatenate(pf_chunks, axis=0)
    pi = np.concatenate(pi_chunks, axis=0)
    pm = np.concatenate(pm_chunks, axis=0)
    if return_ids:
        return y, pf, pi, pm, np.array(id_chunks, dtype=object)
    return y, pf, pi, pm


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
