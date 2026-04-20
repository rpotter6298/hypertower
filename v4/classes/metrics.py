"""metrics — loss, scoring, calibration, and threshold/bias tuning."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def focal_loss(
    logits:    torch.Tensor,
    targets:   torch.Tensor,
    gamma:     float = 0.0,
    weight:    Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal loss; reduces to cross-entropy when gamma=0."""
    if gamma <= 0:
        return F.cross_entropy(logits, targets, weight=weight, reduction=reduction)
    log_probs = F.log_softmax(logits, dim=1)
    probs     = log_probs.exp()
    targets   = targets.long().view(-1, 1)
    logpt     = log_probs.gather(1, targets)
    pt        = probs.gather(1, targets)
    loss      = -(((1.0 - pt).clamp_min(0.0) ** gamma) * logpt)
    if weight is not None:
        loss = loss * weight.gather(0, targets.view(-1)).view(-1, 1)
    loss = loss.view(-1)
    if reduction == "sum":  return loss.sum()
    if reduction == "mean": return loss.mean()
    return loss


# ---------------------------------------------------------------------------
# Basic array scoring
# ---------------------------------------------------------------------------

def score_arrays(y_true: np.ndarray, probs: np.ndarray, num_classes: int):
    """Return (acc, auc, n)."""
    if y_true.size == 0:
        return float("nan"), float("nan"), 0
    acc = float((probs.argmax(1) == y_true).mean())
    try:
        auc = (
            float(roc_auc_score(y_true, probs[:, 1]))
            if num_classes == 2
            else float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
        )
    except Exception:
        auc = float("nan")
    return acc, auc, int(len(y_true))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def compute_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error: weighted mean |confidence − accuracy| per bin."""
    if y_true.size == 0:
        return float("nan")
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    bin_edges   = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n   = len(y_true)
    for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = (confidences >= lo) & (
            confidences <= hi if i == n_bins - 1 else confidences < hi
        )
        if not mask.any():
            continue
        ece += float(mask.sum()) / n * abs(
            float(confidences[mask].mean()) - float((predictions[mask] == y_true[mask]).mean())
        )
    return float(ece)


def compute_extended_metrics(
    y_true:          np.ndarray,
    probs:           np.ndarray,
    num_classes:     int,
    n_bins:          int = 10,
    preds_override:  Optional[np.ndarray] = None,
) -> dict:
    nan = float("nan")
    if y_true.size == 0:
        return dict(
            kappa=nan, mcc=nan, macro_f1=nan,
            per_class_recall=np.full(num_classes, nan), ece=nan,
        )
    preds = preds_override if preds_override is not None else probs.argmax(axis=1)
    try:    kappa    = float(cohen_kappa_score(y_true, preds))
    except: kappa    = nan
    try:    mcc      = float(matthews_corrcoef(y_true, preds))
    except: mcc      = nan
    try:    macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
    except: macro_f1 = nan
    try:
        pcr = recall_score(
            y_true, preds, average=None,
            labels=list(range(num_classes)), zero_division=0,
        ).astype(float)
    except:
        pcr = np.full(num_classes, nan)
    return dict(
        kappa=kappa, mcc=mcc, macro_f1=macro_f1,
        per_class_recall=pcr, ece=compute_ece(y_true, probs, n_bins=n_bins),
    )


# ---------------------------------------------------------------------------
# Threshold / bias tuning
# ---------------------------------------------------------------------------

def tune_binary_threshold(y_true: np.ndarray, p1: np.ndarray) -> float:
    """Pick threshold via Youden's J (sensitivity + specificity − 1).

    Class-distribution independent; falls back to 0.5 if fewer than two
    classes are present in y_true.
    """
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, p1)
    return float(thresholds[np.argmax(tpr + (1.0 - fpr) - 1.0)])


def multiclass_acc_with_bias(
    y_true: np.ndarray, probs: np.ndarray, bias: np.ndarray
) -> float:
    """Balanced accuracy (mean per-class recall) after applying log-space bias."""
    if y_true.size == 0:
        return float("nan")
    logits  = np.log(np.clip(probs, 1e-8, 1.0)) + bias.reshape(1, -1)
    preds   = np.argmax(logits, axis=1)
    classes = np.unique(y_true)
    return float(np.mean([(preds[y_true == c] == c).mean() for c in classes]))


def tune_multiclass_bias(
    y_true: np.ndarray, probs: np.ndarray, *, iters: int = 2
) -> np.ndarray:
    """Grid-search per-class log-space bias to maximise balanced accuracy."""
    if y_true.size == 0 or probs.size == 0:
        return np.zeros((0,), dtype=float)
    c    = probs.shape[1]
    bias = np.zeros((c,), dtype=float)
    grid = np.linspace(-1.0, 1.0, 41)
    for _ in range(iters):
        for k in range(c):
            best_v   = bias[k]
            best_acc = multiclass_acc_with_bias(y_true, probs, bias)
            old      = bias[k]
            for v in grid:
                bias[k] = float(v)
                acc = multiclass_acc_with_bias(y_true, probs, bias)
                if acc > best_acc or (acc == best_acc and abs(v) < abs(best_v)):
                    best_acc, best_v = acc, float(v)
            bias[k] = best_v
            if np.isnan(best_acc):
                bias[k] = old
    return bias
