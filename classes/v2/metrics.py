"""Metric computation, calibration, and threshold/bias tuning for V2."""
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
)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 0.0,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Standard focal loss wrapper. When gamma=0 it reduces to cross entropy.
    weight should be per-class weights (same semantics as CrossEntropyLoss).
    """
    if gamma <= 0:
        return F.cross_entropy(logits, targets, weight=weight, reduction=reduction)

    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()

    targets = targets.long().view(-1, 1)
    logpt = log_probs.gather(1, targets)
    pt = probs.gather(1, targets)

    focal_factor = (1.0 - pt).clamp_min(0.0) ** gamma
    loss = -focal_factor * logpt

    if weight is not None:
        class_weight = weight.gather(0, targets.view(-1))
        loss = loss * class_weight.view(-1, 1)

    loss = loss.view(-1)
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


# ---------------------------------------------------------------------------
# Basic array scoring
# ---------------------------------------------------------------------------

def _score_arrays(y_true: np.ndarray, probs: np.ndarray, num_classes: int):
    """Returns (acc, auc, n)."""
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
    """Expected Calibration Error: weighted mean of |confidence - accuracy| per bin."""
    if y_true.size == 0:
        return float("nan")
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = (confidences >= lo) & (
            confidences <= hi if i == n_bins - 1 else confidences < hi
        )
        if not mask.any():
            continue
        bin_acc  = float((predictions[mask] == y_true[mask]).mean())
        bin_conf = float(confidences[mask].mean())
        ece += float(mask.sum()) / n * abs(bin_conf - bin_acc)
    return float(ece)


def compute_extended_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    num_classes: int,
    n_bins: int = 10,
    preds_override: Optional[np.ndarray] = None,
) -> dict:
    nan = float("nan")
    if y_true.size == 0:
        return dict(
            kappa=nan, mcc=nan, macro_f1=nan,
            per_class_recall=np.full(num_classes, nan), ece=nan,
        )
    preds = preds_override if preds_override is not None else probs.argmax(axis=1)
    try:
        kappa = float(cohen_kappa_score(y_true, preds))
    except Exception:
        kappa = nan
    try:
        mcc = float(matthews_corrcoef(y_true, preds))
    except Exception:
        mcc = nan
    try:
        macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
    except Exception:
        macro_f1 = nan
    try:
        pcr = recall_score(
            y_true, preds, average=None,
            labels=list(range(num_classes)), zero_division=0,
        ).astype(float)
    except Exception:
        pcr = np.full(num_classes, nan)
    ece = compute_ece(y_true, probs, n_bins=n_bins)
    return dict(kappa=kappa, mcc=mcc, macro_f1=macro_f1, per_class_recall=pcr, ece=ece)


# ---------------------------------------------------------------------------
# Threshold / bias tuning
# ---------------------------------------------------------------------------

def tune_binary_threshold(y_true: np.ndarray, p1: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.5
    grid = np.linspace(0.0, 1.0, 1001)
    best_t, best_acc = 0.5, -1.0
    for t in grid:
        pred = (p1 >= t).astype(int)
        acc = float((pred == y_true).mean())
        if acc > best_acc or (acc == best_acc and abs(t - 0.5) < abs(best_t - 0.5)):
            best_acc, best_t = acc, float(t)
    return best_t


def multiclass_acc_with_bias(y_true: np.ndarray, probs: np.ndarray, bias: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    logits = np.log(np.clip(probs, 1e-8, 1.0)) + bias.reshape(1, -1)
    return float((np.argmax(logits, axis=1) == y_true).mean())


def tune_multiclass_bias(y_true: np.ndarray, probs: np.ndarray, *, iters: int = 2) -> np.ndarray:
    if y_true.size == 0 or probs.size == 0:
        return np.zeros((0,), dtype=float)
    c = probs.shape[1]
    bias = np.zeros((c,), dtype=float)
    grid = np.linspace(-1.0, 1.0, 41)
    for _ in range(iters):
        for k in range(c):
            best_v = bias[k]
            best_acc = multiclass_acc_with_bias(y_true, probs, bias)
            old = bias[k]
            for v in grid:
                bias[k] = float(v)
                acc = multiclass_acc_with_bias(y_true, probs, bias)
                if acc > best_acc or (acc == best_acc and abs(v) < abs(best_v)):
                    best_acc, best_v = acc, float(v)
            bias[k] = best_v
            if np.isnan(best_acc):
                bias[k] = old
    return bias


def _svf(vec) -> Optional[str]:
    """Serialise a float vector to pipe-separated string, or None if empty."""
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=float)
    if arr.size == 0:
        return None
    return "|".join(f"{float(v):.4f}" for v in arr.tolist())


def _tune_and_snap(
    y: np.ndarray,
    p: np.ndarray,
    acc: float,
    num_classes: int,
    args,
    n_bins: int,
) -> tuple[dict, float, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Apply threshold/bias tuning and compute extended metrics.
    Returns (snap_dict, tuned_auc, threshold, bias).
    """
    thr  = 0.5 if num_classes == 2 else float("nan")
    bias = None
    ext_preds = None

    if args.tune_binary_threshold and num_classes == 2 and y.size > 0:
        thr = tune_binary_threshold(y, p[:, 1])
        ext_preds = (p[:, 1] >= thr).astype(int)
        acc = float((ext_preds == y).mean())
    elif args.tune_multiclass_bias and num_classes > 2 and y.size > 0:
        bias = tune_multiclass_bias(y, p)
        logits = np.log(np.clip(p, 1e-8, 1.0)) + bias.reshape(1, -1)
        ext_preds = np.argmax(logits, axis=1)
        acc = float((ext_preds == y).mean())

    ext = compute_extended_metrics(y, p, num_classes, n_bins=n_bins, preds_override=ext_preds)
    _, auc, n = _score_arrays(y, p, num_classes)

    snap = dict(
        auc=auc, acc=acc, n=n,
        kappa=ext["kappa"], mcc=ext["mcc"], macro_f1=ext["macro_f1"],
        per_class_recall=ext["per_class_recall"], ece=ext["ece"],
        threshold=thr, bias=bias,
    )
    return snap, auc, thr, bias
