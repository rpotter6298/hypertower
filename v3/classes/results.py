"""Result dataclasses and serialisation helpers for V3 fold outputs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _nan() -> float:
    return float("nan")


def _f(v) -> Optional[float]:
    """Round a scalar to 6 dp, return None for nan/None."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), 6)


def _sv(vec) -> Optional[str]:
    """Serialise a vector to a pipe-separated string, or None."""
    if vec is None:
        return None
    return "|".join(f"{float(v):.4f}" for v in vec)


@dataclass
class FoldResult:
    mode: str
    fold: int
    # Epoch where each model hit its peak val AUC
    best_epoch_single: int
    best_epoch_bilat:  int
    # Classic (eye-level eval of SingleEyeHT)
    classic_val_auc:       float
    classic_val_acc:       float
    classic_val_kappa:     float
    classic_val_mcc:       float
    classic_val_f1:        float
    classic_val_recall:    Optional[str]
    classic_val_ece:       float
    classic_val_threshold: float
    classic_val_bias:      Optional[str]
    classic_val_n:         int
    # Ensemble (patient-level eval of SingleEyeHT)
    ensemble_val_auc:       float
    ensemble_val_acc:       float
    ensemble_val_kappa:     float
    ensemble_val_mcc:       float
    ensemble_val_f1:        float
    ensemble_val_recall:    Optional[str]
    ensemble_val_ece:       float
    ensemble_val_threshold: float
    ensemble_val_bias:      Optional[str]
    ensemble_val_n:         int
    # Bilateral (BilateralHT patient-level)
    bilat_val_auc:       float
    bilat_val_acc:       float
    bilat_val_kappa:     float
    bilat_val_mcc:       float
    bilat_val_f1:        float
    bilat_val_recall:    Optional[str]
    bilat_val_ece:       float
    bilat_val_threshold: float
    bilat_val_bias:      Optional[str]
    bilat_val_n:         int
    # Test metrics (evaluated once after training on final-epoch model)
    ensemble_test_auc:   float = float("nan")
    ensemble_test_acc:   float = float("nan")
    ensemble_test_kappa: float = float("nan")
    ensemble_test_f1:    float = float("nan")
    ensemble_test_ece:   float = float("nan")
    classic_test_auc:    float = float("nan")
    classic_test_acc:    float = float("nan")
    classic_test_kappa:  float = float("nan")
    classic_test_f1:     float = float("nan")
    classic_test_ece:    float = float("nan")
    bilat_test_auc:      float = float("nan")
    bilat_test_acc:      float = float("nan")
    bilat_test_kappa:    float = float("nan")
    bilat_test_f1:       float = float("nan")
    bilat_test_ece:      float = float("nan")
    test_n:              int   = 0
    # Training sample counts
    single_train_n: int = 0
    bilat_train_n:  int = 0
    # Fused head (optional)
    fused_val_auc:       float        = float("nan")
    fused_val_acc:       float        = float("nan")
    fused_val_kappa:     float        = float("nan")
    fused_val_mcc:       float        = float("nan")
    fused_val_f1:        float        = float("nan")
    fused_val_recall:    Optional[str] = None
    fused_val_ece:       float        = float("nan")
    fused_val_threshold: float        = float("nan")
    fused_val_bias:      Optional[str] = None
    fused_val_n:         int          = 0
    fused_test_auc:      float        = float("nan")
    fused_test_acc:      float        = float("nan")
    fused_test_kappa:    float        = float("nan")
    fused_test_f1:       float        = float("nan")
    fused_test_ece:      float        = float("nan")
    fused_test_n:        int          = 0


@dataclass
class FoldArtifacts:
    # Val split
    y_true_classic:  Optional[np.ndarray]
    probs_classic:   Optional[np.ndarray]
    y_true_ensemble: Optional[np.ndarray]
    probs_ensemble:  Optional[np.ndarray]
    y_true_bilat:    Optional[np.ndarray]
    probs_bilat:     Optional[np.ndarray]
    y_true_fused:    Optional[np.ndarray] = None
    probs_fused:     Optional[np.ndarray] = None
    probs_ensemble_img: Optional[np.ndarray] = None
    probs_ensemble_md:  Optional[np.ndarray] = None
    probs_classic_img:  Optional[np.ndarray] = None
    probs_classic_md:   Optional[np.ndarray] = None
    # per-eye (pre-averaged) for ensemble mode
    y_true_ensemble_pereye:    Optional[np.ndarray] = None
    probs_ensemble_pereye:     Optional[np.ndarray] = None
    probs_ensemble_img_pereye: Optional[np.ndarray] = None
    probs_ensemble_md_pereye:  Optional[np.ndarray] = None
    # raw logits — patient-level
    logits_ensemble:     Optional[np.ndarray] = None
    logits_ensemble_img: Optional[np.ndarray] = None
    logits_ensemble_md:  Optional[np.ndarray] = None
    logits_classic:      Optional[np.ndarray] = None
    logits_classic_img:  Optional[np.ndarray] = None
    logits_classic_md:   Optional[np.ndarray] = None
    # raw logits — per-eye
    logits_ensemble_pereye:     Optional[np.ndarray] = None
    logits_ensemble_img_pereye: Optional[np.ndarray] = None
    logits_ensemble_md_pereye:  Optional[np.ndarray] = None
    # Test split equivalents
    y_true_test:         Optional[np.ndarray] = None
    probs_test:          Optional[np.ndarray] = None
    probs_test_img:      Optional[np.ndarray] = None
    probs_test_md:       Optional[np.ndarray] = None
    y_true_fused_test:   Optional[np.ndarray] = None
    probs_fused_test:    Optional[np.ndarray] = None
