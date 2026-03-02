"""Result dataclasses and serialisation helpers for V2 fold outputs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Primitive helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    mode: str
    fold: int
    # Epoch where each model hit its peak val AUC
    best_epoch_single: int   # SingleEyeHT — selected by ensemble val AUC
    best_epoch_bilat:  int   # BilateralHT — selected by bilateral val AUC
    # Classic (eye-level eval of SingleEyeHT; n = 2 * ensemble_val_n)
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
    # Ensemble (patient-level eval of same SingleEyeHT)
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
    # Holdout metrics (evaluated at best val epoch; nan if no holdout)
    classic_holdout_auc:  float
    classic_holdout_acc:  float
    ensemble_holdout_auc: float
    ensemble_holdout_acc: float
    bilat_holdout_auc:    float
    bilat_holdout_acc:    float
    holdout_n:            int    # number of holdout bilateral samples
    # Training sample counts
    single_train_n: int
    bilat_train_n:  int
    # Fused head (ensemble + --fused-head; nan / None if --fused-head not used)
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
    fused_holdout_auc:   float        = float("nan")
    fused_holdout_acc:   float        = float("nan")


@dataclass
class FoldArtifacts:
    y_true_classic:  Optional[np.ndarray]
    probs_classic:   Optional[np.ndarray]
    y_true_ensemble: Optional[np.ndarray]
    probs_ensemble:  Optional[np.ndarray]
    y_true_bilat:    Optional[np.ndarray]
    probs_bilat:     Optional[np.ndarray]
    y_true_fused:    Optional[np.ndarray] = None
    probs_fused:     Optional[np.ndarray] = None
