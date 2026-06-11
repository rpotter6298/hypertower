"""regression — RegressionHead for continuous-target tasks.

Self-contained head: implements the four opt-in hooks expected by the v4
stage runners (`target_key`, `compute_loss`, `to_probs`, `score`), so no
runner-side classification logic interferes.

Usage in a stage config:

    {
      "name":       "nt_head",
      "type":       "head",
      "input":      "nt",
      "train_with": "nt",
      "module":     "v4.classes.heads.regression",
      "class":      "RegressionHead",
      "args": {
        "target_key": "vf_md",
        "dropout":    0.3,
        "loss":       "mse"
      }
    }
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


_LOSS_FNS = {
    "mse":   F.mse_loss,
    "l1":    F.l1_loss,
    "huber": F.huber_loss,
}


class RegressionHead(nn.Module):
    """Minimal regression head: ReLU → Dropout → Linear(in_dim → 1).

    Parameters
    ----------
    in_dim      : input embedding dimension (set by the stage runner)
    num_classes : ignored — present only so the runner's `h_cls(in_dim, num_classes, **args)`
                  invocation keeps working.  The head always outputs 1 scalar.
    dropout     : dropout fraction before the final linear
    target_key  : which batch field to read as the ground-truth value
                  (must match a field populated by the dataset; default 'vf_md')
    loss        : 'mse' | 'l1' | 'huber'
    """

    def __init__(
        self,
        in_dim:     int,
        num_classes: int = 1,
        dropout:     float = 0.3,
        target_key:  str   = "vf_md",
        loss:        str   = "mse",
    ):
        super().__init__()
        del num_classes  # explicitly ignored; always single-output
        self.target_key = target_key
        if loss not in _LOSS_FNS:
            raise ValueError(f"unknown loss {loss!r}; choose from {list(_LOSS_FNS)}")
        self._loss_name = loss
        self._loss_fn   = _LOSS_FNS[loss]
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, 1),
        )

    # ── nn.Module ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)   # (B,) — per-sample predicted scalar

    # ── opt-in hooks consumed by the stage runners ─────────────────────────────

    def compute_loss(self, logits: torch.Tensor, batch: dict) -> torch.Tensor:
        """Run regression loss against this head's target field in the batch."""
        target = batch[self.target_key]
        if not torch.is_tensor(target):
            target = torch.as_tensor(target)
        target = target.float().to(logits.device)
        # logits is (B,), target is (B,) — match shapes for any loss fn
        return self._loss_fn(logits, target)

    def to_probs(self, logits: torch.Tensor) -> np.ndarray:
        """For regression there is no 'probability' — return raw predictions."""
        return logits.detach().cpu().numpy()

    def score(self, y_true: np.ndarray, predictions: np.ndarray) -> dict:
        """Compute regression metrics. Returns dict with 'primary' = neg-MSE.

        We report negative MSE as 'primary' so that 'higher is better' matches
        the convention of AUC — easier for monitoring code that expects to
        maximise the primary metric.  MSE itself is also stored separately.
        """
        y    = np.asarray(y_true,    dtype=np.float32).ravel()
        pred = np.asarray(predictions, dtype=np.float32).ravel()
        n    = y.size
        if n == 0:
            return {"primary":      float("nan"),
                    "primary_name": "neg_mse",
                    "neg_mse":      float("nan"),
                    "mse":          float("nan"),
                    "mae":          float("nan"),
                    "r2":           float("nan"),
                    "spearman":     float("nan"),
                    "n":            0}

        diff = pred - y
        mse  = float(np.mean(diff ** 2))
        mae  = float(np.mean(np.abs(diff)))
        # R²: 1 - SSres / SStot; falls back to NaN if SStot = 0
        ss_res = float(np.sum(diff ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2     = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
        # Spearman rank correlation (no scipy dependency — manual rank-Pearson)
        spearman = _spearmanr(y, pred)
        return {
            "primary":      -mse,          # neg-MSE so 'higher is better'
            "primary_name": "neg_mse",
            "neg_mse":      -mse,          # mirror under its primary_name (matches AUC pattern)
            "mse":          mse,
            "mae":          mae,
            "r2":           r2,
            "spearman":     spearman,
            "n":            int(n),
        }


def _spearmanr(a: np.ndarray, b: np.ndarray) -> float:
    """Simple Spearman correlation — Pearson on ranks; handles ties via average ranks."""
    if a.size < 2:
        return float("nan")
    ra = _rankdata(a)
    rb = _rankdata(b)
    ra -= ra.mean(); rb -= rb.mean()
    den = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    if den < 1e-12:
        return float("nan")
    return float((ra * rb).sum() / den)


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Assign average ranks; equivalent to scipy.stats.rankdata."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    # Average ties
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i:j + 1]].mean())
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks
