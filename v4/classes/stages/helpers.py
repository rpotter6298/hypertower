"""stages/helpers — shared utilities for stage runners."""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F


def class_weights_from_shell(
    shell, num_classes: int, device, *, enabled: bool = True
) -> torch.Tensor | None:
    """Return inverse-frequency CE weights normalised to mean 1, or None.

    weights[i] = (N_total / num_classes) / N_class_i  → rare classes weighted higher.
    Mean(weights) ≈ 1 so overall loss magnitude is unchanged.

    Classes absent from the shell get weight 1.0 (no division-by-zero).
    """
    if not enabled or shell is None or not shell.entries:
        return None
    counts  = Counter(int(e.label) for e in shell.entries)
    n_total = sum(counts.values())
    weights = []
    for c in range(num_classes):
        n_c = counts.get(c, 0)
        weights.append(1.0 if n_c == 0 else n_total / (num_classes * n_c))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def get_out_dim(name: str, towers: dict, stage_models: dict) -> int:
    if name in towers:
        return towers[name].out_dim
    if name in stage_models:
        return stage_models[name].out_dim
    raise KeyError(f"No out_dim for input {name!r}")


def resolve_input_dims(inputs, towers: dict, stage_models: dict):
    """Return list[int] for list inputs, dict[str, int] for dict inputs."""
    if isinstance(inputs, list):
        return [get_out_dim(n, towers, stage_models) for n in inputs]
    return {k: get_out_dim(v, towers, stage_models) for k, v in inputs.items()}


def encode_embedding(
    name:         str,
    batch:        dict,
    side:         str | None,
    towers:       dict,
    stage_models: dict,
    cfg_stages:   list[dict],
    device,
) -> torch.Tensor:
    """Return the embedding for a named tower or prior frozen fusion stage.

    For bilateral batches, *side* selects which side dict entry to use.
    Fusion stages are re-encoded recursively from their own inputs.
    Runs under torch.no_grad() — only used for frozen passes.
    """
    if name in towers:
        t = batch[name]
        if side is not None and isinstance(t, dict):
            t = t[side]
        with torch.no_grad():
            return towers[name](t.to(device))

    s_cfg  = next(s for s in cfg_stages if s["name"] == name)
    model  = stage_models[name]
    inputs = s_cfg["inputs"]

    if isinstance(inputs, list):
        embeddings = [
            encode_embedding(n, batch, side, towers, stage_models, cfg_stages, device)
            for n in inputs
        ]
        with torch.no_grad():
            return model(embeddings)

    raise NotImplementedError(
        f"Stage {name!r} uses dict inputs and cannot be used as a fusion input."
    )


def phase_for_epoch(epoch: int, warmup_tower: int, warmup_fused: int) -> str:
    if epoch < warmup_tower:
        return "tower_warmup"
    if epoch < warmup_tower + warmup_fused:
        return "fused_warmup"
    return "main"


# ─────────────────────────────────────────────────────────────────────────────
# Head dispatch helpers — opt-in hooks that let new head types (regression,
# ordinal, etc.) plug in without touching the runners.  Heads that don't
# implement these methods fall back to the classification defaults.
# ─────────────────────────────────────────────────────────────────────────────

def head_target_key(head) -> str:
    """Which batch field this head consumes as ground truth (default: 'label')."""
    return getattr(head, "target_key", "label")


def head_compute_loss(
    head, logits: torch.Tensor, batch: dict, y_t: torch.Tensor,
    *, class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute one head's training loss.

    If the head provides `compute_loss(logits, batch)`, that wins — the head
    is responsible for reading its own target from batch and applying whatever
    loss function it wants.  Otherwise we fall back to weighted CE against y_t
    (the standard classification path).
    """
    if hasattr(head, "compute_loss"):
        return head.compute_loss(logits, batch)
    return F.cross_entropy(logits, y_t, weight=class_weights)


def head_to_probs(head, logits: torch.Tensor) -> np.ndarray:
    """Convert raw head outputs to a per-sample numpy array.

    For classification heads, this is the softmax over class logits.
    For regression heads, this is just the raw predicted scalar(s).
    Heads override `to_probs(logits)` to define their own conversion.
    """
    if hasattr(head, "to_probs"):
        return head.to_probs(logits)
    return F.softmax(logits, dim=1).cpu().numpy()


def head_score(head, y_true: np.ndarray, predictions: np.ndarray,
               num_classes: int) -> dict:
    """Compute evaluation metrics for one head.

    Returns a dict with at minimum:
      - 'primary'      : the headline metric value (float)
      - 'primary_name' : how to label it (e.g. 'auc', 'mse')
      - 'n'            : sample count

    Classification heads fall through to the existing score_arrays-based
    metric set.  Regression heads override `score(y_true, predictions)` and
    return their own dict (which may include the standard fields plus extras
    like 'mae', 'r2', 'spearman').
    """
    if hasattr(head, "score"):
        return head.score(y_true, predictions)
    # Default: classification scoring
    from v4.classes.metrics import score_arrays, compute_extended_metrics
    if not y_true.size:
        return {"primary": float("nan"), "primary_name": "auc",
                "auc": float("nan"), "acc": float("nan"), "n": 0}
    acc, auc, n = score_arrays(y_true, predictions, num_classes)
    ext         = compute_extended_metrics(y_true, predictions, num_classes)
    return {
        "primary":      float(auc),
        "primary_name": "auc",
        "auc":          float(auc),
        "acc":          float(acc),
        "n":            int(n),
        **{k: float(v) for k, v in ext.items() if isinstance(v, (int, float))},
    }
