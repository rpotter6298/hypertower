"""Generic explainability primitives — GradCAM + region-of-interest analysis
+ column-shuffling permutation importance.

This module is dataset-agnostic. All PAPILA-specific knowledge (where the disc
contour lives, eye orientation conventions, clinical feature names, etc.) is
supplied by the profile through optional methods consumed via getattr — see
v4.classes.profiles.v4papila.ImageDataView and ClinicalDataView for examples.

Public surface
--------------
GradCAM                 — hooks-based class-activation maps via fwd/bwd hooks
overlay_gradcam         — render a heatmap blended onto a PIL image
cam_region_stats        — quantify CAM attention inside/outside a binary ROI
cam_region_patch        — crop a square patch centered on the ROI centroid
permutation_importance  — per-feature drop-in-score under column shuffling
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import cm
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# GradCAM
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """Minimal GradCAM via forward/backward hooks.

    Architecture-agnostic: the caller supplies a forward_fn that returns a
    (1, num_classes) logits tensor, so this class doesn't need to know whether
    the model takes (img,), (img, meta), or composed tower/bridge/head inputs.

    Usage
    -----
        cam_module = GradCAM(target_layer)
        cam, pred = cam_module.compute(
            forward_fn=lambda: head(bridge([tower_img(x), tower_cd(c)])),
            output_shape=(224, 224),
        )
        cam_module.remove()
    """

    def __init__(self, target_layer: nn.Module) -> None:
        self._acts: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None
        self._h1 = target_layer.register_forward_hook(self._save_acts)
        self._h2 = target_layer.register_full_backward_hook(self._save_grads)

    def _save_acts(self, _module, _inputs, output):
        self._acts = output.detach()

    def _save_grads(self, _module, _grad_in, grad_out):
        self._grads = grad_out[0].detach()

    def compute(
        self,
        forward_fn: Callable[[], torch.Tensor],
        *,
        output_shape: tuple[int, int],
        target_class: int | None = None,
    ) -> tuple[np.ndarray, int]:
        """Run forward via forward_fn, backward against target_class, return CAM.

        Parameters
        ----------
        forward_fn
            Zero-arg callable returning a (1, C) logits tensor. Caller is
            responsible for placing model in eval mode and zeroing grads.
        output_shape
            (H, W) at which the CAM is bilinearly upsampled. Typically the
            eval-crop image size, e.g. (224, 224).
        target_class
            Class index to backprop against. If None, uses the predicted class.

        Returns
        -------
        (cam, pred)
            cam   : float32 ndarray of shape output_shape, normalised to [0, 1]
            pred  : int, the predicted class index
        """
        with torch.enable_grad():
            out = forward_fn()
            pred = int(out.argmax(1).item())
            tc = pred if target_class is None else target_class
            out[0, tc].backward()

        assert self._grads is not None and self._acts is not None, (
            "GradCAM hooks did not fire — check that target_layer is in the "
            "forward path of forward_fn()."
        )
        weights = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, output_shape, mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().cpu().numpy()
        lo, hi = cam_np.min(), cam_np.max()
        return ((cam_np - lo) / (hi - lo + 1e-8)).astype(np.float32), pred

    def remove(self) -> None:
        """Detach the forward/backward hooks. Call when done with this instance."""
        self._h1.remove()
        self._h2.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def overlay_gradcam(pil: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a jet-colored heatmap of `cam` onto `pil` at alpha blending weight."""
    cam_u8 = (np.clip(cam, 0.0, 1.0) * 255).astype(np.uint8)
    cam_r = np.array(Image.fromarray(cam_u8).resize(pil.size, Image.BILINEAR)) / 255.0
    colored = (cm.jet(cam_r)[:, :, :3] * 255).astype(np.uint8)
    return Image.blend(pil.convert("RGB"), Image.fromarray(colored), alpha)


# ─────────────────────────────────────────────────────────────────────────────
# Region-of-interest analysis (generic — no PAPILA knowledge)
# ─────────────────────────────────────────────────────────────────────────────

def cam_region_stats(cam: np.ndarray, roi_mask: np.ndarray) -> dict[str, float]:
    """Quantify CAM attention restricted to / outside a binary ROI mask.

    The CAM and the mask must be at the same resolution.

    Returns
    -------
    dict with:
      inside_mean      : mean CAM intensity inside the ROI
      outside_mean     : mean CAM intensity outside the ROI
      inside_fraction  : sum(CAM inside) / sum(CAM total); fraction of total
                         attention that falls inside the ROI
      roi_area_frac    : ROI area / total area; how much of the image the ROI
                         covers — useful for normalising inside_fraction
    """
    if cam.shape != roi_mask.shape:
        raise ValueError(
            f"cam shape {cam.shape} != roi_mask shape {roi_mask.shape}"
        )
    mask = roi_mask.astype(bool)
    cam_sum = float(cam.sum())
    inside_sum = float(cam[mask].sum()) if mask.any() else 0.0
    outside_sum = cam_sum - inside_sum
    n_inside = int(mask.sum())
    n_outside = int(mask.size - n_inside)
    return {
        "inside_mean": inside_sum / max(1, n_inside),
        "outside_mean": outside_sum / max(1, n_outside),
        "inside_fraction": inside_sum / cam_sum if cam_sum > 0 else 0.0,
        "roi_area_frac": n_inside / mask.size,
    }


def cam_region_patch(
    cam: np.ndarray,
    roi_mask: np.ndarray,
    *,
    span: float = 5.0,
    patch_size: int = 96,
) -> tuple[np.ndarray | None, float | None]:
    """Crop a square patch centered on the ROI centroid.

    The crop side is `span × roi_radius` (in CAM pixels), where roi_radius is
    derived from the mask area assuming a circular ROI. The output is resized
    to (patch_size, patch_size).

    Returns
    -------
    (patch, roi_radius_in_patch_pixels)
        patch : float32 ndarray of shape (patch_size, patch_size), values in
                [0, 1]. Padded with zeros if the centered crop window extends
                outside the CAM.
        roi_radius_in_patch_pixels : float, the ROI radius re-expressed in
                output-patch pixels (useful for drawing ROI circles in figures).
    Returns (None, None) if the mask is empty.
    """
    mask = roi_mask.astype(bool)
    if not mask.any():
        return None, None

    ys, xs = np.where(mask)
    cy, cx = ys.mean(), xs.mean()
    roi_r = float(np.sqrt(mask.sum() / np.pi))
    half = max(1, int(round(span * roi_r / 2)))
    h, w = cam.shape

    y0, y1 = int(round(cy)) - half, int(round(cy)) + half
    x0, x1 = int(round(cx)) - half, int(round(cx)) + half
    pt = max(0, -y0)
    pb = max(0, y1 - h)
    pl = max(0, -x0)
    pr = max(0, x1 - w)
    cam_pad = np.pad(cam, ((pt, pb), (pl, pr)), constant_values=0.0)
    window = cam_pad[y0 + pt: y1 + pt, x0 + pl: x1 + pl]

    patch = np.array(
        Image.fromarray((np.clip(window, 0, 1) * 255).astype(np.uint8))
             .resize((patch_size, patch_size), Image.BILINEAR)
    ) / 255.0
    roi_r_out = patch_size * roi_r / (2 * half)
    return patch.astype(np.float32), roi_r_out


# ─────────────────────────────────────────────────────────────────────────────
# Permutation importance (column-shuffling)
# ─────────────────────────────────────────────────────────────────────────────

def permutation_importance(
    *,
    score_fn: Callable[[np.ndarray], float],
    X:        np.ndarray,
    groups:   Sequence[Sequence[int]] | None = None,
    n_permutations: int = 30,
    seed:     int = 0,
    feature_names: Sequence[str] | None = None,
) -> dict:
    """Per-feature drop-in-score from column shuffling.

    Generic and feature-agnostic — the caller wires the model (or any
    arbitrary scoring pipeline) into ``score_fn``. Higher score = better;
    drop = baseline_score − score(X_with_column_permuted).

    Parameters
    ----------
    score_fn
        Callable taking the input matrix ``X`` (shape (N, F)) and returning
        a scalar metric. Caller is responsible for any constants the metric
        depends on (e.g. labels, held-out image embeddings).
    X
        Float ndarray of shape (N, F). The base matrix to permute.
    groups
        Optional list of column-index groups. If provided, each group is
        permuted together (useful when several model dims encode one
        conceptual feature, e.g. one-hot encodings of a categorical column).
        Defaults to one group per column.
    n_permutations
        Number of independent shuffles per group; mean and std of the drop
        are reported across these.
    seed
        Seed for the shuffling RNG.
    feature_names
        Optional human-readable names, one per group. If not provided, names
        default to f"group_{i}" or f"col_{j}" when groups is None.

    Returns
    -------
    dict with keys:
        baseline       : float, the score on unmodified X
        mean_drop      : ndarray of shape (G,), mean (baseline − shuffled) per group
        std_drop       : ndarray of shape (G,), std of per-permutation drops
        feature_names  : list[str] of length G
        groups         : list[list[int]] of resolved column groups
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (N, F); got shape {X.shape}")
    n, f_dim = X.shape

    if groups is None:
        resolved_groups: list[list[int]] = [[j] for j in range(f_dim)]
    else:
        resolved_groups = [list(g) for g in groups]
        all_cols = [c for g in resolved_groups for c in g]
        if any(c < 0 or c >= f_dim for c in all_cols):
            raise ValueError(
                f"groups contain column indices outside [0, {f_dim})"
            )

    n_groups = len(resolved_groups)
    if feature_names is None:
        names = (
            [f"col_{g[0]}" for g in resolved_groups] if groups is None
            else [f"group_{i}" for i in range(n_groups)]
        )
    else:
        names = list(feature_names)
        if len(names) != n_groups:
            raise ValueError(
                f"feature_names has length {len(names)} but there are "
                f"{n_groups} groups"
            )

    rng = np.random.default_rng(seed)
    baseline = float(score_fn(X))
    drops = np.empty((n_groups, n_permutations), dtype=float)
    for gi, cols in enumerate(resolved_groups):
        for p in range(n_permutations):
            X_perm = X.copy()
            perm = rng.permutation(n)
            X_perm[:, cols] = X_perm[perm][:, cols]
            drops[gi, p] = baseline - float(score_fn(X_perm))

    return {
        "baseline": baseline,
        "mean_drop": drops.mean(axis=1),
        "std_drop": drops.std(axis=1),
        "feature_names": names,
        "groups": resolved_groups,
    }
