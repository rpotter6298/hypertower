#!/usr/bin/env python3
"""
Post-hoc explainability for a single saved fold.

Phase 1 — MD permutation feature importance (bar chart + CSV).
Phase 2 — GradCAM overlays on all holdout (or val) patients.

Usage:
    python scripts/output_analysis/explainability/explain_fold.py \
        --fold-dir analysis_data/.../binary/single/fold0 \
        [--checkpoint best_single.pt | best_holdout_single.pt] \
        [--split holdout]              # falls back to val if no holdout
        [--image-dir Papila/FundusImages] \
        [--clinical-dir Papila/ClinicalData] \
        [--n-permutations 30] \
        [--seed 0] \
        [--alpha 0.45]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classes.v2.data_bundle import DataBundle
from classes.v2.papila_builders import build_papila_data
from classes.v2.profiles.papila import build_papila_profile
from classes.v2.split_manager import PatientFirstSplitManager
from classes.v2.loader_factory import filter_bilateral_samples, make_loader
from classes.v2.metrics import _score_arrays
from classes.v2.models import SingleEyeHT
from classes.v2.transforms import build_eval_transform

# ---------------------------------------------------------------------------
# Label display helpers
# ---------------------------------------------------------------------------

BINARY_LABELS = {0: "Normal", 1: "Glaucoma"}
MULTICLASS_LABELS = {0: "Normal", 1: "Glaucoma", 2: "Suspect"}


def label_name(label: int, eval_mode: str) -> str:
    mapping = BINARY_LABELS if eval_mode == "binary" else MULTICLASS_LABELS
    return mapping.get(int(label), str(label))


# ---------------------------------------------------------------------------
# GradCAM
# ---------------------------------------------------------------------------


class GradCAM:
    """Minimal GradCAM using forward/backward hooks. No extra dependencies."""

    def __init__(self, target_layer: torch.nn.Module) -> None:
        self._acts: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None
        self._h1 = target_layer.register_forward_hook(self._save_acts)
        self._h2 = target_layer.register_full_backward_hook(self._save_grads)

    def _save_acts(self, _m, _i, output):
        self._acts = output.detach()

    def _save_grads(self, _m, _gi, grad_output):
        self._grads = grad_output[0].detach()

    def compute(
        self,
        img: torch.Tensor,
        meta: torch.Tensor,
        model: torch.nn.Module,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, int]:
        """Return (cam [H,W] in [0,1], predicted_class_index)."""
        model.eval()
        with torch.enable_grad():
            out = model(img, meta)
            pred = int(out.argmax(1).item())
            tc = pred if target_class is None else target_class
            model.zero_grad()
            out[0, tc].backward()

        if self._acts is None or self._grads is None:
            raise RuntimeError("GradCAM hooks did not fire — check target_layer.")

        weights = self._grads.mean(dim=(2, 3), keepdim=True)  # [1,C,1,1]
        cam = F.relu((weights * self._acts).sum(dim=1, keepdim=True))  # [1,1,h,w]
        cam = F.interpolate(cam, img.shape[-2:], mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().cpu().numpy()
        lo, hi = cam_np.min(), cam_np.max()
        cam_np = (cam_np - lo) / (hi - lo + 1e-8)
        return cam_np, pred

    def remove(self) -> None:
        self._h1.remove()
        self._h2.remove()


def get_gradcam_layer(model: SingleEyeHT, backbone: str) -> torch.nn.Module:
    """Return the final spatial feature map layer for GradCAM."""
    bb = model.img_tower.backbone
    key = backbone.lower()
    if key in ("refugelike",) or "resnet" in key:
        return bb.layer4[-1]
    if "efficientnet" in key or "refuge_efficient" in key:
        return bb.features[-1]
    if "densenet" in key or key == "refuge_densenet":
        return bb.features.denseblock4
    if "mobilenet" in key:
        return bb.features[-1]
    if "vgg" in key:
        return bb.features[-1]
    raise ValueError(f"Unknown backbone for GradCAM target layer: {backbone!r}")


def overlay_gradcam(
    original_pil: Image.Image, cam: np.ndarray, alpha: float = 0.45
) -> Image.Image:
    """Blend a jet-coloured GradCAM map onto the original image."""
    cam_u8 = (cam * 255).astype(np.uint8)
    cam_resized = (
        np.array(Image.fromarray(cam_u8).resize(original_pil.size, Image.BILINEAR))
        / 255.0
    )
    colored = (cm.jet(cam_resized)[:, :, :3] * 255).astype(np.uint8)
    return Image.blend(original_pil.convert("RGB"), Image.fromarray(colored), alpha)


# ---------------------------------------------------------------------------
# Feature index map
# ---------------------------------------------------------------------------


def build_feature_index_map(data: DataBundle) -> dict[str, dict]:
    """
    Return a mapping  feature_name → {"value_dims": [...], "missing_dims": [...]}
    that covers every input dimension of the MD tower vector.

    Layout (from DataBundle.vectorize_row):
        [scalar_0..scalar_n-1 | cat_onehot | scalar_missing_0..scalar_missing_n-1]
    """
    n_scalar = len(data.scalar_cols)
    cat_expanded = sum(len(m) for m in data.cat_maps.values())

    feature_map: dict[str, dict] = {}
    idx = 0

    # Scalar features: value_dim + corresponding missing flag
    for i, col in enumerate(data.scalar_cols):
        missing_dim = n_scalar + cat_expanded + i
        feature_map[col] = {"value_dims": [i], "missing_dims": [missing_dim]}
        idx += 1

    # Categorical features: permute the entire one-hot block
    cat_offset = n_scalar
    for col in data.cat_cols:
        n_cats = len(data.cat_maps[col])
        dims = list(range(cat_offset, cat_offset + n_cats))
        feature_map[col] = {"value_dims": dims, "missing_dims": []}
        cat_offset += n_cats

    return feature_map


# ---------------------------------------------------------------------------
# Phase 1 — MD permutation importance
# ---------------------------------------------------------------------------


def run_permutation_importance(
    model: SingleEyeHT,
    loader,
    data: DataBundle,
    num_classes: int,
    device: torch.device,
    n_permutations: int,
    seed: int,
    out_dir: Path,
) -> None:
    print("\n[Phase 1] MD permutation importance ...", flush=True)

    # ---- cache bilateral image embeddings + metadata tensors + labels ----
    img1_feats_list, img2_feats_list = [], []
    md1_list, md2_list, label_list = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            img1 = batch["image_1"].to(device)
            img2 = batch["image_2"].to(device)
            md1 = batch["matrix_1"].to(device)
            md2 = batch["matrix_2"].to(device)
            labels = batch["label_1"]
            img1_feats_list.append(model.img_tower(img1))
            img2_feats_list.append(model.img_tower(img2))
            md1_list.append(md1)
            md2_list.append(md2)
            if isinstance(labels, torch.Tensor):
                label_list.append(labels)
            else:
                label_list.append(torch.tensor(labels, dtype=torch.long))

    img1_feats = torch.cat(img1_feats_list)  # [N, img_dim]
    img2_feats = torch.cat(img2_feats_list)  # [N, img_dim]
    md1_tensor = torch.cat(md1_list)  # [N, feature_dim]
    md2_tensor = torch.cat(md2_list)  # [N, feature_dim]
    y_true = torch.cat(label_list).numpy()
    N = len(y_true)

    if N == 0:
        print("  [Phase 1] No samples — skipping.", flush=True)
        return

    # ---- baseline AUC (patient-level: average OD/OS fused probabilities) ----
    with torch.no_grad():
        md1_feats = model.md_tower(md1_tensor)
        md2_feats = model.md_tower(md2_tensor)
        fused1, _, _ = model.bridge(img1_feats, md1_feats)
        fused2, _, _ = model.bridge(img2_feats, md2_feats)
        probs_baseline = (
            0.5 * (torch.softmax(fused1, dim=1) + torch.softmax(fused2, dim=1))
        ).cpu().numpy()
    _, baseline_auc, _ = _score_arrays(y_true, probs_baseline, num_classes)
    print(f"  Baseline AUC: {baseline_auc:.4f}  (N={N})", flush=True)

    # ---- feature index map ----
    feat_map = build_feature_index_map(data)
    rng = np.random.default_rng(seed)

    results = []
    for feat_name, dims in feat_map.items():
        all_dims = dims["value_dims"] + dims["missing_dims"]
        drops = []
        for _ in range(n_permutations):
            perm1 = md1_tensor.clone()
            perm2 = md2_tensor.clone()
            perm_idx = torch.from_numpy(rng.permutation(N)).to(device)
            # Apply the same donor patient permutation to both eyes to preserve
            # within-patient coherence while breaking feature-label association.
            perm1[:, all_dims] = perm1[perm_idx][:, all_dims]
            perm2[:, all_dims] = perm2[perm_idx][:, all_dims]
            with torch.no_grad():
                md1_p = model.md_tower(perm1)
                md2_p = model.md_tower(perm2)
                fused1_p, _, _ = model.bridge(img1_feats, md1_p)
                fused2_p, _, _ = model.bridge(img2_feats, md2_p)
                probs_p = (
                    0.5
                    * (
                        torch.softmax(fused1_p, dim=1)
                        + torch.softmax(fused2_p, dim=1)
                    )
                ).cpu().numpy()
            _, auc_p, _ = _score_arrays(y_true, probs_p, num_classes)
            drops.append(baseline_auc - auc_p)

        mean_drop = float(np.mean(drops))
        std_drop = float(np.std(drops))
        results.append({"feature": feat_name, "importance": mean_drop, "std": std_drop})
        print(
            f"  {feat_name:30s}  Δ AUC = {mean_drop:+.4f} ± {std_drop:.4f}", flush=True
        )

    results.sort(key=lambda r: r["importance"], reverse=True)

    # ---- total MD ablation (all features permuted simultaneously) ----
    print("  Running total MD ablation ...", flush=True)
    total_drops = []
    for _ in range(n_permutations):
        perm_idx = torch.from_numpy(rng.permutation(N)).to(device)
        perm1_all = md1_tensor[perm_idx]
        perm2_all = md2_tensor[perm_idx]
        with torch.no_grad():
            md1_all = model.md_tower(perm1_all)
            md2_all = model.md_tower(perm2_all)
            f1, _, _ = model.bridge(img1_feats, md1_all)
            f2, _, _ = model.bridge(img2_feats, md2_all)
            probs_all = (
                0.5 * (torch.softmax(f1, dim=1) + torch.softmax(f2, dim=1))
            ).cpu().numpy()
        _, auc_all, _ = _score_arrays(y_true, probs_all, num_classes)
        total_drops.append(baseline_auc - auc_all)
    total_mean = float(np.mean(total_drops))
    total_std = float(np.std(total_drops))
    print(
        f"  Total MD ablation  Δ AUC = {total_mean:+.4f} ± {total_std:.4f}", flush=True
    )

    # ---- Gaussian noise ablation (tests architectural vs informational benefit) ----
    print("  Running Gaussian noise ablation ...", flush=True)
    noise_drops = []
    for _ in range(n_permutations):
        noise1 = torch.randn_like(md1_tensor)
        noise2 = torch.randn_like(md2_tensor)
        with torch.no_grad():
            md1_noise = model.md_tower(noise1)
            md2_noise = model.md_tower(noise2)
            f1, _, _ = model.bridge(img1_feats, md1_noise)
            f2, _, _ = model.bridge(img2_feats, md2_noise)
            probs_noise = (
                0.5 * (torch.softmax(f1, dim=1) + torch.softmax(f2, dim=1))
            ).cpu().numpy()
        _, auc_noise, _ = _score_arrays(y_true, probs_noise, num_classes)
        noise_drops.append(baseline_auc - auc_noise)
    noise_mean = float(np.mean(noise_drops))
    noise_std = float(np.std(noise_drops))
    print(
        f"  Gaussian noise ablation  Δ AUC = {noise_mean:+.4f} ± {noise_std:.4f}", flush=True
    )
    print(
        f"  [interpretation] permutation Δ={total_mean:+.4f}  noise Δ={noise_mean:+.4f}  "
        f"informational gain = {total_mean - noise_mean:+.4f}",
        flush=True,
    )

    # ---- save CSV ----
    import csv

    csv_path = out_dir / "md_permutation_importance.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "importance", "std"])
        writer.writeheader()
        writer.writerows(results)
        writer.writerow({"feature": "TOTAL_MD_ABLATION", "importance": total_mean, "std": total_std})
        writer.writerow({"feature": "GAUSSIAN_NOISE_ABLATION", "importance": noise_mean, "std": noise_std})

    # ---- bar chart ----
    names = [r["feature"] for r in results]
    imps = [r["importance"] for r in results]
    stds = [r["std"] for r in results]
    colors = ["#e05c5c" if v >= 0 else "#5c9ee0" for v in imps]

    fig, ax = plt.subplots(figsize=(9, max(4, (len(names) + 3) * 0.45)))
    y_pos = np.arange(len(names))
    ax.barh(
        y_pos, imps, xerr=stds, color=colors, ecolor="grey", capsize=3, height=0.6
    )
    ax.axhline(len(names) - 0.25, color="grey", linewidth=0.6, linestyle="--")
    # total ablation
    ax.barh(
        len(names) + 0.5, total_mean, xerr=total_std,
        color="#c45ce0" if total_mean >= 0 else "#5c9ee0",
        ecolor="grey", capsize=3, height=0.6,
    )
    # gaussian noise ablation
    ax.barh(
        len(names) + 1.5, noise_mean, xerr=noise_std,
        color="#e08c2a" if noise_mean >= 0 else "#5c9ee0",
        ecolor="grey", capsize=3, height=0.6,
    )
    ax.set_yticks(list(y_pos) + [len(names) + 0.5, len(names) + 1.5])
    ax.set_yticklabels(names + ["ALL MD (permute)", "ALL MD (noise)"], fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean AUC drop (baseline − permuted)", fontsize=10)
    ax.set_title(
        f"MD Tower — Permutation Feature Importance\n"
        f"baseline AUC={baseline_auc:.4f}  N={N}  repeats={n_permutations}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "md_permutation_importance.png", dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_dir / 'md_permutation_importance.png'}", flush=True)


# ---------------------------------------------------------------------------
# Phase 2 — GradCAM overlays
# ---------------------------------------------------------------------------


def run_gradcam(
    model: SingleEyeHT,
    loader,
    data: DataBundle,
    eval_df,
    eval_mode: str,
    backbone: str,
    device: torch.device,
    alpha: float,
    out_dir: Path,
) -> None:
    print("\n[Phase 2] GradCAM overlays ...", flush=True)
    gradcam_dir = out_dir / "gradcam"
    gradcam_dir.mkdir(exist_ok=True)

    target_layer = get_gradcam_layer(model, backbone)
    gcam = GradCAM(target_layer)

    num_classes = model.bridge.classifier_fused[-1].out_features

    overlay_grid_items: list[
        tuple[Image.Image | None, Image.Image | None, str, bool]
    ] = []
    index_rows: list[dict] = []

    model.eval()
    for batch in loader:
        img_od = batch["image_1"].to(device)  # [1, 3, H, W]
        img_os = batch["image_2"].to(device)  # [1, 3, H, W]
        meta_od = batch["matrix_1"].to(device)  # [1, feature_dim]
        meta_os = batch["matrix_2"].to(device)
        lbl_raw = batch["label_1"][0]
        label = int(lbl_raw.item() if isinstance(lbl_raw, torch.Tensor) else lbl_raw)
        pid = batch["id_1"][0]

        # GradCAM for each eye (OD drives the prediction label)
        cam_od, pred = gcam.compute(img_od, meta_od, model)
        cam_os, _ = gcam.compute(img_os, meta_os, model)

        # Confidence of predicted class
        with torch.no_grad():
            out_od = model(img_od, meta_od)
            conf = float(torch.softmax(out_od, dim=1)[0, pred].item())

        # Load original (un-normalised) images from disk
        row_od = eval_df[
            (eval_df["Patient ID"] == int(pid)) & (eval_df["eyeID"] == "OD")
        ]
        row_os = eval_df[
            (eval_df["Patient ID"] == int(pid)) & (eval_df["eyeID"] == "OS")
        ]
        orig_od = (
            Image.open(data.get_image_path(row_od.iloc[0])).convert("RGB")
            if len(row_od)
            else None
        )
        orig_os = (
            Image.open(data.get_image_path(row_os.iloc[0])).convert("RGB")
            if len(row_os)
            else None
        )

        true_name = label_name(label, eval_mode)
        pred_name = label_name(pred, eval_mode)
        correct = label == pred
        title = (
            f"Patient {pid}  |  True: {true_name}  |  Pred: {pred_name}  "
            f"|  conf={conf:.2f}  {'✓' if correct else '✗'}"
        )

        # ---- per-patient 2×2 figure (OD raw | OD overlay / OS raw | OS overlay) ----
        fig, axes = plt.subplots(2, 2, figsize=(10, 9))
        fig.suptitle(
            title, fontsize=11, fontweight="bold", color="green" if correct else "red"
        )

        # Row 0: OD
        if orig_od is not None:
            axes[0, 0].imshow(orig_od)
            axes[0, 0].set_title("OD — original", fontsize=9)
            axes[0, 1].imshow(overlay_gradcam(orig_od, cam_od, alpha))
            axes[0, 1].set_title("OD — GradCAM", fontsize=9)
        else:
            axes[0, 0].set_title("OD — (missing)", fontsize=9)
        axes[0, 0].axis("off")
        axes[0, 1].axis("off")

        # Row 1: OS
        if orig_os is not None:
            axes[1, 0].imshow(orig_os)
            axes[1, 0].set_title("OS — original", fontsize=9)
            axes[1, 1].imshow(overlay_gradcam(orig_os, cam_os, alpha))
            axes[1, 1].set_title("OS — GradCAM", fontsize=9)
        else:
            axes[1, 0].set_title("OS — (missing)", fontsize=9)
        axes[1, 0].axis("off")
        axes[1, 1].axis("off")

        fig.tight_layout()
        out_path = gradcam_dir / f"patient_{pid}_OD_OS.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(
            f"  Patient {pid}: {true_name} → {pred_name} ({conf:.2f})  → {out_path.name}",
            flush=True,
        )

        # Save raw CAM arrays
        np.save(gradcam_dir / f"patient_{pid}_OD_cam.npy", cam_od)
        np.save(gradcam_dir / f"patient_{pid}_OS_cam.npy", cam_os)
        index_rows.append({
            "patient_id":  pid,
            "true_label":  label,
            "true_name":   true_name,
            "pred_label":  pred,
            "pred_name":   pred_name,
            "confidence":  conf,
            "correct":     correct,
        })

        # Accumulate for summary grid
        od_overlay = overlay_gradcam(orig_od, cam_od, alpha) if orig_od else None
        os_overlay = overlay_gradcam(orig_os, cam_os, alpha) if orig_os else None
        short_lbl = f"P{pid}  {true_name[:3]}→{pred_name[:3]} {'✓' if correct else '✗'}"
        overlay_grid_items.append((od_overlay, os_overlay, short_lbl, correct))

    gcam.remove()

    # ---- save index CSV ----
    if index_rows:
        import csv
        idx_path = gradcam_dir / "gradcam_index.csv"
        with idx_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
            writer.writeheader()
            writer.writerows(index_rows)
        print(f"  Index CSV → {idx_path}", flush=True)

    # ---- summary grid: N_patients rows × 2 cols (OD overlay | OS overlay) ----
    n = len(overlay_grid_items)
    if n == 0:
        print("  [Phase 2] No patients to visualise.", flush=True)
        return

    fig, axes = plt.subplots(n, 2, figsize=(8, n * 3.2 + 0.8))
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("GradCAM Summary Grid — all holdout patients", fontsize=12)

    for i, (od_ov, os_ov, lbl, correct) in enumerate(overlay_grid_items):
        color = "green" if correct else "red"
        for j in range(2):
            axes[i, j].axis("off")
        if od_ov is not None:
            axes[i, 0].imshow(od_ov)
        axes[i, 0].set_title(f"{lbl}\nOD", fontsize=7, color=color)
        if os_ov is not None:
            axes[i, 1].imshow(os_ov)
        axes[i, 1].set_title(f"{lbl}\nOS", fontsize=7, color=color)

    fig.tight_layout()
    grid_path = out_dir / "gradcam_summary_grid.png"
    fig.savefig(grid_path, dpi=120)
    plt.close(fig)
    print(f"  Summary grid → {grid_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    ap = argparse.ArgumentParser(
        description="Post-hoc explainability for a saved fold."
    )
    ap.add_argument(
        "--fold-dir",
        type=Path,
        required=True,
        help="Path to fold directory, e.g. analysis_data/.../binary/single/fold0",
    )
    ap.add_argument(
        "--checkpoint",
        default="best_single.pt",
        help="Checkpoint filename inside fold_dir (default: best_single.pt; "
        "use best_holdout_single.pt for holdout-selected model)",
    )
    ap.add_argument(
        "--split",
        choices=["holdout", "val"],
        default="holdout",
        help="Which patient set to analyse (default: holdout, falls back to val)",
    )
    ap.add_argument("--image-dir", default="Papila/FundusImages")
    ap.add_argument("--clinical-dir", default="Papila/ClinicalData")
    ap.add_argument("--label-col", default="Diagnosis")
    ap.add_argument("--cat-cols", nargs="*", default=["Gender", "Phakic/Pseudophakic"])
    ap.add_argument("--fold-seed", type=int, default=42)
    ap.add_argument("--holdout-seed", type=int, default=123)
    ap.add_argument("--holdout-per-class", type=int, default=5)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument(
        "--n-permutations",
        type=int,
        default=30,
        help="Repetitions per feature for permutation importance (default: 30)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="GradCAM overlay opacity (default: 0.45)",
    )
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--no-phase1", action="store_true", help="Skip MD importance")
    ap.add_argument("--no-phase2", action="store_true", help="Skip GradCAM")
    ap.add_argument("--no-phase3", action="store_true", help="Skip fusion event analysis")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Phase 3 — Fusion event analysis
# ---------------------------------------------------------------------------


def run_fusion_event_analysis(
    model: SingleEyeHT,
    loader,
    device: torch.device,
    out_dir: Path,
) -> None:
    print("\n[Phase 3] Fusion event analysis ...", flush=True)

    from classes.v2.models import collect_probs_single_components

    y_true, pf, pi, pm = collect_probs_single_components(
        model, loader, device, aggregate_patient=True
    )
    N = len(y_true)
    if N == 0:
        print("  [Phase 3] No samples — skipping.", flush=True)
        return

    pred_f = pf.argmax(axis=1)
    pred_i = pi.argmax(axis=1)
    pred_m = pm.argmax(axis=1)

    # confidence of the predicted class for each head
    conf_f = np.take_along_axis(pf, pred_f[:, None], axis=1).squeeze(1)
    conf_i = np.take_along_axis(pi, pred_i[:, None], axis=1).squeeze(1)
    conf_m = np.take_along_axis(pm, pred_m[:, None], axis=1).squeeze(1)
    # how much did fusion shift confidence vs the average of the two towers?
    conf_delta = conf_f - 0.5 * (conf_i + conf_m)

    f_ok = pred_f == y_true
    i_ok = pred_i == y_true
    m_ok = pred_m == y_true

    # 6 non-trivial bridge-effect event types
    full_correction = f_ok  & ~i_ok & ~m_ok   # both towers wrong → fused right
    img_assist      = f_ok  & ~i_ok &  m_ok   # img wrong, md right → fused right  (md carried it)
    md_assist       = f_ok  &  i_ok & ~m_ok   # md wrong, img right → fused right  (img carried it)
    full_error      = ~f_ok &  i_ok &  m_ok   # both towers right  → fused wrong
    img_drag        = ~f_ok & ~i_ok &  m_ok   # img wrong, md right → fused wrong  (img dragged it down)
    md_drag         = ~f_ok &  i_ok & ~m_ok   # md wrong, img right → fused wrong  (md dragged it down)
    concordant_ok   =  f_ok &  i_ok &  m_ok
    concordant_bad  = ~f_ok & ~i_ok & ~m_ok

    event_labels  = [
        "full correction\n(both wrong→fused right)",
        "img assist\n(img wrong, md right→right)",
        "md assist\n(md wrong, img right→right)",
        "full error\n(both right→fused wrong)",
        "img drag\n(img wrong, md right→wrong)",
        "md drag\n(md wrong, img right→wrong)",
    ]
    event_masks  = [full_correction, img_assist, md_assist, full_error, img_drag, md_drag]
    event_colors = ["#2ca02c", "#98df8a", "#b5d46e", "#d62728", "#ff9896", "#ffbf9b"]
    event_keys   = ["full_correction", "img_assist", "md_assist",
                    "full_error", "img_drag", "md_drag"]
    counts = [int(m.sum()) for m in event_masks]

    print(f"  N={N}", flush=True)
    for label, count in zip(event_labels, counts):
        print(f"  {label.replace(chr(10), ' '):55s}: {count}", flush=True)
    n_corr, n_err = counts[0], counts[3]
    ratio_str = f"{n_corr}/{n_err}" if n_err > 0 else f"{n_corr}/0"
    print(f"  full correction/error ratio: {ratio_str}", flush=True)
    print(f"  conf_delta  mean={conf_delta.mean():+.4f}  median={np.median(conf_delta):+.4f}",
          flush=True)

    # ---- CSV ----
    import csv
    event_type = np.where(concordant_ok, "concordant_correct",
                 np.where(concordant_bad, "concordant_wrong", "other")).astype(object)
    for mask, key in zip(event_masks, event_keys):
        event_type[mask] = key

    rows = []
    for idx in range(N):
        rows.append({
            "patient_idx":  idx,
            "y_true":       int(y_true[idx]),
            "pred_fused":   int(pred_f[idx]),
            "pred_img":     int(pred_i[idx]),
            "pred_md":      int(pred_m[idx]),
            "conf_fused":   float(conf_f[idx]),
            "conf_img":     float(conf_i[idx]),
            "conf_md":      float(conf_m[idx]),
            "conf_delta":   float(conf_delta[idx]),
            "event_type":   event_type[idx],
        })
    csv_path = out_dir / "fusion_events.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved → {csv_path}", flush=True)

    # ---- plot ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: stacked bar — positive events vs negative events
    pos_counts = counts[:3]
    neg_counts  = counts[3:]
    pos_colors  = event_colors[:3]
    neg_colors  = event_colors[3:]
    for bar_x, bar_counts, bar_colors in ((0, pos_counts, pos_colors),
                                           (1, neg_counts, neg_colors)):
        bot = 0
        for c, col in zip(bar_counts, bar_colors):
            axes[0].bar(bar_x, c, bottom=bot, color=col, width=0.5)
            if c > 0:
                axes[0].text(bar_x, bot + c / 2, str(c), ha="center", va="center",
                             fontsize=9, fontweight="bold")
            bot += c
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["Positive\nevents", "Negative\nevents"])
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Fusion Events  (N={N})")
    patches = [mpatches.Patch(color=c, label=l.replace("\n", " "))
               for c, l in zip(event_colors, event_labels)]
    axes[0].legend(handles=patches, fontsize=6, loc="upper right")

    # Panel 2: conf_delta boxplot per event type (only non-empty)
    box_data   = [conf_delta[m] for m in event_masks if m.sum() > 0]
    box_labels = [l.split("\n")[0] for m, l in zip(event_masks, event_labels) if m.sum() > 0]
    box_cols   = [c for m, c in zip(event_masks, event_colors) if m.sum() > 0]
    if box_data:
        bp = axes[1].boxplot(box_data, patch_artist=True, widths=0.5)
        for patch, color in zip(bp["boxes"], box_cols):
            patch.set_facecolor(color)
        axes[1].set_xticks(range(1, len(box_labels) + 1))
        axes[1].set_xticklabels(box_labels, rotation=35, ha="right", fontsize=7)
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel("conf_delta\n(fused − avg(img, md))")
    axes[1].set_title("Confidence delta by event type")

    # Panel 3: img vs md confidence space, coloured by event type
    for mask, color, label in zip(event_masks, event_colors, event_labels):
        if mask.sum() > 0:
            axes[2].scatter(conf_i[mask], conf_m[mask], c=color,
                            label=label.split("\n")[0], alpha=0.85, s=45, edgecolors="none")
    if concordant_ok.sum() > 0:
        axes[2].scatter(conf_i[concordant_ok], conf_m[concordant_ok],
                        c="lightgrey", alpha=0.4, s=20, edgecolors="none", label="concordant correct")
    if concordant_bad.sum() > 0:
        axes[2].scatter(conf_i[concordant_bad], conf_m[concordant_bad],
                        c="darkgrey", alpha=0.4, s=20, edgecolors="none", label="concordant wrong")
    axes[2].plot([0, 1], [0, 1], "k--", linewidth=0.5, alpha=0.4)
    axes[2].set_xlabel("conf_img")
    axes[2].set_ylabel("conf_md")
    axes[2].set_title("Tower confidence space\ncoloured by fusion event")
    axes[2].legend(fontsize=6, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_dir / "fusion_events.png", dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_dir / 'fusion_events.png'}", flush=True)


def main():
    args = parse_args()
    fold_dir = args.fold_dir.resolve()
    if not fold_dir.is_dir():
        sys.exit(f"[ERROR] fold_dir does not exist: {fold_dir}")

    ckpt_path = fold_dir / args.checkpoint
    if not ckpt_path.exists():
        sys.exit(
            f"[ERROR] Checkpoint not found: {ckpt_path}\n"
            f"  Run training with --save-checkpoints (now the default) to produce checkpoints."
        )

    # ---- read config from summary.json in parent (tower-mode) dir ----
    summary_path = fold_dir.parent / "summary.json"
    if not summary_path.exists():
        sys.exit(f"[ERROR] summary.json not found: {summary_path}")
    summary = json.loads(summary_path.read_text())
    backbone = summary["backbone"]
    eval_mode = summary["eval_mode"]
    tower_mode = summary.get("tower_mode", "single")
    fold_idx = int(fold_dir.name.replace("fold", ""))
    print(
        f"[explain_fold] fold={fold_idx}  backbone={backbone}  eval_mode={eval_mode}  tower_mode={tower_mode}"
    )

    if tower_mode not in ("single", "ensemble"):
        sys.exit(
            f"[ERROR] explain_fold currently supports single/ensemble tower modes, got: {tower_mode!r}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[explain_fold] device={device}  checkpoint={args.checkpoint}")

    # ---- build DataBundle ----
    print("[explain_fold] Loading clinical data ...", flush=True)
    data = build_papila_data(
        image_dir=args.image_dir,
        clinical_dir=args.clinical_dir,
        label_col=args.label_col,
        cat_cols=args.cat_cols,
        n_splits=args.n_splits,
        random_seed=args.fold_seed,
    )
    df_mode = data.df.copy()
    if eval_mode == "binary":
        df_mode = df_mode[df_mode[args.label_col].isin([0, 1])].reset_index(drop=True)
    num_classes = 2 if eval_mode == "binary" else int(df_mode[args.label_col].nunique())

    # ---- reconstruct the exact same split ----
    print("[explain_fold] Reconstructing split ...", flush=True)
    splitter = PatientFirstSplitManager(
        patient_col="Patient ID", label_col=args.label_col
    )
    split_args = SimpleNamespace(
        eval_mode=eval_mode,
        holdout_per_class=args.holdout_per_class,
        holdout_seed=args.holdout_seed,
        n_splits=args.n_splits,
        fold_seed=args.fold_seed,
    )
    clinical_ns = SimpleNamespace(df=df_mode, label_col=args.label_col)
    plans = splitter.build_plans(clinical=clinical_ns, args=split_args, profile=None)
    if fold_idx >= len(plans):
        sys.exit(f"[ERROR] fold_idx={fold_idx} but only {len(plans)} plans built.")
    split = plans[fold_idx]

    if (
        args.split == "holdout"
        and split.holdout is not None
        and not split.holdout.empty
    ):
        eval_df = split.holdout
        split_name = "holdout"
    else:
        if args.split == "holdout":
            print("  [WARN] No holdout set available; falling back to val.", flush=True)
        eval_df = split.val
        split_name = "val"
    print(
        f"  Using {split_name} set: {eval_df['Patient ID'].nunique()} patients",
        flush=True,
    )

    # ---- build loader ----
    profile_patient = build_papila_profile(
        patient_col="Patient ID", label_col=args.label_col, sample_mode="patient"
    )
    samples = filter_bilateral_samples(
        profile_patient.build_samples(df=eval_df, clinical=data)
    )
    if not samples:
        sys.exit("[ERROR] No bilateral samples found in the eval set.")
    loader = make_loader(
        samples,
        profile_patient.slot_descriptors(),
        image_transform=build_eval_transform(backbone),
        image_preprocessor=None,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # ---- load model ----
    print(f"[explain_fold] Loading model from {ckpt_path} ...", flush=True)
    model = SingleEyeHT(
        backbone=backbone,
        freeze_ratio=0.0,
        augment=False,
        clinical_data=data,
        num_classes=num_classes,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # ---- output directory ----
    out_dir = fold_dir / "explainability"
    out_dir.mkdir(exist_ok=True)
    print(f"[explain_fold] Output → {out_dir}", flush=True)

    # ---- Phase 1 ----
    if not args.no_phase1:
        run_permutation_importance(
            model=model,
            loader=loader,
            data=data,
            num_classes=num_classes,
            device=device,
            n_permutations=args.n_permutations,
            seed=args.seed,
            out_dir=out_dir,
        )

    # ---- Phase 2 ----
    if not args.no_phase2:
        run_gradcam(
            model=model,
            loader=loader,
            data=data,
            eval_df=eval_df,
            eval_mode=eval_mode,
            backbone=backbone,
            device=device,
            alpha=args.alpha,
            out_dir=out_dir,
        )

    # ---- Phase 3 ----
    if not args.no_phase3:
        run_fusion_event_analysis(
            model=model,
            loader=loader,
            device=device,
            out_dir=out_dir,
        )

    print("\n[explain_fold] Done.", flush=True)


if __name__ == "__main__":
    main()
