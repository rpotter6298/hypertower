"""
GradCAM analysis for Phase 5 — logit_mlp_head checkpointed run.

Produces (all in figures/explainability/gradcam/):
  mean_cam_normal.png       — average heatmap across all normal test eyes
  mean_cam_glaucoma.png     — average heatmap across all glaucoma test eyes
  mean_cam_comparison.png   — side-by-side normal vs glaucoma mean CAMs
  overlay_grid_normal.png   — grid of individual overlays (normal eyes)
  overlay_grid_glaucoma.png — grid of individual overlays (glaucoma eyes)

Checkpoints loaded from:
  v3/results/phase5/logit_mlp_head_ckpt/rep00/binary/ensemble/fold{0..4}/best_single.pt

Usage:
    python -m v3.scripts.output_analysis.explainability.gradcam_phase5
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

REPO_ROOT     = Path(__file__).resolve().parents[4]
CKPT_RUN      = REPO_ROOT / "v3" / "results" / "phase5" / "logit_mlp_head_ckpt"
FIGURES_ROOT  = REPO_ROOT / "v3" / "figures" / "explainability" / "gradcam"
CLINICAL_DIR  = REPO_ROOT / "Papila" / "ClinicalData"
IMAGE_DIR     = REPO_ROOT / "Papila" / "FundusImages"
CONTOUR_DIR   = REPO_ROOT / "Papila" / "ExpertsSegmentations" / "Contours"

DISC_SPAN    = 5    # patch side = DISC_SPAN × disc diameter
PATCH_SIZE   = 96   # output thumbnail pixels

# Model hyperparameters (inferred from checkpoint weight shapes)
BACKBONE     = "resnet50"
NUM_CLASSES  = 2
CD_HIDDEN    = 128
FUSION_DIM   = 256

LABEL_NAMES  = {0: "Normal", 1: "Glaucoma"}


# ── GradCAM ──────────────────────────────────────────────────────────────────

class GradCAM:
    """Minimal GradCAM via forward/backward hooks."""

    def __init__(self, target_layer: torch.nn.Module) -> None:
        self._acts = None
        self._grads = None
        self._h1 = target_layer.register_forward_hook(self._save_acts)
        self._h2 = target_layer.register_full_backward_hook(self._save_grads)

    def _save_acts(self, _m, _i, output):
        self._acts = output.detach()

    def _save_grads(self, _m, _gi, grad_output):
        self._grads = grad_output[0].detach()

    def compute(self, img: torch.Tensor, meta: torch.Tensor,
                model: torch.nn.Module, target_class: int | None = None) -> tuple[np.ndarray, int]:
        """Return (cam [H,W] normalised 0-1, predicted_class)."""
        model.eval()
        with torch.enable_grad():
            out = model(img, meta)
            pred = int(out.argmax(1).item())
            tc = pred if target_class is None else target_class
            model.zero_grad()
            out[0, tc].backward()

        weights = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, img.shape[-2:], mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().cpu().numpy()
        lo, hi = cam_np.min(), cam_np.max()
        return (cam_np - lo) / (hi - lo + 1e-8), pred

    def remove(self) -> None:
        self._h1.remove(); self._h2.remove()


def overlay_gradcam(pil: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    cam_u8 = (cam * 255).astype(np.uint8)
    cam_r = np.array(Image.fromarray(cam_u8).resize(pil.size, Image.BILINEAR)) / 255.0
    colored = (cm.jet(cam_r)[:, :, :3] * 255).astype(np.uint8)
    return Image.blend(pil.convert("RGB"), Image.fromarray(colored), alpha)


# ── Disc-centred attention helpers ────────────────────────────────────────────

def _disc_contour_path(pid: int, eye: str, expert: int = 1) -> Path:
    return CONTOUR_DIR / f"RET{pid:03d}{eye}_disc_exp{expert}.txt"


def _load_disc_mask(pid: int, eye: str, cam_h: int, cam_w: int) -> np.ndarray | None:
    """Load expert disc contour, polygon-fill, resize to (cam_h, cam_w)."""
    from PIL import ImageDraw as _ID
    p = _disc_contour_path(pid, eye)
    if not p.exists():
        return None
    try:
        arr = np.loadtxt(str(p), dtype=np.float32)
    except Exception:
        return None
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    if arr.shape[0] < 3:
        return None
    # Get original image size
    img_path = get_image_path(pid, eye)
    try:
        with Image.open(img_path) as im:
            orig_w, orig_h = im.size
    except Exception:
        return None
    canvas = Image.new("L", (orig_w, orig_h), 0)
    _ID.Draw(canvas).polygon([tuple(pt) for pt in arr[:, :2]], fill=1)
    return np.array(canvas.resize((cam_w, cam_h), Image.NEAREST), dtype=bool)


def _disc_centred_patch(cam: np.ndarray, disc_mask: np.ndarray,
                        span: int = DISC_SPAN, out: int = PATCH_SIZE
                        ) -> tuple[np.ndarray | None, float | None]:
    """Translate+scale cam so disc centroid is centred; return (patch, disc_r_out)."""
    if disc_mask is None or disc_mask.sum() == 0:
        return None, None
    ys, xs = np.where(disc_mask)
    cy, cx = ys.mean(), xs.mean()
    disc_r = float(np.sqrt(disc_mask.sum() / np.pi))
    half   = max(1, int(round(span * disc_r / 2)))
    h, w   = cam.shape
    y0, y1 = int(round(cy)) - half, int(round(cy)) + half
    x0, x1 = int(round(cx)) - half, int(round(cx)) + half
    pt = max(0, -y0); pb = max(0, y1 - h)
    pl = max(0, -x0); pr = max(0, x1 - w)
    cam_pad = np.pad(cam, ((pt, pb), (pl, pr)), constant_values=0.0)
    patch   = cam_pad[y0 + pt: y1 + pt, x0 + pl: x1 + pl]
    patch_out = np.array(
        Image.fromarray((np.clip(patch, 0, 1) * 255).astype(np.uint8))
             .resize((out, out), Image.BILINEAR)
    ) / 255.0
    disc_r_out = out * disc_r / (2 * half)
    return patch_out.astype(np.float32), disc_r_out


def make_disc_attention_detail(
    mean_patches: dict,
    stats_rows: list[dict],
    out_path: Path,
) -> None:  # noqa: C901
    """
    2-row (Normal / Glaucoma) × 3-col (correct cam | incorrect cam | disc_frac strip).

    mean_patches: {(cls_name, split): (mean_patch_array, mean_disc_r, count)}
    stats_rows:   list of {true_name, correct, disc_frac} dicts (floats only, no arrays)
    """
    import pandas as pd
    from matplotlib.patches import Circle

    classes     = ["Normal", "Glaucoma"]
    splits      = ["correct", "incorrect"]
    corr_colors = {"correct": "steelblue", "incorrect": "tomato"}
    stats       = pd.DataFrame(stats_rows)

    # 2 rows (Normal / Glaucoma) × 3 cols (correct cam | incorrect cam | disc_frac strip)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8),
                             gridspec_kw={"width_ratios": [1, 1, 0.75]})
    fig.patch.set_facecolor("#f4f4f4")

    rng = np.random.default_rng(42)

    for ri, cls in enumerate(classes):
        # Col 0 & 1: correct / incorrect mean CAMs
        for ci, split in enumerate(splits):
            ax  = axes[ri, ci]
            ax.set_facecolor("#222")
            key = (cls, split)
            if key in mean_patches:
                mp, disc_r_out, count = mean_patches[key]
                ax.imshow(mp, cmap="jet", vmin=0, vmax=1, origin="upper",
                          extent=[0, PATCH_SIZE, PATCH_SIZE, 0])
                cx = cy = PATCH_SIZE / 2
                ax.add_patch(Circle((cx, cy), disc_r_out,
                                    fill=False, edgecolor="white",
                                    linewidth=2, linestyle="--"))
                ax.set_title(f"{split.capitalize()}  (N={count})", fontsize=9)
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="grey")
                ax.set_title(split.capitalize(), fontsize=9)
            ax.axis("off")

        # Row label on leftmost column
        axes[ri, 0].set_ylabel(cls, fontsize=11, fontweight="bold", labelpad=8)

        # Col 2: disc_frac strip
        ax = axes[ri, 2]
        ax.set_facecolor("#f4f4f4")
        sub = stats[stats["true_name"] == cls].dropna(subset=["disc_frac"])
        for xi, split in enumerate(splits):
            pts = sub[sub["correct"] == (split == "correct")]["disc_frac"].values
            if len(pts) == 0:
                continue
            color  = corr_colors[split]
            jitter = rng.uniform(-0.18, 0.18, size=len(pts))
            ax.scatter(xi + jitter, pts, color=color, alpha=0.7, s=28, edgecolors="none")
            ax.hlines(pts.mean(), xi - 0.28, xi + 0.28, colors=color, linewidth=2.5, zorder=5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Correct", "Incorrect"], fontsize=9)
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylim(0, 1)
        ax.set_title("Disc fraction", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        if ri == 0:
            ax.set_ylabel("Attention mass inside GT disc", fontsize=9)

    fig.suptitle(
        "Disc-centred GradCAM attention  |  dashed circle = GT disc boundary\n"
        "Phase 5, logit_mlp_head, fold 0–4",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Model loading ─────────────────────────────────────────────────────────────

def build_model(ckpt_path: Path, device: torch.device):
    """Reconstruct SingleEyeHT from checkpoint and load weights."""
    from types import SimpleNamespace
    from v3.classes.models import SingleEyeHT
    sd = torch.load(ckpt_path, map_location="cpu")
    # ClinicalTower only reads clinical_data.feature_dim at init time
    cd_in = sd["cd_tower.block0.0.weight"].shape[1]
    clinical_shim = SimpleNamespace(feature_dim=cd_in)
    model = SingleEyeHT(
        backbone=BACKBONE,
        freeze_ratio=0.0,
        augment=False,
        clinical_data=clinical_shim,
        num_classes=NUM_CLASSES,
        cd_hidden_dim=CD_HIDDEN,
        fusion_dim=FUSION_DIM,
    )
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


# ── Data helpers ──────────────────────────────────────────────────────────────

def build_data_bundle():
    """Build the PAPILA DataBundle matching the checkpointed run's feature config."""
    from v3.classes.papila_builders import build_papila_data
    import torch as _t
    # Auto-detect cd_in from the majority of checkpoints (excludes stale reps).
    import collections as _col
    all_ckpts = list(CKPT_RUN.glob("rep*/binary/ensemble/fold*/best_single.pt"))
    if all_ckpts:
        counts = _col.Counter(
            _t.load(c, map_location="cpu")["cd_tower.block0.0.weight"].shape[1]
            for c in all_ckpts
        )
        cd_in = counts.most_common(1)[0][0]
    else:
        cd_in = 25
    drop_raw = cd_in <= 21
    excl = ["Axial_Length"] if cd_in in (21, 23) else []
    return build_papila_data(
        image_dir=str(IMAGE_DIR),
        clinical_dir=str(CLINICAL_DIR),
        label_col="Diagnosis",
        cat_cols=["Gender", "Phakic/Pseudophakic"],
        iop_corr_method="ratio",
        iop_drop_raw=drop_raw,
        exclude_cols=excl,
    )


def get_image_path(pid: int, eye: str) -> Path:
    return IMAGE_DIR / f"RET{pid:03d}{eye}.jpg"


def build_meta_vector(row, data) -> torch.Tensor:
    """Build the training-compatible feature vector via DataBundle.vectorize_row."""
    vec = data.vectorize_row(row)
    return torch.tensor(vec, dtype=torch.float32).unsqueeze(0)


# ── Eval transform ────────────────────────────────────────────────────────────

def get_eval_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])


# ── Main loop ─────────────────────────────────────────────────────────────────

def _discover_checkpoints(ckpt_run: Path) -> list[tuple[int, int, Path]]:
    """
    Scan ckpt_run for all available best_single.pt files.
    Skips checkpoints whose cd_in doesn't match the majority (to exclude stale reps).
    Returns sorted list of (rep_idx, fold_idx, ckpt_path).
    """
    import collections
    candidates = []
    for rep_dir in sorted(ckpt_run.glob("rep*")):
        try:
            rep_idx = int(rep_dir.name.replace("rep", ""))
        except ValueError:
            continue
        for fold_dir in sorted((rep_dir / "binary" / "ensemble").glob("fold[0-9]")):
            ckpt = fold_dir / "best_single.pt"
            if ckpt.exists():
                fold_idx = int(fold_dir.name.replace("fold", ""))
                cd_in = torch.load(ckpt, map_location="cpu")[
                    "cd_tower.block0.0.weight"
                ].shape[1]
                candidates.append((rep_idx, fold_idx, ckpt, cd_in))

    if not candidates:
        return []

    # Use the majority cd_in so stale reps are automatically excluded
    counts = collections.Counter(c[3] for c in candidates)
    target_cd_in = counts.most_common(1)[0][0]
    skipped = sum(1 for c in candidates if c[3] != target_cd_in)
    if skipped:
        print(f"  [discover] skipping {skipped} checkpoint(s) with cd_in≠{target_cd_in}")

    return [(rep, fold, ckpt) for rep, fold, ckpt, cd in candidates if cd == target_cd_in]


def run(n_grid: int = 16, alpha: float = 0.45, target_class: int | None = None):
    """
    Loop over all available checkpoints in the run directory (all reps × folds).
    Aggregate CAMs per class, collect overlay grids.
    """
    import pandas as pd
    from v3.scripts.output_analysis.explainability.fold_patient_ids import (
        get_test_patient_ids,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Building DataBundle ...")
    data = build_data_bundle()
    clinical = data.df
    print(f"  feature_dim={data.feature_dim}  rows={len(clinical)}")
    transform = get_eval_transform()

    checkpoints = _discover_checkpoints(CKPT_RUN)
    print(f"Found {len(checkpoints)} checkpoint(s) across "
          f"{len(set(r for r,f,_ in checkpoints))} rep(s)")

    if not checkpoints:
        print("No checkpoints found — run with --save-checkpoints first.")
        return

    # ── Incremental accumulators (no full-res arrays kept after each eye) ────────
    # Mean CAM per class: running sum
    cam_sum   = {0: None, 1: None}
    cam_count = {0: 0,    1: 0}

    # Overlay grid: keep at most n_grid PIL images per class (capped)
    overlay_items = {0: [], 1: []}

    # Disc-attention detail: running sum of disc patches (not list of arrays)
    disc_patch_sum   = {}   # (cls_name, split) → np.ndarray sum
    disc_patch_count = {}   # (cls_name, split) → int
    disc_radius_sum  = {}   # (cls_name, split) → float sum
    disc_stats_rows  = []   # floats only — no arrays

    ckpt_bar = tqdm(checkpoints, desc="Folds", unit="fold")
    for rep_idx, fold_idx, ckpt_path in ckpt_bar:
        ckpt_bar.set_postfix(rep=rep_idx, fold=fold_idx)
        model = build_model(ckpt_path, device)

        # GradCAM target: last ResNet block
        target_layer = model.img_tower.backbone.layer4[-1]
        gcam = GradCAM(target_layer)

        pids = get_test_patient_ids(rep_idx, fold_idx, clinical_dir=CLINICAL_DIR)

        for pid in tqdm(pids, desc=f"  rep{rep_idx:02d}/fold{fold_idx}", leave=False, unit="pt"):
            for eye in ("OD", "OS"):
                img_path = get_image_path(pid, eye)
                if not img_path.exists():
                    continue

                row = clinical[
                    (clinical["Patient ID"] == pid) & (clinical["eyeID"] == eye)
                ]
                if len(row) == 0:
                    continue
                row = row.iloc[0]
                label = int(row["Diagnosis"])

                pil_orig = Image.open(img_path).convert("RGB")
                img_t = transform(pil_orig).unsqueeze(0).to(device)
                meta_t = build_meta_vector(row, data).to(device)

                cam_np, pred = gcam.compute(img_t, meta_t, model,
                                            target_class=target_class)

                # Running mean CAM
                if cam_sum[label] is None:
                    cam_sum[label] = cam_np.copy()
                else:
                    cam_sum[label] += cam_np
                cam_count[label] += 1

                # Overlay grid — only keep up to n_grid per class
                if len(overlay_items[label]) < n_grid:
                    ov = overlay_gradcam(pil_orig, cam_np, alpha=alpha)
                    overlay_items[label].append((ov, pid, eye, pred))

                # Disc-attention: extract patch now, accumulate into running sum
                h, w   = cam_np.shape
                disc_mask = _load_disc_mask(pid, eye, h, w)
                disc_frac = None
                if disc_mask is not None and disc_mask.sum() > 0:
                    disc_frac = float(cam_np[disc_mask].sum() / (cam_np.sum() + 1e-8))

                cls_name = LABEL_NAMES[label]
                split    = "correct" if (pred == label) else "incorrect"
                key      = (cls_name, split)

                patch, disc_r_out = _disc_centred_patch(cam_np, disc_mask)
                if patch is not None:
                    if key not in disc_patch_sum:
                        disc_patch_sum[key]   = patch.copy()
                        disc_patch_count[key] = 1
                        disc_radius_sum[key]  = disc_r_out
                    else:
                        disc_patch_sum[key]   += patch
                        disc_patch_count[key] += 1
                        disc_radius_sum[key]  += disc_r_out

                disc_stats_rows.append({
                    "true_name": cls_name,
                    "correct":   pred == label,
                    "disc_frac": disc_frac,
                })

                # Release per-eye tensors immediately
                del img_t, meta_t, cam_np
                if disc_mask is not None:
                    del disc_mask

        gcam.remove()
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Build mean_patches dict for disc detail plot
    mean_patches = {
        key: (
            disc_patch_sum[key] / disc_patch_count[key],
            disc_radius_sum[key] / disc_patch_count[key],
            disc_patch_count[key],
        )
        for key in disc_patch_sum
    }

    # ── Save outputs ─────────────────────────────────────────────────────────
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    for cls in [0, 1]:
        if cam_count[cls] == 0:
            continue
        mean_cam = cam_sum[cls] / cam_count[cls]
        lo, hi = mean_cam.min(), mean_cam.max()
        mean_cam = (mean_cam - lo) / (hi - lo + 1e-8)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(mean_cam, cmap="jet", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(f"Mean GradCAM — {LABEL_NAMES[cls]}\n(n={cam_count[cls]} eyes, fold 0–4)",
                     fontsize=11, fontweight="bold")
        plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
        out = FIGURES_ROOT / f"mean_cam_{LABEL_NAMES[cls].lower()}.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")

    # Side-by-side comparison
    if cam_count[0] > 0 and cam_count[1] > 0:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle("Mean GradCAM — Normal vs Glaucoma (Phase 5, fold 0–4)",
                     fontsize=12, fontweight="bold")
        for ax, cls in zip(axes, [0, 1]):
            mean_cam = cam_sum[cls] / cam_count[cls]
            lo, hi = mean_cam.min(), mean_cam.max()
            mean_cam = (mean_cam - lo) / (hi - lo + 1e-8)
            im = ax.imshow(mean_cam, cmap="jet", vmin=0, vmax=1)
            ax.axis("off")
            ax.set_title(f"{LABEL_NAMES[cls]}  (n={cam_count[cls]})", fontsize=11)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        out = FIGURES_ROOT / "mean_cam_comparison.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")

    # Overlay grids
    for cls in [0, 1]:
        items = overlay_items[cls]
        if not items:
            continue
        # Sort: misclassified first (more interesting)
        items.sort(key=lambda x: x[3] == cls)  # wrong preds first
        items = items[:n_grid]
        ncols = 4
        nrows = int(np.ceil(len(items) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 3.2))
        axes = np.array(axes).reshape(-1)
        fig.suptitle(f"GradCAM Overlays — {LABEL_NAMES[cls]} (Phase 5)",
                     fontsize=12, fontweight="bold")
        for i, ax in enumerate(axes):
            if i < len(items):
                ov, pid, eye, pred = items[i]
                ax.imshow(ov)
                correct = pred == cls
                col = "#2e7d32" if correct else "#c62828"
                ax.set_title(f"RET{pid:03d}{eye}\n→ {LABEL_NAMES[pred]}",
                             fontsize=7.5, color=col)
            ax.axis("off")
        out = FIGURES_ROOT / f"overlay_grid_{LABEL_NAMES[cls].lower()}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


    # Disc-centred detail plot
    if disc_patch_sum:
        make_disc_attention_detail(mean_patches, disc_stats_rows,
                                   FIGURES_ROOT / "disc_attention_detail.png")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-grid",       type=int,   default=16,
                    help="Max overlays per class in grid (default 16)")
    ap.add_argument("--alpha",        type=float, default=0.45,
                    help="GradCAM overlay opacity (default 0.45)")
    ap.add_argument("--target-class", type=int,   default=None,
                    help="GradCAM target class (default: predicted class)")
    args = ap.parse_args()
    run(n_grid=args.n_grid, alpha=args.alpha, target_class=args.target_class)
