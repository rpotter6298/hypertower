#!/usr/bin/env python3
"""
Granular disc-attention visualisation.

Layout (3 rows × N_class cols):
  Row 0 — disc-centred mean GradCAM patch for CORRECT predictions
  Row 1 — disc-centred mean GradCAM patch for INCORRECT predictions
  Row 2 — per-patient strip plot of disc_frac (blue=correct, red=incorrect)

Disc-centred patches: each patient's CAM is translated and scaled so the GT
disc centroid sits at the patch centre before averaging.  A dashed white circle
marks the average GT disc size.  This makes cross-patient averaging meaningful
regardless of where the disc sits in the original image.

Usage
-----
    python scripts/output_analysis/explainability/plot_disc_attention_detail.py \
        --agg-dir analysis_data/pipeline_nocrop/binary/single/gradcam_aggregate \
        --manifest manifest.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image, ImageDraw


DISC_SPAN   = 5     # patch side = DISC_SPAN × disc diameter
OUTPUT_SIZE = 96    # pixel size of each thumbnail


# ---------------------------------------------------------------------------
# Disc mask helpers
# ---------------------------------------------------------------------------

def _load_disc_mask(contour_path: Path, orig_size: tuple, cam_h: int, cam_w: int):
    try:
        arr = np.loadtxt(str(contour_path), dtype=np.float32)
    except Exception:
        return None
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    if arr.shape[0] < 3 or arr.shape[1] < 2:
        return None
    img = Image.new("L", orig_size, 0)
    ImageDraw.Draw(img).polygon([tuple(pt) for pt in arr[:, :2]], fill=1)
    return np.array(img.resize((cam_w, cam_h), Image.NEAREST), dtype=bool)


def build_disc_lookup(manifest_path: Path) -> dict:
    mf = pd.read_csv(manifest_path)
    lookup: dict = {}
    for _, row in mf.iterrows():
        sid = str(row["sample_id"])
        if not sid.startswith("papila_RET"):
            continue
        suffix = sid[len("papila_RET"):]
        eye = suffix[-2:]
        pid = int(suffix[:-2])
        disc_path = Path(str(row["annotation_disc"]))
        img_path  = Path(str(row["image_path"]))
        if not disc_path.exists():
            continue
        try:
            with Image.open(img_path) as im:
                orig_size = im.size
        except Exception:
            continue
        lookup[(pid, eye)] = (disc_path, orig_size)
    return lookup


# ---------------------------------------------------------------------------
# Disc-centred patch extraction
# ---------------------------------------------------------------------------

def disc_centered_patch(
    cam: np.ndarray,
    disc_mask: np.ndarray,
    span: int = DISC_SPAN,
    out: int = OUTPUT_SIZE,
) -> tuple[np.ndarray | None, float | None]:
    """
    Return (patch, disc_r_out):
      patch       — (out, out) float32 in [0, 1]
      disc_r_out  — disc radius in patch-pixel units (for drawing reference circle)
    """
    if disc_mask is None or disc_mask.sum() == 0:
        return None, None

    ys, xs = np.where(disc_mask)
    cy, cx = ys.mean(), xs.mean()
    disc_r = float(np.sqrt(disc_mask.sum() / np.pi))
    half   = max(1, int(round(span * disc_r / 2)))

    h, w = cam.shape
    y0, y1 = int(round(cy)) - half, int(round(cy)) + half
    x0, x1 = int(round(cx)) - half, int(round(cx)) + half

    pt = max(0, -y0);  pb = max(0, y1 - h)
    pl = max(0, -x0);  pr = max(0, x1 - w)
    cam_pad = np.pad(cam, ((pt, pb), (pl, pr)), constant_values=0.0)

    patch = cam_pad[y0 + pt : y1 + pt, x0 + pl : x1 + pl]
    patch_img = Image.fromarray((np.clip(patch, 0, 1) * 255).astype(np.uint8))
    patch_out = np.array(patch_img.resize((out, out), Image.BILINEAR)) / 255.0

    disc_r_out = out * disc_r / (2 * half)
    return patch_out.astype(np.float32), disc_r_out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_cam_records(mode_dir: Path) -> list[dict]:
    records = []
    fold_dirs = sorted(
        [d for d in mode_dir.iterdir() if d.is_dir() and d.name.startswith("fold")],
        key=lambda p: int(p.name.replace("fold", "")),
    )
    for fd in fold_dirs:
        gcam_dir = fd / "explainability" / "gradcam"
        idx_path = gcam_dir / "gradcam_index.csv"
        if not idx_path.exists():
            continue
        idx = pd.read_csv(idx_path)
        for _, row in idx.iterrows():
            pid = int(row["patient_id"])
            for eye in ("OD", "OS"):
                npy = gcam_dir / f"patient_{pid}_{eye}_cam.npy"
                if not npy.exists():
                    continue
                records.append({
                    "patient_id": pid,
                    "eye":        eye,
                    "true_name":  row["true_name"],
                    "correct":    bool(row["correct"]),
                    "cam":        np.load(npy),
                })
    return records


def build_mean_patches(
    records: list[dict],
    disc_lookup: dict,
    classes: list[str],
) -> dict[tuple, tuple]:
    """
    Returns {(cls, split): (mean_patch, mean_disc_r_out, count)}
    split = 'correct' | 'incorrect'
    """
    buckets:  dict[tuple, list] = {}
    radii:    dict[tuple, list] = {}

    for r in records:
        split = "correct" if r["correct"] else "incorrect"
        key   = (r["true_name"], split)
        pid, eye = r["patient_id"], r["eye"]
        if (pid, eye) not in disc_lookup:
            continue
        disc_path, orig_size = disc_lookup[(pid, eye)]
        h, w = r["cam"].shape
        disc_mask = _load_disc_mask(disc_path, orig_size, h, w)
        patch, disc_r_out = disc_centered_patch(r["cam"], disc_mask)
        if patch is None:
            continue
        buckets.setdefault(key, []).append(patch)
        radii.setdefault(key, []).append(disc_r_out)

    return {
        key: (np.stack(ps).mean(0), float(np.mean(radii[key])), len(ps))
        for key, ps in buckets.items()
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg-dir",  required=True)
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--out",      default=None)
    args = ap.parse_args()

    agg_dir  = Path(args.agg_dir)
    mode_dir = agg_dir.parent
    out_path = Path(args.out) if args.out else agg_dir / "disc_attention_detail.png"

    print("Loading disc lookup…")
    disc_lookup = build_disc_lookup(Path(args.manifest))
    print(f"  {len(disc_lookup)} entries")

    print("Loading CAM records…")
    records = load_all_cam_records(mode_dir)
    print(f"  {len(records)} eye records")

    stats   = pd.read_csv(agg_dir / "attention_stats.csv")
    classes = sorted({r["true_name"] for r in records})
    n_cls   = len(classes)
    print(f"Classes: {classes}")

    print("Building disc-centred mean patches…")
    mean_patches = build_mean_patches(records, disc_lookup, classes)

    # -----------------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------------
    corr_colors = {"correct": "steelblue", "incorrect": "tomato"}
    splits      = ["correct", "incorrect"]
    row_labels  = ["Correct", "Incorrect", "Disc fraction\n(strip plot)"]

    fig, axes = plt.subplots(3, n_cls, figsize=(4.2 * n_cls, 13))
    if n_cls == 1:
        axes = axes[:, np.newaxis]

    # ---- rows 0 & 1: disc-centred heatmaps ----
    for ri, split in enumerate(splits):
        for ci, cls in enumerate(classes):
            ax  = axes[ri, ci]
            key = (cls, split)
            if key in mean_patches:
                mean_patch, disc_r_out, count = mean_patches[key]
                ax.imshow(mean_patch, cmap="jet", vmin=0, vmax=1, origin="upper",
                          extent=[0, OUTPUT_SIZE, OUTPUT_SIZE, 0])
                cx = cy = OUTPUT_SIZE / 2
                ax.add_patch(Circle((cx, cy), disc_r_out,
                                    fill=False, edgecolor="white",
                                    linewidth=2, linestyle="--"))
                ax.set_title(f"{cls}  |  {split}\n(N={count})", fontsize=9)
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="grey")
                ax.set_title(f"{cls}  |  {split}", fontsize=9)
            ax.axis("off")
        axes[ri, 0].set_ylabel(row_labels[ri], fontsize=10, labelpad=6)

    # ---- row 2: strip plots ----
    rng = np.random.default_rng(42)
    for ci, cls in enumerate(classes):
        ax  = axes[2, ci]
        sub = stats[stats["true_name"] == cls].dropna(subset=["disc_frac"])

        for xi, split in enumerate(splits):
            correct_val = (split == "correct")
            pts = sub[sub["correct"] == correct_val]["disc_frac"].values
            if len(pts) == 0:
                continue
            color  = corr_colors[split]
            jitter = rng.uniform(-0.18, 0.18, size=len(pts))
            ax.scatter(xi + jitter, pts, color=color, alpha=0.7, s=28, edgecolors="none")
            ax.hlines(pts.mean(), xi - 0.28, xi + 0.28,
                      colors=color, linewidth=2.5, zorder=5)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Correct", "Incorrect"], fontsize=9)
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylim(0, 1)
        ax.set_title(cls, fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        if ci == 0:
            ax.set_ylabel("Disc fraction\n(GT disc attention)", fontsize=9)

    fig.suptitle(
        "Disc-centred GradCAM attention  |  dashed circle = GT disc boundary",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
