#!/usr/bin/env python3
"""
Aggregate raw GradCAM heatmaps across all folds for a run.

For each combination of (eye, class, correct/incorrect) computes:
  - mean heatmap
  - std heatmap
  - count

Also computes a scalar per patient: fraction of GradCAM attention mass that
falls within the expert-segmented optic disc region (from GT contour files),
using the manifest.csv to locate the contour for each patient/eye.

Outputs
-------
  {out_dir}/mean_heatmaps.npz
      Keys: {eye}_{class_name}_{correct|incorrect}_{mean|std|count}
      e.g.  OD_Glaucoma_correct_mean   shape (224, 224)

  {out_dir}/attention_stats.csv
      per-patient scalars: patient_id, fold, eye, true_name, pred_name,
      correct, confidence, disc_frac, entropy

Usage
-----
    python scripts/output_analysis/explainability/aggregate_gradcam.py \
        --run-dir   analysis_data/pipeline_nocrop \
        --eval-mode binary \
        --tower-mode single
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def load_disc_mask(contour_path: Path, orig_size: tuple[int, int],
                   cam_h: int, cam_w: int) -> np.ndarray | None:
    """
    Load a PAPILA disc contour TXT file, polygon-fill at original image
    dimensions, then resize to (cam_h, cam_w).  Returns a bool array or
    None if the contour cannot be loaded.
    """
    try:
        arr = np.loadtxt(str(contour_path), dtype=np.float32)
    except Exception:
        return None
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    if arr.shape[0] < 3 or arr.shape[1] < 2:
        return None

    # orig_size is (W, H) as PIL convention
    img = Image.new("L", orig_size, 0)
    draw = ImageDraw.Draw(img)
    draw.polygon([tuple(pt) for pt in arr[:, :2]], fill=1)
    mask = np.array(img.resize((cam_w, cam_h), Image.NEAREST), dtype=bool)
    return mask


def build_disc_lookup(manifest_path: Path) -> dict[tuple[int, str], tuple[Path, tuple[int, int]]]:
    """
    Returns {(patient_id_int, eye): (disc_contour_path, (img_W, img_H))}.
    Only PAPILA rows are included.
    """
    mf = pd.read_csv(manifest_path)
    lookup: dict[tuple[int, str], tuple[Path, tuple[int, int]]] = {}
    for _, row in mf.iterrows():
        sid = str(row["sample_id"])
        if not sid.startswith("papila_RET"):
            continue
        # sample_id: papila_RET002OD  or  papila_RET002OS
        suffix = sid[len("papila_RET"):]   # e.g. "002OD"
        eye    = suffix[-2:]                # "OD" or "OS"
        pid    = int(suffix[:-2])           # 2
        disc_path = Path(str(row["annotation_disc"]))
        img_path  = Path(str(row["image_path"]))
        if not disc_path.exists():
            continue
        # read original image size once
        try:
            with Image.open(img_path) as im:
                orig_size = im.size   # (W, H)
        except Exception:
            continue
        lookup[(pid, eye)] = (disc_path, orig_size)
    return lookup


def attention_entropy(cam: np.ndarray) -> float:
    flat = cam.flatten().astype(np.float64)
    flat = flat / (flat.sum() + 1e-12)
    return float(-np.sum(flat * np.log(flat + 1e-12)))


def load_fold(gradcam_dir: Path):
    idx_path = gradcam_dir / "gradcam_index.csv"
    if not idx_path.exists():
        return None
    idx = pd.read_csv(idx_path)
    records = []
    for _, row in idx.iterrows():
        pid = row["patient_id"]
        for eye in ("OD", "OS"):
            npy = gradcam_dir / f"patient_{pid}_{eye}_cam.npy"
            if not npy.exists():
                continue
            cam = np.load(npy)
            records.append({
                "patient_id": pid,
                "eye":        eye,
                "true_label": int(row["true_label"]),
                "true_name":  row["true_name"],
                "pred_label": int(row["pred_label"]),
                "pred_name":  row["pred_name"],
                "confidence": float(row["confidence"]),
                "correct":    bool(row["correct"]),
                "cam":        cam,
            })
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir",    default="analysis_data/pipeline_nocrop")
    ap.add_argument("--eval-mode",  default="binary")
    ap.add_argument("--tower-mode", default="single")
    ap.add_argument("--manifest",   default="manifest.csv")
    ap.add_argument("--out",        default=None)
    args = ap.parse_args()

    run_dir  = Path(args.run_dir)
    mode_dir = run_dir / args.eval_mode / args.tower_mode
    out_dir  = mode_dir / "gradcam_aggregate"
    if args.out:
        out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- build disc mask lookup ----
    manifest_path = Path(args.manifest)
    disc_lookup = build_disc_lookup(manifest_path)
    print(f"Disc mask lookup: {len(disc_lookup)} entries from {manifest_path}")

    # ---- collect all records ----
    all_records = []
    stat_rows   = []
    fold_dirs   = sorted(
        [d for d in mode_dir.iterdir() if d.is_dir() and d.name.startswith("fold")],
        key=lambda p: int(p.name.replace("fold", "")),
    )
    if not fold_dirs:
        print(f"No fold dirs found under {mode_dir}")
        return

    for fd in fold_dirs:
        gcam_dir = fd / "explainability" / "gradcam"
        records  = load_fold(gcam_dir)
        if records is None:
            print(f"  [skip] {fd.name}: no gradcam_index.csv")
            continue
        print(f"  {fd.name}: {len(records)} eye records")
        for r in records:
            r["fold"] = fd.name
            all_records.append(r)

    if not all_records:
        print("No records found — re-run explain_fold.py first.")
        return

    print(f"\nTotal eye records: {len(all_records)}")

    h, w = all_records[0]["cam"].shape

    # ---- per-record stats ----
    n_missing = 0
    for r in all_records:
        cam   = r["cam"]
        total = cam.sum() + 1e-12
        pid   = int(r["patient_id"])
        eye   = r["eye"]

        disc_mask = None
        key = (pid, eye)
        if key in disc_lookup:
            disc_path, orig_size = disc_lookup[key]
            disc_mask = load_disc_mask(disc_path, orig_size, h, w)
        if disc_mask is None:
            n_missing += 1
            disc_frac = float("nan")
        else:
            disc_frac = float(cam[disc_mask].sum() / total)

        stat_rows.append({
            "patient_id": r["patient_id"],
            "fold":       r["fold"],
            "eye":        r["eye"],
            "true_name":  r["true_name"],
            "pred_name":  r["pred_name"],
            "correct":    r["correct"],
            "confidence": r["confidence"],
            "disc_frac":  disc_frac,
            "entropy":    attention_entropy(cam),
        })

    if n_missing:
        print(f"  Warning: {n_missing} records had no disc mask (disc_frac=NaN)")

    stats_df = pd.DataFrame(stat_rows)
    stats_path = out_dir / "attention_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved attention stats → {stats_path}")

    # ---- mean heatmaps ----
    npz_arrays = {}
    groups: dict[tuple, list[np.ndarray]] = {}
    for r in all_records:
        key = (r["eye"], r["true_name"], "correct" if r["correct"] else "incorrect")
        groups.setdefault(key, []).append(r["cam"])
    for r in all_records:
        key = (r["eye"], r["true_name"], "all")
        groups.setdefault(key, []).append(r["cam"])

    for (eye, cls, split), cams in groups.items():
        stack = np.stack(cams, axis=0)
        key_base = f"{eye}_{cls}_{split}"
        npz_arrays[f"{key_base}_mean"]  = stack.mean(axis=0).astype(np.float32)
        npz_arrays[f"{key_base}_std"]   = stack.std(axis=0).astype(np.float32)
        npz_arrays[f"{key_base}_count"] = np.array(len(cams))
        print(f"  {key_base}: N={len(cams)}")

    npz_path = out_dir / "mean_heatmaps.npz"
    np.savez_compressed(npz_path, **npz_arrays)
    print(f"Saved mean heatmaps → {npz_path}")


if __name__ == "__main__":
    main()
