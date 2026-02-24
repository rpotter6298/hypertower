#!/usr/bin/env python3
"""
Inspect saved validation/holdout logits for a multifold run.
Prints per-class AUCs and sample counts so we can sanity-check unusually high scores.
Can also print per-fold confusion matrices.

Example:
    python scripts/fold_confusion_matrix.py \
        --run-dir analysis_data/1030_Balanced_Unet_Perimg_Resnet_SE16NormB_SE16NormT_multi_fused/1030_Balanced_Unet_Perimg_Resnet_SE16NormB_SE16NormT_multi_fused_20251030_091842 \
        --head fused
    python scripts/fold_confusion_matrix.py --run-dir ... --head fused --use-holdout --confusion
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Inspect saved logits for a run and report per-class AUCs.")
    ap.add_argument("--run-dir", required=True, type=Path, help="Path to the run directory under analysis_data.")
    ap.add_argument("--head", choices=["fused", "image", "metadata"], default="fused",
                    help="Which prediction head's saved probabilities to load.")
    ap.add_argument("--use-holdout", action="store_true",
                    help="Look for *_holdout.npy dumps instead of validation splits.")
    ap.add_argument("--class-names", nargs="*", default=None,
                    help="Optional override for class labels (order should match numeric labels).")
    ap.add_argument("--macro", action="store_true", help="Also print macro-average AUC across classes.")
    ap.add_argument("--confusion", action="store_true", help="Print confusion matrix for each fold.")
    return ap.parse_args()


def load_cli_args(run_dir: Path) -> Dict:
    path = run_dir / "cli_args.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing cli_args.json in {run_dir}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def find_fold_files(run_dir: Path, suffix: str) -> Dict[int, Dict[str, Path]]:
    files: Dict[int, Dict[str, Path]] = {}
    for y_file in run_dir.glob(f"fold*_y_true{suffix}.npy"):
        fold_str = y_file.stem.split("_")[0].replace("fold", "")
        try:
            fold_idx = int(fold_str)
        except ValueError:
            continue
        files.setdefault(fold_idx, {})["y_true"] = y_file
    for head_key, glob_pat in [
        ("fused", f"fold*_probs_fused{suffix}.npy"),
        ("image", f"fold*_probs_img{suffix}.npy"),
        ("metadata", f"fold*_probs_md{suffix}.npy"),
    ]:
        for p_file in run_dir.glob(glob_pat):
            fold_str = p_file.stem.split("_")[0].replace("fold", "")
            try:
                fold_idx = int(fold_str)
            except ValueError:
                continue
            files.setdefault(fold_idx, {})[head_key] = p_file
    return files


def compute_auc(y_true: np.ndarray, probs: np.ndarray, class_names: List[str], macro: bool) -> List[int]:
    num_classes = probs.shape[1]
    unique = np.unique(y_true)
    print(f"    classes present: {sorted(unique.tolist())}")

    aucs = []
    seen_classes: List[int] = []
    for cls in range(num_classes):
        name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
        mask = (y_true == cls)
        pos = int(mask.sum())
        neg = len(y_true) - pos
        if pos == 0 or neg == 0:
            print(f"      {name:<15} -> insufficient positives/negatives (pos={pos}, neg={neg}); skipping AUC")
            continue
        try:
            auc = roc_auc_score((y_true == cls).astype(int), probs[:, cls])
        except ValueError as exc:
            print(f"      {name:<15} -> AUC error: {exc}")
            continue
        aucs.append(auc)
        seen_classes.append(cls)
        print(f"      {name:<15} -> AUC={auc:.4f}  (pos={pos}, neg={neg})")

    if macro and aucs:
        mean = float(np.mean(aucs))
        std = float(np.std(aucs, ddof=0)) if len(aucs) > 1 else math.nan
        print(f"      macro AUC across reported classes: {mean:.4f} (std={std:.4f})")
    return seen_classes


def print_confusion(y_true: np.ndarray, probs: np.ndarray, class_names: List[str]) -> None:
    num_classes = probs.shape[1]
    preds = probs.argmax(axis=1)
    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, preds, labels=labels)
    names = [class_names[i] if i < len(class_names) else f"class_{i}" for i in labels]
    header = " " * 14 + "".join(f"{name:>12}" for name in names)
    print("      Confusion matrix (rows=true, cols=pred):")
    print(header)
    for idx, row in enumerate(cm):
        label = names[idx]
        row_str = "".join(f"{int(val):>12}" for val in row)
        print(f"        {label:<12}{row_str}")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    cli_args = load_cli_args(run_dir)
    eval_mode = cli_args.get("eval_mode", "multiclass")
    if args.class_names:
        class_names = args.class_names
    else:
        if eval_mode == "binary":
            class_names = ["Healthy", "Glaucoma"]
        else:
            class_names = cli_args.get("class_names") or ["Healthy", "Glaucoma", "Suspect"]

    suffix = "_holdout" if args.use_holdout else ""
    files = find_fold_files(run_dir, suffix)
    if not files:
        raise SystemExit(f"No saved probability files matching suffix '{suffix}' found in {run_dir}. "
                         "Run scripts/rebuild_run_best_plots.py first if needed.")

    print(f"[info] Inspecting head='{args.head}' ({'holdout' if args.use_holdout else 'validation'})")
    for fold_idx in sorted(files.keys()):
        fold = files[fold_idx]
        if "y_true" not in fold:
            print(f"[warning] Fold {fold_idx}: missing y_true file; skipping.")
            continue
        head_key = {
            "fused": "fused",
            "image": "image",
            "metadata": "metadata",
        }[args.head]
        prob_path = fold.get(head_key)
        if prob_path is None:
            print(f"[warning] Fold {fold_idx}: missing probability file for head '{args.head}'; skipping.")
            continue

        y_true = np.load(fold["y_true"])
        probs = np.load(prob_path)
        print(f"\n  Fold {fold_idx} -> samples={len(y_true)}  file={prob_path.name}")
        compute_auc(y_true, probs, class_names, args.macro)
        if args.confusion:
            print_confusion(y_true, probs, class_names)


if __name__ == "__main__":
    main()
