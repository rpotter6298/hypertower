#!/usr/bin/env python3
"""
Per-class ROC: one figure per class (multiclass) OR one figure total (binary),
with ALL models (runs under a tag) plotted as separate lines.

Outputs under analysis_data/:
  - multiclass:
      <tag>_class0_roc.png  (e.g., Healthy)
      <tag>_class1_roc.png  (e.g., Glaucoma)
      <tag>_class2_roc.png  (e.g., Suspect)
      <tag>_perclass_summary.json
  - binary:
      <tag>_binary_roc.png
      <tag>_perclass_summary.json
"""

import argparse, json, re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score

HEAD_ALIASES = {"image": ["image","img"], "fused": ["fused"], "metadata": ["metadata","md"]}

def find_run_dirs(tag_prefix: str, analysis_dir: Path):
    return sorted([p for p in analysis_dir.glob(f"{tag_prefix}_*") if p.is_dir()])

def read_summary(run_dir: Path) -> dict:
    p = run_dir / "summary.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}

def find_folds(run_dir: Path, head: str):
    variants = HEAD_ALIASES.get(head, [head])
    y_files = sorted(run_dir.glob("fold*_y_true.npy"))
    folds = []
    for yf in y_files:
        m = re.search(r"fold(\d+)_y_true\.npy$", yf.name)
        if not m: continue
        idx = int(m.group(1))
        if any((run_dir / f"fold{idx}_probs_{v}.npy").exists() for v in variants):
            folds.append(idx)
    return folds

def load_probs(run_dir: Path, fold: int, head: str):
    variants = HEAD_ALIASES.get(head, [head])
    y = np.load(run_dir / f"fold{fold}_y_true.npy")
    p = None
    tried = []
    for v in variants:
        pp = run_dir / f"fold{fold}_probs_{v}.npy"
        tried.append(pp.name)
        if pp.exists():
            p = np.load(pp); break
    if p is None:
        raise FileNotFoundError(f"Missing probs for fold {fold} in {run_dir}; tried {tried}")
    return y, p

def infer_mode_from_files(run_dir: Path, head: str):
    f = find_folds(run_dir, head)
    if not f: return None
    _, p = load_probs(run_dir, f[0], head)
    if p.ndim == 2 and p.shape[1] == 2: return "binary"
    if p.ndim == 2 and p.shape[1] >= 3: return "multiclass"
    return None

def per_class_roc(y, p):
    """Return {k: (fpr, tpr, auc)} for OVR."""
    K = p.shape[1]
    out = {}
    for k in range(K):
        yb = (y == k).astype(np.uint8)
        fpr, tpr, _ = roc_curve(yb, p[:, k])
        out[k] = (fpr, tpr, auc(fpr, tpr) if len(fpr) > 1 else np.nan)
    return out

def make_per_model_class_curves(run_dir: Path, head: str, mode: str):
    """
    Returns:
      label (model/backbone name),
      class_curves: dict[k] -> dict with keys:
         'fpr': grid, 'tpr_mean': mean across folds on grid, 'auc_mean': mean across folds,
         'tpr_std' and 'auc_std' also included.
      K = number of classes (2 or 3+)
    """
    summary = read_summary(run_dir)
    label = summary.get("backbone") or run_dir.name
    folds = find_folds(run_dir, head)
    if not folds:
        return None

    # collect per-fold per-class curves
    per_fold = []
    for f in folds:
        y, p = load_probs(run_dir, f, head)
        if mode == "binary":
            keep = np.isin(y, [0,1])
            if keep.sum() == 0: 
                continue
            y, p = y[keep], p[keep]
            if p.shape[1] > 2:  # safety; binary should have 2 cols
                p = p[:, :2]
        else:
            if p.ndim != 2 or p.shape[1] < 3:
                continue
        per_fold.append(per_class_roc(y, p))
    if not per_fold:
        return None

    # interpolate on a common grid, avg across folds
    grid = np.linspace(0, 1, 501)
    K = max(per_fold[0].keys()) + 1
    class_curves = {}
    for k in range(K):
        tprs, aucs = [], []
        for d in per_fold:
            if k not in d: 
                continue
            fpr, tpr, a = d[k]
            tprs.append(np.interp(grid, fpr, tpr))
            aucs.append(a)
        if not tprs:
            continue
        tprs = np.vstack(tprs)
        class_curves[k] = {
            "fpr": grid,
            "tpr_mean": tprs.mean(axis=0),
            "tpr_std":  tprs.std(axis=0),
            "auc_mean": float(np.nanmean(aucs)),
            "auc_std":  float(np.nanstd(aucs)),
        }
    return label, class_curves

def main():
    ap = argparse.ArgumentParser(description="Per-class ROC with all models as separate lines.")
    ap.add_argument("--tag", required=True, help="analysis_data prefix like 'papergrid'")
    ap.add_argument("--head", default="image", choices=["image","fused","metadata"])
    ap.add_argument("--mode", choices=["binary","multiclass"], required=True,
                    help="Select which experiment style to aggregate.")
    ap.add_argument("--fusion-mode", choices=["image_only","fused","metadata_only","vote"], default=None,
                    help="Filter runs by fusion mode to avoid mixing.")
    ap.add_argument("--analysis-dir", default="analysis_data")
    ap.add_argument("--class-names", nargs="*", default=["Healthy","Glaucoma","Suspect"])
    ap.add_argument("--shade", action="store_true", help="Shade ±1 SD per model (can get busy).")
    args = ap.parse_args()

    analysis_dir = Path(args.analysis_dir) / args.tag
    run_dirs_all = find_run_dirs(args.tag, analysis_dir)
    if not run_dirs_all:
        raise SystemExit(f"No run directories found starting with '{args.tag}_' under {analysis_dir}")

    # filter runs
    selected = []
    skipped = []
    for rd in run_dirs_all:
        sj = read_summary(rd)
        m = sj.get("eval_mode") or infer_mode_from_files(rd, args.head)
        if m != args.mode:
            skipped.append((rd, f"mode={m}")); continue
        if args.fusion_mode:
            fm = sj.get("fusion_mode")
            if fm and fm != args.fusion_mode:
                skipped.append((rd, f"fusion_mode={fm}")); continue
        selected.append(rd)

    if not selected:
        raise SystemExit("No runs matched filters (mode/fusion-mode).")

    # build per-model curves
    per_model = []  # list of (label, class_curves)
    for rd in selected:
        res = make_per_model_class_curves(rd, args.head, args.mode)
        if res is None:
            skipped.append((rd, "no_usable_folds")); continue
        per_model.append(res)

    if not per_model:
        raise SystemExit("No usable runs after fold parsing/interpolation.")

    # determine classes to plot
    maxK = max((max(curves.keys())+1) for _, curves in per_model)
    if args.mode == "binary":
        # Only class 1 (positive) is typically plotted
        classes_to_plot = [1]
        class_names = [args.class_names[1] if len(args.class_names) > 1 else "Positive"]
        outfile_names = [f"{args.tag}_binary_roc.png"]
        title_suffixes = ["Binary (positive class)"]
    else:
        classes_to_plot = list(range(min(3, maxK)))  # usually 0,1,2
        class_names = [args.class_names[i] if i < len(args.class_names) else f"class {i}" for i in classes_to_plot]
        outfile_names = [f"{args.tag}_class{i}_roc.png" for i in classes_to_plot]
        title_suffixes = [f"Class: {name}" for name in class_names]

    # plot per class: all models on same axes
    out_json = {"tag": args.tag, "mode": args.mode, "head": args.head,
                "fusion_mode_filter": args.fusion_mode, "figures": []}

    for k, cname, out_name, t_suffix in zip(classes_to_plot, class_names, outfile_names, title_suffixes):
        fig = plt.figure(figsize=(10, 8)); ax = fig.add_subplot(111)
        ax.plot([0,1],[0,1], linestyle="--", linewidth=1)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        title_bits = [f"Combined ROC — {args.tag}", t_suffix, f"[{args.head}]"]
        if args.fusion_mode: title_bits.append(f"[{args.fusion_mode}]")
        ax.set_title(" — ".join(title_bits))

        entries = []
        for label, curves in per_model:
            if k not in curves:
                continue
            c = curves[k]
            ax.plot(c["fpr"], c["tpr_mean"], linewidth=2,
                    label=f"{label} (AUC {c['auc_mean']:.3f}±{c['auc_std']:.3f})")
            if args.shade:
                ax.fill_between(c["fpr"],
                                np.maximum(c["tpr_mean"] - c["tpr_std"], 0),
                                np.minimum(c["tpr_mean"] + c["tpr_std"], 1),
                                alpha=0.10)
            entries.append({"label": label, "auc_mean": c["auc_mean"], "auc_std": c["auc_std"]})

        ax.legend(loc="lower right")
        fig.tight_layout()

        out_path = analysis_dir / out_name
        fig.savefig(out_path, dpi=160); plt.close(fig)

        out_json["figures"].append({
            "class_index": k, "class_name": cname, "output_png": str(out_path),
            "models": entries
        })

    # metadata file
    meta_path = analysis_dir / f"{args.tag}_perclass_summary.json"
    meta_path.write_text(json.dumps(out_json, indent=2), encoding="utf-8")
    print(f"Wrote figures + {meta_path}")
    
if __name__ == "__main__":
    main()
