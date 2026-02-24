#!/usr/bin/env python3
import argparse, subprocess, sys, time, json
from pathlib import Path

# Backbones in the paper that torchvision supports
BACKBONES = [
    "efficientnet_b0",
    "resnet50",
    "densenet121",
    "vgg16",
    "mobilenet_v2",
    "inception_v3",
    # (Xception omitted; not in torchvision — add via timm later if needed)
]

MODES = [
    ("multiclass", ["Healthy", "Glaucoma", "Suspect"]),
    ("binary", ["Healthy", "Glaucoma"]),
]

def run(cmd):
    print("\n$ " + " ".join(map(str, cmd)))
    res = subprocess.run(cmd, check=True)
    return res.returncode

def main():
    ap = argparse.ArgumentParser(description="Run all paper CNNs across folds in multiclass + binary, then compile plots.")
    ap.add_argument("--epochs", type=int, default=5, help="Epochs per fold (fast sanity first).")
    ap.add_argument("--shortname", type=str, default="papergrid", help="Prefix for run IDs.")
    ap.add_argument("--n-splits", type=int, default=5, help="Number of folds.")
    ap.add_argument("--fusion-mode", type=str, default="fused", choices=["image_only","fused","metadata_only","vote"],
                    help="Paper CNNs are image-only; leave as image_only unless you’re testing others.")
    ap.add_argument("--freeze-ratio", type=float, default=0.0, help="0.0 = full fine-tune (as in the paper).")
    # You can override data roots if needed
    ap.add_argument("--image-dir", default="Papila/FundusImages")
    ap.add_argument("--clinical-dir", default="Papila/ClinicalData")
    ap.add_argument("--label-col", default="Diagnosis")
    ap.add_argument("--cat-cols", nargs="*", default=["Gender", "Phakic/Pseudophakic"])
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    master_tag = f"{args.shortname}_{ts}"
    master_dir = Path("analysis_data") / master_tag
    master_dir.mkdir(parents=True, exist_ok=True)

    # Keep a log of all subruns for the master report
    index = []

    for backbone in BACKBONES:
        for eval_mode, class_names in MODES:
            # build a child shortname per (backbone, mode)
            sub_prefix = f"{args.shortname}_{backbone}_{eval_mode}"
            cmd = [
                sys.executable, "scripts/run_multifold.py",
                "--backbone", backbone,
                "--freeze-ratio", str(args.freeze_ratio),
                "--fusion-mode", args.fusion_mode,
                "--epochs", str(args.epochs),
                "--n-splits", str(args.n_splits),
                "--shortname", sub_prefix,
                "--eval_mode", eval_mode,
                "--image-dir", args.image_dir,
                "--clinical-dir", args.clinical_dir,
                "--label-col", args.label_col,
            ]

            # class names by mode (ensures plot legends are correct)
            cmd += ["--class-names", *class_names]

            plot_head_map = {
                "image_only": "image",
                "fused" : "fused",
                "metadata_only": "metadata",
                "vote": "fused",
            }
            # We always aggregate/plot the image head for paper CNNs
            cmd += ["--plot-head", plot_head_map.get(args.fusion_mode)]

            # Delegate the whole run to run_multifold.py
            run(cmd)

            # Discover the child run folder (the newest folder matching the shortname prefix)
            # We do this because run_multifold appends its own timestamp.
            adir = Path("analysis_data")
            children = sorted([p for p in adir.glob(f"{sub_prefix}_*") if p.is_dir()])
            if not children:
                print(f"[WARN] No analysis_data folder found for {sub_prefix}; skipping index entry.")
                continue
            run_dir = children[-1]
            summary_json = run_dir / "summary.json"
            plots_dir = run_dir / "plots"

            # Record entry
            entry = {
                "backbone": backbone,
                "eval_mode": eval_mode,
                "run_dir": str(run_dir),
                "summary_json": str(summary_json) if summary_json.exists() else None,
                "plots": {
                    "mean": str(plots_dir / "roc_image_mean_ovr.png"),
                    "overlay": str(plots_dir / "roc_image_perfold_overlay.png"),
                }
            }
            # Try to read AUCs
            try:
                if summary_json.exists():
                    entry.update(json.loads(summary_json.read_text()))
            except Exception:
                pass
            index.append(entry)

    # Write a master JSON + markdown report
    (master_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    # Simple markdown table of results with links
    lines = [
        f"# Multimodel grid — {master_tag}",
        "",
        f"- Epochs per fold: **{args.epochs}**",
        f"- Folds: **{args.n_splits}**",
        f"- Fusion mode: **{args.fusion_mode}** (paper CNNs = image-only)",
        f"- Freeze ratio: **{args.freeze_ratio}**",
        "",
        "| Backbone | Mode | Mean AUC (macro/mc or ROC-AUC/bin) | Plots | Run folder |",
        "|---|---|---:|---|---|",
    ]
    for e in index:
        auc_mean = e.get("macro_ovr_auc_mean", None)
        if auc_mean is not None:
            auc_str = f"{auc_mean:.3f}"
        else:
            auc_str = "—"
        mean_png = e["plots"]["mean"]
        overlay_png = e["plots"]["overlay"]
        plots_md = f"[mean]({mean_png}) / [overlay]({overlay_png})"
        lines.append(
            f"| `{e['backbone']}` | `{e['eval_mode']}` | {auc_str} | {plots_md} | `{e['run_dir']}` |"
        )
    (master_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nAll done.\n- Master index: {master_dir/'index.json'}\n- Report: {master_dir/'README.md'}")
    print(f"- Individual runs live under analysis_data/<shortname_backbone_mode_*> with plots and summaries.")

if __name__ == "__main__":
    main()
