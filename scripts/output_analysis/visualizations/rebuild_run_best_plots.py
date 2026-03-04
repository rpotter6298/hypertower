#!/usr/bin/env python3
"""
Recompute per-fold ROC plots for a completed multifold run using the saved
best checkpoints instead of the final epoch.

Example:
  python scripts/rebuild_run_best_plots.py \
      --run-dir analysis_data/1029_Baseline_Balanced_Resnet/1029_Baseline_Balanced_Resnet_20251029_163906 \
      --head image
"""


import argparse
import logging
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import auc, roc_auc_score, roc_curve  # noqa: E402

from classes import build_papila_clinical  # noqa: E402
from classes.hypertower import HyperTower  # noqa: E402

try:  # Allow checkpoints that stored pandas DataFrames in their args.
    from torch.serialization import add_safe_globals  # type: ignore

    add_safe_globals([pd.DataFrame])
except (ImportError, AttributeError):
    pass


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Rebuild ROC plots for an existing multifold run.")
    ap.add_argument("--run-dir", required=True, type=Path, help="Path to the run directory under analysis_data.")
    ap.add_argument("--head", default="image", choices=["image", "fused", "metadata"], help="Which prediction head to plot.")
    ap.add_argument("--class-names", nargs="*", default=None, help="Optional class names to control plot labels.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .npy probability dumps if present.")
    ap.add_argument(
        "--use-holdout",
        action="store_true",
        help="Evaluate checkpoints on the saved holdout set instead of the fold validation splits.",
    )
    return ap.parse_args()


def load_cli_args(run_dir: Path) -> dict:
    cli_path = run_dir / "cli_args.json"
    if not cli_path.exists():
        raise FileNotFoundError(f"Missing cli_args.json in {run_dir}")
    with cli_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_summary(run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json in {run_dir}")
    with summary_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_data_dir(raw_dir: str) -> str:
    """
    Resolve dataset paths saved in legacy cli_args.json.

    Older runs often store "ClinicalData"/"FundusImages" relative to a
    dataset root, while current repo layout uses "Papila/<dir>".
    """
    p = Path(raw_dir)
    if p.exists():
        return str(p)

    candidates = [
        REPO_ROOT / p,
        REPO_ROOT / "Papila" / p,
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand)

    return str(p)


def prepare_clinical(cli_args: dict, run_dir: Path) -> tuple:
    image_dir = resolve_data_dir(cli_args["image_dir"])
    clinical_dir = resolve_data_dir(cli_args["clinical_dir"])
    clinical = build_papila_clinical(
        image_dir,
        clinical_dir,
        cli_args["label_col"],
        cli_args["cat_cols"],
        n_splits=cli_args["n_splits"],
        random_seed=cli_args["fold_seed"],
    )

    holdout_path = run_dir / "holdout.csv"
    holdout_df = pd.read_csv(holdout_path) if holdout_path.exists() else None
    if holdout_df is not None:
        if cli_args["eval_mode"] == "binary":
            holdout_df = holdout_df[holdout_df[cli_args["label_col"]].isin([0, 1])].reset_index(drop=True)

        join_cols = [c for c in holdout_df.columns if c in clinical.df.columns]
        if not join_cols:
            raise RuntimeError("Holdout CSV found but no overlapping columns with clinical dataframe.")
        marker = holdout_df.assign(_holdout_marker=1)
        merged = clinical.df.merge(marker, on=join_cols, how="left")
        train_df = merged[merged["_holdout_marker"].isna()].drop(columns=["_holdout_marker"]).reset_index(drop=True)
        clinical.frames = [train_df.copy()]
        clinical.df = train_df.copy()
        clinical._infer_or_validate_feature_types()
        clinical._compute_numeric_stats()
        clinical._build_cat_maps()
        clinical._compute_feature_dim()
        clinical._build_kfold_indices()
    return clinical, holdout_df


def build_ht_args(cli_args: dict, fold: int, run_dir: Path, models_dir: Path, holdout_df):
    image_dir = resolve_data_dir(cli_args["image_dir"])
    clinical_dir = resolve_data_dir(cli_args["clinical_dir"])
    # Copy of the training-time namespace so HyperTower can be re-instantiated.
    return SimpleNamespace(
        image_dir=image_dir,
        clinical_dir=clinical_dir,
        label_col=cli_args["label_col"],
        cat_cols=cli_args["cat_cols"],
        batch_size=cli_args["batch_size"],
        epochs=cli_args["epochs"],
        lr=cli_args["lr"],
        num_classes=cli_args["num_classes"],
        img_augment=cli_args.get("img_augment", True),
        focal_gamma=cli_args.get("focal_gamma", 0.0),
        eval_mode=cli_args["eval_mode"],
        fold=fold,
        run_dir=str(run_dir),
        models_dir=str(models_dir),
        backbone=cli_args["backbone"],
        freeze_ratio=cli_args["freeze_ratio"],
        fusion_mode=cli_args["fusion_mode"],
        use_se=cli_args.get("use_se", True),
        se_reduction=cli_args.get("se_reduction", 16),
        se_pre_norm=cli_args.get("se_pre_norm", True),
        se_where=cli_args.get("se_where", "bridge"),
        se_reduction_tower=cli_args.get("se_reduction_tower", cli_args.get("se_reduction", 16)),
        se_pre_norm_tower=cli_args.get("se_pre_norm_tower", cli_args.get("se_pre_norm", True)),
        warmup_tower_epochs=cli_args.get("warmup_tower_epochs", 0),
        warmup_fused_epochs=cli_args.get("warmup_fused_epochs", 0),
        gradual_thaw=cli_args.get("gradual_thaw", False),
        thaw_phase_duration=cli_args.get("thaw_phase_duration", 5),
        thaw_ratio=cli_args.get("thaw_ratio", 0.33),
        thaw_target=cli_args.get("thaw_target", "image"),
        thaw_start_epoch=cli_args.get("thaw_start_epoch", -1),
        initial_freeze=cli_args.get("initial_freeze", False),
        bcd_prob=0.5,
        bcd_p0=0.20,
        bcd_min=0.05,
        bcd_max=0.30,
        bcd_k=0.4,
        bcd_metric="auc",
        bcd_alpha_batch=0.2,
        bcd_alpha_tower=0.3,
        bcd_explore_floor=0.15,
        aux_img=0.05,
        aux_md=0.05,
        aux_detach=True,
        ema_alpha=0.9,
        entropy_ema=0.7,
        early_stop=cli_args.get("early_stop", False),
        early_metric=cli_args.get("early_metric"),
        early_mode=cli_args.get("early_mode", "auto"),
        early_patience=cli_args.get("early_patience", 7),
        early_min_delta=cli_args.get("early_min_delta", 0.0),
        checkpoint_best=cli_args.get("checkpoint_best", False),
        holdout_df=holdout_df,
        img_crop_manifest=cli_args.get("img_crop_manifest"),
        img_crop_weights=cli_args.get("img_crop_weights"),
        img_crop_normalize=cli_args.get("img_crop_normalize"),
        img_crop_threshold=cli_args.get("img_crop_threshold"),
        img_crop_scale=cli_args.get("img_crop_scale", 2.5),
        img_crop_size=cli_args.get("img_crop_size", 224),
        img_crop_cache=cli_args.get("img_crop_cache"),
        img_crop_tta=cli_args.get("img_crop_tta", False),
        img_crop_gt=cli_args.get("img_crop_gt", False),
        img_geometry_features=cli_args.get("img_geometry_features", False),
        balanced_sampler=cli_args.get("balanced_sampler", False),
    )


def collect_logits(ht, loader):
    """Mirror Multifold.eval_collect_logits but for a provided loader."""
    device = ht.device
    ht.img_tower.eval()
    ht.md_tower.eval()
    outputs = []
    with torch.no_grad():
        if ht.mode == "vote":
            ht.head_img.eval()
            ht.head_md.eval()
            ht.vote.eval()
        else:
            ht.bridge.eval()

    for batch in loader:
        if len(batch) == 4:
            imgs, metas, geometry, labels = batch
        else:
            imgs, metas, labels = batch
            geometry = None
        imgs = imgs.to(device)
        metas = metas.to(device)
        labels = labels.to(device)
        if geometry is not None and geometry.numel() > 0:
            geometry = geometry.to(device)
        else:
            geometry = None
        if ht.mode == "vote":
            img_feats = ht.img_tower(imgs, geometry)
            md_feats = ht.md_tower(metas)
            out_img = ht.head_img(img_feats)
            out_md = ht.head_md(md_feats)
            out_fused = ht.vote(out_img, out_md)
        else:
            img_feats = ht.img_tower(imgs, geometry)
            md_feats = ht.md_tower(metas)
            result = ht.bridge(img_feats, md_feats)
            if isinstance(result, tuple):
                out_fused, out_img, out_md = result
            else:
                out_fused, out_img, out_md = result, None, None

            outputs.append(
                (
                    labels.detach().cpu().numpy(),
                    torch.softmax(out_fused, dim=1).detach().cpu().numpy() if out_fused is not None else None,
                    torch.softmax(out_img, dim=1).detach().cpu().numpy() if out_img is not None else None,
                    torch.softmax(out_md, dim=1).detach().cpu().numpy() if out_md is not None else None,
                )
            )

    if not outputs:
        return np.array([]), None, None, None

    y_all, pf, pi, pm = zip(*outputs)
    y_true = np.concatenate(y_all, axis=0)
    probs_f = np.concatenate([p for p in pf if p is not None], axis=0) if any(p is not None for p in pf) else None
    probs_i = np.concatenate([p for p in pi if p is not None], axis=0) if any(p is not None for p in pi) else None
    probs_m = np.concatenate([p for p in pm if p is not None], axis=0) if any(p is not None for p in pm) else None
    return y_true, probs_f, probs_i, probs_m


def compute_per_class_curves(y_true, probs):
    if probs is None:
        return {}
    num_classes = probs.shape[1]
    curves = {}
    for k in range(num_classes):
        y_bin = (y_true == k).astype(np.uint8)
        fpr, tpr, _ = roc_curve(y_bin, probs[:, k])
        curves[k] = {"fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr) if len(fpr) > 1 else np.nan}
    return curves


def choose_head_probs(head: str, probs_f, probs_i, probs_m):
    if head == "fused":
        return probs_f
    if head == "metadata":
        return probs_m
    return probs_i


def legacy_probs_suffix(head: str) -> str:
    if head == "image":
        return "img"
    if head == "metadata":
        return "md"
    return "fused"


def ensure_binary_slice(y_true, *arrays):
    mask = np.isin(y_true, [0, 1])
    filtered = [y_true[mask]]
    for arr in arrays:
        if arr is None:
            filtered.append(None)
        else:
            filtered.append(arr[mask])
    return filtered


def plot_overlays(
    per_fold_curves,
    out_dir: Path,
    class_names: list[str],
    head: str,
    eval_mode: str,
    suffix: str = "",
):
    keys = sorted({k for _, curves in per_fold_curves for k in curves.keys()})
    if eval_mode == "binary":
        keys = [k for k in keys if k == 1]
    if not keys:
        return
    name_map = {k: (class_names[k] if k < len(class_names) else f"class_{k}") for k in keys}
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in keys:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="grey")
        for fold_idx, curves in per_fold_curves:
            if k not in curves:
                continue
            fpr = curves[k]["fpr"]
            tpr = curves[k]["tpr"]
            auc_val = curves[k]["auc"]
            label = f"Fold {fold_idx} (AUC {auc_val:.3f})" if auc_val == auc_val else f"Fold {fold_idx}"
            ax.plot(fpr, tpr, linewidth=1.5, label=label)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{head} head — {name_map[k]} ROC per fold")
        ax.legend(loc="lower right")
        fig.tight_layout()
        safe_name = name_map[k].replace(" ", "_")
        suffix_str = suffix if suffix else ""
        fig.savefig(out_dir / f"roc_{head}_{safe_name}_perfold{suffix_str}.png", dpi=160)
        plt.close(fig)


def plot_mean_sd(
    per_fold_curves,
    out_dir: Path,
    class_names: list[str],
    head: str,
    eval_mode: str,
    suffix: str = "",
):
    keys = sorted({k for _, curves in per_fold_curves for k in curves.keys()})
    if eval_mode == "binary":
        keys = [k for k in keys if k == 1]
    if not keys:
        return
    grid = np.linspace(0, 1, 501)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="grey")
    for k in keys:
        tprs = []
        aucs = []
        for _, curves in per_fold_curves:
            if k not in curves:
                continue
            fpr = curves[k]["fpr"]
            tpr = curves[k]["tpr"]
            aucs.append(curves[k]["auc"])
            tprs.append(np.interp(grid, fpr, tpr))
        if not tprs:
            continue
        tprs = np.vstack(tprs)
        mean = tprs.mean(axis=0)
        std = tprs.std(axis=0)
        label = class_names[k] if k < len(class_names) else f"class_{k}"
        label = f"{label} (AUC {np.nanmean(aucs):.3f}±{np.nanstd(aucs):.3f})"
        ax.plot(grid, mean, linewidth=2, label=label)
        ax.fill_between(grid, np.maximum(mean - std, 0), np.minimum(mean + std, 1), alpha=0.15)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Mean OVR ROC (±1 SD) — {head} head")
    ax.legend(loc="lower right")
    fig.tight_layout()
    suffix_str = suffix if suffix else ""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"roc_{head}_mean_ovr{suffix_str}.png", dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    cli_args = load_cli_args(run_dir)
    summary = load_summary(run_dir)

    class_names = (
        args.class_names
        if args.class_names
        else (cli_args.get("class_names") or (["Healthy", "Glaucoma"] if cli_args["eval_mode"] == "binary" else ["Healthy", "Glaucoma", "Suspect"]))
    )

    shortname = cli_args.get("shortname") or run_dir.parent.name
    run_id = cli_args.get("run_id") or run_dir.name
    base_models_dir = Path("models") / shortname / run_id

    clinical, holdout_df = prepare_clinical(cli_args, run_dir)
    if args.use_holdout and holdout_df is None:
        raise SystemExit("Holdout metrics requested but no holdout.csv found for this run.")

    per_fold_curves = []
    fold_aucs = []
    head = args.head
    file_suffix = "_holdout" if args.use_holdout else ""

    for fold_entry in summary.get("fold_metrics", []):
        fold_idx = int(fold_entry["fold"])
        best_epoch = fold_entry.get("best_epoch")
        if not best_epoch:
            print(f"[skip] Fold {fold_idx}: no best_epoch recorded.")
            continue

        # Prefer saved fold arrays when available. This avoids reconstructing
        # HyperTower for legacy runs whose external weight paths no longer exist.
        base = run_dir / f"fold{fold_idx}{file_suffix}"
        y_path = Path(f"{base}_y_true.npy")
        p_path = Path(f"{base}_probs_{legacy_probs_suffix(head)}.npy")
        if y_path.exists() and p_path.exists():
            y_true = np.load(y_path)
            head_probs = np.load(p_path)
            if cli_args["eval_mode"] == "binary" and head_probs.shape[1] >= 2:
                head_probs = head_probs[:, :2]
            curves = compute_per_class_curves(y_true, head_probs)
            per_fold_curves.append((fold_idx, curves))
            try:
                if head_probs.shape[1] > 2:
                    fold_auc = roc_auc_score(y_true, head_probs, multi_class="ovr", average="macro")
                else:
                    target_scores = head_probs[:, 1] if head_probs.shape[1] > 1 else head_probs[:, 0]
                    fold_auc = roc_auc_score(y_true, target_scores)
                fold_aucs.append(fold_auc)
                print(
                    f"[info] Fold {fold_idx}: using saved arrays "
                    f"({y_path.name}, {p_path.name}), AUC={fold_auc:.4f}"
                )
            except Exception:
                print(
                    f"[warning] Fold {fold_idx}: using saved arrays "
                    f"({y_path.name}, {p_path.name}) but AUC failed."
                )
            continue

        fold_models_dir = base_models_dir / f"fold{fold_idx}"
        best_checkpoint = fold_models_dir / "model_best.pt"
        if not best_checkpoint.exists():
            print(
                f"[warning] Fold {fold_idx}: missing model_best.pt at {best_checkpoint} "
                f"and missing fallback arrays {y_path.name}/{p_path.name}"
            )
            continue

        ht_args = build_ht_args(cli_args, fold_idx, run_dir, fold_models_dir, holdout_df)
        ht = HyperTower(clinical, ht_args)
        for handler in list(ht.logger.handlers):
            handler.close()
        ht.logger.handlers = [logging.NullHandler()]
        train_log_path = Path("train.log")
        if train_log_path.exists() and train_log_path.stat().st_size == 0:
            try:
                train_log_path.unlink()
            except OSError:
                pass
        try:
            state = torch.load(best_checkpoint, map_location=ht.device, weights_only=False)
        except TypeError:
            state = torch.load(best_checkpoint, map_location=ht.device)
        ht._restore_from_state(state)
        if args.use_holdout:
            eval_df = holdout_df.copy()
        else:
            _, eval_df = clinical.get_split_dfs(fold_idx)
        if cli_args["eval_mode"] == "binary":
            eval_df = eval_df[eval_df[cli_args["label_col"]].isin([0, 1])].reset_index(drop=True)
        if eval_df.empty:
            print(f"[warning] Fold {fold_idx}: evaluation dataframe is empty; skipping.")
            continue
        ht.test_loader = ht._make_loader_for_df(eval_df, is_train=False)

        y_true, probs_f, probs_i, probs_m = collect_logits(ht, ht.test_loader)
        if cli_args["eval_mode"] == "binary":
            y_true, probs_f, probs_i, probs_m = ensure_binary_slice(y_true, probs_f, probs_i, probs_m)

        head_probs = choose_head_probs(head, probs_f, probs_i, probs_m)
        if head_probs is None:
            print(f"[skip] Fold {fold_idx}: head '{head}' not available.")
            continue

        if head_probs.shape[1] >= 2:
            head_probs = head_probs[:, :2]

        if args.overwrite:
            base = run_dir / f"fold{fold_idx}{file_suffix}"
            np.save(f"{base}_y_true.npy", y_true)
            if probs_f is not None:
                np.save(f"{base}_probs_fused.npy", probs_f)
            if probs_i is not None:
                np.save(f"{base}_probs_img.npy", probs_i)
            if probs_m is not None:
                np.save(f"{base}_probs_md.npy", probs_m)

        curves = compute_per_class_curves(y_true, head_probs)
        per_fold_curves.append((fold_idx, curves))
        try:
            if head_probs.shape[1] > 2:
                fold_auc = roc_auc_score(y_true, head_probs, multi_class="ovr", average="macro")
            else:
                target_scores = head_probs[:, 1] if head_probs.shape[1] > 1 else head_probs[:, 0]
                fold_auc = roc_auc_score(y_true, target_scores)
            fold_aucs.append(fold_auc)
            print(f"[info] Fold {fold_idx}: best epoch {best_epoch}, AUC={fold_auc:.4f}")
        except Exception:
            print(f"[warning] Fold {fold_idx}: unable to compute AUC.")

    if not per_fold_curves:
        raise SystemExit("No folds processed; nothing to plot.")

    plots_dir = run_dir / "plots"
    plot_overlays(per_fold_curves, plots_dir, class_names, head, cli_args["eval_mode"], file_suffix)
    plot_mean_sd(per_fold_curves, plots_dir, class_names, head, cli_args["eval_mode"], file_suffix)

    if fold_aucs:
        print(f"[info] {head} head mean AUC across folds: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"Plots regenerated under {plots_dir}")


if __name__ == "__main__":
    main()
