#!/usr/bin/env python3
"""Compare suspect AUC from image tower vs crop-derived geometry (CDR)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classes import build_papila_clinical
from classes.hypertower import UNetImageCropper, ManifestImageCropper


# ---------------------------
# Config (edit in IDE)
# ---------------------------
RUN_DIRS = [
    Path("analysis_data/1030_Balanced_Unet_Perimg_Resnet_SE16NormB_SE16NormT_multi_fused/1030_Balanced_Unet_Perimg_Resnet_SE16NormB_SE16NormT_multi_fused_20251030_091842"),
    Path("analysis_data/1030_Balanced_GT_Perimg_Resnet_SE16NormB_SE16NormT_multi_fused/1030_Balanced_GT_Perimg_Resnet_SE16NormB_SE16NormT_multi_fused_20251030_113730"),
]
GEOM_CACHE_ROOT = Path("analysis_data/geometry_cache")
SUSPECT_LABEL = 2


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _drop_holdout_rows(clinical, holdout_path: Path) -> None:
    if not holdout_path.exists():
        return
    holdout = pd.read_csv(holdout_path)
    if holdout.empty:
        return
    if "Patient ID" not in holdout.columns or "eyeID" not in holdout.columns:
        return
    holdout_keys = set(zip(holdout["Patient ID"].astype(int), holdout["eyeID"].astype(str)))
    df = clinical.df.copy()
    df["_key"] = list(zip(df["Patient ID"].astype(int), df["eyeID"].astype(str)))
    df = df[~df["_key"].isin(holdout_keys)].drop(columns=["_key"]).reset_index(drop=True)

    clinical.frames = [df.copy()]
    clinical.df = df.copy()
    clinical._infer_or_validate_feature_types()
    clinical._compute_numeric_stats()
    clinical._build_cat_maps()
    clinical._compute_feature_dim()
    clinical._build_kfold_indices()


def _make_cropper(args: Dict, cache_dir: Path):
    manifest = args.get("img_crop_manifest")
    if not manifest:
        raise RuntimeError("img_crop_manifest missing; cannot compute geometry features.")

    scale = float(args.get("img_crop_scale", 2.5))
    target_size = int(args.get("img_crop_size", 224))
    use_gt = bool(args.get("img_crop_gt", False))

    if use_gt:
        return ManifestImageCropper(
            manifest_path=Path(manifest),
            scale=scale,
            target_size=target_size,
            cache_dir=cache_dir,
        )

    weights = args.get("img_crop_weights")
    if not weights:
        raise RuntimeError("img_crop_weights missing for UNet cropper.")

    normalize = args.get("img_crop_normalize", "per_image")
    threshold = float(args.get("img_crop_threshold", 0.5))
    tta = bool(args.get("img_crop_tta", False))
    return UNetImageCropper(
        manifest_path=Path(manifest),
        weights_path=Path(weights),
        normalize=normalize,
        threshold=threshold,
        tta=tta,
        scale=scale,
        target_size=target_size,
        cache_dir=cache_dir,
    )


def _geometry_scores(
    clinical,
    cropper,
    test_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    scores: List[float] = []
    keep_mask: List[bool] = []
    for _, row in test_df.iterrows():
        img_path = clinical.get_image_path(row)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            scores.append(float("nan"))
            keep_mask.append(False)
            continue
        feats = cropper.geometry_features(image, img_path)
        if feats is None or len(feats) == 0:
            scores.append(float("nan"))
            keep_mask.append(False)
        else:
            scores.append(float(feats[0]))  # area_ratio (CDR)
            keep_mask.append(True)
    return np.asarray(scores, dtype=float), np.asarray(keep_mask, dtype=bool)


def _suspect_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = (y_true == SUSPECT_LABEL).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, scores))


def main() -> None:
    rows: List[Dict[str, object]] = []

    for run_dir in RUN_DIRS:
        cli_path = run_dir / "cli_args.json"
        cli_args = _load_json(cli_path)
        if not cli_args:
            print(f"[warn] Missing cli_args.json in {run_dir}")
            continue

        label_col = cli_args.get("label_col", "Diagnosis")
        cat_cols = cli_args.get("cat_cols", ["Gender", "Phakic/Pseudophakic"])
        n_splits = int(cli_args.get("n_splits", 5))
        fold_seed = int(cli_args.get("fold_seed", 42))
        eval_mode = str(cli_args.get("eval_mode", "multiclass")).lower()

        clinical = build_papila_clinical(
            image_dir=cli_args.get("image_dir", "Papila/FundusImages"),
            clinical_dir=cli_args.get("clinical_dir", "Papila/ClinicalData"),
            label_col=label_col,
            cat_cols=cat_cols,
            n_splits=n_splits,
            random_seed=fold_seed,
        )

        if eval_mode == "binary":
            clinical.df = clinical.df[clinical.df[label_col].isin([0, 1])].reset_index(drop=True)
            clinical.frames = [clinical.df.copy()]
            clinical._infer_or_validate_feature_types()
            clinical._compute_numeric_stats()
            clinical._build_cat_maps()
            clinical._compute_feature_dim()
            clinical._build_kfold_indices()

        _drop_holdout_rows(clinical, run_dir / "holdout.csv")

        cache_dir = GEOM_CACHE_ROOT / run_dir.name
        cache_dir.mkdir(parents=True, exist_ok=True)
        cropper = _make_cropper(cli_args, cache_dir=cache_dir)

        all_geom_scores: List[float] = []
        all_img_scores: List[float] = []
        all_y: List[int] = []

        for fold in range(n_splits):
            y_path = run_dir / f"fold{fold}_y_true.npy"
            p_img_path = run_dir / f"fold{fold}_probs_img.npy"
            if not y_path.exists() or not p_img_path.exists():
                continue

            y_true = np.load(y_path)
            probs_img = np.load(p_img_path)
            if probs_img.ndim != 2 or probs_img.shape[1] <= SUSPECT_LABEL:
                continue

            _, test_df = clinical.get_split_dfs(fold)
            if len(test_df) != len(y_true):
                print(
                    f"[warn] {run_dir.name} fold{fold}: test_df len {len(test_df)} != y_true len {len(y_true)}"
                )

            geom_scores, keep_mask = _geometry_scores(clinical, cropper, test_df)
            if keep_mask.sum() == 0:
                print(f"[warn] {run_dir.name} fold{fold}: no valid geometry features")
                continue

            y_fold = y_true[: len(geom_scores)][keep_mask]
            geom_fold = geom_scores[keep_mask]
            img_fold = probs_img[: len(geom_scores), SUSPECT_LABEL][keep_mask]

            geom_auc = _suspect_auc(y_fold, geom_fold)
            img_auc = _suspect_auc(y_fold, img_fold)

            rows.append(
                {
                    "run": run_dir.name,
                    "fold": fold,
                    "metric": "suspect_auc",
                    "image_auc": img_auc,
                    "geometry_auc": geom_auc,
                    "n": int(len(y_fold)),
                }
            )

            all_geom_scores.append(geom_fold)
            all_img_scores.append(img_fold)
            all_y.append(y_fold)

        if all_y:
            y_all = np.concatenate(all_y)
            geom_all = np.concatenate(all_geom_scores)
            img_all = np.concatenate(all_img_scores)
            rows.append(
                {
                    "run": run_dir.name,
                    "fold": "all",
                    "metric": "suspect_auc",
                    "image_auc": _suspect_auc(y_all, img_all),
                    "geometry_auc": _suspect_auc(y_all, geom_all),
                    "n": int(len(y_all)),
                }
            )

    if not rows:
        raise SystemExit("No results produced; check run paths and files.")

    df = pd.DataFrame(rows)
    out_path = GEOM_CACHE_ROOT / "suspect_auc_geometry_vs_image.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
