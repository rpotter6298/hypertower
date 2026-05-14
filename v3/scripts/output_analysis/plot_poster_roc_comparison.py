#!/usr/bin/env python3
"""
Poster-facing ROC comparison for PAPILA binary glaucoma classification.

Builds one combined ROC figure with four curves:
  - MD Only
  - Image Only
  - Single Fusion
  - Ensemble Fusion

The default setup mixes two artifact layouts:
  1. Dedicated unimodal runs saved under analysis_data/.../fold*/{y_true,probs}.npy
  2. Repeated CV fusion runs saved under v3/results/.../rep*/.../predictions_test.csv

Update DEFAULT_CURVES below if you want different source runs or labels.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO_ROOT / "v3" / "results"
DEFAULT_OUT = REPO_ROOT / "results" / "poster" / "papila_binary_roc_comparison.png"


@dataclass(frozen=True)
class CurveSpec:
    label: str
    source_kind: str
    color: str
    mode_dir: str | None = None
    probs_stem: str | None = None
    run: str | None = None
    tower_path: str | None = None
    score_col: str | None = None


DEFAULT_CURVES = [
    CurveSpec(
        label="Clinical Data Only",
        source_kind="legacy_npy",
        color="#4c72b0",
        mode_dir="analysis_data/pipeline_mdonly_500ep/binary/single",
        probs_stem="probs_classic",
    ),
    CurveSpec(
        label="Image Only",
        source_kind="v3_csv",
        color="#dd8452",
        run="phase2/imageonly_resnet50_proper",
        tower_path="binary/single",
        score_col="prob_img_c1",
    ),
    CurveSpec(
        label="Single Fusion",
        source_kind="v3_csv",
        color="#55a868",
        run="phase5/single_fused",
        tower_path="binary/single",
        score_col="prob_fused_c1",
    ),
    CurveSpec(
        label="Ensemble Fusion",
        source_kind="v3_csv",
        color="#c44e52",
        run="phase5/logit_mlp_head",
        tower_path="binary/ensemble",
        score_col="prob_fused_c1",
    ),
]


def load_v3_csv_predictions(run: str, tower_path: str, score_col: str) -> pd.DataFrame:
    run_dir = RESULTS_ROOT / run
    rows: list[pd.DataFrame] = []
    for rep_dir in sorted(run_dir.glob("rep*")):
        mode_dir = rep_dir / tower_path
        if not mode_dir.exists():
            continue
        for fold_dir in sorted(mode_dir.glob("fold[0-9]")):
            csv_path = fold_dir / "predictions_test.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path, usecols=["y_true", score_col])
            df["rep"] = rep_dir.name
            df["fold"] = fold_dir.name
            rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No predictions found under {run_dir}/{tower_path}")
    return pd.concat(rows, ignore_index=True)


def load_legacy_npy_predictions(mode_dir: str, probs_stem: str) -> pd.DataFrame:
    root = REPO_ROOT / mode_dir
    rows: list[pd.DataFrame] = []
    for fold_dir in sorted(root.glob("fold[0-9]")):
        y_path = fold_dir / "y_true.npy"
        p_path = fold_dir / f"{probs_stem}.npy"
        if not y_path.exists() or not p_path.exists():
            continue
        y_true = np.load(y_path)
        probs = np.load(p_path)
        if probs.ndim == 2:
            if probs.shape[1] < 2:
                raise ValueError(f"Expected 2-class probs in {p_path}")
            scores = probs[:, 1]
        else:
            scores = probs
        rows.append(
            pd.DataFrame(
                {
                    "y_true": y_true,
                    "score": scores,
                    "rep": "rep00",
                    "fold": fold_dir.name,
                }
            )
        )
    if not rows:
        raise FileNotFoundError(f"No fold artifacts found under {root}")
    return pd.concat(rows, ignore_index=True)


def load_curve_predictions(spec: CurveSpec) -> tuple[pd.DataFrame, str]:
    if spec.source_kind == "v3_csv":
        if spec.run is None or spec.tower_path is None or spec.score_col is None:
            raise ValueError(f"Incomplete v3_csv spec: {spec}")
        return load_v3_csv_predictions(spec.run, spec.tower_path, spec.score_col), spec.score_col
    if spec.source_kind == "legacy_npy":
        if spec.mode_dir is None or spec.probs_stem is None:
            raise ValueError(f"Incomplete legacy_npy spec: {spec}")
        return load_legacy_npy_predictions(spec.mode_dir, spec.probs_stem), "score"
    raise ValueError(f"Unknown source_kind: {spec.source_kind}")


def build_curve_summary(df: pd.DataFrame, score_col: str) -> dict[str, object]:
    if score_col not in df.columns:
        raise KeyError(f"Missing score column '{score_col}'")

    mean_fpr = np.linspace(0.0, 1.0, 501)
    tprs: list[np.ndarray] = []
    aucs: list[float] = []
    fold_count = 0

    for (_, _), fold_df in df.groupby(["rep", "fold"], sort=True):
        y_true = fold_df["y_true"].to_numpy()
        scores = fold_df[score_col].to_numpy()
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, scores)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0
        tprs.append(interp_tpr)
        aucs.append(float(roc_auc_score(y_true, scores)))
        fold_count += 1

    if not tprs:
        raise ValueError(f"No valid binary folds found for '{score_col}'")

    tpr_arr = np.vstack(tprs)
    y_all = df["y_true"].to_numpy()
    s_all = df[score_col].to_numpy()

    return {
        "fpr": mean_fpr,
        "tpr_mean": tpr_arr.mean(axis=0),
        "tpr_std": tpr_arr.std(axis=0),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "auc_pooled": float(roc_auc_score(y_all, s_all)),
        "fold_count": fold_count,
        "n_total": int(len(df)),
    }


def plot_curves(curve_summaries: list[tuple[CurveSpec, dict[str, object]]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.8, 7.0))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="0.5", alpha=0.8)

    for spec, summary in curve_summaries:
        fpr = np.asarray(summary["fpr"])
        tpr_mean = np.asarray(summary["tpr_mean"])
        tpr_std = np.asarray(summary["tpr_std"])
        auc_mean = float(summary["auc_mean"])
        auc_std = float(summary["auc_std"])

        ax.plot(
            fpr,
            tpr_mean,
            color=spec.color,
            linewidth=2.4,
            label=f"{spec.label} (AUC {auc_mean:.3f} ± {auc_std:.3f})",
        )
        ax.fill_between(
            fpr,
            np.clip(tpr_mean - tpr_std, 0, 1),
            np.clip(tpr_mean + tpr_std, 0, 1),
            color=spec.color,
            alpha=0.12,
        )

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("PAPILA Binary ROC Comparison")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary(curve_summaries: list[tuple[CurveSpec, dict[str, object]]], out_path: Path) -> None:
    summary_path = out_path.with_suffix(".json")
    payload = {
        "figure": str(out_path),
        "curves": [
            {
                "label": spec.label,
                "source_kind": spec.source_kind,
                "mode_dir": spec.mode_dir,
                "run": spec.run,
                "tower_path": spec.tower_path,
                "score_col": spec.score_col,
                "probs_stem": spec.probs_stem,
                "auc_mean": float(summary["auc_mean"]),
                "auc_std": float(summary["auc_std"]),
                "auc_pooled": float(summary["auc_pooled"]),
                "fold_count": int(summary["fold_count"]),
                "n_total": int(summary["n_total"]),
            }
            for spec, summary in curve_summaries
        ],
    }
    summary_path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output PNG path (default: {DEFAULT_OUT})",
    )
    args = ap.parse_args()

    curve_summaries: list[tuple[CurveSpec, dict[str, object]]] = []
    for spec in DEFAULT_CURVES:
        df, score_col = load_curve_predictions(spec)
        summary = build_curve_summary(df, score_col)
        curve_summaries.append((spec, summary))
        print(
            f"{spec.label:16s} "
            f"AUC={summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}  "
            f"pooled={summary['auc_pooled']:.4f}  "
            f"folds={summary['fold_count']}"
        )

    plot_curves(curve_summaries, args.output)
    write_summary(curve_summaries, args.output)
    print(f"Saved figure: {args.output}")
    print(f"Saved summary: {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
