"""
Phase 5 explainability — confidence strips and head comparison.

Works from saved prediction CSVs. If patient_id column is present (requires
a re-run after the v3_hypertower.py update), points are colored by VFI
severity group. Otherwise falls back to a single color per true class.

VFI severity groups (VF_MD from clinical data):
  Early    VF_MD > -6
  Moderate VF_MD -6 to -12
  Severe   VF_MD < -12

Produces (all in figures/explainability/):
  confidence_strips.png   — vertical strip: P(glaucoma) by true class, VFI colored
  head_comparison.png     — fused vs img vs md distributions side by side

Usage:
    python -m v3.scripts.output_analysis.explainability.confidence_strips
    python -m v3.scripts.output_analysis.explainability.confidence_strips \
        --run phase5/logit_mlp_head --clinical-dir Papila/ClinicalData
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
RESULTS_ROOT = REPO_ROOT / "v3" / "results"
FIGURES_ROOT = REPO_ROOT / "v3" / "figures"
CLINICAL_DIR = REPO_ROOT / "Papila" / "ClinicalData"

FONT = "DejaVu Sans"

# Colour palette
C_NORMAL = "#78909C"  # blue-grey   — healthy controls (no VFI staging)
C_EARLY = "#29B6F6"  # sky-blue    — glaucoma, early VFI loss
C_MODERATE = "#FFB300"  # amber       — glaucoma, moderate VFI loss
C_SEVERE = "#E53935"  # vivid red   — glaucoma, severe VFI loss
C_UNKNOWN = "#BDBDBD"  # light grey  — glaucoma, VFI not recorded

SEV_LABELS = {
    "normal": "Normal",
    "early": "Glaucoma — early (VF_MD > −6)",
    "moderate": "Glaucoma — moderate (−12 to −6)",
    "severe": "Glaucoma — severe (VF_MD < −12)",
    "unknown": "Glaucoma — VF_MD not recorded",
}
SEV_COLORS = {
    "normal": C_NORMAL,
    "early": C_EARLY,
    "moderate": C_MODERATE,
    "severe": C_SEVERE,
    "unknown": C_UNKNOWN,
}

HEAD_COLORS = {"fused": "#d4a017", "img": "#4e8d3a", "md": "#4c72b0"}
HEAD_LABELS = {
    "fused": "Fused head",
    "img": "Image-only head",
    "md": "Clinical-only head",
}


# ── VFI data ──────────────────────────────────────────────────────────────────


def load_vfi(clinical_dir: Path) -> pd.DataFrame:
    """Return DataFrame with columns [patient_id (int), vf_md (float), severity (str)].
    Only includes patients in the binary study (Diagnosis 0=Normal, 1=Glaucoma).
    """
    od = pd.read_excel(clinical_dir / "patient_data_od.xlsx", header=1)
    os_ = pd.read_excel(clinical_dir / "patient_data_os.xlsx", header=1)

    def _clean(df, eye):
        df = df.copy()
        if "Patient ID" not in df.columns and "ID" in df.columns:
            df.rename(columns={"ID": "Patient ID"}, inplace=True)
        df["Patient ID"] = (
            df["Patient ID"].astype(str).str.extract(r"(\d+)")[0].astype(int)
        )
        df["Diagnosis"] = pd.to_numeric(df["Diagnosis"], errors="coerce")
        df["VF_MD"] = pd.to_numeric(df["VF_MD"], errors="coerce")
        # PAPILA: 0=Normal, 1=Glaucoma, 2=Suspect — keep only binary patients
        df = df[df["Diagnosis"].isin([0, 1])].copy()
        df["eye"] = eye
        return df[["Patient ID", "Diagnosis", "VF_MD", "eye"]]

    combined = pd.concat([_clean(od, "OD"), _clean(os_, "OS")], ignore_index=True)

    # Per patient: modal diagnosis, worst (most negative) VF_MD across eyes
    diag = (
        combined.groupby("Patient ID")["Diagnosis"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
    )
    vf = combined.groupby("Patient ID")["VF_MD"].min().reset_index()
    worst = diag.merge(vf, on="Patient ID").rename(
        columns={"Patient ID": "patient_id", "VF_MD": "vf_md", "Diagnosis": "diagnosis"}
    )

    def _severity(row):
        if int(row["diagnosis"]) == 0:
            return "normal"  # healthy control — no VFI staging
        v = row["vf_md"]
        if pd.isna(v):
            return "unknown"  # glaucoma, no VFI recorded
        if v > -6:
            return "early"
        if v > -12:
            return "moderate"
        return "severe"

    worst["severity"] = worst.apply(_severity, axis=1)
    return worst


# ── Prediction loading ────────────────────────────────────────────────────────


def load_all_predictions(
    run_dir: Path, tower_path: str = "binary/ensemble"
) -> pd.DataFrame:
    """Pool predictions_test.csv across all reps and folds."""
    rows = []
    for rep_dir in sorted(run_dir.glob("rep*")):
        tm_dir = rep_dir / tower_path
        if not tm_dir.exists():
            continue
        for fold_dir in sorted(tm_dir.glob("fold[0-9]")):
            csv_path = fold_dir / "predictions_test.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            df["rep"] = rep_dir.name
            df["fold"] = fold_dir.name
            rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No predictions_test.csv found under {run_dir}")
    return pd.concat(rows, ignore_index=True)


# ── Figure 1: Confidence strips (vertical) ───────────────────────────────────


def make_confidence_strips(
    df: pd.DataFrame, vfi: pd.DataFrame | None, out_path: Path
) -> None:
    """
    Vertical strip plot: x = true class, y = P(glaucoma).
    Points colored by VFI severity if patient_id column available, else uniform.
    """
    rng = np.random.default_rng(42)
    has_vfi = (
        vfi is not None
        and "patient_id" in df.columns
        and df["patient_id"].notna().any()
    )

    if has_vfi:
        df = df.copy()
        df["patient_id"] = pd.to_numeric(df["patient_id"], errors="coerce").astype(
            "Int64"
        )
        vfi_merge = vfi.copy()
        vfi_merge["patient_id"] = vfi_merge["patient_id"].astype("Int64")
        df = df.merge(
            vfi_merge[["patient_id", "severity"]], on="patient_id", how="left"
        )
        df["severity"] = df["severity"].fillna("unknown")
    else:
        df["severity"] = "unknown"

    fig, ax = plt.subplots(figsize=(6, 7))
    fig.patch.set_facecolor("#e8e8e8")
    ax.set_facecolor("#e8e8e8")

    x_pos = {0: 0.0, 1: 1.0}
    jitter_scale = 0.18

    # Draw in severity order so severe is on top
    sev_order = ["normal", "unknown", "early", "moderate", "severe"]
    sev_alpha = {
        "normal": 0.40,
        "early": 0.55,
        "moderate": 0.70,
        "severe": 0.85,
        "unknown": 0.35,
    }
    sev_size = {"normal": 6, "early": 8, "moderate": 10, "severe": 12, "unknown": 6}

    for sev in sev_order:
        mask = df["severity"] == sev
        if not mask.any():
            continue
        sub = df[mask]
        jitter = rng.uniform(-jitter_scale, jitter_scale, len(sub))
        x = np.array([x_pos[int(v)] for v in sub["y_true"]]) + jitter
        ax.scatter(
            x,
            sub["prob_fused_c1"].values,
            c=SEV_COLORS[sev],
            s=sev_size[sev],
            alpha=sev_alpha[sev],
            linewidths=0,
            zorder=3,
            label=SEV_LABELS[sev],
        )

    # Median lines per class
    for cls, xc in x_pos.items():
        med = np.median(df.loc[df["y_true"] == cls, "prob_fused_c1"])
        ax.plot(
            [xc - jitter_scale - 0.04, xc + jitter_scale + 0.04],
            [med, med],
            color="#222",
            lw=2.0,
            zorder=5,
        )

    ax.axhline(0.5, color="#888", lw=1.2, ls="--", alpha=0.7, zorder=2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Normal", "Glaucoma"], fontsize=11)
    ax.set_ylabel("Predicted P(Glaucoma)", fontsize=11)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlim(-0.55, 1.55)
    ax.set_title(
        "Confidence Strips — Fused Head\n(Phase 5, all folds)",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3, zorder=1)

    # Legend — only show groups that appear
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles=handles,
            labels=labels,
            fontsize=8.5,
            loc="upper center",
            framealpha=0.75,
            ncol=2,
        )

    if not has_vfi:
        ax.text(
            0.98,
            0.02,
            "Re-run with updated v3_hypertower.py\nto enable VFI severity coloring",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.5,
            color="#888",
            style="italic",
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Figure 2: Head comparison ─────────────────────────────────────────────────


def make_head_comparison(df: pd.DataFrame, out_path: Path) -> None:
    """Side-by-side violin + strip of P(glaucoma) by true class for each head."""
    heads = ["fused", "img", "md"]
    prob_cols = {"fused": "prob_fused_c1", "img": "prob_img_c1", "md": "prob_md_c1"}

    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.5), sharey=True)
    fig.patch.set_facecolor("#e8e8e8")
    fig.suptitle(
        "Head Comparison — P(Glaucoma) by True Class (Phase 5, all folds)",
        fontsize=12,
        fontweight="bold",
    )

    c_normal = "#4c72b0"
    c_glaucoma = "#c44e52"

    for ax, head in zip(axes, heads):
        ax.set_facecolor("#e8e8e8")
        col = prob_cols[head]
        data_by_class = [df.loc[df["y_true"] == cls, col].values for cls in [0, 1]]

        vp = ax.violinplot(
            data_by_class,
            positions=[0, 1],
            widths=0.6,
            showmedians=True,
            showextrema=False,
        )
        for body, color in zip(vp["bodies"], [c_normal, c_glaucoma]):
            body.set_facecolor(color)
            body.set_alpha(0.35)
        vp["cmedians"].set_color("#222")
        vp["cmedians"].set_linewidth(2)

        for cls, color in zip([0, 1], [c_normal, c_glaucoma]):
            vals = data_by_class[cls]
            jitter = rng.uniform(-0.12, 0.12, len(vals))
            ax.scatter(
                cls + jitter, vals, color=color, s=4, alpha=0.30, linewidths=0, zorder=3
            )

        ax.axhline(0.5, color="#888", lw=1.0, ls="--", alpha=0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Normal", "Glaucoma"], fontsize=9)
        ax.set_title(
            HEAD_LABELS[head], fontsize=10, fontweight="bold", color=HEAD_COLORS[head]
        )
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", alpha=0.3)
        if head == "fused":
            ax.set_ylabel("Predicted P(Glaucoma)", fontsize=10)

        from sklearn.metrics import roc_auc_score

        try:
            auc = roc_auc_score(df["y_true"], df[col])
            ax.text(
                0.97,
                0.04,
                f"AUC = {auc:.3f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                color="#333",
                bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=2),
            )
        except Exception:
            pass

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Multi-model comparison strip ──────────────────────────────────────────────


def make_comparison_strips(
    run_configs: list[dict], vfi: pd.DataFrame, out_path: Path
) -> None:
    """
    Side-by-side confidence strips for multiple runs.
    Each config: {"label": str, "df": DataFrame}.
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(42)

    n = len(run_configs)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 7), sharey=True)
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor("#e8e8e8")
    fig.suptitle(
        "Confidence Strips by Model (Phase 5, all folds)",
        fontsize=13,
        fontweight="bold",
    )

    sev_order = ["normal", "unknown", "early", "moderate", "severe"]
    sev_alpha = {
        "normal": 0.40,
        "early": 0.55,
        "moderate": 0.70,
        "severe": 0.85,
        "unknown": 0.35,
    }
    sev_size = {"normal": 6, "early": 8, "moderate": 10, "severe": 12, "unknown": 6}
    x_pos = {0: 0.0, 1: 1.0}
    jitter_scale = 0.18

    for ax, cfg in zip(axes, run_configs):
        ax.set_facecolor("#e8e8e8")
        df = cfg["df"]

        # Attach VFI severity
        df = df.copy()
        df["patient_id"] = pd.to_numeric(df["patient_id"], errors="coerce").astype(
            "Int64"
        )
        vfi_m = vfi.copy()
        vfi_m["patient_id"] = vfi_m["patient_id"].astype("Int64")
        df = df.merge(vfi_m[["patient_id", "severity"]], on="patient_id", how="left")
        df["severity"] = df["severity"].fillna("unknown")

        for sev in sev_order:
            mask = df["severity"] == sev
            if not mask.any():
                continue
            sub = df[mask]
            jitter = rng.uniform(-jitter_scale, jitter_scale, len(sub))
            x = np.array([x_pos[int(v)] for v in sub["y_true"]]) + jitter
            ax.scatter(
                x,
                sub["prob_fused_c1"].values,
                c=SEV_COLORS[sev],
                s=sev_size[sev],
                alpha=sev_alpha[sev],
                linewidths=0,
                zorder=3,
            )

        # IQR box + median line per class
        iqr_w = 0.06
        xticklabels = []
        for cls, xc in x_pos.items():
            vals = df.loc[df["y_true"] == cls, "prob_fused_c1"]
            med = np.median(vals)
            q25 = np.percentile(vals, 25)
            q75 = np.percentile(vals, 75)
            # Subtle translucent IQR box
            ax.add_patch(
                plt.Rectangle(
                    (xc - iqr_w, q25),
                    2 * iqr_w,
                    q75 - q25,
                    facecolor="#555",
                    alpha=0.18,
                    linewidth=0,
                    zorder=4,
                )
            )
            # Median line
            ax.plot(
                [xc - jitter_scale - 0.04, xc + jitter_scale + 0.04],
                [med, med],
                color="#222",
                lw=2.0,
                zorder=5,
                label="Median" if cls == 0 else None,
            )
            # TP / TN rate below x-label
            if cls == 1:
                rate = (vals > 0.5).mean() * 100
                xticklabels.append(f"Glaucoma\nTP {rate:.0f}%")
            else:
                rate = (vals <= 0.5).mean() * 100
                xticklabels.append(f"Normal\nTN {rate:.0f}%")

        ax.axhline(0.5, color="#888", lw=1.2, ls="--", alpha=0.7, zorder=2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(xticklabels, fontsize=10)
        ax.set_title(cfg["label"], fontsize=11, fontweight="bold")
        ax.set_ylim(-0.04, 1.04)
        ax.set_xlim(-0.55, 1.55)
        ax.grid(axis="y", alpha=0.3, zorder=1)

        try:
            fold_aucs = [
                roc_auc_score(g["y_true"], g["prob_fused_c1"])
                for _, g in df.groupby(["rep", "fold"])
                if g["y_true"].nunique() > 1
            ]
            mean_auc = np.mean(fold_aucs)
            std_auc = np.std(fold_aucs)
            ax.text(
                0.65,
                0.0,
                f"AUC = {mean_auc:.3f} ± {std_auc:.3f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                color="#333",
                bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=2),
            )
        except Exception:
            pass

    axes[0].set_ylabel("Predicted P(Glaucoma)", fontsize=11)

    # Shared legend
    legend_patches = [
        mpatches.Patch(color=SEV_COLORS[s], label=SEV_LABELS[s])
        for s in ["normal", "early", "moderate", "severe", "unknown"]
    ]
    fig.legend(
        handles=legend_patches,
        fontsize=9,
        loc="lower center",
        ncol=len(legend_patches),
        framealpha=0.75,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

# Default runs shown in the comparison
DEFAULT_RUNS = [
    {
        "run": "phase4/single",
        "tower_path": "binary/single",
        "label": "Single HyperTower",
    },
    {
        "run": "phase5/ensemble_fused",
        "tower_path": "binary/ensemble",
        "label": "Bilateral Ensemble",
    },
    {
        "run": "phase5/logit_mlp_head",
        "tower_path": "binary/ensemble",
        "label": "Fused Head",
    },
]


def _aggregate_eye_to_patient(df: pd.DataFrame) -> pd.DataFrame:
    """
    Single-mode predictions are eye-level (2 rows per patient per fold).
    In the test loader, OD rows come first (sorted patient ID order) then OS.
    Average the two eyes to get one patient-level row per fold.
    """
    rows = []
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    pred_cols = [c for c in df.columns if c.startswith("pred_")]

    for (rep, fold), grp in df.groupby(["rep", "fold"]):
        n = len(grp)
        half = n // 2
        od = grp.iloc[:half].reset_index(drop=True)
        os_ = grp.iloc[half:].reset_index(drop=True)
        pat = od.copy()
        for col in prob_cols:
            pat[col] = (od[col].values + os_[col].values) / 2
        for col in pred_cols:
            pat[col] = (pat[col.replace("pred_", "prob_") + "_c1"] >= 0.5).astype(int)
        rows.append(pat)

    return pd.concat(rows, ignore_index=True)


def _load_run(run: str, tower_path: str, clinical_dir: Path) -> pd.DataFrame:
    run_dir = RESULTS_ROOT / run
    print(f"  Loading {run} ...")
    df = load_all_predictions(run_dir, tower_path=tower_path)
    if tower_path.endswith("/single"):
        from v3.scripts.output_analysis.explainability.fold_patient_ids import (
            attach_patient_ids_single,
        )

        df = attach_patient_ids_single(df, clinical_dir=clinical_dir, batch_size=8)
    else:
        from v3.scripts.output_analysis.explainability.fold_patient_ids import (
            attach_patient_ids,
        )

        df = attach_patient_ids(df, clinical_dir=clinical_dir)
    return df


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--run", default="phase5/logit_mlp_head", help="Single run for standalone plots"
    )
    ap.add_argument("--tower-path", default="binary/ensemble")
    ap.add_argument("--clinical-dir", type=Path, default=CLINICAL_DIR)
    ap.add_argument("--out", type=Path, default=FIGURES_ROOT / "explainability")
    args = ap.parse_args()

    print("Loading VFI data ...")
    vfi = load_vfi(args.clinical_dir)

    # ── Single-run plots (strips + head comparison) ──────────────────────────
    df = _load_run(args.run, args.tower_path, args.clinical_dir)
    make_confidence_strips(df, vfi, args.out / "confidence_strips.png")
    make_head_comparison(df, args.out / "head_comparison.png")

    # ── Multi-model comparison ───────────────────────────────────────────────
    print("Building comparison strips ...")
    run_configs = []
    for cfg in DEFAULT_RUNS:
        try:
            df_r = _load_run(cfg["run"], cfg["tower_path"], args.clinical_dir)
            run_configs.append({"label": cfg["label"], "df": df_r})
        except FileNotFoundError as e:
            print(f"  Skipping {cfg['run']}: {e}")
    if run_configs:
        make_comparison_strips(
            run_configs, vfi, args.out / "confidence_strips_comparison.png"
        )


if __name__ == "__main__":
    main()
