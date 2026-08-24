"""F3 (main text) - Clinical, Image, Hadamard L1 fusion under the pruned clinical panel.

Two-row, three-column figure focused on the headline architecture (Hadamard L1
fusion) and its two single-modality references. All three columns share the
pruned clinical panel: astigmatism, dioptre_1, dioptre_2, and Phakic/
Pseudophakic dropped (see S8e for the fused-head permutation importance that
motivated the prune; the paired-rep effect of dropping them is nil at fusion
and +5 AUC pts at clinical-only). The image column has no clinical inputs so
is unaffected by the prune.

Columns (left -> right):
    Clinical only          (cd_solo_single_dropzero, cd_fuse)
    Image only             (img_solo_single_refugelike, img_fuse)
    Hadamard L1 fusion     (ensemble_single_refugelike_dropzero, nt)

Rows:
    top:    confidence strip - predicted P(Glaucoma) coloured by VF-MD tier
            (severe / moderate / early) with normals in grey. Legend AUC is
            the per-fold-rep mean +/- SD across the 50 fold-reps.
    bottom: ROC per severity tier, each tier vs all normals; pooled ROC curve
            plotted on top of a shaded band showing the per-fold-rep TPR
            mean +/- SD at each FPR grid point; legend AUC is per-fold-rep
            mean +/- SD.

Glaucoma - VF_MD not recorded is dropped from both rows (n = 0 patients at
the patient-worst-eye level).

Re-run:
    python -m v4.figures.F3_hadamard_focus
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from v4.figures.util.loaders import REPO_ROOT, RESULTS_ROOT

OUT = Path(__file__).parent / "output" / "F3_hadamard_focus.png"

# ── Palette (matches other manuscript figures) ────────────────────────────────
C_NORMAL   = "#78909C"
C_EARLY    = "#29B6F6"
C_MODERATE = "#FFB300"
C_SEVERE   = "#E53935"

SEV_LABELS = {
    "normal":   "Normal",
    "early":    "Glaucoma - early (VF_MD > -6)",
    "moderate": "Glaucoma - moderate (-12 to -6)",
    "severe":   "Glaucoma - severe (VF_MD <= -12)",
}
SEV_COLORS = {
    "normal":   C_NORMAL,
    "early":    C_EARLY,
    "moderate": C_MODERATE,
    "severe":   C_SEVERE,
}
STRIP_ORDER = ["severe", "moderate", "early", "normal"]
ROC_ORDER   = ["severe", "moderate", "early"]
SEV_ALPHA   = 0.55
SEV_SIZE    = 8

# Class-density violin fill
C_NORMAL_VIOLIN   = "#4c72b0"
C_GLAUCOMA_VIOLIN = "#c44e52"

# ── Sources ───────────────────────────────────────────────────────────────────
SOURCES = [
    ("Clinical only",
     RESULTS_ROOT / "explainability" / "cd_solo_single_dropzero",
     "cd_fuse"),
    ("Image only",
     RESULTS_ROOT / "refuge_v2m_baseline" / "img_solo_single_refugelike",
     "img_fuse"),
    ("Hadamard L1 fusion",
     RESULTS_ROOT / "explainability" / "ensemble_single_refugelike_dropzero",
     "nt"),
]

CLINICAL_DIR = REPO_ROOT / "Papila" / "ClinicalData"

# Common FPR grid for per-fold-rep ROC interpolation.
FPR_GRID = np.linspace(0.0, 1.0, 201)


def _per_foldrep_auc_and_band(
    neg_df: pd.DataFrame, pos_df: pd.DataFrame,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Return (auc_mean, auc_sd, tpr_mean, tpr_sd) computed across fold-reps.

    Each (rep, fold) is one observation: build an ROC on its own negatives +
    positives, interpolate TPR to FPR_GRID, record its AUC. Report the mean
    and standard deviation across the 50 fold-reps. Fold-reps whose subset
    has only one class present (rare, e.g. a tier missing from a fold) are
    dropped from that tier's calculation.
    """
    nan_curve = np.full_like(FPR_GRID, np.nan, dtype=float)
    if len(pos_df) == 0 or len(neg_df) == 0:
        return float("nan"), float("nan"), nan_curve, nan_curve

    keys = sorted(set(zip(neg_df["rep"], neg_df["fold"])) |
                  set(zip(pos_df["rep"], pos_df["fold"])))
    aucs: list[float] = []
    tprs: list[np.ndarray] = []
    for rep, fold in keys:
        n = neg_df[(neg_df["rep"] == rep) & (neg_df["fold"] == fold)]
        p = pos_df[(pos_df["rep"] == rep) & (pos_df["fold"] == fold)]
        if len(n) == 0 or len(p) == 0:
            continue
        y = np.concatenate([np.zeros(len(n), dtype=int),
                            np.ones(len(p), dtype=int)])
        s = np.concatenate([n["prob_glaucoma"].values,
                            p["prob_glaucoma"].values])
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        # roc_curve returns duplicate fpr=0 rows: (0, 0) then (0, y_first_step).
        # np.interp resolves the tie to the last value, which lifts the curve
        # off the origin. Force the origin to (0, 0) so the mean ROC actually
        # starts where it should.
        tpr_i = np.interp(FPR_GRID, fpr, tpr)
        tpr_i[FPR_GRID <= 0.0] = 0.0
        tprs.append(tpr_i)
        aucs.append(roc_auc_score(y, s))

    if not aucs:
        return float("nan"), float("nan"), nan_curve, nan_curve
    aucs_a = np.array(aucs)
    tprs_a = np.stack(tprs, axis=0)
    return (float(aucs_a.mean()),
            float(aucs_a.std(ddof=1)) if len(aucs_a) > 1 else 0.0,
            tprs_a.mean(axis=0),
            tprs_a.std(axis=0, ddof=1) if len(tprs_a) > 1
            else np.zeros_like(FPR_GRID))


# ── VFI loader (patient-level worst-eye severity) ─────────────────────────────

def load_vfi() -> pd.DataFrame:
    od = pd.read_excel(CLINICAL_DIR / "patient_data_od.xlsx", header=1)
    os_= pd.read_excel(CLINICAL_DIR / "patient_data_os.xlsx", header=1)

    def _clean(df):
        df = df.copy()
        if "Patient ID" not in df.columns and "ID" in df.columns:
            df.rename(columns={"ID": "Patient ID"}, inplace=True)
        df["Patient ID"] = df["Patient ID"].astype(str).str.extract(r"(\d+)")[0].astype(int)
        df["Diagnosis"]  = pd.to_numeric(df["Diagnosis"], errors="coerce")
        df["VF_MD"]      = pd.to_numeric(df["VF_MD"], errors="coerce")
        df = df[df["Diagnosis"].isin([0, 1])].copy()
        return df[["Patient ID", "Diagnosis", "VF_MD"]]

    both = pd.concat([_clean(od), _clean(os_)], ignore_index=True)
    diag = both.groupby("Patient ID")["Diagnosis"].agg(lambda x: x.mode().iloc[0]).reset_index()
    vf   = both.groupby("Patient ID")["VF_MD"].min().reset_index()
    out  = diag.merge(vf, on="Patient ID").rename(
        columns={"Patient ID": "patient_id", "Diagnosis": "diagnosis", "VF_MD": "vf_md"}
    )

    def _sev(row):
        if int(row["diagnosis"]) == 0: return "normal"
        v = row["vf_md"]
        if pd.isna(v): return "unknown"
        if v > -6:    return "early"
        if v > -12:   return "moderate"
        return "severe"
    out["severity"] = out.apply(_sev, axis=1)
    return out


# ── Prediction pooler ─────────────────────────────────────────────────────────

def collect_predictions(run_dir: Path, eval_stage: str) -> pd.DataFrame:
    if not run_dir.exists():
        return pd.DataFrame()
    rows: list[dict] = []
    for rep in sorted(run_dir.glob("rep*")):
        fp = next(iter(rep.rglob("predictions.h5")), None)
        if fp is None: continue
        with h5py.File(fp, "r") as f:
            if eval_stage not in f: continue
            grp = f[eval_stage]
            logits = grp["logits"][:]
            y_true = grp["y_true"][:].astype(int)
            split  = grp["split"][:]
            eid0   = grp["entity_id_0"][:]
            n_folds, n_epochs, _, n_heads, n_outputs = logits.shape
            if n_outputs != 2: continue
            ep, head = n_epochs - 1, n_heads - 1
            for fold in range(n_folds):
                labels = np.array([s.decode() if isinstance(s, bytes) else str(s) for s in split[fold]])
                test_mask = labels == "test"
                if not test_mask.any(): continue
                # The 'logits' field in predictions.h5 is misnamed: the values
                # already sum to 1 across the output axis (i.e. they are softmax
                # probabilities). Use column 1 directly as P(Glaucoma); applying
                # another softmax here would compress everything into
                # [sigmoid(-1), sigmoid(1)] = [0.269, 0.731] (rank preserved so
                # AUC is unchanged, but probabilities and thresholds become
                # meaningless).
                p = logits[fold, ep, test_mask, head, :]
                for k, idx in enumerate(np.where(test_mask)[0]):
                    rows.append({
                        "rep":           rep.name,
                        "fold":          fold,
                        "patient_id":    int(eid0[idx]),
                        "y_true":        int(y_true[idx]),
                        "prob_glaucoma": float(p[k, 1]),
                    })
    return pd.DataFrame(rows)


# ── Confidence strip ──────────────────────────────────────────────────────────

def _merge_and_filter(df: pd.DataFrame, vfi: pd.DataFrame) -> pd.DataFrame:
    """Attach patient-level severity and apply the consistent-label filter.

    Two exclusions:
      1. Patients whose patient-level severity is 'unknown' (glaucoma without
         a recorded VF_MD; 0 patients in PAPILA at the patient level).
      2. Mixed-label predictions where the model's eye-level label is
         glaucoma (y_true == 1) but the patient's severity resolves to
         'normal' via the load_vfi mode-of-both-eyes rule. ~70 predictions
         in this dataset; excluding them makes the strip-corner AUC equal to
         the sample-size-weighted mean of the per-tier ROC AUCs.
    """
    df = df.merge(vfi[["patient_id", "severity"]], on="patient_id", how="left")
    df["severity"] = df["severity"].fillna("unknown")
    df = df[df["severity"] != "unknown"]
    df = df[~((df["y_true"] == 1) & (df["severity"] == "normal"))]
    return df.copy()


def _draw_strip(ax, df: pd.DataFrame, vfi: pd.DataFrame, label: str):
    df = _merge_and_filter(df, vfi)

    rng = np.random.default_rng(42)
    x_pos = {0: 0.0, 1: 1.0}
    jitter_scale = 0.18

    data_by_class = [df.loc[df["y_true"] == cls, "prob_glaucoma"].values for cls in [0, 1]]
    if all(len(d) > 0 for d in data_by_class):
        vp = ax.violinplot(data_by_class, positions=[0, 1], widths=0.7,
                           showmedians=False, showextrema=False)
        for body, color in zip(vp["bodies"], [C_NORMAL_VIOLIN, C_GLAUCOMA_VIOLIN]):
            body.set_facecolor(color); body.set_alpha(0.30)
            body.set_edgecolor("none"); body.set_zorder(2)

    for sev in STRIP_ORDER:
        mask = df["severity"] == sev
        if not mask.any(): continue
        sub = df[mask]
        jitter = rng.uniform(-jitter_scale, jitter_scale, len(sub))
        x = np.array([x_pos[int(v)] for v in sub["y_true"]]) + jitter
        ax.scatter(x, sub["prob_glaucoma"].values,
                   c=SEV_COLORS[sev], s=SEV_SIZE,
                   alpha=SEV_ALPHA, linewidths=0, zorder=3)

    xtick_labels = []
    for cls, xc in x_pos.items():
        vals = df.loc[df["y_true"] == cls, "prob_glaucoma"]
        if not len(vals):
            xtick_labels.append("Normal" if cls == 0 else "Glaucoma"); continue
        med = float(np.median(vals))
        ax.plot([xc - jitter_scale - 0.04, xc + jitter_scale + 0.04],
                [med, med], color="#222", lw=2.0, zorder=5)
        if cls == 0:
            rate = (vals <= 0.5).mean() * 100
            xtick_labels.append(f"Normal\nTN {rate:.0f}%")
        else:
            rate = (vals > 0.5).mean() * 100
            xtick_labels.append(f"Glaucoma\nTP {rate:.0f}%")

    ax.axhline(0.5, color="#888", lw=1.2, ls="--", alpha=0.7, zorder=2)
    ax.set_xticks([0, 1]); ax.set_xticklabels(xtick_labels, fontsize=10)
    ax.set_ylim(-0.04, 1.04); ax.set_xlim(-0.55, 1.55)
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, zorder=1)

    # Per-fold-rep AUC mean +/- SD across the 50 fold-reps. Uses the same
    # computation as the ROC panel legend for consistency.
    neg_df = df[df["y_true"] == 0]
    pos_df = df[df["y_true"] == 1]
    if len(pos_df) > 0 and len(neg_df) > 0:
        auc_mean, auc_sd, *_ = _per_foldrep_auc_and_band(neg_df, pos_df)
        ax.text(0.98, 0.02,
                f"AUC = {auc_mean:.3f} +/- {auc_sd:.3f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, color="#333",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2))


# ── ROC by severity ───────────────────────────────────────────────────────────

def _draw_roc(ax, df: pd.DataFrame, vfi: pd.DataFrame, label: str):
    df = _merge_and_filter(df, vfi)

    # Diagonal reference behind everything
    ax.plot([0, 1], [0, 1], color="#aaa", linewidth=0.7, linestyle=":", zorder=1)

    for tier in ROC_ORDER:
        neg = df[df["y_true"] == 0]
        pos = df[(df["y_true"] == 1) & (df["severity"] == tier)]
        if len(pos) == 0 or len(neg) == 0:
            continue

        # Per-fold-rep mean ROC (plotted on FPR_GRID) with a +/- SD band and
        # legend AUC as per-fold-rep mean +/- SD.
        auc_mean, auc_sd, tpr_mean, tpr_sd = _per_foldrep_auc_and_band(neg, pos)
        tpr_lo = np.clip(tpr_mean - tpr_sd, 0.0, 1.0)
        tpr_hi = np.clip(tpr_mean + tpr_sd, 0.0, 1.0)

        color = SEV_COLORS[tier]
        ax.fill_between(FPR_GRID, tpr_lo, tpr_hi,
                        color=color, alpha=0.16, linewidth=0, zorder=2)
        label_txt = f"{tier.capitalize()}  AUC = {auc_mean:.3f} +/- {auc_sd:.3f}"
        ax.plot(FPR_GRID, tpr_mean, color=color, linewidth=2.0, zorder=3, label=label_txt)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate", fontsize=10)
    ax.grid(alpha=0.25, linestyle="--", zorder=1)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.94)
    ax.set_title(label, fontsize=11, fontweight="bold")


# ── Render ────────────────────────────────────────────────────────────────────

def render() -> None:
    vfi = load_vfi()
    frames = [(lbl, collect_predictions(p, s)) for lbl, p, s in SOURCES]
    for lbl, df in frames:
        if len(df):
            print(f"  {lbl:<20s}  n_rows={len(df):>5d}  (pid={df['patient_id'].nunique()}, reps={df['rep'].nunique()})")
        else:
            print(f"  {lbl:<20s}  no data")

    n_cols = len(SOURCES)
    fig = plt.figure(figsize=(4.6 * n_cols, 10.4))
    fig.patch.set_facecolor("#e8e8e8")

    gs = fig.add_gridspec(2, n_cols, hspace=0.30, wspace=0.16,
                          left=0.06, right=0.985, top=0.94, bottom=0.10)

    strip_axes, roc_axes = [], []
    strip_sharey, roc_sharey = None, None
    for i, (lbl, df) in enumerate(frames):
        ax_s = fig.add_subplot(gs[0, i], sharey=strip_sharey)
        ax_r = fig.add_subplot(gs[1, i], sharey=roc_sharey)
        strip_sharey = strip_sharey or ax_s
        roc_sharey   = roc_sharey   or ax_r
        strip_axes.append(ax_s); roc_axes.append(ax_r)
        ax_s.set_facecolor("#e8e8e8"); ax_r.set_facecolor("#e8e8e8")
        if len(df):
            _draw_strip(ax_s, df, vfi, lbl)
            _draw_roc(ax_r,   df, vfi, lbl)
        else:
            for ax in (ax_s, ax_r):
                ax.text(0.5, 0.5, "(pending)", ha="center", va="center",
                        fontsize=12, color="#888", transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(lbl, fontsize=11, fontweight="bold")

    strip_axes[0].set_ylabel("Predicted P(Glaucoma)", fontsize=11)
    roc_axes[0].set_ylabel("True positive rate", fontsize=11)
    # Force numeric ticks on the leftmost panel of each row (matplotlib
    # sometimes drops them under sharey after set_ylim); hide on the rest.
    strip_axes[0].set_yticks(np.linspace(0, 1, 6))
    roc_axes[0].set_yticks(np.linspace(0, 1, 6))
    strip_axes[0].tick_params(axis="y", labelleft=True)
    roc_axes[0].tick_params(axis="y", labelleft=True)
    for ax in strip_axes[1:] + roc_axes[1:]:
        ax.tick_params(axis="y", labelleft=False)

    # Shared severity legend (strip row) at figure bottom
    legend_patches = [mpatches.Patch(color=SEV_COLORS[s], label=SEV_LABELS[s])
                      for s in ["normal", "early", "moderate", "severe"]]
    fig.legend(handles=legend_patches, fontsize=9,
               loc="lower center", ncol=len(legend_patches),
               framealpha=0.75, bbox_to_anchor=(0.5, 0.0))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}")


if __name__ == "__main__":
    render()
