"""F3 — Single-mode comparison via confidence strips.

Four panels showing per-eye predicted P(Glaucoma) coloured by VF-MD severity
(when known), in the style of v3/figures/explainability/confidence_strips_comparison.png.

Panels (left → right):
    Clinical only          (cd_solo_single)
    Image only             (img_solo_single_refugelike)
    Hadamard fusion        (ensemble_single_refugelike — baseline)
    Concat fusion          (phase3_v4/single_bcd_concat)

Each panel shows test predictions pooled across all reps × folds. Points are
jittered around their true-class column and coloured by VF-MD severity tier
(early / moderate / severe / unknown for glaucoma rows; grey for healthy).

Re-run anytime:
    python -m v4.figures.F3_hyperfeature_ablation
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
from sklearn.metrics import roc_auc_score

from v4.figures.util.loaders import REPO_ROOT, RESULTS_ROOT

OUT = Path(__file__).parent / "output" / "F3_single_mode_strips.png"

# ── Palette (matches v3 confidence_strips) ───────────────────────────────────
C_NORMAL    = "#78909C"
C_EARLY     = "#29B6F6"
C_MODERATE  = "#FFB300"
C_SEVERE    = "#E53935"
C_UNKNOWN   = "#BDBDBD"

SEV_LABELS = {
    "normal":   "Normal",
    "early":    "Glaucoma — early (VF_MD > −6)",
    "moderate": "Glaucoma — moderate (−12 to −6)",
    "severe":   "Glaucoma — severe (VF_MD < −12)",
    "unknown":  "Glaucoma — VF_MD not recorded",
}
SEV_COLORS = {
    "normal":   C_NORMAL,
    "early":    C_EARLY,
    "moderate": C_MODERATE,
    "severe":   C_SEVERE,
    "unknown":  C_UNKNOWN,
}

SEV_ORDER = ["severe", "moderate", "unknown", "early", "normal"]
SEV_ALPHA = {"normal": 0.55, "unknown": 0.55, "early": 0.55, "moderate": 0.55, "severe": 0.55}
SEV_SIZE  = {"normal": 8,    "unknown": 8,    "early": 8,    "moderate": 8,    "severe": 8}

# ── Per-panel definitions: (label, results dir, eval_stage) ──────────────────
# Top row: single-modality reference runs
TOP_ROW = [
    ("Clinical only",   RESULTS_ROOT / "phase2_v4" / "cd_solo_single",                       "cd_fuse"),
    ("Image only",      RESULTS_ROOT / "refuge_v2m_baseline" / "img_solo_single_refugelike", "img_fuse"),
]

# Bottom row: L1 fusion bridge variants (eye-level img+cd ensembles)
BOTTOM_ROW = [
    ("Concat fusion",   RESULTS_ROOT / "phase3_v4" / "single_bcd_concat",                    "nt"),
    ("Pairwise fusion", RESULTS_ROOT / "phase3_v4" / "single_bcd_pairwise",                  "nt"),
    ("Gated fusion",    RESULTS_ROOT / "phase3_v4" / "single_bcd_gated",                     "nt"),
    ("Hadamard fusion", RESULTS_ROOT / "refuge_v2m_baseline" / "ensemble_single_refugelike", "nt"),
]

CLINICAL_DIR = REPO_ROOT / "Papila" / "ClinicalData"


# ── VFI loader (patient-level worst-eye severity, matches v3) ────────────────

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
        # PAPILA: 0=Normal, 1=Glaucoma, 2=Suspect — keep only binary subjects
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


# ── Prediction pooler ────────────────────────────────────────────────────────

def collect_predictions(run_dir: Path, eval_stage: str) -> pd.DataFrame:
    """Pool test rows across reps × folds. Returns DataFrame with
    patient_id, y_true, prob_glaucoma, rep, fold.
    Applies softmax to the 2-class logits."""
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
            n_folds, n_epochs, n_samples, n_heads, n_outputs = logits.shape
            if n_outputs != 2: continue
            ep, head = n_epochs - 1, n_heads - 1
            for fold in range(n_folds):
                labels = np.array([s.decode() if isinstance(s, bytes) else str(s) for s in split[fold]])
                test_mask = (labels == "test")
                if not test_mask.any(): continue
                lg = logits[fold, ep, test_mask, head, :]   # (n_test, 2)
                # softmax
                e = np.exp(lg - lg.max(axis=1, keepdims=True))
                p = e / e.sum(axis=1, keepdims=True)
                for k, idx in enumerate(np.where(test_mask)[0]):
                    rows.append({
                        "rep":             rep.name,
                        "fold":            fold,
                        "patient_id":      int(eid0[idx]),
                        "y_true":          int(y_true[idx]),
                        "prob_glaucoma":   float(p[k, 1]),
                    })
    return pd.DataFrame(rows)


# ── Panel render ─────────────────────────────────────────────────────────────

C_NORMAL_VIOLIN   = "#4c72b0"
C_GLAUCOMA_VIOLIN = "#c44e52"


def _draw_panel(ax, df: pd.DataFrame, vfi: pd.DataFrame, label: str):
    df = df.merge(vfi[["patient_id", "severity"]], on="patient_id", how="left")
    df["severity"] = df["severity"].fillna("unknown")

    rng = np.random.default_rng(42)
    x_pos = {0: 0.0, 1: 1.0}
    jitter_scale = 0.18

    # Violin density behind everything (per true class)
    data_by_class = [df.loc[df["y_true"] == cls, "prob_glaucoma"].values for cls in [0, 1]]
    if all(len(d) > 0 for d in data_by_class):
        vp = ax.violinplot(data_by_class, positions=[0, 1], widths=0.7,
                           showmedians=False, showextrema=False)
        for body, color in zip(vp["bodies"], [C_NORMAL_VIOLIN, C_GLAUCOMA_VIOLIN]):
            body.set_facecolor(color); body.set_alpha(0.30)
            body.set_edgecolor("none"); body.set_zorder(2)

    for sev in SEV_ORDER:
        mask = df["severity"] == sev
        if not mask.any(): continue
        sub = df[mask]
        jitter = rng.uniform(-jitter_scale, jitter_scale, len(sub))
        x = np.array([x_pos[int(v)] for v in sub["y_true"]]) + jitter
        ax.scatter(x, sub["prob_glaucoma"].values,
                   c=SEV_COLORS[sev], s=SEV_SIZE[sev],
                   alpha=SEV_ALPHA[sev], linewidths=0, zorder=3)

    # Median lines + TN/TP rate labels per class
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

    # AUC across reps×folds (per-fold AUC averaged)
    fold_aucs = []
    for _, g in df.groupby(["rep", "fold"]):
        if g["y_true"].nunique() < 2: continue
        try: fold_aucs.append(roc_auc_score(g["y_true"], g["prob_glaucoma"]))
        except Exception: pass
    if fold_aucs:
        ax.text(0.66, 0.0,
                f"AUC = {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, color="#333",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))


def render() -> None:
    vfi = load_vfi()
    top_dfs    = [(lbl, collect_predictions(p, s)) for lbl, p, s in TOP_ROW]
    bottom_dfs = [(lbl, collect_predictions(p, s)) for lbl, p, s in BOTTOM_ROW]

    for lbl, df in top_dfs + bottom_dfs:
        if len(df):
            print(f"  {lbl:<18s}  n_rows={len(df):>5d}  (pid={df['patient_id'].nunique()}, reps={df['rep'].nunique()})")
        else:
            print(f"  {lbl:<18s}  no data")

    n_cols = len(BOTTOM_ROW)
    fig = plt.figure(figsize=(4.4 * n_cols, 12))
    fig.patch.set_facecolor("#e8e8e8")
    fig.suptitle("L1 Fusion Comparison", fontsize=14, fontweight="bold")

    gs = fig.add_gridspec(2, n_cols, hspace=0.30, wspace=0.15)

    # Top row: 2 reference panels at same width as bottom panels, centered.
    # In a 4-column bottom grid, that's columns 1 and 2.
    n_top = len(top_dfs)
    top_offset = (n_cols - n_top) // 2  # leading empty columns
    top_axes = []
    for i, (lbl, df) in enumerate(top_dfs):
        ax = fig.add_subplot(gs[0, top_offset + i])
        top_axes.append(ax)
        ax.set_facecolor("#e8e8e8")
        if len(df):
            _draw_panel(ax, df, vfi, lbl)
        else:
            ax.text(0.5, 0.5, "(pending)", ha="center", va="center",
                    fontsize=12, color="#888", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(lbl, fontsize=11, fontweight="bold")

    # Bottom row: 4 fusion variants
    bottom_axes = []
    sharey = None
    for i, (lbl, df) in enumerate(bottom_dfs):
        ax = fig.add_subplot(gs[1, i], sharey=sharey)
        sharey = sharey or ax
        bottom_axes.append(ax)
        ax.set_facecolor("#e8e8e8")
        if len(df):
            _draw_panel(ax, df, vfi, lbl)
        else:
            ax.text(0.5, 0.5, "(pending)", ha="center", va="center",
                    fontsize=12, color="#888", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(lbl, fontsize=11, fontweight="bold")

    top_axes[0].set_ylabel("Predicted P(Glaucoma)", fontsize=11)
    bottom_axes[0].set_ylabel("Predicted P(Glaucoma)", fontsize=11)
    for ax in bottom_axes[1:]:
        ax.set_yticklabels([])

    legend_patches = [mpatches.Patch(color=SEV_COLORS[s], label=SEV_LABELS[s])
                      for s in ["normal", "early", "moderate", "severe", "unknown"]]
    fig.legend(handles=legend_patches, fontsize=9,
               loc="lower center", ncol=len(legend_patches),
               framealpha=0.75, bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}")


if __name__ == "__main__":
    render()
