"""V2M_F6 — Regression VF_MD with severity grouping (refuge V2-M backbone).

V2-M counterpart to F6. Reads predictions from the refuge V2-M variant of
baseline_reg_nt50; otherwise identical layout to F6 so panels can be
compared side-by-side.

Re-run anytime predictions.h5 changes:
    python -m v4.figures.V2M_F6_regression
"""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

from v4.figures.util.loaders import RESULTS_ROOT

OUT = Path(__file__).parent / "output" / "V2M_F6_regression.png"
RUN_DIR = RESULTS_ROOT / "v2m_variants" / "baseline_reg_nt50_v2m"

# Prediction-side bin boundaries
NP_THRESH = -1.097  # mean of measured-healthy MD
SEV_PRED = -9.14  # midpoint of HAP -12 and mean predicted MD for severe truth

# Actual (HAP / Mills) clinical boundaries
HAP_SEV = -12.0
HAP_MOD = -6.0

LABELS = ["severe", "moderate", "low"]

# Severity colors (consistent across figures)
C_SEVERE = "#E53935"
C_MODERATE = "#FFB300"
C_LOW = "#3B6FB5"


def _decode(arr):
    return np.array(
        [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in arr]
    )


def _collect():
    actuals, preds = [], []
    for fp in sorted(RUN_DIR.rglob("predictions.h5")):
        with h5py.File(fp, "r") as f:
            if "hb" not in f:
                continue
            grp = f["hb"]
            logits = grp["logits"][:]
            y_true = grp["y_true"][:].astype(float)
            split = grp["split"][:]
            n_folds, n_epochs, _, n_heads, _ = logits.shape
            ep, head, out = n_epochs - 1, n_heads - 1, 0
            for fold in range(n_folds):
                labels = _decode(split[fold])
                m = (
                    (labels == "test")
                    & np.isfinite(y_true)
                    & np.isfinite(logits[fold, ep, :, head, out])
                )
                actuals.append(y_true[m])
                preds.append(logits[fold, ep, m, head, out].astype(float))
    if not actuals:
        return None, None
    return np.concatenate(actuals), np.concatenate(preds)


def _bin(values, sev, np_th):
    bins = np.full(values.shape, 2, dtype=int)
    bins[values <= np_th] = 1
    bins[values <= sev] = 0
    return bins


def render() -> None:
    if not RUN_DIR.exists() or not any(RUN_DIR.rglob("predictions.h5")):
        print(f"[F6] no predictions.h5 yet under {RUN_DIR}. Run after data lands.")
        return
    a, p = _collect()
    if a is None:
        print("[F6] no usable predictions")
        return
    print(f"[F6] pooled n={a.size}")

    fig = plt.figure(figsize=(14, 10.5))
    gs = fig.add_gridspec(
        2, 2, hspace=0.40, wspace=0.30, left=0.07, right=0.96, top=0.92, bottom=0.07
    )
    ax_sc = fig.add_subplot(gs[0, 0])
    ax_rc = fig.add_subplot(gs[0, 1])
    ax_cm = fig.add_subplot(gs[1, 0])
    ax_tb = fig.add_subplot(gs[1, 1])
    ax_tb.axis("off")

    # ── (a) scatter ──────────────────────────────────────────────────────────
    ax_sc.scatter(a, p, s=10, alpha=0.4, color="#2563eb", edgecolor="none")
    lo, hi = -30, 6
    ax_sc.plot([lo, hi], [lo, hi], ls="--", color="#9ca3af", lw=1, label="ideal y=x")
    ax_sc.axvline(HAP_SEV, ls=":", color="#dc2626", lw=0.8, alpha=0.5)
    ax_sc.axvline(HAP_MOD, ls=":", color="#dc2626", lw=0.8, alpha=0.5)
    ax_sc.set_xlim(lo, hi)
    ax_sc.set_ylim(lo, hi)
    ax_sc.set_xlabel("Actual VF_MD (dB)")
    ax_sc.set_ylabel("Predicted VF_MD (dB)")
    ax_sc.set_title(f"(a) Predicted vs Actual MD  (n={a.size})", fontsize=11)
    r = np.corrcoef(a, p)[0, 1]
    mae = float(np.mean(np.abs(p - a)))
    ax_sc.text(
        0.04,
        0.95,
        f"r = {r:.3f}\nMAE = {mae:.2f} dB",
        transform=ax_sc.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="#d1d5db"),
    )

    # ── (b) three one-vs-rest ROCs ───────────────────────────────────────────
    # Severe vs rest:    score = -p (more negative pred → more severe)
    # Low vs rest:       score = +p (more positive pred → more "low" / no-problem)
    # Moderate vs rest:  score = -|p - midpoint of moderate range|
    #                    (closer to midpoint → more moderate-like)
    mod_midpoint = 0.5 * (HAP_SEV + HAP_MOD)  # -9 dB

    truth_severe = (a <= HAP_SEV).astype(int)
    truth_low = (a > HAP_MOD).astype(int)
    truth_moderate = ((a > HAP_SEV) & (a <= HAP_MOD)).astype(int)

    series = [
        ("Severe (≤ −12 dB) vs rest", truth_severe, -p, C_SEVERE),
        (
            "Moderate (−12..−6) vs rest",
            truth_moderate,
            -np.abs(p - mod_midpoint),
            C_MODERATE,
        ),
        ("Low (> −6 dB) vs rest", truth_low, p, C_LOW),
    ]
    for label, ybin, score, color in series:
        if len(np.unique(ybin)) < 2:
            continue
        fpr, tpr, _ = roc_curve(ybin, score)
        auc = roc_auc_score(ybin, score)
        ax_rc.plot(fpr, tpr, color=color, lw=1.8, label=f"{label}  (AUC = {auc:.3f})")
    ax_rc.plot([0, 1], [0, 1], ls="--", color="#9ca3af", lw=0.8)
    ax_rc.set_xlim(0, 1)
    ax_rc.set_ylim(0, 1.02)
    ax_rc.set_xlabel("False positive rate")
    ax_rc.set_ylabel("True positive rate")
    ax_rc.set_title("(b) One-vs-rest ROC per severity tier", fontsize=11)
    ax_rc.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax_rc.grid(alpha=0.25, linestyle="--")

    # ── (c) confusion matrix ─────────────────────────────────────────────────
    t_act = _bin(a, HAP_SEV, HAP_MOD)
    t_pred = _bin(p, SEV_PRED, NP_THRESH)
    cm = np.zeros((3, 3), dtype=int)
    for x, y in zip(t_act, t_pred):
        cm[x, y] += 1
    cm_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax_cm.imshow(cm_pct, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    for i in range(3):
        for j in range(3):
            text_color = "white" if cm_pct[i, j] > 0.55 else "black"
            ax_cm.text(
                j,
                i,
                f"{cm[i,j]}\n({cm_pct[i,j]*100:.0f}%)",
                ha="center",
                va="center",
                fontsize=10,
                color=text_color,
            )
    ax_cm.set_xticks(range(3))
    ax_cm.set_xticklabels(LABELS, fontsize=10)
    ax_cm.set_yticks(range(3))
    ax_cm.set_yticklabels(LABELS, fontsize=10)
    ax_cm.set_xlabel("Predicted", fontsize=10)
    ax_cm.set_ylabel("Actual", fontsize=10)
    ax_cm.set_title("(c) 3-tier confusion", fontsize=11)

    # ── (d) per-class stats — sens / spec / PPV / NPV only ─────────────────
    # We deliberately drop TP/FN/FP/TN here because in a 3-tier setting a
    # "false negative" for severe could land in moderate (clinically
    # different from landing in low). The confusion matrix in panel (c)
    # already shows that distinction; sens/spec/PPV/NPV summarise the
    # one-vs-rest performance without the blanket-count obfuscation.
    ax_tb.set_title("(d) Per-class statistics", fontsize=11)
    ax_tb.set_xlim(0, 10)
    ax_tb.set_ylim(0, 5)

    headers = ["class", "n", "sens", "spec", "PPV", "NPV"]
    # Make the class column wider than the numeric columns to avoid clipping.
    col_widths = np.array([2.4, 1.1, 1.4, 1.4, 1.4, 1.4])
    col_widths *= 10.0 / col_widths.sum()  # normalise to total width 10
    col_edges = np.concatenate([[0], np.cumsum(col_widths)])
    col_x = (col_edges[:-1] + col_edges[1:]) / 2  # column centers
    row_y = [3.5, 2.5, 1.5, 0.5]  # 1 header + 3 data rows

    rows = []
    for c in range(3):
        ac = t_act == c
        pc = t_pred == c
        tp = int(np.sum(ac & pc))
        fn = int(np.sum(ac & ~pc))
        fp = int(np.sum(~ac & pc))
        tn = int(np.sum(~ac & ~pc))
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        ppv = tp / max(tp + fp, 1)
        npv = tn / max(tn + fn, 1)
        rows.append(
            [
                LABELS[c],
                int(ac.sum()),
                f"{sens:.3f}",
                f"{spec:.3f}",
                f"{ppv:.3f}",
                f"{npv:.3f}",
            ]
        )

    # Header band
    ax_tb.add_patch(
        plt.Rectangle(
            (0, 3.05), 10, 0.9, facecolor="#dbeafe", edgecolor="none", zorder=1
        )
    )
    for x, h in zip(col_x, headers):
        ax_tb.text(
            x,
            row_y[0],
            h,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="#1e3a8a",
            zorder=2,
        )

    # Data rows with zebra shading
    row_colors = ["#f8fafc", "#eef2f6", "#f8fafc"]
    severity_color = {"severe": C_SEVERE, "moderate": C_MODERATE, "low": C_LOW}
    for ri, row in enumerate(rows):
        ax_tb.add_patch(
            plt.Rectangle(
                (0, row_y[ri + 1] - 0.45),
                10,
                0.9,
                facecolor=row_colors[ri],
                edgecolor="none",
                zorder=1,
            )
        )
        for ci, val in enumerate(row):
            txt_color = "#222"
            weight = "normal"
            if ci == 0:
                txt_color = severity_color.get(val, "#222")
                weight = "bold"
            ax_tb.text(
                col_x[ci],
                row_y[ri + 1],
                str(val),
                ha="center",
                va="center",
                fontsize=11,
                fontweight=weight,
                color=txt_color,
                zorder=2,
            )

    # Subtle horizontal grid lines
    for y in [
        row_y[0] - 0.45,
        row_y[0] + 0.45,
        row_y[1] - 0.45,
        row_y[2] - 0.45,
        row_y[3] - 0.45,
    ]:
        ax_tb.plot([0, 10], [y, y], color="#cbd5e1", lw=0.6, zorder=1.5)

    fig.suptitle(
        "Regression predicting VF_MD with severity grouping — refuge V2-M backbone",
        fontsize=13,
        fontweight="bold",
        y=0.97,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}")


if __name__ == "__main__":
    render()
