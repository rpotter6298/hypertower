#!/usr/bin/env python3
"""
Compare Perkins→Pneumatic IOP conversion methods:
  - Current: fixed ratio (1.158)
  - OLS linear regression (minimises MSE)
  - LAD linear regression (minimises MAE — robust to outliers)
  - Multiple regression: Perkins + Pachymetry (OLS)

Also reports on the impact of dropping IOP_raw.

Run from repo root:
    python scripts/exploratory/iop_correction_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

CURRENT_RATIO = 1.158

# ── load raw clinical data ────────────────────────────────────────────────────

def load_raw() -> pd.DataFrame:
    dfs = []
    for fname in ("patient_data_od.xlsx", "patient_data_os.xlsx"):
        df = pd.read_excel(REPO / "Papila/ClinicalData" / fname, header=1)
        eye = "OD" if "od" in fname else "OS"
        df["eyeID"] = eye
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

df = load_raw()
print(f"Total rows: {len(df)}")
print(f"Columns with IOP: {[c for c in df.columns if 'iop' in c.lower() or c in ('Pneumatic','Perkins')]}")

# ── find paired rows ──────────────────────────────────────────────────────────

paired = df.dropna(subset=["Pneumatic", "Perkins", "Pachymetry"]).copy()
pneumatic  = paired["Pneumatic"].values.astype(float)
perkins    = paired["Perkins"].values.astype(float)
pachymetry = paired["Pachymetry"].values.astype(float)

print(f"\nPaired obs (Pneumatic + Perkins + Pachymetry): n={len(paired)}")
print(f"  Pneumatic: mean={pneumatic.mean():.2f}  std={pneumatic.std():.2f}  "
      f"range=[{pneumatic.min():.1f}, {pneumatic.max():.1f}]")
print(f"  Perkins:   mean={perkins.mean():.2f}  std={perkins.std():.2f}  "
      f"range=[{perkins.min():.1f}, {perkins.max():.1f}]")

# ── current ratio ─────────────────────────────────────────────────────────────

ratio_obs   = pneumatic / perkins
ratio_mean  = ratio_obs.mean()
ratio_pred  = perkins * CURRENT_RATIO
ratio_resid = pneumatic - ratio_pred
ratio_mae   = np.abs(ratio_resid).mean()
ratio_rmse  = np.sqrt((ratio_resid ** 2).mean())

print(f"\n── Current ratio approach ────────────────────────────────")
print(f"  Observed Pneumatic/Perkins ratio: mean={ratio_mean:.4f}  "
      f"std={ratio_obs.std():.4f}  range=[{ratio_obs.min():.3f}, {ratio_obs.max():.3f}]")
print(f"  Hardcoded ratio used: {CURRENT_RATIO}")
print(f"  MAE:  {ratio_mae:.3f} mmHg")
print(f"  RMSE: {ratio_rmse:.3f} mmHg")

# ── OLS regression ────────────────────────────────────────────────────────────

slope, intercept, r, p, se = stats.linregress(perkins, pneumatic)
reg_pred  = slope * perkins + intercept
reg_resid = pneumatic - reg_pred
reg_mae   = np.abs(reg_resid).mean()
reg_rmse  = np.sqrt((reg_resid ** 2).mean())

print(f"\n── OLS regression (minimises MSE): Pneumatic = slope * Perkins + intercept ──")
print(f"  slope={slope:.4f}  intercept={intercept:.4f}")
print(f"  R²={r**2:.4f}  p={p:.4e}")
print(f"  MAE:  {reg_mae:.3f} mmHg")
print(f"  RMSE: {reg_rmse:.3f} mmHg")
print(f"  Improvement over ratio — MAE: {ratio_mae - reg_mae:+.3f}  RMSE: {ratio_rmse - reg_rmse:+.3f}")

# LAD regression (minimises MAE) — more robust to outliers
from scipy.optimize import minimize

def lad_loss(params):
    a, b = params
    return np.abs(pneumatic - (a * perkins + b)).mean()

lad_res      = minimize(lad_loss, x0=[slope, intercept], method="Nelder-Mead")
lad_slope, lad_intercept = lad_res.x
lad_pred     = lad_slope * perkins + lad_intercept
lad_resid    = pneumatic - lad_pred
lad_mae      = np.abs(lad_resid).mean()
lad_rmse     = np.sqrt((lad_resid ** 2).mean())

print(f"\n── LAD regression (minimises MAE — robust to outliers) ──────")
print(f"  slope={lad_slope:.4f}  intercept={lad_intercept:.4f}")
print(f"  MAE:  {lad_mae:.3f} mmHg")
print(f"  RMSE: {lad_rmse:.3f} mmHg")
print(f"  Improvement over ratio — MAE: {ratio_mae - lad_mae:+.3f}  RMSE: {ratio_rmse - lad_rmse:+.3f}")

# ── Multiple regression: Perkins + Pachymetry ─────────────────────────────────
# Pachymetry affects Perkins (applanation) more than Pneumatic, so including
# CCT should absorb some instrument-specific bias.
# Design matrix: [Perkins, Pachymetry, 1]
from numpy.linalg import lstsq

X_multi = np.column_stack([perkins, pachymetry, np.ones(len(perkins))])
coeffs, _, _, _ = lstsq(X_multi, pneumatic, rcond=None)
coef_perkins, coef_pachy, coef_intercept = coeffs
multi_pred  = X_multi @ coeffs
multi_resid = pneumatic - multi_pred
multi_mae   = np.abs(multi_resid).mean()
multi_rmse  = np.sqrt((multi_resid ** 2).mean())

# R² for the multiple model
ss_res = (multi_resid ** 2).sum()
ss_tot = ((pneumatic - pneumatic.mean()) ** 2).sum()
multi_r2 = 1 - ss_res / ss_tot

# Partial correlation of Pachymetry with residual after removing Perkins effect
perkins_resid  = pneumatic - (slope * perkins + intercept)
pachy_r, pachy_p = stats.pearsonr(pachymetry, perkins_resid)

print(f"\n── Multiple OLS (Perkins + Pachymetry, n={len(paired)}) ──────────────")
print(f"  Pneumatic = {coef_perkins:.4f}×Perkins + {coef_pachy:.5f}×Pachymetry + {coef_intercept:.4f}")
print(f"  R²={multi_r2:.4f}  (vs simple OLS R²={r**2:.4f})")
print(f"  Pachymetry partial corr with OLS residuals: r={pachy_r:.3f}  p={pachy_p:.4f}")
print(f"  MAE:  {multi_mae:.3f} mmHg  (vs ratio {ratio_mae:.3f})")
print(f"  RMSE: {multi_rmse:.3f} mmHg  (vs ratio {ratio_rmse:.3f})")
print(f"  Improvement over ratio — MAE: {ratio_mae - multi_mae:+.3f}  RMSE: {ratio_rmse - multi_rmse:+.3f}")
print(f"  NOTE: n={len(paired)} with 3 parameters — interpret with caution.")

# How much does the intercept matter at typical IOP values?
typical_iop = np.array([10, 15, 20, 25])
print(f"\n  Comparison at typical Perkins values:")
print(f"  {'Perkins':>8} {'Ratio':>10} {'OLS':>10} {'LAD':>10}")
for v in typical_iop:
    ratio_v = v * CURRENT_RATIO
    ols_v   = slope * v + intercept
    lad_v   = lad_slope * v + lad_intercept
    print(f"  {v:>8.1f} {ratio_v:>10.2f} {ols_v:>10.2f} {lad_v:>10.2f}")

# ── IOP_raw coverage ──────────────────────────────────────────────────────────

pneumatic_only = df["Pneumatic"].notna() & df["Perkins"].isna()
perkins_only   = df["Pneumatic"].isna()  & df["Perkins"].notna()
both           = df["Pneumatic"].notna() & df["Perkins"].notna()
neither        = df["Pneumatic"].isna()  & df["Perkins"].isna()

print(f"\n── IOP measurement coverage ──────────────────────────────")
print(f"  Pneumatic only: {pneumatic_only.sum()}")
print(f"  Perkins only:   {perkins_only.sum()}")
print(f"  Both:           {both.sum()}")
print(f"  Neither:        {neither.sum()}")
print(f"  Rows where IOP_raw == IOP_corr (no pachy correction): ", end="")

# ── plot ──────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 5), layout="constrained")
gs  = gridspec.GridSpec(1, 3, figure=fig)

x_line = np.linspace(perkins.min() - 1, perkins.max() + 1, 100)

# ── Row 1: conversion fits ────────────────────────────────────────────────────

# 1a. Scatter with all four fits
ax1 = fig.add_subplot(gs[0, :2])  # spans first two columns
ax1.scatter(perkins, pneumatic, alpha=0.6, s=35, color="gray", zorder=3, label="Observed pairs (n=41)")
ax1.plot(x_line, x_line * CURRENT_RATIO,             "r--", lw=2, label=f"Ratio ×{CURRENT_RATIO}  MAE={ratio_mae:.2f}")
ax1.plot(x_line, slope * x_line + intercept,          "b-",  lw=2, label=f"OLS  (slope={slope:.3f}, int={intercept:.2f})  MAE={reg_mae:.2f}")
ax1.plot(x_line, lad_slope * x_line + lad_intercept, "g-",  lw=2, label=f"LAD  (slope={lad_slope:.3f}, int={lad_intercept:.2f})  MAE={lad_mae:.2f}")

# Multi-reg projected at mean CCT
pachy_mean, pachy_sd = pachymetry.mean(), pachymetry.std()
multi_mean_line = coef_perkins * x_line + coef_pachy * pachy_mean + coef_intercept
ax1.plot(x_line, multi_mean_line, color="purple", lw=2, ls="-.",
         label=f"Multi (Perkins+CCT) @ mean CCT  MAE={multi_mae:.2f}")

ax1.set_xlabel("Perkins IOP (mmHg)")
ax1.set_ylabel("Pneumatic IOP (mmHg)")
ax1.set_title("Perkins → Pneumatic conversion: all methods")
ax1.legend(fontsize=8)

# 1b. MAE / RMSE bar chart
ax_bar = fig.add_subplot(gs[0, 2])  # third column
bar_methods = ["Ratio", "OLS", "LAD", "Multi\n(+CCT)"]
maes  = [ratio_mae,  reg_mae,  lad_mae,  multi_mae]
rmses = [ratio_rmse, reg_rmse, lad_rmse, multi_rmse]
bar_colors = ["#e05c5c", "#5c7de0", "#5cc97c", "#9b59b6"]
bx = np.arange(4)
w = 0.35
ax_bar.bar(bx - w/2, maes,  w, label="MAE",  color=bar_colors)
ax_bar.bar(bx + w/2, rmses, w, label="RMSE", color=bar_colors, alpha=0.5)
ax_bar.set_xticks(bx); ax_bar.set_xticklabels(bar_methods, fontsize=8)
ax_bar.set_ylabel("Error (mmHg)")
ax_bar.set_title("MAE & RMSE comparison")
ax_bar.legend(fontsize=9)
ax_bar.set_ylim(0, max(rmses) * 1.25)
for i, (mae, rmse) in enumerate(zip(maes, rmses)):
    ax_bar.text(i - w/2, mae  + 0.05, f"{mae:.2f}",  ha="center", va="bottom", fontsize=8)
    ax_bar.text(i + w/2, rmse + 0.05, f"{rmse:.2f}", ha="center", va="bottom", fontsize=8)

out = REPO / "analysis_data/iop_correction_comparison.png"
fig.savefig(out, dpi=150)
print(f"\nPlot saved to {out}")

# ── Figure 2: Pachymetry analysis ─────────────────────────────────────────────

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

pachy_norm = (pachymetry - pachymetry.mean()) / pachymetry.std()

# 2a. Pachymetry vs Pneumatic–Perkins difference
ax = axes2[0]
diff = pneumatic - perkins
m, b, rr, pp, _ = stats.linregress(pachymetry, diff)
ax.scatter(pachymetry, diff, alpha=0.6, s=35, color="steelblue")
px = np.linspace(pachymetry.min() - 5, pachymetry.max() + 5, 100)
ax.plot(px, m * px + b, "r-", lw=2, label=f"r={rr:.2f}  p={pp:.3f}")
ax.axhline(0, color="k", lw=0.7, ls="--")
ax.set_xlabel("Pachymetry (μm)")
ax.set_ylabel("Pneumatic − Perkins (mmHg)")
ax.set_title("Does CCT predict the instrument gap?")
ax.legend(fontsize=9)

# 2b. Residuals from ratio vs Pachymetry (coloured by size)
ax = axes2[1]
sc = ax.scatter(pachymetry, ratio_resid, c=perkins, cmap="viridis", alpha=0.7, s=35)
plt.colorbar(sc, ax=ax, label="Perkins IOP")
m2, b2, rr2, pp2, _ = stats.linregress(pachymetry, ratio_resid)
ax.plot(px, m2 * px + b2, "r-", lw=2, label=f"r={rr2:.2f}  p={pp2:.3f}")
ax.axhline(0, color="k", lw=0.7, ls="--")
ax.set_xlabel("Pachymetry (μm)")
ax.set_ylabel("Ratio residual (mmHg)")
ax.set_title("Ratio residuals vs Pachymetry\n(colour = Perkins IOP)")
ax.legend(fontsize=9)

# 2c. Multiple regression residuals vs Pachymetry (should be flat if absorbed)
ax = axes2[2]
m3, b3, rr3, pp3, _ = stats.linregress(pachymetry, multi_resid)
ax.scatter(pachymetry, multi_resid, alpha=0.6, s=35, color="darkorange")
ax.plot(px, m3 * px + b3, "r-", lw=2, label=f"r={rr3:.2f}  p={pp3:.3f}")
ax.axhline(0, color="k", lw=0.7, ls="--")
ax.set_xlabel("Pachymetry (μm)")
ax.set_ylabel("Multi-reg residual (mmHg)")
ax.set_title("Multi-reg residuals vs Pachymetry\n(flat = Pachymetry effect absorbed)")
ax.legend(fontsize=9)

# shared y-axis
ylim2 = max(np.abs(diff).max(), np.abs(ratio_resid).max(), np.abs(multi_resid).max()) + 1
for ax in axes2[1:]:
    ax.set_ylim(-ylim2, ylim2)

fig2.suptitle(
    f"Pachymetry as a covariate  (n={len(paired)}, CCT mean={pachymetry.mean():.0f}±{pachymetry.std():.0f} μm)",
    fontsize=11,
)
fig2.tight_layout()
out2 = REPO / "analysis_data/iop_pachymetry_analysis.png"
fig2.savefig(out2, dpi=150)
print(f"Plot saved to {out2}")
