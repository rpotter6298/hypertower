"""
Plot predicted-probability strip charts for V2 runs.

Three styles available via --style:

  strips      (default)
    X = true class, Y = P(Glaucoma), color = true class × fold shade.
    Works for binary and multiclass.

  confidence
    X = predicted class (major) subdivided by true class (minor sub-column).
    Y = model confidence in its own prediction (P of the predicted class).
    Color = true class × fold shade.
    Makes high-confidence mistakes immediately visible.

  triangle    (multiclass only)
    Ternary / simplex plot. Each corner = 100% probability for one class.
    Every sample is a dot placed at its softmax probability vector
    (p_H, p_G, p_S) using barycentric coordinates. The centroid is maximum
    uncertainty (1/3, 1/3, 1/3). Correctly classified samples cluster near
    their true-class corner; mistakes drift toward the wrong corner.

Colour families (light → dark = fold 0 → fold N-1):
  Blues  = Healthy / Normal eyes
  Reds   = Glaucoma eyes
  Greens = Suspect eyes

Usage
-----
    python scripts/output_analysis/visualizations/plot_prob_strips.py \
        --run-dir analysis_data/v2.3_single_multiclass_nocrop/multiclass/single \
        --style triangle --head fused
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


# ── colour families ────────────────────────────────────────────────────────────
_CLASS_FAMILIES = {
    0: ("#aac4ff", "#0a3d91"),   # blues  (healthy / normal)
    1: ("#ffaaaa", "#8b0000"),   # reds   (glaucoma)
    2: ("#aaffcc", "#1a6b3c"),   # greens (suspect)
}

_CLASS_LABELS = {
    0: "Healthy",
    1: "Glaucoma",
    2: "Suspect",
}

_CLASS_LABELS_SHORT = {
    0: "H",
    1: "G",
    2: "S",
}


def _lerp_hex(c1: str, c2: str, t: float) -> tuple:
    def h(c): return tuple(int(c.lstrip("#")[i*2:i*2+2], 16) / 255 for i in range(3))
    r1, g1, b1 = h(c1)
    r2, g2, b2 = h(c2)
    return (r1 + t*(r2-r1), g1 + t*(g2-g1), b1 + t*(b2-b1))


def _fold_colours(n_folds: int) -> dict[int, dict[int, tuple]]:
    out: dict[int, dict[int, tuple]] = {}
    for cls, (light, dark) in _CLASS_FAMILIES.items():
        out[cls] = {}
        for f in range(n_folds):
            t = f / max(n_folds - 1, 1)
            out[cls][f] = _lerp_hex(light, dark, t)
    return out


def _load_folds(run_dir: Path, head: str) -> list[pd.DataFrame]:
    fold_dirs = sorted(
        [p for p in run_dir.glob("fold*") if p.is_dir() and re.search(r"\d+", p.name)],
        key=lambda p: int(re.search(r"\d+", p.name).group()),
    )
    if not fold_dirs:
        raise FileNotFoundError(f"No fold* directories found under {run_dir}")
    frames = []
    for fd in fold_dirs:
        csv = fd / "predictions_classic.csv"
        if not csv.exists():
            print(f"  [warn] {csv} not found, skipping")
            continue
        df = pd.read_csv(csv)
        df["_fold"] = int(re.search(r"\d+", fd.name).group())
        frames.append(df)
    return frames


def _detect_num_classes(df: pd.DataFrame, head: str) -> int:
    return len([c for c in df.columns if c.startswith(f"prob_{head}_c")])


def _draw_grid_legend(ax_leg: plt.Axes, n_folds: int, classes: list[int],
                      colours: dict, title: str = "True class") -> None:
    """Rows = folds, columns = classes grid of coloured dots."""
    from matplotlib.patches import FancyBboxPatch
    ax_leg.axis("off")

    col_xs = np.linspace(0.55, 0.88, len(classes)) if len(classes) > 1 else [0.72]
    row_ys = np.linspace(0.88, 0.05, n_folds + 1)
    header_y, dot_ys = row_ys[0], row_ys[1:]

    for ci, cls in enumerate(classes):
        ax_leg.text(col_xs[ci], header_y, _CLASS_LABELS_SHORT.get(cls, f"C{cls}"),
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                    transform=ax_leg.transAxes)

    for fi in range(n_folds):
        y = dot_ys[fi]
        ax_leg.text(0.05, y, f"Fold {fi}", ha="left", va="center", fontsize=9,
                    transform=ax_leg.transAxes)
        for ci, cls in enumerate(classes):
            ax_leg.scatter([col_xs[ci]], [y], color=colours[cls][fi], s=70, zorder=3,
                           transform=ax_leg.transAxes, clip_on=False)

    ax_leg.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02",
                                    linewidth=0.8, edgecolor="#aaaaaa",
                                    facecolor="#f9f9f9", zorder=0,
                                    transform=ax_leg.transAxes))
    ax_leg.set_title(title, fontsize=9, pad=4)


# ── strips style ───────────────────────────────────────────────────────────────

def plot_strip(run_dir: Path, head: str = "fused", out_dir: Path | None = None,
               jitter_strength: float = 0.08) -> Path:
    """X = true class, Y = P(Glaucoma)."""
    frames   = _load_folds(run_dir, head)
    all_true = sorted({int(v) for df in frames for v in df["y_true"]})
    n_folds  = len(frames)
    colours  = _fold_colours(n_folds)
    rng      = np.random.default_rng(seed=0)
    prob_col = f"prob_{head}_c1"
    x_pos    = {cls: i for i, cls in enumerate(all_true)}

    leg_width = 0.8 + 0.55 * len(all_true)
    fig = plt.figure(figsize=(max(6, 2.2 * len(all_true) + 1), 7))
    gs  = fig.add_gridspec(1, 2, width_ratios=[max(6, 2.2 * len(all_true)), leg_width],
                           wspace=0.08)
    ax     = fig.add_subplot(gs[0])
    ax_leg = fig.add_subplot(gs[1])

    fig.suptitle(f"{run_dir.parent.parent.name}  —  P(Glaucoma)  [{head}]",
                 fontsize=11, y=1.01)

    for fold_idx, df in enumerate(frames):
        for cls in all_true:
            sub = df[df["y_true"] == cls]
            if sub.empty:
                continue
            probs  = sub[prob_col].values
            jitter = rng.uniform(-jitter_strength, jitter_strength, size=len(probs))
            ax.scatter(x_pos[cls] + jitter, probs, color=colours[cls][fold_idx],
                       s=45, alpha=0.88, linewidths=0, zorder=3)

    ax.set_xticks(list(x_pos.values()))
    ax.set_xticklabels([_CLASS_LABELS.get(c, f"C{c}") for c in all_true], fontsize=12)
    ax.set_xlim(-0.5, len(all_true) - 0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Predicted P(Glaucoma)", fontsize=12)
    ax.set_xlabel("True class", fontsize=12)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    _draw_grid_legend(ax_leg, n_folds, all_true, colours, title="Legend")

    fig.tight_layout()
    return _save(fig, out_dir or run_dir / "plots", f"prob_strips_{head}.png")


# ── confidence style ───────────────────────────────────────────────────────────

def plot_confidence(run_dir: Path, head: str = "fused", out_dir: Path | None = None,
                    jitter_strength: float = 0.06) -> Path:
    """
    X = predicted class (major) × true class (minor sub-column).
    Y = model confidence = P(predicted class).
    Color = true class × fold shade.
    """
    frames    = _load_folds(run_dir, head)
    num_cls   = _detect_num_classes(frames[0], head)
    all_cls   = list(range(num_cls))
    all_true  = sorted({int(v) for df in frames for v in df["y_true"]})
    n_folds   = len(frames)
    colours   = _fold_colours(n_folds)
    rng       = np.random.default_rng(seed=0)

    # Sub-column spacing within each predicted-class group
    # E.g. for 3 classes: sub-offsets at -0.25, 0, +0.25
    n_sub       = len(all_true)
    sub_spacing = 0.22
    sub_offsets = np.linspace(-(n_sub - 1) * sub_spacing / 2,
                               (n_sub - 1) * sub_spacing / 2,
                               n_sub)
    sub_off     = {cls: sub_offsets[i] for i, cls in enumerate(all_true)}

    # Major x positions for each predicted class, spaced so sub-columns don't bleed
    group_gap = sub_spacing * n_sub + 0.35
    major_x   = {pc: i * group_gap for i, pc in enumerate(all_cls)}

    leg_width = 0.8 + 0.55 * n_sub
    fig_w     = max(7, group_gap * len(all_cls) * 1.8 + 1)
    fig = plt.figure(figsize=(fig_w, 7))
    gs  = fig.add_gridspec(1, 2, width_ratios=[fig_w - leg_width, leg_width],
                           wspace=0.08)
    ax     = fig.add_subplot(gs[0])
    ax_leg = fig.add_subplot(gs[1])

    fig.suptitle(f"{run_dir.parent.parent.name}  —  Prediction confidence  [{head}]",
                 fontsize=11, y=1.01)

    for fold_idx, df in enumerate(frames):
        prob_cols = [f"prob_{head}_c{c}" for c in all_cls]
        pred_col  = f"pred_{head}"
        for true_cls in all_true:
            sub = df[df["y_true"] == true_cls].copy()
            if sub.empty:
                continue
            for pred_cls in all_cls:
                rows = sub[sub[pred_col] == pred_cls]
                if rows.empty:
                    continue
                # confidence = probability assigned to the predicted class
                conf   = rows[f"prob_{head}_c{pred_cls}"].values
                x_base = major_x[pred_cls] + sub_off[true_cls]
                jitter = rng.uniform(-jitter_strength * 0.5,
                                      jitter_strength * 0.5, size=len(conf))
                ax.scatter(x_base + jitter, conf, color=colours[true_cls][fold_idx],
                           s=45, alpha=0.88, linewidths=0, zorder=3)

    # X-axis: major ticks with predicted-class labels, minor sub-column markers
    ax.set_xlim(-group_gap * 0.5, group_gap * len(all_cls) - group_gap * 0.5)
    ax.set_xticks([major_x[pc] for pc in all_cls])
    ax.set_xticklabels([_CLASS_LABELS.get(pc, f"C{pc}") for pc in all_cls], fontsize=12)

    # Light vertical separators between predicted-class groups
    for i in range(1, len(all_cls)):
        sep_x = (major_x[all_cls[i-1]] + major_x[all_cls[i]]) / 2
        ax.axvline(sep_x, color="#cccccc", linewidth=1.0, zorder=1)

    # Sub-column labels (H/G/S) just below the x-axis
    for pred_cls in all_cls:
        for true_cls in all_true:
            lbl = _CLASS_LABELS_SHORT.get(true_cls, f"C{true_cls}")
            ax.text(major_x[pred_cls] + sub_off[true_cls], -0.085, lbl,
                    ha="center", va="top", fontsize=7, color="#555555",
                    transform=ax.get_xaxis_transform())

    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Confidence  P(predicted class)", fontsize=12)
    ax.set_xlabel("Predicted class", fontsize=12, labelpad=18)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    _draw_grid_legend(ax_leg, n_folds, all_true, colours, title="True class")

    fig.tight_layout()
    return _save(fig, out_dir or run_dir / "plots", f"prob_confidence_{head}.png")


# ── triangle / ternary style ───────────────────────────────────────────────────

# Equilateral triangle vertices in Cartesian space:
#   H (Healthy)  = bottom-left  (0, 0)
#   G (Glaucoma) = bottom-right (1, 0)
#   S (Suspect)  = apex         (0.5, sqrt(3)/2)
_TRI_VERTICES = np.array([
    [0.0, 0.0],               # class 0 – Healthy
    [1.0, 0.0],               # class 1 – Glaucoma
    [0.5, np.sqrt(3) / 2],    # class 2 – Suspect
])


def _bary_to_cart(probs: np.ndarray) -> np.ndarray:
    """
    Convert Nx3 barycentric coordinates (softmax probs) to Nx2 Cartesian.
    probs rows must sum to 1.
    """
    return probs @ _TRI_VERTICES


def _draw_triangle_grid(ax: plt.Axes, levels: tuple = (0.25, 0.5, 0.75)) -> None:
    """Draw the triangle border and iso-probability grid lines."""
    from matplotlib.patches import Polygon
    from matplotlib.lines import Line2D

    # Outer triangle
    tri = Polygon(_TRI_VERTICES, fill=False, edgecolor="#333333", linewidth=1.5, zorder=2)
    ax.add_patch(tri)

    # Grid lines: for each class, lines where p_class = level,
    # parallel to the opposite edge.
    for level in levels:
        for cls in range(3):
            # Points on the two edges adjacent to this vertex at distance `level`
            v0 = _TRI_VERTICES[cls]
            v1 = _TRI_VERTICES[(cls + 1) % 3]
            v2 = _TRI_VERTICES[(cls + 2) % 3]
            # A line at p_cls = level divides the triangle:
            # p1 = level * v0 + (1-level) * v1
            # p2 = level * v0 + (1-level) * v2
            p1 = level * v0 + (1 - level) * v1
            p2 = level * v0 + (1 - level) * v2
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color="#cccccc", linewidth=0.6, zorder=1, linestyle="--")


def plot_triangle(run_dir: Path, head: str = "fused", out_dir: Path | None = None) -> Path:
    """
    Ternary simplex plot of softmax probabilities for a 3-class run.
    Each dot = one eye; position = (p_H, p_G, p_S) in barycentric coords.
    Color = true class × fold shade.
    """
    frames = _load_folds(run_dir, head)
    num_cls = _detect_num_classes(frames[0], head)
    if num_cls != 3:
        raise ValueError(f"Triangle plot requires 3 classes; got {num_cls}. "
                         "Use --style strips for binary.")

    all_true = sorted({int(v) for df in frames for v in df["y_true"]})
    n_folds  = len(frames)
    colours  = _fold_colours(n_folds)

    leg_width = 0.8 + 0.55 * len(all_true)
    fig = plt.figure(figsize=(8, 7))
    gs  = fig.add_gridspec(1, 2, width_ratios=[8 - leg_width, leg_width], wspace=0.05)
    ax     = fig.add_subplot(gs[0], aspect="equal")
    ax_leg = fig.add_subplot(gs[1])

    fig.suptitle(f"{run_dir.parent.parent.name}  —  Simplex  [{head}]",
                 fontsize=11, y=1.01)

    _draw_triangle_grid(ax)

    # Vertex labels — placed just outside each corner
    offsets = [(-0.07, -0.06), (0.07, -0.06), (0.0, 0.06)]
    for cls_idx in range(3):
        lbl = _CLASS_LABELS.get(cls_idx, f"C{cls_idx}")
        vx, vy = _TRI_VERTICES[cls_idx]
        dx, dy = offsets[cls_idx]
        ax.text(vx + dx, vy + dy, lbl, ha="center", va="center",
                fontsize=12, fontweight="bold")

    # Centroid marker
    cx, cy = _TRI_VERTICES.mean(axis=0)
    ax.scatter([cx], [cy], color="#aaaaaa", s=30, marker="+", zorder=2, linewidths=1)
    ax.text(cx + 0.02, cy - 0.04, "1/3 each", fontsize=7, color="#999999", ha="left")

    # Data dots
    rng = np.random.default_rng(seed=0)
    for fold_idx, df in enumerate(frames):
        prob_cols = [f"prob_{head}_c{c}" for c in range(3)]
        probs_all = df[prob_cols].values          # Nx3
        y_true    = df["y_true"].values.astype(int)
        for true_cls in all_true:
            mask  = y_true == true_cls
            if not mask.any():
                continue
            probs = probs_all[mask]               # Kx3
            xy    = _bary_to_cart(probs)          # Kx2
            noise = rng.normal(0, 0.004, xy.shape)
            ax.scatter(xy[:, 0] + noise[:, 0],
                       xy[:, 1] + noise[:, 1],
                       color=colours[true_cls][fold_idx],
                       s=30, alpha=0.80, linewidths=0, zorder=3)

    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.12, 1.02)
    ax.axis("off")

    _draw_grid_legend(ax_leg, n_folds, all_true, colours, title="True class")

    fig.tight_layout()
    return _save(fig, out_dir or run_dir / "plots", f"prob_triangle_{head}.png")


# ── triangle3d style ──────────────────────────────────────────────────────────

# Triangle vertices for the 3D "bread-slice" view.
# Triangles stand upright in the x-z plane; y is the depth (layer) axis.
#   H (Healthy)  = bottom-left  (0,   0)   — back corner of the base
#   G (Glaucoma) = top-centre   (0.5, √3/2) — apex  (visually "at the top")
#   S (Suspect)  = bottom-right (1,   0)   — front corner toward the viewer
_TRI3D_VERTS = np.array([
    [0.0, 0.0],                # class 0 – Healthy   (bottom-left / back)
    [0.5, np.sqrt(3) / 2],     # class 1 – Glaucoma  (top)
    [1.0, 0.0],                # class 2 – Suspect   (bottom-right / front)
])


def _bary_to_cart_3d(probs: np.ndarray) -> np.ndarray:
    """Convert Nx3 softmax probs to Nx2 Cartesian using the 3D vertex layout."""
    return probs @ _TRI3D_VERTS


def plot_triangle_3d(
    run_dir: Path,
    head: str = "fused",
    out_dir: Path | None = None,
    elev: float = 20,
    azim: float = -45,
    y_spacing: float = 0.4,
) -> Path:
    """
    3-D ternary "bread-slice" plot.

    Each true class gets its own vertical triangle slice standing in the x-z
    plane, stacked along the y (depth) axis (camera at ~4:30 / SE position):
      • Healthy  layer is at y=0  (front-left from 4:30 view)
      • Glaucoma layer is at y=1
      • Suspect  layer is at y=2  (back-right from 4:30 view)

    Within every slice the simplex corners are:
      • G (Glaucoma) at the top apex
      • H (Healthy)  at the bottom-left  (back corner of the base)
      • S (Suspect)  at the bottom-right (front corner toward viewer)

    View with elev/azim to see the slices as near-vertical planes with slight
    perspective depth.
    """
    from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.patches import Patch

    frames  = _load_folds(run_dir, head)
    num_cls = _detect_num_classes(frames[0], head)
    if num_cls != 3:
        raise ValueError("triangle3d requires 3 classes.")

    all_true = sorted({int(v) for df in frames for v in df["y_true"]})
    n_folds  = len(frames)
    colours  = _fold_colours(n_folds)
    rng      = np.random.default_rng(seed=0)

    # Layer y positions: H front-left (y=0), S back-right (y=2) from 4:30 view
    y_pos = {cls: i * y_spacing for i, cls in enumerate(all_true)}

    # Triangle wireframe in (x, z) — loop closed
    wire_x = np.append(_TRI3D_VERTS[:, 0], _TRI3D_VERTS[0, 0])
    wire_z = np.append(_TRI3D_VERTS[:, 1], _TRI3D_VERTS[0, 1])

    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)

    fig.suptitle(run_dir.parent.parent.name, fontsize=11)

    # ── pre-compute all xz points per class (needed for KDE) ─────────────────
    from scipy.stats import gaussian_kde
    import matplotlib.colors as mcolors

    _xz_by_cls: dict[int, np.ndarray] = {}
    for df in frames:
        prob_cols_all = [f"prob_{head}_c{c}" for c in range(3)]
        p_all  = df[prob_cols_all].values
        yt_all = df["y_true"].values.astype(int)
        for cls in all_true:
            m = yt_all == cls
            if m.any():
                xz_pts = _bary_to_cart_3d(p_all[m])
                _xz_by_cls[cls] = (
                    np.vstack([_xz_by_cls[cls], xz_pts])
                    if cls in _xz_by_cls else xz_pts
                )

    # KDE grid setup — evaluate on a 40×40 grid masked to the triangle interior
    _G = 40
    _x_lin = np.linspace(0.0, 1.0, _G)
    _z_lin = np.linspace(0.0, np.sqrt(3) / 2, _G)
    _dx    = _x_lin[1] - _x_lin[0]
    _dz    = _z_lin[1] - _z_lin[0]
    _XX, _ZZ = np.meshgrid(_x_lin, _z_lin)          # (_G, _G)

    # Vectorised inside-triangle test (barycentric)
    vH, vG, vS = _TRI3D_VERTS
    _denom = (vG[1] - vS[1]) * (vH[0] - vS[0]) + (vS[0] - vG[0]) * (vH[1] - vS[1])
    def _inside(px, pz):
        la = ((vG[1] - vS[1]) * (px - vS[0]) + (vS[0] - vG[0]) * (pz - vS[1])) / _denom
        lb = ((vS[1] - vH[1]) * (px - vS[0]) + (vH[0] - vS[0]) * (pz - vS[1])) / _denom
        return (la >= 0) & (lb >= 0) & ((1 - la - lb) >= 0)

    _inside_mask = _inside(_XX.ravel(), _ZZ.ravel()).reshape(_G, _G)
    _grid_pts    = np.vstack([_XX.ravel(), _ZZ.ravel()])  # 2×(_G²)

    # ── draw triangle wireframe + density heatmap at each y layer ─────────────
    for cls in all_true:
        y = y_pos[cls]
        base_col = colours[cls][n_folds // 2]
        rgba_base = np.array(mcolors.to_rgba(base_col))

        # Wireframe
        ax.plot(wire_x, np.full_like(wire_x, y), wire_z,
                color=base_col, linewidth=1.2, alpha=0.65, zorder=1)

        # ── density heatmap ────────────────────────────────────────────────
        xz_pts = _xz_by_cls.get(cls)
        if xz_pts is not None and len(xz_pts) >= 2:
            kde     = gaussian_kde(xz_pts.T, bw_method="silverman")
            density = kde(_grid_pts).reshape(_G, _G)
            density[~_inside_mask] = 0.0
            inside_vals = density[_inside_mask]
            d_max = inside_vals.max()
            if d_max > 0:
                # Normalise by 95th-percentile density so a tight peak at one
                # corner doesn't wash out the rest of the triangle.  Values
                # above the cap are clipped to the max alpha.
                d_ref = np.percentile(inside_vals[inside_vals > 0], 95)
                if d_ref == 0:
                    d_ref = d_max
                alpha_grid = np.clip(density / d_ref, 0, 1) * 0.50
                # Build one quad per inside cell, coloured by density alpha
                quads, face_cols = [], []
                for i in range(_G):
                    for j in range(_G):
                        if not _inside_mask[i, j]:
                            continue
                        xi, zj = _x_lin[j], _z_lin[i]
                        x0, x1 = xi - _dx / 2, xi + _dx / 2
                        z0, z1 = zj - _dz / 2, zj + _dz / 2
                        quads.append([(x0, y, z0), (x1, y, z0),
                                      (x1, y, z1), (x0, y, z1)])
                        fc = rgba_base.copy()
                        fc[3] = float(alpha_grid[i, j])
                        face_cols.append(fc)
                heat = Poly3DCollection(quads, facecolors=face_cols,
                                        edgecolors="none", zorder=0)
                ax.add_collection3d(heat)

        # Iso-prob grid lines
        for level in (0.25, 0.5, 0.75):
            for vi in range(3):
                v0 = _TRI3D_VERTS[vi]
                v1 = _TRI3D_VERTS[(vi + 1) % 3]
                v2 = _TRI3D_VERTS[(vi + 2) % 3]
                p1 = level * v0 + (1 - level) * v1
                p2 = level * v0 + (1 - level) * v2
                ax.plot([p1[0], p2[0]], [y, y], [p1[1], p2[1]],
                        color="#cccccc", linewidth=0.4, linestyle="--",
                        alpha=0.45, zorder=1)

    # ── vertex labels just outside the front face of the merged volume ────────
    # Place labels at the H-layer plane (y=0, front-left) but pushed slightly
    # in front of the y-axis so they don't collide with the wireframe.
    lbl_info = [
        (0, -0.13, -0.08),   # H: bottom-left
        (1,  0.00,  0.10),   # G: top
        (2,  0.13, -0.08),   # S: bottom-right
    ]
    front_y = min(y_pos.values()) - 0.08
    for ci, dx, dz in lbl_info:
        vx, vz = _TRI3D_VERTS[ci]
        ax.text(vx + dx, front_y, vz + dz,
                _CLASS_LABELS.get(ci, f"C{ci}"),
                fontsize=10, fontweight="bold", ha="center", va="center",
                zorder=10)

    # ── data dots ─────────────────────────────────────────────────────────────
    for fold_idx, df in enumerate(frames):
        prob_cols = [f"prob_{head}_c{c}" for c in range(3)]
        probs_all = df[prob_cols].values
        y_true    = df["y_true"].values.astype(int)
        for true_cls in all_true:
            mask = y_true == true_cls
            if not mask.any():
                continue
            probs = probs_all[mask]
            xz    = _bary_to_cart_3d(probs)          # Nx2 (x, z)
            noise = rng.normal(0, 0.004, xz.shape)
            y_vals = np.full(mask.sum(), y_pos[true_cls])
            ax.scatter(xz[:, 0] + noise[:, 0],
                       y_vals,
                       xz[:, 1] + noise[:, 1],
                       color=colours[true_cls][fold_idx],
                       s=22, alpha=0.82, linewidths=0,
                       depthshade=False, zorder=5)

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.3, max(y_pos.values()) + 0.3)
    ax.set_zlim(-0.1, np.sqrt(3) / 2 + 0.1)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)

    # ── legend: same grid style as 2D plots, placed as an inset axes ─────────
    ax_leg = fig.add_axes([0.68, 0.65, 0.28, 0.30])
    _draw_grid_legend(ax_leg, n_folds, all_true, colours, title="True class")

    fig.tight_layout()
    return _save(fig, out_dir or run_dir / "plots", f"prob_triangle3d_{head}.png")


# ── shared save helper ─────────────────────────────────────────────────────────

def _save(fig: plt.Figure, out_dir: Path, filename: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--head", default="fused", choices=["fused", "img", "md"])
    ap.add_argument("--style", default="strips",
                    choices=["strips", "confidence", "triangle", "triangle3d"],
                    help="strips: X=true class, Y=P(Glaucoma).  "
                         "confidence: X=predicted class × true sub-column, Y=confidence.  "
                         "triangle: ternary simplex plot (multiclass only).")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rd = Path(args.run_dir)
    od = Path(args.out) if args.out else None
    if args.style == "confidence":
        plot_confidence(rd, head=args.head, out_dir=od)
    elif args.style == "triangle":
        plot_triangle(rd, head=args.head, out_dir=od)
    elif args.style == "triangle3d":
        plot_triangle_3d(rd, head=args.head, out_dir=od)
    else:
        plot_strip(rd, head=args.head, out_dir=od)


if __name__ == "__main__":
    main()
