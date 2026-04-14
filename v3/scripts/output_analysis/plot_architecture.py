"""
Publication-quality architecture diagrams for HyperTower.

Generates:
  architecture_single_tower.png   — single-eye image-only tower
  architecture_hypertower.png     — single-eye image + clinical fusion
  architecture_ensemble.png       — bilateral ensemble (two HyperTowers + average)
  architecture_fused_head.png     — bilateral ensemble + learned head

Usage:
    python -m v3.scripts.output_analysis.plot_architecture
    python -m v3.scripts.output_analysis.plot_architecture --out figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ── Colour palette ────────────────────────────────────────────────────────────
C_IMG    = "#4e8d3a"  # green  — image / CNN
C_MD     = "#4c72b0"  # blue   — clinical / MLP
C_BRIDGE = "#c44e52"  # red    — bridge / fusion
C_EMB    = "#2a9d8f"  # teal   — embedding vectors (z)
C_OUT    = "#8c6bb1"  # purple — output nodes
C_HEAD   = "#d4a017"  # gold   — learned head / average
C_INPUT  = "#a0a0a0"  # grey   — raw input nodes
C_BG     = "#e8e8e8"
C_ARROW  = "#444444"
FONT     = "DejaVu Sans"


# ── Low-level primitives ──────────────────────────────────────────────────────


def _box(
    ax,
    cx,
    cy,
    w,
    h,
    color,
    text="",
    fontsize=9,
    text_color="white",
    bold=False,
    alpha=0.92,
    radius=0.12,
    lw=1.5,
):
    """Rounded rectangle centered at (cx, cy)."""
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=color,
        edgecolor="white",
        linewidth=lw,
        alpha=alpha,
        zorder=3,
        transform=ax.transData,
    )
    ax.add_patch(patch)
    if text:
        ax.text(
            cx,
            cy,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=text_color,
            fontweight="bold" if bold else "normal",
            fontfamily=FONT,
            zorder=4,
        )
    return patch


def _arrow(ax, x0, y0, x1, y1, lw=1.6, color=C_ARROW, style="->", rad=0.0):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle=style,
            color=color,
            lw=lw,
            connectionstyle=f"arc3,rad={rad}",
        ),
        zorder=2,
    )


def _text(
    ax, x, y, s, fontsize=8, color="#333333", ha="center", va="center", bold=False
):
    ax.text(
        x,
        y,
        s,
        ha=ha,
        va=va,
        fontsize=fontsize,
        color=color,
        fontfamily=FONT,
        fontweight="bold" if bold else "normal",
        zorder=5,
    )


def _bracket(
    ax,
    x,
    y0,
    y1,
    text="",
    fontsize=8.5,
    color="#888888",
    pad=0.15,
    lw=1.4,
    badge_color=None,
):
    """Vertical C-bracket on the right side.
    If badge_color is set, the label is drawn as white text on a filled badge."""
    mid = (y0 + y1) / 2
    ax.plot(
        [x, x + pad, x + pad, x],
        [y1, y1, y0, y0],
        color=color,
        lw=lw,
        solid_capstyle="round",
        zorder=2,
    )
    if text:
        if badge_color:
            ax.text(
                x + pad * 1.4,
                mid,
                text,
                ha="left",
                va="center",
                fontsize=fontsize,
                color="white",
                fontfamily=FONT,
                fontweight="bold",
                zorder=6,
                bbox=dict(
                    facecolor=badge_color,
                    edgecolor="none",
                    pad=3.5,
                    boxstyle="round,pad=0.3",
                ),
            )
        else:
            ax.text(
                x + pad * 1.4,
                mid,
                text,
                ha="left",
                va="center",
                fontsize=fontsize,
                color=color,
                fontfamily=FONT,
                style="italic",
            )


def _setup(fig, ax, w, h, title):
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    ax.set_facecolor(C_BG)
    fig.patch.set_facecolor(C_BG)
    if title:
        ax.set_title(
            title,
            fontsize=12,
            fontweight="bold",
            fontfamily=FONT,
            pad=10,
            color="#222222",
        )


# ── Reusable sub-blocks ───────────────────────────────────────────────────────


def _draw_cnn_block(ax, x_center, y, w=2.0, h=0.75):
    """Three-layer CNN block with labels: Conv Layers → Conv Layers → GAP."""
    labels = ["Conv\nLayers", "Conv\nLayers", "GAP"]
    sub_w = [w * 0.42, w * 0.30, w * 0.22]
    sub_h = [h, h * 0.82, h * 0.65]
    alphas = [0.82, 0.74, 0.66]
    fsizes = [8.0, 7.5, 7.5]
    gap = (w - sum(sub_w)) / 2
    xs = [
        x_center - w / 2 + sub_w[0] / 2,
        x_center - w / 2 + sub_w[0] + gap + sub_w[1] / 2,
        x_center - w / 2 + sub_w[0] + gap + sub_w[1] + gap + sub_w[2] / 2,
    ]
    for i, (sx, sw, sh, lbl, alp, fs) in enumerate(
        zip(xs, sub_w, sub_h, labels, alphas, fsizes)
    ):
        _box(ax, sx, y, sw, sh, C_IMG, lbl, fontsize=fs, alpha=alp, radius=0.08)
        if i < 2:
            _arrow(
                ax, sx + sw / 2, y, xs[i + 1] - sub_w[i + 1] / 2, y, lw=1.2, style="-|>"
            )
    return xs[-1] + sub_w[-1] / 2


def _draw_mlp_block(ax, x_center, y, w=1.4, h=0.65):
    """Two-layer MLP block: FC(128) → FC(128) (hidden_dim=128 both layers)."""
    labels = ["FC\n(128)", "FC\n(128)"]
    w0, w1 = w * 0.55, w * 0.45
    gap = w - w0 - w1
    x0 = x_center - w / 2 + w0 / 2
    x1 = x0 + w0 / 2 + gap + w1 / 2
    _box(ax, x0, y, w0, h, C_MD, labels[0], fontsize=8.0, alpha=0.82, radius=0.08)
    _arrow(ax, x0 + w0 / 2, y, x1 - w1 / 2, y, lw=1.2, style="-|>")
    _box(
        ax, x1, y, w1, h * 0.88, C_MD, labels[1], fontsize=7.5, alpha=0.72, radius=0.08
    )
    return x1 + w1 / 2


def _draw_embedding(ax, x, y, w=0.40, h=0.75, label="z\n(emb)"):
    _box(ax, x + w / 2, y, w, h, C_EMB, label, fontsize=8, bold=True, radius=0.08)
    return x + w


def _draw_output(ax, x, y, dy=0.45, classes=("Glaucoma", "Normal")):
    """Stacked output class boxes, connected from (x, y) via arrows."""
    n = len(classes)
    bw = 1.10
    bh = 0.38
    gap = 0.08
    total = n * bh + (n - 1) * gap
    y_top = y + total / 2 - bh / 2

    for i, cls in enumerate(classes):
        cy = y_top - i * (bh + gap)
        _box(ax, x + bw / 2, cy, bw, bh, C_OUT, cls, fontsize=8.5, radius=0.08)
        _arrow(ax, x, y, x, cy, lw=1.1, style="-|>", rad=0.0)

    _text(ax, x + bw / 2, y - total / 2 - 0.20, "Softmax", fontsize=7.5, color=C_OUT)


def _draw_compact_ht(ax, x_left, y_img, y_md, eye_label):
    """Compact HyperTower block: Image+Clinical boxes → Bridge.
    Returns (x_right_of_bridge, y_bridge_center).
    """
    bw_img = 1.40
    bh_img = 0.72
    bw_md = 1.20
    bh_md = 0.62
    bw_br = 0.72
    cy_br = (y_img + y_md) / 2
    bh_br = abs(y_img - y_md) * 0.60

    # Image box: CNN Backbone
    _box(
        ax,
        x_left + bw_img / 2,
        y_img,
        bw_img,
        bh_img,
        C_IMG,
        f"{eye_label}\nCNN Backbone",
        fontsize=8.5,
        radius=0.08,
    )
    # MD box: Clinical MLP
    _box(
        ax,
        x_left + bw_md / 2,
        y_md,
        bw_md,
        bh_md,
        C_MD,
        f"{eye_label}\nClinical MLP",
        fontsize=8.5,
        radius=0.08,
    )

    # Arrows to bridge
    br_x = x_left + max(bw_img, bw_md) + 0.60
    _arrow(ax, x_left + bw_img, y_img, br_x - bw_br / 2, cy_br, lw=1.2, style="-|>")
    _arrow(ax, x_left + bw_md, y_md, br_x - bw_br / 2, cy_br, lw=1.2, style="-|>")

    # Bridge label kept simple — detail lives in the hypertower diagram
    _box(
        ax,
        br_x,
        cy_br,
        bw_br,
        max(bh_br, 0.70),
        C_BRIDGE,
        "Bridge",
        fontsize=8.0,
        radius=0.08,
    )

    return br_x + bw_br / 2, cy_br


# ── Figure 1: Single Tower ────────────────────────────────────────────────────


def make_single_tower(out_dir: Path):
    W, H = 9.0, 3.2
    fig, ax = plt.subplots(figsize=(W, H))
    _setup(fig, ax, W, H, "Single Tower")

    cy = H / 2

    # Input
    _box(
        ax,
        0.75,
        cy,
        0.95,
        0.60,
        C_INPUT,
        "Fundus\nImage",
        fontsize=8.5,
        radius=0.08,
        alpha=0.75,
        text_color="#333",
    )
    _arrow(ax, 1.22, cy, 1.60, cy)

    # CNN Backbone
    cnn_x_right = _draw_cnn_block(ax, x_center=3.10, y=cy, w=2.80, h=0.78)
    _text(ax, 3.10, cy - 0.68, "CNN Backbone", fontsize=8.5, color=C_IMG, bold=True)
    _arrow(ax, 1.60, cy, 1.73, cy, lw=1.4, style="-|>")

    # Embedding
    emb_x_right = _draw_embedding(ax, x=cnn_x_right + 0.28, y=cy, w=0.48, h=0.78)
    _arrow(ax, cnn_x_right, cy, cnn_x_right + 0.28, cy, lw=1.4, style="-|>")

    # Classifier
    _arrow(ax, emb_x_right, cy, emb_x_right + 0.25, cy, lw=1.4, style="-|>")
    _draw_output(ax, emb_x_right + 0.25, cy)

    path = out_dir / "architecture_single_tower.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 2: HyperTower (single eye) ────────────────────────────────────────


def make_hypertower(out_dir: Path):
    W, H = 11.0, 5.5
    fig, ax = plt.subplots(figsize=(W, H))
    _setup(fig, ax, W, H, "HyperTower — Single Eye")

    y_img = 3.70
    y_md = 1.60

    # ── Image tower ────────────────────────────────────────────────
    _box(
        ax,
        0.80,
        y_img,
        1.00,
        0.60,
        C_INPUT,
        "Fundus\nImage",
        fontsize=8.5,
        radius=0.08,
        alpha=0.75,
        text_color="#333",
    )
    _arrow(ax, 1.30, y_img, 1.85, y_img)
    cnn_x_r = _draw_cnn_block(ax, x_center=3.50, y=y_img, w=2.80, h=0.75)
    _text(ax, 3.50, y_img - 0.65, "CNN Backbone", fontsize=8, color=C_IMG, bold=True)
    _arrow(ax, 1.85, y_img, 1.98, y_img, lw=1.4, style="-|>")

    emb_img_x = _draw_embedding(ax, x=cnn_x_r + 0.30, y=y_img, w=0.65, h=0.75)
    _arrow(ax, cnn_x_r, y_img, cnn_x_r + 0.30, y_img, lw=1.4, style="-|>")
    _text(
        ax,
        (1.30 + emb_img_x) / 2,
        y_img + 0.65,
        "Image Tower",
        fontsize=9,
        color=C_IMG,
        bold=True,
    )

    # ── Clinical tower ─────────────────────────────────────────────
    _box(
        ax,
        0.80,
        y_md,
        1.00,
        0.55,
        C_INPUT,
        "Clinical\nData",
        fontsize=8.5,
        radius=0.08,
        alpha=0.75,
        text_color="#333",
    )
    _arrow(ax, 1.30, y_md, 1.65, y_md)
    mlp_x_r = _draw_mlp_block(ax, x_center=2.90, y=y_md, w=1.60, h=0.65)
    _arrow(ax, 1.65, y_md, 1.74, y_md, lw=1.4, style="-|>")

    emb_md_x = _draw_embedding(ax, x=mlp_x_r + 0.30, y=y_md, w=0.65, h=0.65)
    _arrow(ax, mlp_x_r, y_md, mlp_x_r + 0.30, y_md, lw=1.4, style="-|>")
    _text(
        ax,
        (1.30 + emb_md_x) / 2,
        y_md - 0.60,
        "Clinical Tower",
        fontsize=9,
        color=C_MD,
        bold=True,
    )

    # ── Bridge ─────────────────────────────────────────────────────
    br_x = max(emb_img_x, emb_md_x) + 0.80
    cy_br = (y_img + y_md) / 2
    bh_br = abs(y_img - y_md) * 0.55

    _arrow(ax, emb_img_x, y_img, br_x - 0.40, cy_br, lw=1.4, style="-|>")
    _arrow(ax, emb_md_x, y_md, br_x - 0.40, cy_br, lw=1.4, style="-|>")
    _box(
        ax,
        br_x,
        cy_br,
        1.40,
        max(bh_br, 1.35),
        C_BRIDGE,
        "Bridge\nFC(img→256)\nFC(md→256)\n⊙ Hadamard\n→ ReLU→FC(2)",
        fontsize=8,
        radius=0.10,
    )

    # ── Output ─────────────────────────────────────────────────────
    out_x = br_x + 0.65 + 0.40
    _arrow(ax, br_x + 0.65, cy_br, out_x, cy_br, lw=1.4, style="-|>")
    _draw_output(ax, out_x, cy_br)

    # ── Bracket (right of output nodes; output bw=1.10 so right edge = out_x+1.10)
    _bracket(
        ax,
        x=out_x + 1.25,
        y0=y_md - 0.50,
        y1=y_img + 0.50,
        text="HyperTower",
        fontsize=9,
        pad=0.22,
        color="#555",
        badge_color="#555",
    )

    path = out_dir / "architecture_hypertower.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 3: Bilateral Ensemble ─────────────────────────────────────────────


def make_ensemble(out_dir: Path):
    W, H = 10.5, 7.5
    fig, ax = plt.subplots(figsize=(W, H))
    _setup(fig, ax, W, H, "Bilateral Ensemble HyperTower")

    x_left = 3.0
    inp_cx = 1.85
    inp_w  = 0.90
    inp_h  = 0.55

    # OD (top)
    od_y_img, od_y_md = 5.90, 4.60
    br_od_x, cy_od = _draw_compact_ht(
        ax, x_left=x_left, y_img=od_y_img, y_md=od_y_md, eye_label="OD"
    )
    _text(ax, 0.45, (od_y_img + od_y_md) / 2, "OD\n(Right Eye)",
          fontsize=9, color="#444", bold=True)
    _box(ax, inp_cx, od_y_img, inp_w, inp_h, C_INPUT, "Fundus\nImage",
         fontsize=8, radius=0.08, alpha=0.75, text_color="#333")
    _box(ax, inp_cx, od_y_md, inp_w, inp_h, C_INPUT, "Clinical\nData",
         fontsize=8, radius=0.08, alpha=0.75, text_color="#333")
    _arrow(ax, inp_cx + inp_w / 2, od_y_img, x_left, od_y_img, lw=1.2, style="-|>")
    _arrow(ax, inp_cx + inp_w / 2, od_y_md,  x_left, od_y_md,  lw=1.2, style="-|>")

    # OS (bottom)
    os_y_img, os_y_md = 2.80, 1.50
    br_os_x, cy_os = _draw_compact_ht(
        ax, x_left=x_left, y_img=os_y_img, y_md=os_y_md, eye_label="OS"
    )
    _text(ax, 0.45, (os_y_img + os_y_md) / 2, "OS\n(Left Eye)",
          fontsize=9, color="#444", bold=True)
    _box(ax, inp_cx, os_y_img, inp_w, inp_h, C_INPUT, "Fundus\nImage",
         fontsize=8, radius=0.08, alpha=0.75, text_color="#333")
    _box(ax, inp_cx, os_y_md, inp_w, inp_h, C_INPUT, "Clinical\nData",
         fontsize=8, radius=0.08, alpha=0.75, text_color="#333")
    _arrow(ax, inp_cx + inp_w / 2, os_y_img, x_left, os_y_img, lw=1.2, style="-|>")
    _arrow(ax, inp_cx + inp_w / 2, os_y_md,  x_left, os_y_md,  lw=1.2, style="-|>")

    # Average node
    avg_x = max(br_od_x, br_os_x) + 1.20
    avg_y = (cy_od + cy_os) / 2
    avg_size = 0.90

    _arrow(ax, br_od_x, cy_od, avg_x - avg_size / 2, avg_y, lw=1.4, style="-|>")
    _arrow(ax, br_os_x, cy_os, avg_x - avg_size / 2, avg_y, lw=1.4, style="-|>")
    _box(
        ax,
        avg_x,
        avg_y,
        avg_size,
        avg_size,
        C_HEAD,
        "Average",
        fontsize=10,
        bold=True,
        radius=0.10,
    )

    # Output
    out_x = avg_x + avg_size / 2 + 0.50
    _arrow(ax, avg_x + avg_size / 2, avg_y, out_x, avg_y, lw=1.5, style="-|>")
    _draw_output(ax, out_x, avg_y)

    # Side brackets — white text on badge
    _bracket(
        ax,
        x=br_od_x + 0.10,
        y0=od_y_md - 0.45,
        y1=od_y_img + 0.45,
        text="OD HyperTower",
        fontsize=8.5,
        pad=0.20,
        color="#555",
        badge_color="#555",
    )
    _bracket(
        ax,
        x=br_os_x + 0.10,
        y0=os_y_md - 0.45,
        y1=os_y_img + 0.45,
        text="OS HyperTower",
        fontsize=8.5,
        pad=0.20,
        color="#555",
        badge_color="#555",
    )

    path = out_dir / "architecture_ensemble.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 4: Fused Head ──────────────────────────────────────────────────────


def make_fused_head(out_dir: Path):
    W, H = 10.5, 7.5
    fig, ax = plt.subplots(figsize=(W, H))
    _setup(fig, ax, W, H, "Fused Head Bilateral HyperTower")

    x_left = 3.0
    inp_cx = 1.85
    inp_w  = 0.90
    inp_h  = 0.55

    # OD (top) — same layout as ensemble
    od_y_img, od_y_md = 5.90, 4.60
    br_od_x, cy_od = _draw_compact_ht(
        ax, x_left=x_left, y_img=od_y_img, y_md=od_y_md, eye_label="OD"
    )
    _text(ax, 0.45, (od_y_img + od_y_md) / 2, "OD\n(Right Eye)",
          fontsize=9, color="#444", bold=True)
    _box(ax, inp_cx, od_y_img, inp_w, inp_h, C_INPUT, "Fundus\nImage",
         fontsize=8, radius=0.08, alpha=0.75, text_color="#333")
    _box(ax, inp_cx, od_y_md, inp_w, inp_h, C_INPUT, "Clinical\nData",
         fontsize=8, radius=0.08, alpha=0.75, text_color="#333")
    _arrow(ax, inp_cx + inp_w / 2, od_y_img, x_left, od_y_img, lw=1.2, style="-|>")
    _arrow(ax, inp_cx + inp_w / 2, od_y_md,  x_left, od_y_md,  lw=1.2, style="-|>")

    # OS (bottom)
    os_y_img, os_y_md = 2.80, 1.50
    br_os_x, cy_os = _draw_compact_ht(
        ax, x_left=x_left, y_img=os_y_img, y_md=os_y_md, eye_label="OS"
    )
    _text(ax, 0.45, (os_y_img + os_y_md) / 2, "OS\n(Left Eye)",
          fontsize=9, color="#444", bold=True)
    _box(ax, inp_cx, os_y_img, inp_w, inp_h, C_INPUT, "Fundus\nImage",
         fontsize=8, radius=0.08, alpha=0.75, text_color="#333")
    _box(ax, inp_cx, os_y_md, inp_w, inp_h, C_INPUT, "Clinical\nData",
         fontsize=8, radius=0.08, alpha=0.75, text_color="#333")
    _arrow(ax, inp_cx + inp_w / 2, os_y_img, x_left, os_y_img, lw=1.2, style="-|>")
    _arrow(ax, inp_cx + inp_w / 2, os_y_md,  x_left, os_y_md,  lw=1.2, style="-|>")

    _bracket(
        ax,
        x=br_od_x + -0.17,
        y0=od_y_md - 0.45,
        y1=od_y_img + 0.45,
        text="OD HyperTower",
        fontsize=8.5,
        pad=0.20,
        color="#555",
        badge_color="#555",
    )
    _bracket(
        ax,
        x=br_os_x + -0.17,
        y0=os_y_md - 0.45,
        y1=os_y_img + 0.45,
        text="OS HyperTower",
        fontsize=8.5,
        pad=0.20,
        color="#555",
        badge_color="#555",
    )

    # Fused Head box with logit MLP detail
    avg_y = (cy_od + cy_os) / 2
    head_x = max(br_od_x, br_os_x) + 2.20
    head_w = 1.80
    head_h = 1.20

    _arrow(ax, br_od_x + 0.05, cy_od, head_x - head_w / 2, avg_y, lw=1.4, style="-|>")
    _arrow(ax, br_os_x + 0.05, cy_os, head_x - head_w / 2, avg_y, lw=1.4, style="-|>")
    _box(
        ax,
        head_x,
        avg_y,
        head_w,
        head_h,
        C_HEAD,
        "Fused Head\ncat(l_OD, l_OS)\n→ FC(64) → logits",
        fontsize=8.5,
        bold=False,
        radius=0.10,
    )

    # Output nodes + softmax
    out_x = head_x + head_w / 2 + 0.50
    _arrow(ax, head_x + head_w / 2, avg_y, out_x, avg_y, lw=1.5, style="-|>")
    _draw_output(ax, out_x, avg_y)

    path = out_dir / "architecture_fused_head.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "v3" / "figures",
        help="Output directory",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("Generating architecture diagrams...")
    make_single_tower(args.out)
    make_hypertower(args.out)
    make_ensemble(args.out)
    make_fused_head(args.out)
    print("Done.")


if __name__ == "__main__":
    main()
