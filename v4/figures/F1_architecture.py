"""F1 — System architecture diagram.

Bilateral multimodal fusion architecture. Modelled on v3's architecture_fused_head
but adapted for v4 + manuscript terminology:

  * "OD HyperTower" / "OS HyperTower"  ->  "OD Fusion" / "OS Fusion"
  * Bridge boxes show their math explicitly (image projection, clinical
    projection, fusion operation), no longer abbreviated "Bridge"
  * Title drops the HyperTower brand

Re-run anytime:
    python -m v4.figures.F1_architecture
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = Path(__file__).parent / "output" / "F1_architecture.png"

# ── Palette (matches v3 plot_architecture.py for visual consistency) ────────
C_IMG    = "#4e8d3a"  # green — image / CNN
C_MD     = "#4c72b0"  # blue  — clinical network
C_BRIDGE = "#c44e52"  # red   — fusion bridge
C_HEAD   = "#d4a017"  # gold  — patient-level head
C_OUT    = "#8c6bb1"  # purple — output classes
C_INPUT  = "#a0a0a0"  # grey  — raw inputs
C_BG     = "#e8e8e8"
C_ARROW  = "#444444"
FONT     = "DejaVu Sans"


# ── Primitives ───────────────────────────────────────────────────────────────

def _box(ax, cx, cy, w, h, color, text="", fontsize=9, text_color="white",
         bold=False, alpha=0.92, radius=0.12, lw=1.5):
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=color, edgecolor="white", linewidth=lw, alpha=alpha, zorder=3,
        transform=ax.transData,
    )
    ax.add_patch(patch)
    if text:
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=fontsize, color=text_color,
                fontweight="bold" if bold else "normal",
                fontfamily=FONT, zorder=4)
    return patch


def _arrow(ax, x0, y0, x1, y1, lw=1.4, color=C_ARROW, style="-|>"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=2)


def _text(ax, x, y, s, fontsize=9, color="#333", ha="center", va="center", bold=False):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fontsize, color=color,
            fontfamily=FONT, fontweight="bold" if bold else "normal", zorder=5)


def _bracket(ax, x, y0, y1, text="", pad=0.20, fontsize=9, badge_color="#555"):
    mid = (y0 + y1) / 2
    ax.plot([x, x + pad, x + pad, x], [y1, y1, y0, y0],
            color=badge_color, lw=1.4, solid_capstyle="round", zorder=2)
    if text:
        ax.text(x + pad * 1.4, mid, text, ha="left", va="center",
                fontsize=fontsize, color="white", fontfamily=FONT, fontweight="bold",
                zorder=6,
                bbox=dict(facecolor=badge_color, edgecolor="none", pad=3.5,
                          boxstyle="round,pad=0.3"))


def _tag(ax, cx, cy, letter, color="#222"):
    """Small square tag in the top-right corner of a main-diagram box."""
    ax.text(cx, cy, letter, ha="center", va="center",
            fontsize=8.5, color="white", fontfamily=FONT, fontweight="bold",
            zorder=6,
            bbox=dict(facecolor=color, edgecolor="white", linewidth=1.0,
                      boxstyle="round,pad=0.20"))


def _draw_output(ax, x, y, classes=("Glaucoma", "Normal")):
    bw, bh, gap = 1.10, 0.38, 0.08
    n = len(classes)
    total = n * bh + (n - 1) * gap
    y_top = y + total / 2 - bh / 2
    for i, cls in enumerate(classes):
        cy = y_top - i * (bh + gap)
        _box(ax, x + bw / 2, cy, bw, bh, C_OUT, cls, fontsize=8.5, radius=0.08)
        _arrow(ax, x, y, x, cy, lw=1.1, style="-|>")
    _text(ax, x + bw / 2, y - total / 2 - 0.20, "Softmax",
          fontsize=7.5, color=C_OUT)


def _draw_eye_fusion(ax, x_left, y_img, y_md, eye_label):
    """One eye's row: Image Network box + Clinical Network box -> Fusion bridge box.

    Returns (x_right_of_bridge, y_bridge_center).
    """
    bw_img, bh_img = 1.45, 0.72
    bw_md,  bh_md  = 1.48, 0.66
    bw_br,  bh_br  = 1.75, 1.30

    # Image network box
    _box(ax, x_left + bw_img / 2, y_img, bw_img, bh_img, C_IMG,
         f"{eye_label}\nCNN", fontsize=8.5, radius=0.08)
    _tag(ax, x_left + bw_img - 0.10, y_img + bh_img / 2 - 0.10, "A", color=C_IMG)
    # Clinical network box
    _box(ax, x_left + bw_md / 2, y_md, bw_md, bh_md, C_MD,
         f"{eye_label}\nMLP", fontsize=8.5, radius=0.08)
    _tag(ax, x_left + bw_md - 0.10, y_md + bh_md / 2 - 0.10, "B", color=C_MD)

    # Fusion bridge — with math detail (replaces compact "Bridge" label)
    br_x = x_left + max(bw_img, bw_md) + 1.40
    cy_br = (y_img + y_md) / 2
    _arrow(ax, x_left + bw_img, y_img, br_x - bw_br / 2, cy_br, lw=1.2, style="-|>")
    _arrow(ax, x_left + bw_md,  y_md,  br_x - bw_br / 2, cy_br, lw=1.2, style="-|>")
    _box(ax, br_x, cy_br, bw_br, bh_br, C_BRIDGE,
         "Eye-Level Fusion\n(Hadamard Product)",
         fontsize=8.5, radius=0.10)
    _tag(ax, br_x + bw_br / 2 - 0.12, cy_br + bh_br / 2 - 0.12, "C", color=C_BRIDGE)
    return br_x + bw_br / 2, cy_br


# ── Detail-row primitives ────────────────────────────────────────────────────

def _rect(ax, cx, cy, w, h, color, alpha=0.92, lw=0.0, edge="none"):
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h,
                           facecolor=color, edgecolor=edge, linewidth=lw,
                           alpha=alpha, zorder=3))


def _vector(ax, cx, cy, n_cells, cell_h=0.14, cell_w=0.30, color=C_INPUT,
            alpha=0.92, gap=0.02):
    """Draw a vertical vector of n_cells stacked cells centred at (cx, cy)."""
    total = n_cells * cell_h + (n_cells - 1) * gap
    y_top = cy + total / 2 - cell_h / 2
    for i in range(n_cells):
        y = y_top - i * (cell_h + gap)
        _rect(ax, cx, y, cell_w, cell_h, color, alpha=alpha, lw=0.6, edge="white")


def _setup_detail_panel(ax, xlim=(0, 5), ylim=(0, 5)):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor(C_BG)
    for spine in ax.spines.values():
        spine.set_color("#999")
        spine.set_linewidth(0.6)


def _draw_cnn_detail(ax):
    """CNN schematic: Image -> shrinking feature maps -> pooled vector."""
    _setup_detail_panel(ax)
    _text(ax, 2.5, 4.65, "CNN", fontsize=10.5, color="#222", bold=True)
    _tag(ax, 0.30, 4.70, "A", color=C_IMG)

    # Image
    _rect(ax, 0.55, 2.5, 0.80, 1.10, C_INPUT, alpha=0.85, lw=0.6, edge="white")
    _text(ax, 0.55, 2.5, "Fundus\nImage", fontsize=7.0, color="#333")
    # Feature maps: shrinking spatially, stacked in a row
    map_xs = [1.55, 2.20, 2.75, 3.20]
    map_hs = [1.00, 0.85, 0.70, 0.55]
    for x, h in zip(map_xs, map_hs):
        _rect(ax, x, 2.5, h * 0.60, h, C_IMG, alpha=0.75, lw=0.6, edge="white")
    # Pooled vector
    _vector(ax, 3.90, 2.50, 8, cell_h=0.13, cell_w=0.30, color=C_IMG, alpha=0.92)
    # Arrows
    prev_x = 0.55 + 0.40
    for x, h in zip(map_xs, map_hs):
        _arrow(ax, prev_x, 2.50, x - h * 0.30, 2.50, lw=0.8)
        prev_x = x + h * 0.30
    _arrow(ax, prev_x, 2.50, 3.90 - 0.15, 2.50, lw=0.8)

    _text(ax, 3.90, 1.05, "z_img\n(2048-d)", fontsize=7.5, color="#333")
    _text(ax, 2.50, 0.35,
          "Conv + pool → global avg pool → flatten",
          fontsize=7.8, color="#555")


def _draw_mlp_detail(ax):
    """MLP schematic: input features -> Linear+LN+ReLU -> Linear+ReLU -> embedding.

    Two Linear layers total (matching ClinicalEncoder's block0 + block1). The
    single circle column represents the hidden 128-d activations after the
    first Linear + LayerNorm + ReLU + Dropout; the output vector represents
    the 128-d embedding after the second Linear + ReLU.
    """
    _setup_detail_panel(ax)
    _text(ax, 2.5, 4.65, "MLP", fontsize=10.5, color="#222", bold=True)
    _tag(ax, 0.30, 4.70, "B", color=C_MD)

    labels = ["Age", "IOP", "Gender", "Pachy.", "..."]
    n_in = len(labels)
    y_top_in = 2.50 + (n_in * 0.28) / 2 - 0.14
    for i, lbl in enumerate(labels):
        y = y_top_in - i * 0.28
        _rect(ax, 0.55, y, 0.75, 0.22, C_MD, alpha=0.85, lw=0.6, edge="white")
        _text(ax, 0.55, y, lbl, fontsize=6.8, color="white")

    # Single hidden column (post-Linear + LN + ReLU + Dropout)
    cx_hidden, n_hidden = 2.15, 5
    y_top_h = 2.50 + (n_hidden * 0.32) / 2 - 0.16
    circle_ys = [y_top_h - j * 0.32 for j in range(n_hidden)]
    for y in circle_ys:
        ax.add_patch(plt.Circle((cx_hidden, y), 0.10, facecolor=C_MD,
                                edgecolor="white", lw=0.6, alpha=0.9, zorder=3))

    # LN badge sitting above the hidden column (annotates LN applied at this stage)
    _box(ax, cx_hidden, 3.55, 0.36, 0.28, C_MD, "LN",
         fontsize=6.5, radius=0.05, alpha=0.75)

    # Output vector (post-Linear + ReLU)
    _vector(ax, 3.55, 2.50, 6, cell_h=0.13, cell_w=0.28, color=C_MD, alpha=0.92)

    # Dense connections: input → hidden (Linear 1)
    for i in range(n_in):
        y0 = y_top_in - i * 0.28
        for y1 in circle_ys:
            ax.plot([0.95, cx_hidden - 0.10], [y0, y1],
                    color="#888", lw=0.25, alpha=0.4, zorder=2)
    # Dense connections: hidden → output vector (Linear 2)
    out_cell_ys = [2.50 + (6 * 0.15) / 2 - 0.075 - k * 0.15 for k in range(6)]
    for y0 in circle_ys:
        for y1 in out_cell_ys:
            ax.plot([cx_hidden + 0.10, 3.55 - 0.14], [y0, y1],
                    color="#888", lw=0.25, alpha=0.4, zorder=2)

    _text(ax, 3.55, 1.05, "z_cd\n(128-d)", fontsize=7.5, color="#333")
    _text(ax, 2.50, 0.30,
          "Linear + LN + ReLU + Drop\n→ Linear + ReLU",
          fontsize=7.5, color="#555")


def _draw_eye_fusion_detail(ax):
    """Eye-level fusion: each modality Linear-projected to shared dim, Hadamard."""
    _setup_detail_panel(ax)
    _text(ax, 2.5, 4.65, "Eye-Level Fusion", fontsize=10.5, color="#222", bold=True)
    _tag(ax, 0.30, 4.70, "C", color=C_BRIDGE)

    # ── Top row: image path (z_img → dense lines → W_img circles → LN → ⊙)
    cell_h_img, cell_w_img = 0.08, 0.20
    n_img = 12
    y_img_ctr = 3.45
    _vector(ax, 0.40, y_img_ctr, n_img, cell_h=cell_h_img, cell_w=cell_w_img,
            color=C_IMG)
    _text(ax, 0.40, 4.20, "z_img", fontsize=6.6, color="#333")

    cx_wimg, n_wimg = 1.70, 6
    y_top_wimg = y_img_ctr + (n_wimg * 0.17) / 2 - 0.085
    circle_ys_wimg = [y_top_wimg - j * 0.17 for j in range(n_wimg)]
    for y in circle_ys_wimg:
        ax.add_patch(plt.Circle((cx_wimg, y), 0.075, facecolor=C_IMG,
                                edgecolor="white", lw=0.5, alpha=0.9, zorder=3))
    _text(ax, cx_wimg, 4.20, "W_img 2048→256", fontsize=6.2, color="#555")

    # Dense connections z_img cells → W_img circles
    cell_step_img = cell_h_img + 0.02
    img_cell_ys = [y_img_ctr + (n_img * cell_h_img + (n_img - 1) * 0.02) / 2
                   - cell_h_img / 2 - j * cell_step_img for j in range(n_img)]
    for y_src in img_cell_ys:
        for y_dst in circle_ys_wimg:
            ax.plot([0.40 + cell_w_img / 2, cx_wimg - 0.075],
                    [y_src, y_dst], color="#888", lw=0.2, alpha=0.25, zorder=1)

    _arrow(ax, cx_wimg + 0.10, y_img_ctr, 2.15, y_img_ctr, lw=0.8)
    _box(ax, 2.35, y_img_ctr, 0.30, 0.30, C_IMG, "LN",
         fontsize=6.5, radius=0.05, alpha=0.75)

    # ── Bottom row: clinical path (z_cd → dense lines → W_cd circles → LN → ⊙)
    cell_h_cd, cell_w_cd = 0.15, 0.20
    n_cd = 4
    y_cd_ctr = 1.55
    _vector(ax, 0.40, y_cd_ctr, n_cd, cell_h=cell_h_cd, cell_w=cell_w_cd,
            color=C_MD)
    _text(ax, 0.40, 0.85, "z_cd", fontsize=6.6, color="#333")

    cx_wcd, n_wcd = 1.70, 6
    y_top_wcd = y_cd_ctr + (n_wcd * 0.17) / 2 - 0.085
    circle_ys_wcd = [y_top_wcd - j * 0.17 for j in range(n_wcd)]
    for y in circle_ys_wcd:
        ax.add_patch(plt.Circle((cx_wcd, y), 0.075, facecolor=C_MD,
                                edgecolor="white", lw=0.5, alpha=0.9, zorder=3))
    _text(ax, cx_wcd, 0.85, "W_cd 128→256", fontsize=6.2, color="#555")

    cell_step_cd = cell_h_cd + 0.02
    cd_cell_ys = [y_cd_ctr + (n_cd * cell_h_cd + (n_cd - 1) * 0.02) / 2
                  - cell_h_cd / 2 - j * cell_step_cd for j in range(n_cd)]
    for y_src in cd_cell_ys:
        for y_dst in circle_ys_wcd:
            ax.plot([0.40 + cell_w_cd / 2, cx_wcd - 0.075],
                    [y_src, y_dst], color="#888", lw=0.2, alpha=0.25, zorder=1)

    _arrow(ax, cx_wcd + 0.10, y_cd_ctr, 2.15, y_cd_ctr, lw=0.8)
    _box(ax, 2.35, y_cd_ctr, 0.30, 0.30, C_MD, "LN",
         fontsize=6.5, radius=0.05, alpha=0.75)

    # ── Convergence at ⊙ → z_fused
    _arrow(ax, 2.55, y_img_ctr, 3.10, 2.50, lw=0.9)
    _arrow(ax, 2.55, y_cd_ctr, 3.10, 2.50, lw=0.9)
    _text(ax, 3.35, 2.50, "⊙", fontsize=20, color="#333")
    _arrow(ax, 3.60, 2.50, 3.90, 2.50, lw=1.0)
    _vector(ax, 4.15, 2.50, 6, cell_h=0.12, cell_w=0.22, color=C_BRIDGE)
    _text(ax, 4.15, 1.65, "z_fused\n(256)", fontsize=6.8, color="#333")

    _text(ax, 2.50, 0.30,
          "Project + LN each modality → Hadamard product",
          fontsize=7.8, color="#555")


def _draw_patient_fusion_detail(ax):
    """Concatenation schematic: two eye embeddings → concat → head → logits.

    Draws z_OD and z_OS as vertically-stacked rectangles pushed close together
    with a right-side concat bracket, then two dense layers as columns of
    circles (matching MLP style): FC 512→256 (hb bridge) and FC 256→2 (hb head).
    """
    _setup_detail_panel(ax)
    _text(ax, 2.5, 4.65, "Patient-Level Fusion", fontsize=10.5, color="#222", bold=True)
    _tag(ax, 0.30, 4.70, "D", color=C_HEAD)

    # z_OD and z_OS: vertically stacked rectangles, close together
    cell_h_od = 0.13
    n_od = 6
    y_od_ctr = 3.25
    y_os_ctr = 1.75
    _vector(ax, 0.65, y_od_ctr, n_od, cell_h=cell_h_od, cell_w=0.28, color=C_BRIDGE)
    _text(ax, 0.65, 4.10, "z_OD", fontsize=7.2, color="#333")
    _vector(ax, 0.65, y_os_ctr, n_od, cell_h=cell_h_od, cell_w=0.28, color=C_BRIDGE)
    _text(ax, 0.65, 0.90, "z_OS", fontsize=7.2, color="#333")

    # Concat bracket [ on the LEFT of the two stacks, grouping them
    y_od_top = y_od_ctr + (n_od * cell_h_od + (n_od - 1) * 0.02) / 2
    y_os_bot = y_os_ctr - (n_od * cell_h_od + (n_od - 1) * 0.02) / 2
    x_bracket = 0.42
    ax.plot([x_bracket + 0.10, x_bracket, x_bracket, x_bracket + 0.10],
            [y_od_top, y_od_top, y_os_bot, y_os_bot],
            color="#555", lw=1.3, solid_capstyle="round", zorder=2)
    _text(ax, x_bracket - 0.06, (y_od_top + y_os_bot) / 2, "cat",
          fontsize=7.0, color="white", ha="right", va="center")
    ax.text(x_bracket - 0.02, (y_od_top + y_os_bot) / 2, "cat",
            ha="right", va="center", fontsize=7.0, color="white",
            fontfamily=FONT, fontweight="bold", zorder=6,
            bbox=dict(facecolor="#555", edgecolor="none", pad=2.5,
                      boxstyle="round,pad=0.20"))

    # FC 512 → 256 as a column of circles
    cx1, n1 = 2.10, 6
    y_top1 = 2.50 + (n1 * 0.30) / 2 - 0.15
    circle_ys_1 = [y_top1 - j * 0.30 for j in range(n1)]
    for y in circle_ys_1:
        ax.add_patch(plt.Circle((cx1, y), 0.10, facecolor=C_HEAD,
                                edgecolor="white", lw=0.6, alpha=0.9, zorder=3))
    _text(ax, cx1, 0.85, "FC\n512→256", fontsize=6.8, color="#333")

    # Dense connections from every z_OD and z_OS cell to every FC circle
    cell_step = cell_h_od + 0.02
    od_cell_ys = [y_od_ctr + (n_od * cell_h_od + (n_od - 1) * 0.02) / 2
                  - cell_h_od / 2 - j * cell_step for j in range(n_od)]
    os_cell_ys = [y_os_ctr + (n_od * cell_h_od + (n_od - 1) * 0.02) / 2
                  - cell_h_od / 2 - j * cell_step for j in range(n_od)]
    for y_src in od_cell_ys + os_cell_ys:
        for y_dst in circle_ys_1:
            ax.plot([0.65 + 0.14, cx1 - 0.10], [y_src, y_dst],
                    color="#888", lw=0.22, alpha=0.30, zorder=1)

    # Arrow with ReLU + Drop label on the connection to the head
    _arrow(ax, cx1 + 0.15, 2.50, 3.05, 2.50, lw=0.9)
    _text(ax, 2.65, 2.80, "ReLU + Drop", fontsize=6.5, color="#555")

    # FC 256 → 2 as a shorter column of circles (2 units)
    cx2, n2 = 3.30, 2
    circle_ys_2 = [2.90 - j * 0.80 for j in range(n2)]
    for y in circle_ys_2:
        ax.add_patch(plt.Circle((cx2, y), 0.10, facecolor=C_HEAD,
                                edgecolor="white", lw=0.6, alpha=0.9, zorder=3))
    _text(ax, cx2, 0.85, "FC\n256→2", fontsize=6.8, color="#333")

    # Dense connections between the two FC circle columns
    for y0 in circle_ys_1:
        for y1 in circle_ys_2:
            ax.plot([cx1 + 0.10, cx2 - 0.10], [y0, y1],
                    color="#888", lw=0.25, alpha=0.35, zorder=2)

    # Two class output boxes
    for i, cls in enumerate(("Glauc.", "Normal")):
        y = 2.90 - i * 0.80
        _arrow(ax, cx2 + 0.10, y, 4.05, y, lw=0.7)
        _box(ax, 4.40, y, 0.65, 0.40, C_OUT, cls, fontsize=6.8, radius=0.08)

    _text(ax, 2.50, 0.30,
          "Concat OD + OS → FC 512→256\n→ ReLU + Drop → FC 256→2 → softmax",
          fontsize=7.5, color="#555")


# ── Main figure ──────────────────────────────────────────────────────────────

def _draw_main(ax) -> None:
    W, H = 13.0, 7.8
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor(C_BG)

    ax.set_title("Bilateral Multimodal Fusion Architecture",
                 fontsize=13, fontweight="bold", fontfamily=FONT, pad=10, color="#222")

    x_left = 3.2
    inp_cx = 1.85
    inp_w  = 0.95
    inp_h  = 0.55

    # OD (top)
    od_y_img, od_y_md = 6.10, 4.80
    br_od_x, cy_od = _draw_eye_fusion(ax, x_left, od_y_img, od_y_md, "OD")
    _text(ax, 0.50, (od_y_img + od_y_md) / 2, "OD\n(Right Eye)",
          fontsize=9.5, color="#444", bold=True)
    _box(ax, inp_cx, od_y_img, inp_w, inp_h, C_INPUT, "Fundus\nImage",
         fontsize=8.5, radius=0.08, alpha=0.78, text_color="#333")
    _box(ax, inp_cx, od_y_md,  inp_w, inp_h, C_INPUT, "Clinical\nData",
         fontsize=8.5, radius=0.08, alpha=0.78, text_color="#333")
    _arrow(ax, inp_cx + inp_w / 2, od_y_img, x_left, od_y_img, lw=1.2, style="-|>")
    _arrow(ax, inp_cx + inp_w / 2, od_y_md,  x_left, od_y_md,  lw=1.2, style="-|>")

    # OS (bottom)
    os_y_img, os_y_md = 2.95, 1.65
    br_os_x, cy_os = _draw_eye_fusion(ax, x_left, os_y_img, os_y_md, "OS")
    _text(ax, 0.50, (os_y_img + os_y_md) / 2, "OS\n(Left Eye)",
          fontsize=9.5, color="#444", bold=True)
    _box(ax, inp_cx, os_y_img, inp_w, inp_h, C_INPUT, "Fundus\nImage",
         fontsize=8.5, radius=0.08, alpha=0.78, text_color="#333")
    _box(ax, inp_cx, os_y_md,  inp_w, inp_h, C_INPUT, "Clinical\nData",
         fontsize=8.5, radius=0.08, alpha=0.78, text_color="#333")
    _arrow(ax, inp_cx + inp_w / 2, os_y_img, x_left, os_y_img, lw=1.2, style="-|>")
    _arrow(ax, inp_cx + inp_w / 2, os_y_md,  x_left, os_y_md,  lw=1.2, style="-|>")

    # Side brackets — re-labelled "OD Fusion" / "OS Fusion"
    _bracket(ax, x=br_od_x + 0.05,
             y0=od_y_md - 0.50, y1=od_y_img + 0.50,
             text="OD Fusion", pad=0.22, fontsize=9, badge_color="#555")
    _bracket(ax, x=br_os_x + 0.05,
             y0=os_y_md - 0.50, y1=os_y_img + 0.50,
             text="OS Fusion", pad=0.22, fontsize=9, badge_color="#555")

    # Patient-level head (Fused Head)
    head_y = (cy_od + cy_os) / 2
    head_x = max(br_od_x, br_os_x) + 2.55
    head_w, head_h = 1.95, 1.30
    _arrow(ax, br_od_x + 0.05, cy_od, head_x - head_w / 2, head_y, lw=1.4, style="-|>")
    _arrow(ax, br_os_x + 0.05, cy_os, head_x - head_w / 2, head_y, lw=1.4, style="-|>")
    _box(ax, head_x, head_y, head_w, head_h, C_HEAD,
         "Patient-Level Fusion\n(Concatenation)",
         fontsize=8.5, radius=0.10)
    _tag(ax, head_x + head_w / 2 - 0.13, head_y + head_h / 2 - 0.13, "D", color=C_HEAD)

    # Output nodes
    out_x = head_x + head_w / 2 + 0.55
    _arrow(ax, head_x + head_w / 2, head_y, out_x, head_y, lw=1.5, style="-|>")
    _draw_output(ax, out_x, head_y)


def main() -> None:
    fig = plt.figure(figsize=(18.0, 8.0))
    fig.patch.set_facecolor(C_BG)

    outer = fig.add_gridspec(
        1, 2,
        width_ratios=[12.0, 6.0],
        wspace=0.02,
        left=0.01, right=0.995, top=0.985, bottom=0.02,
    )

    ax_main = fig.add_subplot(outer[0, 0])
    _draw_main(ax_main)

    # 2x2 grid of component details to the right of the main diagram.
    details = outer[0, 1].subgridspec(2, 2, wspace=0.0, hspace=0.0)
    ax_cnn  = fig.add_subplot(details[0, 0])
    ax_mlp  = fig.add_subplot(details[0, 1])
    ax_eye  = fig.add_subplot(details[1, 0])
    ax_pat  = fig.add_subplot(details[1, 1])
    _draw_cnn_detail(ax_cnn)
    _draw_mlp_detail(ax_mlp)
    _draw_eye_fusion_detail(ax_eye)
    _draw_patient_fusion_detail(ax_pat)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
