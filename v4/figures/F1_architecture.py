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
from matplotlib.patches import FancyBboxPatch

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
         f"{eye_label}\nImage Network", fontsize=8.5, radius=0.08)
    # Clinical network box
    _box(ax, x_left + bw_md / 2, y_md, bw_md, bh_md, C_MD,
         f"{eye_label}\nClinical Network", fontsize=8.5, radius=0.08)

    # Fusion bridge — with math detail (replaces compact "Bridge" label)
    br_x = x_left + max(bw_img, bw_md) + 1.40
    cy_br = (y_img + y_md) / 2
    _arrow(ax, x_left + bw_img, y_img, br_x - bw_br / 2, cy_br, lw=1.2, style="-|>")
    _arrow(ax, x_left + bw_md,  y_md,  br_x - bw_br / 2, cy_br, lw=1.2, style="-|>")
    _box(ax, br_x, cy_br, bw_br, bh_br, C_BRIDGE,
         "Fusion Bridge\nFC(img → 256)\nFC(md → 256)\nHadamard product",
         fontsize=8, radius=0.10)
    return br_x + bw_br / 2, cy_br


# ── Main figure ──────────────────────────────────────────────────────────────

def main() -> None:
    W, H = 13.0, 7.8
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ax.set_facecolor(C_BG); fig.patch.set_facecolor(C_BG)

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
         "Patient Head\ncat(z_OD, z_OS)\n→ FC(256) → FC(2)",
         fontsize=8.5, radius=0.10)

    # Output nodes
    out_x = head_x + head_w / 2 + 0.55
    _arrow(ax, head_x + head_w / 2, head_y, out_x, head_y, lw=1.5, style="-|>")
    _draw_output(ax, out_x, head_y)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
