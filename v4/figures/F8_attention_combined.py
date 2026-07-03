"""F8 combined explainability panel: disc-centred attention + quadrant breakdown.

Composes a single A/B figure from two existing renderings:
    A: ``F8_gradcam/disc_attention_detail.png`` — 2x2 grid of mean Grad-CAM
       heatmaps for {correct, incorrect} x {Normal, Glaucoma} cells, with the
       mean disc boundary annotated as a dashed circle.
    B: ``F8_quadrant_attention.png`` — grouped-bar chart of mean full-image
       Grad-CAM fraction per optic-disc quadrant, by cell.

Both source panels are produced by ``v4.figures.F8_explainability`` and
``v4.figures.F8_quadrant_plot`` respectively; this script just stitches the
two PNGs into a single combined figure with A/B subfigure labels.

Re-run:
    python -m v4.figures.F8_attention_combined
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


SRC_A = Path(__file__).parent / "output" / "F8_gradcam" / "disc_attention_detail.png"
SRC_B = Path(__file__).parent / "output" / "F8_quadrant_attention.png"
OUT   = Path(__file__).parent / "output" / "F8_attention_combined.png"


def render() -> None:
    for p in (SRC_A, SRC_B):
        if not p.exists():
            raise SystemExit(
                f"Source panel missing: {p}\n"
                "Run F8_explainability (for A) and F8_quadrant_plot (for B) first."
            )

    img_a = Image.open(SRC_A)
    img_b = Image.open(SRC_B)

    # Stack vertically: A on top (square), B below (wider).
    fig = plt.figure(figsize=(13.0, 13.6))
    gs = fig.add_gridspec(
        2, 1,
        height_ratios=[img_a.size[1] / img_a.size[0],
                       img_b.size[1] / img_b.size[0] * 13.0 / 13.0],
        hspace=0.06,
    )

    ax_a = fig.add_subplot(gs[0])
    ax_a.imshow(img_a)
    ax_a.axis("off")
    ax_a.text(-0.01, 1.01, "A", transform=ax_a.transAxes,
              ha="left", va="bottom", fontsize=22, fontweight="bold")

    ax_b = fig.add_subplot(gs[1])
    ax_b.imshow(img_b)
    ax_b.axis("off")
    ax_b.text(-0.01, 1.01, "B", transform=ax_b.transAxes,
              ha="left", va="bottom", fontsize=22, fontweight="bold")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}")


if __name__ == "__main__":
    render()
