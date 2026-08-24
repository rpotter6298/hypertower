"""F4 (main text) - Bilateral clinical, image, and Hadamard L2 fusion.

Patient-level analogue of F3: same two-row / three-column layout, same pruned
clinical panel, but every column is the bilateral (both-eyes) model reading
from the patient-level 'hb' stage. All three columns share the pruned clinical
panel: astigmatism, dioptre_1, dioptre_2, and Phakic/Pseudophakic dropped
(see S8e). The image column has no clinical inputs so is unaffected by the
prune.

Columns (left -> right):
    Clinical only (bilateral)      (cd_solo_bilateral_dropzero, hb)
    Image only (bilateral)         (img_solo_bilateral_refugelike, hb)
    Hadamard L2 fusion (bilateral) (ensemble_refugelike_ckpt_dropzero, hb)

Rows:
    top:    confidence strip - predicted P(Glaucoma) coloured by VF-MD tier
            (severe / moderate / early) with normals in grey. Patients are the
            unit of prediction here (~1 prediction per patient per fold-rep).
    bottom: ROC per severity tier, each tier vs all normals; pooled ROC curve
            with a 95% CI band from patient-level bootstrap; pooled AUC with
            95% CI annotated.

Glaucoma - VF_MD not recorded is dropped from both rows (n = 0 patients at
the patient-worst-eye level).

Re-run:
    python -m v4.figures.F4_bilateral
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from v4.figures.util.loaders import RESULTS_ROOT
from v4.figures import F3_hadamard_focus as F3

OUT = Path(__file__).parent / "output" / "F4_bilateral.png"

SOURCES = [
    ("Clinical only (bilateral)",
     RESULTS_ROOT / "explainability" / "cd_solo_bilateral_dropzero",
     "hb"),
    ("Image only (bilateral)",
     RESULTS_ROOT / "refuge_v2m_baseline" / "img_solo_bilateral_refugelike",
     "hb"),
    ("Hadamard L2 fusion (bilateral)",
     RESULTS_ROOT / "explainability" / "ensemble_refugelike_ckpt_dropzero",
     "hb"),
]


def render() -> None:
    """Reuse every drawing primitive from F3; only source paths and out-file differ."""
    F3.SOURCES = SOURCES
    F3.OUT     = OUT
    F3.render()


if __name__ == "__main__":
    render()
