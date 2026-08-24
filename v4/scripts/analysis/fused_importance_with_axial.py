"""Re-run the fused-head permutation importance on the with-Axial_Length model.

Points F8's make_fused_clinical_importance at the sensitivity checkpoints
(refugelike_ensemble_with_axial_length, 10 reps with save_checkpoints=true and
exclude_cols=[]) instead of the standard ensemble_refugelike_ckpt run. This
tests whether Axial_Length lands at ~0 fused-head importance like the other
four dropzero features, closing the circle on the historical decision to
exclude it upstream.

Output is routed to output/S8e_fused_clinical_importance_with_axial.{png,csv}
so it does not overwrite the main S8e figure.

Usage:
    python -m v4.scripts.analysis.fused_importance_with_axial [--n-permutations N]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from v4.figures import F8_explainability as F8
from v4.figures.util.loaders import REPO_ROOT


AXIAL_RUN = (
    REPO_ROOT / "v4" / "results" / "experiments" / "sensitivity"
    / "refugelike_ensemble_with_axial_length" / "rep00" / "binary"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-permutations", type=int, default=30,
                    help="Permutation repeats per feature (default 30).")
    args = ap.parse_args()

    if not AXIAL_RUN.exists():
        raise SystemExit(f"Sensitivity run not found: {AXIAL_RUN}")

    # Point the F8 module at the sensitivity run. rep_base is derived as
    # V4_CKPT_RUN.parent.parent inside make_fused_clinical_importance, so
    # all 10 reps under refugelike_ensemble_with_axial_length are picked up.
    F8.V4_CKPT_RUN = AXIAL_RUN

    # Route the output filenames so we do not overwrite the primary S8e run.
    orig_savefig = F8.plt.Figure.savefig  # noqa: E501  (untouched, just noted)
    orig_out_dir = F8.OUT_DIR
    tag = "with_axial"

    # Monkey-patch pandas.DataFrame.to_csv and Figure.savefig only for calls
    # that name the S8e file. Simplest: swap out OUT_DIR and rename the two
    # target filenames post-hoc via a wrapper.
    src_png = orig_out_dir / "S8e_fused_clinical_importance.png"
    src_csv = orig_out_dir / "S8e_fused_clinical_importance.csv"
    dst_png = orig_out_dir / f"S8e_fused_clinical_importance_{tag}.png"
    dst_csv = orig_out_dir / f"S8e_fused_clinical_importance_{tag}.csv"

    # Move originals aside if present, restore after; simpler than hooking I/O.
    def _stash(p: Path):
        return p.rename(p.with_suffix(p.suffix + ".bak")) if p.exists() else None

    stashed = [_stash(src_png), _stash(src_csv)]
    try:
        F8.make_fused_clinical_importance(n_permutations=args.n_permutations)
        if src_png.exists(): src_png.rename(dst_png)
        if src_csv.exists(): src_csv.rename(dst_csv)
        print(f"\nRenamed outputs -> {dst_png.name}, {dst_csv.name}")
    finally:
        for p in stashed:
            if p is not None:
                p.rename(p.with_suffix(""))


if __name__ == "__main__":
    main()
