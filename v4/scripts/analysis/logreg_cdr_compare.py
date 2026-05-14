"""logreg_cdr_compare — compare LogReg AUCs across GT / base-UNet / fine-tuned-UNet CDR sources.

Lets us answer: how much of the apparent +0.11 AUC from the LogReg(clinical+CDR-GT)
result is from the GT source quality vs the CDR features themselves?  Tritower
uses per-fold-fine-tuned UNet — this script reproduces that pipeline as a 5-fold
LogReg baseline so the numbers are directly comparable to v4 cd_solo_geom and
tritower runs.

5-fold StratifiedGroupKFold (patient-grouped), LogReg with StandardScaler.

Run:
    python -m v4.scripts.analysis.logreg_cdr_compare
"""
from __future__ import annotations

import sys, time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from v4.classes.profiles.v4papila import build_data
from v4.classes.profiles.fundus_images import (
    build_geometry_loader,
    _PapilaUNetMaskPipeline,
    compute_geometry_features,
)

CONTOUR_DIR  = REPO_ROOT / "Papila/ExpertsSegmentations/Contours"
UNET_WEIGHTS = REPO_ROOT / "models/v2/refuge/segmentation/per_image/best.pt"


def main():
    data = build_data({
        "image_dir":       "Papila/FundusImages",
        "clinical_dir":    "Papila/ClinicalData",
        "iop_corr_method": "ratio",
        "iop_drop_raw":    True,
        "exclude_cols":    ["Axial_Length"],
    })

    df = data.df[data.df[data.label_col].isin([0, 1])].reset_index(drop=True)
    samples = data.collect_samples(df)
    y       = df[data.label_col].astype(int).values
    groups  = df["Patient ID"].astype(int).values
    view    = data.matrix
    X_clin  = np.array([view.vectorize_entity(int(r["Patient ID"]), str(r["eyeID"]))
                        for _, r in df.iterrows()])
    print(f"n eyes={len(df)}  n patients={len(set(groups))}  "
          f"label balance={np.bincount(y).tolist()}")

    # ─── GT CDR ───────────────────────────────────────────────────────────────
    print("\n[1/3] GT CDR (rasterise expert contours; split-independent)")
    gt = build_geometry_loader("gt", contour_dir=str(CONTOUR_DIR))
    gt.precompute(samples)
    gt_vecs   = gt.all_vectors()
    X_cdr_gt  = np.array([gt_vecs[(int(r["Patient ID"]), str(r["eyeID"]))]
                          for _, r in df.iterrows()])

    # ─── Base UNet CDR (no fine-tune; same masks every fold) ──────────────────
    print("\n[2/3] base UNet CDR (REFUGE weights, no fine-tune)")
    t0 = time.time()
    unet_base = build_geometry_loader("unet",
        weights_path=str(UNET_WEIGHTS),
        contour_dir=str(CONTOUR_DIR),
        finetune_epochs=0,
    )
    unet_base.precompute(samples)
    base_vecs = unet_base.all_vectors()
    X_cdr_unet_base = np.array([base_vecs[(int(r["Patient ID"]), str(r["eyeID"]))]
                                for _, r in df.iterrows()])
    print(f"  base inference done in {time.time()-t0:.1f}s")

    # ─── Build the 5 folds (used for both per-fold UNet ft AND for LogReg CV) ─
    sgkf   = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(sgkf.split(np.arange(len(df)), y, groups))

    # ─── Per-fold fine-tuned UNet CDR ─────────────────────────────────────────
    print("\n[3/3] per-fold-fine-tuned UNet CDR (matches tritower / cd_solo_geom)")
    X_cdr_unet_ft = np.zeros((len(df), 5), dtype=np.float32)
    pipe = _PapilaUNetMaskPipeline(
        str(UNET_WEIGHTS),
        contour_dir=str(CONTOUR_DIR),
        finetune_epochs=10,
        finetune_lr=1e-5,
        finetune_batch_size=4,
    )
    for fold, (tr, te) in enumerate(splits):
        t1 = time.time()
        train_samples = [samples[i] for i in tr]
        test_samples  = [samples[i] for i in te]
        pipe.reset_weights()
        pipe.finetune(train_samples)
        masks = pipe.predict(test_samples)
        for j, (pid, eye, _) in zip(te, test_samples):
            disc, cup = masks[(pid, eye)]
            X_cdr_unet_ft[j] = compute_geometry_features(disc, cup)
        print(f"  fold {fold+1}/5 done ({time.time()-t1:.0f}s)")

    # ─── LogReg CV ────────────────────────────────────────────────────────────
    def cv(X, label):
        aucs = []
        for tr, te in splits:
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, C=1.0))
            clf.fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
        print(f"  {label:32s}: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    print("\n=== StratifiedGroupKFold LogReg AUCs ===")
    cv(X_clin,                                                "clinical only")
    cv(X_cdr_gt,                                              "CDR-GT only")
    cv(X_cdr_unet_base,                                       "CDR-UNet (base) only")
    cv(X_cdr_unet_ft,                                         "CDR-UNet (per-fold ft) only")
    print()
    cv(np.concatenate([X_clin, X_cdr_gt],         axis=1),    "clinical + CDR-GT")
    cv(np.concatenate([X_clin, X_cdr_unet_base],  axis=1),    "clinical + CDR-UNet (base)")
    cv(np.concatenate([X_clin, X_cdr_unet_ft],    axis=1),    "clinical + CDR-UNet (ft)")


if __name__ == "__main__":
    main()
