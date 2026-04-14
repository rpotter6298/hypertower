"""
MD permutation feature importance for Phase 5 — logit_mlp_head checkpointed run.

For each of the 5 fold checkpoints:
  - loads test images + clinical metadata
  - caches image features (no grad)
  - permutes each clinical feature N times and measures AUC drop

Produces (in figures/explainability/):
  md_importance_phase5.png   — aggregated bar chart across 5 folds
  md_importance_phase5.csv   — mean/std per feature

Usage:
    python -m v3.scripts.output_analysis.explainability.permutation_importance_phase5
"""
from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

REPO_ROOT    = Path(__file__).resolve().parents[4]
CKPT_RUN     = REPO_ROOT / "v3" / "results" / "phase5" / "logit_mlp_head_ckpt"
FIGURES_ROOT = REPO_ROOT / "v3" / "figures" / "explainability"
CLINICAL_DIR = REPO_ROOT / "Papila" / "ClinicalData"
IMAGE_DIR    = REPO_ROOT / "Papila" / "FundusImages"

BACKBONE   = "resnet50"
NUM_CLASSES = 2
CD_HIDDEN  = 128
FUSION_DIM = 256
N_PERMUTATIONS = 30
SEED = 0


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(ckpt_path: Path, device: torch.device):
    from v3.classes.models import SingleEyeHT
    sd = torch.load(ckpt_path, map_location="cpu")
    cd_in = sd["cd_tower.block0.0.weight"].shape[1]
    model = SingleEyeHT(
        backbone=BACKBONE, freeze_ratio=0.0, augment=False,
        clinical_data=SimpleNamespace(feature_dim=cd_in),
        num_classes=NUM_CLASSES, cd_hidden_dim=CD_HIDDEN, fusion_dim=FUSION_DIM,
    )
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


# ── Data ──────────────────────────────────────────────────────────────────────

def build_data_bundle():
    from v3.classes.papila_builders import build_papila_data
    # Infer settings from available checkpoint to stay compatible.
    # Once the 10x5 run (--iop-drop-raw --exclude-cols Axial_Length) completes,
    # these will automatically match (feature_dim will drop from 25 → 21).
    ckpt = next(CKPT_RUN.glob("rep*/binary/ensemble/fold*/best_single.pt"), None)
    import torch as _t
    cd_in = _t.load(ckpt, map_location="cpu")["cd_tower.block0.0.weight"].shape[1] if ckpt else 25
    # cd_in=25 → old run (no iop_drop_raw, no excl); cd_in=21 → new run
    drop_raw = cd_in <= 21
    excl = ["Axial_Length"] if cd_in in (21, 23) else []
    return build_papila_data(
        image_dir=str(IMAGE_DIR), clinical_dir=str(CLINICAL_DIR),
        label_col="Diagnosis", cat_cols=["Gender", "Phakic/Pseudophakic"],
        iop_corr_method="ratio", iop_drop_raw=drop_raw, exclude_cols=excl,
    )


def get_eval_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def get_image_path(pid: int, eye: str) -> Path:
    return IMAGE_DIR / f"RET{pid:03d}{eye}.jpg"


# ── Feature index map ─────────────────────────────────────────────────────────

def build_feature_index_map(data) -> dict[str, list[int]]:
    """
    Map feature name → list of dimension indices in the vectorize_row output.
    Layout: [scalars (min-max scaled)] + [cat one-hots] + [scalar missing flags]
    """
    n_scalar = len(data.scalar_cols)
    cat_expanded = sum(len(m) for m in data.cat_maps.values())
    feat_map: dict[str, list[int]] = {}

    # Scalar: value dim + missing flag dim
    for i, col in enumerate(data.scalar_cols):
        feat_map[col] = [i, n_scalar + cat_expanded + i]

    # Categorical: whole one-hot block
    cat_offset = n_scalar
    for col in data.cat_cols:
        n = len(data.cat_maps[col])
        feat_map[col] = list(range(cat_offset, cat_offset + n))
        cat_offset += n

    return feat_map


# ── Per-fold importance ───────────────────────────────────────────────────────

def run_fold(rep_idx: int, fold_idx: int, model, data, device: torch.device,
             n_permutations: int, seed: int) -> dict[str, tuple[float, float]]:
    """
    Returns {feature_name: (mean_auc_drop, std_auc_drop)}.
    """
    from v3.scripts.output_analysis.explainability.fold_patient_ids import (
        get_test_patient_ids,
    )
    transform = get_eval_transform()
    clinical = data.df

    pids = get_test_patient_ids(rep_idx, fold_idx, clinical_dir=CLINICAL_DIR)

    # Cache image features + build meta tensors + labels
    img_feats_list, meta_list, label_list = [], [], []
    model.eval()
    with torch.no_grad():
        for pid in pids:
            for eye in ("OD", "OS"):
                img_path = get_image_path(pid, eye)
                if not img_path.exists():
                    continue
                row = clinical[
                    (clinical["Patient ID"] == pid) & (clinical["eyeID"] == eye)
                ]
                if len(row) == 0:
                    continue
                row = row.iloc[0]
                label = int(row["Diagnosis"])

                from PIL import Image
                pil = Image.open(img_path).convert("RGB")
                img_t = transform(pil).unsqueeze(0).to(device)
                feats = model.img_tower(img_t)          # [1, img_dim]
                meta_vec = torch.tensor(data.vectorize_row(row),
                                        dtype=torch.float32).unsqueeze(0)

                img_feats_list.append(feats.cpu())
                meta_list.append(meta_vec)
                label_list.append(label)

    if not label_list or len(set(label_list)) < 2:
        print(f"  fold{fold_idx}: insufficient data, skipping.")
        return {}

    img_feats = torch.cat(img_feats_list).to(device)   # [N, img_dim]
    meta_all  = torch.cat(meta_list)                    # [N, feat_dim]  on CPU
    y_true    = np.array(label_list)

    # Baseline AUC
    with torch.no_grad():
        md_feats = model.cd_tower(meta_all.to(device))
        out_f, _, _ = model.bridge(img_feats, md_feats)
        probs_base = F.softmax(out_f, dim=1)[:, 1].cpu().numpy()
    baseline_auc = roc_auc_score(y_true, probs_base)
    print(f"  fold{fold_idx}: baseline AUC={baseline_auc:.4f}  N={len(y_true)}")

    feat_map = build_feature_index_map(data)
    rng = np.random.default_rng(seed + fold_idx)
    results: dict[str, tuple[float, float]] = {}

    for feat_name, dims in feat_map.items():
        drops = []
        for _ in range(n_permutations):
            meta_perm = meta_all.clone()
            perm_idx = rng.permutation(len(meta_perm))
            meta_perm[:, dims] = meta_perm[perm_idx][:, dims]
            with torch.no_grad():
                md_p = model.cd_tower(meta_perm.to(device))
                out_p, _, _ = model.bridge(img_feats, md_p)
                probs_p = F.softmax(out_p, dim=1)[:, 1].cpu().numpy()
            try:
                drops.append(baseline_auc - roc_auc_score(y_true, probs_p))
            except Exception:
                pass
        if drops:
            results[feat_name] = (float(np.mean(drops)), float(np.std(drops)))

    return results


# ── Aggregate and plot ────────────────────────────────────────────────────────

def plot_importance(all_results: list[dict], out_png: Path, out_csv: Path) -> None:
    # Aggregate across folds
    all_feats = sorted({f for r in all_results for f in r})
    agg = {}
    for feat in all_feats:
        vals = [r[feat][0] for r in all_results if feat in r]
        if vals:
            agg[feat] = (float(np.mean(vals)), float(np.std(vals)))

    # Sort by mean importance descending
    sorted_feats = sorted(agg, key=lambda f: agg[f][0], reverse=True)
    names = sorted_feats
    imps  = [agg[f][0] for f in names]
    stds  = [agg[f][1] for f in names]
    colors = ["#e05c5c" if v >= 0 else "#5c9ee0" for v in imps]

    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.45 + 1.5)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, imps, xerr=stds, color=colors, ecolor="grey", capsize=3, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean AUC drop  (baseline − permuted)", fontsize=10)
    n_reps  = len(set(r for r in range(len(all_results))))  # placeholder
    ax.set_title(
        f"MD Tower — Permutation Feature Importance\n"
        f"Phase 5 logit_mlp_head_ckpt  ({len(all_results)} folds, "
        f"error bars = std across folds)",
        fontsize=11,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["feature", "mean_importance", "std_importance"])
        w.writeheader()
        for feat in sorted_feats:
            w.writerow({"feature": feat,
                        "mean_importance": agg[feat][0],
                        "std_importance":  agg[feat][1]})
    print(f"Saved: {out_csv}")


# ── Main ──────────────────────────────────────────────────────────────────────

def _discover_checkpoints(ckpt_run: Path) -> list[tuple[int, int, Path]]:
    found = []
    for rep_dir in sorted(ckpt_run.glob("rep*")):
        try:
            rep_idx = int(rep_dir.name.replace("rep", ""))
        except ValueError:
            continue
        for fold_dir in sorted((rep_dir / "binary" / "ensemble").glob("fold[0-9]")):
            ckpt = fold_dir / "best_single.pt"
            if ckpt.exists():
                found.append((rep_idx, int(fold_dir.name.replace("fold", "")), ckpt))
    return found


def main(n_permutations: int = N_PERMUTATIONS, seed: int = SEED):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Building DataBundle ...")
    data = build_data_bundle()
    print(f"  feature_dim={data.feature_dim}")

    checkpoints = _discover_checkpoints(CKPT_RUN)
    print(f"Found {len(checkpoints)} checkpoint(s) across "
          f"{len(set(r for r,f,_ in checkpoints))} rep(s)")

    if not checkpoints:
        print("No checkpoints found.")
        return

    all_results = []
    for rep_idx, fold_idx, ckpt in checkpoints:
        print(f"\n── rep{rep_idx:02d} fold{fold_idx} ──")
        model = build_model(ckpt, device)
        result = run_fold(rep_idx, fold_idx, model, data, device, n_permutations, seed)
        if result:
            all_results.append(result)
        del model

    if not all_results:
        print("No results — nothing to plot.")
        return

    plot_importance(
        all_results,
        FIGURES_ROOT / "md_importance_phase5.png",
        FIGURES_ROOT / "md_importance_phase5.csv",
    )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    ap.add_argument("--seed",           type=int, default=SEED)
    args = ap.parse_args()
    main(n_permutations=args.n_permutations, seed=args.seed)
