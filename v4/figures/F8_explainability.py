"""F8 - Explainability figures from the R50 checkpointed v4 run.

Sources predictions and Grad-CAM panels exclusively from the
``experiments/explainability/ensemble_refugelike_ckpt`` run (img+cd ensemble
with the refugelike R50 backbone, save_checkpoints=true). rep00 (seed=1234)
is the single rep used for the figure; matches the headline configuration in
section 3.

GradCAM machinery lives in ``v4.classes.accessory.explainability``; PAPILA
specific knowledge (disc contour rasterisation, OS→OD orientation flip) is
attached to ``ImageDataView`` in ``v4.classes.profiles.v4papila`` and consumed
here via getattr so future non-PAPILA profiles can opt in without changing
this file.

Usage
    python -m v4.figures.F8_explainability --only-fusion
    python -m v4.figures.F8_explainability --only-gradcam --run-gradcam
    python -m v4.figures.F8_explainability --run-gradcam        # both
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

from v4.figures.util.loaders import REPO_ROOT


OUT_DIR = Path(__file__).parent / "output"
GRADCAM_DIR = OUT_DIR / "F8_gradcam"
V4_CKPT_RUN = REPO_ROOT / "v4" / "results" / "experiments" / "explainability" / "ensemble_refugelike_ckpt" / "rep00" / "binary"

LABEL_NAMES = {0: "Normal", 1: "Glaucoma"}
EVENT_ORDER = [
    "full_correction", "img_assist", "md_assist",
    "full_error", "img_drag", "md_drag",
    "concordant_correct", "concordant_wrong",
]
EVENT_LABELS = {
    "full_correction": "Both wrong -> fused right",
    "img_assist": "Image right, clinical wrong",
    "md_assist": "Clinical right, image wrong",
    "full_error": "Both right -> fused wrong",
    "img_drag": "Clinical right, image wrong -> fused wrong",
    "md_drag": "Image right, clinical wrong -> fused wrong",
    "concordant_correct": "All correct",
    "concordant_wrong": "All wrong",
}
EVENT_COLORS = {
    "full_correction": "#2f8f5b",
    "img_assist": "#74b66b",
    "md_assist": "#b7c85a",
    "full_error": "#b23a48",
    "img_drag": "#df7f5f",
    "md_drag": "#d2a24c",
    "concordant_correct": "#7da5c9",
    "concordant_wrong": "#9b8fc2",
    "net_positive": "#1f7a4d",
    "net_negative": "#9f2735",
}


def _prediction_frame_from_probs(
    *,
    y_true: np.ndarray,
    probs_fused: np.ndarray,
    probs_img: np.ndarray,
    probs_md: np.ndarray,
    rep: str,
    fold: str,
    source: str,
) -> pd.DataFrame:
    pred_fused = probs_fused.argmax(axis=1)
    pred_img = probs_img.argmax(axis=1)
    pred_md = probs_md.argmax(axis=1)
    df = pd.DataFrame(
        {
            "idx": np.arange(len(y_true)),
            "y_true": y_true.astype(int),
            "pred_fused": pred_fused.astype(int),
            "prob_fused_c0": probs_fused[:, 0],
            "prob_fused_c1": probs_fused[:, 1],
            "pred_img": pred_img.astype(int),
            "prob_img_c0": probs_img[:, 0],
            "prob_img_c1": probs_img[:, 1],
            "pred_md": pred_md.astype(int),
            "prob_md_c0": probs_md[:, 0],
            "prob_md_c1": probs_md[:, 1],
            "rep": rep,
            "fold": fold,
            "source": source,
        }
    )
    for head in ("fused", "img", "md"):
        pred = df[f"pred_{head}"].to_numpy()
        y = df["y_true"].to_numpy()
        df[f"tp_{head}"] = ((pred == 1) & (y == 1)).astype(int)
        df[f"fp_{head}"] = ((pred == 1) & (y == 0)).astype(int)
        df[f"tn_{head}"] = ((pred == 0) & (y == 0)).astype(int)
        df[f"fn_{head}"] = ((pred == 0) & (y == 1)).astype(int)
    return df


def _load_ckpt_predictions(split: str = "test") -> pd.DataFrame:
    """Reconstruct v4 component predictions from checkpointed fold modules.

    predictions.h5 only stores the final hb head, so the image/clinical
    component predictions needed for F8a are rebuilt from tower/head checkpoints.
    """
    if split not in {"test", "val", "both"}:
        raise ValueError(f"split must be 'test', 'val', or 'both', got {split!r}")

    import importlib
    import json
    import torch
    import torch.nn.functional as F
    from v4.classes.split_manager import SplitManager
    from v4.classes.v4_hypertower import load_data, build_towers, _make_loader
    import v4.classes.v4_hypertower as orch
    from v4.classes.heads.classifier import ClassificationHead

    summary = json.loads((V4_CKPT_RUN / "summary.json").read_text())
    cfg = summary["config"]
    orch.cfg_ref = cfg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(cfg)
    label_filter = cfg.get("label_filter", None)
    df_mode = data.df.copy()
    if label_filter is not None:
        df_mode = df_mode[df_mode[data.label_col].isin(label_filter)].reset_index(drop=True)
    identity_cols = getattr(data, "identity_cols", [])
    identity_level = cfg.get("split_identity_level", 1)
    group_col = identity_cols[identity_level - 1] if identity_level and identity_cols else None
    splits = SplitManager(group_col=group_col).build_plans(
        df_mode,
        label_col=data.label_col,
        n_splits=cfg.get("folds", 5),
        seed=cfg.get("fold_seed", 100),
    )

    stage_by_name = {s["name"]: s for s in cfg["stages"]}
    nt_cfg = stage_by_name["nt"]
    hb_cfg = stage_by_name["hb"]
    rows: list[pd.DataFrame] = []

    for fold_idx, split_obj in enumerate(splits):
        ckpt_dir = V4_CKPT_RUN / "checkpoints" / f"fold{fold_idx}"
        if not ckpt_dir.exists():
            continue

        towers = build_towers(cfg["towers"], data)
        for name, tower in towers.items():
            tower.load_state_dict(torch.load(ckpt_dir / f"tower_{name}.pt", map_location="cpu"))
            tower.to(device).eval()

        nt_mod = importlib.import_module(nt_cfg["module"])
        nt = getattr(nt_mod, nt_cfg["class"])(
            [towers[n].out_dim for n in nt_cfg["inputs"]],
            **nt_cfg.get("args", {}),
        ).to(device)
        nt.load_state_dict(torch.load(ckpt_dir / "stage_nt.pt", map_location="cpu"))
        nt.eval()

        hb_mod = importlib.import_module(hb_cfg["module"])
        hb = getattr(hb_mod, hb_cfg["class"])(
            {"a": nt.out_dim, "b": nt.out_dim},
            **hb_cfg.get("args", {}),
        ).to(device)
        hb.load_state_dict(torch.load(ckpt_dir / "stage_hb.pt", map_location="cpu"))
        hb.eval()

        img_aux = ClassificationHead(towers["img"].out_dim, cfg["num_classes"]).to(device)
        cd_aux = ClassificationHead(towers["cd"].out_dim, cfg["num_classes"]).to(device)
        hb_head = ClassificationHead(hb.out_dim, cfg["num_classes"], dropout=0.3).to(device)
        img_aux.load_state_dict(torch.load(ckpt_dir / "stage_img_aux.pt", map_location="cpu"))
        cd_aux.load_state_dict(torch.load(ckpt_dir / "stage_cd_aux.pt", map_location="cpu"))
        hb_head.load_state_dict(torch.load(ckpt_dir / "stage_hb_head.pt", map_location="cpu"))
        img_aux.eval(); cd_aux.eval(); hb_head.eval()

        split_frames = []
        if split in {"val", "both"}:
            split_frames.append(("val", split_obj.val))
        if split in {"test", "both"} and split_obj.test is not None:
            split_frames.append(("test", split_obj.test))

        for split_name, split_df in split_frames:
            shell = data.build_shells(split_df, level="patient", label_filter=label_filter)
            loader = _make_loader(
                shell,
                towers,
                batch_size=cfg["training"].get("batch_size", 8),
                shuffle=False,
            )
            y_all, pf_all, pi_all, pc_all, ids_all = [], [], [], [], []
            with torch.no_grad():
                for batch in loader:
                    y = batch["label"].detach().cpu().numpy()
                    img_a = towers["img"](batch["img"]["a"].to(device))
                    img_b = towers["img"](batch["img"]["b"].to(device))
                    cd_a = towers["cd"](batch["cd"]["a"].to(device))
                    cd_b = towers["cd"](batch["cd"]["b"].to(device))

                    z_a = nt([img_a, cd_a])
                    z_b = nt([img_b, cd_b])
                    z_hb = hb({"a": z_a, "b": z_b})

                    p_fused = F.softmax(hb_head(z_hb), dim=1).cpu().numpy()
                    p_img = 0.5 * (
                        F.softmax(img_aux(img_a), dim=1).cpu().numpy()
                        + F.softmax(img_aux(img_b), dim=1).cpu().numpy()
                    )
                    p_cd = 0.5 * (
                        F.softmax(cd_aux(cd_a), dim=1).cpu().numpy()
                        + F.softmax(cd_aux(cd_b), dim=1).cpu().numpy()
                    )
                    y_all.append(y)
                    pf_all.append(p_fused)
                    pi_all.append(p_img)
                    pc_all.append(p_cd)
                    ids_all.extend(batch.get("entity_id", []))

            if y_all:
                part = _prediction_frame_from_probs(
                    y_true=np.concatenate(y_all),
                    probs_fused=np.concatenate(pf_all, axis=0),
                    probs_img=np.concatenate(pi_all, axis=0),
                    probs_md=np.concatenate(pc_all, axis=0),
                    rep="v4",
                    fold=f"fold{fold_idx}",
                    source=split_name,
                )
                part["entity_id"] = [str(e) for e in ids_all]
                rows.append(part)

        del towers, nt, hb, img_aux, cd_aux, hb_head
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not rows:
        raise FileNotFoundError(f"No reconstructed predictions under {V4_CKPT_RUN}")
    return pd.concat(rows, ignore_index=True)


def _classify_events(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    y = df["y_true"].to_numpy()
    fused_ok = df["pred_fused"].to_numpy() == y
    img_ok = df["pred_img"].to_numpy() == y
    md_ok = df["pred_md"].to_numpy() == y

    def classify(fo: bool, io: bool, mo: bool) -> str:
        if fo and io and mo:
            return "concordant_correct"
        if not fo and not io and not mo:
            return "concordant_wrong"
        if fo and not io and not mo:
            return "full_correction"
        if fo and io and not mo:
            return "img_assist"
        if fo and not io and mo:
            return "md_assist"
        if not fo and io and mo:
            return "full_error"
        if not fo and not io and mo:
            return "img_drag"
        if not fo and io and not mo:
            return "md_drag"
        return "other"

    df["event_type"] = [classify(fo, io, mo) for fo, io, mo in zip(fused_ok, img_ok, md_ok)]
    df["fused_ok"] = fused_ok
    df["img_ok"] = img_ok
    df["md_ok"] = md_ok
    df["tower_state"] = np.select(
        [
            (~img_ok) & (~md_ok),
            img_ok & (~md_ok),
            (~img_ok) & md_ok,
            img_ok & md_ok,
        ],
        [
            "Both networks wrong",
            "Image only correct",
            "Clinical only correct",
            "Both networks correct",
        ],
        default="Other",
    )
    df["tower_mean_c1"] = 0.5 * (df["prob_img_c1"] + df["prob_md_c1"])
    df["fusion_delta_c1"] = df["prob_fused_c1"] - df["tower_mean_c1"]
    df["fusion_margin"] = np.where(
        df["y_true"] == 1,
        df["prob_fused_c1"] - df["tower_mean_c1"],
        (1.0 - df["prob_fused_c1"]) - (1.0 - df["tower_mean_c1"]),
    )
    df["tower_gap_abs"] = (df["prob_img_c1"] - df["prob_md_c1"]).abs()
    return df


def make_fusion_event_panel(split: str = "test") -> None:
    df = _classify_events(_load_ckpt_predictions(split=split))
    auc = roc_auc_score(df["y_true"], df["prob_fused_c1"])
    fold_groups = list(df.groupby(["rep", "fold"]))
    counts = df["event_type"].value_counts().reindex(EVENT_ORDER, fill_value=0)
    positive_keys = EVENT_ORDER[:3]
    negative_keys = EVENT_ORDER[3:6]
    total_positive = int(counts[positive_keys].sum())
    total_negative = int(counts[negative_keys].sum())

    fig = plt.figure(figsize=(15, 8.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.45], width_ratios=[1.05, 1.0],
                          wspace=0.32, hspace=0.38)

    ax = fig.add_subplot(gs[0, 0])
    display_counts = counts.to_dict()
    display_counts["net_positive"] = total_positive
    display_counts["net_negative"] = total_negative
    display_labels = {
        **EVENT_LABELS,
        "net_positive": "Net positive",
        "net_negative": "Net negative",
    }
    display_order = [
        "full_correction", "img_assist", "md_assist",
        "full_error", "img_drag", "md_drag",
        "net_positive", "net_negative",
    ]
    bars = [k for k in display_order if display_counts.get(k, 0) > 0]
    y = np.arange(len(bars))
    ax.barh(y, [display_counts[k] for k in bars],
            color=[EVENT_COLORS[k] for k in bars], height=0.68)
    ax.set_yticks(y)
    ax.set_yticklabels([display_labels[k] for k in bars], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title("Fusion event taxonomy", fontsize=10, fontweight="bold")
    max_count = max(display_counts[k] for k in bars) if bars else 0
    ax.set_xlim(0, max_count * 1.10 + 1)
    for yi, k in enumerate(bars):
        ax.text(display_counts[k] + 0.8, yi, str(int(display_counts[k])),
                va="center", fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    per_fold = pd.DataFrame(
        {
            "fold": [str(f) for (_, f), _ in fold_groups],
            "positive": [sum((g["event_type"] == k).sum() for k in positive_keys) for _, g in fold_groups],
            "negative": [sum((g["event_type"] == k).sum() for k in negative_keys) for _, g in fold_groups],
        }
    )
    x = np.arange(len(per_fold))
    ax.bar(x, per_fold["positive"], color="#2f8f5b", width=0.72, label="positive correction")
    ax.bar(x, -per_fold["negative"], color="#b23a48", width=0.72, label="negative correction")
    ax.plot(x, per_fold["positive"] - per_fold["negative"], color="#222", lw=1.2,
            marker="o", ms=3, label="net")
    ax.axhline(0, color="#222", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(per_fold["fold"], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Events per fold")
    ax.set_title("Per-fold correction balance", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", frameon=False)

    ax = fig.add_subplot(gs[1, :])
    disagree = df[df["event_type"].isin(positive_keys + negative_keys)].copy()
    concordant = df[df["event_type"].isin(["concordant_correct", "concordant_wrong"])].copy()
    n_bins = 28
    agreement_grid = np.zeros((n_bins, n_bins), dtype=float)
    pos_grid = np.zeros((n_bins, n_bins), dtype=float)
    neg_grid = np.zeros((n_bins, n_bins), dtype=float)
    for _, row in concordant.iterrows():
        xi = min(n_bins - 1, max(0, int(row["prob_img_c1"] * n_bins)))
        yi = min(n_bins - 1, max(0, int(row["prob_md_c1"] * n_bins)))
        agreement_grid[yi, xi] += 1
    for _, row in disagree.iterrows():
        xi = min(n_bins - 1, max(0, int(row["prob_img_c1"] * n_bins)))
        yi = min(n_bins - 1, max(0, int(row["prob_md_c1"] * n_bins)))
        if row["event_type"] in positive_keys:
            pos_grid[yi, xi] += 1
        else:
            neg_grid[yi, xi] += 1

    grey_rgba = np.zeros((n_bins, n_bins, 4), dtype=float)
    grey_rgba[:, :, :3] = 0.52
    if agreement_grid.max() > 0:
        grey_rgba[:, :, 3] = 0.06 + 0.22 * np.sqrt(agreement_grid / agreement_grid.max())
    grey_rgba[agreement_grid == 0, 3] = 0.0
    ax.imshow(grey_rgba, extent=[0, 1, 0, 1], origin="lower", aspect="auto",
              interpolation="nearest")

    support = pos_grid + neg_grid
    dominance = np.divide(pos_grid - neg_grid, support,
                          out=np.zeros_like(pos_grid), where=support > 0)
    cmap = plt.get_cmap("RdYlGn")
    rgba = cmap((dominance + 1.0) / 2.0)
    if support.max() > 0:
        rgba[:, :, 3] = 0.08 + 0.46 * np.sqrt(support / support.max())
    rgba[support == 0, 3] = 0.0
    ax.imshow(rgba, extent=[0, 1, 0, 1], origin="lower", aspect="auto",
              interpolation="nearest")

    point_handles = []
    for correct, label, color in [
        (True, "fused correct", "#2f8f5b"),
        (False, "fused wrong", "#b23a48"),
    ]:
        sub = df[df["fused_ok"] == correct]
        h = ax.scatter(
            sub["prob_img_c1"], sub["prob_md_c1"], s=18, alpha=0.45,
            color=color, edgecolors="white", linewidths=0.18,
            label=label,
        )
        point_handles.append(h)
    ax.axvline(0.5, color="#222", lw=0.9, ls="--", alpha=0.75)
    ax.axhline(0.5, color="#222", lw=0.9, ls="--", alpha=0.75)
    ax.plot([0, 1], [0, 1], color="#222", lw=0.7, ls=":", alpha=0.55)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Image network P(glaucoma)")
    ax.set_ylabel("Clinical network P(glaucoma)")
    ax.set_title(
        "Network confidence space - shaded by dominant empirical fusion outcome "
        "(grey=agreement, green=positive correction, red=negative correction)",
        fontsize=10, fontweight="bold",
    )
    ax.text(0.02, 0.96, "clinical says glaucoma", transform=ax.transAxes,
            ha="left", va="top", fontsize=8, color="#333")
    ax.text(0.98, 0.04, "image says glaucoma", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="#333")
    shade_handles = [
        mpatches.Patch(facecolor="#858585", alpha=0.28, label="agreement density"),
        mpatches.Patch(facecolor="#2f8f5b", alpha=0.34, label="positive correction shade"),
        mpatches.Patch(facecolor="#b23a48", alpha=0.34, label="negative correction shade"),
    ]
    ax.legend(handles=point_handles + shade_handles, ncol=5, fontsize=7,
              loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)

    out = OUT_DIR / "S8a_comparison_panel.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def make_clinical_importance(n_permutations: int = 30, seed: int = 0) -> None:
    """S8e clinical permutation importance via the cd-tower → cd_aux head.

    Isolates the clinical-only prediction path at the R50 ckpt run, then
    column-shuffles the encoded clinical vector to measure per-feature AUC
    drop. Per-original-column grouping comes from
    ``ClinicalDataView.feature_groups`` (one-hot encoded dims for a
    categorical column shuffle together).

    Aggregates per-fold mean drops, then averages across folds.
    """
    import json
    import torch
    import torch.nn.functional as TF_
    from tqdm import tqdm
    from v4.classes.accessory.explainability import permutation_importance
    from v4.classes.split_manager import SplitManager
    from v4.classes.v4_hypertower import load_data, _make_loader
    import v4.classes.v4_hypertower as orch

    summary_path = V4_CKPT_RUN / "summary.json"
    if not summary_path.exists():
        print(f"[F8] make_clinical_importance: no summary.json under {V4_CKPT_RUN}")
        return

    cfg = json.loads(summary_path.read_text())["config"]
    orch.cfg_ref = cfg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_data(cfg)
    clinical_view = data.matrix
    feature_names = getattr(clinical_view, "feature_names", None)
    feature_groups = getattr(clinical_view, "feature_groups", None)
    if feature_names is None or feature_groups is None:
        print(
            "[F8] make_clinical_importance: profile lacks feature_names / "
            "feature_groups; skipping S8e."
        )
        return

    label_filter = cfg.get("label_filter", None)
    df_mode = data.df.copy()
    if label_filter is not None:
        df_mode = df_mode[df_mode[data.label_col].isin(label_filter)].reset_index(drop=True)
    identity_cols = getattr(data, "identity_cols", [])
    identity_level = cfg.get("split_identity_level", 1)
    group_col = identity_cols[identity_level - 1] if identity_level and identity_cols else None
    splits = SplitManager(group_col=group_col).build_plans(
        df_mode,
        label_col=data.label_col,
        n_splits=cfg.get("folds", 5),
        seed=cfg.get("fold_seed", 100),
    )

    all_results = []
    for fold_idx in tqdm(range(cfg.get("folds", 5)), desc="Importance folds", unit="fold"):
        ckpt_dir = V4_CKPT_RUN / "checkpoints" / f"fold{fold_idx}"
        if not ckpt_dir.exists():
            continue
        towers, nt, hb, img_aux, cd_aux, hb_head = _build_v4_fold_modules(
            fold_idx, cfg, data, device,
        )

        split_obj = splits[fold_idx]
        if split_obj.test is None:
            del towers, nt, hb, img_aux, cd_aux, hb_head
            continue
        shell = data.build_shells(split_obj.test, level="patient", label_filter=label_filter)
        loader = _make_loader(
            shell,
            towers,
            batch_size=cfg["training"].get("batch_size", 8),
            shuffle=False,
        )

        X_list, y_list = [], []
        with torch.no_grad():
            for batch in loader:
                X_list.append(batch["cd"]["a"].cpu().numpy())   # OD-side clinical
                y_list.append(batch["label"].cpu().numpy())
        if not X_list:
            del towers, nt, hb, img_aux, cd_aux, hb_head
            continue
        X = np.concatenate(X_list).astype(np.float32)
        y = np.concatenate(y_list).astype(int)
        if len(np.unique(y)) < 2:
            del towers, nt, hb, img_aux, cd_aux, hb_head
            continue

        def score_fn(X_in: np.ndarray) -> float:
            x_t = torch.from_numpy(X_in).to(device)
            with torch.no_grad():
                z = towers["cd"](x_t)
                logits = cd_aux(z)
                probs = TF_.softmax(logits, dim=1).cpu().numpy()
            try:
                return roc_auc_score(y, probs[:, 1])
            except Exception:
                return float("nan")

        result = permutation_importance(
            score_fn=score_fn,
            X=X,
            groups=feature_groups,
            n_permutations=n_permutations,
            seed=seed + fold_idx,
            feature_names=feature_names,
        )
        all_results.append(result)

        del towers, nt, hb, img_aux, cd_aux, hb_head
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_results:
        print("[F8] make_clinical_importance: no folds with usable data; skipping S8e.")
        return

    fold_drops = np.stack([r["mean_drop"] for r in all_results])
    mean_across_folds = fold_drops.mean(axis=0)
    std_across_folds = fold_drops.std(axis=0)
    baseline_mean = float(np.mean([r["baseline"] for r in all_results]))
    names = all_results[0]["feature_names"]

    order = np.argsort(mean_across_folds)[::-1]
    sorted_names = [names[i] for i in order]
    sorted_means = mean_across_folds[order]
    sorted_stds = std_across_folds[order]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, max(3.5, 0.32 * len(sorted_names))))
    y_pos = np.arange(len(sorted_names))
    ax.barh(
        y_pos, sorted_means, xerr=sorted_stds,
        color="#3B6FB5", edgecolor="black", height=0.7,
        error_kw=dict(ecolor="#444", lw=0.8, capsize=2),
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean AUC drop on shuffling (averaged across folds)", fontsize=10)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_title(
        f"S8e — Clinical permutation importance (cd-only head, R50 ckpt run)\n"
        f"baseline AUC = {baseline_mean:.3f}; n_permutations = {n_permutations}",
        fontsize=10, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    fig.tight_layout()

    out_png = OUT_DIR / "S8e_clinical_importance.png"
    out_csv = OUT_DIR / "S8e_clinical_importance.csv"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame({
        "feature": sorted_names,
        "mean_drop": sorted_means,
        "std_drop": sorted_stds,
        "baseline_auc_mean": baseline_mean,
    }).to_csv(out_csv, index=False)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_csv}")


def _build_v4_fold_modules(fold_idx: int, cfg: dict, data, device):
    import importlib
    import torch
    from v4.classes.v4_hypertower import build_towers
    from v4.classes.heads.classifier import ClassificationHead

    stage_by_name = {s["name"]: s for s in cfg["stages"]}
    nt_cfg = stage_by_name["nt"]
    hb_cfg = stage_by_name["hb"]
    ckpt_dir = V4_CKPT_RUN / "checkpoints" / f"fold{fold_idx}"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No v4 checkpoint directory: {ckpt_dir}")

    towers = build_towers(cfg["towers"], data)
    for name, tower in towers.items():
        tower.load_state_dict(torch.load(ckpt_dir / f"tower_{name}.pt", map_location="cpu"))
        tower.to(device).eval()

    nt_mod = importlib.import_module(nt_cfg["module"])
    nt = getattr(nt_mod, nt_cfg["class"])(
        [towers[n].out_dim for n in nt_cfg["inputs"]],
        **nt_cfg.get("args", {}),
    ).to(device)
    nt.load_state_dict(torch.load(ckpt_dir / "stage_nt.pt", map_location="cpu"))
    nt.eval()

    hb_mod = importlib.import_module(hb_cfg["module"])
    hb = getattr(hb_mod, hb_cfg["class"])(
        {"a": nt.out_dim, "b": nt.out_dim},
        **hb_cfg.get("args", {}),
    ).to(device)
    hb.load_state_dict(torch.load(ckpt_dir / "stage_hb.pt", map_location="cpu"))
    hb.eval()

    img_aux = ClassificationHead(towers["img"].out_dim, cfg["num_classes"]).to(device)
    cd_aux = ClassificationHead(towers["cd"].out_dim, cfg["num_classes"]).to(device)
    hb_head = ClassificationHead(hb.out_dim, cfg["num_classes"], dropout=0.3).to(device)
    img_aux.load_state_dict(torch.load(ckpt_dir / "stage_img_aux.pt", map_location="cpu"))
    cd_aux.load_state_dict(torch.load(ckpt_dir / "stage_cd_aux.pt", map_location="cpu"))
    hb_head.load_state_dict(torch.load(ckpt_dir / "stage_hb_head.pt", map_location="cpu"))
    img_aux.eval(); cd_aux.eval(); hb_head.eval()
    return towers, nt, hb, img_aux, cd_aux, hb_head


def _orient_eye_array(arr: np.ndarray, eye: str) -> np.ndarray:
    """Mirror OS so temporal/nasal anatomy is aligned to OD-style orientation."""
    return np.fliplr(arr) if eye.upper() == "OS" else arr


def _disc_centred_patch_array(
    arr: np.ndarray,
    disc_mask: np.ndarray,
    *,
    span: int = 5,
    out: int = 96,
) -> tuple[np.ndarray | None, float | None]:
    if disc_mask is None or disc_mask.sum() == 0:
        return None, None
    ys, xs = np.where(disc_mask)
    cy, cx = ys.mean(), xs.mean()
    disc_r = float(np.sqrt(disc_mask.sum() / np.pi))
    half = max(1, int(round(span * disc_r / 2)))
    h, w = disc_mask.shape
    y0, y1 = int(round(cy)) - half, int(round(cy)) + half
    x0, x1 = int(round(cx)) - half, int(round(cx)) + half
    pad_spec = ((max(0, -y0), max(0, y1 - h)), (max(0, -x0), max(0, x1 - w)))
    if arr.ndim == 3:
        pad_spec = (*pad_spec, (0, 0))
    arr_pad = np.pad(arr, pad_spec, constant_values=0)
    patch = arr_pad[y0 + pad_spec[0][0]: y1 + pad_spec[0][0],
                    x0 + pad_spec[1][0]: x1 + pad_spec[1][0]]
    if arr.ndim == 2:
        pil = Image.fromarray((np.clip(patch, 0, 1) * 255).astype(np.uint8))
        patch_out = np.array(pil.resize((out, out), Image.BILINEAR)) / 255.0
    else:
        patch_out = np.array(Image.fromarray(patch.astype(np.uint8)).resize((out, out), Image.BILINEAR))
    return patch_out.astype(np.float32), out * disc_r / (2 * half)


QUAD_ORDER = ("ST", "SN", "IT", "IN")  # superotemporal, superonasal, inferotemporal, inferonasal


def _quadrant_fractions(
    cam: np.ndarray,
    disc_mask: np.ndarray,
    *,
    peri_inner: float = 1.0,
    peri_outer: float = 2.0,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]] | tuple[None, None, None]:
    """Per-quadrant Grad-CAM fractions in OD-oriented coordinates, for three
    region scopes.

    Quadrant boundaries are the disc-mask centroid (cx, cy). In the OD-oriented
    frame nasal is left (x < cx) and temporal is right (x > cx); superior is
    top (y < cy) and inferior is bottom (y > cy):

        ST = x > cx,  y < cy
        SN = x < cx,  y < cy
        IT = x > cx,  y > cy
        IN = x < cx,  y > cy

    Three region scopes are returned:
        disc_q : fractions of CAM intensity that fall inside the GT disc mask
        peri_q : fractions inside a peri-disc annulus of disc-radius units
                 (peri_inner to peri_outer, default 1x-2x), excluding the disc
        full_q : fractions over the entire image
    Each dict sums to 1 (within floating-point error). Returns (None, None, None)
    if the disc mask is empty.
    """
    if disc_mask is None or disc_mask.sum() == 0:
        return None, None, None
    ys, xs = np.where(disc_mask)
    cy = float(ys.mean()); cx = float(xs.mean())
    disc_r = float(np.sqrt(disc_mask.sum() / np.pi))
    h, w = cam.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    peri_mask = (dist >= peri_inner * disc_r) & (dist <= peri_outer * disc_r) & ~disc_mask
    quads = {
        "ST": (xx > cx) & (yy < cy),
        "SN": (xx < cx) & (yy < cy),
        "IT": (xx > cx) & (yy > cy),
        "IN": (xx < cx) & (yy > cy),
    }
    disc_total = float(cam[disc_mask].sum()) + 1e-8
    peri_total = float(cam[peri_mask].sum()) + 1e-8
    full_total = float(cam.sum()) + 1e-8
    disc_q = {k: float(cam[disc_mask & q].sum()) / disc_total for k, q in quads.items()}
    peri_q = {k: float(cam[peri_mask & q].sum()) / peri_total for k, q in quads.items()}
    full_q = {k: float(cam[q].sum())             / full_total for k, q in quads.items()}
    return disc_q, peri_q, full_q


def _annotate_nasal_temporal(ax, *, fontsize: int = 9, color: str = "white",
                             pad: float = 2.5) -> None:
    """Label the disc-side (nasal) and macula-side (temporal) edges of an
    OD-oriented fundus axis. After OS is mirrored to OD orientation, the disc
    sits on the LEFT (nasal) side and the macula on the RIGHT (temporal) side.
    """
    ax.text(0.015, 0.5, "N", transform=ax.transAxes,
            ha="left", va="center", fontsize=fontsize, fontweight="bold",
            color=color,
            bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=pad))
    ax.text(0.985, 0.5, "T", transform=ax.transAxes,
            ha="right", va="center", fontsize=fontsize, fontweight="bold",
            color=color,
            bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=pad))


def _make_oriented_disc_detail(mean_patches: dict, examples: dict, out_path: Path) -> None:
    """2x2 grid of mean Grad-CAM patches centred on the optic disc.

    Rows = ground-truth class (Normal / Glaucoma)
    Cols = prediction outcome relative to truth (Correct / Incorrect)

    Each cell shows the per-pixel mean Grad-CAM intensity in a disc-centred,
    OD-oriented patch (OS mirrored). A dashed circle traces the mean disc
    boundary. Nasal (N) and Temporal (T) edges are labelled.

    Sample overlays are intentionally NOT shown here; see overlay_grid_*.png
    for browseable per-image overlays and hand-pick from those for any
    figure that wants concrete examples.
    """
    from matplotlib.patches import Circle

    classes = ["Normal", "Glaucoma"]
    splits  = ["correct", "incorrect"]

    fig, axes = plt.subplots(2, 2, figsize=(8, 7.8), constrained_layout=True)
    fig.patch.set_facecolor("#f6f6f6")

    # Column headers
    for ci, split in enumerate(splits):
        axes[0, ci].set_title(
            f"{split.capitalize()} predictions",
            fontsize=11, fontweight="bold", pad=10, color="#222",
        )

    for ri, cls in enumerate(classes):
        # Row label drawn outside the leftmost cell
        axes[ri, 0].text(
            -0.13, 0.5, cls, transform=axes[ri, 0].transAxes,
            ha="right", va="center", fontsize=12, fontweight="bold",
            color="#c44e52" if cls == "Glaucoma" else "#3B6FB5", rotation=90,
        )
        for ci, split in enumerate(splits):
            ax  = axes[ri, ci]
            ax.set_facecolor("#202020")
            key = (cls, split)
            if key in mean_patches:
                patch, radius, count, *rest = mean_patches[key]
                disc_frac = rest[0] if rest else None
                ax.imshow(patch, cmap="jet", vmin=0, vmax=1)
                ax.add_patch(Circle((48, 48), radius, fill=False,
                                    edgecolor="white", linewidth=1.8, linestyle="--"))
                _annotate_nasal_temporal(ax)
                cap = f"n = {count}"
                if disc_frac is not None and not np.isnan(disc_frac):
                    cap += f"   ·   disc-frac = {disc_frac:.2f}"
                ax.text(0.5, -0.06, cap, transform=ax.transAxes,
                        ha="center", va="top", fontsize=9, color="#222")
            else:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", color="white", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        "Mean image-tower Grad-CAM, disc-centred (OD-oriented)\n"
        "rows: ground truth   ·   columns: prediction outcome   ·   dashed circle = mean disc boundary",
        fontsize=11, fontweight="bold",
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _patient_id_from_entity_id(entity_id) -> int:
    if isinstance(entity_id, (tuple, list)):
        return int(entity_id[0])
    return int(entity_id)


def _v4_gradcam_target_layer(img_tower):
    blocks = getattr(img_tower, "_blocks", None)
    if blocks:
        return blocks[-1]
    backbone = getattr(img_tower, "backbone", None)
    if hasattr(backbone, "features"):
        return backbone.features[-1]
    children = list(backbone.children()) if backbone is not None else []
    if children:
        return children[-1]
    raise RuntimeError(f"Could not infer GradCAM target layer for {type(img_tower).__name__}")


def _maybe(profile_view, method: str, *args, **kwargs):
    """Call profile_view.method(*args, **kwargs) if it exists, else return None."""
    fn = getattr(profile_view, method, None)
    return fn(*args, **kwargs) if fn is not None else None


def _orient_via_profile(profile_view, image_or_array, side: str):
    """Use the profile's orient_for_display if it provides one; else passthrough."""
    return _maybe(profile_view, "orient_for_display", image_or_array, side) \
        if hasattr(profile_view, "orient_for_display") else image_or_array


def _compute_image_aux_cam(
    gcam,
    batch: dict,
    eye: str,
    towers: dict,
    img_aux,
    device,
    target_class: int | None,
) -> tuple[np.ndarray, int]:
    """Run image-tower GradCAM against the img_aux head for one (batch, eye) pair."""
    for mod in [towers["img"], img_aux]:
        mod.zero_grad(set_to_none=True)

    side = "a" if eye.upper() == "OD" else "b"
    img_t = batch["img"][side].to(device)

    return gcam.compute(
        forward_fn=lambda: img_aux(towers["img"](img_t)),
        output_shape=img_t.shape[-2:],
        target_class=target_class,
    )


def _make_oriented_gradcam(n_grid: int = 16, alpha: float = 0.45,
                            target_class: int | None = None) -> None:
    import json
    import torch
    from tqdm import tqdm
    from v4.classes.accessory.explainability import GradCAM, overlay_gradcam
    from v4.classes.split_manager import SplitManager
    from v4.classes.v4_hypertower import load_data, _make_loader
    import v4.classes.v4_hypertower as orch

    summary = json.loads((V4_CKPT_RUN / "summary.json").read_text())
    cfg = summary["config"]
    orch.cfg_ref = cfg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"GradCAM device: {device}")

    data = load_data(cfg)
    image_view = data.image  # profile data view — owns dataset-specific knowledge
    label_filter = cfg.get("label_filter", None)
    df_mode = data.df.copy()
    if label_filter is not None:
        df_mode = df_mode[df_mode[data.label_col].isin(label_filter)].reset_index(drop=True)
    identity_cols = getattr(data, "identity_cols", [])
    identity_level = cfg.get("split_identity_level", 1)
    group_col = identity_cols[identity_level - 1] if identity_level and identity_cols else None
    splits = SplitManager(group_col=group_col).build_plans(
        df_mode,
        label_col=data.label_col,
        n_splits=cfg.get("folds", 5),
        seed=cfg.get("fold_seed", 100),
    )

    cam_sum = {0: None, 1: None}
    cam_count = {0: 0, 1: 0}
    overlay_items = {0: [], 1: []}
    disc_patch_sum: dict[tuple[str, str], np.ndarray] = {}
    disc_patch_count: dict[tuple[str, str], int] = {}
    disc_radius_sum: dict[tuple[str, str], float] = {}
    disc_frac_sum: dict[tuple[str, str], float] = {}
    # Per-eye quadrant fractions for three region scopes; aggregated per cell.
    quad_disc_list: dict[tuple[str, str], list[dict[str, float]]] = {}
    quad_peri_list: dict[tuple[str, str], list[dict[str, float]]] = {}
    quad_full_list: dict[tuple[str, str], list[dict[str, float]]] = {}
    examples: dict[tuple[str, str], tuple[np.ndarray, int, str, int]] = {}

    fold_range = range(cfg.get("folds", 5))
    for fold_idx in tqdm(fold_range, desc="GradCAM folds", unit="fold"):
        ckpt_dir = V4_CKPT_RUN / "checkpoints" / f"fold{fold_idx}"
        if not ckpt_dir.exists():
            continue
        towers, nt, hb, img_aux, _cd_aux, hb_head = _build_v4_fold_modules(
            fold_idx, cfg, data, device,
        )
        target_layer = _v4_gradcam_target_layer(towers["img"])
        gcam = GradCAM(target_layer)

        split_obj = splits[fold_idx]
        if split_obj.test is None:
            continue
        shell = data.build_shells(split_obj.test, level="patient", label_filter=label_filter)
        loader = _make_loader(shell, towers, batch_size=1, shuffle=False)

        for batch in tqdm(loader, desc=f"fold{fold_idx}", leave=False, unit="pt"):
            pid = _patient_id_from_entity_id(batch["entity_id"][0])
            label = int(batch["label"][0].item())
            for eye in ("OD", "OS"):
                if not image_view.get_image_path(pid, eye).exists():
                    continue
                pil_eval = image_view.eval_image_pil(pid, eye)
                roi_mask = _maybe(image_view, "build_roi_mask",
                                  pid, eye, target_h=pil_eval.size[1], target_w=pil_eval.size[0])

                cam_np, pred = _compute_image_aux_cam(
                    gcam, batch, eye, towers, img_aux, device, target_class,
                )

                # Orient for OD-style display (profile decides; passthrough if it doesn't define)
                cam_np = _orient_via_profile(image_view, cam_np, eye)
                pil_oriented = _orient_via_profile(image_view, pil_eval, eye)
                if roi_mask is not None:
                    roi_mask = _orient_via_profile(image_view, roi_mask, eye)

                if cam_sum[label] is None:
                    cam_sum[label] = cam_np.copy()
                else:
                    cam_sum[label] += cam_np
                cam_count[label] += 1

                if len(overlay_items[label]) < n_grid:
                    overlay_items[label].append(
                        (overlay_gradcam(pil_oriented, cam_np, alpha), pid, eye, pred)
                    )

                patch, roi_r_out = _disc_centred_patch_array(cam_np, roi_mask)
                if patch is not None:
                    key = (LABEL_NAMES[label], "correct" if pred == label else "incorrect")
                    disc_patch_sum[key] = disc_patch_sum.get(key, 0) + patch
                    disc_patch_count[key] = disc_patch_count.get(key, 0) + 1
                    disc_radius_sum[key] = disc_radius_sum.get(key, 0.0) + float(roi_r_out)
                    disc_frac_sum[key] = disc_frac_sum.get(key, 0.0) + float(
                        cam_np[roi_mask].sum() / (cam_np.sum() + 1e-8)
                    )
                    disc_q, peri_q, full_q = _quadrant_fractions(cam_np, roi_mask)
                    if disc_q is not None:
                        quad_disc_list.setdefault(key, []).append(disc_q)
                        quad_peri_list.setdefault(key, []).append(peri_q)
                        quad_full_list.setdefault(key, []).append(full_q)
                    if key not in examples:
                        ov = overlay_gradcam(pil_oriented, cam_np, alpha)
                        ov_small = np.array(ov.resize(cam_np.shape[::-1], Image.BILINEAR))
                        ex_patch, _ = _disc_centred_patch_array(ov_small, roi_mask)
                        if ex_patch is not None:
                            examples[key] = (ex_patch, pid, eye, pred)

                del cam_np

        gcam.remove()
        del towers, nt, hb, img_aux, _cd_aux, hb_head
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

    for cls in (0, 1):
        if cam_count[cls] == 0:
            continue
        mean_cam = cam_sum[cls] / cam_count[cls]
        mean_cam = (mean_cam - mean_cam.min()) / (mean_cam.max() - mean_cam.min() + 1e-8)
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(mean_cam, cmap="jet", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(f"Mean v4 image-aux GradCAM - {LABEL_NAMES[cls]} (oriented, n={cam_count[cls]})",
                     fontsize=10, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.savefig(GRADCAM_DIR / f"mean_cam_{LABEL_NAMES[cls].lower()}.png",
                    dpi=180, bbox_inches="tight")
        plt.close(fig)

    if cam_count[0] and cam_count[1]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), constrained_layout=True)
        fig.patch.set_facecolor("#f6f6f6")
        fig.suptitle(
            "Mean image-tower Grad-CAM by ground-truth class (OD-oriented)",
            fontsize=12, fontweight="bold",
        )
        cls_colors = {0: "#3B6FB5", 1: "#c44e52"}
        for ax, cls in zip(axes, (0, 1)):
            mean_cam = cam_sum[cls] / cam_count[cls]
            mean_cam = (mean_cam - mean_cam.min()) / (mean_cam.max() - mean_cam.min() + 1e-8)
            im = ax.imshow(mean_cam, cmap="jet", vmin=0, vmax=1)
            _annotate_nasal_temporal(ax)
            ax.set_title(
                f"{LABEL_NAMES[cls]}    n = {cam_count[cls]}",
                fontsize=12, fontweight="bold", color=cls_colors[cls],
            )
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.savefig(GRADCAM_DIR / "mean_cam_comparison.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    for cls in (0, 1):
        items = overlay_items[cls]
        if not items:
            continue
        items.sort(key=lambda x: x[3] == cls)
        ncols = 4
        nrows = int(np.ceil(len(items) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.1, nrows * 3.1))
        axes = np.array(axes).reshape(-1)
        fig.suptitle(
            f"Per-eye Grad-CAM samples for browsing — ground truth = {LABEL_NAMES[cls]}\n"
            "(OD-oriented; green title = model agreed, red = disagreed)",
            fontsize=11, fontweight="bold",
        )
        for i, ax in enumerate(axes):
            if i < len(items):
                ov, pid, eye, pred = items[i]
                ax.imshow(ov)
                _annotate_nasal_temporal(ax, fontsize=7, pad=1.5)
                color = "#2f7d46" if pred == cls else "#b23a48"
                ax.set_title(f"RET{pid:03d}{eye} -> {LABEL_NAMES[pred]}",
                             fontsize=7.5, color=color)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(GRADCAM_DIR / f"overlay_grid_{LABEL_NAMES[cls].lower()}.png",
                    dpi=160, bbox_inches="tight")
        plt.close(fig)

    mean_patches = {
        key: (
            disc_patch_sum[key] / disc_patch_count[key],
            disc_radius_sum[key] / disc_patch_count[key],
            disc_patch_count[key],
            disc_frac_sum.get(key, float("nan")) / disc_patch_count[key],
        )
        for key in disc_patch_sum
    }
    if mean_patches:
        _make_oriented_disc_detail(mean_patches, examples, GRADCAM_DIR / "disc_attention_detail.png")

    # Per-quadrant CAM fractions (within-disc and full-image scopes).
    # One CSV row per (class, outcome, scope, quadrant) cell, with mean and SD
    # computed over the per-eye fractions in that cell.
    rows = []
    for key in sorted(quad_disc_list.keys()):
        cls_name, outcome = key
        n_eyes = len(quad_disc_list[key])
        for scope_label, scope_list in (("disc", quad_disc_list[key]),
                                        ("peri", quad_peri_list[key]),
                                        ("full", quad_full_list[key])):
            for q in QUAD_ORDER:
                vals = np.array([d[q] for d in scope_list], dtype=np.float64)
                rows.append({
                    "class":    cls_name,
                    "outcome":  outcome,
                    "scope":    scope_label,
                    "quadrant": q,
                    "n_eyes":   n_eyes,
                    "mean":     float(vals.mean()),
                    "sd":       float(vals.std(ddof=1)) if n_eyes > 1 else float("nan"),
                })
    if rows:
        out_csv = OUT_DIR / "F8_quadrant_fractions.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"saved quadrant fractions: {out_csv}")


def make_oriented_gradcam(n_grid: int = 16, alpha: float = 0.45,
                          target_class: int | None = None) -> None:
    _make_oriented_gradcam(n_grid=n_grid, alpha=alpha, target_class=target_class)


def main(*, run_gradcam: bool = False, n_permutations: int = 30,
         fusion_split: str = "test") -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_fusion_event_panel(split=fusion_split)
    make_clinical_importance(n_permutations=n_permutations)
    if run_gradcam:
        make_oriented_gradcam()
    else:
        print("Skipped GradCAM. Re-run with --run-gradcam to generate oriented heatmaps.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only-fusion", action="store_true",
                        help="Generate only the fusion event panel; skip clinical and GradCAM figures.")
    parser.add_argument("--only-gradcam", action="store_true",
                        help="Generate only oriented GradCAM outputs; skip fusion and clinical figures.")
    parser.add_argument("--run-gradcam", action="store_true",
                        help="Generate GPU-heavy oriented GradCAM outputs.")
    parser.add_argument("--n-grid", type=int, default=16,
                        help="Max overlays per class for GradCAM grids.")
    parser.add_argument("--alpha", type=float, default=0.45,
                        help="GradCAM overlay opacity.")
    parser.add_argument("--target-class", type=int, default=None,
                        help="GradCAM target class; default uses predicted class.")
    parser.add_argument("--n-permutations", type=int, default=30,
                        help="Clinical permutation repeats per feature.")
    parser.add_argument("--fusion-split", choices=["test", "val", "both"], default="test",
                        help="Prediction split for the fusion event panel.")
    args = parser.parse_args()
    if args.only_fusion:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        make_fusion_event_panel(split=args.fusion_split)
    elif args.only_gradcam:
        make_oriented_gradcam(
            n_grid=args.n_grid,
            alpha=args.alpha,
            target_class=args.target_class,
        )
    else:
        main(
            run_gradcam=args.run_gradcam,
            n_permutations=args.n_permutations,
            fusion_split=args.fusion_split,
        )
