"""Per-tower gate / contribution readout + per-head AUC sanity check.

For each gated_<backbone> run (and ortho_<backbone>), loads each fold's
checkpoints and runs inference on the test set, capturing in a single pass:
  • per-sample per-stream bridge attention (sigmoid gates for
    GatedAdditiveBridge; ||W_i·z_i|| pre-LN projection norms for OrthoBridge)
  • per-fold AUCs of the eye-level aux heads (img_aux, cd_aux, nt_head) —
    sanity check that the image tower really is getting stronger as the
    backbone improves, and that the clinical tower stays consistent

Outputs:
    v4/results/experiments/bridge_attention/_summary/gated_gates.csv
    v4/results/experiments/bridge_attention/_summary/ortho_stream_norms.csv
        Both CSVs include img_aux_auc, cd_aux_auc, nt_head_auc columns.

Run:
    python -m v4.scripts.analysis.bridge_attention_readout
    python -m v4.scripts.analysis.bridge_attention_readout --bridges gated
    python -m v4.scripts.analysis.bridge_attention_readout --reps 3   # subsample for speed
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from v4.classes.split_manager import SplitManager
from v4.classes.v4_hypertower import _make_loader, build_towers, load_data
import v4.classes.v4_hypertower as orch


SWEEP_ROOT = Path("v4/results/experiments/bridge_attention")
OUT_DIR = SWEEP_ROOT / "_summary"

BACKBONES = [
    "mobilenet_v2",
    "resnet50",
    "efficientnet_b0",
    "efficientnet_v2_m",
    "refugelike",
    "refuge_efficientnet_v2_m",
]


# ─────────────────────────────────────────────────────────────────────────────
# Shared fold-module loader (trimmed version of F8's helper)
# ─────────────────────────────────────────────────────────────────────────────

def _load_fold_bridge(ckpt_dir: Path, cfg: dict, data, device):
    """Load towers + nt bridge + eye-level aux heads from a single fold ckpt dir."""
    from v4.classes.heads.classifier import ClassificationHead

    towers = build_towers(cfg["towers"], data)
    for name, tower in towers.items():
        tower.load_state_dict(
            torch.load(ckpt_dir / f"tower_{name}.pt", map_location="cpu")
        )
        tower.to(device).eval()

    stage_by_name = {s["name"]: s for s in cfg["stages"]}
    nt_cfg = stage_by_name["nt"]
    nt_mod = importlib.import_module(nt_cfg["module"])
    nt = getattr(nt_mod, nt_cfg["class"])(
        [towers[n].out_dim for n in nt_cfg["inputs"]],
        **nt_cfg.get("args", {}),
    ).to(device)
    nt.load_state_dict(torch.load(ckpt_dir / "stage_nt.pt", map_location="cpu"))
    nt.eval()

    nc = cfg["num_classes"]
    img_aux = ClassificationHead(towers["img"].out_dim, nc).to(device)
    cd_aux = ClassificationHead(towers["cd"].out_dim, nc).to(device)
    nt_head = ClassificationHead(nt.out_dim, nc).to(device)
    img_aux.load_state_dict(torch.load(ckpt_dir / "stage_img_aux.pt", map_location="cpu"))
    cd_aux.load_state_dict(torch.load(ckpt_dir / "stage_cd_aux.pt", map_location="cpu"))
    nt_head.load_state_dict(torch.load(ckpt_dir / "stage_nt_head.pt", map_location="cpu"))
    img_aux.eval(); cd_aux.eval(); nt_head.eval()
    aux = {"img_aux": img_aux, "cd_aux": cd_aux, "nt_head": nt_head}
    return towers, nt, nt_cfg, aux


def _splits_from_cfg(cfg: dict, data):
    label_filter = cfg.get("label_filter", None)
    df_mode = data.df.copy()
    if label_filter is not None:
        df_mode = df_mode[df_mode[data.label_col].isin(label_filter)].reset_index(drop=True)
    identity_cols = getattr(data, "identity_cols", [])
    identity_level = cfg.get("split_identity_level", 1)
    group_col = (
        identity_cols[identity_level - 1]
        if identity_level and identity_cols
        else None
    )
    return SplitManager(group_col=group_col).build_plans(
        df_mode,
        label_col=data.label_col,
        n_splits=cfg.get("folds", 5),
        seed=cfg.get("fold_seed", 100),
    ), label_filter


# ─────────────────────────────────────────────────────────────────────────────
# Unified inference: capture bridge attention + per-head logits in one pass
# ─────────────────────────────────────────────────────────────────────────────

def _install_attention_hooks(nt, bridge_kind: str):
    """Register the right hook(s) for the bridge kind. Returns (handles, finalize)
    where finalize() reads back the captured per-sample (N, n_streams) array."""
    if bridge_kind == "gated":
        captured: list[torch.Tensor] = []
        def hook(_module, _input, output):
            captured.append(output.detach().cpu())
        handles = [nt.gate.register_forward_hook(hook)]
        def finalize():
            return (
                torch.cat(captured, dim=0).numpy() if captured else np.empty((0, 0))
            )
        return handles, finalize

    if bridge_kind == "ortho":
        # Pre-LN projection norm ||W_i z_i||_2 per stream. Post-LN norm is
        # √fusion_dim by construction, so we hook the linear projection itself.
        per_stream: list[list[torch.Tensor]] = [[] for _ in range(len(nt.inner.W))]
        handles = []
        for i, w in enumerate(nt.inner.W):
            def make_hook(idx):
                def hook(_module, _input, output):
                    per_stream[idx].append(output.detach().norm(dim=-1).cpu())
                return hook
            handles.append(w.register_forward_hook(make_hook(i)))
        def finalize():
            cols = [
                torch.cat(c, dim=0).numpy() if c else np.array([])
                for c in per_stream
            ]
            if not all(s.size for s in cols):
                return np.empty((0, 0))
            return np.stack(cols, axis=1)
        return handles, finalize

    raise ValueError(f"unknown bridge_kind {bridge_kind!r}")


def extract_one_fold(
    bridge_kind: str,
    towers, nt, aux: dict,
    data, split_obj, label_filter, device,
):
    """Single inference pass that returns:
        attention : ndarray (N, n_streams) — bridge-specific attention signal
        head_aucs : dict[str, float] — AUC of softmax[:,1] for each aux head
        n_samples : int
    """
    from sklearn.metrics import roc_auc_score
    import torch.nn.functional as F

    if split_obj.test is None:
        return np.empty((0, 0)), {}, 0

    shell = data.build_shells(
        split_obj.test, level="patient", label_filter=label_filter
    )
    loader = _make_loader(shell, towers, batch_size=8, shuffle=False)

    handles, finalize = _install_attention_hooks(nt, bridge_kind)
    img_logits, cd_logits, nt_logits, ys = [], [], [], []

    try:
        with torch.no_grad():
            for batch in loader:
                y = batch["label"].detach().cpu().numpy()
                for side in ("a", "b"):
                    z_img = towers["img"](batch["img"][side].to(device))
                    z_cd = towers["cd"](batch["cd"][side].to(device))
                    img_logits.append(aux["img_aux"](z_img).cpu())
                    cd_logits.append(aux["cd_aux"](z_cd).cpu())
                    z_fused = nt([z_img, z_cd])  # attention hooks capture here
                    nt_logits.append(aux["nt_head"](z_fused).cpu())
                    ys.append(y)
        attention = finalize()
    finally:
        for h in handles:
            h.remove()

    if not ys:
        return attention, {}, 0

    y_all = np.concatenate(ys)
    head_aucs: dict[str, float] = {}
    for name, buf in [
        ("img_aux", img_logits),
        ("cd_aux",  cd_logits),
        ("nt_head", nt_logits),
    ]:
        logits = torch.cat(buf, dim=0).numpy()
        probs = (np.exp(logits - logits.max(axis=1, keepdims=True))
                 / np.exp(logits - logits.max(axis=1, keepdims=True))
                   .sum(axis=1, keepdims=True))
        try:
            head_aucs[name] = float(roc_auc_score(y_all, probs[:, 1]))
        except Exception:
            head_aucs[name] = float("nan")

    return attention, head_aucs, int(attention.shape[0])


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def run_for_bridge(bridge_kind: str, max_reps: int | None, device) -> pd.DataFrame:
    rows: list[dict] = []
    for backbone in BACKBONES:
        run_dir = SWEEP_ROOT / f"{bridge_kind}_{backbone}"
        if not run_dir.exists():
            print(f"  [{bridge_kind}/{backbone}] (missing)")
            continue
        rep_dirs = sorted(run_dir.glob("rep*"))
        if max_reps is not None:
            rep_dirs = rep_dirs[:max_reps]
        for rep_dir in tqdm(rep_dirs, desc=f"{bridge_kind}/{backbone}", unit="rep"):
            summary_path = rep_dir / "binary" / "summary.json"
            if not summary_path.exists():
                continue
            cfg = json.loads(summary_path.read_text())["config"]
            orch.cfg_ref = cfg
            data = load_data(cfg)
            splits, label_filter = _splits_from_cfg(cfg, data)

            try:
                rep_idx = int(rep_dir.name.replace("rep", ""))
            except ValueError:
                continue

            for fold_idx in range(cfg.get("folds", 5)):
                ckpt_dir = rep_dir / "binary" / "checkpoints" / f"fold{fold_idx}"
                if not ckpt_dir.exists():
                    continue
                towers, nt, _, aux = _load_fold_bridge(ckpt_dir, cfg, data, device)
                arr, head_aucs, _ = extract_one_fold(
                    bridge_kind, towers, nt, aux, data,
                    splits[fold_idx], label_filter, device,
                )

                if arr.size:
                    means = arr.mean(axis=0)
                    rows.append({
                        "bridge": bridge_kind,
                        "backbone": backbone,
                        "rep": rep_idx,
                        "fold": fold_idx,
                        "n_samples": arr.shape[0],
                        "img_value": float(means[0]),
                        "cd_value": float(means[1]),
                        "img_share": float(means[0] / (means.sum() + 1e-8)),
                        "img_aux_auc": head_aucs.get("img_aux", float("nan")),
                        "cd_aux_auc":  head_aucs.get("cd_aux",  float("nan")),
                        "nt_head_auc": head_aucs.get("nt_head", float("nan")),
                    })

                del towers, nt, aux
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bridges", default="gated,ortho",
        help="Comma-separated bridge kinds to process (default: gated,ortho)",
    )
    ap.add_argument(
        "--reps", type=int, default=None,
        help="Cap reps per backbone for a quick first pass (default: all)",
    )
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for bridge in args.bridges.split(","):
        bridge = bridge.strip()
        if not bridge:
            continue
        df = run_for_bridge(bridge, args.reps, device)
        suffix = "gates" if bridge == "gated" else "stream_norms"
        out = OUT_DIR / f"{bridge}_{suffix}.csv"
        df.to_csv(out, index=False)
        print(f"saved {out}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
