#!/usr/bin/env python
"""
V4 HyperTower orchestrator — config-driven stage pipeline.

Stage logic lives in v4/classes/stages/:
  warm.py    — pre-trains a single tower with a temporary linear probe
  fusion.py  — trains a bridge + associated head stages, freezes for downstream use
  helpers.py — encode_embedding, resolve_input_dims, phase_for_epoch, etc.

Usage:
  python -m v4.classes.v4_hypertower --config v4/configs/ensemble_fused.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from v4.classes.dataset import LoaderShell, HTDataset, ht_collate
from v4.classes.utils import seed_everything, choose_device
from v4.classes.split_manager import SplitManager
from v4.classes.stages import warm, fusion


# ---------------------------------------------------------------------------
# Early-pass protocol
# ---------------------------------------------------------------------------

class EarlyPassContext:
    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def put(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def require(self, key: str) -> Any:
        if key not in self._store:
            raise KeyError(
                f"EarlyPassContext: required key '{key}' not present. "
                f"Available: {sorted(self._store.keys())}"
            )
        return self._store[key]

    def keys(self) -> set[str]:
        return set(self._store.keys())


def _validate_epc_requests(towers_cfg: list[dict], provided_keys: set[str]) -> None:
    available = set(provided_keys)
    for t in towers_cfg:
        for req in t.get("epc_requests", []):
            if req not in available:
                raise ValueError(
                    f"Tower '{t['name']}' requests EPC key '{req}' "
                    f"but no supplier provides it. Available: {sorted(available)}"
                )
        available.update(t.get("epc_supplies", []))


# ---------------------------------------------------------------------------
# Data + tower helpers
# ---------------------------------------------------------------------------

cfg_ref: dict = {}


def load_data(cfg: dict):
    data_cfg = cfg["data"]
    args = dict(data_cfg.get("args", {}))
    for key in ("image_dir", "clinical_dir"):
        if key in args:
            p = Path(args[key])
            if not p.is_absolute():
                args[key] = str(REPO_ROOT / p)
    mod = importlib.import_module(data_cfg["module"])
    return mod.build_data(args)


def build_towers(towers_cfg: list[dict], data) -> dict:
    def _resolve(path: str):
        mod_name = cfg_ref.get("data", {}).get("module", "")
        if mod_name:
            try:
                m = importlib.import_module(mod_name)
                if hasattr(m, "resolve_data_source"):
                    return m.resolve_data_source(data, path)
            except Exception:
                pass
        obj = data
        for part in path.split("."):
            obj = getattr(obj, part)
        return obj

    towers = {}
    for t in towers_cfg:
        mod = importlib.import_module(t["module"])
        cls = getattr(mod, t["class"])
        kwargs = dict(t.get("args", {}))
        if "data_source" in t:
            towers[t["name"]] = cls(_resolve(t["data_source"]), **kwargs)
        elif "data_arg" in t:
            kwargs[t["data_arg"]] = data
            towers[t["name"]] = cls(**kwargs)
        else:
            towers[t["name"]] = cls(**kwargs)
    return towers


def _make_loader(shell: LoaderShell, towers: dict, *, batch_size: int,
                 shuffle: bool, sampler=None) -> DataLoader:
    dataset = HTDataset(shell, towers)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        collate_fn=ht_collate,
        num_workers=0,
        persistent_workers=False,
    )


def _balanced_sampler(shell: LoaderShell):
    from torch.utils.data import WeightedRandomSampler
    labels  = [e.label for e in shell.entries]
    counts  = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    weights = [1.0 / counts[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ---------------------------------------------------------------------------
# Fold runner
# ---------------------------------------------------------------------------

def run_fold(fold: int, splits, cfg: dict, data, num_classes: int, device) -> dict:
    seed_everything(cfg["seed"] + fold * 100)
    split        = splits[fold]
    label_filter = cfg.get("label_filter", None)
    cfg_stages   = cfg["stages"]

    towers = build_towers(cfg["towers"], data)
    for t in towers.values():
        t.to(device)

    context = EarlyPassContext()
    context.put("device",       device)
    context.put("data",         data)
    context.put("split",        split)
    context.put("label_filter", label_filter)
    _validate_epc_requests(cfg["towers"], context.keys())
    for tower in towers.values():
        if hasattr(tower, "early_pass"):
            tower.early_pass(context)

    stage_models: dict = {}
    fold_result         = {"fold": fold}

    for stage_cfg in cfg_stages:
        stype = stage_cfg["type"]

        if stype == "warm":
            warm.run(stage_cfg, towers, data, split, label_filter,
                     cfg, num_classes, device, fold,
                     _make_loader, _balanced_sampler)

        elif stype == "fusion":
            stage_models, metrics = fusion.run(
                stage_cfg, cfg, towers, stage_models, data, split,
                label_filter, num_classes, device, fold, cfg_stages,
                _make_loader,
            )
            fold_result.update(metrics)

        # head stages are handled inside fusion.run

    return fold_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global cfg_ref

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    cfg_ref = cfg

    device      = choose_device(args.device or cfg.get("device"))
    num_classes = cfg.get("num_classes", 2)
    label_filter = cfg.get("label_filter", None)
    print(f"Device: {device}", flush=True)

    print("Loading data ...", flush=True)
    data = load_data(cfg)
    print(f"  feature_dim={data.feature_dim}", flush=True)

    label_col = cfg["data"]["args"].get("label_col", "Diagnosis")
    df_mode   = data.df.copy()
    if label_filter is not None:
        df_mode = df_mode[df_mode[label_col].isin(label_filter)].reset_index(drop=True)

    identity_level = cfg.get("split_identity_level", 1)
    identity_cols  = getattr(data, "identity_cols", [])
    group_col = identity_cols[identity_level - 1] if identity_level and identity_cols else None

    splits = SplitManager(group_col=group_col).build_plans(
        df_mode,
        label_col=label_col,
        n_splits=cfg.get("folds", 5),
        seed=cfg.get("fold_seed", 100),
    )

    out_dir_tags = cfg.get("out_dir_tags", [])
    out_dir = REPO_ROOT / cfg.get("output_root", "v4/results") / cfg["run_name"]
    for tag in out_dir_tags:
        out_dir = out_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_stage   = cfg.get("eval_stage", "hb")
    fold_results = []
    t0 = time.time()

    for fold in range(cfg.get("folds", 5)):
        split   = splits[fold]
        n_train = split.train[group_col].nunique() if group_col else len(split.train)
        print(f"\n── fold {fold+1}/{cfg.get('folds', 5)}  train_groups={n_train} ──",
              flush=True)
        result = run_fold(fold, splits, cfg, data, num_classes, device)
        fold_results.append(result)
        print(
            f"  fold{fold+1} DONE"
            f"  val_auc={result.get(f'{eval_stage}_val_auc', float('nan')):.4f}"
            f"  test_auc={result.get(f'{eval_stage}_test_auc', float('nan')):.4f}",
            flush=True,
        )

    if fold_results:
        val_aucs  = [r.get(f"{eval_stage}_val_auc",  float("nan")) for r in fold_results]
        test_aucs = [r.get(f"{eval_stage}_test_auc", float("nan")) for r in fold_results]
        val_aucs  = [v for v in val_aucs  if not np.isnan(v)]
        test_aucs = [v for v in test_aucs if not np.isnan(v)]
        summary = {
            "run_name":      cfg["run_name"],
            "eval_stage":    eval_stage,
            "config":        cfg,
            "mean_val_auc":  float(np.mean(val_aucs))  if val_aucs  else float("nan"),
            "std_val_auc":   float(np.std(val_aucs))   if val_aucs  else float("nan"),
            "mean_test_auc": float(np.mean(test_aucs)) if test_aucs else float("nan"),
            "std_test_auc":  float(np.std(test_aucs))  if test_aucs else float("nan"),
            "elapsed_s":     round(time.time() - t0, 1),
            "fold_results":  fold_results,
        }
        summary_path = out_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"\n{'='*60}", flush=True)
        print(f"Val  AUC: {summary['mean_val_auc']:.4f} ± {summary['std_val_auc']:.4f}", flush=True)
        print(f"Test AUC: {summary['mean_test_auc']:.4f} ± {summary['std_test_auc']:.4f}", flush=True)
        print(f"Saved: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
