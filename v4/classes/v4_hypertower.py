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
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from v4.classes.dataset import LoaderShell, HTDataset, ht_collate
from v4.classes.utils import seed_everything, choose_device
from v4.classes.split_manager import SplitManager
from v4.classes.stages import warm, fusion, parallel
from v4.classes.logging.prediction_store import PredictionStore, FeatureStore


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

def run_fold(fold: int, splits, cfg: dict, data, num_classes: int, device):
    """Train + evaluate one fold.

    Returns (fold_result, fold_preds, towers, stage_models). The latter two are
    handy for opt-in artefact saving (e.g. checkpoint dumps for explainability
    runs) without forcing the orchestrator to know about every saved tensor.
    """
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

    # Flatten parallel wrappers so sub-stage configs are addressable by name.
    flat_stages: list[dict] = []
    for s in cfg_stages:
        if s.get("type") == "parallel":
            flat_stages.extend(s["stages"])
        else:
            flat_stages.append(s)

    stage_models: dict = {}
    fold_result         = {"fold": fold}
    fold_preds:   dict  = {}   # stage_name → pred_data

    for stage_cfg in cfg_stages:
        stype = stage_cfg["type"]

        if stype == "warm":
            stage_models = warm.run(
                stage_cfg, towers, data, split, label_filter,
                cfg, num_classes, device, fold,
                _make_loader, _balanced_sampler, stage_models, flat_stages,
            )

        elif stype == "fusion":
            stage_models, metrics, preds = fusion.run(
                stage_cfg, cfg, towers, stage_models, data, split,
                label_filter, num_classes, device, fold, flat_stages,
                _make_loader,
            )
            fold_result.update(metrics)
            fold_preds.update(preds)

        elif stype == "parallel":
            stage_models, metrics, preds = parallel.run(
                stage_cfg, cfg, towers, stage_models, data, split,
                label_filter, num_classes, device, fold, flat_stages,
                _make_loader, _balanced_sampler,
            )
            fold_result.update(metrics)
            fold_preds.update(preds)

        # head stages are handled inside fusion.run

    return fold_result, fold_preds, towers, stage_models


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

    eval_stage        = cfg.get("eval_stage", "hb")
    save_predictions  = cfg.get("save_predictions", False)
    save_features     = cfg.get("save_features", False)
    save_checkpoints  = cfg.get("save_checkpoints", False)
    fold_results      = []
    eval_stage_preds  = []   # list[dict] — one per fold, only for eval_stage
    all_phase_preds: dict[str, list[dict]] = {}  # phase → list[dict] across folds
    t0 = time.time()

    for fold in range(cfg.get("folds", 5)):
        split   = splits[fold]
        n_train = split.train[group_col].nunique() if group_col else len(split.train)
        print(f"\n── fold {fold+1}/{cfg.get('folds', 5)}  train_groups={n_train} ──",
              flush=True)
        result, fold_preds, towers, stage_models = run_fold(
            fold, splits, cfg, data, num_classes, device,
        )
        fold_results.append(result)

        if save_checkpoints:
            ckpt_dir = out_dir / "checkpoints" / f"fold{fold}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            for name, mod in towers.items():
                torch.save(mod.state_dict(), ckpt_dir / f"tower_{name}.pt")
            for name, mod in stage_models.items():
                # Skip non-Module entries (defensive); only nn.Modules have state_dict
                if hasattr(mod, "state_dict"):
                    torch.save(mod.state_dict(), ckpt_dir / f"stage_{name}.pt")
            print(f"  Checkpoints saved: {ckpt_dir}", flush=True)
        if save_predictions and eval_stage in fold_preds:
            eval_stage_preds.append(fold_preds[eval_stage])
        if save_features:
            for ph, pdata in fold_preds.items():
                if pdata.get("val_z") is None:
                    continue
                all_phase_preds.setdefault(ph, []).append(pdata)
        # Look up the eval stage's primary metric name (set by the stage runner).
        primary_name = result.get(f"{eval_stage}_val_primary_name", "auc")
        val_primary  = result.get(f"{eval_stage}_val_{primary_name}",  float("nan"))
        test_primary = result.get(f"{eval_stage}_test_{primary_name}", float("nan"))
        print(
            f"  fold{fold+1} DONE"
            f"  val_{primary_name}={val_primary:.4f}"
            f"  test_{primary_name}={test_primary:.4f}",
            flush=True,
        )

    if fold_results:
        # Resolve primary metric name from first valid fold result.
        primary_name = next(
            (r.get(f"{eval_stage}_val_primary_name", "auc") for r in fold_results
             if r.get(f"{eval_stage}_val_primary_name") is not None),
            "auc",
        )
        val_primaries  = [r.get(f"{eval_stage}_val_{primary_name}",  float("nan"))
                          for r in fold_results]
        test_primaries = [r.get(f"{eval_stage}_test_{primary_name}", float("nan"))
                          for r in fold_results]
        val_primaries  = [v for v in val_primaries  if not np.isnan(v)]
        test_primaries = [v for v in test_primaries if not np.isnan(v)]
        summary = {
            "run_name":         cfg["run_name"],
            "eval_stage":       eval_stage,
            "primary_metric":   primary_name,
            "config":           cfg,
            # Canonical primary-metric stats
            f"mean_val_{primary_name}":  float(np.mean(val_primaries))  if val_primaries  else float("nan"),
            f"std_val_{primary_name}":   float(np.std(val_primaries))   if val_primaries  else float("nan"),
            f"mean_test_{primary_name}": float(np.mean(test_primaries)) if test_primaries else float("nan"),
            f"std_test_{primary_name}":  float(np.std(test_primaries))  if test_primaries else float("nan"),
            # Backward-compat aliases so existing analysis tooling (summarize_run.py,
            # compare_grid.py) still reads correctly for classification runs.
            "mean_val_auc":  float(np.mean(val_primaries))  if val_primaries  else float("nan"),
            "std_val_auc":   float(np.std(val_primaries))   if val_primaries  else float("nan"),
            "mean_test_auc": float(np.mean(test_primaries)) if test_primaries else float("nan"),
            "std_test_auc":  float(np.std(test_primaries))  if test_primaries else float("nan"),
            "elapsed_s":     round(time.time() - t0, 1),
            "fold_results":  fold_results,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"\n{'='*60}", flush=True)
        print(f"Val  {primary_name}: {summary[f'mean_val_{primary_name}']:.4f} ± "
              f"{summary[f'std_val_{primary_name}']:.4f}", flush=True)
        print(f"Test {primary_name}: {summary[f'mean_test_{primary_name}']:.4f} ± "
              f"{summary[f'std_test_{primary_name}']:.4f}", flush=True)
        print(f"Saved: {summary_path}", flush=True)

        if save_predictions and eval_stage_preds:
            # Collect all unique entity_ids across val+test sets of all folds.
            seen, all_ids, id_to_y = set(), [], {}
            for fp in eval_stage_preds:
                for eid, y in zip(fp["val_ids"], fp["val_y"]):
                    k = str(eid)
                    if k not in seen:
                        seen.add(k); all_ids.append(eid)
                    id_to_y[k] = int(y)
                if fp.get("test_ids"):
                    for eid, y in zip(fp["test_ids"], fp["test_y"]):
                        k = str(eid)
                        if k not in seen:
                            seen.add(k); all_ids.append(eid)
                        id_to_y[k] = int(y)

            y_true = np.array([id_to_y.get(str(e), -1) for e in all_ids], dtype=np.int64)
            store  = PredictionStore(n_folds=len(eval_stage_preds), n_classes=num_classes)
            store.register_phase(
                phase=eval_stage,
                entity_ids=all_ids,
                y_true=y_true,
                head_names=[f"{eval_stage}_head"],
                n_epochs=1,
            )
            for fold_idx, fp in enumerate(eval_stage_preds):
                store.record(eval_stage, fold_idx, 0, fp["val_ids"],
                             f"{eval_stage}_head", fp["val_p"])
                store.set_split(eval_stage, fold_idx, fp["val_ids"], "val")
                if fp.get("test_ids"):
                    store.record(eval_stage, fold_idx, 0, fp["test_ids"],
                                 f"{eval_stage}_head", fp["test_p"])
                    store.set_split(eval_stage, fold_idx, fp["test_ids"], "test")

            pred_path = out_dir / "predictions.h5"
            store.save(pred_path)
            print(f"Predictions saved: {pred_path}", flush=True)

        if save_features and all_phase_preds:
            # One FeatureStore covers all phases; each phase gets its own group.
            n_folds_any = max(len(v) for v in all_phase_preds.values())
            fstore = FeatureStore(n_folds=n_folds_any)
            for phase, phase_preds in all_phase_preds.items():
                emb_dim = phase_preds[0]["val_z"].shape[-1]
                seen, all_ids, id_to_y = set(), [], {}
                for fp in phase_preds:
                    for eid, y in zip(fp["val_ids"], fp["val_y"]):
                        k = str(eid)
                        if k not in seen:
                            seen.add(k); all_ids.append(eid)
                        id_to_y[k] = int(y)
                    if fp.get("test_ids"):
                        for eid, y in zip(fp["test_ids"], fp["test_y"]):
                            k = str(eid)
                            if k not in seen:
                                seen.add(k); all_ids.append(eid)
                            id_to_y[k] = int(y)
                y_true = np.array([id_to_y.get(str(e), -1) for e in all_ids], dtype=np.int64)
                fstore.register_phase(phase=phase, entity_ids=all_ids, y_true=y_true)
                fstore.register_head(phase=phase, head=f"{phase}_embedding",
                                     n_epochs=1, embedding_dim=emb_dim)

                for fold_idx, fp in enumerate(phase_preds):
                    fstore.record(phase, fold_idx, 0, fp["val_ids"],
                                  f"{phase}_embedding", fp["val_z"])
                    fstore.set_split(phase, fold_idx, fp["val_ids"], "val")
                    if fp.get("test_ids") and fp.get("test_z") is not None:
                        fstore.record(phase, fold_idx, 0, fp["test_ids"],
                                      f"{phase}_embedding", fp["test_z"])
                        fstore.set_split(phase, fold_idx, fp["test_ids"], "test")

            feat_path = out_dir / "features.h5"
            fstore.save(feat_path)
            print(f"Features saved (phases: {sorted(all_phase_preds.keys())}): {feat_path}",
                  flush=True)


if __name__ == "__main__":
    main()
