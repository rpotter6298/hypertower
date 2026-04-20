"""stages/fusion — fusion stage runner: trains a bridge + associated head stages."""
from __future__ import annotations

import importlib
from random import choice, random as _random

import numpy as np
import torch
import torch.nn.functional as F

from v4.classes.dataset import LoaderShell, to_label_tensor
from v4.classes.metrics import score_arrays, compute_extended_metrics, tune_binary_threshold
from v4.classes.stages.helpers import (
    encode_embedding, get_out_dim, resolve_input_dims, phase_for_epoch,
)


def collect_probs(
    bridge,
    primary_head,
    stage_cfg:    dict,
    towers:       dict,
    stage_models: dict,
    cfg_stages:   list[dict],
    loader,
    device,
    num_classes:  int,
) -> tuple[np.ndarray, np.ndarray]:
    """Eval pass for one fusion stage; returns (y_true, softmax_probs)."""
    from v4.classes.dataset import to_label_tensor
    bridge.eval(); primary_head.eval()
    for t in towers.values():
        t.eval()
    inputs       = stage_cfg["inputs"]
    is_bilateral = isinstance(inputs, dict)
    y_all, p_all = [], []

    with torch.no_grad():
        for batch in loader:
            y = batch.get("label")
            if not torch.is_tensor(y):
                continue

            if is_bilateral:
                side_embs = {
                    side: encode_embedding(src, batch, side, towers, stage_models, cfg_stages, device)
                    for side, src in inputs.items()
                }
                z = bridge(side_embs)
            else:
                embs = [
                    encode_embedding(n, batch, None, towers, stage_models, cfg_stages, device)
                    for n in inputs
                ]
                z = bridge(embs)

            logits = primary_head(z)
            y_all.append(to_label_tensor(y, device).cpu().numpy())
            p_all.append(F.softmax(logits, dim=1).cpu().numpy())

    if not y_all:
        return np.zeros(0, dtype=np.int64), np.zeros((0, num_classes), dtype=np.float32)
    return np.concatenate(y_all), np.concatenate(p_all, axis=0)


def run(
    stage_cfg:    dict,
    cfg:          dict,
    towers:       dict,
    stage_models: dict,
    data,
    split,
    label_filter,
    num_classes:  int,
    device,
    fold:         int,
    cfg_stages:   list[dict],
    _make_loader,
) -> tuple[dict, dict]:
    """Train one fusion stage + its associated head stages.

    Returns (updated_stage_models, metrics_dict).
    """
    nan          = float("nan")
    name         = stage_cfg["name"]
    level        = stage_cfg["level"]
    epochs       = stage_cfg["epochs"]
    inputs       = stage_cfg["inputs"]
    is_bilateral = isinstance(inputs, dict)

    s_train = data.build_shells(split.train, level=level, label_filter=label_filter)
    s_val   = data.build_shells(split.val,   level=level, label_filter=label_filter)
    s_test  = (data.build_shells(split.test, level=level, label_filter=label_filter)
               if split.test is not None else LoaderShell(entries=[]))

    if not s_val.entries:
        print(f"  fold{fold+1}: no val samples for stage {name!r}, skipping.", flush=True)
        return stage_models, {}

    bs           = cfg["training"]["batch_size"]
    train_loader = _make_loader(s_train, towers, batch_size=bs, shuffle=True)
    val_loader   = _make_loader(s_val,   towers, batch_size=bs, shuffle=False)
    test_loader  = (_make_loader(s_test, towers, batch_size=bs, shuffle=False)
                    if s_test.entries else None)

    # ── Bridge ───────────────────────────────────────────────────────────────
    input_dims = resolve_input_dims(inputs, towers, stage_models)
    bmod   = importlib.import_module(stage_cfg["module"])
    bridge = getattr(bmod, stage_cfg["class"])(input_dims, **stage_cfg.get("args", {})).to(device)

    # ── Head stages ──────────────────────────────────────────────────────────
    head_stage_cfgs = [s for s in cfg_stages if s["type"] == "head"
                       and s.get("train_with") == name]
    head_models: dict[str, torch.nn.Module] = {}
    for hs in head_stage_cfgs:
        h_dim = get_out_dim(hs["input"], towers, {**stage_models, name: bridge})
        h_mod = importlib.import_module(hs.get("module", "v4.classes.heads.classifier"))
        h_cls = getattr(h_mod, hs.get("class", "ClassificationHead"))
        head_models[hs["name"]] = h_cls(h_dim, num_classes).to(device)

    primary_hs_cfg = next((hs for hs in head_stage_cfgs if not hs.get("bcd", False)), None)
    bcd_head_cfgs  = [hs for hs in head_stage_cfgs if hs.get("bcd", False)]

    if primary_hs_cfg is None:
        print(f"  WARNING: no primary head for stage {name!r}; skipping.", flush=True)
        return stage_models, {}

    primary_head = head_models[primary_hs_cfg["name"]]

    # ── Freeze prior stages ───────────────────────────────────────────────────
    for m in stage_models.values():
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()

    # ── Optimizer ────────────────────────────────────────────────────────────
    train_towers = stage_cfg.get("train_towers", False)
    opt_params = (
        ([p for t in towers.values() for p in t.parameters()] if train_towers else []) +
        list(bridge.parameters()) +
        [p for h in head_models.values() for p in h.parameters()]
    )
    opt = torch.optim.Adam(opt_params, lr=cfg["training"]["lr"])

    warmup_cfg = stage_cfg.get("warmup", {})
    wt         = 0 if is_bilateral else warmup_cfg.get("tower_epochs", 0)
    wf         = 0 if is_bilateral else warmup_cfg.get("fused_epochs",  0)
    bcd_prob   = cfg["training"].get("bcd_prob", 0.5)

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(epochs):
        bridge.train()
        for h in head_models.values():
            h.train()
        for t in towers.values():
            if train_towers:
                t.train()
            else:
                t.eval()

        if not is_bilateral:
            phase = phase_for_epoch(epoch, wt, wf)
            if hasattr(bridge, "set_phase"):
                bridge.set_phase(phase)
            for t in towers.values():
                if hasattr(t, "set_phase"):
                    t.set_phase(phase)
        else:
            phase = "fusion"

        total_loss = total_correct = total_n = 0

        for batch in train_loader:
            y = batch.get("label")
            if not torch.is_tensor(y):
                continue
            y_t = to_label_tensor(y, device)
            if y_t.numel() == 0:
                continue

            if is_bilateral:
                side_embs  = {
                    side: encode_embedding(src, batch, side, towers, stage_models, cfg_stages, device)
                    for side, src in inputs.items()
                }
                local_embs = {name: bridge(side_embs)}
            else:
                local_embs = {n: towers[n](batch[n].to(device)) for n in inputs
                              if n in batch and torch.is_tensor(batch[n])}
                if len(local_embs) != len(inputs):
                    continue
                local_embs[name] = bridge(list(local_embs[n] for n in inputs))

            head_logits = {
                hs["name"]: head_models[hs["name"]](local_embs[hs["input"]])
                for hs in head_stage_cfgs
                if hs["input"] in local_embs
            }

            if is_bilateral or phase == "fused_warmup":
                logits = head_logits.get(primary_hs_cfg["name"])
            elif phase == "tower_warmup" and bcd_head_cfgs:
                losses = [F.cross_entropy(head_logits[hs["name"]], y_t)
                          for hs in bcd_head_cfgs if hs["name"] in head_logits]
                if not losses:
                    continue
                loss = sum(losses) / len(losses)
                opt.zero_grad(); loss.backward(); opt.step()
                total_loss += loss.item() * len(y_t)
                total_n    += len(y_t)
                continue
            else:
                if bcd_head_cfgs and _random() < bcd_prob:
                    logits = head_logits.get(choice(bcd_head_cfgs)["name"])
                else:
                    logits = head_logits.get(primary_hs_cfg["name"])

            if logits is None:
                continue
            loss = F.cross_entropy(logits, y_t)
            opt.zero_grad(); loss.backward(); opt.step()
            total_correct += int((logits.argmax(1) == y_t).sum())
            total_loss    += loss.item() * len(y_t)
            total_n       += len(y_t)

        tr_loss = total_loss / total_n if total_n else nan
        tr_acc  = total_correct / total_n if total_n else nan

        y_v, p_v = collect_probs(bridge, primary_head, stage_cfg, towers,
                                  stage_models, cfg_stages, val_loader, device, num_classes)
        _, val_auc, _ = score_arrays(y_v, p_v, num_classes) if y_v.size else (nan, nan, nan)
        print(
            f"  fold{fold+1} [{name}] ep{epoch+1:03d}/{epochs} [{phase:14s}]"
            f"  loss={tr_loss:.4f}  acc={tr_acc:.3f}  val_auc={val_auc:.4f}",
            flush=True,
        )

    # ── Final eval ────────────────────────────────────────────────────────────
    y_val, p_val = collect_probs(bridge, primary_head, stage_cfg, towers,
                                  stage_models, cfg_stages, val_loader, device, num_classes)
    val_acc, val_auc, val_n = (score_arrays(y_val, p_val, num_classes)
                               if y_val.size else (nan, nan, nan))
    ext = compute_extended_metrics(y_val, p_val, num_classes) if y_val.size else {}

    val_threshold = 0.5
    if (cfg["training"].get("tune_binary_threshold")
            and num_classes == 2 and y_val.size >= 2):
        val_threshold = tune_binary_threshold(y_val, p_val[:, 1])

    test_auc = test_acc = test_n = nan
    if test_loader is not None:
        y_te, p_te = collect_probs(bridge, primary_head, stage_cfg, towers,
                                    stage_models, cfg_stages, test_loader, device, num_classes)
        test_acc, test_auc, test_n = (score_arrays(y_te, p_te, num_classes)
                                      if y_te.size else (nan, nan, nan))

    updated = dict(stage_models)
    updated[name] = bridge
    updated.update(head_models)

    metrics = {
        f"{name}_val_auc":       val_auc,
        f"{name}_val_acc":       val_acc,
        f"{name}_val_n":         val_n,
        f"{name}_val_kappa":     ext.get("kappa", nan),
        f"{name}_val_mcc":       ext.get("mcc", nan),
        f"{name}_val_f1":        ext.get("macro_f1", nan),
        f"{name}_val_threshold": val_threshold,
        f"{name}_test_auc":      test_auc,
        f"{name}_test_acc":      test_acc,
        f"{name}_test_n":        test_n,
    }
    return updated, metrics
