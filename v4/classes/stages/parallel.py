"""stages/parallel — runs multiple same-type sub-stages in a shared epoch loop.

Used to train bilateral pairs (OD + OS) simultaneously rather than sequentially.
Sub-stages must all be the same type: either all 'warm' or all 'fusion'.
"""
from __future__ import annotations

import importlib
from random import choice, random as _random

import numpy as np
import torch
import torch.nn.functional as F

from v4.classes.dataset import LoaderShell, to_label_tensor
from v4.classes.metrics import score_arrays, compute_extended_metrics, tune_binary_threshold
from v4.classes.stages.helpers import (
    class_weights_from_shell, get_out_dim, resolve_input_dims, phase_for_epoch,
    head_compute_loss, head_to_probs, head_score, head_target_key,
)
from v4.classes.stages.fusion import collect_probs


def run(
    stage_cfg:        dict,
    cfg:              dict,
    towers:           dict,
    stage_models:     dict,
    data,
    split,
    label_filter,
    num_classes:      int,
    device,
    fold:             int,
    cfg_stages:       list[dict],
    _make_loader,
    _balanced_sampler,
) -> tuple[dict, dict, dict]:
    sub_cfgs  = stage_cfg["stages"]
    sub_types = {s["type"] for s in sub_cfgs}

    if sub_types == {"warm"}:
        return _parallel_warm(sub_cfgs, cfg, towers, stage_models, data, split,
                              label_filter, num_classes, device, fold,
                              cfg_stages, _make_loader, _balanced_sampler)
    elif sub_types == {"fusion"}:
        return _parallel_fusion(sub_cfgs, cfg, towers, stage_models, data, split,
                                label_filter, num_classes, device, fold,
                                cfg_stages, _make_loader)
    else:
        raise ValueError(
            f"parallel stage sub-stages must all be the same type (warm or fusion), "
            f"got {sub_types}"
        )


# ── Parallel warm ─────────────────────────────────────────────────────────────

def _parallel_warm(
    sub_cfgs, cfg, towers, stage_models, data, split,
    label_filter, num_classes, device, fold, cfg_stages, _make_loader, _balanced_sampler,
):
    bs = cfg["training"]["batch_size"]

    contexts = []
    for sc in sub_cfgs:
        tower_name   = sc["tower"]
        level        = sc["level"]
        shell_filter = sc.get("shell_filter", {})
        n_epochs     = sc.get("epochs", 0)

        s_train = data.build_shells(split.train, level=level, label_filter=label_filter,
                                    **shell_filter)
        loader  = _make_loader(s_train, {tower_name: towers[tower_name]},
                               batch_size=bs, shuffle=False,
                               sampler=_balanced_sampler(s_train))
        head_name = sc.get("head_name")
        if head_name:
            head_cfg = next((s for s in cfg_stages if s.get("name") == head_name), None)
            if head_cfg is None:
                raise ValueError(f"warm stage requested head_name={head_name!r}, but no such head exists")
            h_mod = importlib.import_module(head_cfg.get("module", "v4.classes.heads.classifier"))
            h_cls = getattr(h_mod, head_cfg.get("class", "ClassificationHead"))
            probe = stage_models.get(head_name)
            if probe is None:
                probe = h_cls(towers[tower_name].out_dim, num_classes, **head_cfg.get("args", {}))
            probe = probe.to(device)
        else:
            probe = torch.nn.Linear(towers[tower_name].out_dim, num_classes).to(device)
        opt   = torch.optim.Adam(
            list(towers[tower_name].parameters()) + list(probe.parameters()),
            lr=cfg["training"]["lr"],
        )
        cw = class_weights_from_shell(
            s_train, num_classes, device,
            enabled=cfg["training"].get("class_weighted", False),
        )
        contexts.append({
            "name": tower_name, "n_epochs": n_epochs,
            "loader": loader, "probe": probe, "opt": opt,
            "head_name": head_name, "class_weights": cw,
        })

    active = {c["name"] for c in contexts if c["n_epochs"] > 0}
    for name, t in towers.items():
        for p in t.parameters():
            p.requires_grad_(name in active)

    max_epochs = max((c["n_epochs"] for c in contexts), default=0)
    for epoch in range(max_epochs):
        for ctx in contexts:
            if epoch >= ctx["n_epochs"]:
                continue
            tower_name = ctx["name"]
            towers[tower_name].train()
            total_loss = total_correct = total_n = 0
            for batch in ctx["loader"]:
                y = batch.get("label")
                x = batch.get(tower_name)
                if not torch.is_tensor(y) or not torch.is_tensor(x):
                    continue
                y_t    = to_label_tensor(y, device)
                logits = ctx["probe"](towers[tower_name](x.to(device)))
                loss   = head_compute_loss(ctx["probe"], logits, batch, y_t,
                                           class_weights=ctx["class_weights"])
                ctx["opt"].zero_grad(); loss.backward(); ctx["opt"].step()
                total_loss    += loss.item() * len(y_t)
                if logits.dim() >= 2:
                    total_correct += int((logits.argmax(1) == y_t).sum())
                    is_class = True
                else:
                    is_class = False
                total_n       += len(y_t)
            if total_n:
                if is_class:
                    print(
                        f"  fold{fold+1} [warm/{tower_name}]"
                        f" ep{epoch+1:03d}/{ctx['n_epochs']}"
                        f"  loss={total_loss/total_n:.4f}  acc={total_correct/total_n:.3f}",
                        flush=True,
                    )
                else:
                    print(
                        f"  fold{fold+1} [warm/{tower_name}]"
                        f" ep{epoch+1:03d}/{ctx['n_epochs']}"
                        f"  loss={total_loss/total_n:.4f}",
                        flush=True,
                    )

    for t in towers.values():
        for p in t.parameters():
            p.requires_grad_(True)

    updated = dict(stage_models)
    for ctx in contexts:
        if ctx.get("head_name"):
            updated[ctx["head_name"]] = ctx["probe"]

    return updated, {}, {}


# ── Parallel fusion ───────────────────────────────────────────────────────────

def _parallel_fusion(
    sub_cfgs, cfg, towers, stage_models, data, split,
    label_filter, num_classes, device, fold, cfg_stages, _make_loader,
):
    nan      = float("nan")
    bs       = cfg["training"]["batch_size"]
    bcd_prob        = cfg["training"].get("bcd_prob", 0.5)
    tower_loss_mode = cfg["training"].get("tower_loss_mode", "bcd")
    if tower_loss_mode not in ("bcd", "all_losses"):
        raise ValueError(f"tower_loss_mode must be 'bcd' or 'all_losses'; got {tower_loss_mode!r}")

    # Freeze all prior stage models once, before building any bridges.
    for m in stage_models.values():
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()

    # ── Per-sub-stage setup ───────────────────────────────────────────────────
    contexts = []
    for sc in sub_cfgs:
        name       = sc["name"]
        level      = sc["level"]
        inputs     = sc["inputs"]
        epochs     = sc["epochs"]
        eye_filter = sc.get("eye_filter", None)

        s_train = data.build_shells(split.train, level=level, label_filter=label_filter,
                                    eye_filter=eye_filter)
        s_val   = data.build_shells(split.val,   level=level, label_filter=label_filter,
                                    eye_filter=eye_filter)
        s_test  = (data.build_shells(split.test, level=level, label_filter=label_filter,
                                     eye_filter=eye_filter)
                   if split.test is not None else LoaderShell(entries=[]))

        if not s_val.entries:
            print(f"  fold{fold+1}: no val samples for stage {name!r}, skipping.", flush=True)
            continue

        input_dims = resolve_input_dims(inputs, towers, stage_models)
        bmod       = importlib.import_module(sc["module"])
        bridge     = getattr(bmod, sc["class"])(input_dims, **sc.get("args", {})).to(device)

        # Head stages for this sub-stage.
        head_stage_cfgs = [s for s in cfg_stages if s["type"] == "head"
                           and s.get("train_with") == name]
        head_models: dict[str, torch.nn.Module] = {}
        for hs in head_stage_cfgs:
            h_dim = get_out_dim(hs["input"], towers, {**stage_models, name: bridge})
            h_mod = importlib.import_module(hs.get("module", "v4.classes.heads.classifier"))
            h_cls = getattr(h_mod, hs.get("class", "ClassificationHead"))
            existing = stage_models.get(hs["name"])
            head_models[hs["name"]] = (
                existing.to(device) if existing is not None
                else h_cls(h_dim, num_classes, **hs.get("args", {})).to(device)
            )
        for h in head_models.values():
            for p in h.parameters():
                p.requires_grad_(True)

        primary_hs_cfg = next((hs for hs in head_stage_cfgs if not hs.get("bcd", False)), None)
        bcd_head_cfgs  = [hs for hs in head_stage_cfgs if hs.get("bcd", False)]

        if primary_hs_cfg is None:
            print(f"  WARNING: no primary head for stage {name!r}; skipping.", flush=True)
            continue

        primary_head = head_models[primary_hs_cfg["name"]]

        train_towers  = sc.get("train_towers", False)
        direct_inputs = inputs if isinstance(inputs, list) else list(inputs.values())
        tower_params  = ([p for n in direct_inputs if n in towers
                          for p in towers[n].parameters()]
                         if train_towers else [])
        opt_params    = (tower_params + list(bridge.parameters()) +
                         [p for h in head_models.values() for p in h.parameters()])
        opt = torch.optim.Adam(opt_params, lr=cfg["training"]["lr"])

        warmup_cfg = sc.get("warmup", {})
        wt         = warmup_cfg.get("tower_epochs", 0)
        wf         = warmup_cfg.get("fused_epochs", 0)

        train_loader = _make_loader(s_train, towers, batch_size=bs, shuffle=True)
        val_loader   = _make_loader(s_val,   towers, batch_size=bs, shuffle=False)
        test_loader  = (_make_loader(s_test, towers, batch_size=bs, shuffle=False)
                        if s_test.entries else None)

        cw = class_weights_from_shell(
            s_train, num_classes, device,
            enabled=cfg["training"].get("class_weighted", False),
        )
        contexts.append({
            "name": name, "inputs": inputs, "epochs": epochs,
            "bridge": bridge, "head_models": head_models,
            "primary_head": primary_head, "primary_hs_cfg": primary_hs_cfg,
            "bcd_head_cfgs": bcd_head_cfgs,
            "opt": opt, "wt": wt, "wf": wf,
            "train_towers": train_towers, "direct_inputs": direct_inputs,
            "train_loader": train_loader, "val_loader": val_loader,
            "test_loader": test_loader, "sc": sc,
            "class_weights": cw,
        })

    if not contexts:
        return stage_models, {}, {}

    max_epochs = max(c["epochs"] for c in contexts)

    # ── Shared epoch loop ─────────────────────────────────────────────────────
    for epoch in range(max_epochs):
        for ctx in contexts:
            if epoch >= ctx["epochs"]:
                continue

            name   = ctx["name"]
            bridge = ctx["bridge"]
            inputs = ctx["inputs"]
            wt, wf = ctx["wt"], ctx["wf"]
            phase  = phase_for_epoch(epoch, wt, wf)

            bridge.train()
            for h in ctx["head_models"].values():
                h.train()
            if ctx["train_towers"]:
                for n in ctx["direct_inputs"]:
                    if n in towers:
                        towers[n].train()
            else:
                for n in ctx["direct_inputs"]:
                    if n in towers:
                        towers[n].eval()

            if hasattr(bridge, "set_phase"):
                bridge.set_phase(phase)

            total_loss = total_correct = total_n = 0

            for batch in ctx["train_loader"]:
                y = batch.get("label")
                if not torch.is_tensor(y):
                    continue
                y_t = to_label_tensor(y, device)
                if y_t.numel() == 0:
                    continue

                local_embs = {n: towers[n](batch[n].to(device)) for n in inputs
                              if n in batch and torch.is_tensor(batch[n])}
                if len(local_embs) != len(inputs):
                    continue
                local_embs[name] = bridge(list(local_embs[n] for n in inputs))

                head_logits = {
                    hs["name"]: ctx["head_models"][hs["name"]](local_embs[hs["input"]])
                    for hs in ([ctx["primary_hs_cfg"]] + ctx["bcd_head_cfgs"])
                    if hs["input"] in local_embs
                }

                if phase == "fused_warmup":
                    logits = head_logits.get(ctx["primary_hs_cfg"]["name"])
                    chosen_head = ctx["head_models"].get(ctx["primary_hs_cfg"]["name"])
                elif phase == "tower_warmup" and ctx["bcd_head_cfgs"]:
                    losses = [
                        head_compute_loss(ctx["head_models"][hs["name"]],
                                          head_logits[hs["name"]], batch, y_t,
                                          class_weights=ctx["class_weights"])
                        for hs in ctx["bcd_head_cfgs"] if hs["name"] in head_logits
                    ]
                    if not losses:
                        continue
                    loss = sum(losses) / len(losses)
                    if hasattr(bridge, "modify_loss"):
                        loss = bridge.modify_loss(loss)
                    ctx["opt"].zero_grad(); loss.backward(); ctx["opt"].step()
                    total_loss += loss.item() * len(y_t)
                    total_n    += len(y_t)
                    continue
                elif tower_loss_mode == "all_losses" and ctx["bcd_head_cfgs"]:
                    all_head_names = ([ctx["primary_hs_cfg"]["name"]]
                                      + [hs["name"] for hs in ctx["bcd_head_cfgs"]])
                    losses = [
                        head_compute_loss(ctx["head_models"][n], head_logits[n],
                                          batch, y_t, class_weights=ctx["class_weights"])
                        for n in all_head_names if n in head_logits
                    ]
                    if not losses:
                        continue
                    loss = sum(losses)
                    if hasattr(bridge, "modify_loss"):
                        loss = bridge.modify_loss(loss)
                    ctx["opt"].zero_grad(); loss.backward(); ctx["opt"].step()
                    total_loss += loss.item() * len(y_t)
                    total_n    += len(y_t)
                    continue
                else:
                    if ctx["bcd_head_cfgs"] and _random() < bcd_prob:
                        chosen_hs = choice(ctx["bcd_head_cfgs"])
                    else:
                        chosen_hs = ctx["primary_hs_cfg"]
                    logits      = head_logits.get(chosen_hs["name"])
                    chosen_head = ctx["head_models"].get(chosen_hs["name"])

                if logits is None:
                    continue
                loss = head_compute_loss(chosen_head, logits, batch, y_t,
                                         class_weights=ctx["class_weights"])
                if hasattr(bridge, "modify_loss"):
                    loss = bridge.modify_loss(loss)
                ctx["opt"].zero_grad(); loss.backward(); ctx["opt"].step()
                if logits.dim() >= 2:
                    total_correct += int((logits.argmax(1) == y_t).sum())
                total_loss    += loss.item() * len(y_t)
                total_n       += len(y_t)

            tr_loss = total_loss / total_n if total_n else nan
            tr_acc  = total_correct / total_n if total_n else nan

            y_v, p_v, _, _ = collect_probs(bridge, ctx["primary_head"], ctx["sc"], towers,
                                           stage_models, cfg_stages, ctx["val_loader"],
                                           device, num_classes)
            epoch_scores = head_score(ctx["primary_head"], y_v, p_v, num_classes)
            val_metric   = epoch_scores.get("primary", nan)
            metric_name  = epoch_scores.get("primary_name", "auc")
            is_class     = not hasattr(ctx["primary_head"], "target_key")
            if is_class:
                print(
                    f"  fold{fold+1} [{name}] ep{epoch+1:03d}/{ctx['epochs']} [{phase:14s}]"
                    f"  loss={tr_loss:.4f}  acc={tr_acc:.3f}  val_{metric_name}={val_metric:.4f}",
                    flush=True,
                )
            else:
                print(
                    f"  fold{fold+1} [{name}] ep{epoch+1:03d}/{ctx['epochs']} [{phase:14s}]"
                    f"  loss={tr_loss:.4f}  val_{metric_name}={val_metric:.4f}",
                    flush=True,
                )

    # ── Final eval + collect results ──────────────────────────────────────────
    updated    = dict(stage_models)
    all_metrics: dict = {}
    all_preds:   dict = {}

    for ctx in contexts:
        name         = ctx["name"]
        bridge       = ctx["bridge"]
        primary_head = ctx["primary_head"]

        updated[name] = bridge
        updated.update(ctx["head_models"])

        y_val, p_val, ids_val, z_val = collect_probs(bridge, primary_head, ctx["sc"], towers,
                                                      stage_models, cfg_stages, ctx["val_loader"],
                                                      device, num_classes)
        val_scores = head_score(primary_head, y_val, p_val, num_classes)

        val_threshold = 0.5
        if (cfg["training"].get("tune_binary_threshold")
                and num_classes == 2 and y_val.size >= 2
                and p_val.ndim == 2 and p_val.shape[1] == 2):
            val_threshold = tune_binary_threshold(y_val, p_val[:, 1])

        y_te = p_te = ids_te = z_te = None
        test_scores: dict = {}
        if ctx["test_loader"] is not None:
            y_te, p_te, ids_te, z_te = collect_probs(bridge, primary_head, ctx["sc"], towers,
                                                      stage_models, cfg_stages, ctx["test_loader"],
                                                      device, num_classes)
            test_scores = head_score(primary_head, y_te, p_te, num_classes)

        per_stage_metrics: dict = {f"{name}_val_threshold": val_threshold}
        for k, v in val_scores.items():
            if k == "primary_name":
                continue
            if isinstance(v, (int, float)):
                per_stage_metrics[f"{name}_val_{k}"] = float(v)
        per_stage_metrics[f"{name}_val_primary_name"] = val_scores.get("primary_name", "auc")
        for k, v in test_scores.items():
            if k == "primary_name":
                continue
            if isinstance(v, (int, float)):
                per_stage_metrics[f"{name}_test_{k}"] = float(v)
        per_stage_metrics[f"{name}_test_primary_name"] = test_scores.get(
            "primary_name", per_stage_metrics[f"{name}_val_primary_name"])
        all_metrics.update(per_stage_metrics)
        all_preds[name] = {
            "val_y": y_val, "val_p": p_val, "val_ids": ids_val, "val_z": z_val,
            "test_y": y_te, "test_p": p_te, "test_ids": ids_te, "test_z": z_te,
        }

    return updated, all_metrics, all_preds
