"""stages/warm — warm stage runner: pre-trains a single tower and optional real head."""
from __future__ import annotations

import importlib

import torch
import torch.nn.functional as F

from v4.classes.dataset import to_label_tensor
from v4.classes.stages.helpers import class_weights_from_shell


def run(
    stage_cfg:   dict,
    towers:      dict,
    data,
    split,
    label_filter,
    cfg:         dict,
    num_classes: int,
    device,
    fold:        int,
    _make_loader,
    _balanced_sampler,
    stage_models: dict | None = None,
    cfg_stages:   list[dict] | None = None,
) -> dict:
    """Pre-train one tower.

    If ``head_name`` is set on the stage config, train that real downstream head
    and return it in ``stage_models``. Otherwise, fall back to a temporary linear
    probe for backward-compatible representation warmup.
    """
    tower_name   = stage_cfg["tower"]
    n_epochs     = stage_cfg.get("epochs", 0)
    level        = stage_cfg["level"]
    shell_filter = stage_cfg.get("shell_filter", {})
    stage_models = dict(stage_models or {})
    cfg_stages   = list(cfg_stages or [])

    if n_epochs == 0:
        return stage_models

    s_train = data.build_shells(split.train, level=level, label_filter=label_filter,
                                **shell_filter)
    bs      = cfg["training"]["batch_size"]
    loader  = _make_loader(
        s_train, {tower_name: towers[tower_name]},
        batch_size=bs, shuffle=False,
        sampler=_balanced_sampler(s_train),
    )

    for n, t in towers.items():
        for p in t.parameters():
            p.requires_grad_(n == tower_name)

    head_name = stage_cfg.get("head_name")
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

    towers[tower_name].train()
    probe.train()
    for epoch in range(n_epochs):
        total_loss = total_correct = total_n = 0
        for batch in loader:
            y = batch.get("label")
            x = batch.get(tower_name)
            if not torch.is_tensor(y) or not torch.is_tensor(x):
                continue
            y_t    = to_label_tensor(y, device)
            logits = probe(towers[tower_name](x.to(device)))
            loss   = F.cross_entropy(logits, y_t, weight=cw)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss    += loss.item() * len(y_t)
            total_correct += int((logits.argmax(1) == y_t).sum())
            total_n       += len(y_t)
        print(
            f"  fold{fold+1} [warm/{tower_name}] ep{epoch+1:03d}/{n_epochs}"
            f"  loss={total_loss/total_n:.4f}  acc={total_correct/total_n:.3f}",
            flush=True,
        )

    for t in towers.values():
        for p in t.parameters():
            p.requires_grad_(True)

    if head_name:
        stage_models[head_name] = probe
    return stage_models
