"""stages/warm — warm stage runner: pre-trains a single tower with a temporary probe."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from v4.classes.dataset import to_label_tensor
from v4.classes.stages.helpers import phase_for_epoch


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
) -> None:
    """Pre-train one tower using a temporary linear probe (probe discarded after)."""
    tower_name = stage_cfg["tower"]
    n_epochs   = stage_cfg.get("epochs", 0)
    level      = stage_cfg["level"]

    if n_epochs == 0:
        return

    s_train = data.build_shells(split.train, level=level, label_filter=label_filter)
    bs      = cfg["training"]["batch_size"]
    loader  = _make_loader(
        s_train, {tower_name: towers[tower_name]},
        batch_size=bs, shuffle=False,
        sampler=_balanced_sampler(s_train),
    )

    for n, t in towers.items():
        for p in t.parameters():
            p.requires_grad_(n == tower_name)

    probe = torch.nn.Linear(towers[tower_name].out_dim, num_classes).to(device)
    opt   = torch.optim.Adam(
        list(towers[tower_name].parameters()) + list(probe.parameters()),
        lr=cfg["training"]["lr"],
    )

    towers[tower_name].train()
    for epoch in range(n_epochs):
        total_loss = total_correct = total_n = 0
        for batch in loader:
            y = batch.get("label")
            x = batch.get(tower_name)
            if not torch.is_tensor(y) or not torch.is_tensor(x):
                continue
            y_t    = to_label_tensor(y, device)
            logits = probe(towers[tower_name](x.to(device)))
            loss   = F.cross_entropy(logits, y_t)
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
