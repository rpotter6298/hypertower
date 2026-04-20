"""stages/helpers — shared utilities for stage runners."""
from __future__ import annotations

import torch


def get_out_dim(name: str, towers: dict, stage_models: dict) -> int:
    if name in towers:
        return towers[name].out_dim
    if name in stage_models:
        return stage_models[name].out_dim
    raise KeyError(f"No out_dim for input {name!r}")


def resolve_input_dims(inputs, towers: dict, stage_models: dict):
    """Return list[int] for list inputs, dict[str, int] for dict inputs."""
    if isinstance(inputs, list):
        return [get_out_dim(n, towers, stage_models) for n in inputs]
    return {k: get_out_dim(v, towers, stage_models) for k, v in inputs.items()}


def encode_embedding(
    name:         str,
    batch:        dict,
    side:         str | None,
    towers:       dict,
    stage_models: dict,
    cfg_stages:   list[dict],
    device,
) -> torch.Tensor:
    """Return the embedding for a named tower or prior frozen fusion stage.

    For bilateral batches, *side* selects which side dict entry to use.
    Fusion stages are re-encoded recursively from their own inputs.
    Runs under torch.no_grad() — only used for frozen passes.
    """
    if name in towers:
        t = batch[name]
        if side is not None and isinstance(t, dict):
            t = t[side]
        with torch.no_grad():
            return towers[name](t.to(device))

    s_cfg  = next(s for s in cfg_stages if s["name"] == name)
    model  = stage_models[name]
    inputs = s_cfg["inputs"]

    if isinstance(inputs, list):
        embeddings = [
            encode_embedding(n, batch, side, towers, stage_models, cfg_stages, device)
            for n in inputs
        ]
        with torch.no_grad():
            return model(embeddings)

    raise NotImplementedError(
        f"Stage {name!r} uses dict inputs and cannot be used as a fusion input."
    )


def phase_for_epoch(epoch: int, warmup_tower: int, warmup_fused: int) -> str:
    if epoch < warmup_tower:
        return "tower_warmup"
    if epoch < warmup_tower + warmup_fused:
        return "fused_warmup"
    return "main"
