"""Memory probe: EfficientNetV2-M at 480x480, bilateral forward+backward, AMP bf16.

Goal: confirm bs=8 fits in 16 GB on the available GPU before committing to the
full sensitivity experiment.

Mimics the bilateral training step (image tower run twice on OD + OS with shared
weights, plus a small downstream head + CE loss + Adam step). The clinical tower
and L1 bridge are omitted; their memory footprint is negligible against V2-M
activations. Synthetic inputs of the correct shape — no v4 dataset needed.

Run: python v4/scripts/experiments/sensitivity/probe_v2m_480_amp.py
Expected output: GPU name, peak memory at each phase, fit/OOM verdict.
"""
from __future__ import annotations

import sys
import torch
from torch import nn
from torchvision.models import efficientnet_v2_m


BATCH = 8
RES = 480
DTYPE = torch.bfloat16


def fmt_gb(bytes_):
    return f"{bytes_ / 1024**3:.2f} GB"


def main():
    if not torch.cuda.is_available():
        print("No CUDA/ROCm device available; probe requires a GPU.", file=sys.stderr)
        sys.exit(2)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_total = torch.cuda.get_device_properties(0).total_memory
    print(f"GPU: {gpu_name}   total VRAM: {fmt_gb(gpu_total)}")
    print(f"Config: V2-M, bs={BATCH}, res={RES}, bilateral 2x forward, AMP={DTYPE}")
    print("-" * 70)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    backbone = efficientnet_v2_m(weights=None).to(device)
    feat_dim = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()
    head = nn.Sequential(
        nn.LayerNorm(feat_dim),
        nn.Linear(feat_dim, 256),
        nn.GELU(),
        nn.Linear(256, 2),
    ).to(device)
    opt = torch.optim.Adam(
        list(backbone.parameters()) + list(head.parameters()),
        lr=1e-4,
    )

    print(f"After model + optimizer load: allocated={fmt_gb(torch.cuda.memory_allocated())}  "
          f"peak={fmt_gb(torch.cuda.max_memory_allocated())}")

    x_od = torch.randn(BATCH, 3, RES, RES, device=device)
    x_os = torch.randn(BATCH, 3, RES, RES, device=device)
    y = torch.randint(0, 2, (BATCH,), device=device)

    print(f"After synthetic inputs:        allocated={fmt_gb(torch.cuda.memory_allocated())}  "
          f"peak={fmt_gb(torch.cuda.max_memory_allocated())}")

    try:
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=DTYPE):
            z_od = backbone(x_od)
            z_os = backbone(x_os)
            z = z_od + z_os
            logits = head(z)
            loss = nn.functional.cross_entropy(logits, y)

        print(f"After forward:                 allocated={fmt_gb(torch.cuda.memory_allocated())}  "
              f"peak={fmt_gb(torch.cuda.max_memory_allocated())}")

        loss.backward()
        print(f"After backward:                allocated={fmt_gb(torch.cuda.memory_allocated())}  "
              f"peak={fmt_gb(torch.cuda.max_memory_allocated())}")

        opt.step()
        print(f"After optimizer step:          allocated={fmt_gb(torch.cuda.memory_allocated())}  "
              f"peak={fmt_gb(torch.cuda.max_memory_allocated())}")

        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        headroom = gpu_total - peak
        print("-" * 70)
        print(f"VERDICT: FIT  | peak={fmt_gb(peak)} of {fmt_gb(gpu_total)}  "
              f"headroom={fmt_gb(headroom)} ({100 * headroom / gpu_total:.1f}%)")
        print(f"Loss value: {loss.item():.4f}")

    except torch.cuda.OutOfMemoryError as e:
        peak = torch.cuda.max_memory_allocated()
        print("-" * 70)
        print(f"VERDICT: OOM  | peak before OOM={fmt_gb(peak)} of {fmt_gb(gpu_total)}")
        print(f"OOM details: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
