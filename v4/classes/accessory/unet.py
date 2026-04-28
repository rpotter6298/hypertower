"""unet — REFUGE-trained UNet wrapper for v4.

Lean accessory module: model definition + a thin segmenter wrapper that handles
weight loading, preprocessing, inference, and fine-tuning.

Used by:
  - GeometrySegEncoder tower (produces disc/cup seg maps as CNN input)
  - (future) ImageEncoder cropping (locates disc bbox for image cropping)

The segmenter is intentionally domain-agnostic: it takes PIL images in and
returns binary (disc, cup) numpy masks.  Fine-tuning consumes any DataLoader
yielding (image_tensor, mask_tensor) pairs — mask preparation (parsing GT
contour files, etc.) lives in the consumer.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PIL.Image import Resampling
from torch import nn
from torchvision import transforms


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32, out_channels: int = 2):
        super().__init__()
        self.enc1 = self._block(in_channels, base_channels)
        self.enc2 = self._block(base_channels, base_channels * 2)
        self.enc3 = self._block(base_channels * 2, base_channels * 4)
        self.enc4 = self._block(base_channels * 4, base_channels * 8)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = self._block(base_channels * 8, base_channels * 16)

        self.up4  = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.dec4 = self._block(base_channels * 16, base_channels * 8)
        self.up3  = nn.ConvTranspose2d(base_channels * 8,  base_channels * 4, 2, stride=2)
        self.dec3 = self._block(base_channels * 8, base_channels * 4)
        self.up2  = nn.ConvTranspose2d(base_channels * 4,  base_channels * 2, 2, stride=2)
        self.dec2 = self._block(base_channels * 4, base_channels * 2)
        self.up1  = nn.ConvTranspose2d(base_channels * 2,  base_channels,     2, stride=2)
        self.dec1 = self._block(base_channels * 2, base_channels)

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


# ---------------------------------------------------------------------------
# Segmenter wrapper
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


class UNetSegmenter:
    """Wraps a UNet with preprocessing, weight loading, inference, and fine-tuning.

    Parameters
    ----------
    target_size  : square resolution UNet operates at (default 512)
    normalize    : "per_image" | "imagenet" | "none"
    device       : torch device string; defaults to cuda if available
    """

    def __init__(
        self,
        *,
        target_size: int = 512,
        normalize:   str = "per_image",
        device:      str | torch.device | None = None,
        in_channels: int = 3,
        base_channels: int = 32,
        out_channels: int = 2,
    ):
        self.target_size = target_size
        self.normalize   = normalize
        self.device      = (
            torch.device(device) if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = UNet(in_channels, base_channels, out_channels).to(self.device)
        self._to_tensor = transforms.ToTensor()

    # ── lifecycle ────────────────────────────────────────────────────────────

    def to(self, device: str | torch.device) -> "UNetSegmenter":
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    def load_weights(self, path: str | Path) -> "UNetSegmenter":
        """Load a UNet checkpoint (raw state_dict or {'model': state_dict})."""
        state = torch.load(Path(path), map_location=self.device, weights_only=False)
        sd    = state["model"] if isinstance(state, dict) and "model" in state else state
        self.model.load_state_dict(sd)
        self.model.eval()
        return self

    # ── preprocessing ────────────────────────────────────────────────────────

    def _normalize_tensor(self, t: torch.Tensor) -> torch.Tensor:
        if self.normalize == "per_image":
            mean = t.mean(dim=(-2, -1), keepdim=True)
            std  = t.std (dim=(-2, -1), keepdim=True).clamp(min=1e-6)
            return (t - mean) / std
        if self.normalize == "imagenet":
            mean = torch.tensor(_IMAGENET_MEAN, device=t.device).view(-1, 1, 1)
            std  = torch.tensor(_IMAGENET_STD,  device=t.device).view(-1, 1, 1)
            return (t - mean) / std
        return t

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """PIL image → normalized (C, H, W) tensor on segmenter device."""
        resized = image.convert("RGB").resize(
            (self.target_size, self.target_size), Resampling.BILINEAR
        )
        return self._normalize_tensor(self._to_tensor(resized).to(self.device))

    # ── inference ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        image:     Image.Image,
        *,
        threshold: float = 0.5,
        tta:       bool  = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Single image → (disc_mask, cup_mask) binary uint8 arrays at target_size.

        cup_mask is restricted to disc area (cup ⊆ disc).
        """
        self.model.eval()
        x      = self.preprocess(image).unsqueeze(0)
        logits = self.model(x)
        if tta:
            log_h  = torch.flip(self.model(torch.flip(x, dims=[3])), dims=[3])
            log_v  = torch.flip(self.model(torch.flip(x, dims=[2])), dims=[2])
            logits = (logits + log_h + log_v) / 3.0
        probs = torch.sigmoid(logits)[0].cpu().numpy()
        disc  = (probs[0] > threshold).astype(np.uint8)
        cup   = ((probs[1] > threshold) & (disc > 0)).astype(np.uint8)
        return disc, cup

    # ── fine-tuning ──────────────────────────────────────────────────────────

    def finetune(
        self,
        dataloader,
        *,
        epochs:     int   = 10,
        lr:         float = 1e-5,
        log_prefix: str   = "[UNetSegmenter]",
    ) -> "UNetSegmenter":
        """Fine-tune on (image_tensor, mask_tensor) pairs.

        image_tensor : (B, C, H, W) — already preprocessed (normalized)
        mask_tensor  : (B, 2, H, W) float32 — channel 0 disc, channel 1 cup
        """
        import time
        opt  = torch.optim.Adam(self.model.parameters(), lr=lr)
        crit = nn.BCEWithLogitsLoss()
        for ep in range(1, epochs + 1):
            self.model.train()
            running, n_batches, t0 = 0.0, 0, time.time()
            for img, mask in dataloader:
                img, mask = img.to(self.device), mask.to(self.device)
                opt.zero_grad()
                loss = crit(self.model(img), mask)
                loss.backward()
                opt.step()
                running   += float(loss.item())
                n_batches += 1
            avg = running / max(n_batches, 1)
            print(
                f"  {log_prefix} ep{ep:03d}/{epochs:03d}  loss={avg:.4f}  "
                f"({time.time() - t0:.1f}s)",
                flush=True,
            )
        self.model.eval()
        return self
