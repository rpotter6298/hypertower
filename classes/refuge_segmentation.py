"""REFUGE optic disc / cup segmentation utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from classes.refuge_preprocessing import RefugePreprocessing, RefugeSample


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _load_rgb(path: Path) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _load_mask_array(path: Path) -> np.ndarray:
    mask_img = Image.open(path).convert("L")
    mask = np.array(mask_img, dtype=np.float32)
    # REFUGE masks encode disc/cup with different intensities; treat any
    # positive value as disc for coarse localisation.
    mask = np.where(mask > 0, 1.0, 0.0)
    return mask


@dataclass
class RefugeSegmentationSample:
    sample: RefugeSample
    image_path: Path
    mask_path: Path


class RefugeSegmentationDataset(Dataset):
    """Simple segmentation dataset returning tensors."""

    def __init__(
        self,
        samples: Sequence[RefugeSegmentationSample],
        image_size: int = 512,
    ) -> None:
        self.samples = list(samples)
        self.image_size = image_size
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.samples[idx]
        image = _load_rgb(rec.image_path)
        mask_arr = _load_mask_array(rec.mask_path)

        if self.image_size is not None:
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
            mask_img = Image.fromarray(mask_arr).resize(
                (self.image_size, self.image_size), Image.NEAREST
            )
            mask_arr = np.array(mask_img, dtype=np.float32)

        image_tensor = self.to_tensor(image)
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)  # [1,H,W]
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "sample_id": rec.sample.sample_id,
        }


# ---------------------------------------------------------------------------
# Model definition (lightweight U-Net)
# ---------------------------------------------------------------------------


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        self.out = nn.Conv2d(base_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.out(d1)


# ---------------------------------------------------------------------------
# Segmentation manager
# ---------------------------------------------------------------------------


class RefugeSegmentation:
    """Train and run coarse-to-fine OD/OC segmentation for REFUGE."""

    def __init__(
        self,
        preprocessing: RefugePreprocessing,
        model: Optional[nn.Module] = None,
    ) -> None:
        self.preprocessing = preprocessing
        self.model = model or UNet()
        self.train_dataset: Optional[RefugeSegmentationDataset] = None
        self.val_dataset: Optional[RefugeSegmentationDataset] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None

    # ------------------------------------------------------------------
    def build_datasets(
        self,
        image_size: int = 512,
        batch_size: int = 8,
        num_workers: int = 4,
    ) -> None:
        manifest = self.preprocessing.build_manifest()

        train_samples: List[RefugeSegmentationSample] = []
        val_samples: List[RefugeSegmentationSample] = []

        for sample in manifest:
            if not sample.mask_path or not sample.mask_path.exists():
                continue
            rec = RefugeSegmentationSample(sample=sample, image_path=sample.image_path, mask_path=sample.mask_path)
            if sample.split == "train":
                train_samples.append(rec)
            elif sample.split in {"val", "validation"}:
                val_samples.append(rec)

        if not val_samples:
            # Fall back to using a subset of training data for validation
            split = max(1, int(0.1 * len(train_samples)))
            val_samples = train_samples[:split]
            train_samples = train_samples[split:]

        self.train_dataset = RefugeSegmentationDataset(train_samples, image_size=image_size)
        self.val_dataset = RefugeSegmentationDataset(val_samples, image_size=image_size)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    # ------------------------------------------------------------------
    def train(
        self,
        epochs: int = 40,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: Optional[str] = None,
        checkpoint_dir: Optional[Path] = None,
    ) -> Dict[str, float]:
        if self.train_loader is None or self.val_loader is None:
            self.build_datasets()

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        best_dice = 0.0
        history: Dict[str, float] = {}

        for epoch in range(1, epochs + 1):
            print(f"[Seg] Processing epoch {epoch}/{epochs}")
            self.model.train()
            running_loss = 0.0
            for batch in self.train_loader:  # type: ignore[arg-type]
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                optimizer.zero_grad()
                logits = self.model(images)
                loss = criterion(logits, masks)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)

            train_loss = running_loss / len(self.train_loader.dataset)  # type: ignore[arg-type]
            val_metrics = self.evaluate(device=device)
            history[f"epoch_{epoch}_loss"] = train_loss
            history[f"epoch_{epoch}_dice"] = val_metrics.get("dice", float("nan"))

            if val_metrics.get("dice", 0.0) > best_dice:
                best_dice = val_metrics["dice"]
                if checkpoint_dir is not None:
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(self.model.state_dict(), checkpoint_dir / "refuge_segmentation_best.pt")

        return {"best_dice": best_dice, **history}

    # ------------------------------------------------------------------
    def evaluate(self, split: str = "val", device: Optional[str] = None) -> Dict[str, float]:
        if split != "val":
            raise ValueError("Only validation split supported currently")
        if self.val_loader is None:
            self.build_datasets()

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        dices: List[float] = []
        criterion = nn.BCEWithLogitsLoss()
        losses: List[float] = []

        with torch.no_grad():
            for batch in self.val_loader:  # type: ignore[arg-type]
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                logits = self.model(images)
                loss = criterion(logits, masks)
                losses.append(loss.item() * images.size(0))
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                dice = self._dice_coefficient(preds, masks)
                dices.extend(dice)

        mean_dice = float(np.mean(dices)) if dices else 0.0
        mean_loss = float(np.sum(losses) / len(self.val_loader.dataset))  # type: ignore[arg-type]
        return {"dice": mean_dice, "loss": mean_loss}

    # ------------------------------------------------------------------
    def predict_mask(self, sample: RefugeSample, device: Optional[str] = None) -> torch.Tensor:
        if self.train_dataset is None:
            self.build_datasets()
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        image = _load_rgb(sample.image_path)
        original_size = image.size  # (width, height)
        image_resized = image.resize((self.train_dataset.image_size, self.train_dataset.image_size), Image.BILINEAR)  # type: ignore[union-attr]
        tensor = transforms.ToTensor()(image_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = self.model(tensor)
            mask_resized = torch.sigmoid(logits)[0, 0]

        mask_np = mask_resized.cpu().numpy()
        mask_np = (mask_np > 0.5).astype(np.float32)
        mask_img = Image.fromarray(mask_np)
        mask_img = mask_img.resize(original_size, Image.NEAREST)
        return torch.from_numpy(np.array(mask_img, dtype=np.float32))

    def infer_disc_geometry(
        self,
        sample: RefugeSample,
        scale: float = 2.5,
    ) -> Dict[str, float]:
        if sample.mask_path and sample.mask_path.exists():
            mask = _load_mask_array(sample.mask_path)
        else:
            mask = self.predict_mask(sample).numpy()

        coords = np.argwhere(mask > 0.5)
        if coords.size == 0:
            raise RuntimeError(f"Unable to locate disc for sample {sample.sample_id}")

        ys, xs = coords[:, 0], coords[:, 1]
        centre_x = float(xs.mean())
        centre_y = float(ys.mean())
        width = float(xs.max() - xs.min())
        height = float(ys.max() - ys.min())
        diameter = max(width, height)
        radius = diameter / 2.0
        crop_radius = radius * scale
        return {
            "centre_x": centre_x,
            "centre_y": centre_y,
            "radius": radius,
            "crop_radius": crop_radius,
            "crop_size": crop_radius * 2.0,
        }

    def batch_crops(
        self,
        samples: Iterable[RefugeSample],
        scale: float = 2.5,
        output_dir: Optional[Path] = None,
        size: int = 256,
    ) -> Dict[str, Path]:
        output_paths: Dict[str, Path] = {}
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        for sample in samples:
            geom = self.infer_disc_geometry(sample, scale=scale)
            image = _load_rgb(sample.image_path)
            cx, cy = geom["centre_x"], geom["centre_y"]
            r = geom["crop_radius"]
            left = max(0.0, cx - r)
            upper = max(0.0, cy - r)
            right = min(image.width, cx + r)
            lower = min(image.height, cy + r)
            crop = image.crop((left, upper, right, lower)).resize((size, size), Image.BILINEAR)
            if output_dir is not None:
                out_path = output_dir / f"{sample.sample_id}_crop.png"
                crop.save(out_path)
                output_paths[sample.sample_id] = out_path
        return output_paths

    # ------------------------------------------------------------------
    @staticmethod
    def _dice_coefficient(preds: torch.Tensor, targets: torch.Tensor) -> List[float]:
        eps = 1e-6
        dices = []
        preds = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        for p, t in zip(preds, targets):
            intersection = float((p * t).sum().item())
            union = float(p.sum().item() + t.sum().item())
            dice = (2.0 * intersection + eps) / (union + eps)
            dices.append(dice)
        return dices
