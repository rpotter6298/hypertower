"""U-Net based optic disc/cup segmenter for REFUGE + Papila."""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from PIL.Image import Resampling
from skimage import measure
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


@dataclass
class ManifestEntry:
    sample_id: str
    dataset: str
    image_path: Path
    annotation_disc: Path
    annotation_cup: Path
    annotation_type_disc: str
    annotation_type_cup: str
    split: str  # train / holdout / etc.


class UNet(nn.Module):
    def __init__(
        self, in_channels: int = 3, base_channels: int = 32, out_channels: int = 2
    ):
        super().__init__()
        self.enc1 = self._block(in_channels, base_channels)
        self.enc2 = self._block(base_channels, base_channels * 2)
        self.enc3 = self._block(base_channels * 2, base_channels * 4)
        self.enc4 = self._block(base_channels * 4, base_channels * 8)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = self._block(base_channels * 8, base_channels * 16)

        self.up4 = nn.ConvTranspose2d(
            base_channels * 16, base_channels * 8, 2, stride=2
        )
        self.dec4 = self._block(base_channels * 16, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = self._block(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = self._block(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = self._block(base_channels * 2, base_channels)

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

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
        return self.out_conv(d1)


class SegmentationDataset(Dataset):
    def __init__(
        self,
        entries: List[ManifestEntry],
        segmenter: "UNetSegmenter",
        augment: bool,
    ) -> None:
        self.entries = entries
        self.segmenter = segmenter
        self.augment = augment

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        image = self.segmenter.load_preprocessed_image(entry)
        disc_mask, cup_mask = self.segmenter.load_masks(entry)

        if self.augment:
            image = self.segmenter.jitter_image(image)
            image, disc_mask, cup_mask = self.segmenter.augment_geometric(
                image, disc_mask, cup_mask
            )
        image_tensor = transforms.ToTensor()(image)
        image_tensor = self.segmenter._normalize_tensor(image_tensor)

        mask = np.stack([disc_mask, cup_mask], axis=0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask)
        return image_tensor, mask_tensor


class UNetSegmenter:
    def __init__(
        self,
        manifest_path: Path,
        device: Optional[str] = None,
        cup_weight: float = 1.0,
        disc_weight: float = 1.0,
        target_size: int = 512,
        val_ratio: float = 0.1,
        train_datasets: Optional[Iterable[str]] = None,
        val_datasets: Optional[Iterable[str]] = None,
        holdout_datasets: Optional[Iterable[str]] = None,
        normalize: str = "none",
        use_stronger_aug: bool = False,
        mask_cache_dir: Optional[Path] = None,
        image_cache_dir: Optional[Path] = None,
        in_memory_cache: bool = False,
        loader_workers: int = 0,
    ) -> None:
        self.manifest_path = manifest_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cup_weight = cup_weight
        self.disc_weight = disc_weight
        self.target_size = target_size
        self.val_ratio = val_ratio
        self.normalize = (normalize or "none").lower()
        self.use_stronger_aug = bool(use_stronger_aug)
        self.mask_cache_dir = Path(mask_cache_dir).resolve() if mask_cache_dir else None
        if self.mask_cache_dir:
            self.mask_cache_dir.mkdir(parents=True, exist_ok=True)
        self.image_cache_dir = Path(image_cache_dir).resolve() if image_cache_dir else None
        if self.image_cache_dir:
            self.image_cache_dir.mkdir(parents=True, exist_ok=True)
        self.in_memory_cache = bool(in_memory_cache)
        self._mem_image_cache: dict[str, np.ndarray] = {}
        self._mem_mask_cache: dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.loader_workers = max(0, int(loader_workers))

        self.train_dataset_filter = self._normalize_filter(train_datasets)
        self.val_dataset_filter = self._normalize_filter(val_datasets)
        self.holdout_dataset_filter = self._normalize_filter(holdout_datasets)

        self.model = UNet().to(self.device)
        self._manifest: List[ManifestEntry] = []
        self.train_entries: List[ManifestEntry] = []
        self.val_entries: List[ManifestEntry] = []
        self.holdout_entries: List[ManifestEntry] = []
        self.read_manifest()

    def prebuild_in_memory_cache(
        self,
        *,
        cache_workers: int = 0,
        include_train: bool = True,
        include_val: bool = True,
        include_holdout: bool = False,
    ) -> None:
        if not self.in_memory_cache:
            return
        selected: List[ManifestEntry] = []
        if include_train:
            selected.extend(self.train_entries)
        if include_val:
            selected.extend(self.val_entries)
        if include_holdout:
            selected.extend(self.holdout_entries)
        if not selected:
            return

        # Deduplicate by cache key.
        dedup = {}
        for entry in selected:
            dedup[self._entry_cache_key(entry)] = entry
        entries = list(dedup.values())
        workers = max(0, int(cache_workers))
        print(
            f"[UNetSegmenter] prebuilding in-memory cache for {len(entries)} samples "
            f"(cache_workers={workers})",
            flush=True,
        )

        def _warm_one(entry: ManifestEntry) -> None:
            self.load_preprocessed_image(entry)
            self.load_masks(entry)

        if workers <= 1:
            for entry in tqdm(entries, desc="Warm cache", unit="sample"):
                _warm_one(entry)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(_warm_one, entry) for entry in entries]
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Warm cache", unit="sample"):
                    fut.result()

    # ------------------------------------------------------------------
    def read_manifest(self) -> None:
        df = pd.read_csv(self.manifest_path)
        entries: List[ManifestEntry] = []
        for _, row in df.iterrows():
            entry = ManifestEntry(
                sample_id=row["sample_id"],
                dataset=row["dataset"],
                image_path=Path(row["image_path"]),
                annotation_disc=Path(row["annotation_disc"]),
                annotation_cup=Path(row["annotation_cup"]),
                annotation_type_disc=row["annotation_type_disc"],
                annotation_type_cup=row["annotation_type_cup"],
                split=row["split"],
            )
            entries.append(entry)
        self._manifest = entries
        self.holdout_entries = [e for e in entries if e.split == "holdout"]
        if self.holdout_dataset_filter is not None:
            self.holdout_entries = [
                e for e in self.holdout_entries if e.dataset in self.holdout_dataset_filter
            ]

        trainable = [e for e in entries if e.split != "holdout"]
        if self.train_dataset_filter is not None:
            trainable = [
                e for e in trainable if e.dataset in self.train_dataset_filter
            ]

        if not trainable:
            self.val_entries = []
            self.train_entries = []
            return

        val_pool = trainable
        if self.val_dataset_filter is not None:
            filtered = [e for e in trainable if e.dataset in self.val_dataset_filter]
            if filtered:
                val_pool = filtered

        if len(trainable) == 1:
            val_count = 0
        else:
            val_count = max(1, int(len(trainable) * self.val_ratio))
            val_count = min(val_count, len(val_pool), len(trainable) - 1)

        selected_val: List[ManifestEntry] = []
        if val_count > 0:
            selected_val = list(val_pool[:val_count])
        self.val_entries = selected_val
        selected_ids = {id(item) for item in selected_val}
        self.train_entries = [e for e in trainable if id(e) not in selected_ids]

        if not self.train_entries and trainable:
            # Fallback when filtering removed all train entries (e.g. val_count forced entire set)
            self.train_entries = trainable
            self.val_entries = []

    # ------------------------------------------------------------------
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        return image.resize((self.target_size, self.target_size), Resampling.BILINEAR)

    def jitter_image(self, image: Image.Image) -> Image.Image:
        # Photometric jitter only; geometric ops are applied jointly (image+mask)
        return transforms.ColorJitter(0.1, 0.1, 0.1, 0.05)(image)

    def augment_geometric(
        self,
        image: Image.Image,
        disc_mask: np.ndarray,
        cup_mask: np.ndarray,
    ) -> tuple[Image.Image, np.ndarray, np.ndarray]:
        if not self.use_stronger_aug:
            return image, disc_mask, cup_mask

        img = image
        disc_pil = Image.fromarray((disc_mask > 0).astype(np.uint8) * 255)
        cup_pil = Image.fromarray((cup_mask > 0).astype(np.uint8) * 255)

        # Random horizontal flip
        if np.random.rand() < 0.5:
            img = ImageOps.mirror(img)
            disc_pil = ImageOps.mirror(disc_pil)
            cup_pil = ImageOps.mirror(cup_pil)
        # Random vertical flip
        if np.random.rand() < 0.5:
            img = ImageOps.flip(img)
            disc_pil = ImageOps.flip(disc_pil)
            cup_pil = ImageOps.flip(cup_pil)
        # Random rotation (multiples of 90° to keep masks aligned)
        rotations = np.random.choice([0, 90, 180, 270])
        if rotations:
            img = img.rotate(rotations, expand=False)
            disc_pil = disc_pil.rotate(rotations, expand=False)
            cup_pil = cup_pil.rotate(rotations, expand=False)

        disc_mask = (np.array(disc_pil) > 0).astype(np.float32)
        cup_mask = (np.array(cup_pil) > 0).astype(np.float32)
        return img, disc_mask, cup_mask

    @staticmethod
    def _slugify(text: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)

    def _entry_cache_key(self, entry: ManifestEntry) -> str:
        return self._slugify(f"{entry.dataset}_{entry.sample_id}_sz{self.target_size}")

    def _mask_cache_path(self, entry: ManifestEntry) -> Optional[Path]:
        if self.mask_cache_dir is None:
            return None
        slug = self._slugify(f"{entry.dataset}_{entry.sample_id}")
        fname = f"{slug}_sz{self.target_size}.npz"
        return self.mask_cache_dir / fname

    def _image_cache_path(self, entry: ManifestEntry) -> Optional[Path]:
        if self.image_cache_dir is None:
            return None
        slug = self._slugify(f"{entry.dataset}_{entry.sample_id}")
        fname = f"{slug}_img_sz{self.target_size}.npz"
        return self.image_cache_dir / fname

    def _load_image_cache(self, cache_path: Path) -> Optional[Image.Image]:
        try:
            data = np.load(str(cache_path), allow_pickle=False)
            arr = data["image"].astype(np.uint8, copy=False)
            if arr.ndim != 3 or arr.shape[2] != 3:
                return None
            return Image.fromarray(arr, mode="RGB")
        except Exception:
            with suppress(OSError, FileNotFoundError):
                cache_path.unlink()
            return None

    def _save_image_cache(self, cache_path: Optional[Path], image: Image.Image) -> None:
        if cache_path is None:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp.npz")
        try:
            arr = np.asarray(image, dtype=np.uint8)
            np.savez_compressed(tmp_path, image=arr)
            os.replace(tmp_path, cache_path)
        except Exception:
            with suppress(OSError, FileNotFoundError):
                tmp_path.unlink()

    def load_preprocessed_image(self, entry: ManifestEntry) -> Image.Image:
        key = self._entry_cache_key(entry)
        if self.in_memory_cache:
            cached = self._mem_image_cache.get(key)
            if cached is not None:
                return Image.fromarray(cached, mode="RGB")
        cache_path = self._image_cache_path(entry)
        if cache_path and cache_path.exists():
            cached = self._load_image_cache(cache_path)
            if cached is not None:
                if self.in_memory_cache:
                    self._mem_image_cache[key] = np.asarray(cached, dtype=np.uint8)
                return cached
        image = Image.open(entry.image_path).convert("RGB")
        image = self.preprocess_image(image)
        if self.in_memory_cache:
            self._mem_image_cache[key] = np.asarray(image, dtype=np.uint8)
        self._save_image_cache(cache_path, image)
        return image

    def _load_mask_cache(self, cache_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        try:
            data = np.load(str(cache_path), allow_pickle=False)
            disc = data["disc"].astype(np.float32)
            cup = data["cup"].astype(np.float32)
            return disc, cup
        except Exception:
            with suppress(OSError, FileNotFoundError):
                cache_path.unlink()
            return None

    def _save_mask_cache(
        self,
        cache_path: Optional[Path],
        disc_mask: np.ndarray,
        cup_mask: np.ndarray,
    ) -> None:
        if cache_path is None:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp.npz")
        try:
            np.savez_compressed(
                tmp_path,
                disc=disc_mask.astype(np.uint8),
                cup=cup_mask.astype(np.uint8),
            )
            os.replace(tmp_path, cache_path)
        except Exception:
            with suppress(OSError, FileNotFoundError):
                tmp_path.unlink()

    def _normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.normalize == "per_image":
            mean = tensor.mean(dim=(1, 2), keepdim=True)
            std = tensor.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
            return (tensor - mean) / std
        if self.normalize == "imagenet":
            mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
            return (tensor - mean) / std
        return tensor

    # ------------------------------------------------------------------
    def extract_masks_from_image(
        self,
        mask_path: Path,
        disc_color: Optional[tuple[int, int, int]] = None,
        cup_color: Optional[tuple[int, int, int]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[int, int]]:
        raw = Image.open(mask_path)
        arr = np.array(raw)
        if arr.ndim == 2:
            h, w = arr.shape
            flat = arr.reshape(-1).astype(np.int64, copy=False)
            edges = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]], axis=0).astype(np.int64, copy=False)
            edge_counts = np.bincount(edges, minlength=256)
            bg_val = int(np.argmax(edge_counts))
            counts = np.bincount(flat, minlength=256)
            counts[bg_val] = 0
            vals = np.where(counts > 0)[0]
            if vals.size < 1:
                raise ValueError(f"Mask {mask_path} does not contain discernible labels")
            # Disc = ALL non-background pixels (full optic disc: rim + cup combined).
            # Previously this was rim-only, which caused the cup structural prior
            # (cup & disc) to produce empty cup masks since cup and rim don't overlap.
            disc_mask = (arr != bg_val).astype(np.uint8)
            # Cup = the darkest non-background value (0 in REFUGE = inner cup region).
            # Using min-value rather than frequency avoids swapping when cup area > rim area.
            cup_val = int(np.min(vals)) if vals.size > 1 else None
            cup_mask = (arr == cup_val).astype(np.uint8) if cup_val is not None else np.zeros_like(disc_mask, dtype=np.uint8)
            return disc_mask, cup_mask if cup_mask.any() else None, (w, h)

        image = raw.convert("RGB")
        arr = np.array(image)
        h, w, c = arr.shape

        if disc_color is None or cup_color is None:
            # Fast color discovery via NumPy (avoid Python-level per-pixel tuple counting).
            edges = np.concatenate(
                [arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]], axis=0
            )
            edge_colors, edge_counts = np.unique(edges.reshape(-1, c), axis=0, return_counts=True)
            bg_color_np = edge_colors[int(np.argmax(edge_counts))]

            colors_np, counts_np = np.unique(arr.reshape(-1, c), axis=0, return_counts=True)
            keep = np.any(colors_np != bg_color_np.reshape(1, -1), axis=1)
            colors_np = colors_np[keep]
            counts_np = counts_np[keep]
            if colors_np.shape[0] < 1:
                raise ValueError(f"Mask {mask_path} does not contain discernible labels")
            order = np.argsort(-counts_np)
            colors_np = colors_np[order]
            disc_color = tuple(int(v) for v in colors_np[0].tolist())
            cup_color = (
                tuple(int(v) for v in colors_np[1].tolist())
                if colors_np.shape[0] > 1
                else None
            )

        disc_mask = np.zeros((h, w), dtype=np.uint8)
        cup_mask = np.zeros((h, w), dtype=np.uint8)

        if disc_color is not None:
            disc_mask[np.all(arr == disc_color, axis=-1)] = 1
        if cup_color is not None:
            cup_mask[np.all(arr == cup_color, axis=-1)] = 1

        return disc_mask, cup_mask if cup_mask.any() else None, (w, h)

    def load_contour_from_file(self, contour_path: Path) -> np.ndarray:
        # Fast path: contour files are typically CSV or whitespace-delimited x,y pairs.
        try:
            arr = np.loadtxt(str(contour_path), delimiter=",", comments="#", dtype=np.float32)
        except Exception:
            try:
                arr = np.loadtxt(str(contour_path), comments="#", dtype=np.float32)
            except Exception:
                return np.zeros((0, 2), dtype=np.float32)
        if arr.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        if arr.ndim == 1:
            if arr.shape[0] < 2:
                return np.zeros((0, 2), dtype=np.float32)
            arr = arr.reshape(1, -1)
        if arr.shape[1] < 2:
            return np.zeros((0, 2), dtype=np.float32)
        return arr[:, :2].astype(np.float32, copy=False)

    def coords_to_mask(
        self,
        coords: Optional[np.ndarray],
        size: Tuple[int, int],
    ) -> np.ndarray:
        if coords is None or len(coords) == 0:
            return np.zeros((self.target_size, self.target_size), dtype=np.float32)

        width, height = map(int, size)
        target_shape = (height, width)
        arr = np.asarray(coords)
        if arr.size == 0:
            return np.zeros((self.target_size, self.target_size), dtype=np.float32)

        if arr.ndim == 2 and arr.shape[-1] != 2:
            mask = (arr > 0).astype(np.uint8)
            return self._resize_mask(mask)

        if arr.ndim > 2:
            arr = arr.reshape(-1, arr.shape[-1])
        arr = arr.astype(float, copy=False)
        if arr.shape[-1] != 2:
            raise ValueError(f"Expected coordinate pairs, got shape {arr.shape}")

        points = [tuple(map(float, pt)) for pt in arr]
        if len(points) < 3:
            return np.zeros(target_shape, dtype=np.float32)

        img = Image.new("L", size, 0)
        draw = ImageDraw.Draw(img)
        draw.polygon(points, outline=1, fill=1)
        mask = np.array(img, dtype=np.uint8)
        return self._resize_mask(mask)

    def _resize_mask(self, mask: np.ndarray) -> np.ndarray:
        img = Image.fromarray((mask > 0).astype(np.uint8) * 255)
        img = img.resize((self.target_size, self.target_size), Resampling.NEAREST)
        return (np.array(img, dtype=np.uint8) > 0).astype(np.float32)

    def load_masks(self, entry: ManifestEntry) -> Tuple[np.ndarray, np.ndarray]:
        key = self._entry_cache_key(entry)
        if self.in_memory_cache:
            cached = self._mem_mask_cache.get(key)
            if cached is not None:
                disc_u8, cup_u8 = cached
                return disc_u8.astype(np.float32), cup_u8.astype(np.float32)
        cache_path = self._mask_cache_path(entry)
        if cache_path and cache_path.exists():
            cached = self._load_mask_cache(cache_path)
            if cached is not None:
                if self.in_memory_cache:
                    disc, cup = cached
                    self._mem_mask_cache[key] = (
                        disc.astype(np.uint8),
                        cup.astype(np.uint8),
                    )
                return cached

        image = Image.open(entry.image_path)
        size = image.size

        disc_coords = cup_coords = None
        if entry.annotation_type_disc == "mask":
            disc_coords, cup_coords_from_disc, size = self.extract_masks_from_image(
                entry.annotation_disc
            )
            if cup_coords_from_disc is not None:
                cup_coords = cup_coords_from_disc
        else:
            disc_coords = self.load_contour_from_file(entry.annotation_disc)

        if entry.annotation_type_cup == "mask":
            _, cup_coords_from_cup, size_cup = self.extract_masks_from_image(
                entry.annotation_cup
            )
            if cup_coords_from_cup is not None:
                cup_coords = cup_coords_from_cup
            if disc_coords is None:
                disc_coords, _, size = self.extract_masks_from_image(
                    entry.annotation_cup
                )
            else:
                size = size_cup
        else:
            cup_coords = self.load_contour_from_file(entry.annotation_cup)

        disc_mask = self.coords_to_mask(disc_coords, size).astype(np.float32)
        cup_mask = self.coords_to_mask(cup_coords, size).astype(np.float32)
        if self.in_memory_cache:
            self._mem_mask_cache[key] = (
                disc_mask.astype(np.uint8),
                cup_mask.astype(np.uint8),
            )
        self._save_mask_cache(cache_path, disc_mask, cup_mask)
        return disc_mask, cup_mask

    # ------------------------------------------------------------------
    def build_loaders(self, batch_size: int = 4, num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
        train_ds = SegmentationDataset(self.train_entries, self, augment=True)
        val_ds = SegmentationDataset(self.val_entries, self, augment=False)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        return train_loader, val_loader

    # ------------------------------------------------------------------
    def dice_score(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = (preds > 0.5).float()
        intersection = (preds * targets).sum(dim=(2, 3))
        union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        return dice.mean(dim=0)

    def train(
        self,
        epochs: int = 40,
        batch_size: int = 4,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        checkpoint_dir: Path = Path("models/unet_segmenter"),
    ) -> None:
        print(
            f"[UNetSegmenter] training on device={self.device} "
            f"(epochs={epochs}, batch_size={batch_size}, workers={self.loader_workers})"
        )
        train_loader, val_loader = self.build_loaders(batch_size=batch_size, num_workers=self.loader_workers)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        criterion = nn.BCEWithLogitsLoss()
        best_dice = -math.inf
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_path = checkpoint_dir / "best.pt"

        epoch_bar = tqdm(range(1, epochs + 1), desc="Epochs", unit="epoch")

        for epoch in epoch_bar:
            self.model.train()
            batch_bar = tqdm(
                train_loader,
                desc=f"Train {epoch}/{epochs}",
                leave=False,
                unit="batch",
                total=len(train_loader),
            )
            train_loss_total = 0.0
            train_samples = 0
            for images, masks in batch_bar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                optimizer.zero_grad()
                logits = self.model(images)
                loss_disc = criterion(logits[:, 0:1], masks[:, 0:1])
                loss_cup = criterion(logits[:, 1:2], masks[:, 1:2])
                loss = self.disc_weight * loss_disc + self.cup_weight * loss_cup
                loss.backward()
                optimizer.step()
                batch_size = images.size(0)
                train_loss_total += loss.item() * batch_size
                train_samples += batch_size

            train_loss = (
                train_loss_total / train_samples if train_samples else float("nan")
            )

            self.model.eval()
            dices = []
            val_bar = tqdm(
                val_loader,
                desc="Validate",
                leave=False,
                unit="batch",
                total=len(val_loader),
            )
            with torch.no_grad():
                for images, masks in val_bar:
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    logits = self.model(images)
                    probs = torch.sigmoid(logits)
                    dice = self.dice_score(probs, masks)
                    dices.append(dice.cpu())
            if dices:
                mean_dice = torch.stack(dices).mean(dim=0)
                disc_dice = mean_dice[0].item()
                cup_dice = mean_dice[1].item()
                weight_sum = self.disc_weight + self.cup_weight
                score = (
                    (self.disc_weight * disc_dice + self.cup_weight * cup_dice)
                    / weight_sum
                    if weight_sum
                    else 0.0
                )
                epoch_bar.set_postfix(
                    loss=f"{train_loss:.4f}",
                    dice_disc=f"{disc_dice:.3f}",
                    dice_cup=f"{cup_dice:.3f}",
                    dice_w=f"{score:.3f}",
                )
            else:
                disc_dice = cup_dice = 0.0
                score = 0.0
                epoch_bar.set_postfix(loss=f"{train_loss:.4f}")

            if score > best_dice:
                best_dice = score
                torch.save({"model": self.model.state_dict()}, best_path)

        if best_path.exists():
            state = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(state["model"])

    # ------------------------------------------------------------------
    def evaluate_holdout(
        self, output_dir: Path = Path("analysis_data/segmenter_eval")
    ) -> pd.DataFrame:
        return self.evaluate_dataset(split_filter={"holdout"}, output_dir=output_dir)

    @staticmethod
    def overlay_masks(
        image: Image.Image, disc: np.ndarray, cup: np.ndarray
    ) -> Image.Image:
        overlay = image.copy()
        disc_img = Image.fromarray((disc * 255).astype(np.uint8))
        cup_img = Image.fromarray((cup * 255).astype(np.uint8))
        disc_color = Image.new("RGBA", image.size, (255, 0, 0, 0))
        cup_color = Image.new("RGBA", image.size, (0, 255, 0, 0))
        disc_color.paste((255, 0, 0, 100), mask=disc_img)
        cup_color.paste((0, 255, 0, 100), mask=cup_img)
        overlay = overlay.convert("RGBA")
        overlay = Image.alpha_composite(overlay, disc_color)
        overlay = Image.alpha_composite(overlay, cup_color)
        return overlay.convert("RGB")

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_filter(values: Optional[Iterable[str]]) -> Optional[Set[str]]:
        if values is None:
            return None
        if isinstance(values, str):
            return {values}
        return {str(item) for item in values}

    @staticmethod
    def _dice_from_masks(pred: np.ndarray, target: np.ndarray) -> float:
        pred = (pred > 0).astype(np.float32)
        target = (target > 0).astype(np.float32)
        intersection = float((pred * target).sum())
        denom = float(pred.sum() + target.sum())
        return (2.0 * intersection + 1e-6) / (denom + 1e-6)

    def get_entries(
        self,
        dataset_filter: Optional[Iterable[str]] = None,
        split_filter: Optional[Iterable[str]] = None,
    ) -> List[ManifestEntry]:
        dataset_set = self._normalize_filter(dataset_filter)
        split_set = self._normalize_filter(split_filter)
        entries = self._manifest
        if dataset_set is not None:
            entries = [e for e in entries if e.dataset in dataset_set]
        if split_set is not None:
            entries = [e for e in entries if e.split in split_set]
        return list(entries)

    def evaluate_dataset(
        self,
        dataset_filter: Optional[Iterable[str]] = None,
        split_filter: Optional[Iterable[str]] = None,
        output_dir: Path = Path("analysis_data/segmenter_eval"),
        save_overlays: bool = True,
        metrics_path: Optional[Path] = None,
        threshold: float = 0.5,
        tta: bool = False,
    ) -> pd.DataFrame:
        entries = self.get_entries(
            dataset_filter=dataset_filter, split_filter=split_filter
        )
        if not entries:
            return pd.DataFrame(
                columns=[
                    "sample_id",
                    "dataset",
                    "split",
                    "dice_disc",
                    "dice_cup",
                ]
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        if metrics_path is None:
            suffix_parts = []
            if dataset_filter is not None:
                suffix_parts.append("-".join(sorted(self._normalize_filter(dataset_filter))))
            if split_filter is not None:
                suffix_parts.append("-".join(sorted(self._normalize_filter(split_filter))))
            suffix = "_".join(part for part in suffix_parts if part)
            csv_name = f"metrics{'_' + suffix if suffix else ''}.csv"
            metrics_path = output_dir / csv_name

        records = []
        self.model.eval()
        progress = tqdm(
            entries,
            desc="Evaluate",
            unit="sample",
            leave=False,
        )
        for entry in progress:
            orig_image = Image.open(entry.image_path).convert("RGB")
            image = self.preprocess_image(orig_image)
            tensor = transforms.ToTensor()(image)
            tensor = self._normalize_tensor(tensor)
            tensor = tensor.unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                if tta:
                    t_h = torch.flip(tensor, dims=[3])
                    log_h = self.model(t_h)
                    log_h = torch.flip(log_h, dims=[3])
                    t_v = torch.flip(tensor, dims=[2])
                    log_v = self.model(t_v)
                    log_v = torch.flip(log_v, dims=[2])
                    logits = (logits + log_h + log_v) / 3.0
                probs = torch.sigmoid(logits)[0].cpu().numpy()

            disc_pred = (probs[0] > threshold).astype(np.uint8)
            cup_pred = (probs[1] > threshold).astype(np.uint8)
            # Structural prior: cup within disc
            cup_pred = (cup_pred > 0) & (disc_pred > 0)
            cup_pred = cup_pred.astype(np.uint8)

            disc_gt, cup_gt = self.load_masks(entry)
            disc_gt = disc_gt.astype(np.uint8)
            cup_gt = cup_gt.astype(np.uint8)

            dice_disc = self._dice_from_masks(disc_pred, disc_gt)
            dice_cup = self._dice_from_masks(cup_pred, cup_gt)

            records.append(
                {
                    "sample_id": entry.sample_id,
                    "dataset": entry.dataset,
                    "split": entry.split,
                    "dice_disc": dice_disc,
                    "dice_cup": dice_cup,
                }
            )

            progress.set_postfix(
                dice_disc=f"{dice_disc:.3f}", dice_cup=f"{dice_cup:.3f}"
            )

            if save_overlays:
                overlay_gt = self.overlay_masks(image, disc_gt, cup_gt)
                overlay_pred = self.overlay_masks(image, disc_pred, cup_pred)
                combined = Image.new("RGB", (image.width * 2, image.height))
                combined.paste(overlay_gt, (0, 0))
                combined.paste(overlay_pred, (image.width, 0))
                combined.save(output_dir / f"{entry.sample_id}_eval.png")

        metrics_df = pd.DataFrame(records)
        summary = metrics_df[["dice_disc", "dice_cup"]].mean()
        summary_row = {
            "sample_id": "__mean__",
            "dataset": "summary",
            "split": "summary",
            "dice_disc": summary["dice_disc"],
            "dice_cup": summary["dice_cup"],
        }
        metrics_with_summary = pd.concat(
            [metrics_df, pd.DataFrame([summary_row])], ignore_index=True
        )
        metrics_with_summary.to_csv(metrics_path, index=False)
        return metrics_with_summary
