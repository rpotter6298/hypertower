"""geometry_towers — GeometryTower and all segmentation-map infrastructure.

Self-contained: absorbs everything that was in seg_cnn.py so that file can
eventually be removed.  Does not import from seg_cnn.py or any other tower file.
Imports only TowerBase from towerbase plus standard infrastructure.

Contents
--------
SegMapRecord           — labelled-eye data record
_combine_masks         — merge disc/cup binary masks → 3-class label map
crop_to_disc           — tight bounding-box crop
seg_map_to_tensor      — (H,W) uint8 → (C,H,W) float32 tensor
load_gt_masks          — load GT disc+cup masks from contour/mask files
UNetFineTuneDataset    — Dataset for fine-tuning the UNet on GT annotations
precompute_unet_seg_maps — batch UNet inference helper
SegMapDataset          — Dataset yielding (seg_tensor, label) pairs
SegCNN                 — pretrained CNN adapted for segmentation-map input
GeometryTower          — TowerBase implementation (the main class to use)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from PIL.Image import Resampling
from torch.utils.data import Dataset
from torchvision import models, transforms
from tqdm import tqdm

import pandas as pd

from v3.classes.towerbase import TowerBase


# ---------------------------------------------------------------------------
# Data record
# ---------------------------------------------------------------------------

@dataclass
class SegMapRecord:
    """One labelled eye sample for the seg-map CNN."""
    sample_id: str
    image_path: Path           # original fundus image (used by unet mode)
    annotation_disc: Path      # contour (.txt) or mask (.bmp/.png)
    annotation_cup: Path
    annotation_type_disc: str  # "contour" or "mask"
    annotation_type_cup: str
    patient_id: int            # for group-CV: keep both eyes of a patient together
    eye: str                   # "OD" or "OS"
    label: int                 # 0 = Normal, 1 = Glaucoma


# ---------------------------------------------------------------------------
# Seg-map utilities
# ---------------------------------------------------------------------------

def _combine_masks(disc_mask: np.ndarray, cup_mask: np.ndarray) -> np.ndarray:
    """Combine binary disc and cup masks into a 3-class label map.

    Returns a uint8 array with values:
      0 — background
      1 — optic disc rim  (disc but not cup)
      2 — optic cup
    """
    disc = (disc_mask > 0).astype(np.uint8)
    cup  = (cup_mask  > 0).astype(np.uint8)
    cup  = (cup & disc)          # structural prior: cup must be inside disc
    seg  = disc + cup            # 0, 1 (rim), or 2 (cup)
    return seg.astype(np.uint8)


def crop_to_disc(seg_map: np.ndarray) -> np.ndarray:
    """Crop a seg map tightly to the disc bounding box.

    The disc is anywhere seg_map > 0 (i.e. rim or cup).
    Returns the original array unchanged if no disc is found.
    """
    rows = np.any(seg_map > 0, axis=1)
    cols = np.any(seg_map > 0, axis=0)
    if not rows.any():
        return seg_map
    r0, r1 = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1]))
    c0, c1 = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1]))
    return seg_map[r0:r1 + 1, c0:c1 + 1]


def seg_map_to_tensor(
    seg_map: np.ndarray,
    channels: int,
    target_size: int,
) -> torch.Tensor:
    """Convert an (H, W) seg map with values {0, 1, 2} to a float tensor.

    channels=1  →  (1, H, W) float in [0, 1]  (values 0/0.5/1.0)
    channels=3  →  (3, H, W) one-hot binary channels [bg, disc_rim, cup]
    """
    pil = Image.fromarray(seg_map.astype(np.uint8), mode="L")
    pil = pil.resize((target_size, target_size), Image.NEAREST)
    seg = np.array(pil, dtype=np.uint8)

    if channels == 1:
        arr = seg.astype(np.float32) / 2.0   # {0, 0.5, 1.0}
        return torch.from_numpy(arr).unsqueeze(0)

    if channels == 3:
        bg       = (seg == 0).astype(np.float32)
        disc_rim = (seg == 1).astype(np.float32)
        cup      = (seg == 2).astype(np.float32)
        return torch.from_numpy(np.stack([bg, disc_rim, cup], axis=0))

    raise ValueError(f"channels must be 1 or 3, got {channels}")


# ---------------------------------------------------------------------------
# GT mask loading (pure NumPy / PIL — no CUDA, safe in DataLoader workers)
# ---------------------------------------------------------------------------

def _load_contour(path: Path) -> np.ndarray:
    """Load x,y contour pairs from a whitespace- or comma-delimited text file."""
    arr = np.zeros((0, 2), dtype=np.float32)
    for delimiter in (",", None):
        try:
            candidate = np.loadtxt(str(path), delimiter=delimiter, comments="#", dtype=np.float32)
            if candidate.size > 0:
                arr = candidate
                break
        except Exception:
            pass
    if arr.size == 0 or arr.ndim == 1:
        return np.zeros((0, 2), dtype=np.float32)
    if arr.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    return arr[:, :2]


def _contour_to_mask(
    coords: np.ndarray, image_size: Tuple[int, int], target_size: int
) -> np.ndarray:
    """Rasterise a polygon defined by (x, y) coords into a binary mask.

    image_size is the (width, height) of the original fundus image — the
    coordinate space the contour was annotated in.
    """
    if coords is None or len(coords) < 3:
        return np.zeros((target_size, target_size), dtype=np.uint8)
    points = [tuple(map(float, pt)) for pt in coords]
    img = Image.new("L", image_size, 0)
    ImageDraw.Draw(img).polygon(points, outline=1, fill=1)
    img = img.resize((target_size, target_size), Resampling.NEAREST)
    return (np.array(img, dtype=np.uint8) > 0).astype(np.uint8)


def _extract_masks_from_image(
    mask_path: Path, target_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract disc and cup binary masks from a segmentation image file.

    Handles both grayscale label images (e.g. REFUGE .bmp) and
    RGB colour-coded masks.  Returns (disc_mask, cup_mask) both at
    target_size × target_size.
    """
    raw = Image.open(mask_path)
    arr = np.array(raw)

    if arr.ndim == 2:
        edges = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
        bg_val = int(np.argmax(np.bincount(edges.astype(np.int64).clip(0, 255), minlength=256)))
        disc_arr = (arr != bg_val).astype(np.uint8)
        vals = np.unique(arr)
        non_bg = vals[vals != bg_val]
        cup_arr: np.ndarray
        if non_bg.size > 1:
            cup_val = int(non_bg.min())
            cup_arr = (arr == cup_val).astype(np.uint8)
        else:
            cup_arr = np.zeros_like(disc_arr, dtype=np.uint8)
    else:
        img_rgb = raw.convert("RGB")
        arr = np.array(img_rgb)
        h, w, c = arr.shape
        edges_rgb = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]], axis=0)
        edge_colors, edge_counts = np.unique(edges_rgb.reshape(-1, c), axis=0, return_counts=True)
        bg_color = edge_colors[int(np.argmax(edge_counts))]
        colors, counts = np.unique(arr.reshape(-1, c), axis=0, return_counts=True)
        not_bg = np.any(colors != bg_color.reshape(1, -1), axis=1)
        colors, counts = colors[not_bg], counts[not_bg]
        disc_arr = np.zeros((h, w), dtype=np.uint8)
        cup_arr  = np.zeros((h, w), dtype=np.uint8)
        if colors.shape[0] >= 1:
            order = np.argsort(-counts)
            disc_color = colors[order[0]]
            disc_arr[np.all(arr == disc_color, axis=-1)] = 1
            if colors.shape[0] >= 2:
                cup_color = colors[order[1]]
                cup_arr[np.all(arr == cup_color, axis=-1)] = 1

    def _resize(m: np.ndarray) -> np.ndarray:
        pil = Image.fromarray((m > 0).astype(np.uint8) * 255)
        pil = pil.resize((target_size, target_size), Resampling.NEAREST)
        return (np.array(pil) > 0).astype(np.uint8)

    return _resize(disc_arr), _resize(cup_arr)


def load_gt_masks(rec: SegMapRecord, target_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load GT disc + cup masks for one record.

    Handles annotation_type "contour" (x,y text file) and "mask" (image file).
    Returns (disc_mask, cup_mask) as uint8 arrays of shape (target_size, target_size).
    """
    disc_mask: Optional[np.ndarray] = None
    cup_mask:  Optional[np.ndarray] = None

    with Image.open(rec.image_path) as _img:
        image_size = _img.size  # (width, height)

    # ---- Disc ----
    if rec.annotation_type_disc == "mask":
        disc_mask, cup_from_disc = _extract_masks_from_image(rec.annotation_disc, target_size)
        if cup_from_disc.any():
            cup_mask = cup_from_disc
    else:  # contour
        coords = _load_contour(rec.annotation_disc)
        disc_mask = _contour_to_mask(coords, image_size, target_size)

    # ---- Cup ----
    if cup_mask is None:
        if rec.annotation_type_cup == "mask":
            _, cup_from_cup = _extract_masks_from_image(rec.annotation_cup, target_size)
            cup_mask = cup_from_cup
        else:  # contour
            coords = _load_contour(rec.annotation_cup)
            cup_mask = _contour_to_mask(coords, image_size, target_size)

    if disc_mask is None:
        disc_mask = np.zeros((target_size, target_size), dtype=np.uint8)
    if cup_mask is None:
        cup_mask = np.zeros((target_size, target_size), dtype=np.uint8)

    cup_mask = (cup_mask > 0) & (disc_mask > 0)
    return disc_mask.astype(np.uint8), cup_mask.astype(np.uint8)


# ---------------------------------------------------------------------------
# U-Net fine-tuning dataset
# ---------------------------------------------------------------------------

class UNetFineTuneDataset(Dataset):
    """Loads (image_tensor, mask_tensor) pairs for fine-tuning the U-Net."""

    def __init__(
        self,
        records: List[SegMapRecord],
        target_size: int = 512,
        normalize: str = "per_image",
    ) -> None:
        self.records = records
        self.target_size = target_size
        self.normalize = normalize
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.records)

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.normalize == "per_image":
            mean = tensor.mean(dim=(1, 2), keepdim=True)
            std  = tensor.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
            return (tensor - mean) / std
        if self.normalize == "imagenet":
            mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
            return (tensor - mean) / std
        return tensor

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        image = Image.open(rec.image_path).convert("RGB")
        image = image.resize((self.target_size, self.target_size), Resampling.BILINEAR)
        img_tensor = self._normalize(self.to_tensor(image))
        disc_mask, cup_mask = load_gt_masks(rec, self.target_size)
        mask_tensor = torch.from_numpy(
            np.stack([disc_mask, cup_mask], axis=0).astype(np.float32)
        )
        return img_tensor, mask_tensor


# ---------------------------------------------------------------------------
# U-Net precomputation
# ---------------------------------------------------------------------------

def precompute_unet_seg_maps(
    records: List[SegMapRecord],
    segmenter,
    threshold: float = 0.5,
) -> List[np.ndarray]:
    """Run the U-Net on every record and return a list of combined seg maps."""
    to_tensor = transforms.ToTensor()
    seg_maps = []
    for rec in tqdm(records, desc="U-Net inference", unit="img", leave=False):
        image = Image.open(rec.image_path).convert("RGB")
        resized = segmenter.preprocess_image(image)
        tensor = segmenter._normalize_tensor(
            to_tensor(resized).to(segmenter.device)
        ).unsqueeze(0)
        with torch.no_grad():
            logits = segmenter.model(tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()
        disc = (probs[0] > threshold).astype(np.uint8)
        cup  = (probs[1] > threshold).astype(np.uint8)
        cup  = (cup & disc)
        seg_maps.append(_combine_masks(disc, cup.astype(np.uint8)))
    return seg_maps


# ---------------------------------------------------------------------------
# SegMapDataset
# ---------------------------------------------------------------------------

class SegMapDataset(Dataset):
    """PyTorch Dataset that yields (seg_tensor, label) pairs."""

    def __init__(
        self,
        records: List[SegMapRecord],
        target_size: int = 224,
        channels: int = 3,
        augment: bool = False,
        unet_segmenter=None,
        unet_threshold: float = 0.5,
        seg_target_size: int = 512,
        crop_to_disc_flag: bool = True,
        precomputed_seg_maps: Optional[List[np.ndarray]] = None,
    ) -> None:
        self.records = records
        self.target_size = target_size
        self.channels = channels
        self.augment = augment
        self.seg_target_size = seg_target_size
        self.crop_to_disc_flag = crop_to_disc_flag

        if precomputed_seg_maps is not None:
            self._seg_maps = precomputed_seg_maps
        elif unet_segmenter is not None:
            self._seg_maps = precompute_unet_seg_maps(
                records, unet_segmenter, unet_threshold
            )
        else:
            self._seg_maps = None

    def __len__(self) -> int:
        return len(self.records)

    def _augment(self, seg_map: np.ndarray) -> np.ndarray:
        if np.random.rand() < 0.5:
            seg_map = np.fliplr(seg_map)
        if np.random.rand() < 0.5:
            seg_map = np.flipud(seg_map)
        k = np.random.randint(0, 4)
        if k:
            seg_map = np.rot90(seg_map, k=k)
        return np.ascontiguousarray(seg_map)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        if self._seg_maps is not None:
            seg_map = self._seg_maps[idx]
        else:
            disc_mask, cup_mask = load_gt_masks(rec, self.seg_target_size)
            seg_map = _combine_masks(disc_mask, cup_mask)

        if self.crop_to_disc_flag:
            seg_map = crop_to_disc(seg_map)
        if self.augment:
            seg_map = self._augment(seg_map)

        tensor = seg_map_to_tensor(seg_map, self.channels, self.target_size)
        return tensor, rec.label


# ---------------------------------------------------------------------------
# SegCNN
# ---------------------------------------------------------------------------

_SEGCNN_FEAT_DIM = {
    "resnet18": 512,
    "resnet50": 2048,
    "efficientnet_b0": 1280,
}


class SegCNN(nn.Module):
    """Pretrained CNN backbone adapted for segmentation-map input.

    Parameters
    ----------
    num_classes : output classes (2 for binary glaucoma grading)
    backbone    : "resnet18" | "resnet50" | "efficientnet_b0"
    pretrained  : initialise with ImageNet weights
    in_channels : 1 (single label map) or 3 (one-hot channels)
    dropout     : dropout rate before the final classifier head
    """

    def __init__(
        self,
        num_classes: int = 2,
        backbone: str = "resnet18",
        pretrained: bool = True,
        in_channels: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        weights_arg = "DEFAULT" if pretrained else None

        if backbone == "resnet18":
            base = models.resnet18(weights=weights_arg)
            feat_dim = base.fc.in_features
            base.fc = nn.Identity()
        elif backbone == "resnet50":
            base = models.resnet50(weights=weights_arg)
            feat_dim = base.fc.in_features
            base.fc = nn.Identity()
        elif backbone == "efficientnet_b0":
            base = models.efficientnet_b0(weights=weights_arg)
            feat_dim = base.classifier[1].in_features
            base.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone!r}")

        if in_channels != 3:
            first_conv = self._find_first_conv(base)
            new_conv = nn.Conv2d(
                in_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None,
            )
            if pretrained:
                with torch.no_grad():
                    new_conv.weight.copy_(
                        first_conv.weight.mean(dim=1, keepdim=True).expand_as(new_conv.weight)
                    )
            self._replace_first_conv(base, new_conv)

        self.backbone = base
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, num_classes),
        )

    @staticmethod
    def _find_first_conv(module: nn.Module) -> nn.Conv2d:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                return m
        raise RuntimeError("No Conv2d found in backbone")

    @staticmethod
    def _replace_first_conv(module: nn.Module, new_conv: nn.Conv2d) -> None:
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                setattr(module, name, new_conv)
                return
            try:
                SegCNN._replace_first_conv(child, new_conv)
                return
            except RuntimeError:
                pass
        raise RuntimeError("Could not replace first Conv2d")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if feats.dim() > 2:
            feats = feats.flatten(1)
        return self.head(feats)


# ---------------------------------------------------------------------------
# GeometryTower — TowerBase implementation
# ---------------------------------------------------------------------------

class GeometryTower(TowerBase, nn.Module):
    """TowerBase implementation for the optic-disc/cup segmentation modality.

    Encodes a 3-class disc/cup segmentation map (bg=0, rim=1, cup=2) through a
    CNN backbone, contributing one spatial embedding to the bridge.

    The seg map is produced from GT annotations (manifest-based) or from a
    trained U-Net, depending on ``geometry_source``.

    ``prepare_fold`` builds a seg-map generator, pre-computes all maps for the
    fold, and caches them keyed by image path.  ``augment_samples`` then
    injects ``seg_map_1`` / ``seg_map_2`` float32 numpy arrays (shape C×H×W)
    into each sample dict so the DataLoader delivers them as tensors to
    ``embed_batch``.

    Parameters
    ----------
    backbone              : CNN backbone — "resnet18" | "resnet50" | "efficientnet_b0"
    in_channels           : 1 (label map) or 3 (one-hot disc/rim/cup channels)
    pretrained            : initialise backbone with ImageNet weights
    frozen                : if True, backbone is always frozen
    target_size           : spatial size the seg map tensor is resized to
    seg_target_size       : resolution at which GT masks are rasterised / U-Net runs
    crop_to_disc          : crop seg map tightly to disc bounding box before resizing
    geometry_source       : "gt" (manifest annotations) or "unet" (U-Net predictions)
    manifest_path         : path to the geometry manifest CSV (required)
    weights_path          : path to UNet checkpoint (required when source="unet")
    unet_normalize        : UNet normalisation mode (default "per_image")
    unet_threshold        : UNet mask threshold (default 0.5)
    finetune_unet_epochs  : epochs to fine-tune U-Net per fold (0 = disabled)
    finetune_unet_lr      : learning rate for U-Net fine-tuning
    """

    def __init__(
        self,
        *,
        backbone: str = "resnet18",
        in_channels: int = 3,
        pretrained: bool = True,
        frozen: bool = False,
        target_size: int = 224,
        seg_target_size: int = 512,
        crop_to_disc: bool = True,
        geometry_source: str = "gt",
        manifest_path=None,
        weights_path=None,
        unet_normalize: str = "per_image",
        unet_threshold: float = 0.5,
        finetune_unet_epochs: int = 0,
        finetune_unet_lr: float = 1e-5,
    ):
        nn.Module.__init__(self)
        self._backbone_name        = backbone
        self._in_channels          = in_channels
        self._frozen               = frozen
        self._target_size          = target_size
        self._seg_target_size      = seg_target_size
        self._crop_to_disc         = crop_to_disc
        self._geometry_source      = geometry_source
        self._manifest_path        = Path(manifest_path) if manifest_path is not None else None
        self._weights_path         = Path(weights_path)  if weights_path  is not None else None
        self._unet_normalize       = unet_normalize
        self._unet_threshold       = unet_threshold
        self._finetune_unet_epochs = finetune_unet_epochs
        self._finetune_unet_lr     = finetune_unet_lr

        self._out_dim  = _SEGCNN_FEAT_DIM.get(backbone, 512)
        self._seg_cnn  = SegCNN(
            num_classes=2,
            backbone=backbone,
            pretrained=pretrained,
            in_channels=in_channels,
        )
        self._seg_cache: dict = {}  # image_path_str → float32 (C, H, W) numpy array

    # ------------------------------------------------------------------
    # TowerBase interface
    # ------------------------------------------------------------------

    @property
    def embed_dims(self) -> list[int]:
        return [self._out_dim]

    @property
    def total_epochs(self) -> int:
        return 0 if self._frozen else 1

    def set_phase(self, phase: str) -> None:
        trainable = not self._frozen and phase in ("tower_warmup", "main")
        for p in self._seg_cnn.parameters():
            p.requires_grad = trainable

    def prepare_fold(
        self,
        *,
        eye_train,
        bilat_train,
        bilat_val,
        bilat_test,
        image_preprocessor,
        image_cache,
        device,
        args,
    ) -> None:
        """Build seg-map generator and pre-compute maps for all fold images."""
        if self._manifest_path is None:
            raise ValueError("GeometryTower requires manifest_path")

        all_paths: dict = {}
        for split in (eye_train, bilat_train, bilat_val, bilat_test):
            for s in split:
                for slot in ("image_1", "image_2"):
                    p = s.get(slot)
                    if p is not None:
                        all_paths[str(Path(p).resolve())] = None

        if self._geometry_source == "unet":
            self._prepare_fold_unet(list(all_paths.keys()), eye_train, device)
        else:
            self._prepare_fold_gt(list(all_paths.keys()))

    def augment_samples(self, samples: list) -> list:
        """Inject ``seg_map_1`` / ``seg_map_2`` float32 arrays into each sample dict.

        Arrays have shape (C, H, W) and are collated by the DataLoader into
        (B, C, H, W) tensors delivered to ``embed_batch``.
        """
        blank = np.zeros(
            (self._in_channels, self._target_size, self._target_size), dtype=np.float32
        )
        for s in samples:
            for img_slot, seg_slot in (("image_1", "seg_map_1"), ("image_2", "seg_map_2")):
                img_path = s.get(img_slot)
                if img_path is None:
                    continue
                key = str(Path(img_path).resolve())
                s[seg_slot] = self._seg_cache.get(key, blank)
        return samples

    def embed_batch(
        self,
        batch: dict,
        *,
        device: torch.device,
        slot: int = 1,
    ) -> list[torch.Tensor]:
        seg = batch.get(f"seg_map_{slot}")
        if seg is None or not torch.is_tensor(seg):
            ref = batch.get(f"image_{slot}")
            bs = ref.shape[0] if torch.is_tensor(ref) else 1
            return [torch.zeros(bs, self._out_dim, device=device)]
        return [self._seg_cnn.backbone(seg.float().to(device))]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _seg_map_to_array(self, seg_map: np.ndarray) -> np.ndarray:
        """Apply crop + resize and return a (C, H, W) float32 numpy array."""
        if self._crop_to_disc:
            seg_map = crop_to_disc(seg_map)
        return seg_map_to_tensor(seg_map, self._in_channels, self._target_size).numpy()

    def _prepare_fold_gt(self, image_paths: list) -> None:
        """Pre-compute GT seg maps from manifest annotations."""
        manifest_df = pd.read_csv(self._manifest_path)
        manifest_df["_img_key"] = manifest_df["image_path"].apply(
            lambda p: str(Path(p).resolve())
        )
        manifest_index = manifest_df.set_index("_img_key").to_dict("index")

        print(
            f"[GeometryTower] pre-computing GT seg maps for {len(image_paths)} images...",
            flush=True,
        )
        n_ok = 0
        blank = np.zeros((self._seg_target_size, self._seg_target_size), dtype=np.uint8)
        for img_path in image_paths:
            entry = manifest_index.get(img_path)
            if entry is None:
                self._seg_cache[img_path] = self._seg_map_to_array(blank)
                continue
            rec = SegMapRecord(
                sample_id="",
                image_path=Path(img_path),
                annotation_disc=Path(entry["annotation_disc"]),
                annotation_cup=Path(entry["annotation_cup"]),
                annotation_type_disc=entry["annotation_type_disc"],
                annotation_type_cup=entry["annotation_type_cup"],
                patient_id=0,
                eye="",
                label=0,
            )
            try:
                disc_mask, cup_mask = load_gt_masks(rec, self._seg_target_size)
                self._seg_cache[img_path] = self._seg_map_to_array(
                    _combine_masks(disc_mask, cup_mask)
                )
                n_ok += 1
            except Exception:
                self._seg_cache[img_path] = self._seg_map_to_array(blank)
        print(f"[GeometryTower] {n_ok}/{len(image_paths)} GT seg maps computed", flush=True)

    def _prepare_fold_unet(self, image_paths: list, eye_train: list, device) -> None:
        """Pre-compute U-Net seg maps, with optional per-fold fine-tuning."""
        from v3.classes.unet_segmenter import UNetSegmenter
        from torch.utils.data import DataLoader as _DL

        if self._weights_path is None:
            raise ValueError("GeometryTower(source='unet') requires weights_path")

        segmenter = UNetSegmenter(
            manifest_path=self._manifest_path,
            normalize=self._unet_normalize,
        )
        state = torch.load(self._weights_path, map_location=segmenter.device)
        segmenter.model.load_state_dict(state.get("model", state))
        segmenter.model.to(segmenter.device).eval()

        if self._finetune_unet_epochs > 0:
            print(
                f"[GeometryTower] fine-tuning U-Net for {self._finetune_unet_epochs} epochs...",
                flush=True,
            )
            ft_loader = _DL(
                UNetFineTuneDataset(
                    self._build_records_from_samples(eye_train),
                    target_size=segmenter.target_size,
                    normalize=self._unet_normalize,
                ),
                batch_size=4, shuffle=True, num_workers=0,
            )
            optimizer = torch.optim.Adam(segmenter.model.parameters(), lr=self._finetune_unet_lr)
            criterion = torch.nn.BCEWithLogitsLoss()
            segmenter.model.train()
            for _ in range(self._finetune_unet_epochs):
                for images, masks in ft_loader:
                    images, masks = images.to(segmenter.device), masks.to(segmenter.device)
                    optimizer.zero_grad()
                    criterion(segmenter.model(images), masks).backward()
                    optimizer.step()
            segmenter.model.eval()

        print(
            f"[GeometryTower] running U-Net inference on {len(image_paths)} images...",
            flush=True,
        )
        records = [
            SegMapRecord(
                sample_id="", image_path=Path(p),
                annotation_disc=Path(p), annotation_cup=Path(p),
                annotation_type_disc="", annotation_type_cup="",
                patient_id=0, eye="", label=0,
            )
            for p in image_paths
        ]
        seg_maps = precompute_unet_seg_maps(records, segmenter, self._unet_threshold)
        for img_path, seg_map in zip(image_paths, seg_maps):
            self._seg_cache[img_path] = self._seg_map_to_array(seg_map)
        print(f"[GeometryTower] {len(seg_maps)} U-Net seg maps cached", flush=True)

    def _build_records_from_samples(self, samples: list) -> list:
        """Build SegMapRecord list from HyperTower sample dicts (for U-Net fine-tuning)."""
        manifest_df = pd.read_csv(self._manifest_path)
        manifest_df["_img_key"] = manifest_df["image_path"].apply(
            lambda p: str(Path(p).resolve())
        )
        manifest_index = manifest_df.set_index("_img_key").to_dict("index")
        records = []
        for s in samples:
            for slot in ("image_1", "image_2"):
                p = s.get(slot)
                if p is None:
                    continue
                key = str(Path(p).resolve())
                entry = manifest_index.get(key)
                if entry is None:
                    continue
                records.append(SegMapRecord(
                    sample_id="",
                    image_path=Path(p),
                    annotation_disc=Path(entry["annotation_disc"]),
                    annotation_cup=Path(entry["annotation_cup"]),
                    annotation_type_disc=entry["annotation_type_disc"],
                    annotation_type_cup=entry["annotation_type_cup"],
                    patient_id=int(s.get("patient_id", 0)),
                    eye=str(s.get("eye", "")),
                    label=int(s.get("label", 0)),
                ))
        return records
