"""Segmentation-map CNN for glaucoma grading.

Trains a CNN on combined disc/cup segmentation maps — pixel values
  0 = background,  1 = disc (rim only),  2 = cup
— instead of raw RGB fundus images, forcing the model to learn purely
from optic nerve head geometry (CDR, rim width, cup location, etc.).

Two segmentation sources are supported:
  gt    – rasterise expert contour/mask annotations directly (pure NumPy/PIL,
          no CUDA — safe in DataLoader worker processes)
  unet  – run a trained UNetSegmenter on the raw fundus image

Usage (import from training script):
    from v3.classes.seg_cnn import SegMapRecord, SegMapDataset, SegCNN, seg_map_to_tensor
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
    """
    Combine binary disc and cup masks into a 3-class label map.

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
    """
    Crop a seg map tightly to the disc bounding box.

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
    """
    Convert an (H, W) seg map with values {0, 1, 2} to a float tensor.

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
    for delimiter in (",", None):
        try:
            arr = np.loadtxt(str(path), delimiter=delimiter, comments="#", dtype=np.float32)
            if arr.size > 0:
                break
        except Exception:
            arr = np.zeros((0, 2), dtype=np.float32)
    if arr.size == 0 or arr.ndim == 1:
        return np.zeros((0, 2), dtype=np.float32)
    if arr.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    return arr[:, :2]


def _contour_to_mask(
    coords: np.ndarray, image_size: Tuple[int, int], target_size: int
) -> np.ndarray:
    """
    Rasterise a polygon defined by (x, y) coords into a binary mask.

    image_size is the (width, height) of the original fundus image — the
    coordinate space the contour was annotated in.  The mask is drawn at
    that resolution then resized to target_size, matching UNetSegmenter's
    behaviour and avoiding off-canvas clipping.
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
    """
    Extract disc and cup binary masks from a segmentation image file.

    Handles both grayscale label images (e.g. REFUGE .bmp) and
    RGB colour-coded masks.  Returns (disc_mask, cup_mask) both at
    target_size × target_size.
    """
    raw = Image.open(mask_path)
    arr = np.array(raw)

    if arr.ndim == 2:
        # Grayscale: identify background from edge statistics
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
        edges_rgb = np.concatenate(
            [arr[0], arr[-1], arr[:, 0], arr[:, -1]], axis=0
        )
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

    # Resize to target_size with nearest-neighbour to preserve binary values
    def _resize(m: np.ndarray) -> np.ndarray:
        pil = Image.fromarray((m > 0).astype(np.uint8) * 255)
        pil = pil.resize((target_size, target_size), Resampling.NEAREST)
        return (np.array(pil) > 0).astype(np.uint8)

    return _resize(disc_arr), _resize(cup_arr)


def load_gt_masks(rec: "SegMapRecord", target_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load GT disc + cup masks for one record.

    Handles annotation_type "contour" (x,y text file) and "mask" (image file).
    Returns (disc_mask, cup_mask) as uint8 arrays of shape (target_size, target_size).
    """
    disc_mask: Optional[np.ndarray] = None
    cup_mask:  Optional[np.ndarray] = None

    # Get original image size so contour coordinates are drawn in the right space
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

    # Structural prior: cup must lie within disc
    cup_mask = (cup_mask > 0) & (disc_mask > 0)
    return disc_mask.astype(np.uint8), cup_mask.astype(np.uint8)


# ---------------------------------------------------------------------------
# U-Net fine-tuning dataset
# ---------------------------------------------------------------------------

class UNetFineTuneDataset(Dataset):
    """
    Loads (image_tensor, mask_tensor) pairs for fine-tuning the U-Net on
    PAPILA GT annotations.  Uses the same preprocessing as UNetSegmenter
    so the fine-tuned weights are compatible with inference.
    """

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
# U-Net precomputation (run once per full record list, not per fold)
# ---------------------------------------------------------------------------

def precompute_unet_seg_maps(
    records: List["SegMapRecord"],
    segmenter,
    threshold: float = 0.5,
) -> List[np.ndarray]:
    """
    Run the U-Net on every record and return a list of combined seg maps.

    Call this once before the CV loop and pass the results to each fold's
    SegMapDataset via precomputed_seg_maps, so the U-Net isn't re-run per fold.
    """
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
# Dataset
# ---------------------------------------------------------------------------

class SegMapDataset(Dataset):
    """
    PyTorch Dataset that yields (seg_tensor, label) pairs.

    Parameters
    ----------
    records      : list of SegMapRecord
    target_size  : CNN input spatial size (images are resized to this)
    channels     : 1 = single-channel label map;  3 = one-hot three channels
    augment      : apply random flips + rotation (for training set)
    unet_segmenter : if provided, use U-Net predictions instead of GT masks;
                     must be a loaded UNetSegmenter with model weights set
    unet_threshold : threshold for U-Net logit → binary mask
    seg_target_size: resolution at which GT masks are rasterised (or U-Net
                     output size).  Default 512 matches UNetSegmenter default.
    crop_to_disc   : crop the seg map tightly to the disc bounding box before
                     resizing to target_size (default True — eliminates the
                     background zeros that make up most of the full image)
    """

    def __init__(
        self,
        records: List[SegMapRecord],
        target_size: int = 224,
        channels: int = 3,
        augment: bool = False,
        unet_segmenter=None,
        unet_threshold: float = 0.5,
        seg_target_size: int = 512,
        crop_to_disc: bool = True,
        precomputed_seg_maps: Optional[List[np.ndarray]] = None,
    ) -> None:
        self.records = records
        self.target_size = target_size
        self.channels = channels
        self.augment = augment
        self.seg_target_size = seg_target_size
        self.crop_to_disc = crop_to_disc

        if precomputed_seg_maps is not None:
            self._seg_maps = precomputed_seg_maps
        elif unet_segmenter is not None:
            self._seg_maps = precompute_unet_seg_maps(
                records, unet_segmenter, unet_threshold
            )
        else:
            self._seg_maps = None

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.records)

    # ------------------------------------------------------------------
    def _augment(self, seg_map: np.ndarray) -> np.ndarray:
        """Random flips + 90° rotations (label-safe since NEAREST resize)."""
        if np.random.rand() < 0.5:
            seg_map = np.fliplr(seg_map)
        if np.random.rand() < 0.5:
            seg_map = np.flipud(seg_map)
        k = np.random.randint(0, 4)
        if k:
            seg_map = np.rot90(seg_map, k=k)
        return np.ascontiguousarray(seg_map)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        rec = self.records[idx]

        if self._seg_maps is not None:
            seg_map = self._seg_maps[idx]
        else:
            disc_mask, cup_mask = load_gt_masks(rec, self.seg_target_size)
            seg_map = _combine_masks(disc_mask, cup_mask)

        if self.crop_to_disc:
            seg_map = crop_to_disc(seg_map)

        if self.augment:
            seg_map = self._augment(seg_map)

        tensor = seg_map_to_tensor(seg_map, self.channels, self.target_size)
        return tensor, rec.label


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SegCNN(nn.Module):
    """
    Pretrained CNN backbone adapted for segmentation-map input.

    Parameters
    ----------
    num_classes : output classes (2 for binary glaucoma grading)
    backbone    : "resnet18" | "resnet50" | "efficientnet_b0"
    pretrained  : initialise with ImageNet weights (recommended even for
                  non-RGB input — transfer generalises across domains)
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

        # Adapt first conv layer if in_channels ≠ 3
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
                # Average pretrained RGB weights across channel dim
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

    # ------------------------------------------------------------------
    @staticmethod
    def _find_first_conv(module: nn.Module) -> nn.Conv2d:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                return m
        raise RuntimeError("No Conv2d found in backbone")

    @staticmethod
    def _replace_first_conv(module: nn.Module, new_conv: nn.Conv2d) -> None:
        """Replace the first Conv2d in-place (handles resnet and efficientnet)."""
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

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if feats.dim() > 2:
            feats = feats.flatten(1)
        return self.head(feats)
