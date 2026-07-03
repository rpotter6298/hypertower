"""fundus_images — disc/cup geometry for fundus image profiles.

Contains all fundus-specific geometry logic: mask parsing, feature computation,
and source-specific loaders.  Profiles whose ImageDataView supports geometry
should implement build_geometry_loader(source, **kwargs) and/or
build_seg_map_loader(source, **kwargs) and delegate here.

Geometry vectors (5 scalar CDR features per eye)
-----------------------------------------------
  build_geometry_loader("gt", contour_dir=...)         → GTGeometryLoader

Seg maps (3-class disc/cup label map per eye, fed to a CNN tower)
-----------------------------------------------------------------
  build_seg_map_loader("gt",   contour_dir=..., ...)   → GTSegMapLoader
  build_seg_map_loader("unet", weights_path=..., ...)  → UNetSegMapLoader
"""
from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from PIL.Image import Resampling
from torch.utils.data import DataLoader, Dataset

EPS = 1e-6

_FEATURE_DIM   = 5
_FEATURE_NAMES = ["area_cdr", "rim_ratio", "vertical_cdr", "horizontal_cdr", "centre_shift"]

_MASK_SIZE = (512, 512)  # canonical rasterisation size; CDR ratios are scale-invariant


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------

def disc_cup_from_mask_image(mask_img: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    """Return binary (disc, cup) masks from a REFUGE-style colour annotation image."""
    arr = np.asarray(mask_img)
    if arr.ndim == 3:
        h, w, c = arr.shape
        border   = np.concatenate(
            [arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]], axis=0
        )
        bg_color = Counter(map(tuple, border)).most_common(1)[0][0]
        colors   = Counter(map(tuple, arr.reshape(-1, c)))
        colors.pop(bg_color, None)
        disc = (~np.all(arr == bg_color, axis=-1)).astype(np.uint8)
        if colors:
            cup_color = min(colors.keys(), key=lambda col: sum(col))
            cup = np.all(arr == cup_color, axis=-1).astype(np.uint8)
        else:
            cup = np.zeros((h, w), dtype=np.uint8)
    else:
        border   = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
        bg_value = Counter(border.tolist()).most_common(1)[0][0]
        disc = (arr != bg_value).astype(np.uint8)
        fg   = arr[arr != bg_value]
        cup  = (arr == int(np.min(fg))).astype(np.uint8) if fg.size > 0 else np.zeros_like(arr)
    cup = (cup > 0) & (disc > 0)
    return disc.astype(np.uint8), cup.astype(np.uint8)


def _contour_to_mask(coords: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Rasterize a polygon contour (Nx2 xy array) into a binary mask of (width, height)."""
    from PIL import ImageDraw
    if coords is None or coords.size == 0:
        return np.zeros((size[1], size[0]), dtype=np.uint8)
    img  = Image.new("L", size, 0)
    draw = ImageDraw.Draw(img)
    draw.polygon([tuple(map(float, pt)) for pt in coords], outline=1, fill=1)
    return np.array(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_geometry_features(disc_mask: np.ndarray, cup_mask: np.ndarray) -> np.ndarray:
    """Compute 5 cup/disc structural descriptors from binary masks.

    Returns float32 [area_cdr, rim_ratio, vertical_cdr, horizontal_cdr, centre_shift].
    """
    disc = (disc_mask > 0).astype(np.float32)
    cup  = (cup_mask  > 0).astype(np.float32)

    disc_area = disc.sum()
    cup_area  = cup.sum()
    area_cdr  = cup_area  / (disc_area + EPS)
    rim_ratio = (disc_area - cup_area) / (disc_area + EPS)

    disc_h = float(np.any(disc > 0, axis=1).sum())
    cup_h  = float(np.any(cup  > 0, axis=1).sum())
    disc_w = float(np.any(disc > 0, axis=0).sum())
    cup_w  = float(np.any(cup  > 0, axis=0).sum())

    vertical_cdr   = cup_h / (disc_h + EPS)
    horizontal_cdr = cup_w / (disc_w + EPS)

    def _centre(m: np.ndarray) -> Tuple[float, float]:
        coords = np.argwhere(m > 0)
        if coords.size == 0:
            return 0.5, 0.5
        ys, xs = coords[:, 0], coords[:, 1]
        return float(xs.mean()) / m.shape[1], float(ys.mean()) / m.shape[0]

    dcx, dcy = _centre(disc)
    ccx, ccy = _centre(cup)
    centre_shift = float(np.hypot(ccx - dcx, ccy - dcy))

    return np.array(
        [area_cdr, rim_ratio, vertical_cdr, horizontal_cdr, centre_shift],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

class GTGeometryLoader:
    """Per-eye 5-feature CDR vectors from PAPILA GT contour annotations.

    Loads contour coordinates from per-expert text files, rasterises them at
    the original image's pixel space (so polygons aren't clipped), then
    computes the CDR feature vector.  Expert masks are merged before feature
    computation to match the seg-map path.
    """

    feature_dim   = _FEATURE_DIM
    feature_names = _FEATURE_NAMES

    def __init__(self, contour_dir: str | Path, *, mask_size: int = _MASK_SIZE[0]) -> None:
        self._dir       = Path(contour_dir)
        self._mask_size = mask_size
        self._cache: dict[tuple, np.ndarray] = {}

    def reset_cache(self) -> None:
        self._cache.clear()

    def precompute(self, samples: Iterable[Tuple[int, str, Path]]) -> None:
        n_ok = 0
        for pid, eye, image_path in samples:
            key = (pid, eye)
            if key in self._cache:
                continue
            with Image.open(image_path) as img:
                image_size = img.size  # (W, H)
            res = _papila_disc_cup_masks(
                pid, eye, self._dir, image_size, self._mask_size,
            )
            if res is None:
                self._cache[key] = np.zeros(self.feature_dim, dtype=np.float32)
            else:
                disc, cup = res
                self._cache[key] = compute_geometry_features(disc, cup)
                n_ok += 1
        print(f"[GTGeometryLoader] {n_ok}/{len(self._cache)} geometry vectors computed", flush=True)

    def all_vectors(self) -> dict:
        return dict(self._cache)


# ---------------------------------------------------------------------------
# Seg-map utilities (shared by GT and UNet seg-map loaders)
# ---------------------------------------------------------------------------

def _combine_disc_cup(disc: np.ndarray, cup: np.ndarray) -> np.ndarray:
    """Merge binary disc + cup masks into a uint8 label map: 0=bg, 1=rim, 2=cup."""
    disc = (disc > 0).astype(np.uint8)
    cup  = ((cup > 0) & (disc > 0)).astype(np.uint8)
    return (disc + cup).astype(np.uint8)


def _crop_to_disc_bbox(seg_map: np.ndarray) -> np.ndarray:
    """Crop a label map tightly to the disc bounding box (anywhere seg_map > 0)."""
    rows = np.any(seg_map > 0, axis=1)
    cols = np.any(seg_map > 0, axis=0)
    if not rows.any():
        return seg_map
    r0, r1 = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1]))
    c0, c1 = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1]))
    return seg_map[r0:r1 + 1, c0:c1 + 1]


def _seg_map_to_array(
    seg_map: np.ndarray, channels: int, target_size: int
) -> np.ndarray:
    """Resize a {0,1,2} seg map and convert to a (C, H, W) float32 array.

    channels=1  →  (1, H, W) values in {0, 0.5, 1.0}
    channels=3  →  (3, H, W) one-hot [bg, rim, cup]
    """
    pil = Image.fromarray(seg_map.astype(np.uint8), mode="L").resize(
        (target_size, target_size), Resampling.NEAREST
    )
    arr = np.array(pil, dtype=np.uint8)
    if channels == 1:
        return (arr.astype(np.float32) / 2.0)[None, :, :]
    if channels == 3:
        return np.stack([
            (arr == 0).astype(np.float32),
            (arr == 1).astype(np.float32),
            (arr == 2).astype(np.float32),
        ], axis=0)
    raise ValueError(f"channels must be 1 or 3, got {channels}")


def _load_papila_contour(path: Path) -> np.ndarray:
    """Load (x, y) contour pairs from a PAPILA whitespace/comma-delimited text file."""
    for delim in (",", None):
        try:
            arr = np.loadtxt(str(path), delimiter=delim, comments="#", dtype=np.float32)
            if arr.size > 0 and arr.ndim >= 1:
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 2)
                if arr.shape[1] >= 2:
                    return arr[:, :2]
        except Exception:
            continue
    return np.zeros((0, 2), dtype=np.float32)


def _papila_disc_cup_masks(
    pid: int, eye: str, contour_dir: Path, image_size: Tuple[int, int],
    mask_size: int, experts: Tuple[int, ...] = (1, 2),
) -> Tuple[np.ndarray, np.ndarray] | None:
    """Average masks across PAPILA experts. Returns (disc, cup) at mask_size, or None."""
    stem = f"RET{pid:03d}{eye}"
    discs, cups = [], []
    for exp in experts:
        disc_path = contour_dir / f"{stem}_disc_exp{exp}.txt"
        cup_path  = contour_dir / f"{stem}_cup_exp{exp}.txt"
        if not disc_path.exists():
            continue
        disc_c = _load_papila_contour(disc_path)
        if len(disc_c) < 3:
            continue
        disc_m = _rasterise_polygon(disc_c, image_size, mask_size)
        if cup_path.exists():
            cup_c = _load_papila_contour(cup_path)
            cup_m = (_rasterise_polygon(cup_c, image_size, mask_size)
                     if len(cup_c) >= 3 else np.zeros_like(disc_m))
        else:
            cup_m = np.zeros_like(disc_m)
        discs.append(disc_m)
        cups.append(cup_m)
    if not discs:
        return None
    disc = (np.mean(discs, axis=0) > 0.5).astype(np.uint8)
    cup  = (np.mean(cups,  axis=0) > 0.5).astype(np.uint8)
    return disc, cup


def _rasterise_polygon(
    coords: np.ndarray, image_size: Tuple[int, int], target_size: int
) -> np.ndarray:
    """Rasterise an (N, 2) polygon contour into a (target_size, target_size) binary mask.

    image_size is (width, height) of the coord space (the original fundus image).
    """
    if coords is None or len(coords) < 3:
        return np.zeros((target_size, target_size), dtype=np.uint8)
    img  = Image.new("L", image_size, 0)
    pts  = [tuple(map(float, p)) for p in coords]
    ImageDraw.Draw(img).polygon(pts, outline=1, fill=1)
    img  = img.resize((target_size, target_size), Resampling.NEAREST)
    return (np.array(img, dtype=np.uint8) > 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Seg-map loaders
# ---------------------------------------------------------------------------

class GTSegMapLoader:
    """Pre-computes per-eye disc/cup seg maps from PAPILA GT contour annotations.

    Output: dict {(pid, eye): np.ndarray (C, H, W) float32} cached for the fold.
    """

    def __init__(
        self,
        contour_dir: str | Path,
        *,
        channels:     int  = 3,
        mask_size:    int  = 512,
        target_size:  int  = 224,
        crop_to_disc: bool = True,
    ) -> None:
        self._dir         = Path(contour_dir)
        self._channels    = channels
        self._mask_size   = mask_size
        self._target_size = target_size
        self._crop        = crop_to_disc
        self._cache: dict[tuple, np.ndarray] = {}

    @property
    def cache_dim(self) -> tuple[int, int, int]:
        return (self._channels, self._target_size, self._target_size)

    def precompute(self, samples: Iterable[Tuple[int, str, Path]]) -> None:
        """samples: iterable of (pid, eye, image_path) tuples."""
        n_ok = 0
        blank = np.zeros((self._mask_size, self._mask_size), dtype=np.uint8)
        for pid, eye, image_path in samples:
            key = (pid, eye)
            if key in self._cache:
                continue
            with Image.open(image_path) as _im:
                im_size = _im.size  # (W, H)
            res = _papila_disc_cup_masks(
                pid, eye, self._dir, im_size, self._mask_size,
            )
            if res is None:
                seg = blank
            else:
                disc, cup = res
                seg = _combine_disc_cup(disc, cup)
                n_ok += 1
            if self._crop:
                seg = _crop_to_disc_bbox(seg)
            self._cache[key] = _seg_map_to_array(seg, self._channels, self._target_size)
        print(
            f"[GTSegMapLoader] {n_ok}/{len(self._cache)} GT seg maps computed "
            f"(channels={self._channels}, target={self._target_size})",
            flush=True,
        )

    def all_seg_maps(self) -> dict:
        return self._cache


class _UNetFTDataset(Dataset):
    """Pre-cached (image_tensor, mask_tensor) pairs for fine-tuning a UNet.

    Decode + resize + normalize + GT mask rasterisation are all deterministic,
    so we do them once at construction and store float32 tensors on CPU.  This
    drops per-batch cost to a tensor lookup + GPU transfer.
    """

    def __init__(
        self,
        records:     list,         # list of (pid, eye, image_path)
        contour_dir: Path,
        segmenter,                 # UNetSegmenter — used for image preprocessing
    ):
        import time
        S = segmenter.target_size
        self._imgs:  list[torch.Tensor] = []
        self._masks: list[torch.Tensor] = []

        print(
            f"[UNetSegMapLoader] pre-caching {len(records)} (image, mask) pairs "
            f"at {S}×{S}...",
            flush=True,
        )
        t0     = time.time()
        report = max(1, len(records) // 4)
        for i, (pid, eye, image_path) in enumerate(records, 1):
            with Image.open(image_path) as raw:
                im_size = raw.size
                img_t   = segmenter.preprocess(raw).detach().cpu()
            res = _papila_disc_cup_masks(pid, eye, contour_dir, im_size, S)
            if res is None:
                disc = np.zeros((S, S), dtype=np.uint8)
                cup  = np.zeros_like(disc)
            else:
                disc, cup = res
            mask_t = torch.from_numpy(np.stack([disc, cup], axis=0).astype(np.float32))
            self._imgs.append(img_t)
            self._masks.append(mask_t)
            if i % report == 0 or i == len(records):
                print(
                    f"  [UNet ft cache] {i}/{len(records)}  ({time.time() - t0:.1f}s)",
                    flush=True,
                )

    def __len__(self) -> int:
        return len(self._imgs)

    def __getitem__(self, idx: int):
        return self._imgs[idx], self._masks[idx]


class _PapilaUNetMaskPipeline:
    """Per-fold UNet mask producer.

    Owns a UNetSegmenter, handles the per-fold lifecycle:
      - reset_weights() restores REFUGE base state (call at start of each fold)
      - finetune(train_samples) fine-tunes on the train split's GT contours
      - predict(samples) returns {(pid, eye): (disc_mask, cup_mask)} raw masks

    Downstream loaders interpret the raw masks differently — seg-map loader
    crops/resizes/encodes for a CNN, geometry-vector loader computes 5 CDR
    features.  Sharing this pipeline avoids duplicating UNet load + finetune
    when both seg-map and feature-vector outputs are needed in one config.
    """

    def __init__(
        self,
        weights_path:        str | Path,
        *,
        contour_dir:         str | Path,
        unet_size:           int   = 512,
        normalize:           str   = "per_image",
        threshold:           float = 0.5,
        finetune_epochs:     int   = 0,
        finetune_lr:         float = 1e-5,
        finetune_batch_size: int   = 4,
        device:              str | None = None,
    ) -> None:
        from v4.classes.accessory.unet import UNetSegmenter

        self._contour_dir   = Path(contour_dir)
        self._threshold     = threshold
        self._ft_epochs     = finetune_epochs
        self._ft_lr         = finetune_lr
        self._ft_batch_size = finetune_batch_size

        self._segmenter = UNetSegmenter(
            target_size=unet_size, normalize=normalize, device=device,
        ).load_weights(Path(weights_path))
        self._base_state = copy.deepcopy(self._segmenter.model.state_dict())

    def reset_weights(self) -> None:
        """Restore base REFUGE weights (undo any prior fine-tuning)."""
        self._segmenter.model.load_state_dict(copy.deepcopy(self._base_state))

    def finetune(self, train_samples: list) -> None:
        """Fine-tune the UNet on the training fold's GT contours."""
        if self._ft_epochs <= 0:
            return
        ds = _UNetFTDataset(train_samples, self._contour_dir, self._segmenter)
        loader = DataLoader(
            ds, batch_size=self._ft_batch_size, shuffle=True, num_workers=0,
        )
        print(
            f"[UNetMaskPipeline] fine-tuning UNet for {self._ft_epochs} epochs "
            f"on {len(train_samples)} samples (lr={self._ft_lr}, "
            f"batch_size={self._ft_batch_size})",
            flush=True,
        )
        self._segmenter.finetune(
            loader, epochs=self._ft_epochs, lr=self._ft_lr,
            log_prefix="[UNet ft]",
        )

    def predict(
        self, samples: Iterable[Tuple[int, str, Path]],
    ) -> dict[tuple, Tuple[np.ndarray, np.ndarray]]:
        """Run inference. Returns {(pid, eye): (disc_mask, cup_mask)} raw uint8."""
        import time
        samples = list(samples)
        if not samples:
            return {}
        print(
            f"[UNetMaskPipeline] running UNet inference on {len(samples)} images...",
            flush=True,
        )
        result: dict[tuple, Tuple[np.ndarray, np.ndarray]] = {}
        t0     = time.time()
        report = max(1, len(samples) // 4)
        for i, (pid, eye, image_path) in enumerate(samples, 1):
            with Image.open(image_path) as raw:
                disc, cup = self._segmenter.predict(raw, threshold=self._threshold)
            result[(pid, eye)] = (disc, cup)
            if i % report == 0 or i == len(samples):
                print(
                    f"  [UNet inf] {i}/{len(samples)}  ({time.time() - t0:.1f}s)",
                    flush=True,
                )
        return result


class UNetSegMapLoader:
    """Per-eye CNN-ready seg maps via a REFUGE-pretrained UNet.

    Wraps a `_PapilaUNetMaskPipeline` and post-processes raw masks into
    (C, H, W) float32 arrays sized for a downstream CNN.

    Output: dict {(pid, eye): np.ndarray (C, H, W) float32}
    """

    def __init__(
        self,
        weights_path: str | Path,
        *,
        contour_dir:         str | Path,
        channels:            int   = 3,
        target_size:         int   = 224,
        unet_size:           int   = 512,
        normalize:           str   = "per_image",
        threshold:           float = 0.5,
        crop_to_disc:        bool  = True,
        finetune_epochs:     int   = 0,
        finetune_lr:         float = 1e-5,
        finetune_batch_size: int   = 4,
        device:              str | None = None,
    ) -> None:
        self._pipeline = _PapilaUNetMaskPipeline(
            weights_path,
            contour_dir=contour_dir,
            unet_size=unet_size,
            normalize=normalize,
            threshold=threshold,
            finetune_epochs=finetune_epochs,
            finetune_lr=finetune_lr,
            finetune_batch_size=finetune_batch_size,
            device=device,
        )
        self._channels    = channels
        self._target_size = target_size
        self._crop        = crop_to_disc
        self._cache: dict[tuple, np.ndarray] = {}

    @property
    def cache_dim(self) -> tuple[int, int, int]:
        return (self._channels, self._target_size, self._target_size)

    def reset_cache(self) -> None:
        self._cache.clear()

    def reset_weights(self) -> None:
        self._pipeline.reset_weights()

    def finetune(self, train_samples: list) -> None:
        self._pipeline.finetune(train_samples)

    def precompute(self, samples: Iterable[Tuple[int, str, Path]]) -> None:
        todo  = [s for s in samples if (s[0], s[1]) not in self._cache]
        masks = self._pipeline.predict(todo)
        for (pid, eye), (disc, cup) in masks.items():
            seg = _combine_disc_cup(disc, cup)
            if self._crop:
                seg = _crop_to_disc_bbox(seg)
            self._cache[(pid, eye)] = _seg_map_to_array(
                seg, self._channels, self._target_size,
            )
        if masks:
            print(
                f"[UNetSegMapLoader] {len(masks)} seg maps cached "
                f"(channels={self._channels}, target={self._target_size})",
                flush=True,
            )

    def all_seg_maps(self) -> dict:
        return self._cache


class UNetGeometryLoader:
    """Per-eye 5-feature CDR vectors derived from UNet-predicted masks.

    Same UNet lifecycle as `UNetSegMapLoader` but the output is a 5-vector
    (compute_geometry_features over the predicted disc/cup masks) rather
    than a CNN-ready seg map.  Designed to slot into ImageEncoder's
    geometry_source mechanism for vector-style geometry injection.
    """

    feature_dim   = _FEATURE_DIM
    feature_names = _FEATURE_NAMES

    def __init__(
        self,
        weights_path: str | Path,
        *,
        contour_dir:         str | Path,
        unet_size:           int   = 512,
        normalize:           str   = "per_image",
        threshold:           float = 0.5,
        finetune_epochs:     int   = 0,
        finetune_lr:         float = 1e-5,
        finetune_batch_size: int   = 4,
        device:              str | None = None,
    ) -> None:
        self._pipeline = _PapilaUNetMaskPipeline(
            weights_path,
            contour_dir=contour_dir,
            unet_size=unet_size,
            normalize=normalize,
            threshold=threshold,
            finetune_epochs=finetune_epochs,
            finetune_lr=finetune_lr,
            finetune_batch_size=finetune_batch_size,
            device=device,
        )
        self._cache: dict[tuple, np.ndarray] = {}

    def reset_cache(self) -> None:
        self._cache.clear()

    def reset_weights(self) -> None:
        self._pipeline.reset_weights()

    def finetune(self, train_samples: list) -> None:
        self._pipeline.finetune(train_samples)

    def precompute(self, samples: Iterable[Tuple[int, str, Path]]) -> None:
        todo  = [s for s in samples if (s[0], s[1]) not in self._cache]
        masks = self._pipeline.predict(todo)
        for (pid, eye), (disc, cup) in masks.items():
            self._cache[(pid, eye)] = compute_geometry_features(disc, cup)
        if masks:
            print(
                f"[UNetGeometryLoader] {len(masks)} geometry vectors cached "
                f"(dim={self.feature_dim})",
                flush=True,
            )

    def all_vectors(self) -> dict:
        return dict(self._cache)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def build_geometry_loader(source: str, **kwargs):
    """Return the appropriate geometry-vector loader for the given source.

    Parameters
    ----------
    source : "gt" | "unet"

    GT kwargs:
        contour_dir
    UNet kwargs:
        weights_path, contour_dir, unet_size=512, normalize="per_image",
        threshold=0.5, finetune_epochs=0, finetune_lr=1e-5,
        finetune_batch_size=4, device=None
    """
    if source == "gt":
        contour_dir = kwargs.get("contour_dir")
        if contour_dir is None:
            raise ValueError("build_geometry_loader source='gt' requires contour_dir")
        return GTGeometryLoader(contour_dir)
    if source == "unet":
        if "weights_path" not in kwargs:
            raise ValueError("build_geometry_loader source='unet' requires weights_path")
        if "contour_dir" not in kwargs:
            raise ValueError(
                "build_geometry_loader source='unet' requires contour_dir "
                "(needed for per-fold fine-tuning, even if finetune_epochs=0)"
            )
        return UNetGeometryLoader(**kwargs)
    raise NotImplementedError(f"build_geometry_loader: source={source!r} not implemented")


class _GTContourBboxLoader:
    """Disc bounding-box loader from PAPILA expert disc contours.

    Computes a square bbox centred on the disc, expanded by ``margin`` ×
    max(disc_w, disc_h). Returned bboxes are in original-image pixel coords
    and may extend past image bounds (clip at crop time).
    """

    def __init__(
        self,
        contour_dir: str | Path,
        *,
        margin:      float = 2.5,
        expert:      int   = 1,
    ) -> None:
        self._contour_dir = Path(contour_dir)
        self._margin      = float(margin)
        self._expert      = int(expert)
        self._cache: dict[tuple, tuple[int, int, int, int] | None] = {}

    def reset_cache(self) -> None:
        self._cache.clear()

    def precompute(self, samples: Iterable[Tuple[int, str, Path]]) -> None:
        for pid, eye, _ in list(samples):
            key = (int(pid), str(eye))
            if key in self._cache:
                continue
            self._cache[key] = self._compute_bbox(*key)

    def bbox_for(self, pid, eye) -> tuple[int, int, int, int] | None:
        key = (int(pid), str(eye))
        if key not in self._cache:
            self._cache[key] = self._compute_bbox(*key)
        return self._cache[key]

    def _compute_bbox(self, pid: int, eye: str) -> tuple[int, int, int, int] | None:
        path = self._contour_dir / f"RET{pid:03d}{eye}_disc_exp{self._expert}.txt"
        if not path.exists():
            return None
        try:
            arr = np.loadtxt(str(path), dtype=np.float32)
        except Exception:
            return None
        if arr.ndim == 1:
            arr = arr.reshape(-1, 2)
        if arr.shape[0] < 3:
            return None
        return _expand_bbox_from_points(arr[:, 0], arr[:, 1], self._margin)


class _UNetBboxLoader:
    """Disc bounding-box loader from U-Net predicted masks.

    Reuses _PapilaUNetMaskPipeline for per-fold fine-tune + inference.
    """

    def __init__(
        self,
        weights_path: str | Path,
        *,
        contour_dir:         str | Path,
        margin:              float = 2.5,
        unet_size:           int   = 512,
        normalize:           str   = "per_image",
        threshold:           float = 0.5,
        finetune_epochs:     int   = 0,
        finetune_lr:         float = 1e-5,
        finetune_batch_size: int   = 4,
        device:              str | None = None,
    ) -> None:
        self._margin    = float(margin)
        self._unet_size = int(unet_size)
        self._pipeline  = _PapilaUNetMaskPipeline(
            weights_path,
            contour_dir=contour_dir,
            unet_size=unet_size,
            normalize=normalize,
            threshold=threshold,
            finetune_epochs=finetune_epochs,
            finetune_lr=finetune_lr,
            finetune_batch_size=finetune_batch_size,
            device=device,
        )
        self._cache: dict[tuple, tuple[int, int, int, int] | None] = {}

    def reset_cache(self) -> None:
        self._cache.clear()

    def reset_weights(self) -> None:
        self._pipeline.reset_weights()

    def finetune(self, train_samples: list) -> None:
        self._pipeline.finetune(train_samples)

    def precompute(self, samples: Iterable[Tuple[int, str, Path]]) -> None:
        samples = list(samples)
        if not samples:
            return
        # Need original-image dims to rescale mask coords back; capture per sample.
        orig_sizes: dict[tuple, tuple[int, int]] = {}
        for pid, eye, image_path in samples:
            try:
                with Image.open(image_path) as im:
                    orig_sizes[(int(pid), str(eye))] = im.size  # (w, h)
            except Exception:
                continue
        masks = self._pipeline.predict(samples)
        for key, (disc, _cup) in masks.items():
            ow, oh = orig_sizes.get(key, (self._unet_size, self._unet_size))
            self._cache[key] = _bbox_from_mask(disc, ow, oh, self._margin)

    def bbox_for(self, pid, eye) -> tuple[int, int, int, int] | None:
        return self._cache.get((int(pid), str(eye)))


def _expand_bbox_from_points(
    xs: np.ndarray, ys: np.ndarray, margin: float
) -> tuple[int, int, int, int]:
    """Square bbox centred on disc centroid, half-side = margin × max(w,h) / 2."""
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half   = max(x1 - x0, y1 - y0) * float(margin) / 2.0
    return (int(round(cx - half)), int(round(cy - half)),
            int(round(cx + half)), int(round(cy + half)))


def _bbox_from_mask(
    mask: np.ndarray, orig_w: int, orig_h: int, margin: float,
) -> tuple[int, int, int, int] | None:
    """Compute original-image bbox from a binary mask at mask resolution."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    sx = float(orig_w) / float(mask.shape[1])
    sy = float(orig_h) / float(mask.shape[0])
    return _expand_bbox_from_points(xs * sx, ys * sy, margin)


def build_disc_bbox_loader(source: str, **kwargs):
    """Return a disc-bbox loader for the given source.

    Parameters
    ----------
    source : "gt" | "unet"

    GT kwargs:
        contour_dir (required), margin=2.5, expert=1
    UNet kwargs:
        weights_path (required), contour_dir (required),
        margin=2.5, unet_size=512, normalize="per_image", threshold=0.5,
        finetune_epochs=0, finetune_lr=1e-5, finetune_batch_size=4, device=None
    """
    if source == "gt":
        if "contour_dir" not in kwargs:
            raise ValueError("build_disc_bbox_loader source='gt' requires contour_dir")
        gt_keys = {"contour_dir", "margin", "expert"}
        return _GTContourBboxLoader(**{k: v for k, v in kwargs.items() if k in gt_keys})
    if source == "unet":
        if "weights_path" not in kwargs:
            raise ValueError("build_disc_bbox_loader source='unet' requires weights_path")
        if "contour_dir" not in kwargs:
            raise ValueError(
                "build_disc_bbox_loader source='unet' requires contour_dir "
                "(needed for per-fold fine-tuning, even if finetune_epochs=0)"
            )
        return _UNetBboxLoader(**kwargs)
    raise NotImplementedError(f"build_disc_bbox_loader: source={source!r} not implemented")


def build_seg_map_loader(source: str, **kwargs):
    """Return the appropriate seg-map loader for the given source string.

    Parameters
    ----------
    source : "gt" | "unet"

    GT kwargs:
        contour_dir, channels=3, mask_size=512, target_size=224, crop_to_disc=True
    UNet kwargs:
        weights_path, contour_dir, channels=3, target_size=224, unet_size=512,
        normalize="per_image", threshold=0.5, crop_to_disc=True,
        finetune_epochs=0, finetune_lr=1e-5, finetune_batch_size=4, device=None
    """
    if source == "gt":
        if "contour_dir" not in kwargs:
            raise ValueError("build_seg_map_loader source='gt' requires contour_dir")
        gt_keys = {"contour_dir", "channels", "mask_size", "target_size", "crop_to_disc"}
        return GTSegMapLoader(**{k: v for k, v in kwargs.items() if k in gt_keys})
    if source == "unet":
        if "weights_path" not in kwargs:
            raise ValueError("build_seg_map_loader source='unet' requires weights_path")
        if "contour_dir" not in kwargs:
            raise ValueError(
                "build_seg_map_loader source='unet' requires contour_dir "
                "(needed for per-fold fine-tuning, even if finetune_epochs=0)"
            )
        return UNetSegMapLoader(**kwargs)
    raise NotImplementedError(f"build_seg_map_loader: source={source!r} not implemented")
