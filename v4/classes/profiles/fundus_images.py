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
    """Pre-computes per-eye geometry vectors from PAPILA GT contour annotations.

    File naming: RET{pid:03d}{eye}_{disc|cup}_exp{n}.txt
    Averages exp1 and exp2 when both are present; zero vector for missing entries.

    Usage:
        loader = GTGeometryLoader(contour_dir)
        loader.precompute(df, patient_col="Patient ID")
        vecs = loader.all_vectors()   # {(pid, eye): ndarray}
    """

    _EXPERTS = (1, 2)

    feature_dim   = _FEATURE_DIM
    feature_names = _FEATURE_NAMES

    def __init__(self, contour_dir: str | Path) -> None:
        self._dir   = Path(contour_dir)
        self._cache: dict[tuple, np.ndarray] = {}

    def precompute(self, df, patient_col: str = "Patient ID") -> None:
        n_ok = 0
        for _, row in df.iterrows():
            pid = int(row[patient_col])
            eye = str(row.get("eyeID", "OD"))
            key = (pid, eye)
            if key in self._cache:
                continue
            vec = self._compute(pid, eye)
            self._cache[key] = vec if vec is not None else np.zeros(self.feature_dim, dtype=np.float32)
            if vec is not None:
                n_ok += 1
        print(f"[GTGeometryLoader] {n_ok}/{len(self._cache)} geometry vectors computed", flush=True)

    def all_vectors(self) -> dict:
        return dict(self._cache)

    def _compute(self, pid: int, eye: str) -> "np.ndarray | None":
        stem = f"RET{pid:03d}{eye}"
        vecs: list[np.ndarray] = []
        for exp in self._EXPERTS:
            disc_path = self._dir / f"{stem}_disc_exp{exp}.txt"
            cup_path  = self._dir / f"{stem}_cup_exp{exp}.txt"
            if not disc_path.exists():
                continue
            try:
                disc_c = np.loadtxt(disc_path)
                if disc_c.ndim == 1:
                    disc_c = disc_c.reshape(-1, 2)
                disc_mask = _contour_to_mask(disc_c, _MASK_SIZE)
                if cup_path.exists():
                    cup_c = np.loadtxt(cup_path)
                    if cup_c.ndim == 1:
                        cup_c = cup_c.reshape(-1, 2)
                    cup_mask = _contour_to_mask(cup_c, _MASK_SIZE)
                else:
                    cup_mask = np.zeros((_MASK_SIZE[1], _MASK_SIZE[0]), dtype=np.uint8)
                cup_mask = ((cup_mask > 0) & (disc_mask > 0)).astype(np.uint8)
                vecs.append(compute_geometry_features(disc_mask, cup_mask))
            except Exception:
                continue
        if not vecs:
            return None
        return np.stack(vecs).mean(axis=0).astype(np.float32)


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


class UNetSegMapLoader:
    """Pre-computes per-eye seg maps via a REFUGE-pretrained UNet.

    Optionally fine-tunes the UNet per fold on the training split's GT contours.

    Output: dict {(pid, eye): np.ndarray (C, H, W) float32} cached for the fold.
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
        from v4.classes.accessory.unet import UNetSegmenter

        self._weights_path        = Path(weights_path)
        self._contour_dir         = Path(contour_dir)
        self._channels            = channels
        self._target_size         = target_size
        self._threshold           = threshold
        self._crop                = crop_to_disc
        self._ft_epochs           = finetune_epochs
        self._ft_lr               = finetune_lr
        self._ft_batch_size       = finetune_batch_size

        self._segmenter = UNetSegmenter(
            target_size=unet_size, normalize=normalize, device=device,
        ).load_weights(self._weights_path)
        self._base_state = copy.deepcopy(self._segmenter.model.state_dict())
        self._cache: dict[tuple, np.ndarray] = {}

    @property
    def cache_dim(self) -> tuple[int, int, int]:
        return (self._channels, self._target_size, self._target_size)

    def reset_cache(self) -> None:
        """Clear cached seg maps (call between folds)."""
        self._cache.clear()

    def reset_weights(self) -> None:
        """Restore base REFUGE weights (undo any prior fine-tuning)."""
        self._segmenter.model.load_state_dict(copy.deepcopy(self._base_state))

    def finetune(self, train_samples: list) -> None:
        """Fine-tune the UNet on the training fold's GT contours.

        train_samples: list of (pid, eye, image_path) tuples — train split only.
        """
        if self._ft_epochs <= 0:
            return
        ds = _UNetFTDataset(train_samples, self._contour_dir, self._segmenter)
        loader = DataLoader(
            ds, batch_size=self._ft_batch_size, shuffle=True, num_workers=0,
        )
        print(
            f"[UNetSegMapLoader] fine-tuning UNet for {self._ft_epochs} epochs "
            f"on {len(train_samples)} samples (lr={self._ft_lr}, "
            f"batch_size={self._ft_batch_size})",
            flush=True,
        )
        self._segmenter.finetune(
            loader, epochs=self._ft_epochs, lr=self._ft_lr,
            log_prefix="[UNet ft]",
        )

    def precompute(self, samples: Iterable[Tuple[int, str, Path]]) -> None:
        """Run UNet inference on every sample and cache the resulting seg map."""
        import time
        samples = list(samples)
        todo    = [s for s in samples if (s[0], s[1]) not in self._cache]
        if not todo:
            return
        print(
            f"[UNetSegMapLoader] running UNet inference on {len(todo)} images...",
            flush=True,
        )
        t0     = time.time()
        report = max(1, len(todo) // 4)
        for i, (pid, eye, image_path) in enumerate(todo, 1):
            with Image.open(image_path) as raw:
                disc, cup = self._segmenter.predict(raw, threshold=self._threshold)
            seg = _combine_disc_cup(disc, cup)
            if self._crop:
                seg = _crop_to_disc_bbox(seg)
            self._cache[(pid, eye)] = _seg_map_to_array(
                seg, self._channels, self._target_size,
            )
            if i % report == 0 or i == len(todo):
                print(
                    f"  [UNet inf] {i}/{len(todo)}  ({time.time() - t0:.1f}s)",
                    flush=True,
                )
        print(
            f"[UNetSegMapLoader] {len(todo)} seg maps cached via UNet "
            f"(channels={self._channels}, target={self._target_size})",
            flush=True,
        )

    def all_seg_maps(self) -> dict:
        return self._cache


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def build_geometry_loader(source: str, **kwargs):
    """Return the appropriate geometry-vector loader for the given source string.

    Parameters
    ----------
    source      : "gt" | "unet"
    contour_dir : (gt) path to contour annotation directory
    """
    if source == "gt":
        contour_dir = kwargs.get("contour_dir")
        if contour_dir is None:
            raise ValueError("build_geometry_loader source='gt' requires contour_dir")
        return GTGeometryLoader(contour_dir)
    raise NotImplementedError(f"build_geometry_loader: source={source!r} not implemented")


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
        return GTSegMapLoader(**kwargs)
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
