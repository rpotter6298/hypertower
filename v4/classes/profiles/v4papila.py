"""v4papila — self-contained PAPILA data module.

Public contract (v4 orchestrator interface)
-------------------------------------------
    bundle = build_data(args: dict) -> PapilaBundle

PapilaBundle exposes:
    .df                     full preprocessed DataFrame (for split building)
    .label_col, .patient_col, .feature_dim
    .id_names               tuple of semantic names for each entity_id slot
                            e.g. ("patient_id", "eye") — used by orchestrator for logging
    .matrix                 ClinicalDataView (all eyes)
    .matrix.od / .matrix.os scoped views (OD or OS only)
    .image                  ImageDataView (all eyes)
    .image.od  / .image.os  scoped views
    .build_shells(df, *, level) -> LoaderShell

DataView interface (consumed by towers' get_sample)
----------------------------------------------------
Both views accept positional id slots (*ids) matching entity_id tuple positions.
Semantic names for each position are in view.id_names.

ClinicalDataView:
    .feature_dim
    .id_names               e.g. ("patient_id", "eye")
    .vectorize_entity(*ids) -> np.ndarray
    .side_map               -> dict mapping generic keys {"a", "b"} to id_1 values
    .od, .os                -> scoped ClinicalDataView

ImageDataView:
    .id_names               e.g. ("patient_id", "eye")
    .get_image_path(*ids)   -> Path
    .load_image(*ids)       -> PIL.Image
    .side_map               -> dict mapping generic keys {"a", "b"} to id_1 values
    .od, .os                -> scoped ImageDataView
"""
from __future__ import annotations

import sys
from functools import cached_property
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from v4.classes.loaders.image_loader import CachedImageLoader, call_preprocessor
from v4.classes.dataset import DataBundle, LoaderShell, ShellEntry


# ---------------------------------------------------------------------------
# Pachymetry → IOP correction (PAPILA Table 3)
# ---------------------------------------------------------------------------

_PACHY_TABLE: Dict[int, int] = {
    475: +5, 485: +4, 495: +4, 505: +3, 515: +2,
    525: +1, 535: +1, 545:  0, 555: -1, 565: -1,
    575: -2, 585: -3, 595: -4, 605: -4, 615: -5,
}
_PACHY_KEYS = np.array(sorted(_PACHY_TABLE.keys()))


def _nearest_pachy_key(x: float) -> int:
    return int(_PACHY_KEYS[int(np.argmin(np.abs(_PACHY_KEYS - float(x))))])


def _fit_perkins_converter(
    frames: List[pd.DataFrame], method: str
) -> Callable[[float, Optional[float]], float]:
    combined  = pd.concat(frames, ignore_index=True)
    paired    = combined.dropna(subset=["Pneumatic", "Perkins"])
    if len(paired) == 0:
        raise ValueError("No paired Pneumatic+Perkins rows; cannot fit converter.")
    pneumatic = paired["Pneumatic"].values.astype(float)
    perkins   = paired["Perkins"].values.astype(float)

    if method == "ratio":
        ratio = float((pneumatic / perkins).mean())
        def _conv(p: float, pachy: Optional[float] = None) -> float:
            return p * ratio
        return _conv

    elif method == "ols":
        from scipy import stats as _stats
        slope, intercept, *_ = _stats.linregress(perkins, pneumatic)
        slope, intercept = float(slope), float(intercept)
        def _conv(p: float, pachy: Optional[float] = None) -> float:
            return p * slope + intercept
        return _conv

    elif method == "lad":
        from scipy import stats as _stats
        from scipy.optimize import minimize as _minimize
        slope0, intercept0, *_ = _stats.linregress(perkins, pneumatic)
        def _lad_loss(params):
            a, b = params
            return np.abs(pneumatic - (a * perkins + b)).mean()
        res = _minimize(_lad_loss, x0=[slope0, intercept0], method="Nelder-Mead")
        slope, intercept = float(res.x[0]), float(res.x[1])
        def _conv(p: float, pachy: Optional[float] = None) -> float:
            return p * slope + intercept
        return _conv

    elif method == "multi":
        from numpy.linalg import lstsq as _lstsq
        pm = combined.dropna(subset=["Pneumatic", "Perkins", "Pachymetry"])
        if len(pm) == 0:
            raise ValueError("No Pneumatic+Perkins+Pachymetry rows; cannot fit multi.")
        pneu  = pm["Pneumatic"].values.astype(float)
        perk  = pm["Perkins"].values.astype(float)
        pv    = pm["Pachymetry"].values.astype(float)
        X     = np.column_stack([perk, pv, np.ones(len(perk))])
        coeffs, *_ = _lstsq(X, pneu, rcond=None)
        slope, pachy_coef, intercept = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
        fallback = float(pv.mean())
        def _conv(p: float, pachy: Optional[float] = None) -> float:
            pval = pachy if (pachy is not None and not np.isnan(pachy)) else fallback
            return p * slope + pachy_coef * pval + intercept
        return _conv

    else:
        raise ValueError(f"Unknown iop_corr_method: {method!r}. Choose ratio/ols/lad/multi.")


def _pick_iop(row: pd.Series, converter: Callable) -> float:
    pneumatic = row.get("Pneumatic", np.nan)
    if not pd.isna(pneumatic):
        return float(pneumatic)
    perkins = row.get("Perkins", np.nan)
    if pd.isna(perkins):
        return np.nan
    pachy = row.get("Pachymetry", np.nan)
    return converter(float(perkins), None if pd.isna(pachy) else float(pachy))


def _correct_iop(raw_iop: float, pachy: float) -> float:
    if pd.isna(raw_iop):
        return np.nan
    if pd.isna(pachy):
        return float(raw_iop)
    key = _nearest_pachy_key(float(pachy))
    return float(raw_iop) + float(_PACHY_TABLE[key])


def _apply_iop_and_drop_md(
    df: pd.DataFrame, converter: Callable, drop_raw: bool = False
) -> pd.DataFrame:
    df["IOP_raw"] = df.apply(lambda row: _pick_iop(row, converter), axis=1)
    pachy = df.get("Pachymetry", pd.Series(np.nan, index=df.index))
    df["IOP_corr"] = [
        _correct_iop(r, p) for r, p in zip(df["IOP_raw"].values, pachy.values)
    ]
    # Drop the raw IOP source columns (Pneumatic, Perkins) since their info
    # is already absorbed into IOP_corr.  VF_MD is intentionally NOT dropped
    # here — it stays available as a target for auxiliary/regression tasks
    # (PapilaBundle exposes it via .vf_md_target / .get_vf_md()).  It must
    # however be added to `exclude_cols` so it is never used as a feature.
    drop = [c for c in ("Pneumatic", "Perkins") if c in df.columns]
    if drop_raw:
        drop.append("IOP_raw")
    if drop:
        df.drop(columns=drop, inplace=True)
    return df


def _canonicalize_eye_column(df: pd.DataFrame) -> None:
    if "eyeID" in df.columns:
        src = "eyeID"
    else:
        src = next((c for c in df.columns if "eye" in c.lower()), None)
        if src is None:
            df["eyeID"] = "OS"
            return

    def norm(v):
        if pd.isna(v):
            return None
        x = str(v).strip().upper()
        if x in {"OS", "L", "LEFT", "0"}:  return "OS"
        if x in {"OD", "R", "RIGHT", "1"}: return "OD"
        try:
            num = int(float(x))
            return "OD" if num % 2 == 1 else "OS"
        except Exception:
            return None

    mapped = df[src].map(norm)
    uniq   = {u for u in mapped.dropna().unique().tolist()}
    if not uniq.issubset({"OS", "OD"}):
        raise ValueError(f"eyeID must be binary; found {sorted(uniq)}")
    df["eyeID"] = mapped.fillna("OS")


# ---------------------------------------------------------------------------
# DataView classes
# ---------------------------------------------------------------------------

class ClinicalDataView:
    """Tabular feature view over a (possibly eye-filtered) clinical DataFrame.

    Exposes feature_dim and vectorize_entity so towers can retrieve
    feature vectors by entity identity without knowing about the DataFrame.

    Scoped views (OD or OS only) are accessed via .od and .os properties.

    id_names gives semantic labels for each positional slot in an entity_id tuple,
    e.g. ("patient_id", "eye").  The orchestrator uses this for logging without
    needing to know PAPILA-specific field names itself.
    """

    # PAPILA canonical side keys used in ShellEntry entity_ids
    SIDE_A = "OD"
    SIDE_B = "OS"

    # Semantic name for each entity_id position (id_0, id_1, ...)
    id_names: tuple[str, ...] = ("patient_id", "eye")

    def __init__(
        self,
        df:           pd.DataFrame,
        patient_col:  str,
        scalar_cols:  list[str],
        cat_cols:     list[str],
        scalar_stats: dict,
        cat_maps:     dict,
        *,
        eye_filter: str | None = None,  # "OD", "OS", or None (all eyes)
    ):
        self._df          = df
        self.patient_col  = patient_col
        self.scalar_cols  = scalar_cols
        self.cat_cols     = cat_cols
        self.scalar_stats = scalar_stats
        self.cat_maps     = cat_maps
        self._eye_filter  = eye_filter

        # Build a (patient_id, eyeID) → row index for fast lookup
        if "eyeID" in df.columns:
            self._idx = df.set_index([patient_col, "eyeID"])
        else:
            self._idx = df.set_index(patient_col)

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def feature_dim(self) -> int:
        n_scalar = len(self.scalar_cols)
        n_cat    = sum(len(m) for m in self.cat_maps.values())
        return n_scalar + n_cat + n_scalar  # scalars + one-hots + missing flags

    # ── Optional explainability hooks ────────────────────────────────────────
    #
    # feature_names + feature_groups describe how the encoded feature_dim
    # vector maps back to the original column space. They're consumed by
    # v4.classes.accessory.explainability.permutation_importance via the
    # caller (F8); profiles without these methods skip per-column importance.

    @cached_property
    def feature_names(self) -> list[str]:
        """Human-readable name for each original column (used as group label)."""
        return list(self.scalar_cols) + list(self.cat_cols)

    @cached_property
    def feature_groups(self) -> list[list[int]]:
        """Encoded-vector indices grouped by original column.

        Each entry is the set of model-input dimensions that encode one
        conceptual feature: a scalar groups its value + missing-flag dim,
        a categorical groups all of its one-hot dims.
        Order matches ``feature_names``.
        """
        n_scalar = len(self.scalar_cols)
        n_cat_total = sum(len(m) for m in self.cat_maps.values())
        groups: list[list[int]] = []
        for i in range(n_scalar):
            groups.append([i, n_scalar + n_cat_total + i])
        offset = n_scalar
        for col in self.cat_cols:
            k = len(self.cat_maps[col])
            groups.append(list(range(offset, offset + k)))
            offset += k
        return groups

    def vectorize_entity(self, *ids) -> np.ndarray:
        """Return the feature vector for an entity identified by positional ids.

        Positional slots match entity_id tuple positions (see id_names).
        For PAPILA: ids = (id_0, id_1) = (patient_id, eye).
        """
        try:
            row = self._idx.loc[ids if len(ids) > 1 else ids[0]].copy()
        except KeyError:
            names = self.id_names[:len(ids)]
            raise KeyError(
                f"ClinicalDataView: no row found for {dict(zip(names, ids))}"
            )
        # set_index removes index-level columns from the row; restore any that
        # _vectorize_row needs (e.g. eyeID is a cat feature AND an index level)
        idx_names = (self._idx.index.names
                     if hasattr(self._idx.index, 'names')
                     else [self._idx.index.name])
        for name, val in zip(idx_names, ids if len(ids) > 1 else [ids[0]]):
            if name not in row.index:
                row[name] = val
        return self._vectorize_row(row)

    # ── Scoped views ─────────────────────────────────────────────────────────

    @property
    def side_map(self) -> dict[str, str]:
        """Generic side-key → dataset side string. Towers use this for patient-level shells."""
        return {"a": self.SIDE_A, "b": self.SIDE_B}

    @cached_property
    def od(self) -> "ClinicalDataView":
        return self._scoped(self.SIDE_A)

    @cached_property
    def os(self) -> "ClinicalDataView":
        return self._scoped(self.SIDE_B)

    def _scoped(self, eye: str) -> "ClinicalDataView":
        sub = self._df[self._df["eyeID"] == eye].reset_index(drop=True)
        return ClinicalDataView(
            df=sub,
            patient_col=self.patient_col,
            scalar_cols=self.scalar_cols,
            cat_cols=self.cat_cols,
            scalar_stats=self.scalar_stats,
            cat_maps=self.cat_maps,
            eye_filter=eye,
        )

    # ── Internal ─────────────────────────────────────────────────────────────

    def _vectorize_row(self, row: pd.Series) -> np.ndarray:
        feats: list[float] = []
        miss:  list[float] = []
        for col in self.scalar_cols:
            v = pd.to_numeric(row.get(col), errors="coerce")
            if pd.isna(v):
                miss.append(1.0)
                v = self.scalar_stats[col]["median"]
            else:
                miss.append(0.0)
            lo = self.scalar_stats[col]["min"]
            hi = self.scalar_stats[col]["max"]
            feats.append((float(v) - lo) / (hi - lo) if hi > lo else 0.0)
        for col in self.cat_cols:
            mapping = self.cat_maps[col]
            one = [0.0] * len(mapping)
            key = row.get(col)
            one[mapping.get(key, 0)] = 1.0
            feats.extend(one)
        feats.extend(miss)
        return np.asarray(feats, dtype=np.float32)


class ImageDataView:
    """Image path and loading view over a (possibly eye-filtered) DataFrame.

    Provides get_image_path and load_image keyed by positional id slots.
    Scoped views (.od, .os) are available for single-side towers.
    The optional image_cache is a shared CachedImageLoader for the run.

    id_names gives semantic labels for each positional slot in an entity_id tuple,
    e.g. ("patient_id", "eye").
    """

    SIDE_A = "OD"
    SIDE_B = "OS"

    id_names: tuple[str, ...] = ("patient_id", "eye")

    def __init__(
        self,
        df:                pd.DataFrame,
        patient_col:       str,
        image_dir:         str,
        filename_template: str,
        preprocessor:      Callable | None = None,
        image_cache:       CachedImageLoader | None = None,
        *,
        eye_filter: str | None = None,
    ):
        self._df               = df
        self.patient_col       = patient_col
        self.image_dir         = Path(image_dir)
        self.filename_template = filename_template
        self.preprocessor      = preprocessor
        self.image_cache       = image_cache
        self._eye_filter       = eye_filter

        if "eyeID" in df.columns:
            self._idx = df.set_index([patient_col, "eyeID"])
        else:
            self._idx = df.set_index(patient_col)

    # ── Public API ──────────────────────────────────────────────────────────

    def get_image_path(self, *ids) -> Path:
        """Return the image path for an entity identified by positional ids.

        For PAPILA: ids = (id_0, id_1) = (patient_id, eye).
        """
        id_0, id_1 = ids  # PAPILA always uses two slots
        try:
            row = self._idx.loc[(id_0, id_1)]
        except KeyError:
            names = self.id_names[:len(ids)]
            raise KeyError(
                f"ImageDataView: no row for {dict(zip(names, ids))}"
            )
        pid = int(row[self.patient_col]) if self.patient_col in row.index else int(id_0)
        return self.image_dir / self.filename_template.format(pid=pid, eye=id_1)

    def load_image(self, *ids) -> Image.Image:
        """Load image for an entity identified by positional ids."""
        path = self.get_image_path(*ids)
        if self.image_cache is not None:
            return self.image_cache.load(path, preprocessor=self.preprocessor)
        img = Image.open(path).convert("RGB")
        if self.preprocessor is not None:
            img = call_preprocessor(self.preprocessor, img, path)
        return img

    # ── Scoped views ─────────────────────────────────────────────────────────

    @cached_property
    def od(self) -> "ImageDataView":
        return self._scoped(self.SIDE_A)

    @cached_property
    def os(self) -> "ImageDataView":
        return self._scoped(self.SIDE_B)

    @property
    def side_map(self) -> dict[str, str]:
        """Generic side-key → dataset side string. Towers use this for patient-level shells."""
        return {"a": self.SIDE_A, "b": self.SIDE_B}

    def _scoped(self, eye: str) -> "ImageDataView":
        sub = self._df[self._df["eyeID"] == eye].reset_index(drop=True)
        return ImageDataView(
            df=sub,
            patient_col=self.patient_col,
            image_dir=str(self.image_dir),
            filename_template=self.filename_template,
            preprocessor=self.preprocessor,
            image_cache=self.image_cache,
            eye_filter=eye,
        )

    # ── Geometry hook ─────────────────────────────────────────────────────────

    def _resolve_paths(self, kwargs: dict) -> dict:
        """Resolve any *_dir / *_path kwargs against the repo root."""
        repo_root = Path(__file__).resolve().parents[3]
        out = {}
        for k, v in kwargs.items():
            if (k.endswith("_dir") or k.endswith("_path")) and v is not None:
                p = Path(v)
                out[k] = str(repo_root / p) if not p.is_absolute() else v
            else:
                out[k] = v
        return out

    def build_geometry_loader(self, source: str, **kwargs):
        """Return a geometry-vector loader (delegates to fundus_images)."""
        from v4.classes.profiles.fundus_images import build_geometry_loader as _build
        return _build(source, **self._resolve_paths(kwargs))

    def build_seg_map_loader(self, source: str, **kwargs):
        """Return a seg-map loader (delegates to fundus_images)."""
        from v4.classes.profiles.fundus_images import build_seg_map_loader as _build
        return _build(source, **self._resolve_paths(kwargs))

    def build_disc_bbox_loader(self, source: str, **kwargs):
        """Return a disc bounding-box loader for crop-to-disc preprocessing."""
        from v4.classes.profiles.fundus_images import build_disc_bbox_loader as _build
        kwargs.setdefault("contour_dir", self._DEFAULT_CONTOUR_DIR)
        return _build(source, **self._resolve_paths(kwargs))

    # ── Optional explainability hooks ────────────────────────────────────────
    #
    # These methods are consumed by v4.classes.accessory.explainability via
    # getattr — they're optional on the data view, so a different profile
    # without disc annotations can simply not define them and GradCAM/overlay
    # will still work (region-of-interest analysis is skipped).
    #
    # Eval-crop convention: PAPILA training resizes the short side to 256
    # then center-crops 224×224. The methods below mirror that so the disc
    # mask and the display image stay aligned with what the image tower sees.

    _EVAL_RESIZE = 256
    _EVAL_CROP   = 224
    _DEFAULT_CONTOUR_DIR = "Papila/ExpertsSegmentations/Contours"

    def eval_image_pil(self, *ids) -> Image.Image:
        """Load image and apply the 256-resize + 224-center-crop used at eval time."""
        from torchvision.transforms import functional as TF
        from torchvision.transforms import InterpolationMode
        pil = self.load_image(*ids).convert("RGB")
        pil = TF.resize(pil, self._EVAL_RESIZE, interpolation=InterpolationMode.BILINEAR)
        return TF.center_crop(pil, [self._EVAL_CROP, self._EVAL_CROP])

    def build_roi_mask(
        self,
        *ids,
        target_h: int,
        target_w: int,
        expert:   int = 1,
        contour_dir: str | None = None,
    ) -> np.ndarray | None:
        """Rasterize the expert disc contour for (pid, eye), aligned to the eval crop.

        Returns a binary ndarray of shape (target_h, target_w), or None if no
        contour file exists for this eye / the file is malformed.

        Alignment: the contour polygon is rasterized at the original image
        resolution, then resized + center-cropped to match the same eval
        transform the image tower uses, then resized to (target_h, target_w).
        """
        from PIL import ImageDraw
        from torchvision.transforms import functional as TF
        from torchvision.transforms import InterpolationMode

        if len(ids) < 2:
            raise TypeError(
                "build_roi_mask requires (patient_id, eye) — got %r" % (ids,)
            )
        pid, eye = int(ids[0]), str(ids[1])

        repo_root = Path(__file__).resolve().parents[3]
        dir_ = Path(contour_dir or self._DEFAULT_CONTOUR_DIR)
        if not dir_.is_absolute():
            dir_ = repo_root / dir_
        contour_path = dir_ / f"RET{pid:03d}{eye}_disc_exp{expert}.txt"
        if not contour_path.exists():
            return None
        try:
            arr = np.loadtxt(str(contour_path), dtype=np.float32)
        except Exception:
            return None
        if arr.ndim == 1:
            arr = arr.reshape(-1, 2)
        if arr.shape[0] < 3:
            return None

        # Original image size for polygon canvas
        try:
            with Image.open(self.get_image_path(*ids)) as im:
                orig_w, orig_h = im.size
        except Exception:
            return None

        canvas = Image.new("L", (orig_w, orig_h), 0)
        ImageDraw.Draw(canvas).polygon([tuple(pt) for pt in arr[:, :2]], fill=1)

        # Match the eval transform exactly (resize → center-crop)
        canvas = TF.resize(canvas, self._EVAL_RESIZE, interpolation=InterpolationMode.NEAREST)
        canvas = TF.center_crop(canvas, [self._EVAL_CROP, self._EVAL_CROP])

        if (target_h, target_w) != (self._EVAL_CROP, self._EVAL_CROP):
            canvas = canvas.resize((target_w, target_h), Image.NEAREST)
        return np.array(canvas, dtype=bool)

    def orient_for_display(self, image, side: str):
        """Mirror OS to align its nasal–temporal axis with the OD convention.

        Accepts a PIL.Image or a numpy array. OD is passed through unchanged.
        """
        if side != self.SIDE_B:  # SIDE_A == OD, unchanged
            return image
        if isinstance(image, Image.Image):
            from PIL import ImageOps
            return ImageOps.mirror(image)
        if isinstance(image, np.ndarray):
            return np.ascontiguousarray(np.fliplr(image))
        raise TypeError(f"orient_for_display: unsupported type {type(image).__name__}")


# ---------------------------------------------------------------------------
# PapilaBundle — the v4 DataBundle returned by build_data
# ---------------------------------------------------------------------------

class PapilaBundle:
    """V4 DataBundle for PAPILA.

    Wraps the v3 DataBundle for backward compatibility (df, feature_dim,
    vectorize_row, get_image_path, patient_col, label_col) while adding
    the v4 DataView interface and build_shells().
    """

    def __init__(
        self,
        bundle:        DataBundle,
        image_dir:     str,
        preprocessor:  Callable | None = None,
        image_cache:   CachedImageLoader | None = None,
        vf_md_targets: dict[tuple, float] | None = None,
    ):
        self._bundle    = bundle
        self._image_dir = image_dir

        # ── ClinicalDataView (all eyes) ──────────────────────────────────────
        self.matrix = ClinicalDataView(
            df=bundle.df,
            patient_col=bundle.patient_col,
            scalar_cols=bundle.scalar_cols,
            cat_cols=bundle.cat_cols,
            scalar_stats=bundle.scalar_stats,
            cat_maps=bundle.cat_maps,
        )

        # ── ImageDataView (all eyes) ─────────────────────────────────────────
        self.image = ImageDataView(
            df=bundle.df,
            patient_col=bundle.patient_col,
            image_dir=image_dir,
            filename_template=bundle.filename_template,
            preprocessor=preprocessor,
            image_cache=image_cache,
        )

        # ── VF_MD target table (for regression / auxiliary heads) ────────────
        # Provided by build_data — captured from raw frames before the master
        # df dropped VF_MD via exclude_cols.  Keys are (pid, eyeID), values are
        # raw floats (NaN where the measurement was missing).
        self._vf_md_raw: dict[tuple, float] = dict(vf_md_targets or {})

        # ── Imputation distribution + pre-sampled imputed table ──────────────
        # Strategy: in PAPILA, the few healthy patients with measured MD are
        # likely biased toward being slightly worse than the typical healthy
        # eye (the test is usually administered when there's some suspicion).
        # We correct for this by centering the imputation distribution on the
        # TOP QUARTILE mean of the measured-healthy MD values (i.e. the
        # healthiest of the measured healthies), while using the std of the
        # full measured-healthy sample.  Missing values are then drawn from
        # N(top_q_mean, std²) once at bundle construction, with a fixed seed,
        # so the same patient always gets the same imputed value.
        self._impute_mean, self._impute_std = self._compute_impute_params()
        self._vf_md_imputed: dict[tuple, float] = self._sample_imputations(seed=42)

    # ── Entity-id metadata (for orchestrator logging) ────────────────────────

    @property
    def id_names(self) -> tuple[str, ...]:
        """Semantic names for each entity_id position, e.g. ('patient_id', 'eye').

        Orchestrators use this to decode entity_ids for logging without
        hardcoding dataset-specific field names.
        """
        return self.matrix.id_names  # both views share the same structure

    # ── Identity column registry (for orchestrator split_identity_level) ────────

    @property
    def identity_cols(self) -> list[str]:
        """Ordered list of grouping columns, one per identity level.

        identity_level=1 → identity_cols[0] → patient column (group by patient)
        identity_level=2 → identity_cols[1] → eye column (group by patient+eye)
        """
        return [self._bundle.patient_col, "eyeID"]

    # ── Sample collection (for tower early_pass) ─────────────────────────────

    def collect_samples(self, df: pd.DataFrame | None) -> list[tuple]:
        """Build (pid, eye, image_path) tuples from a split DataFrame.

        Used by tower early_pass implementations that need per-eye image paths
        (UNet inference, contour rasterisation, etc.).  Returns [] for an
        empty/None df.
        """
        if df is None or len(df) == 0:
            return []
        pc = self._bundle.patient_col
        out: list[tuple] = []
        for _, row in df.iterrows():
            pid = int(row[pc])
            eye = str(row.get("eyeID", "OD"))
            out.append((pid, eye, self.image.get_image_path(pid, eye)))
        return out

    # ── VF_MD target accessor (for regression / auxiliary heads) ─────────────

    def get_vf_md(self, pid: int, eye: str, *, impute: bool = True) -> float:
        """Return VF_MD for one eye.

        impute=True  → look up the precomputed imputed value (raw if measured,
                       distribution sample if missing).  This is what training
                       should use — deterministic, fixed-seed, same value every
                       call across the run.
        impute=False → return the raw value (NaN if unmeasured).
        """
        key = (int(pid), str(eye))
        if impute:
            return self._vf_md_imputed.get(key, self._impute_mean)
        return self._vf_md_raw.get(key, float("nan"))

    @property
    def has_vf_md(self) -> bool:
        return bool(self._vf_md_raw)

    @property
    def impute_params(self) -> dict:
        """Inspection: which mean/std were used to draw imputations."""
        return {"mean": self._impute_mean, "std": self._impute_std}

    # ── Internal helpers for imputation -------------------------------------

    def _compute_impute_params(self) -> tuple[float, float]:
        """Mean = top-quartile mean of measured healthy MDs; std = std of all.

        Falls back to (0.0, 0.0) if there are too few measured-healthy samples
        to fit a sensible distribution.
        """
        # Build (pid, eye) → diagnosis from the underlying bundle df.
        diag_lookup: dict[tuple, int] = {}
        label_col = self._bundle.label_col
        pc        = self._bundle.patient_col
        if label_col in self._bundle.df.columns:
            for _, row in self._bundle.df.iterrows():
                key = (int(row[pc]), str(row.get("eyeID", "OD")))
                diag_lookup[key] = int(row[label_col])

        measured_healthy = [
            v for key, v in self._vf_md_raw.items()
            if v == v and diag_lookup.get(key, -1) == 0  # not NaN and healthy
        ]
        if len(measured_healthy) < 4:
            return 0.0, 0.0

        arr = np.asarray(measured_healthy, dtype=np.float64)
        q3  = float(np.percentile(arr, 75))
        top = arr[arr >= q3]
        mean = float(top.mean()) if top.size else float(arr.mean())
        std  = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        return mean, std

    def _sample_imputations(self, *, seed: int) -> dict[tuple, float]:
        """One-shot pre-sample: every (pid, eye) gets a fixed value for the run."""
        rng = np.random.default_rng(seed)
        out: dict[tuple, float] = {}
        for key in sorted(self._vf_md_raw.keys()):
            v = self._vf_md_raw[key]
            if v == v:  # measured
                out[key] = float(v)
            elif self._impute_std > 0:
                out[key] = float(rng.normal(self._impute_mean, self._impute_std))
            else:
                out[key] = self._impute_mean
        return out

    # ── Backward-compat delegates ────────────────────────────────────────────

    @property
    def df(self) -> pd.DataFrame:
        return self._bundle.df

    @property
    def label_col(self) -> str:
        return self._bundle.label_col

    @property
    def patient_col(self) -> str:
        return self._bundle.patient_col

    @property
    def feature_dim(self) -> int:
        return self._bundle.feature_dim

    def vectorize_row(self, row: pd.Series) -> np.ndarray:
        return self._bundle.vectorize_row(row)

    def get_image_path(self, row: pd.Series):
        return self._bundle.get_image_path(row)

    # ── Shell building ───────────────────────────────────────────────────────

    def build_shells(
        self,
        df: pd.DataFrame,
        *,
        level: str = "eye",
        label_filter: list[int] | None = None,
        eye_filter: str | None = None,
    ) -> LoaderShell:
        """Build a LoaderShell from a split DataFrame.

        level="eye"     — one ShellEntry per eye row.
                          entity_id = (patient_id, side_key)
                          e.g. (42, "OD") or (42, "OS")

        level="patient" — one ShellEntry per patient.
                          entity_id = (patient_id,)  — 1-tuple
                          Towers that need both sides use data_view.side_map
                          to assemble them in get_sample.
                          Patients missing either eye are excluded.
        """
        pc = self.patient_col
        lc = self.label_col

        if label_filter is not None:
            df = df[df[lc].isin(label_filter)]
        if eye_filter is not None and "eyeID" in df.columns:
            df = df[df["eyeID"] == eye_filter]

        entries: list[ShellEntry] = []

        if level == "eye":
            for _, row in df.iterrows():
                pid   = int(row[pc])
                label = int(row[lc])
                side  = str(row.get("eyeID", "OD"))
                meta  = {"vf_md": self.get_vf_md(pid, side)} if self.has_vf_md else {}
                entries.append(ShellEntry(entity_id=(pid, side), label=label, meta=meta))

        elif level == "patient":
            for pid, grp in df.groupby(pc):
                if "eyeID" in grp.columns:
                    eyes = set(grp["eyeID"].unique())
                    if "OD" not in eyes or "OS" not in eyes:
                        continue
                label_mode = grp[lc].mode()
                label = int(label_mode.iloc[0]) if not label_mode.empty else int(grp[lc].iloc[0])
                meta = {}
                if self.has_vf_md:
                    # Patient-level target: mean of the two eyes' imputed MDs
                    meta["vf_md"] = 0.5 * (
                        self.get_vf_md(int(pid), "OD") + self.get_vf_md(int(pid), "OS")
                    )
                entries.append(ShellEntry(entity_id=(int(pid),), label=label, meta=meta))

        else:
            raise ValueError(f"Unknown shell level: {level!r}. Choose 'eye' or 'patient'.")

        return LoaderShell(entries=entries)


# ---------------------------------------------------------------------------
# Resolve helper — used by the orchestrator to inject DataView into towers
# ---------------------------------------------------------------------------

def resolve_data_source(bundle: PapilaBundle, path: str):
    """Resolve a dot-path data source string against a PapilaBundle.

    Examples
    --------
    "matrix"     → bundle.matrix
    "matrix.od"  → bundle.matrix.od
    "image"      → bundle.image
    "image.os"   → bundle.image.os
    """
    obj = bundle
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


# ---------------------------------------------------------------------------
# Public contract: build_data(args: dict) -> PapilaBundle
# ---------------------------------------------------------------------------

_DEFAULT_CAT_COLS = ["Gender", "Phakic/Pseudophakic"]


def build_data(args: dict) -> PapilaBundle:
    """Build and return a PapilaBundle.

    args keys
    ---------
    image_dir        (required)
    clinical_dir     (required)
    label_col        (default: "Diagnosis")
    iop_corr_method  (default: "ratio")
    iop_drop_raw     (default: False)
    exclude_cols     (default: [])
    cat_cols         (default: ["Gender", "Phakic/Pseudophakic"])
    n_splits         (default: 5)
    random_seed      (default: 42)
    in_memory_cache  (default: False) — enable shared image cache for the run
    """
    image_dir      = args["image_dir"]
    clinical_dir   = args["clinical_dir"]
    label_col      = args.get("label_col",       "Diagnosis")
    iop_method     = args.get("iop_corr_method", "ratio")
    iop_drop_raw   = bool(args.get("iop_drop_raw", False))
    exclude_cols   = list(args.get("exclude_cols", []))
    cat_cols       = list(args.get("cat_cols",    _DEFAULT_CAT_COLS))
    n_splits       = int(args.get("n_splits",     5))
    random_seed    = int(args.get("random_seed",  42))
    use_cache      = bool(args.get("in_memory_cache", False))

    # Always exclude VF_MD from feature vectorization — it's a target/label
    # column (used by regression heads), never an input.
    if "VF_MD" not in exclude_cols:
        exclude_cols = exclude_cols + ["VF_MD"]

    effective_cat = [c for c in cat_cols if c not in exclude_cols]

    bundle = DataBundle(
        image_dir=image_dir,
        clinical_dir=clinical_dir,
        label_col=label_col,
        patient_col="Patient ID",
        cat_cols=effective_cat,
        n_splits=n_splits,
        random_seed=random_seed,
        filename_template="RET{pid:03d}{eye}.jpg",
    )

    od  = pd.read_excel(f"{clinical_dir}/patient_data_od.xlsx", header=1)
    od["eyeID"] = "OD"
    os_ = pd.read_excel(f"{clinical_dir}/patient_data_os.xlsx", header=1)
    os_["eyeID"] = "OS"

    for frame in (od, os_):
        if "Patient ID" not in frame.columns and "ID" in frame.columns:
            frame.rename(columns={"ID": "Patient ID"}, inplace=True)
        frame["Patient ID"] = (
            frame["Patient ID"].astype(str).str.extract(r"(\d+)")[0].astype(int)
        )
        _canonicalize_eye_column(frame)

    bundle.add_df(od,  id_column="ID", exclude_cols=exclude_cols or None)
    bundle.add_df(os_, id_column="ID", exclude_cols=exclude_cols or None)

    converter = _fit_perkins_converter(bundle.frames, method=iop_method)
    for i in range(len(bundle.frames)):
        bundle.frames[i] = _apply_iop_and_drop_md(
            bundle.frames[i], converter=converter, drop_raw=iop_drop_raw
        )

    # Capture VF_MD per (pid, eyeID) from the raw frames BEFORE the master df
    # is refreshed (which would drop VF_MD via exclude_cols).
    vf_md_targets: dict[tuple, float] = {}
    for frame in bundle.frames:
        if "VF_MD" not in frame.columns:
            continue
        for _, row in frame.iterrows():
            pid = int(row["Patient ID"])
            eye = str(row.get("eyeID", "OD"))
            v   = row["VF_MD"]
            vf_md_targets[(pid, eye)] = (
                float(v) if not pd.isna(v) else float("nan")
            )

    bundle._refresh_master_df(exclude_cols=exclude_cols or None)
    bundle._infer_or_validate_feature_types(exclude_cols=exclude_cols or None)
    bundle._compute_numeric_stats()
    bundle._build_cat_maps()
    bundle._compute_feature_dim()

    image_cache = CachedImageLoader() if use_cache else None

    return PapilaBundle(
        bundle=bundle,
        image_dir=image_dir,
        image_cache=image_cache,
        vf_md_targets=vf_md_targets,
    )
