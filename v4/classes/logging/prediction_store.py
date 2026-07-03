"""prediction_store — per-epoch logit and embedding recording across folds and phases.

PredictionStore  — records logits for every head, fold, phase, and epoch.
FeatureStore     — records embeddings (opt-in); same structure but per-head
                   tensors since embedding dims vary across heads.

HDF5 layout — PredictionStore
------------------------------
/{phase}/logits          float32          (n_folds, n_epochs, n_samples, n_heads, n_classes)
/{phase}/head_names      str              (n_heads,)
/{phase}/y_true          int64 | float64  (n_samples,)  — float64 for regression targets, int64 otherwise
/{phase}/entity_id_{k}  int64|str         (n_samples,)  — one dataset per id component
/{phase}/split           str              (n_folds, n_samples)
/{phase}/loss            float32          (n_folds, n_epochs)

HDF5 layout — FeatureStore
---------------------------
/{phase}/{head_name}     float32          (n_folds, n_epochs, n_samples, embedding_dim)
/{phase}/y_true          int64 | float64  (n_samples,)  — float64 for regression targets, int64 otherwise
/{phase}/entity_id_{k}  int64|str         (n_samples,)
/{phase}/split           str              (n_folds, n_samples)
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import h5py
except ImportError as e:
    raise ImportError("PredictionStore requires h5py: pip install h5py") from e

_STR_DT = h5py.string_dtype()


def _coerce_y_true(y_true) -> np.ndarray:
    """Coerce y_true to int64 for integer-typed input, float64 otherwise.

    Forcing int64 unconditionally would silently round regression targets
    (e.g. VF_MD), so we honour float input by storing as float64.
    """
    arr = np.asarray(y_true)
    if np.issubdtype(arr.dtype, np.floating):
        return arr.astype(np.float64)
    return arr.astype(np.int64)


# ---------------------------------------------------------------------------
# Internal phase buffer
# ---------------------------------------------------------------------------

class _PhaseBuffer:
    def __init__(
        self,
        entity_ids: list[tuple],
        y_true:     np.ndarray,
        head_names: list[str],
        n_epochs:   int,
        n_folds:    int,
        n_classes:  int,
    ):
        n_s = len(entity_ids)
        n_h = len(head_names)
        self.entity_ids = list(entity_ids)
        self.y_true     = _coerce_y_true(y_true)
        self.head_names = list(head_names)
        self.n_epochs   = n_epochs
        self.logits     = np.full((n_folds, n_epochs, n_s, n_h, n_classes), np.nan, dtype=np.float32)
        self.split      = np.full((n_folds, n_s), "", dtype=object)
        self.loss       = np.full((n_folds, n_epochs), np.nan, dtype=np.float32)
        self._sid       = {str(eid): i for i, eid in enumerate(entity_ids)}
        self._hid       = {h: i for i, h in enumerate(head_names)}


# ---------------------------------------------------------------------------
# Internal feature buffer (per-head, variable embedding_dim)
# ---------------------------------------------------------------------------

class _FeaturePhaseBuffer:
    def __init__(
        self,
        entity_ids: list[tuple],
        y_true:     np.ndarray,
        n_folds:    int,
    ):
        self.entity_ids = list(entity_ids)
        self.y_true     = _coerce_y_true(y_true)
        self.split      = np.full((n_folds, len(entity_ids)), "", dtype=object)
        self._sid       = {str(eid): i for i, eid in enumerate(entity_ids)}
        # head_name → (buffer array, n_epochs)
        self._heads: dict[str, tuple[np.ndarray, int]] = {}

    def register_head(self, head: str, n_epochs: int, embedding_dim: int, n_folds: int) -> None:
        n_s = len(self.entity_ids)
        self._heads[head] = (
            np.full((n_folds, n_epochs, n_s, embedding_dim), np.nan, dtype=np.float32),
            n_epochs,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_entity_ids(grp: h5py.Group, entity_ids: list[tuple]) -> None:
    if not entity_ids:
        return
    n_components = max(len(eid) for eid in entity_ids)
    for k in range(n_components):
        vals = [eid[k] if k < len(eid) else "" for eid in entity_ids]
        if all(isinstance(v, (int, np.integer)) for v in vals):
            grp.create_dataset(f"entity_id_{k}", data=np.array(vals, dtype=np.int64))
        else:
            grp.create_dataset(f"entity_id_{k}", data=np.array(vals, dtype=object), dtype=_STR_DT)


def _read_entity_ids(grp: h5py.Group, n_samples: int) -> list[tuple]:
    k, components = 0, []
    while f"entity_id_{k}" in grp:
        arr = grp[f"entity_id_{k}"][:]
        if arr.dtype.kind in ("S", "O", "U"):
            arr = np.array([v.decode() if isinstance(v, bytes) else str(v) for v in arr])
        components.append(arr)
        k += 1
    if not components:
        return [() for _ in range(n_samples)]
    return [tuple(c[i] for c in components) for i in range(n_samples)]


def _decode_str_array(arr: np.ndarray) -> list[str]:
    return [v.decode() if isinstance(v, bytes) else str(v) for v in arr.flat]


# ---------------------------------------------------------------------------
# PredictionStore
# ---------------------------------------------------------------------------

class PredictionStore:
    """Records per-epoch logits across all folds and phases, saves to HDF5.

    The store is generic — it knows nothing about what heads or phases exist.
    The orchestrator registers phases and records whatever heads it builds.
    """

    def __init__(self, n_folds: int, n_classes: int) -> None:
        self.n_folds   = n_folds
        self.n_classes = n_classes
        self._phases: dict[str, _PhaseBuffer] = {}

    def register_phase(
        self,
        phase:      str,
        entity_ids: list[tuple],
        y_true:     Sequence[int],
        head_names: list[str],
        n_epochs:   int,
    ) -> None:
        """Register a training phase before recording begins."""
        self._phases[phase] = _PhaseBuffer(
            entity_ids=list(entity_ids),
            y_true=_coerce_y_true(y_true),
            head_names=list(head_names),
            n_epochs=n_epochs,
            n_folds=self.n_folds,
            n_classes=self.n_classes,
        )

    def record(
        self,
        phase:      str,
        fold:       int,
        epoch:      int,
        entity_ids: Sequence[tuple],
        head:       str,
        logits:     np.ndarray,
    ) -> None:
        """Record a batch of logits for one head at one epoch."""
        buf  = self._phases[phase]
        hidx = buf._hid.get(head)
        if hidx is None:
            return
        for i, eid in enumerate(entity_ids):
            sidx = buf._sid.get(str(eid))
            if sidx is not None:
                buf.logits[fold, epoch, sidx, hidx, :] = logits[i]

    def record_loss(self, phase: str, fold: int, epoch: int, loss: float) -> None:
        self._phases[phase].loss[fold, epoch] = float(loss)

    def set_split(
        self,
        phase:      str,
        fold:       int,
        entity_ids: Sequence[tuple],
        label:      str,
    ) -> None:
        """Mark samples as 'train', 'val', or 'test' for a fold."""
        buf = self._phases[phase]
        for eid in entity_ids:
            sidx = buf._sid.get(str(eid))
            if sidx is not None:
                buf.split[fold, sidx] = label

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            for phase, buf in self._phases.items():
                grp = f.create_group(phase)
                grp.create_dataset("logits",     data=buf.logits, compression="gzip", compression_opts=4)
                grp.create_dataset("y_true",     data=buf.y_true)
                grp.create_dataset("loss",       data=buf.loss)
                grp.create_dataset("head_names", data=np.array(buf.head_names, dtype=object), dtype=_STR_DT)
                grp.create_dataset("split",      data=buf.split,                              dtype=_STR_DT)
                _write_entity_ids(grp, buf.entity_ids)

    @classmethod
    def load(cls, path: str | Path) -> "PredictionStore":
        """Load all phases into memory."""
        with h5py.File(path, "r") as f:
            first  = next(iter(f.values()))
            n_folds, _, _, _, n_classes = first["logits"].shape
            store  = cls(n_folds=n_folds, n_classes=n_classes)
            for phase in f:
                grp        = f[phase]
                logits     = grp["logits"][:]
                n_folds_, n_epochs, n_samples, n_heads, _ = logits.shape
                head_names = _decode_str_array(grp["head_names"][:])
                entity_ids = _read_entity_ids(grp, n_samples)
                buf            = _PhaseBuffer(
                    entity_ids=entity_ids,
                    y_true=grp["y_true"][:],
                    head_names=head_names,
                    n_epochs=n_epochs,
                    n_folds=n_folds_,
                    n_classes=n_classes,
                )
                buf.logits     = logits
                buf.loss       = grp["loss"][:]
                split_raw      = grp["split"][:]
                buf.split      = np.array(
                    [[v.decode() if isinstance(v, bytes) else str(v) for v in row]
                     for row in split_raw],
                    dtype=object,
                )
                store._phases[phase] = buf
        return store

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    @property
    def phases(self) -> list[str]:
        return list(self._phases.keys())

    def head_names(self, phase: str) -> list[str]:
        return self._phases[phase].head_names

    def entity_ids(self, phase: str) -> list[tuple]:
        return self._phases[phase].entity_ids

    def get_logits(
        self,
        phase: str,
        head:  str,
        fold:  int | None = None,
        epoch: int | None = None,
    ) -> np.ndarray:
        """Slice logits for one head. Unspecified dims return the full axis.

        Returns shape (folds, epochs, samples, classes) by default,
        with leading dims dropped for each specified index.
        """
        buf  = self._phases[phase]
        hidx = buf._hid[head]
        data = buf.logits[:, :, :, hidx, :]       # (folds, epochs, samples, classes)
        if fold  is not None: data = data[fold]   # (epochs, samples, classes)
        if epoch is not None: data = data[..., epoch, :, :]  if fold is None else data[epoch]
        return data

    def get_split(self, phase: str, fold: int) -> dict[str, list[tuple]]:
        """Return {'train': [...], 'val': [...], 'test': [...]} entity_id lists."""
        buf    = self._phases[phase]
        labels = buf.split[fold]
        out: dict[str, list[tuple]] = {}
        for eid, lbl in zip(buf.entity_ids, labels):
            out.setdefault(lbl, []).append(eid)
        return out


# ---------------------------------------------------------------------------
# FeatureStore
# ---------------------------------------------------------------------------

class FeatureStore:
    """Records per-epoch embeddings (variable dim per head), saves to HDF5.

    Opt-in companion to PredictionStore. Typically written only on checkpoint
    runs where you want to do dimensionality reduction or cluster analysis.
    """

    def __init__(self, n_folds: int) -> None:
        self.n_folds = n_folds
        self._phases: dict[str, _FeaturePhaseBuffer] = {}

    def register_phase(
        self,
        phase:      str,
        entity_ids: list[tuple],
        y_true:     Sequence[int],
    ) -> None:
        self._phases[phase] = _FeaturePhaseBuffer(
            entity_ids=list(entity_ids),
            y_true=_coerce_y_true(y_true),
            n_folds=self.n_folds,
        )

    def register_head(
        self,
        phase:         str,
        head:          str,
        n_epochs:      int,
        embedding_dim: int,
    ) -> None:
        self._phases[phase].register_head(head, n_epochs, embedding_dim, self.n_folds)

    def record(
        self,
        phase:      str,
        fold:       int,
        epoch:      int,
        entity_ids: Sequence[tuple],
        head:       str,
        embeddings: np.ndarray,
    ) -> None:
        buf   = self._phases[phase]
        arr, _= buf._heads[head]
        for i, eid in enumerate(entity_ids):
            sidx = buf._sid.get(str(eid))
            if sidx is not None:
                arr[fold, epoch, sidx, :] = embeddings[i]

    def set_split(
        self,
        phase:      str,
        fold:       int,
        entity_ids: Sequence[tuple],
        label:      str,
    ) -> None:
        buf = self._phases[phase]
        for eid in entity_ids:
            sidx = buf._sid.get(str(eid))
            if sidx is not None:
                buf.split[fold, sidx] = label

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            for phase, buf in self._phases.items():
                grp = f.create_group(phase)
                grp.create_dataset("y_true", data=buf.y_true)
                grp.create_dataset("split",  data=buf.split,             dtype=_STR_DT)
                _write_entity_ids(grp, buf.entity_ids)
                for head, (arr, _) in buf._heads.items():
                    grp.create_dataset(head, data=arr, compression="gzip", compression_opts=4)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureStore":
        with h5py.File(path, "r") as f:
            n_folds = next(
                arr.shape[0]
                for grp in f.values()
                for k, arr in grp.items()
                if k not in ("y_true", "split") and not k.startswith("entity_id_")
            )
            store = cls(n_folds=n_folds)
            _meta = {"y_true", "split"}
            for phase in f:
                grp        = f[phase]
                n_samples  = grp["y_true"].shape[0]
                entity_ids = _read_entity_ids(grp, n_samples)
                buf        = _FeaturePhaseBuffer(
                    entity_ids=entity_ids,
                    y_true=grp["y_true"][:],
                    n_folds=n_folds,
                )
                buf.split = np.array(
                    [[v.decode() if isinstance(v, bytes) else str(v) for v in row]
                     for row in grp["split"][:]],
                    dtype=object,
                )
                for key in grp:
                    if key in _meta or key.startswith("entity_id_"):
                        continue
                    arr = grp[key][:]
                    buf._heads[key] = (arr, arr.shape[1])
                store._phases[phase] = buf
        return store
