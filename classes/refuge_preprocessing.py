"""Utilities for preparing REFUGE (REFUGE1/REFUGE2) datasets.

Builds a unified manifest across all provided splits (REFUGE1 train/val/test
and REFUGE2 validation/test), exposing image paths, glaucoma labels, disc/cup
masks, and fovea coordinates so downstream segmentation/classification modules
can operate without additional bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


@dataclass
class RefugeSample:
    """Lightweight container describing a REFUGE sample."""

    sample_id: str
    dataset: str
    split: str
    image_path: Path
    label: Optional[int]
    device: Optional[str]
    mask_path: Optional[Path]
    fovea_coord: Optional[Tuple[float, float]]


class RefugePreprocessing:
    """Builds manifests and provides shared helpers for REFUGE workflows.

    Responsibilities:
      * scan the REFUGE directory structure and build a consistent manifest
        (train/val/test, device vendor, ground-truth labels)
      * expose convenience loaders for raw RGB frames, OD/OC masks, and
        optional fovea landmarks
      * compute geometric metadata (disc centres, diameters) so downstream
        stages can crop ROIs lazily instead of storing pre-rendered tiles
    """

    def __init__(self, root_dir: Path | str) -> None:
        self.root_dir = Path(root_dir)
        self._manifest = None  # populated by build_manifest()

    # ------------------------------------------------------------------
    # Manifest handling
    # ------------------------------------------------------------------
    def build_manifest(self, refresh: bool = False) -> Iterable[RefugeSample]:
        """Return an iterable of :class:`RefugeSample` records.

        Parameters
        ----------
        refresh:
            when True, force a rescan of the filesystem instead of reusing the
            cached manifest.

        Returns
        -------
        Iterable[RefugeSample]
            A sequence containing one entry per sample in the REFUGE datasets.

        Notes
        -----
        The actual manifest-building logic will live here: parsing the
        directory structure, reading any provided CSV/Excel metadata, and
        aligning masks/labels. For now, this method raises ``NotImplementedError``
        so callers are reminded to hook it up before use.
        """

        if self._manifest is not None and not refresh:
            return self._manifest

        manifest: List[RefugeSample] = []

        manifest.extend(self._collect_refuge1_train())
        manifest.extend(self._collect_refuge1_val())
        manifest.extend(self._collect_refuge1_test())
        manifest.extend(self._collect_refuge2_val())
        manifest.extend(self._collect_refuge2_test())

        self._manifest = manifest
        return self._manifest

    # ------------------------------------------------------------------
    # Accessors for downstream modules
    # ------------------------------------------------------------------
    def load_image(self, sample: RefugeSample):
        """Return the RGB fundus image for ``sample``.

        Implementors should handle color-space consistency (e.g., ensure RGB vs
        BGR) and any global normalisation desired across devices.
        """

        raise NotImplementedError("Image loading to be implemented")

    def load_mask(self, sample: RefugeSample):
        """Return the optic disc/cup mask for ``sample`` if available."""

        raise NotImplementedError("Mask loading to be implemented")

    def disc_geometry(self, sample: RefugeSample) -> Dict[str, float]:
        """Compute disc centre and diameter from the mask.

        The segmentation module will rely on this to crop 2.5–3× disc-diameter
        ROIs at training time.
        """

        raise NotImplementedError("Disc geometry helper to be implemented")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _collect_refuge1_train(self) -> List[RefugeSample]:
        base = self.root_dir / "Train" / "REFUGE1-train"
        if not base.exists():
            return []

        fovea_path = base / "Fovea_location.xlsx"
        fovea_map = self._read_fovea_table(fovea_path, img_col="ImgName")

        samples: List[RefugeSample] = []
        image_root = base / "Training400"
        mask_root = base / "Disc_Cup_Masks"

        for label_name, label_val in ("Glaucoma", 1), ("Non-Glaucoma", 0):
            img_dir = image_root / label_name
            mask_dir = mask_root / label_name
            if not img_dir.exists():
                continue
            for image_path in sorted(img_dir.glob("*.jpg")):
                img_name = image_path.name
                mask_path = (mask_dir / image_path.with_suffix(".bmp").name)
                fovea = fovea_map.get(img_name)
                sample_id = f"refuge1_train_{image_path.stem}"
                samples.append(
                    RefugeSample(
                        sample_id=sample_id,
                        dataset="refuge1",
                        split="train",
                        image_path=image_path,
                        label=label_val,
                        device=None,
                        mask_path=mask_path if mask_path.exists() else None,
                        fovea_coord=fovea,
                    )
                )
        return samples

    def _collect_refuge1_val(self) -> List[RefugeSample]:
        base = self.root_dir / "Train" / "REFUGE1-val"
        if not base.exists():
            return []

        fovea_path = base / "Fovea_locations.xlsx"
        df = pd.read_excel(fovea_path)
        samples: List[RefugeSample] = []
        image_root = base / "REFUGE-Validation400"
        mask_root = base / "Disc_Cup_Masks"

        for _, row in df.iterrows():
            img_name = row["ImgName"]
            image_path = image_root / img_name
            mask_path = mask_root / Path(img_name).with_suffix(".bmp").name
            fovea = self._extract_fovea(row, x_key="Fovea_X", y_key="Fovea_Y")
            label = int(row.get("Glaucoma Label", 0)) if not pd.isna(row.get("Glaucoma Label", 0)) else None
            sample_id = f"refuge1_val_{Path(img_name).stem}"
            samples.append(
                RefugeSample(
                    sample_id=sample_id,
                    dataset="refuge1",
                    split="val",
                    image_path=image_path,
                    label=label,
                    device=None,
                    mask_path=mask_path if mask_path.exists() else None,
                    fovea_coord=fovea,
                )
            )
        return samples

    def _collect_refuge1_test(self) -> List[RefugeSample]:
        base = self.root_dir / "Train" / "REFUGE1-test"
        if not base.exists():
            return []

        df = pd.read_excel(base / "Glaucoma_label_and_Fovea_location.xlsx")
        image_root = base / "Test400"
        mask_root = base / "Disc_Cup_Masks"
        samples: List[RefugeSample] = []

        for _, row in df.iterrows():
            img_name = row["ImgName"]
            image_path = image_root / img_name
            mask_path = mask_root / Path(img_name).with_suffix(".bmp").name
            fovea = self._extract_fovea(row, x_key="Fovea_X", y_key="Fovea_Y")
            label = int(row.get("Label(Glaucoma=1)", 0)) if not pd.isna(row.get("Label(Glaucoma=1)", 0)) else None
            sample_id = f"refuge1_test_{Path(img_name).stem}"
            samples.append(
                RefugeSample(
                    sample_id=sample_id,
                    dataset="refuge1",
                    split="test",
                    image_path=image_path,
                    label=label,
                    device=None,
                    mask_path=mask_path if mask_path.exists() else None,
                    fovea_coord=fovea,
                )
            )
        return samples

    def _collect_refuge2_val(self) -> List[RefugeSample]:
        base = self.root_dir / "Validation"
        if not base.exists():
            return []

        label_df = pd.read_csv(base / "glaucoma.csv")
        fovea_df = pd.read_csv(base / "fovea.csv")
        fovea_map = {
            row["ImageName"]: (float(row["Fovea_X"]), float(row["Fovea_Y"]))
            for _, row in fovea_df.iterrows()
        }
        samples: List[RefugeSample] = []
        image_root = base / "Images"
        mask_root = base / "Disc_Masks"

        for _, row in label_df.iterrows():
            img_name = row["FileName"]
            image_path = image_root / img_name
            mask_path = mask_root / Path(img_name).with_suffix(".png").name
            label = row.get("Glaucoma Risk")
            label = int(label) if label == label else None
            sample_id = f"refuge2_val_{Path(img_name).stem}"
            samples.append(
                RefugeSample(
                    sample_id=sample_id,
                    dataset="refuge2",
                    split="val",
                    image_path=image_path,
                    label=label,
                    device=None,
                    mask_path=mask_path if mask_path.exists() else None,
                    fovea_coord=fovea_map.get(img_name),
                )
            )
        return samples

    def _collect_refuge2_test(self) -> List[RefugeSample]:
        base = self.root_dir / "Test"
        if not base.exists():
            return []

        label_df = pd.read_excel(base / "task1.xls", header=None, names=["ImgName", "Glaucoma"])
        fovea_df = pd.read_excel(base / "fovea.xlsx")
        fovea_map = {
            row["ImageName"]: (float(row["Fovea_X"]), float(row["Fovea_Y"]))
            for _, row in fovea_df.iterrows()
        }
        samples: List[RefugeSample] = []
        image_root = base / "refuge2-test"
        mask_root = base / "Disc_Mask"

        for _, row in label_df.iterrows():
            img_name = row["ImgName"]
            image_path = image_root / img_name
            mask_path = mask_root / Path(img_name).with_suffix(".png").name
            label = row.get("Glaucoma")
            label = int(label) if label == label else None
            sample_id = f"refuge2_test_{Path(img_name).stem}"
            samples.append(
                RefugeSample(
                    sample_id=sample_id,
                    dataset="refuge2",
                    split="test",
                    image_path=image_path,
                    label=label,
                    device=None,
                    mask_path=mask_path if mask_path.exists() else None,
            fovea_coord=fovea_map.get(img_name),
                )
            )
        return samples

    @staticmethod
    def _read_fovea_table(path: Path, img_col: str) -> Dict[str, Tuple[float, float]]:
        if not path.exists():
            return {}
        df = pd.read_excel(path)
        mapping: Dict[str, Tuple[float, float]] = {}
        for _, row in df.iterrows():
            mapping[row[img_col]] = (
                float(row.get("Fovea_X", float("nan"))),
                float(row.get("Fovea_Y", float("nan"))),
            )
        return mapping

    @staticmethod
    def _extract_fovea(row: pd.Series, x_key: str, y_key: str) -> Optional[Tuple[float, float]]:
        x_val = row.get(x_key)
        y_val = row.get(y_key)
        if pd.isna(x_val) or pd.isna(y_val):
            return None
        return float(x_val), float(y_val)
