from __future__ import annotations

import pandas as pd

from dataclasses import dataclass

from .base import SimpleDatasetProfile, SlotDescriptor


@dataclass(frozen=True)
class PapilaProfile(SimpleDatasetProfile):
    sample_mode: str = "patient"  # "patient" | "eye"

    def build_samples(self, *, df: pd.DataFrame, clinical) -> list[dict[str, object]]:
        samples: list[dict[str, object]] = []
        patient_col = self.patient_col
        label_col = self.label_col

        mode = (self.sample_mode or "patient").lower()
        if mode not in {"patient", "eye"}:
            raise ValueError(f"Unsupported sample_mode '{self.sample_mode}'. Expected 'patient' or 'eye'.")

        if mode == "eye":
            for _, row in df.iterrows():
                pid = row[patient_col]
                label = row[label_col]
                image_1 = clinical.get_image_path(row) if hasattr(clinical, "get_image_path") else None
                matrix_1 = clinical.vectorize_row(row) if hasattr(clinical, "vectorize_row") else None
                samples.append(
                    {
                        "id_1": pid,
                        "label_1": label,
                        "image_1": image_1,
                        "matrix_1": matrix_1,
                    }
                )
            return samples

        for pid, grp in df.groupby(patient_col):
            label_series = grp[label_col]
            if label_series.empty:
                continue
            mode_vals = label_series.mode()
            label = mode_vals.iloc[0] if not mode_vals.empty else label_series.iloc[0]

            def _row_for_eye(eye: str):
                if "eyeID" not in grp.columns:
                    return None
                match = grp[grp["eyeID"].astype(str).str.upper() == eye]
                if match.empty:
                    return None
                return match.iloc[0]

            row_od = _row_for_eye("OD")
            row_os = _row_for_eye("OS")
            row_any = grp.iloc[0]

            image_1 = clinical.get_image_path(row_od) if row_od is not None else None
            image_2 = clinical.get_image_path(row_os) if row_os is not None else None
            matrix_1 = clinical.vectorize_row(row_od) if row_od is not None else None
            matrix_2 = clinical.vectorize_row(row_os) if row_os is not None else None

            if image_1 is None and hasattr(clinical, "get_image_path"):
                image_1 = clinical.get_image_path(row_any)
            if matrix_1 is None and hasattr(clinical, "vectorize_row"):
                matrix_1 = clinical.vectorize_row(row_any)

            samples.append(
                {
                    "id_1": pid,
                    "label_1": label,
                    "image_1": image_1,
                    "image_2": image_2,
                    "matrix_1": matrix_1,
                    "matrix_2": matrix_2,
                }
            )
        return samples


def build_papila_profile(
    *,
    patient_col: str = "Patient ID",
    label_col: str = "Diagnosis",
    sample_mode: str = "patient",
) -> PapilaProfile:
    """
    PAPILA-specific semantic map for generic V2 slot keys.

    The engine remains slot-based (image_1/image_2/matrix_1/...).
    PAPILA meaning is captured here so run config stays dataset-local.
    """

    slots = {
        "id_1": SlotDescriptor(
            key="id_1",
            kind="id",
            description=f"Patient identifier column ({patient_col})",
            required=True,
            shape_hint="scalar",
        ),
        "label_1": SlotDescriptor(
            key="label_1",
            kind="label",
            description=f"Diagnosis label column ({label_col})",
            required=True,
            shape_hint="scalar",
        ),
        "image_1": SlotDescriptor(
            key="image_1",
            kind="image",
            description="Fundus image slot 1 (PAPILA: OD / right eye)",
            required=False,
            shape_hint="HWC or CHW",
        ),
        "image_2": SlotDescriptor(
            key="image_2",
            kind="image",
            description="Fundus image slot 2 (PAPILA: OS / left eye)",
            required=False,
            shape_hint="HWC or CHW",
        ),
        "matrix_1": SlotDescriptor(
            key="matrix_1",
            kind="matrix",
            description="Clinical metadata feature vector",
            required=False,
            shape_hint="[feature_dim]",
        ),
        "matrix_2": SlotDescriptor(
            key="matrix_2",
            kind="matrix",
            description="Optional auxiliary tabular vector (reserved for experiments)",
            required=False,
            shape_hint="[feature_dim_2]",
        ),
    }

    aliases = {
        "id_1": "patient_id",
        "label_1": "diagnosis",
        "image_1": "od_fundus",
        "image_2": "os_fundus",
        "matrix_1": "clinical_metadata",
        "matrix_2": "aux_metadata",
    }

    return PapilaProfile(
        name="papila",
        patient_col=patient_col,
        label_col=label_col,
        slots=slots,
        aliases=aliases,
        sample_mode=sample_mode,
    )
