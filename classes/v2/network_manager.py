from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

import pandas as pd


@dataclass
class PatientSplit:
    """Patient-disjoint split definition for a fold."""

    train: pd.DataFrame
    val: pd.DataFrame
    holdout: Optional[pd.DataFrame] = None


@dataclass
class LoaderBundle:
    """All loaders needed by a training run."""

    train: Any
    val: Any
    holdout: Optional[Any] = None


@dataclass
class FoldResult:
    """Normalized fold output from trainer implementations."""

    fold: int
    metrics: dict[str, Any]
    artifacts: dict[str, Any]


class SplitManager(Protocol):
    def build_plans(
        self,
        *,
        clinical: Any,
        args: Any,
        profile: Optional[Any] = None,
    ) -> list[PatientSplit]:
        ...


class GraphFactory(Protocol):
    def build(
        self,
        *,
        clinical: Any,
        args: Any,
        fold: int,
        profile: Optional[Any] = None,
    ) -> Any:
        ...


class LoaderFactory(Protocol):
    def build(
        self,
        *,
        clinical: Any,
        split: PatientSplit,
        args: Any,
        fold: int,
        profile: Optional[Any] = None,
    ) -> LoaderBundle:
        ...


class Trainer(Protocol):
    def fit(
        self,
        *,
        graph: Any,
        loaders: LoaderBundle,
        args: Any,
        fold: int,
        profile: Optional[Any] = None,
    ) -> FoldResult:
        ...


class NetworkManager:
    """
    V2 orchestration entrypoint.

    This class is intentionally small and modular:
    - split policy is delegated to a SplitManager
    - graph assembly is delegated to a GraphFactory
    - dataloaders are delegated to a LoaderFactory
    - train/eval/checkpoint lifecycle is delegated to a Trainer
    """

    def __init__(
        self,
        *,
        clinical: Any,
        args: Any,
        split_manager: SplitManager,
        graph_factory: GraphFactory,
        loader_factory: LoaderFactory,
        trainer: Trainer,
        profile: Optional[Any] = None,
    ) -> None:
        self.clinical = clinical
        self.args = args
        self.split_manager = split_manager
        self.graph_factory = graph_factory
        self.loader_factory = loader_factory
        self.trainer = trainer
        self.profile = profile
        self._split_plans: Optional[list[PatientSplit]] = None

    def run_fold(self, fold: int) -> FoldResult:
        plans = self._get_split_plans()
        if fold < 0 or fold >= len(plans):
            raise IndexError(f"Requested fold {fold} but only {len(plans)} fold plans are available")
        split = plans[fold]
        self._validate_patient_disjointness(split)
        self._validate_labels(split)

        graph = self.graph_factory.build(
            clinical=self.clinical,
            args=self.args,
            fold=fold,
            profile=self.profile,
        )
        loaders = self.loader_factory.build(
            clinical=self.clinical,
            split=split,
            args=self.args,
            fold=fold,
            profile=self.profile,
        )
        return self.trainer.fit(
            graph=graph,
            loaders=loaders,
            args=self.args,
            fold=fold,
            profile=self.profile,
        )

    def run_all_folds(self, n_splits: Optional[int] = None) -> list[FoldResult]:
        plans = self._get_split_plans()
        max_folds = len(plans)
        if n_splits is None:
            n = max_folds
        else:
            n = int(n_splits)
            if n < 1:
                raise ValueError("n_splits must be >= 1")
            if n > max_folds:
                raise ValueError(f"Requested {n} folds but only {max_folds} fold plans are available")
        return [self.run_fold(fold) for fold in range(n)]

    def _get_split_plans(self) -> list[PatientSplit]:
        if self._split_plans is None:
            self._split_plans = self.split_manager.build_plans(
                clinical=self.clinical,
                args=self.args,
                profile=self.profile,
            )
            if not self._split_plans:
                raise ValueError("SplitManager returned no fold plans")
        return self._split_plans

    def _validate_patient_disjointness(self, split: PatientSplit) -> None:
        train_ids = self._patient_ids(split.train)
        val_ids = self._patient_ids(split.val)
        holdout_ids = self._patient_ids(split.holdout) if split.holdout is not None else set()

        if train_ids & val_ids:
            overlap = sorted(train_ids & val_ids)[:10]
            raise ValueError(f"Patient leakage between train/val: {overlap}")
        if train_ids & holdout_ids:
            overlap = sorted(train_ids & holdout_ids)[:10]
            raise ValueError(f"Patient leakage between train/holdout: {overlap}")
        if val_ids & holdout_ids:
            overlap = sorted(val_ids & holdout_ids)[:10]
            raise ValueError(f"Patient leakage between val/holdout: {overlap}")

    def _validate_labels(self, split: PatientSplit) -> None:
        label_col = getattr(self.clinical, "label_col", None)
        if not label_col:
            return
        for name, df in (("train", split.train), ("val", split.val), ("holdout", split.holdout)):
            if df is None:
                continue
            if label_col not in df.columns:
                raise ValueError(f"{name} split is missing label column {label_col!r}")

    @staticmethod
    def _patient_ids(df: Optional[pd.DataFrame]) -> set[Any]:
        if df is None or df.empty:
            return set()
        if "Patient ID" not in df.columns:
            raise ValueError("Split dataframes must include 'Patient ID'")
        return set(df["Patient ID"].tolist())
