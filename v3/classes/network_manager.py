from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

import pandas as pd


@dataclass
class PatientSplit:
    """Patient-disjoint split definition for a fold (V3: train/val/test)."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: Optional[pd.DataFrame] = None


@dataclass
class LoaderBundle:
    """All loaders needed by a training run."""

    train: Any
    val: Any
    test: Optional[Any] = None


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
    """V3 orchestration entrypoint. No holdout — test = current fold."""

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
            clinical=self.clinical, args=self.args, fold=fold, profile=self.profile,
        )
        loaders = self.loader_factory.build(
            clinical=self.clinical, split=split, args=self.args, fold=fold, profile=self.profile,
        )
        return self.trainer.fit(
            graph=graph, loaders=loaders, args=self.args, fold=fold, profile=self.profile,
        )

    def run_all_folds(self, n_splits: Optional[int] = None) -> list[FoldResult]:
        plans = self._get_split_plans()
        max_folds = len(plans)
        n = max_folds if n_splits is None else int(n_splits)
        if n < 1:
            raise ValueError("n_splits must be >= 1")
        if n > max_folds:
            raise ValueError(f"Requested {n} folds but only {max_folds} available")
        return [self.run_fold(fold) for fold in range(n)]

    def _get_split_plans(self) -> list[PatientSplit]:
        if self._split_plans is None:
            self._split_plans = self.split_manager.build_plans(
                clinical=self.clinical, args=self.args, profile=self.profile,
            )
            if not self._split_plans:
                raise ValueError("SplitManager returned no fold plans")
        return self._split_plans

    def _validate_patient_disjointness(self, split: PatientSplit) -> None:
        train_ids = self._patient_ids(split.train)
        val_ids   = self._patient_ids(split.val)
        test_ids  = self._patient_ids(split.test) if split.test is not None else set()

        if train_ids & val_ids:
            raise ValueError(f"Patient leakage train/val: {sorted(train_ids & val_ids)[:10]}")
        if train_ids & test_ids:
            raise ValueError(f"Patient leakage train/test: {sorted(train_ids & test_ids)[:10]}")
        if val_ids & test_ids:
            raise ValueError(f"Patient leakage val/test: {sorted(val_ids & test_ids)[:10]}")

    def _validate_labels(self, split: PatientSplit) -> None:
        label_col = getattr(self.clinical, "label_col", None)
        if not label_col:
            return
        for name, df in (("train", split.train), ("val", split.val), ("test", split.test)):
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
