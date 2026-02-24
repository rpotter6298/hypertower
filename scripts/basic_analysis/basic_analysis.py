#!/usr/bin/env python3
"""Basic analytics helpers for PAPILA clinical data."""
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from classes import build_papila_clinical


class basic_analytics:
    def __init__(
        self,
        image_dir: str = "Papila/FundusImages",
        clinical_dir: str = "Papila/ClinicalData",
        label_col: str = "Diagnosis",
        cat_cols: Iterable[str] | None = None,
        exclude_cols: Iterable[str] | None = None,
        positive_label: int = 1,
        negative_label: int = 0,
        drop_labels: Iterable[int] = (2,),
        output_dir: Path | str = Path("analysis_data/basic_analysis"),
        debug: bool = False,
    ) -> None:
        self.image_dir = image_dir
        self.clinical_dir = clinical_dir
        self.label_col = label_col
        self.cat_cols = (
            list(cat_cols)
            if cat_cols is not None
            else ["Gender", "Phakic/Pseudophakic"]
        )
        base_excludes = {"Pneumatic", "Perkins"}
        self.exclude_cols = base_excludes | set(exclude_cols or [])
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.drop_labels = list(drop_labels or [])
        self.output_dir = Path(output_dir)
        self.debug = debug

    @staticmethod
    def _sanitize(name: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")
        return safe or "var"

    def _build_clinical(self):
        return build_papila_clinical(
            image_dir=self.image_dir,
            clinical_dir=self.clinical_dir,
            label_col=self.label_col,
            cat_cols=self.cat_cols,
        )

    def _select_binary_labels(
        self,
        labels: pd.Series,
    ) -> Tuple[np.ndarray, np.ndarray]:
        labels_num = pd.to_numeric(labels, errors="coerce")
        use_num = labels_num.notna().any()
        lab = labels_num if use_num else labels.astype(str)

        drop_set = set(self.drop_labels or [])
        keep = lab.isin([self.positive_label, self.negative_label])
        if drop_set:
            keep &= ~lab.isin(drop_set)

        y = (lab == self.positive_label).astype(int)
        return y.values, keep.values

    def _base_exclude(self) -> set:
        base_exclude = (
            {self.label_col, "Patient ID"} | self.exclude_cols | set(self.cat_cols)
        )
        if "eyeID" not in self.cat_cols:
            base_exclude.add("eyeID")
        return base_exclude

    def _numeric_columns(self, df: pd.DataFrame) -> List[str]:
        base_exclude = self._base_exclude()
        candidate_cols = [c for c in df.columns if c not in base_exclude]
        numeric_cols: List[str] = []
        for col in candidate_cols:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().any():
                numeric_cols.append(col)
        return numeric_cols

    @staticmethod
    def _compute_roc(
        y: np.ndarray, scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        fpr, tpr, thresholds = roc_curve(y, scores, pos_label=1)
        auc_val = float(auc(fpr, tpr))
        return fpr, tpr, thresholds, auc_val

    @staticmethod
    def _best_threshold(
        fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray
    ) -> Tuple[float, float, float]:
        youden = tpr - fpr
        idx = int(np.nanargmax(youden))
        return float(thresholds[idx]), float(tpr[idx]), float(fpr[idx])

    @staticmethod
    def _plot_overlay(curves, title: str, out_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        cmap = plt.get_cmap("tab20")
        for i, (name, fpr, tpr, auc_val) in enumerate(curves):
            color = cmap(i % cmap.N)
            ax.plot(fpr, tpr, lw=1.6, color=color, label=f"{name} (AUC={auc_val:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize="small")
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.tight_layout()
        fig.savefig(out_path, dpi=170)
        plt.close(fig)

    @staticmethod
    def _plot_per_feature(
        fpr: np.ndarray, tpr: np.ndarray, auc_val: float, title: str, out_path: Path
    ) -> None:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.plot(fpr, tpr, lw=1.8, label=f"AUC={auc_val:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.tight_layout()
        fig.savefig(out_path, dpi=170)
        plt.close(fig)

    @staticmethod
    def _plot_roc_line(
        fpr: np.ndarray, tpr: np.ndarray, auc_val: float, title: str, out_path: Path
    ) -> None:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.plot(fpr, tpr, lw=1.8, label=f"AUC={auc_val:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.tight_layout()
        fig.savefig(out_path, dpi=170)
        plt.close(fig)

    def _oof_scores(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int,
        random_state: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        scores = np.zeros(len(y), dtype=float)
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]
            if np.unique(y_train).size < 2:
                continue
            fitted = clone(model)
            fitted.fit(X_train, y_train)
            if hasattr(fitted, "predict_proba"):
                fold_scores = fitted.predict_proba(X_test)[:, 1]
            elif hasattr(fitted, "decision_function"):
                fold_scores = fitted.decision_function(X_test)
            else:
                fold_scores = fitted.predict(X_test)
            scores[test_idx] = fold_scores
        return y.astype(int), scores

    def _cv_roc_curves(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int,
        random_state: int,
    ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        curves = []
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
                continue
            fitted = clone(model)
            fitted.fit(X_train, y_train)
            if hasattr(fitted, "predict_proba"):
                scores = fitted.predict_proba(X_test)[:, 1]
            elif hasattr(fitted, "decision_function"):
                scores = fitted.decision_function(X_test)
            else:
                scores = fitted.predict(X_test)
            fpr, tpr, _ = roc_curve(y_test, scores, pos_label=1)
            auc_val = float(auc(fpr, tpr))
            curves.append((fpr, tpr, auc_val))
        return curves

    @staticmethod
    def _plot_mean_roc(
        curves: List[Tuple[np.ndarray, np.ndarray, float]],
        title: str,
        out_path: Path,
    ) -> None:
        if not curves:
            return
        mean_fpr = np.linspace(0.0, 1.0, 200)
        tprs = []
        aucs = []
        for fpr, tpr, auc_val in curves:
            tpr_interp = np.interp(mean_fpr, fpr, tpr)
            tpr_interp[0] = 0.0
            tprs.append(tpr_interp)
            aucs.append(auc_val)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(tprs, axis=0)
        mean_auc = float(np.mean(aucs))
        std_auc = float(np.std(aucs, ddof=0))

        fig, ax = plt.subplots(figsize=(5.8, 4.6))
        ax.plot(mean_fpr, mean_tpr, lw=2, label=f"AUC={mean_auc:.3f}±{std_auc:.3f}")
        ax.fill_between(
            mean_fpr,
            np.maximum(mean_tpr - std_tpr, 0),
            np.minimum(mean_tpr + std_tpr, 1),
            color="grey",
            alpha=0.2,
        )
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.tight_layout()
        fig.savefig(out_path, dpi=170)
        plt.close(fig)

    def _iter_categorical(self, df: pd.DataFrame, cols: List[str]):
        for col in cols:
            if col not in df.columns:
                continue
            s = df[col]
            vals = s.dropna().unique().tolist()
            try:
                vals = sorted(vals)
            except Exception:
                pass
            for v in vals:
                name = f"{col}=={v}"
                ind = (s == v).astype(int)
                yield name, ind

    def _feature_matrix(
        self, df: pd.DataFrame, include_categorical: bool
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        numeric_cols = self._numeric_columns(df)
        X_num = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        for col in numeric_cols:
            med = pd.to_numeric(X_num[col], errors="coerce").median()
            X_num[col] = pd.to_numeric(X_num[col], errors="coerce").fillna(med)

        parts = [X_num]
        feat_names = list(X_num.columns)

        if include_categorical and self.cat_cols:
            cat_cols = [c for c in self.cat_cols if c in df.columns]
            if cat_cols:
                df_cats = pd.get_dummies(
                    df[cat_cols].astype("category"), drop_first=False, prefix=cat_cols
                )
                parts.append(df_cats)
                feat_names.extend(list(df_cats.columns))

        X = pd.concat(parts, axis=1).values.astype(np.float32)
        labels = df[self.label_col]
        y_all, keep_mask = self._select_binary_labels(labels)
        y = y_all[keep_mask]
        X = X[keep_mask]
        return X, y.astype(int), feat_names

    def univariate_roc(
        self, merge: bool = False, include_categorical: bool = False
    ) -> pd.DataFrame:
        clinical = self._build_clinical()
        df = clinical.df.copy()
        labels = df[self.label_col]
        y_all, keep_mask = self._select_binary_labels(labels)

        if self.debug:
            for col in ("IOP_raw", "IOP_corr"):
                if col not in df.columns:
                    print(f"[debug] {col} missing from df")
                    continue
                s = pd.to_numeric(df[col], errors="coerce")
                print(
                    f"[debug] {col}: non-null={int(s.notna().sum())}, unique={int(s.nunique(dropna=True))}"
                )

        plot_dir = self.output_dir / "papila_univariate_roc" / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        curves = []

        numeric_cols = self._numeric_columns(df)
        for col in numeric_cols:
            series = pd.to_numeric(df[col], errors="coerce")
            mask = keep_mask & series.notna().values
            y = y_all[mask]
            scores = series.values[mask].astype(float)
            if y.size < 2 or np.unique(y).size < 2:
                continue
            if np.nanmin(scores) == np.nanmax(scores):
                continue
            fpr, tpr, thresholds, auc_val = self._compute_roc(y, scores)
            thr, best_tpr, best_fpr = self._best_threshold(fpr, tpr, thresholds)
            direction = "high" if auc_val >= 0.5 else "low"
            title = f"{col} (n={y.size}, direction={direction})"
            if merge:
                out_path = plot_dir / f"roc_{self._sanitize(col)}.png"
                self._plot_per_feature(fpr, tpr, auc_val, title, out_path)
            curves.append((col, fpr, tpr, auc_val))
            rows.append(
                {
                    "feature": col,
                    "kind": "numeric",
                    "n": int(y.size),
                    "auc": auc_val,
                    "direction": direction,
                    "best_threshold": thr,
                    "best_tpr": best_tpr,
                    "best_fpr": best_fpr,
                    "best_specificity": 1.0 - best_fpr,
                }
            )

        if include_categorical:
            cat_cols_use = [c for c in self.cat_cols if c not in self.exclude_cols]
            for name, ind in self._iter_categorical(df, cat_cols_use):
                mask = keep_mask & ind.notna().values
                y = y_all[mask]
                scores = ind.values[mask].astype(float)
                if y.size < 2 or np.unique(y).size < 2:
                    continue
                if np.nanmin(scores) == np.nanmax(scores):
                    continue
                fpr, tpr, thresholds, auc_val = self._compute_roc(y, scores)
                thr, best_tpr, best_fpr = self._best_threshold(fpr, tpr, thresholds)
                direction = "high" if auc_val >= 0.5 else "low"
                title = f"{name} (n={y.size}, direction={direction})"
                if merge:
                    out_path = plot_dir / f"roc_{self._sanitize(name)}.png"
                    self._plot_per_feature(fpr, tpr, auc_val, title, out_path)
                curves.append((name, fpr, tpr, auc_val))
                rows.append(
                    {
                        "feature": name,
                        "kind": "categorical",
                        "n": int(y.size),
                        "auc": auc_val,
                        "direction": direction,
                        "best_threshold": thr,
                        "best_tpr": best_tpr,
                        "best_fpr": best_fpr,
                        "best_specificity": 1.0 - best_fpr,
                    }
                )

        if not rows:
            raise SystemExit(
                "No valid features produced ROC curves. Check labels and feature columns."
            )

        overlay_path = plot_dir / "roc_overlay.png"
        if not merge:
            self._plot_overlay(curves, "Univariate ROC curves", overlay_path)

        out_df = pd.DataFrame(rows).sort_values(by="auc", ascending=False)
        out_csv = self.output_dir / "papila_univariate_roc" / "summary.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_csv, index=False)
        return out_df

    def random_forest(
        self,
        include_categorical: bool = True,
        n_estimators: int = 500,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        max_features: str | None = "sqrt",
        class_weight: str | None = "balanced",
        max_samples: float | None = None,
        random_state: int = 42,
        top_n: int = 25,
        n_splits: int = 5,
        drop_missing: bool = False,
        nerf: bool = False,
        drop_age: bool = False,
    ) -> pd.DataFrame:
        clinical = self._build_clinical()
        df = clinical.df.copy()
        original_exclude = set(self.exclude_cols)
        if nerf:
            self.exclude_cols = set(self.exclude_cols)
            drop_missing = True
        if drop_age:
            self.exclude_cols = set(self.exclude_cols) | {"Age"}
        if drop_missing:
            numeric_cols = self._numeric_columns(df)
            df = df.dropna(subset=numeric_cols)
        X, y, feat_names = self._feature_matrix(
            df, include_categorical=include_categorical
        )

        if X.size == 0 or np.unique(y).size < 2:
            raise SystemExit(
                "Not enough data after filtering labels for Random Forest."
            )

        if nerf:
            n_estimators = 200
            max_depth = 5
            min_samples_leaf = 5
            max_features = "sqrt"
            class_weight = None
            max_samples = 0.7
        self.exclude_cols = original_exclude

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1,
        )
        clf.fit(X, y)
        importances = clf.feature_importances_.astype(float)

        rows = []
        for name, val in zip(feat_names, importances):
            rows.append({"feature": name, "importance": float(val)})

        out_df = pd.DataFrame(rows).sort_values(by="importance", ascending=False)
        out_dir = self.output_dir / (
            "papila_random_forest_nerfed" if nerf else "papila_random_forest"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_dir / "feature_importance.csv", index=False)

        top_df = out_df.head(top_n)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.barh(top_df["feature"], top_df["importance"], color="steelblue")
        ax.invert_yaxis()
        ax.set_xlabel("Importance (Gini)")
        ax.set_title(f"Random Forest Feature Importance")
        fig.tight_layout()
        fig.savefig(out_dir / "feature_importance_top.png", dpi=170)
        plt.close(fig)
        if self.debug:
            age_rows = out_df[out_df["feature"] == "Age"]
            if not age_rows.empty:
                age_imp = float(age_rows["importance"].iloc[0])
                print(f"[debug] RF importance Age = {age_imp:.4f}")

        y_oof, scores_oof = self._oof_scores(clf, X, y, n_splits, random_state)
        fpr, tpr, _, auc_val = self._compute_roc(y_oof, scores_oof)
        curves = self._cv_roc_curves(clf, X, y, n_splits, random_state)
        self._plot_mean_roc(
            curves,
            "Random Forest ROC (mean ± SD)",
            out_dir / "roc_mean.png",
        )

        return out_df

    def _cv_binary_metrics(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int,
        random_state: int,
    ) -> pd.DataFrame:
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        rows = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
                continue
            model.fit(X_train, y_train)
            if hasattr(model, "predict_proba"):
                scores = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(X_test)
            else:
                scores = model.predict(X_test)
            preds = model.predict(X_test)
            auc_val = float(roc_auc_score(y_test, scores))
            acc_val = float(accuracy_score(y_test, preds))
            rows.append(
                {
                    "fold": int(fold),
                    "n": int(len(y_test)),
                    "auc": auc_val,
                    "acc": acc_val,
                }
            )
        return pd.DataFrame(rows)

    def svm(
        self,
        include_categorical: bool = True,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str = "scale",
        n_splits: int = 5,
        random_state: int = 42,
    ) -> pd.DataFrame:
        clinical = self._build_clinical()
        df = clinical.df.copy()
        X, y, feat_names = self._feature_matrix(
            df, include_categorical=include_categorical
        )
        if X.size == 0 or np.unique(y).size < 2:
            raise SystemExit("Not enough data after filtering labels for SVM.")

        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "svm",
                    SVC(
                        kernel=kernel,
                        C=C,
                        gamma=gamma,
                        probability=True,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )
        fold_df = self._cv_binary_metrics(model, X, y, n_splits, random_state)
        if fold_df.empty:
            raise SystemExit("SVM produced no valid folds (check class balance).")

        y_oof, scores_oof = self._oof_scores(model, X, y, n_splits, random_state)
        fpr, tpr, _, auc_val = self._compute_roc(y_oof, scores_oof)
        curves = self._cv_roc_curves(model, X, y, n_splits, random_state)

        summary = pd.DataFrame(
            [
                {
                    "metric": "auc",
                    "mean": float(fold_df["auc"].mean()),
                    "std": float(fold_df["auc"].std(ddof=0)),
                    "oof_auc": float(auc_val),
                },
                {
                    "metric": "acc",
                    "mean": float(fold_df["acc"].mean()),
                    "std": float(fold_df["acc"].std(ddof=0)),
                },
            ]
        )
        out_dir = self.output_dir / "papila_svm"
        out_dir.mkdir(parents=True, exist_ok=True)
        fold_df.to_csv(out_dir / "fold_metrics.csv", index=False)
        summary.to_csv(out_dir / "summary.csv", index=False)
        self._plot_mean_roc(
            curves,
            "SVM ROC (mean ± SD)",
            out_dir / "roc_mean.png",
        )
        return fold_df

    def knn(
        self,
        include_categorical: bool = True,
        n_neighbors: int = 5,
        weights: str = "distance",
        n_splits: int = 5,
        random_state: int = 42,
    ) -> pd.DataFrame:
        clinical = self._build_clinical()
        df = clinical.df.copy()
        X, y, feat_names = self._feature_matrix(
            df, include_categorical=include_categorical
        )
        if X.size == 0 or np.unique(y).size < 2:
            raise SystemExit("Not enough data after filtering labels for KNN.")

        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("knn", KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)),
            ]
        )
        fold_df = self._cv_binary_metrics(model, X, y, n_splits, random_state)
        if fold_df.empty:
            raise SystemExit("KNN produced no valid folds (check class balance).")

        y_oof, scores_oof = self._oof_scores(model, X, y, n_splits, random_state)
        fpr, tpr, _, auc_val = self._compute_roc(y_oof, scores_oof)
        curves = self._cv_roc_curves(model, X, y, n_splits, random_state)

        summary = pd.DataFrame(
            [
                {
                    "metric": "auc",
                    "mean": float(fold_df["auc"].mean()),
                    "std": float(fold_df["auc"].std(ddof=0)),
                    "oof_auc": float(auc_val),
                },
                {
                    "metric": "acc",
                    "mean": float(fold_df["acc"].mean()),
                    "std": float(fold_df["acc"].std(ddof=0)),
                },
            ]
        )
        out_dir = self.output_dir / "papila_knn"
        out_dir.mkdir(parents=True, exist_ok=True)
        fold_df.to_csv(out_dir / "fold_metrics.csv", index=False)
        summary.to_csv(out_dir / "summary.csv", index=False)
        self._plot_mean_roc(
            curves,
            "KNN ROC (mean ± SD)",
            out_dir / "roc_mean.png",
        )
        return fold_df

    def logistic_regression(
        self,
        include_categorical: bool = True,
        C: float = 1.0,
        max_iter: int = 1000,
        n_splits: int = 5,
        random_state: int = 42,
        drop_age: bool = False,
        nerf: bool = False,
    ) -> pd.DataFrame:
        clinical = self._build_clinical()
        df = clinical.df.copy()
        original_exclude = set(self.exclude_cols)
        if drop_age:
            self.exclude_cols = set(self.exclude_cols) | {"Age"}
        X, y, feat_names = self._feature_matrix(
            df, include_categorical=include_categorical
        )
        self.exclude_cols = original_exclude
        if X.size == 0 or np.unique(y).size < 2:
            raise SystemExit(
                "Not enough data after filtering labels for Logistic Regression."
            )

        class_weight = "balanced"
        penalty = "l2"
        solver = "lbfgs"
        if nerf:
            C = 0.05
            class_weight = None
            penalty = "l1"
            solver = "liblinear"

        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        C=C,
                        max_iter=max_iter,
                        class_weight=class_weight,
                        penalty=penalty,
                        solver=solver,
                    ),
                ),
            ]
        )
        fold_df = self._cv_binary_metrics(model, X, y, n_splits, random_state)
        if fold_df.empty:
            raise SystemExit(
                "Logistic Regression produced no valid folds (check class balance)."
            )

        y_oof, scores_oof = self._oof_scores(model, X, y, n_splits, random_state)
        fpr, tpr, _, auc_val = self._compute_roc(y_oof, scores_oof)
        curves = self._cv_roc_curves(model, X, y, n_splits, random_state)

        summary = pd.DataFrame(
            [
                {
                    "metric": "auc",
                    "mean": float(fold_df["auc"].mean()),
                    "std": float(fold_df["auc"].std(ddof=0)),
                    "oof_auc": float(auc_val),
                },
                {
                    "metric": "acc",
                    "mean": float(fold_df["acc"].mean()),
                    "std": float(fold_df["acc"].std(ddof=0)),
                },
            ]
        )
        out_dir = self.output_dir / "papila_logistic_regression"
        out_dir.mkdir(parents=True, exist_ok=True)
        fold_df.to_csv(out_dir / "fold_metrics.csv", index=False)
        summary.to_csv(out_dir / "summary.csv", index=False)
        self._plot_mean_roc(
            curves,
            "Logistic Regression ROC (mean ± SD)",
            out_dir / "roc_mean.png",
        )
        return fold_df


ba = basic_analytics()
roc_df = ba.univariate_roc(merge=False, include_categorical=False)
rf_df = ba.random_forest(include_categorical=True, nerf=False)
svm_df = ba.svm(include_categorical=True)
knn_df = ba.knn(include_categorical=True)
lr_df = ba.logistic_regression(include_categorical=True, nerf=True)
clinical = ba._build_clinical()
clinical.df["Diagnosis"].value_counts()
