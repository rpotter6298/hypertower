#!/usr/bin/env python3
"""Derived analysis wrapper for grid search analytics."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm


class derived_analysis:
    def __init__(
        self,
        analysis_dir: Path | str,
        exclude_keys: Optional[Iterable[str]] = None,
        classification_mode: str = "binary",
    ) -> None:
        self.analysis_dir = Path(analysis_dir)
        if not self.analysis_dir.exists():
            raise FileNotFoundError(f"analysis_dir does not exist: {self.analysis_dir}")

        mode = str(classification_mode).strip().lower()
        if mode not in {"binary", "multiclass"}:
            raise ValueError(
                f"classification_mode must be 'binary' or 'multiclass' (got {classification_mode!r})"
            )
        self.classification_mode = mode
        self.exclude_keys = set(exclude_keys or [])

        self.fusion_corrections = pd.DataFrame()
        self.fusion_errors = pd.DataFrame()
        self.statistics_df = pd.DataFrame()
        self.primary_metrics = pd.DataFrame()
        self.fusion_performance_corr = pd.DataFrame()
        self.param_perf_corr = pd.DataFrame()
        self.se_mode_effects = {}

    def identify_fusion_corrections(
        self, shallow: bool = True, existing: bool = True
    ) -> pd.DataFrame:
        cache_path = self._fusion_corrections_path()
        if existing and cache_path.exists():
            df = self._read_fusion_corrections(cache_path)
            self.fusion_corrections = df
            errors_path = self._fusion_errors_path()
            if errors_path.exists():
                self.fusion_errors = self._read_fusion_errors(errors_path)
            else:
                self.fusion_errors = pd.DataFrame()
            self.statistics_df = self._build_statistics_df(df, shallow=shallow)
            return self.fusion_corrections

        rows: List[Dict[str, object]] = []
        error_rows: List[Dict[str, object]] = []
        stats_rows: List[Dict[str, object]] = []

        for run_dir in self._iter_run_dirs(shallow=shallow, show_progress=True):
            summary = self._read_summary(run_dir)
            cli = self._read_cli_args(run_dir)
            mode = self._infer_mode(summary, cli)
            if mode != self.classification_mode:
                continue

            run_id = self._read_run_id(run_dir, summary)
            folds = self._available_folds(run_dir, summary)
            run_count = 0
            grid_params = self._grid_params_from_cli(cli, summary)

            for fold in folds:
                y_true = self._load_y_true(run_dir, fold)
                if y_true is None:
                    continue
                epoch_prob_paths = self._collect_epoch_prob_paths(run_dir, fold)
                if not epoch_prob_paths:
                    base_paths = self._collect_base_prob_paths(run_dir, fold)
                    if base_paths:
                        epoch_hint = self._fold_epoch_hint(summary, fold)
                        epoch_prob_paths = {
                            epoch_hint if epoch_hint is not None else 0: base_paths
                        }

                for epoch, paths in epoch_prob_paths.items():
                    arrays = {
                        head: self._load_probs_array(path)
                        for head, path in paths.items()
                    }
                    if not self._has_all_heads(arrays):
                        continue
                    events = self._fusion_corrections_for_probs(
                        y_true, arrays, run_id, fold, epoch
                    )
                    run_count += len(events)
                    rows.extend(events)
                    errors = self._fusion_errors_for_probs(
                        y_true, arrays, run_id, fold, epoch
                    )
                    error_rows.extend(errors)

            stats_rows.append(
                {
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "classification_mode": mode,
                    "fusion_corrections": int(run_count),
                    **grid_params,
                }
            )

        self.fusion_corrections = pd.DataFrame(rows)
        self.fusion_errors = pd.DataFrame(error_rows)
        self.statistics_df = pd.DataFrame(stats_rows)
        return self.fusion_corrections

    def write_fusion_corrections(self, output_path: Path | str | None = None) -> Path:
        if self.fusion_corrections.empty:
            self.identify_fusion_corrections(existing=True)
        path = Path(output_path) if output_path else self._fusion_corrections_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fusion_corrections.to_csv(path, index=False)
        return path

    def write_fusion_errors(self, output_path: Path | str | None = None) -> Path:
        if self.fusion_errors.empty:
            self.identify_fusion_corrections(existing=True)
        path = Path(output_path) if output_path else self._fusion_errors_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fusion_errors.to_csv(path, index=False)
        return path

    def populate_primary_metrics(
        self, shallow: bool = True, show_progress: bool = True, existing: bool = True
    ) -> pd.DataFrame:
        cache_path = self._primary_metrics_path()
        if existing and cache_path.exists():
            df = self._read_primary_metrics(cache_path)
            self.primary_metrics = df
            summary_df = self._aggregate_primary_metrics(df)
            if summary_df.empty:
                if self.statistics_df.empty:
                    self.statistics_df = summary_df
            else:
                if self.statistics_df.empty:
                    self.statistics_df = summary_df
                else:
                    self.statistics_df = self.statistics_df.merge(
                        summary_df,
                        on=["run_id", "run_dir", "classification_mode"],
                        how="left",
                    )
            return self.primary_metrics

        rows: List[Dict[str, object]] = []
        for run_dir in self._iter_run_dirs(
            shallow=shallow, show_progress=show_progress
        ):
            summary = self._read_summary(run_dir)
            cli = self._read_cli_args(run_dir)
            mode = self._infer_mode(summary, cli)
            if mode != self.classification_mode:
                continue

            run_id = self._read_run_id(run_dir, summary)
            folds = self._available_folds(run_dir, summary)
            if not folds:
                continue

            for fold in folds:
                log_df = self._read_epoch_log(run_dir, fold)
                best_epoch, holdout_best_epoch = self._extract_best_epochs(log_df)
                if best_epoch is None or holdout_best_epoch is None:
                    continue

                row: Dict[str, object] = {
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "classification_mode": mode,
                    "fold": int(fold),
                    "best_epoch": int(best_epoch),
                    "holdout_best_epoch": int(holdout_best_epoch),
                }

                for head_key, log_suffix, roc_suffix in (
                    ("fused", "fused", "fused"),
                    ("image", "img", "image"),
                    ("metadata", "md", "metadata"),
                ):
                    best_auc = self._load_auc_for_epoch(
                        run_dir, fold, best_epoch, roc_suffix, holdout=False
                    )
                    hold_auc = self._load_auc_for_epoch(
                        run_dir, fold, holdout_best_epoch, roc_suffix, holdout=True
                    )
                    row[f"best_auc_{head_key}"] = best_auc
                    row[f"holdout_best_auc_{head_key}"] = hold_auc

                    if log_df is not None:
                        best_row = self._row_for_epoch(log_df, best_epoch)
                        hold_row = self._row_for_epoch(log_df, holdout_best_epoch)
                        best_acc = self._metric_from_row(best_row, f"acc_{log_suffix}")
                        hold_acc = self._metric_from_row(
                            hold_row, f"holdout_acc_{log_suffix}"
                        )
                        row[f"best_acc_{head_key}"] = best_acc
                        row[f"holdout_best_acc_{head_key}"] = hold_acc

                rows.append(row)

        self.primary_metrics = pd.DataFrame(rows)
        summary_df = self._aggregate_primary_metrics(self.primary_metrics)
        if summary_df.empty:
            if self.statistics_df.empty:
                self.statistics_df = summary_df
        else:
            if self.statistics_df.empty:
                self.statistics_df = summary_df
            else:
                self.statistics_df = self.statistics_df.merge(
                    summary_df,
                    on=["run_id", "run_dir", "classification_mode"],
                    how="left",
                )
        return self.primary_metrics

    def write_primary_metrics(self, output_path: Path | str | None = None) -> Path:
        if self.primary_metrics.empty:
            self.populate_primary_metrics()
        path = Path(output_path) if output_path else self._primary_metrics_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.primary_metrics.to_csv(path, index=False)
        return path

    def plot_fusion_corrections_errors(
        self,
        output_path: Path | str | None = None,
        shallow: bool = True,
        existing: bool = True,
        top_n: int | None = None,
    ) -> pd.DataFrame:
        if self.fusion_corrections.empty:
            self.identify_fusion_corrections(shallow=shallow, existing=existing)
        if self.fusion_corrections.empty:
            raise RuntimeError(
                "fusion_corrections is empty; run identify_fusion_corrections() first."
            )

        corrections = (
            self.fusion_corrections.groupby("run_id")
            .size()
            .rename("fusion_corrections")
        )
        if self.fusion_errors.empty:
            errors = corrections.copy() * 0
            errors.name = "fusion_errors"
        else:
            errors = self.fusion_errors.groupby("run_id").size().rename("fusion_errors")

        df = pd.concat([corrections, errors], axis=1).fillna(0).reset_index()

        df = df.sort_values(by="run_id")
        if top_n is not None:
            df = df.head(int(top_n))

        out_path = (
            Path(output_path)
            if output_path
            else (
                self.analysis_dir
                / "plots"
                / f"fusion_corrections_errors_{self.classification_mode}.png"
            )
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 4.8))
        x = np.arange(len(df))
        ax.bar(
            x,
            df["fusion_corrections"],
            color="steelblue",
            width=1.0,
            label="fusion_corrections",
        )
        ax.bar(
            x,
            df["fusion_errors"],
            bottom=df["fusion_corrections"],
            color="tomato",
            width=1.0,
            label="fusion_errors",
        )
        ax.set_xticks([])
        ax.set_ylabel("Count")
        ax.set_title(
            f"Fusion corrections + errors per run ({self.classification_mode})"
        )
        ax.legend(loc="upper right")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        fig.tight_layout()
        fig.savefig(out_path, dpi=170)
        plt.close(fig)

        return df

    def plot_conf_delta_boxplot(
        self,
        output_path: Path | str | None = None,
        shallow: bool = True,
        existing: bool = True,
        top_n: int | None = None,
    ) -> pd.DataFrame:
        if self.fusion_corrections.empty:
            self.identify_fusion_corrections(shallow=shallow, existing=existing)
        if self.fusion_corrections.empty:
            raise RuntimeError(
                "fusion_corrections is empty; run identify_fusion_corrections() first."
            )

        df = self.fusion_corrections.copy()
        df["conf_delta"] = df["conf_fused"] - 0.5 * (df["conf_img"] + df["conf_md"])

        mean_order = df.groupby("run_id")["conf_delta"].mean().sort_values()
        run_order = mean_order.index.tolist()
        if top_n is not None:
            run_order = run_order[: int(top_n)]

        data = [
            df.loc[df["run_id"] == run_id, "conf_delta"].values for run_id in run_order
        ]

        out_path = (
            Path(output_path)
            if output_path
            else (
                self.analysis_dir
                / "plots"
                / f"conf_delta_box_{self.classification_mode}.png"
            )
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.boxplot(data, widths=0.6, showfliers=False)
        ax.set_xticks([])
        ax.set_ylabel("conf_delta (fused - mean(towers))")
        ax.set_title(f"Confidence delta per run ({self.classification_mode})")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        fig.tight_layout()
        fig.savefig(out_path, dpi=170)
        plt.close(fig)

        return df

    def param_performance_correlations(
        self,
        output_path: Path | str | None = None,
        shallow: bool = True,
        existing: bool = True,
        method: str = "spearman",
        cat_method: str = "kruskal",
    ) -> pd.DataFrame:
        if self.statistics_df.empty:
            self.identify_fusion_corrections(shallow=shallow, existing=existing)
        if self.primary_metrics.empty:
            self.populate_primary_metrics(shallow=shallow, existing=existing)

        df = self.statistics_df.copy()
        if df.empty:
            raise RuntimeError(
                "statistics_df is empty; run identify_fusion_corrections() first."
            )

        if self.fusion_corrections.empty:
            self.identify_fusion_corrections(shallow=shallow, existing=existing)

        conf_delta = None
        if not self.fusion_corrections.empty:
            fc = self.fusion_corrections.copy()
            fc["conf_delta"] = fc["conf_fused"] - 0.5 * (fc["conf_img"] + fc["conf_md"])
            conf_delta = (
                fc.groupby("run_id")["conf_delta"].mean().rename("conf_delta_mean")
            )
            df = df.merge(conf_delta.reset_index(), on="run_id", how="left")

        if self.fusion_errors.empty:
            errors_path = self._fusion_errors_path()
            if errors_path.exists():
                self.fusion_errors = self._read_fusion_errors(errors_path)
        if not self.fusion_errors.empty and "fusion_corrections" in df.columns:
            err_counts = (
                self.fusion_errors.groupby("run_id").size().rename("fusion_errors")
            )
            df = df.merge(err_counts.reset_index(), on="run_id", how="left")
            df["fusion_errors"] = df["fusion_errors"].fillna(0)
            eps = 1e-6
            df["error_correction_ratio"] = (df["fusion_errors"] + eps) / (
                df["fusion_corrections"] + eps
            )

        metric_cols = []
        for cand in ("holdout_best_acc_fused_mean", "best_acc_fused_mean"):
            if cand in df.columns:
                metric_cols.append(("acc", cand))
                break
        for cand in ("holdout_best_auc_fused_mean", "best_auc_fused_mean"):
            if cand in df.columns:
                metric_cols.append(("auc", cand))
                break
        if "fusion_corrections" in df.columns:
            metric_cols.append(("fusion_corrections", "fusion_corrections"))
        if conf_delta is not None and "conf_delta_mean" in df.columns:
            metric_cols.append(("conf_delta", "conf_delta_mean"))
        if "error_correction_ratio" in df.columns:
            metric_cols.append(("error_correction_ratio", "error_correction_ratio"))

        if not metric_cols:
            raise RuntimeError("No metrics found in statistics_df for correlation.")

        grid_param_keys = [
            "crop_variant",
            "crop_normalize",
            "crop_weights",
            "crop_tta",
            "loss_mode",
            "thaw_mode",
            "se_mode",
            "se_bridge_pre_norm",
            "se_tower_pre_norm",
        ]
        param_cols = [c for c in grid_param_keys if c in df.columns]

        def _to_float(v):
            if v is None:
                return None
            if isinstance(v, bool):
                return None
            if isinstance(v, (int, float)) and not math.isnan(float(v)):
                return float(v)
            try:
                return float(v)
            except Exception:
                return None

        def _format_value(v: object) -> str:
            if v is None:
                return ""
            if isinstance(v, bool):
                return "true" if v else "false"
            if isinstance(v, int):
                return str(v)
            if isinstance(v, float):
                return f"{v:.6g}"
            return str(v)

        def _rankdata(vals: List[float]) -> List[float]:
            order = sorted(range(len(vals)), key=lambda i: vals[i])
            ranks = [0.0] * len(vals)
            i = 0
            while i < len(vals):
                j = i
                while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
                    j += 1
                avg_rank = (i + j) / 2.0 + 1.0
                for k in range(i, j + 1):
                    ranks[order[k]] = avg_rank
                i = j + 1
            return ranks

        def _pearson(x: List[float], y: List[float]) -> Optional[float]:
            if len(x) < 2:
                return None
            mx = sum(x) / len(x)
            my = sum(y) / len(y)
            num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
            denx = sum((xi - mx) ** 2 for xi in x)
            deny = sum((yi - my) ** 2 for yi in y)
            if denx <= 0 or deny <= 0:
                return None
            return num / math.sqrt(denx * deny)

        def _spearman(x: List[float], y: List[float]) -> Optional[float]:
            return _pearson(_rankdata(x), _rankdata(y))

        def _eta(categories: List[object], values: List[float]) -> Optional[float]:
            if len(values) < 2:
                return None
            overall = sum(values) / len(values)
            total = sum((v - overall) ** 2 for v in values)
            if total <= 0:
                return None
            groups = {}
            for cat, val in zip(categories, values):
                groups.setdefault(cat, []).append(val)
            between = 0.0
            for vals in groups.values():
                avg = sum(vals) / len(vals)
                between += len(vals) * (avg - overall) ** 2
            return math.sqrt(between / total)

        cat_method_norm = cat_method.strip().lower()
        if cat_method_norm not in {"eta", "anova", "kruskal"}:
            raise ValueError(
                f"cat_method must be 'eta', 'anova', or 'kruskal' (got {cat_method!r})"
            )

        rows: List[Dict[str, object]] = []
        for name, metric_col in metric_cols:
            metric_vals = df[metric_col]
            for param in param_cols:
                param_vals = df[param]
                pairs = [
                    (p, m)
                    for p, m in zip(param_vals, metric_vals)
                    if m is not None and not (isinstance(m, float) and math.isnan(m))
                ]
                if len(pairs) < 3:
                    continue
                p_vals, m_vals = zip(*pairs)
                group_means: Dict[object, float] = {}
                for p, m in pairs:
                    group_means.setdefault(p, []).append(m)
                group_means = {
                    k: float(sum(v) / len(v)) for k, v in group_means.items()
                }
                if name == "error_correction_ratio":
                    best_value = min(group_means.items(), key=lambda item: item[1])[0]
                else:
                    best_value = max(group_means.items(), key=lambda item: item[1])[0]
                num_vals = []
                numeric_ok = True
                for v in p_vals:
                    num = _to_float(v)
                    if num is None:
                        numeric_ok = False
                        break
                    num_vals.append(num)
                if numeric_ok and len(set(num_vals)) >= 3:
                    p_val = None
                    if method == "spearman":
                        try:
                            corr, p_val = stats.spearmanr(num_vals, list(m_vals))
                        except Exception:
                            corr = None
                    else:
                        try:
                            corr, p_val = stats.pearsonr(num_vals, list(m_vals))
                        except Exception:
                            corr = None
                    if corr is not None and corr != corr:
                        corr = None
                    rows.append(
                        {
                            "metric": name,
                            "metric_col": metric_col,
                            "param": param,
                            "type": "numeric",
                            "n": len(pairs),
                            "corr": corr,
                            "stat": corr,
                            "p_value": p_val,
                            "method": method,
                            "best": _format_value(best_value),
                        }
                    )
                else:
                    stat_val = None
                    p_val = None
                    corr = None
                    groups: Dict[object, List[float]] = {}
                    for p, m in pairs:
                        groups.setdefault(p, []).append(m)
                    group_vals = [vals for vals in groups.values() if len(vals) > 0]

                    if cat_method_norm == "eta":
                        corr = _eta(list(p_vals), list(m_vals))
                        stat_val = corr
                    elif cat_method_norm == "anova":
                        if len(group_vals) >= 2:
                            try:
                                stat_val, p_val = stats.f_oneway(*group_vals)
                                corr = stat_val
                            except Exception:
                                stat_val = None
                    elif cat_method_norm == "kruskal":
                        if len(group_vals) >= 2:
                            try:
                                stat_val, p_val = stats.kruskal(*group_vals)
                                corr = stat_val
                            except Exception:
                                stat_val = None

                    rows.append(
                        {
                            "metric": name,
                            "metric_col": metric_col,
                            "param": param,
                            "type": "categorical",
                            "n": len(pairs),
                            "corr": corr,
                            "stat": stat_val,
                            "p_value": p_val,
                            "method": cat_method_norm,
                            "best": _format_value(best_value),
                        }
                    )

        out_df = pd.DataFrame(rows).sort_values(
            by=["metric", "corr"], ascending=[True, False]
        )
        self.param_perf_corr = out_df
        out_path = (
            Path(output_path)
            if output_path
            else (
                self.analysis_dir
                / "plots"
                / f"param_perf_corr_{self.classification_mode}.csv"
            )
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        return out_df

    def plot_param_perf_corr_panels(
        self,
        corr_df: pd.DataFrame,
        output_path: Path | str | None = None,
    ) -> pd.DataFrame:
        if corr_df.empty:
            raise RuntimeError(
                "corr_df is empty; run param_performance_correlations() first."
            )

        metrics = ["auc", "acc", "fusion_corrections", "conf_delta"]
        auc_df = corr_df[corr_df["metric"] == "auc"].copy()
        if auc_df.empty:
            raise RuntimeError("No 'auc' metric rows found in corr_df.")

        auc_df = auc_df.sort_values(by="corr", ascending=False)
        order = auc_df["param"].tolist()

        out_path = (
            Path(output_path)
            if output_path
            else (
                self.analysis_dir
                / "plots"
                / f"param_perf_corr_panels_{self.classification_mode}.png"
            )
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            sub = corr_df[corr_df["metric"] == metric].set_index("param")
            sub = sub.reindex(order)
            values = sub["corr"].astype(float).values
            y = np.arange(len(order))
            ax.barh(y, values, color="steelblue")
            ax.axvline(0.0, color="black", lw=1)
            ax.set_title(metric)
            ax.set_yticks(y)
            ax.set_yticklabels(order, fontsize=7)
            ax.grid(True, axis="x", alpha=0.3, linestyle="--")

            # annotate p-values when available
            for i, param in enumerate(order):
                if param not in sub.index:
                    continue
                p_val = sub.loc[param, "p_value"]
                if p_val is None or (isinstance(p_val, float) and np.isnan(p_val)):
                    continue
                ax.text(
                    values[i] if not np.isnan(values[i]) else 0.0,
                    i,
                    f" p={p_val:.3g}",
                    va="center",
                    ha="left" if values[i] >= 0 else "right",
                    fontsize=7,
                )

            # print p-values to console for each metric
            print(f"\n[{metric}] p-values")
            for param in order:
                if param not in sub.index:
                    continue
                p_val = sub.loc[param, "p_value"]
                if p_val is None or (isinstance(p_val, float) and np.isnan(p_val)):
                    continue
                print(f"  {param}: p={p_val:.4g}")

        fig.suptitle("Parameter correlations (ordered by AUC correlation)", fontsize=12)
        fig.tight_layout(rect=[0, 0.02, 1, 0.96])
        fig.savefig(out_path, dpi=170)
        plt.close(fig)

        return corr_df

    def se_mode_effects_summary(
        self,
        metric: str = "auc",
        metric_col: str | None = None,
        head: str = "fused",
        prefer_holdout: bool = True,
        top_n: int | None = None,
        top_metric_col: str | None = None,
        output_dir: Path | str | None = None,
        pairwise_method: str = "mannwhitney",
        shallow: bool = True,
        existing: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        if self.statistics_df.empty:
            self.identify_fusion_corrections(shallow=shallow, existing=existing)
        if self.primary_metrics.empty:
            self.populate_primary_metrics(shallow=shallow, existing=existing)

        df = self.statistics_df.copy()
        if df.empty:
            raise RuntimeError(
                "statistics_df is empty; run identify_fusion_corrections() first."
            )

        metric_norm = metric.strip().lower()
        if metric_norm not in {"acc", "auc"}:
            raise ValueError(f"metric must be 'acc' or 'auc' (got {metric!r})")

        if metric_col is None:
            candidates = []
            if prefer_holdout:
                candidates.append(f"holdout_best_{metric_norm}_{head}_mean")
                candidates.append(f"best_{metric_norm}_{head}_mean")
            else:
                candidates.append(f"best_{metric_norm}_{head}_mean")
                candidates.append(f"holdout_best_{metric_norm}_{head}_mean")
            for cand in candidates:
                if cand in df.columns:
                    metric_col = cand
                    break

        if metric_col is None or metric_col not in df.columns:
            raise RuntimeError(
                "Could not find a metric column to summarize; run populate_primary_metrics() "
                "or pass metric_col explicitly."
            )

        if "se_mode" not in df.columns:
            raise RuntimeError("statistics_df is missing se_mode column.")

        use_cols = ["run_id", "se_mode", metric_col]
        if "se_bridge_pre_norm" in df.columns:
            use_cols.append("se_bridge_pre_norm")
        if "se_tower_pre_norm" in df.columns:
            use_cols.append("se_tower_pre_norm")

        df = df[use_cols].copy()
        df = df.dropna(subset=[metric_col, "se_mode"])
        if df.empty:
            raise RuntimeError(
                "No rows available after filtering for se_mode and metric."
            )

        if top_n is not None:
            top_metric = top_metric_col or metric_col
            if top_metric not in df.columns:
                raise RuntimeError(
                    f"top_metric_col {top_metric!r} not found in statistics_df."
                )
            df = df.sort_values(by=top_metric, ascending=False).head(int(top_n))
            if df.empty:
                raise RuntimeError("No rows available after applying top_n filter.")

        summary = (
            df.groupby("se_mode")[metric_col]
            .agg(["count", "mean", "median", "std"])
            .reset_index()
            .rename(columns={"count": "n"})
        )

        # Pairwise comparisons
        pairwise_rows: List[Dict[str, object]] = []
        modes = summary["se_mode"].tolist()
        pairwise_method_norm = pairwise_method.strip().lower()
        if pairwise_method_norm not in {"mannwhitney", "ttest"}:
            raise ValueError(
                f"pairwise_method must be 'mannwhitney' or 'ttest' (got {pairwise_method!r})"
            )

        def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
            if len(a) < 2 or len(b) < 2:
                return float("nan")
            va = np.var(a, ddof=1)
            vb = np.var(b, ddof=1)
            pooled = ((len(a) - 1) * va + (len(b) - 1) * vb) / max(
                len(a) + len(b) - 2, 1
            )
            if pooled <= 0:
                return float("nan")
            return (np.mean(a) - np.mean(b)) / math.sqrt(pooled)

        for i, m1 in enumerate(modes):
            vals1 = df.loc[df["se_mode"] == m1, metric_col].astype(float).values
            if vals1.size == 0:
                continue
            for m2 in modes[i + 1 :]:
                vals2 = df.loc[df["se_mode"] == m2, metric_col].astype(float).values
                if vals2.size == 0:
                    continue
                p_val = None
                stat_val = None
                if pairwise_method_norm == "mannwhitney":
                    try:
                        stat_val, p_val = stats.mannwhitneyu(
                            vals1, vals2, alternative="two-sided"
                        )
                    except Exception:
                        stat_val, p_val = None, None
                else:
                    try:
                        stat_val, p_val = stats.ttest_ind(vals1, vals2, equal_var=False)
                    except Exception:
                        stat_val, p_val = None, None

                pairwise_rows.append(
                    {
                        "metric_col": metric_col,
                        "se_mode_a": m1,
                        "se_mode_b": m2,
                        "n_a": int(vals1.size),
                        "n_b": int(vals2.size),
                        "mean_a": float(np.mean(vals1)),
                        "mean_b": float(np.mean(vals2)),
                        "mean_diff": float(np.mean(vals1) - np.mean(vals2)),
                        "median_a": float(np.median(vals1)),
                        "median_b": float(np.median(vals2)),
                        "median_diff": float(np.median(vals1) - np.median(vals2)),
                        "cohens_d": _cohens_d(vals1, vals2),
                        "stat": stat_val,
                        "p_value": p_val,
                        "method": pairwise_method_norm,
                    }
                )

        pairwise_df = pd.DataFrame(pairwise_rows)

        # Stratified by pre-norm options (within relevant se_mode)
        bridge_df = pd.DataFrame()
        if "se_bridge_pre_norm" in df.columns:
            bridge_df = (
                df[df["se_mode"].isin(["bridge", "both"])]
                .groupby(["se_mode", "se_bridge_pre_norm"])[metric_col]
                .agg(["count", "mean", "median", "std"])
                .reset_index()
                .rename(columns={"count": "n"})
            )
        tower_df = pd.DataFrame()
        if "se_tower_pre_norm" in df.columns:
            tower_df = (
                df[df["se_mode"].isin(["tower", "both"])]
                .groupby(["se_mode", "se_tower_pre_norm"])[metric_col]
                .agg(["count", "mean", "median", "std"])
                .reset_index()
                .rename(columns={"count": "n"})
            )

        result = {
            "summary": summary,
            "pairwise": pairwise_df,
            "bridge_pre_norm": bridge_df,
            "tower_pre_norm": tower_df,
        }
        self.se_mode_effects = result

        if output_dir is not None:
            out_dir = Path(output_dir)
        else:
            out_dir = self.analysis_dir / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_dir / f"se_mode_summary_{metric_col}.csv", index=False)
        if not pairwise_df.empty:
            pairwise_df.to_csv(
                out_dir / f"se_mode_pairwise_{metric_col}.csv", index=False
            )
        if not bridge_df.empty:
            bridge_df.to_csv(
                out_dir / f"se_mode_bridge_pre_norm_{metric_col}.csv", index=False
            )
        if not tower_df.empty:
            tower_df.to_csv(
                out_dir / f"se_mode_tower_pre_norm_{metric_col}.csv", index=False
            )

        return result

    def fusion_corrections_correlation(
        self,
        output_path: Path | str | None = None,
        method: str = "pearson",
        metric_type: str = "acc",
    ) -> pd.DataFrame:
        if self.fusion_corrections.empty:
            raise RuntimeError(
                "fusion_corrections is empty; run identify_fusion_corrections() first."
            )
        if self.primary_metrics.empty:
            raise RuntimeError(
                "primary_metrics is empty; run populate_primary_metrics() first."
            )
        if "fold" not in self.primary_metrics.columns:
            raise RuntimeError(
                "primary_metrics missing fold column; refresh populate_primary_metrics()."
            )

        method_norm = method.strip().lower()
        if method_norm not in {"pearson", "spearman"}:
            raise ValueError(f"method must be 'pearson' or 'spearman' (got {method!r})")

        metric_norm = metric_type.strip().lower()
        if metric_norm not in {"acc", "auc"}:
            raise ValueError(
                f"metric_type must be 'acc' or 'auc' (got {metric_type!r})"
            )

        metric_cols = [
            c
            for c in self.primary_metrics.columns
            if c.startswith(("best_", "holdout_best_"))
            and f"_{metric_norm}_" in c
            and not c.endswith(("_mean", "_sd"))
        ]
        if not metric_cols:
            raise RuntimeError(
                "No primary metric columns found in primary_metrics; run populate_primary_metrics() first."
            )

        fold_counts = self._fold_sample_and_opportunity_counts(self.primary_metrics)

        best_epochs = self.primary_metrics[
            ["run_id", "run_dir", "fold", "best_epoch"]
        ].dropna()
        warmup_map = self._build_warmup_map(best_epochs)
        best_epochs = best_epochs.merge(warmup_map, on="run_id", how="left")
        best_epochs["warmup_end"] = best_epochs["warmup_end"].fillna(0).astype(int)
        best_epochs["best_epoch"] = best_epochs["best_epoch"].astype(int)

        events = self.fusion_corrections.merge(
            best_epochs[["run_id", "fold", "best_epoch", "warmup_end"]],
            on=["run_id", "fold"],
            how="inner",
        )
        if "epoch" in events.columns:
            events = events[
                (events["epoch"] >= events["warmup_end"])
                & (events["epoch"] <= events["best_epoch"])
            ]

        counts = (
            events.groupby(["run_id", "fold"], as_index=False)
            .size()
            .rename(columns={"size": "fusion_corrections"})
        )
        merged = self.primary_metrics.merge(counts, on=["run_id", "fold"], how="left")
        merged = merged.merge(fold_counts, on=["run_id", "run_dir", "fold"], how="left")
        merged["fusion_corrections"] = merged["fusion_corrections"].fillna(0)
        merged["n_samples"] = merged["n_samples"].replace(0, np.nan)
        merged["both_wrong"] = merged["both_wrong"].replace(0, np.nan)
        merged["fusion_corrections_rate"] = (
            merged["fusion_corrections"] / merged["n_samples"]
        )
        merged["fusion_corrections_per_opportunity"] = (
            merged["fusion_corrections"] / merged["both_wrong"]
        )

        merged = self._add_fusion_gain_columns(merged)
        gain_cols = [
            c
            for c in merged.columns
            if c.endswith("_fusion_gain")
            and c.startswith(("best_", "holdout_best_"))
            and f"_{metric_norm}_" in c
        ]

        error_counts = self._fusion_errors_counts(merged)
        merged = merged.merge(error_counts, on=["run_id", "fold"], how="left")
        merged["fusion_errors"] = merged["fusion_errors"].fillna(0)
        eps = 1e-6
        merged["correction_error_rate"] = (merged["fusion_corrections"] + eps) / (
            merged["fusion_errors"] + eps
        )

        rows = []
        x_metrics = [
            "fusion_corrections",
            "fusion_corrections_rate",
            "fusion_corrections_per_opportunity",
            "fusion_errors",
            "correction_error_rate",
        ]
        all_metrics = metric_cols + gain_cols
        for x in x_metrics:
            if x not in merged.columns:
                continue
            for col in all_metrics:
                sub = merged[[x, col]].dropna()
                if len(sub) < 2:
                    corr = np.nan
                else:
                    corr = float(sub[x].corr(sub[col], method=method_norm))
                rows.append(
                    {
                        "x_metric": x,
                        "metric": col,
                        "corr": corr,
                        "n": int(len(sub)),
                        "metric_type": metric_norm,
                    }
                )

        out_df = pd.DataFrame(rows).sort_values(
            by=["x_metric", "corr"], ascending=[True, False]
        )
        self.fusion_performance_corr = out_df

        plot_x = "fusion_corrections_per_opportunity"
        if plot_x not in out_df["x_metric"].unique():
            plot_x = "fusion_corrections"
        plot_df = out_df[out_df["x_metric"] == plot_x]

        path = (
            Path(output_path)
            if output_path
            else self._fusion_performance_corr_path(method_norm, metric_norm)
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        self._plot_correlation_bars(
            plot_df, path, method=method_norm, x_metric=plot_x, metric_type=metric_norm
        )
        return out_df

    def plot_fusion_perf_summary(
        self,
        corr_acc: pd.DataFrame,
        corr_auc: pd.DataFrame,
        output_path: Path | str | None = None,
        method: str = "spearman",
        x_metric: str = "fusion_corrections",
    ) -> pd.DataFrame:
        keep_templates = [
            "best_acc_fused",
            "holdout_best_acc_fused",
            "best_acc_fusion_gain",
            "holdout_best_acc_fusion_gain",
            "best_auc_fused",
            "holdout_best_auc_fused",
            "best_auc_fusion_gain",
            "holdout_best_auc_fusion_gain",
        ]

        def _select(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            sub = df[df["x_metric"] == x_metric].copy()
            sub = sub[sub["metric"].isin(keep_templates)]
            sub = sub.drop_duplicates(subset=["metric"])
            sub["metric"] = sub["metric"].str.replace("_fused", "", regex=False)
            return sub

        acc_df = _select(corr_acc)
        auc_df = _select(corr_auc)
        merged = pd.concat([acc_df, auc_df], axis=0, ignore_index=True)
        if merged.empty:
            raise RuntimeError("No matching rows found in corr_acc/corr_auc.")

        merged = (
            merged.set_index("metric")
            .loc[[m.replace("_fused", "") for m in keep_templates]]
            .reset_index()
        )

        safe_x = x_metric.replace("fusion_", "")
        out_path = (
            Path(output_path)
            if output_path
            else (
                self.analysis_dir
                / "plots"
                / f"fusion_{safe_x}_vs_performance_{method}.png"
            )
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._plot_correlation_bars(
            merged,
            out_path,
            method=method,
            x_metric=x_metric,
            metric_type="acc/auc",
            title=f"Fusion corrections vs performance ({method})",
        )
        return merged

    def _iter_run_dirs(
        self, shallow: bool = True, show_progress: bool = False
    ) -> Iterator[Path]:
        if shallow:
            candidates = [p for p in sorted(self.analysis_dir.iterdir()) if p.is_dir()]
        else:
            candidates = [p for p in self.analysis_dir.rglob("*") if p.is_dir()]

        iterator = (
            tqdm(candidates, desc="Scanning runs", unit="run", leave=False)
            if show_progress
            else candidates
        )
        for run_dir in iterator:
            summary = run_dir / "summary.json"
            cli = run_dir / "cli_args.json"
            if summary.exists() or cli.exists():
                yield run_dir

    def _fusion_corrections_path(self) -> Path:
        fname = f"fusion_corrections_{self.classification_mode}.csv"
        return self.analysis_dir / fname

    def _fusion_errors_path(self) -> Path:
        fname = f"fusion_errors_{self.classification_mode}.csv"
        return self.analysis_dir / fname

    @staticmethod
    def _read_fusion_errors(path: Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
        return df

    def _primary_metrics_path(self) -> Path:
        fname = f"primary_metrics_{self.classification_mode}.csv"
        return self.analysis_dir / fname

    def _fusion_performance_corr_path(self, method: str, metric_type: str) -> Path:
        fname = f"fusion_performance_corr_{self.classification_mode}_{metric_type}_{method}.png"
        return self.analysis_dir / fname

    @staticmethod
    def _read_primary_metrics(path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def _read_fusion_corrections(path: Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
        return df

    def _build_statistics_df(
        self, df: pd.DataFrame, shallow: bool = True
    ) -> pd.DataFrame:
        counts = {}
        if not df.empty and "run_id" in df.columns:
            counts = df.groupby("run_id").size().to_dict()

        rows: List[Dict[str, object]] = []
        for run_dir in self._iter_run_dirs(shallow=shallow, show_progress=False):
            summary = self._read_summary(run_dir)
            cli = self._read_cli_args(run_dir)
            mode = self._infer_mode(summary, cli)
            if mode != self.classification_mode:
                continue
            run_id = self._read_run_id(run_dir, summary)
            grid_params = self._grid_params_from_cli(cli, summary)
            rows.append(
                {
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "classification_mode": mode,
                    "fusion_corrections": int(counts.get(run_id, 0)),
                    **grid_params,
                }
            )

        return pd.DataFrame(rows)

    @staticmethod
    def _aggregate_primary_metrics(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        required = ["run_id", "run_dir", "classification_mode"]
        if not all(col in df.columns for col in required):
            return pd.DataFrame()
        metric_cols = [
            c
            for c in df.columns
            if c.startswith(("best_", "holdout_best_"))
            and not c.endswith(("_mean", "_sd"))
        ]
        if not metric_cols:
            return pd.DataFrame()

        grouped = df.groupby(required)
        agg = grouped[metric_cols].agg(["mean", "std"])
        agg.columns = [
            f"{col}_{stat}".replace("std", "sd") for col, stat in agg.columns
        ]
        return agg.reset_index()

    def _fold_sample_and_opportunity_counts(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        cols = ["run_id", "run_dir", "fold"]
        rows = []
        for run_id, run_dir, fold in df[cols].drop_duplicates().itertuples(index=False):
            run_path = Path(run_dir)
            y_true = self._load_y_true(run_path, int(fold))
            n_samples = int(len(y_true)) if y_true is not None else np.nan
            both_wrong = self._count_both_wrong(run_path, int(fold), y_true)
            rows.append(
                {
                    "run_id": run_id,
                    "run_dir": str(run_path),
                    "fold": int(fold),
                    "n_samples": n_samples,
                    "both_wrong": both_wrong,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _count_both_wrong(
        run_dir: Path, fold: int, y_true: Optional[np.ndarray]
    ) -> Optional[int]:
        if y_true is None:
            return np.nan
        img_path = run_dir / f"fold{fold}_probs_img.npy"
        md_path = run_dir / f"fold{fold}_probs_md.npy"
        if not img_path.exists() or not md_path.exists():
            return np.nan
        try:
            p_img = np.load(img_path)
            p_md = np.load(md_path)
        except Exception:
            return np.nan
        if p_img.ndim != 2 or p_md.ndim != 2:
            return np.nan
        if len(p_img) != len(y_true) or len(p_md) != len(y_true):
            return np.nan
        img_pred = p_img.argmax(axis=1)
        md_pred = p_md.argmax(axis=1)
        both_wrong = (img_pred != y_true) & (md_pred != y_true)
        return int(both_wrong.sum())

    @staticmethod
    def _add_fusion_gain_columns(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for prefix in ("best", "holdout_best"):
            acc_cols = [
                f"{prefix}_acc_fused",
                f"{prefix}_acc_image",
                f"{prefix}_acc_metadata",
            ]
            auc_cols = [
                f"{prefix}_auc_fused",
                f"{prefix}_auc_image",
                f"{prefix}_auc_metadata",
            ]
            if all(c in out.columns for c in acc_cols):
                max_acc = out[[acc_cols[1], acc_cols[2]]].max(axis=1)
                out[f"{prefix}_acc_fusion_gain"] = out[acc_cols[0]] - max_acc
            if all(c in out.columns for c in auc_cols):
                max_auc = out[[auc_cols[1], auc_cols[2]]].max(axis=1)
                out[f"{prefix}_auc_fusion_gain"] = out[auc_cols[0]] - max_auc
        return out

    def _fusion_errors_counts(self, merged: pd.DataFrame) -> pd.DataFrame:
        if self.fusion_errors.empty:
            return pd.DataFrame(columns=["run_id", "fold", "fusion_errors"])
        counts = (
            self.fusion_errors.groupby(["run_id", "fold"], as_index=False)
            .size()
            .rename(columns={"size": "fusion_errors"})
        )
        return counts

    def _build_warmup_map(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "run_id" not in df.columns or "run_dir" not in df.columns:
            return pd.DataFrame(columns=["run_id", "warmup_end"])
        rows = []
        for run_id, run_dir in (
            df[["run_id", "run_dir"]].drop_duplicates().itertuples(index=False)
        ):
            cli = self._read_cli_args(Path(run_dir))
            warmup_end = self._warmup_end_epoch(cli)
            rows.append({"run_id": run_id, "warmup_end": warmup_end})
        return pd.DataFrame(rows)

    @staticmethod
    def _warmup_end_epoch(cli: Optional[Dict[str, object]]) -> int:
        if not cli:
            return 0
        tower = cli.get("warmup_tower_epochs")
        fused = cli.get("warmup_fused_epochs")
        try:
            tower_val = int(tower) if tower is not None else 0
        except Exception:
            tower_val = 0
        try:
            fused_val = int(fused) if fused is not None else 0
        except Exception:
            fused_val = 0
        return max(0, tower_val + fused_val)

    def _grid_params_from_cli(
        self, cli: Optional[Dict[str, object]], summary: Optional[Dict[str, object]]
    ) -> Dict[str, object]:
        params: Dict[str, object] = {
            "eval_mode": None,
            "crop_variant": None,
            "crop_normalize": None,
            "crop_weights": None,
            "crop_tta": None,
            "loss_mode": None,
            "thaw_mode": None,
            "se_mode": None,
            "se_bridge_pre_norm": None,
            "se_tower_pre_norm": None,
        }

        eval_mode = None
        for payload in (cli, summary):
            if payload and isinstance(payload.get("eval_mode"), str):
                eval_mode = payload["eval_mode"]
                break
        params["eval_mode"] = eval_mode

        if cli:
            crop_weights = cli.get("img_crop_weights")
            crop_normalize = cli.get("img_crop_normalize")
            crop_tta = cli.get("img_crop_tta")
            params["crop_weights"] = crop_weights
            params["crop_normalize"] = crop_normalize
            params["crop_tta"] = crop_tta
            params["crop_variant"] = self._infer_crop_variant(crop_weights)
            params["loss_mode"] = self._infer_loss_mode(cli)
            params["thaw_mode"] = self._infer_thaw_mode(cli)
            params["se_mode"] = self._infer_se_mode(cli)
            params["se_bridge_pre_norm"] = cli.get("se_pre_norm")
            params["se_tower_pre_norm"] = cli.get("se_pre_norm_tower")

        return params

    @staticmethod
    def _infer_crop_variant(crop_weights: object) -> Optional[str]:
        if not crop_weights:
            return None
        text = str(crop_weights)
        for key in ("norm_imagenet", "normalize_none", "norm_per_image"):
            if key in text:
                return key
        return None

    @staticmethod
    def _infer_loss_mode(cli: Dict[str, object]) -> Optional[str]:
        if cli.get("balanced_sampler"):
            return "balanced"
        gamma = cli.get("focal_gamma")
        try:
            if gamma is not None and float(gamma) > 0:
                return "focal"
        except Exception:
            pass
        return "none"

    @staticmethod
    def _infer_thaw_mode(cli: Dict[str, object]) -> Optional[str]:
        return "gradual" if cli.get("gradual_thaw") else "none"

    @staticmethod
    def _infer_se_mode(cli: Dict[str, object]) -> Optional[str]:
        if not cli.get("use_se"):
            return "none"
        se_where = cli.get("se_where")
        if isinstance(se_where, str) and se_where.strip():
            return se_where.strip()
        return "bridge"

    @staticmethod
    def _read_json(path: Path) -> Optional[Dict[str, object]]:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        return data

    @staticmethod
    def _read_epoch_log(run_dir: Path, fold: int) -> Optional[pd.DataFrame]:
        path = run_dir / f"fold{fold}_epoch_log.csv"
        if not path.exists():
            return None
        try:
            return pd.read_csv(path)
        except Exception:
            return None

    def _read_summary(self, run_dir: Path) -> Optional[Dict[str, object]]:
        return self._read_json(run_dir / "summary.json")

    def _read_cli_args(self, run_dir: Path) -> Optional[Dict[str, object]]:
        return self._read_json(run_dir / "cli_args.json")

    @staticmethod
    def _read_run_id(run_dir: Path, summary: Optional[Dict[str, object]]) -> str:
        if summary:
            rid = summary.get("run_id")
            if isinstance(rid, str) and rid:
                return rid
        return run_dir.name

    @staticmethod
    def _infer_mode(
        summary: Optional[Dict[str, object]], cli: Optional[Dict[str, object]]
    ) -> Optional[str]:
        for payload in (summary, cli):
            if not payload:
                continue
            eval_mode = payload.get("eval_mode")
            if isinstance(eval_mode, str):
                mode = eval_mode.strip().lower()
                if mode == "binary":
                    return "binary"
                if mode in {"multiclass", "multi", "multi-class"}:
                    return "multiclass"
            num_classes = payload.get("num_classes")
            if isinstance(num_classes, (int, float)):
                return "binary" if int(num_classes) <= 2 else "multiclass"
            class_names = payload.get("class_names")
            if isinstance(class_names, list) and class_names:
                return "binary" if len(class_names) <= 2 else "multiclass"
        return None

    @staticmethod
    def _fold_epoch_hint(
        summary: Optional[Dict[str, object]], fold: int
    ) -> Optional[int]:
        if not summary:
            return None
        fold_metrics = summary.get("fold_metrics") or []
        for entry in fold_metrics:
            if not isinstance(entry, dict):
                continue
            if entry.get("fold") == fold:
                stats = (
                    entry.get("stats") if isinstance(entry.get("stats"), dict) else {}
                )
                epoch = stats.get("epoch") or entry.get("best_epoch")
                if isinstance(epoch, (int, float)):
                    return int(epoch)
        return None

    @staticmethod
    def _extract_best_epochs(
        log_df: Optional[pd.DataFrame],
    ) -> tuple[Optional[int], Optional[int]]:
        if log_df is None or log_df.empty:
            return None, None
        best_epoch = None
        holdout_best_epoch = None
        if "best_epoch" in log_df.columns:
            try:
                best_epoch = int(log_df["best_epoch"].iloc[-1])
            except Exception:
                best_epoch = None
        if "holdout_best_epoch" in log_df.columns:
            try:
                holdout_best_epoch = int(log_df["holdout_best_epoch"].iloc[-1])
            except Exception:
                holdout_best_epoch = None
        return best_epoch, holdout_best_epoch

    @staticmethod
    def _row_for_epoch(
        log_df: Optional[pd.DataFrame], epoch: Optional[int]
    ) -> Optional[pd.Series]:
        if log_df is None or epoch is None:
            return None
        if "epoch" not in log_df.columns:
            return None
        rows = log_df[log_df["epoch"] == epoch]
        if rows.empty:
            return None
        return rows.iloc[-1]

    @staticmethod
    def _metric_from_row(row: Optional[pd.Series], key: str) -> Optional[float]:
        if row is None or key not in row:
            return None
        try:
            val = float(row[key])
        except Exception:
            return None
        if np.isnan(val):
            return None
        return val

    @staticmethod
    def _available_folds(
        run_dir: Path, summary: Optional[Dict[str, object]]
    ) -> List[int]:
        folds: List[int] = []
        if summary:
            for entry in summary.get("fold_metrics") or []:
                if not isinstance(entry, dict):
                    continue
                fold_idx = entry.get("fold")
                if isinstance(fold_idx, int):
                    folds.append(fold_idx)
        if not folds:
            pattern = re.compile(r"fold(\d+)_y_true\.npy$")
            for path in run_dir.glob("fold*_y_true.npy"):
                match = pattern.match(path.name)
                if match:
                    folds.append(int(match.group(1)))
        return sorted(set(folds))

    @staticmethod
    def _load_y_true(run_dir: Path, fold: int) -> Optional[np.ndarray]:
        path = run_dir / f"fold{fold}_y_true.npy"
        if not path.exists():
            return None
        try:
            return np.load(path)
        except Exception:
            return None

    @staticmethod
    def _collect_epoch_prob_paths(
        run_dir: Path, fold: int
    ) -> Dict[int, Dict[str, Path]]:
        pattern = re.compile(rf"fold{fold}_epoch(\d+)_probs_(\w+)\.npy$")
        epoch_paths: Dict[int, Dict[str, Path]] = {}
        for path in run_dir.glob(f"fold{fold}_epoch*_probs_*.npy"):
            match = pattern.match(path.name)
            if not match:
                continue
            epoch = int(match.group(1))
            head = match.group(2)
            epoch_paths.setdefault(epoch, {})[head] = path
        return epoch_paths

    @staticmethod
    def _collect_base_prob_paths(run_dir: Path, fold: int) -> Dict[str, Path]:
        paths: Dict[str, Path] = {}
        for head in ("fused", "img", "md"):
            candidate = run_dir / f"fold{fold}_probs_{head}.npy"
            if candidate.exists():
                paths[head] = candidate
        return paths

    @staticmethod
    def _load_probs_array(path: Path) -> Optional[np.ndarray]:
        try:
            return np.load(path)
        except Exception:
            return None

    def _load_auc_for_epoch(
        self, run_dir: Path, fold: int, epoch: int, head: str, holdout: bool
    ) -> Optional[float]:
        if epoch is None:
            return None
        tag = "holdout_" if holdout else ""
        # Prefer per-epoch ROC curves (same evaluation used for epoch_log ACC).
        epoch_dir = run_dir / f"fold{fold}_roc_curves"
        if epoch_dir.exists():
            path = epoch_dir / f"epoch{epoch}_{tag}{head}.json"
        else:
            path = None

        # Fallback to best/holdout_best exports if epoch curves missing.
        if path is None or not path.exists():
            folder = (
                "fold{}_roc_curves_holdout_best".format(fold)
                if holdout
                else "fold{}_roc_curves_best".format(fold)
            )
            target_dir = run_dir / folder
            path = target_dir / f"epoch{epoch}_{tag}{head}.json"
            if not path.exists():
                return None
        return self._extract_auc_from_json(path)

    @staticmethod
    def _extract_auc_from_json(path: Path) -> Optional[float]:
        try:
            payload = json.loads(path.read_text())
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        per_class = payload.get("per_class")
        if isinstance(per_class, dict):

            def _to_float(val):
                try:
                    f = float(val)
                except Exception:
                    return None
                if np.isnan(f):
                    return None
                return f

            # Binary fix: class 0 stored with class-1 labels against class-0 scores.
            if "0" in per_class and "1" in per_class:
                v0 = _to_float(per_class.get("0", {}).get("auc"))
                v1 = _to_float(per_class.get("1", {}).get("auc"))
                if v0 is not None and v1 is None:
                    return float(1.0 - v0)
                if v1 is not None and v0 is None:
                    return float(1.0 - v1)
                if v0 is not None and v1 is not None:
                    return float(np.mean([v0, v1]))
            vals = []
            for entry in per_class.values():
                if not isinstance(entry, dict):
                    continue
                auc_val = entry.get("auc")
                val = _to_float(auc_val)
                if val is not None:
                    vals.append(val)
            if vals:
                return float(np.mean(vals))
        macro_auc = payload.get("macro_auc")
        try:
            if macro_auc is not None and not np.isnan(float(macro_auc)):
                return float(macro_auc)
        except Exception:
            pass
        return None

    @staticmethod
    def _plot_correlation_bars(
        df: pd.DataFrame,
        path: Path,
        method: str = "pearson",
        x_metric: str = "fusion_corrections",
        metric_type: str = "acc",
        title: Optional[str] = None,
    ) -> None:
        if df.empty:
            return
        height = max(4.0, 0.28 * len(df))
        fig, ax = plt.subplots(figsize=(8.5, height))
        ax.barh(df["metric"], df["corr"], color="steelblue")
        ax.axvline(0.0, color="black", lw=1)
        label = "Spearman ρ" if method == "spearman" else "Pearson r"
        ax.set_xlabel(label)
        ax.set_title(title or f"{x_metric} vs {metric_type} metrics ({method})")
        fig.tight_layout()
        fig.savefig(path, dpi=170)
        plt.close(fig)

    @staticmethod
    def _prepare_probs(arr: np.ndarray) -> Optional[np.ndarray]:
        if arr is None:
            return None
        probs = np.asarray(arr, dtype=float)
        if probs.ndim == 1:
            probs = np.stack([1.0 - probs, probs], axis=1)
        if probs.ndim != 2:
            return None
        return probs

    @staticmethod
    def _has_all_heads(arrays: Dict[str, Optional[np.ndarray]]) -> bool:
        needed = ("fused", "img", "md")
        return all(arrays.get(head) is not None for head in needed)

    def _fusion_corrections_for_probs(
        self,
        y_true: np.ndarray,
        arrays: Dict[str, np.ndarray],
        run_id: str,
        fold: int,
        epoch: int,
    ) -> List[Dict[str, object]]:
        fused = self._prepare_probs(arrays.get("fused"))
        img = self._prepare_probs(arrays.get("img"))
        md = self._prepare_probs(arrays.get("md"))
        if fused is None or img is None or md is None:
            return []
        if not (len(fused) == len(img) == len(md) == len(y_true)):
            return []

        fused_pred = fused.argmax(axis=1)
        img_pred = img.argmax(axis=1)
        md_pred = md.argmax(axis=1)

        fused_conf = np.take_along_axis(fused, fused_pred[:, None], axis=1).squeeze(1)
        img_conf = np.take_along_axis(img, img_pred[:, None], axis=1).squeeze(1)
        md_conf = np.take_along_axis(md, md_pred[:, None], axis=1).squeeze(1)

        mask = (fused_pred == y_true) & (img_pred != y_true) & (md_pred != y_true)
        indices = np.nonzero(mask)[0]

        events: List[Dict[str, object]] = []
        for idx in indices:
            events.append(
                {
                    "run_id": run_id,
                    "fold": fold,
                    "epoch": epoch,
                    "index": int(idx),
                    "y_true": int(y_true[idx]),
                    "pred_fused": int(fused_pred[idx]),
                    "pred_img": int(img_pred[idx]),
                    "pred_md": int(md_pred[idx]),
                    "conf_fused": float(fused_conf[idx]),
                    "conf_img": float(img_conf[idx]),
                    "conf_md": float(md_conf[idx]),
                }
            )
        return events

    def _fusion_errors_for_probs(
        self,
        y_true: np.ndarray,
        arrays: Dict[str, np.ndarray],
        run_id: str,
        fold: int,
        epoch: int,
    ) -> List[Dict[str, object]]:
        fused = self._prepare_probs(arrays.get("fused"))
        img = self._prepare_probs(arrays.get("img"))
        md = self._prepare_probs(arrays.get("md"))
        if fused is None or img is None or md is None:
            return []
        if not (len(fused) == len(img) == len(md) == len(y_true)):
            return []

        fused_pred = fused.argmax(axis=1)
        img_pred = img.argmax(axis=1)
        md_pred = md.argmax(axis=1)

        fused_conf = np.take_along_axis(fused, fused_pred[:, None], axis=1).squeeze(1)
        img_conf = np.take_along_axis(img, img_pred[:, None], axis=1).squeeze(1)
        md_conf = np.take_along_axis(md, md_pred[:, None], axis=1).squeeze(1)

        mask = (fused_pred != y_true) & (img_pred == y_true) & (md_pred == y_true)
        indices = np.nonzero(mask)[0]

        events: List[Dict[str, object]] = []
        for idx in indices:
            events.append(
                {
                    "run_id": run_id,
                    "fold": fold,
                    "epoch": epoch,
                    "index": int(idx),
                    "y_true": int(y_true[idx]),
                    "pred_fused": int(fused_pred[idx]),
                    "pred_img": int(img_pred[idx]),
                    "pred_md": int(md_pred[idx]),
                    "conf_fused": float(fused_conf[idx]),
                    "conf_img": float(img_conf[idx]),
                    "conf_md": float(md_conf[idx]),
                }
            )
        return events


if __name__ == "__main__":
    analysis = derived_analysis(
        Path("analysis_data/grid_search"), classification_mode="binary"
    )
    
    df = analysis.identify_fusion_corrections(existing=True)
    #    analysis.write_fusion_corrections()
    #    analysis.write_fusion_errors()
    print(f"Fusion correction events: {len(df)}")
    analysis.primary_metrics = analysis.populate_primary_metrics(existing=True)
    #    analysis.write_primary_metrics()
    # corr_acc = analysis.fusion_corrections_correlation(
    #     method="spearman", metric_type="acc"
    # )
    # corr_auc = analysis.fusion_corrections_correlation(
    #     method="spearman", metric_type="auc"
    # )
    # print(corr_acc)
    # summary = analysis.plot_fusion_perf_summary(
    #     corr_acc,
    #     corr_auc,
    #     method="spearman",
    #     x_metric="fusion_corrections_per_opportunity",
    # )
    analysis.primary_metrics 
    df = analysis.param_performance_correlations(method="spearman")
    df.columns
    # analysis.plot_param_perf_corr_panels(df)
    out = analysis.se_mode_effects_summary(metric="auc")
    out2 = analysis.se_mode_effects_summary(metric="acc")
 
    out3 = analysis.se_mode_effects_summary(metric="auc", head="fused", prefer_holdout=True, top_n=25)
    out4 = analysis.se_mode_effects_summary(metric="acc", head="fused", prefer_holdout=True, top_n=25)
    print(out["summary"])
    print(out2["summary"])
    print(out3["summary"])
    print(out4["summary"])
    print(out["pairwise"])
    print(out3["pairwise"])
# analysis.fusion_corrections
# analysis.plot_fusion_corrections_errors()
# analysis.plot_conf_delta_boxplot()
