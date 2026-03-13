#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.grid_search_analytics.derived_analysis import derived_analysis


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate derived grid-search analytics artifacts (fusion/error + param-performance)."
    )
    ap.add_argument("--analysis-dir", default="analysis_data/grid_search")
    ap.add_argument("--mode", choices=["binary", "multiclass"], default="multiclass")
    ap.add_argument("--method", choices=["pearson", "spearman"], default="spearman")
    ap.add_argument(
        "--x-metric",
        choices=["fusion_corrections", "fusion_corrections_per_opportunity"],
        default="fusion_corrections_per_opportunity",
        help="Fusion-correlation x-axis metric for summary bar plot.",
    )
    ap.add_argument(
        "--cat-method",
        choices=["eta", "anova", "kruskal"],
        default="kruskal",
        help="Categorical-test method for param-performance correlations.",
    )
    ap.add_argument("--top-n", type=int, default=None, help="Optional cap for per-run plots.")
    ap.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute from run artifacts instead of preferring cached CSVs.",
    )
    ap.add_argument(
        "--deep-scan",
        action="store_true",
        help="Scan nested directories instead of direct children only.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    analysis = derived_analysis(
        Path(args.analysis_dir),
        classification_mode=args.mode,
    )

    shallow = not args.deep_scan
    existing = not args.recompute

    analysis.identify_fusion_corrections(shallow=shallow, existing=existing)
    analysis.populate_primary_metrics(shallow=shallow, existing=existing)

    analysis.write_fusion_corrections()
    analysis.write_fusion_errors()
    analysis.write_primary_metrics()

    analysis.plot_fusion_corrections_errors(
        shallow=shallow, existing=existing, top_n=args.top_n
    )
    analysis.plot_conf_delta_boxplot(
        shallow=shallow, existing=existing, top_n=args.top_n
    )

    corr_df = analysis.param_performance_correlations(
        shallow=shallow,
        existing=existing,
        method=args.method,
        cat_method=args.cat_method,
    )
    analysis.plot_param_perf_corr_panels(corr_df)

    corr_acc = analysis.fusion_corrections_correlation(
        method=args.method, metric_type="acc"
    )
    corr_auc = analysis.fusion_corrections_correlation(
        method=args.method, metric_type="auc"
    )
    try:
        analysis.plot_fusion_perf_summary(
            corr_acc,
            corr_auc,
            method=args.method,
            x_metric=args.x_metric,
        )
    except RuntimeError:
        analysis.plot_fusion_perf_summary(
            corr_acc,
            corr_auc,
            method=args.method,
            x_metric="fusion_corrections",
        )

    print(f"Done. Outputs written under: {Path(args.analysis_dir) / 'plots'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
