#!/usr/bin/env python3
"""
Shared helpers for building grid search analytics.

The class below will gradually accumulate reusable utilities for working with
grid search outputs (summary.json, cli_args.json, etc).
"""

from __future__ import annotations

import json
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import numpy as np

DEFAULT_EXCLUDE_KEYS = {
    "run_id",
    "fold_metrics",
    "best_metric",
    "best_metric_mode",
    "best_metric_mean",
    "best_metric_std",
    "eval_mode",
    "n_splits",
    "num_classes",
}


class GridSearchAnalytics:
    """Utility wrapper for inspecting grid search result directories."""

    def __init__(self,
                 analysis_dir: Path | str,
                 exclude_keys: Optional[Iterable[str]] = None) -> None:
        self.analysis_dir = Path(analysis_dir)
        if not self.analysis_dir.exists():
            raise FileNotFoundError(f"analysis_dir does not exist: {self.analysis_dir}")
        self.exclude_keys = set(exclude_keys or DEFAULT_EXCLUDE_KEYS)

    def iter_run_dirs(self, shallow: bool = True) -> Iterator[Path]:
        """
        Yield run directories containing grid search artifacts.

        Shallow iteration only walks direct children. Deep iteration scans the
        entire subtree.
        """
        candidates: Iterable[Path]
        if shallow:
            candidates = (p for p in sorted(self.analysis_dir.iterdir()) if p.is_dir())
        else:
            candidates = (p for p in self.analysis_dir.rglob("*") if p.is_dir())
        for run_dir in candidates:
            summary = run_dir / "summary.json"
            cli = run_dir / "cli_args.json"
            if summary.exists() or cli.exists():
                yield run_dir

    def read_summary(self, run_dir: Path) -> Optional[Dict[str, object]]:
        """Load summary.json for a run directory."""
        return self._read_json(run_dir / "summary.json")

    def read_cli_args(self, run_dir: Path) -> Optional[Dict[str, object]]:
        """Load cli_args.json for a run directory."""
        return self._read_json(run_dir / "cli_args.json")

    def read_run_id(self,
                    run_dir: Path,
                    summary: Optional[Dict[str, object]]) -> str:
        if summary:
            rid = summary.get("run_id")
            if isinstance(rid, str) and rid:
                return rid
        return run_dir.name

    def task_from_summary(self, summary: Optional[Dict[str, object]]) -> Optional[str]:
        """Infer task (binary vs multiclass) from a summary payload."""
        if not summary:
            return None
        eval_mode = summary.get("eval_mode")
        if isinstance(eval_mode, str):
            mode = eval_mode.strip().lower()
            if mode == "binary":
                return "binary"
            if mode in {"multiclass", "multi", "multi-class"}:
                return "multiclass"
        num_classes = summary.get("num_classes")
        if isinstance(num_classes, (int, float)):
            return "binary" if int(num_classes) <= 2 else "multiclass"
        return None

    def flatten_config(self,
                       data: Dict[str, object],
                       prefix: str = "",
                       exclude_keys: Optional[Iterable[str]] = None) -> Dict[str, object]:
        """Flatten nested CLI args or config dictionaries for analysis."""
        out: Dict[str, object] = {}
        excludes = set(exclude_keys or self.exclude_keys)
        for key, value in data.items():
            if key in excludes or key.startswith("best_"):
                continue
            full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
            if isinstance(value, dict):
                out.update(self.flatten_config(value, full_key, exclude_keys=excludes))
                continue
            if isinstance(value, list):
                continue
            out[full_key] = value
        return out

    def fusion_correction_events(self,
                                 output_csv: Path | str | None = None,
                                 shallow: bool = True) -> Path:
        """
        Build a table of cases where the fused head is correct while both towers
        are wrong. Rows are written to CSV for downstream analysis.
        """
        output_path = Path(output_csv) if output_csv else Path("analysis_data/grid_search_analytics/fusion_corrections.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rows: List[Dict[str, object]] = []
        for run_dir in self.iter_run_dirs(shallow=shallow):
            summary = self.read_summary(run_dir)
            run_id = self.read_run_id(run_dir, summary)
            folds = self._available_folds(run_dir, summary)
            for fold in folds:
                y_true = self._load_y_true(run_dir, fold)
                if y_true is None:
                    continue
                epoch_prob_paths = self._collect_epoch_prob_paths(run_dir, fold)
                if not epoch_prob_paths:
                    # Per-epoch dumps were not found; fall back to the saved fold-level probabilities.
                    base_paths = self._collect_base_prob_paths(run_dir, fold)
                    if base_paths:
                        epoch_hint = self._fold_epoch_hint(summary, fold)
                        epoch_prob_paths = {epoch_hint if epoch_hint is not None else 0: base_paths}
                for epoch, paths in epoch_prob_paths.items():
                    arrays = {head: self._load_probs_array(path) for head, path in paths.items()}
                    if not self._has_all_heads(arrays):
                        continue
                    events = self._fusion_corrections_for_probs(y_true, arrays, run_id, fold, epoch)
                    rows.extend(events)

        if rows:
            fieldnames = [
                "run_id",
                "fold",
                "epoch",
                "index",
                "y_true",
                "pred_fused",
                "pred_img",
                "pred_md",
                "conf_fused",
                "conf_img",
                "conf_md",
            ]
            with output_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        else:
            output_path.write_text("")
        return output_path

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
    def _fold_epoch_hint(summary: Optional[Dict[str, object]], fold: int) -> Optional[int]:
        if not summary:
            return None
        fold_metrics = summary.get("fold_metrics") or []
        for entry in fold_metrics:
            if not isinstance(entry, dict):
                continue
            if entry.get("fold") == fold:
                stats = entry.get("stats") if isinstance(entry.get("stats"), dict) else {}
                epoch = stats.get("epoch") or entry.get("best_epoch")
                if isinstance(epoch, (int, float)):
                    return int(epoch)
        return None

    @staticmethod
    def _available_folds(run_dir: Path, summary: Optional[Dict[str, object]]) -> List[int]:
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
    def _collect_epoch_prob_paths(run_dir: Path, fold: int) -> Dict[int, Dict[str, Path]]:
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

    def _fusion_corrections_for_probs(self,
                                      y_true: np.ndarray,
                                      arrays: Dict[str, np.ndarray],
                                      run_id: str,
                                      fold: int,
                                      epoch: int) -> List[Dict[str, object]]:
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
            events.append({
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
            })
        return events


if __name__ == "__main__":
    analytics = GridSearchAnalytics(Path("analysis_data/grid_search"))
    output = analytics.fusion_correction_events()
    print(f"Fusion correction events written to {output}")
