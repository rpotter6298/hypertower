#!/usr/bin/env python3
"""Plot holdout ROC curves for a specific grid search run."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


# ---------------------------
# Config (edit in IDE)
# ---------------------------
RUN_ID = "20251129-0312"
ANALYSIS_ROOT = Path("analysis_data/grid_search")
HEADS = ["fused", "image", "metadata"]
OUTPUT_SUBDIR = Path("plots/holdout_rocs")
BEST_OUTPUT_SUBDIR = Path("plots/best_rocs")
POSITIVE_CLASS = 1
DEBUG = True
USE_JSON_ROC = True
USE_HOLDOUT_PROBS = True
ALLOW_FALLBACK_TO_VALIDATION = False
PLOT_VALIDATION_FROM_HOLDOUT_EPOCH = True
PLOT_BEST_EPOCH = True
PLOT_HOLDOUT_FROM_BEST_EPOCH = True
PLOT_ALL_CLASSES = True
FORCE_PROBS_FOR_BEST_BINARY = True
FORCE_PROBS_FOR_HOLDOUT_BINARY = False


HEAD_FILE_KEYS = {
    "fused": "fused",
    "image": "img",
    "metadata": "md",
}

JSON_DIR_NAMES = [
    "roc_curves_holdout_best",
    "roc_curves",
]


def _epoch_from_name(path: Path) -> int:
    m = re.search(r"epoch(\d+)", path.name)
    return int(m.group(1)) if m else -1


def _load_json(path: Path) -> Dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _infer_run_info(run_dir: Path) -> Tuple[str | None, int | None, List[str] | None]:
    cli = _load_json(run_dir / "cli_args.json")
    summary = _load_json(run_dir / "summary.json")
    payloads = [cli, summary]
    eval_mode = None
    num_classes = None
    class_names = None
    for payload in payloads:
        if not payload:
            continue
        if eval_mode is None:
            em = payload.get("eval_mode")
            if isinstance(em, str):
                eval_mode = em.strip().lower()
        if num_classes is None:
            nc = payload.get("num_classes")
            if isinstance(nc, (int, float)):
                num_classes = int(nc)
        if class_names is None:
            cn = payload.get("class_names")
            if isinstance(cn, list) and cn:
                class_names = [str(x) for x in cn]
    if num_classes is None and eval_mode:
        num_classes = 2 if eval_mode == "binary" else 3
    return eval_mode, num_classes, class_names


def _collect_holdout_json_files(run_dir: Path, head: str) -> Dict[int, Path]:
    fold_files: Dict[int, Path] = {}
    # Fold-scoped folders
    for folder_name in JSON_DIR_NAMES:
        for fold_dir in run_dir.glob(f"fold*_{folder_name}"):
            fold_match = re.search(r"fold(\d+)_", fold_dir.name)
            if not fold_match:
                continue
            fold_idx = int(fold_match.group(1))
            candidates = list(fold_dir.glob(f"epoch*_holdout_{head}.json"))
            if not candidates:
                candidates = list(fold_dir.glob(f"epoch*_{head}.json"))
            if candidates:
                candidates.sort(key=_epoch_from_name)
                fold_files[fold_idx] = candidates[-1]
        if fold_files:
            return fold_files
    # Fallback: unscoped roc_curves in run_dir (single-fold or in-progress)
    for folder_name in JSON_DIR_NAMES:
        base_dir = run_dir / folder_name
        if not base_dir.exists():
            continue
        candidates = list(base_dir.glob(f"epoch*_holdout_{head}.json"))
        if not candidates:
            candidates = list(base_dir.glob(f"epoch*_{head}.json"))
        if candidates:
            candidates.sort(key=_epoch_from_name)
            fold_files[0] = candidates[-1]
            break
    return fold_files


def _collect_validation_json_files(
    holdout_files: Dict[int, Path], head: str
) -> Dict[int, Path]:
    validation_files: Dict[int, Path] = {}
    for fold_idx, holdout_path in holdout_files.items():
        epoch = _epoch_from_name(holdout_path)
        if epoch < 0:
            continue
        candidate = holdout_path.parent / f"epoch{epoch}_{head}.json"
        if candidate.exists():
            validation_files[fold_idx] = candidate
            continue
        # Fallback: try the same epoch under roc_curves (if holdout_best folder omitted it).
        for folder_name in JSON_DIR_NAMES:
            alt_dir = holdout_path.parent.parent / f"fold{fold_idx}_{folder_name}"
            alt_candidate = alt_dir / f"epoch{epoch}_{head}.json"
            if alt_candidate.exists():
                validation_files[fold_idx] = alt_candidate
                break
    return validation_files


def _collect_holdout_from_validation_files(
    validation_files: Dict[int, Path], head: str
) -> Dict[int, Path]:
    holdout_files: Dict[int, Path] = {}
    for fold_idx, val_path in validation_files.items():
        epoch = _epoch_from_name(val_path)
        if epoch < 0:
            continue
        candidate = val_path.parent / f"epoch{epoch}_holdout_{head}.json"
        if candidate.exists():
            holdout_files[fold_idx] = candidate
            continue
        for folder_name in ("roc_curves", "roc_curves_holdout_best"):
            alt_dir = val_path.parent.parent / f"fold{fold_idx}_{folder_name}"
            alt_candidate = alt_dir / f"epoch{epoch}_holdout_{head}.json"
            if alt_candidate.exists():
                holdout_files[fold_idx] = alt_candidate
                break
    return holdout_files


def _collect_best_json_files(run_dir: Path, head: str) -> Dict[int, Path]:
    fold_files: Dict[int, Path] = {}
    for fold_dir in run_dir.glob("fold*_roc_curves_best"):
        fold_match = re.search(r"fold(\d+)_", fold_dir.name)
        if not fold_match:
            continue
        fold_idx = int(fold_match.group(1))
        candidates = list(fold_dir.glob(f"epoch*_{head}.json"))
        if candidates:
            candidates.sort(key=_epoch_from_name)
            fold_files[fold_idx] = candidates[-1]
    return fold_files


def _extract_curves(data: Dict) -> Dict[str, Tuple[List[float], List[float], float]]:
    curves: Dict[str, Tuple[List[float], List[float], float]] = {}
    per_class = data.get("per_class") if isinstance(data, dict) else None
    if not isinstance(per_class, dict):
        return curves
    for cls, entry in per_class.items():
        if not isinstance(entry, dict):
            continue
        fpr = entry.get("fpr")
        tpr = entry.get("tpr")
        auc_val = entry.get("auc")
        if not isinstance(fpr, list) or not isinstance(tpr, list):
            continue
        try:
            auc_f = float(auc_val) if auc_val is not None else float("nan")
        except Exception:
            auc_f = float("nan")
        curves[str(cls)] = (fpr, tpr, auc_f)
    return curves


def _derive_positive_from_class0(
    curves_by_class: Dict[str, List[Tuple[int, List[float], List[float], float]]],
    positive_class: int,
) -> None:
    zero_key = "0"
    if zero_key not in curves_by_class:
        return
    derived = []
    for fold_idx, fpr0, tpr0, auc0 in curves_by_class.get(zero_key, []):
        # The JSON for binary currently stores class-1 labels with class-0 scores,
        # so invert the curve to recover the true class-1 ROC.
        fpr1 = [1.0 - float(x) for x in fpr0]
        tpr1 = [1.0 - float(x) for x in tpr0]
        # Ensure increasing FPR for plotting.
        if len(fpr1) > 1 and fpr1[0] > fpr1[-1]:
            fpr1 = list(reversed(fpr1))
            tpr1 = list(reversed(tpr1))
        auc1 = 1.0 - auc0 if auc0 == auc0 else auc0
        derived.append((fold_idx, fpr1, tpr1, auc1))
    curves_by_class[str(positive_class)] = derived


def _needs_positive_derivation(
    curves_by_class: Dict[str, List[Tuple[int, List[float], List[float], float]]],
    positive_class: int,
) -> bool:
    curves = curves_by_class.get(str(positive_class))
    if not curves:
        return True
    for _, fpr, tpr, auc_val in curves:
        if auc_val == auc_val and len(fpr) > 2 and len(tpr) > 2:
            return False
    return True


def _collect_prob_files(run_dir: Path, suffix: str) -> Dict[int, Dict[str, Path]]:
    files: Dict[int, Dict[str, Path]] = {}
    for y_file in run_dir.glob(f"fold*_y_true{suffix}.npy"):
        fold_str = y_file.stem.split("_")[0].replace("fold", "")
        try:
            fold_idx = int(fold_str)
        except ValueError:
            continue
        files.setdefault(fold_idx, {})["y_true"] = y_file
    for head, key in HEAD_FILE_KEYS.items():
        for p_file in run_dir.glob(f"fold*_probs_{key}{suffix}.npy"):
            fold_str = p_file.stem.split("_")[0].replace("fold", "")
            try:
                fold_idx = int(fold_str)
            except ValueError:
                continue
            files.setdefault(fold_idx, {})[head] = p_file
    return files


def _load_array(path: Path) -> np.ndarray | None:
    try:
        return np.load(path)
    except Exception:
        return None


def _compute_binary_curve(
    y_true: np.ndarray, probs: np.ndarray, positive_class: int
) -> Tuple[List[float], List[float], float] | None:
    if probs.ndim == 1:
        scores = probs
    elif probs.ndim == 2 and probs.shape[1] > positive_class:
        scores = probs[:, positive_class]
    else:
        return None
    y_bin = (y_true == positive_class).astype(int)
    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return None
    fpr, tpr, _ = roc_curve(y_bin, scores)
    auc_val = float(auc(fpr, tpr))
    return fpr.tolist(), tpr.tolist(), auc_val


def _compute_multiclass_curves(
    y_true: np.ndarray, probs: np.ndarray
) -> Dict[str, Tuple[List[float], List[float], float]]:
    curves: Dict[str, Tuple[List[float], List[float], float]] = {}
    if probs.ndim != 2:
        return curves
    num_classes = probs.shape[1]
    for cls in range(num_classes):
        y_bin = (y_true == cls).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        fpr, tpr, _ = roc_curve(y_bin, probs[:, cls])
        curves[str(cls)] = (fpr.tolist(), tpr.tolist(), float(auc(fpr, tpr)))
    return curves


def _plot_overlays(
    curves_by_fold: List[Tuple[int, List[float], List[float], float]],
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for fold_idx, fpr, tpr, auc_val in curves_by_fold:
        label = (
            f"fold{fold_idx} AUC={auc_val:.3f}"
            if auc_val == auc_val
            else f"fold{fold_idx}"
        )
        ax.plot(fpr, tpr, lw=1.4, label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize="small")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> None:
    run_dir = ANALYSIS_ROOT / RUN_ID
    if not run_dir.exists():
        raise SystemExit(f"Run not found: {run_dir}")
    eval_mode, num_classes, class_names = _infer_run_info(run_dir)
    is_binary = eval_mode == "binary" or num_classes == 2

    def _plot_set(
        label: str,
        head: str,
        json_files: Dict[int, Path],
        out_dir: Path,
        paired_files: Dict[int, Path] | None,
        paired_suffix: str,
        class_names: List[str] | None,
    ) -> None:
        if not json_files:
            return
        curves_by_class: Dict[
            str, List[Tuple[int, List[float], List[float], float]]
        ] = {}
        paired_curves_by_class: Dict[
            str, List[Tuple[int, List[float], List[float], float]]
        ] = {}
        for fold_idx, path in sorted(json_files.items()):
            if DEBUG:
                print(f"[debug] {label} head={head} fold={fold_idx} json={path}")
            data = _load_json(path)
            curves = _extract_curves(data)
            for cls, (fpr, tpr, auc_val) in curves.items():
                curves_by_class.setdefault(cls, []).append(
                    (fold_idx, fpr, tpr, auc_val)
                )
            if paired_files:
                p_path = paired_files.get(fold_idx)
                if p_path is not None:
                    if DEBUG:
                        print(
                            f"[debug] {label} head={head} fold={fold_idx} paired_json={p_path}"
                        )
                    p_data = _load_json(p_path)
                    p_curves = _extract_curves(p_data)
                    for cls, (fpr, tpr, auc_val) in p_curves.items():
                        paired_curves_by_class.setdefault(cls, []).append(
                            (fold_idx, fpr, tpr, auc_val)
                        )
        if _needs_positive_derivation(curves_by_class, POSITIVE_CLASS):
            _derive_positive_from_class0(curves_by_class, POSITIVE_CLASS)
        if paired_curves_by_class and _needs_positive_derivation(
            paired_curves_by_class, POSITIVE_CLASS
        ):
            _derive_positive_from_class0(paired_curves_by_class, POSITIVE_CLASS)

        if PLOT_ALL_CLASSES:
            classes = list(curves_by_class.keys())
        else:
            classes = (
                [str(POSITIVE_CLASS)]
                if str(POSITIVE_CLASS) in curves_by_class
                else list(curves_by_class.keys())
            )
        if not classes:
            return

        for cls in classes:
            fold_curves = curves_by_class.get(cls, [])
            if not fold_curves:
                continue
            class_label = cls
            if class_names is not None:
                try:
                    idx = int(cls)
                    if 0 <= idx < len(class_names):
                        class_label = f"{cls} ({class_names[idx]})"
                except Exception:
                    pass
            title = f"{RUN_ID} {label} ROC — head={head} class={class_label}"
            out_path = out_dir / f"{label}_{head}_class{cls}.png"
            _plot_overlays(fold_curves, title, out_path)
            print(f"[ok] {out_path}")

            if paired_curves_by_class:
                p_curves = paired_curves_by_class.get(cls, [])
                if p_curves:
                    p_title = f"{RUN_ID} {label} {paired_suffix} ROC — head={head} class={class_label}"
                    p_path = out_dir / f"{label}_{head}_class{cls}_{paired_suffix}.png"
                    _plot_overlays(p_curves, p_title, p_path)
                    print(f"[ok] {p_path}")

    def _plot_from_probs(
        label: str,
        head: str,
        out_dir: Path,
        suffix: str,
        class_names: List[str] | None,
    ) -> None:
        curves_by_class: Dict[
            str, List[Tuple[int, List[float], List[float], float]]
        ] = {}
        files = _collect_prob_files(run_dir, suffix)
        if not files and suffix and ALLOW_FALLBACK_TO_VALIDATION:
            files = _collect_prob_files(run_dir, "")
            if files:
                print(
                    "[warn] Holdout probability dumps not found; using validation probabilities instead."
                )
        if not files:
            return
        for fold_idx in sorted(files.keys()):
            fold_files = files[fold_idx]
            y_path = fold_files.get("y_true")
            p_path = fold_files.get(head)
            if y_path is None or p_path is None:
                continue
            y_true = _load_array(y_path)
            probs = _load_array(p_path)
            if y_true is None or probs is None:
                continue
            curves = _compute_multiclass_curves(y_true, probs)
            for cls, payload in curves.items():
                curves_by_class.setdefault(cls, []).append((fold_idx, *payload))
        if not curves_by_class:
            return
        classes = list(curves_by_class.keys())
        for cls in classes:
            fold_curves = curves_by_class.get(cls, [])
            if not fold_curves:
                continue
            class_label = cls
            if class_names is not None:
                try:
                    idx = int(cls)
                    if 0 <= idx < len(class_names):
                        class_label = f"{cls} ({class_names[idx]})"
                except Exception:
                    pass
            title = f"{RUN_ID} {label} ROC — head={head} class={class_label}"
            out_path = out_dir / f"{label}_{head}_class{cls}.png"
            _plot_overlays(fold_curves, title, out_path)
            print(f"[ok] {out_path}")

    any_holdout_json = False
    if USE_JSON_ROC:
        out_dir = run_dir / OUTPUT_SUBDIR
        for head in HEADS:
            files = _collect_holdout_json_files(run_dir, head)
            if files:
                any_holdout_json = True
                paired = (
                    _collect_validation_json_files(files, head)
                    if PLOT_VALIDATION_FROM_HOLDOUT_EPOCH
                    else None
                )
                if is_binary and FORCE_PROBS_FOR_HOLDOUT_BINARY:
                    _plot_from_probs("holdout", head, out_dir, "_holdout", class_names)
                else:
                    _plot_set(
                        "holdout",
                        head,
                        files,
                        out_dir,
                        paired,
                        "validation",
                        class_names,
                    )
        if not any_holdout_json and DEBUG:
            print("[debug] no JSON ROC files found; falling back to probs")

        if PLOT_BEST_EPOCH:
            best_out_dir = run_dir / BEST_OUTPUT_SUBDIR
            for head in HEADS:
                if is_binary and FORCE_PROBS_FOR_BEST_BINARY:
                    _plot_from_probs("best", head, best_out_dir, "", class_names)
                    continue
                best_files = _collect_best_json_files(run_dir, head)
                if best_files:
                    paired = (
                        _collect_holdout_from_validation_files(best_files, head)
                        if PLOT_HOLDOUT_FROM_BEST_EPOCH
                        else None
                    )
                    _plot_set(
                        "best",
                        head,
                        best_files,
                        best_out_dir,
                        paired,
                        "holdout",
                        class_names,
                    )

    # Fallback to probs for holdout plots if JSON wasn't found.
    if USE_JSON_ROC and any_holdout_json:
        return

    out_dir = run_dir / OUTPUT_SUBDIR
    for head in HEADS:
        curves_by_class: Dict[
            str, List[Tuple[int, List[float], List[float], float]]
        ] = {}
        suffix = "_holdout" if USE_HOLDOUT_PROBS else ""
        files = _collect_prob_files(run_dir, suffix)
        if not files and USE_HOLDOUT_PROBS and ALLOW_FALLBACK_TO_VALIDATION:
            suffix = ""
            files = _collect_prob_files(run_dir, suffix)
            if files:
                print(
                    "[warn] Holdout probability dumps not found; using validation probabilities instead."
                )
        if not files:
            raise SystemExit(
                "No saved probability dumps found. If you want holdout ROC curves, "
                "run scripts/rebuild_run_best_plots.py with --use-holdout --overwrite "
                "to generate fold*_y_true_holdout.npy and fold*_probs_*_holdout.npy files."
            )
        for fold_idx in sorted(files.keys()):
            fold_files = files[fold_idx]
            y_path = fold_files.get("y_true")
            p_path = fold_files.get(head)
            if y_path is None or p_path is None:
                continue
            y_true = _load_array(y_path)
            probs = _load_array(p_path)
            if y_true is None or probs is None:
                continue
            curves = _compute_multiclass_curves(y_true, probs)
            for cls, payload in curves.items():
                if payload is None:
                    continue
                fpr, tpr, auc_val = payload
                curves_by_class.setdefault(cls, []).append(
                    (fold_idx, fpr, tpr, auc_val)
                )
        if not curves_by_class:
            continue
        classes = sorted(curves_by_class.keys(), key=lambda x: (float(x), str(x)))
        for cls in classes:
            fold_curves = curves_by_class.get(cls, [])
            if not fold_curves:
                continue
            class_label = cls
            if class_names is not None:
                try:
                    idx = int(cls)
                    if 0 <= idx < len(class_names):
                        class_label = f"{cls} ({class_names[idx]})"
                except Exception:
                    pass
            title = f"{RUN_ID} holdout ROC — head={head} class={class_label}"
            out_path = out_dir / f"holdout_{head}_class{cls}.png"
            _plot_overlays(fold_curves, title, out_path)
            print(f"[ok] {out_path}")


if __name__ == "__main__":
    main()
