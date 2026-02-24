from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Optional


DEFAULT_OPTIONAL_EPOCH_COLS = [
    "pct_fused",
    "pct_img",
    "pct_md",
    "phase",
    "se_mean",
    "se_std",
    "se_pct_lt_0.2",
    "se_pct_gt_0.8",
    "holdout_loss",
    "holdout_acc_fused",
    "holdout_acc_img",
    "holdout_acc_md",
    "holdout_auc_fused",
    "holdout_auc_img",
    "holdout_auc_md",
    "best_monitor",
    "best_so_far",
    "best_epoch",
    "early_best_so_far",
    "early_bad_epochs",
    "early_improved",
    "early_monitor",
    "holdout_best_monitor",
    "holdout_best_so_far",
    "holdout_best_epoch",
]


class HypertowerLogger:
    """
    Shared logging utility for V2 tower workflows.
    - train.log line logging
    - epoch_log.csv row logging with stable header
    - lightweight JSON/array artifact helpers
    """

    def __init__(
        self,
        *,
        run_dir: Path,
        train_log_path: Optional[Path] = None,
        epoch_log_path: Optional[Path] = None,
        logger_name: Optional[str] = None,
    ) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.train_log_path = Path(train_log_path) if train_log_path else (self.run_dir / "train.log")
        self.epoch_log_path = Path(epoch_log_path) if epoch_log_path else (self.run_dir / "epoch_log.csv")

        self._logger_name = logger_name or f"hypertower.{id(self)}"
        self.logger = logging.getLogger(self._logger_name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        fh = logging.FileHandler(str(self.train_log_path))
        fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        self.logger.addHandler(fh)
        self.logger.propagate = False

        self._epoch_log_fp = None
        self._epoch_log_writer = None
        self._epoch_log_fields: list[str] | None = None

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)

    def write_epoch_row(
        self,
        row: dict,
        *,
        path: str | Path | None = None,
        optional_cols: Optional[list[str]] = None,
    ) -> None:
        optional = optional_cols if optional_cols is not None else DEFAULT_OPTIONAL_EPOCH_COLS
        if self._epoch_log_writer is None:
            fieldnames = list(dict.fromkeys([*row.keys(), *optional]))
            target_path = Path(path) if path is not None else self.epoch_log_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            self._epoch_log_fp = open(target_path, "w", newline="", encoding="utf-8")
            self._epoch_log_writer = csv.DictWriter(self._epoch_log_fp, fieldnames=fieldnames)
            self._epoch_log_writer.writeheader()
            self._epoch_log_fields = fieldnames

        assert self._epoch_log_fields is not None
        assert self._epoch_log_writer is not None
        assert self._epoch_log_fp is not None
        for key in self._epoch_log_fields:
            row.setdefault(key, None)
        self._epoch_log_writer.writerow({k: row.get(k) for k in self._epoch_log_fields})
        self._epoch_log_fp.flush()

    def write_json(self, path: str | Path, payload: dict) -> None:
        target = Path(path)
        if not target.is_absolute():
            target = self.run_dir / target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def close(self) -> None:
        if self._epoch_log_fp is not None:
            try:
                self._epoch_log_fp.close()
            except Exception:
                pass
            self._epoch_log_fp = None
            self._epoch_log_writer = None
            self._epoch_log_fields = None
        for handler in list(self.logger.handlers):
            try:
                handler.close()
            except Exception:
                pass
            self.logger.removeHandler(handler)
