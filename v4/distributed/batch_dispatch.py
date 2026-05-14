"""
v4 batch dispatch — merge a batch.json with a base config and submit jobs.

Each batch entry specifies only what differs from the base config. Each entry
is dispatched as N_REPS independent jobs (full 5-fold CV per rep, different seeds),
writing to {run_name}/rep00/, rep01/, etc.

Usage:
    python -m v4.distributed.batch_dispatch \\
        --server http://apollo:8765 \\
        --token <secret> \\
        --config v4/configs/ensemble_fused.json \\
        --batch v4/scripts/experiments/my_batch.json \\
        [--reps 10] \\
        [--seed-start 1234] \\
        [--seed-step 100] \\
        [--fold-seed-start 100] \\
        [--fold-seed-step 100] \\
        [--output-root v4/results] \\
        [--priority 0] \\
        [--dry-run]

batch.json format:
    [
      {
        "run_name":  "experiments/lr_sweep/lr1e3",   // required
        "overrides": {                               // optional — deep-merged into base
          "training": { "lr": 0.001 }
        },
        "stage_overrides": {                         // optional — patched by stage name
          "nt": { "epochs": 40 }
        },
        "reps":     10,                              // optional — overrides --reps
        "priority": 0                                // optional
      }
    ]

Global flags (can also be set via env vars):
    --server  HT_SERVER
    --token   HT_TOKEN
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

# ──────────────────────────────────────────────────────────────
# Config helpers
# ──────────────────────────────────────────────────────────────

def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into a copy of base.

    - Dicts are merged recursively.
    - All other types (scalars, lists) are replaced by the override value.
    """
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def apply_stage_overrides(stages: list[dict], stage_overrides: dict) -> list[dict]:
    """Patch individual stages by name without replacing the entire list."""
    stages = copy.deepcopy(stages)
    for stage in stages:
        name = stage.get("name")
        if name in stage_overrides:
            merged = deep_merge(stage, stage_overrides[name])
            stage.clear()
            stage.update(merged)
    return stages


def build_config(base_cfg: dict, entry: dict, rep: int, seed: int, fold_seed: int,
                 output_root: str) -> dict:
    """Produce the final merged config for one rep of one batch entry."""
    cfg = copy.deepcopy(base_cfg)

    # Deep-merge top-level overrides
    cfg = deep_merge(cfg, entry.get("overrides", {}))

    # Patch individual stages by name
    if "stage_overrides" in entry and "stages" in cfg:
        cfg["stages"] = apply_stage_overrides(cfg["stages"], entry["stage_overrides"])

    # Stamp run_name, model seed, split seed, output_root.
    base_run_name = entry["run_name"]
    cfg["run_name"]    = f"{base_run_name}/rep{rep:02d}"
    cfg["seed"]        = seed
    cfg["fold_seed"]   = fold_seed
    cfg["output_root"] = output_root

    return cfg


def save_dispatched_config(cfg: dict, base_run_name: str, rep: int,
                           repo_root: Path) -> Path:
    """Write the merged config to v4/configs/dispatched/ and return its path."""
    out_dir = repo_root / "v4" / "configs" / "dispatched" / base_run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"rep{rep:02d}.json"
    path.write_text(json.dumps(cfg, indent=2))
    return path


# ──────────────────────────────────────────────────────────────
# Server HTTP helper
# ──────────────────────────────────────────────────────────────

class _API:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self._h = {"x-token": token}

    def post(self, path: str, body: dict) -> dict:
        r = requests.post(f"{self.base_url}{path}", headers=self._h,
                          json=body, timeout=10)
        r.raise_for_status()
        return r.json()


# ──────────────────────────────────────────────────────────────
# Dispatch
# ──────────────────────────────────────────────────────────────

def dispatch_batch(
    api: _API,
    base_cfg: dict,
    batch: list[dict],
    *,
    default_reps:    int,
    seed_start:      int,
    seed_step:       int,
    fold_seed_start: int,
    fold_seed_step:  int,
    output_root:     str,
    default_priority: int,
    server_path:     str,
    repo_root:       Path,
    dry_run:         bool,
) -> None:
    total = skipped = 0
    for entry in batch:
        run_name = entry["run_name"]
        reps     = entry.get("reps", default_reps)
        priority = entry.get("priority", default_priority)

        print(f"\n[dispatch] {run_name}  ({reps} reps)")

        for rep in range(reps):
            seed      = seed_start + rep * seed_step
            fold_seed = fold_seed_start + rep * fold_seed_step
            cfg       = build_config(base_cfg, entry, rep, seed, fold_seed, output_root)
            cfg_path = save_dispatched_config(cfg, run_name, rep, repo_root)

            rel_path        = cfg_path.relative_to(repo_root)
            server_cfg_path = str(rel_path)

            job_body = {
                "run_name":   cfg["run_name"],   # per-rep, e.g. ".../rep05"
                "module":     "v4.classes.v4_hypertower",
                "args":       ["--config", server_cfg_path],
                "output_dir": output_root,
                "priority":   priority,
            }

            if dry_run:
                print(
                    f"  [dry-run] rep{rep:02d}  seed={seed}  "
                    f"fold_seed={fold_seed}  config={server_cfg_path}"
                )
                total += 1
            else:
                resp = api.post("/jobs", job_body)
                if resp.get("skipped"):
                    print(f"  rep{rep:02d}  [skip — results exist on disk]")
                    skipped += 1
                elif resp.get("duplicate"):
                    print(f"  rep{rep:02d}  [skip — already queued]  job_id={resp['job_id']}")
                    skipped += 1
                else:
                    print(
                        f"  rep{rep:02d}  seed={seed}  fold_seed={fold_seed}  "
                        f"job_id={resp['job_id']}  config={server_cfg_path}"
                    )
                    total += 1

    action = "would submit" if dry_run else "submitted"
    skip_note = f"  ({skipped} already queued/done, skipped)" if skipped else ""
    print(f"\n[dispatch] {action} {total} jobs across {len(batch)} experiment(s){skip_note}")


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def main():
    repo_root = Path(__file__).resolve().parents[2]

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--server", default=os.environ.get("HT_SERVER", ""),
                    help="Server URL (or set HT_SERVER)")
    ap.add_argument("--token", default=os.environ.get("HT_TOKEN", ""),
                    help="Shared secret (or set HT_TOKEN)")
    ap.add_argument("--config", required=True,
                    help="Base config JSON file")
    ap.add_argument("--batch", required=True,
                    help="Batch JSON file listing experiments")
    ap.add_argument("--reps", type=int, default=10,
                    help="Repetitions per experiment (default: 10)")
    ap.add_argument("--seed-start", type=int, default=1234,
                    help="Seed for rep00 (default: 1234)")
    ap.add_argument("--seed-step", type=int, default=100,
                    help="Seed increment per rep (default: 100)")
    ap.add_argument("--fold-seed-start", type=int, default=100,
                    help="Split fold_seed for rep00 (default: 100; matches v3)")
    ap.add_argument("--fold-seed-step", type=int, default=100,
                    help="Split fold_seed increment per rep (default: 100; matches v3)")
    ap.add_argument("--output-root", default="v4/results",
                    help="Output root written into each config (default: v4/results)")
    ap.add_argument("--server-path", default="",
                    help="Absolute path to hypertower root on server "
                         "(used to build config paths in job args; "
                         "if omitted, relative paths are used)")
    ap.add_argument("--priority", type=int, default=0,
                    help="Default job priority (default: 0)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print jobs without submitting")
    args = ap.parse_args()

    if not args.dry_run:
        if not args.server:
            ap.error("--server is required (or set HT_SERVER)")
        if not args.token:
            ap.error("--token is required (or set HT_TOKEN)")

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = repo_root / cfg_path
    base_cfg = json.loads(cfg_path.read_text())

    batch_path = Path(args.batch)
    if not batch_path.is_absolute():
        batch_path = repo_root / batch_path
    batch = json.loads(batch_path.read_text())

    if not isinstance(batch, list):
        sys.exit("batch.json must be a JSON array")
    for i, entry in enumerate(batch):
        if "run_name" not in entry:
            sys.exit(f"batch entry {i} is missing required 'run_name'")

    api = _API(args.server, args.token) if not args.dry_run else None

    dispatch_batch(
        api,
        base_cfg,
        batch,
        default_reps=args.reps,
        seed_start=args.seed_start,
        seed_step=args.seed_step,
        fold_seed_start=args.fold_seed_start,
        fold_seed_step=args.fold_seed_step,
        output_root=args.output_root,
        default_priority=args.priority,
        server_path=args.server_path,
        repo_root=repo_root,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
