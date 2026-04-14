"""
Dispatch phase 6a experiment runs (geometry vector injection) to the distributed job server.

Covers the geometry_vector_gt and geometry_vector_unet groups from experiment_grid.json:
  - GT geometry vector × {single, ensemble, fused-head}
  - U-Net geometry vector × {single, ensemble, fused-head}

Skips geometry_tower_* groups (needs_implementation — will get dispatch_phase6b.py).

Usage:
    python -m v3.scripts.main.phase6.dispatch_phase6a \
        --server http://hades:8765 --token hypertower

    # Dry run (print what would be submitted, don't actually submit):
    python -m v3.scripts.main.phase6.dispatch_phase6a \
        --server http://hades:8765 --token hypertower --dry-run

    # Override number of reps (default 10):
    python -m v3.scripts.main.phase6.dispatch_phase6a \
        --server http://hades:8765 --token hypertower --reps 4

NOTE: Requires --geometry-dim and --geometry-source to be wired into
v3_hypertower.py before these jobs will run successfully.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests

# Allow running as `python v3/scripts/main/phase6/dispatch_phase6a.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

GRID_PATH    = Path(__file__).parent / "experiment_grid.json"
RESULTS_ROOT = Path(__file__).resolve().parents[4] / "v3" / "results"
MODULE       = "v3.scripts.main.run_cv"
OUTPUT_DIR   = "v3/results"
REP_SEED_START = 100
REP_SEED_STEP  = 100


# ── Completion check ──────────────────────────────────────────────────────────

def _completed_reps(run_name: str, reps: int) -> list[int]:
    """Return list of rep indices that already have a summary.json (any tower mode)."""
    done = []
    for i in range(reps):
        rep_dir = RESULTS_ROOT / run_name / f"rep{i:02d}" / "binary"
        # Accept any tower mode subdir
        if rep_dir.exists() and any((rep_dir / tm / "summary.json").exists()
                                     for tm in ("single", "bilateral", "siamese", "ensemble", "tri", "tri_bilateral")):
            done.append(i)
    return done


# ── Server API ────────────────────────────────────────────────────────────────

class _API:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self._h = {"x-token": token}

    def get(self, path: str, **params) -> object:
        r = requests.get(f"{self.base_url}{path}", headers=self._h, params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    def post(self, path: str, body: dict) -> dict:
        r = requests.post(f"{self.base_url}{path}", headers=self._h, json=body, timeout=10)
        r.raise_for_status()
        return r.json()


def _queued_reps(jobs: list[dict], run_name: str) -> set[int]:
    """Return rep indices already pending or running in the server queue."""
    active = set()
    for job in jobs:
        if job["run_name"] != run_name:
            continue
        if job["state"] not in ("pending", "running"):
            continue
        try:
            args = job["args"] if isinstance(job["args"], list) else json.loads(job["args"])
            if "--rep-index" in args:
                active.add(int(args[args.index("--rep-index") + 1]))
        except Exception:
            pass
    return active


def _submit_cv(api: _API, run_name: str, run_args: list[str],
               reps: int, missing: list[int], dry_run: bool):
    """Submit one job per missing rep."""
    for i in missing:
        seed = REP_SEED_START + i * REP_SEED_STEP
        rep_args = run_args + [
            "--run-name", run_name,
            "--reps", "1",
            "--rep-seed-start", str(seed),
            "--rep-index", str(i),
        ]
        body = {
            "run_name": run_name,
            "module": MODULE,
            "args": rep_args,
            "output_dir": OUTPUT_DIR,
            "priority": 0,
        }
        if dry_run:
            print(f"  [dry-run] would queue rep{i:02d}  seed={seed}")
        else:
            resp = api.post("/jobs", body)
            print(f"  queued rep{i:02d}  seed={seed}  job_id={resp['job_id']}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default=os.environ.get("HT_SERVER", ""),
                    help="Server URL (or set HT_SERVER)")
    ap.add_argument("--token",  default=os.environ.get("HT_TOKEN",  ""),
                    help="Shared secret (or set HT_TOKEN)")
    ap.add_argument("--reps",   type=int, default=10,
                    help="Expected number of reps per run (default: 10)")
    ap.add_argument("--grid",   type=Path, default=GRID_PATH,
                    help="Path to experiment grid JSON (default: experiment_grid.json)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be submitted without actually submitting")
    args = ap.parse_args()

    if not args.dry_run:
        if not args.server:
            ap.error("--server is required (or set HT_SERVER)")
        if not args.token:
            ap.error("--token is required (or set HT_TOKEN)")
    elif not args.server or not args.token:
        print("[dry-run] no --server/--token provided — skipping queue check, showing disk state only")

    grid = json.loads(args.grid.read_text())
    common_args = grid["common_args"]
    api = _API(args.server, args.token) if (args.server and args.token) else None

    # Fetch current server queue once (pending + running)
    server_jobs: list[dict] = []
    if api:
        try:
            all_jobs = api.get("/jobs")
            server_jobs = [j for j in all_jobs if j["state"] in ("pending", "running")]
            print(f"[server] {len(server_jobs)} job(s) currently pending/running in queue")
        except Exception as e:
            print(f"[warn] could not fetch server queue: {e}")

    # Collect all runs: baseline + every group's runs (skip needs_implementation)
    all_runs = [grid["baseline"]]
    for group in grid["groups"]:
        if group.get("needs_implementation"):
            print(f"\n[skip] group '{group['name']}' — {group['needs_implementation']}")
            continue
        all_runs.extend(group["runs"])

    submitted_total = 0
    skipped_total   = 0

    for run in all_runs:
        run_name  = run["run_name"]
        run_args  = common_args + run.get("extra_args", [])
        done      = set(_completed_reps(run_name, args.reps))
        queued    = _queued_reps(server_jobs, run_name)
        accounted = done | queued
        missing   = [i for i in range(args.reps) if i not in accounted]

        if not missing:
            if len(done) == args.reps:
                print(f"\n[done]  {run_name}  ({args.reps}/{args.reps} reps complete)")
            else:
                in_q = sorted(queued - done)
                print(f"\n[skip]  {run_name}  ({len(done)} done, {len(in_q)} queued: {[f'rep{i:02d}' for i in in_q]})")
            skipped_total += 1
            continue

        parts = []
        if done:    parts.append(f"{len(done)} done")
        if queued:  parts.append(f"{len(queued - done)} queued")
        status = ", ".join(parts) if parts else "not started"
        print(f"\n[queue] {run_name}  ({status}) — submitting {len(missing)} rep(s)")
        _submit_cv(api, run_name, run_args, args.reps, missing, args.dry_run)
        submitted_total += len(missing)

    print(f"\n{'='*50}")
    print(f"Submitted: {submitted_total} jobs  |  Already accounted for: {skipped_total} runs")
    if grid.get("groups"):
        needs_impl = sum(1 for g in grid["groups"] if g.get("needs_implementation"))
        if needs_impl:
            print(f"Skipped (needs implementation): {needs_impl} group(s)")


if __name__ == "__main__":
    main()
