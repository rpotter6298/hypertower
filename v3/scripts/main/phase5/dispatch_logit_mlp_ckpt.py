"""
Dispatch a 10×5 rep-CV of logit_mlp_head with --save-checkpoints.

Results land in v3/results/phase5/logit_mlp_head_ckpt/{rep00..rep09}/binary/ensemble/

Usage:
    # Dry run
    python -m v3.scripts.main.phase5.dispatch_logit_mlp_ckpt --dry-run

    # Submit to server
    python -m v3.scripts.main.phase5.dispatch_logit_mlp_ckpt \
        --server http://hades:8765 --token hypertower

    # Skip reps already done, re-queue only missing ones:
    python -m v3.scripts.main.phase5.dispatch_logit_mlp_ckpt \
        --server http://hades:8765 --token hypertower
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

RUN_NAME       = "phase5/logit_mlp_head_ckpt"
MODULE         = "v3.scripts.main.run_cv"
OUTPUT_DIR     = "v3/results"
RESULTS_ROOT   = Path(__file__).resolve().parents[4] / "v3" / "results"
REP_SEED_START = 100
REP_SEED_STEP  = 100
N_REPS         = 10

RUN_ARGS = [
    "--eval-mode",            "binary",
    "--bridge-mode",          "fused",
    "--tower-mode",           "ensemble",
    "--fused-head",
    "--head-type",            "logit_mlp",
    "--epochs",               "30",
    "--in-memory-cache",
    "--augment",
    "--tune-binary-threshold",
    "--backbone",             "refugelike",
    "--iop-corr-method",      "ratio",
    "--iop-drop-raw",
    "--exclude-cols",         "Axial_Length",
    "--output-root",          "v3/results",
    "--save-checkpoints",
]


# ── Completion check ──────────────────────────────────────────────────────────

def _completed_reps(reps: int) -> list[int]:
    done = []
    for i in range(reps):
        rep_dir = RESULTS_ROOT / RUN_NAME / f"rep{i:02d}" / "binary" / "ensemble"
        if (rep_dir / "summary.json").exists():
            # Also verify at least one checkpoint exists
            if any(rep_dir.glob("fold*/best_single.pt")):
                done.append(i)
            else:
                print(f"  [warn] rep{i:02d} has summary.json but no checkpoints — will re-queue")
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


def _queued_reps(jobs: list[dict]) -> set[int]:
    active = set()
    for job in jobs:
        if job.get("run_name") != RUN_NAME:
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server",  default=os.environ.get("HT_SERVER", ""))
    ap.add_argument("--token",   default=os.environ.get("HT_TOKEN",  ""))
    ap.add_argument("--reps",    type=int, default=N_REPS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.dry_run:
        if not args.server:
            ap.error("--server required (or set HT_SERVER)")
        if not args.token:
            ap.error("--token required (or set HT_TOKEN)")

    api = _API(args.server, args.token) if (args.server and args.token) else None

    server_jobs: list[dict] = []
    if api:
        try:
            all_jobs = api.get("/jobs")
            server_jobs = [j for j in all_jobs if j["state"] in ("pending", "running")]
            print(f"[server] {len(server_jobs)} job(s) pending/running")
        except Exception as e:
            print(f"[warn] could not fetch queue: {e}")

    done   = set(_completed_reps(args.reps))
    queued = _queued_reps(server_jobs)
    missing = [i for i in range(args.reps) if i not in (done | queued)]

    print(f"\nRun: {RUN_NAME}")
    print(f"  Done:    {sorted(done)}")
    print(f"  Queued:  {sorted(queued - done)}")
    print(f"  Missing: {missing}")

    if not missing:
        print("Nothing to submit.")
        return

    for i in missing:
        seed = REP_SEED_START + i * REP_SEED_STEP
        rep_args = RUN_ARGS + [
            "--run-name",       RUN_NAME,
            "--reps",           "1",
            "--rep-seed-start", str(seed),
            "--rep-index",      str(i),
        ]
        body = {
            "run_name":   RUN_NAME,
            "module":     MODULE,
            "args":       rep_args,
            "output_dir": OUTPUT_DIR,
            "priority":   0,
        }
        if args.dry_run:
            print(f"  [dry-run] rep{i:02d}  seed={seed}")
        else:
            resp = api.post("/jobs", body)
            print(f"  queued rep{i:02d}  seed={seed}  job_id={resp['job_id']}")


if __name__ == "__main__":
    main()
