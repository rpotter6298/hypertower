"""
HyperTower distributed job CLI — submit jobs, view status.

Usage:
    # View all connected clients
    python -m v3.distributed.cli clients

    # View a specific client
    python -m v3.distributed.cli clients <client_id>

    # View jobs (optionally filter by state)
    python -m v3.distributed.cli jobs [--state pending|running|done|failed]

    # Submit a job
    python -m v3.distributed.cli submit \\
        --run-name phase2/leaky \\
        -- --bridge-mode image_only --backbone resnet50 --reps 10 ...

    # Submit all jobs from a batch file (JSON)
    python -m v3.distributed.cli submit-batch jobs.json

    # Cancel a pending job
    python -m v3.distributed.cli cancel <job_id>

Global flags (can also be set via env vars):
    --server  HT_SERVER   e.g. http://apollo:8765
    --token   HT_TOKEN
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests

# ──────────────────────────────────────────────────────────────
# HTTP helpers
# ──────────────────────────────────────────────────────────────

class _API:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self._h = {"x-token": token}

    def get(self, path: str, **params) -> object:
        r = requests.get(f"{self.base_url}{path}", headers=self._h,
                         params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    def post(self, path: str, body: dict) -> object:
        r = requests.post(f"{self.base_url}{path}", headers=self._h,
                          json=body, timeout=10)
        r.raise_for_status()
        return r.json()

    def delete(self, path: str) -> object:
        r = requests.delete(f"{self.base_url}{path}", headers=self._h, timeout=10)
        r.raise_for_status()
        return r.json()


# ──────────────────────────────────────────────────────────────
# Formatting helpers
# ──────────────────────────────────────────────────────────────

def _ago(ts: Optional[str]) -> str:
    if not ts:
        return "-"
    try:
        dt = datetime.fromisoformat(ts)
        delta = datetime.now(timezone.utc) - dt
        secs = int(delta.total_seconds())
        if secs < 60:
            return f"{secs}s ago"
        elif secs < 3600:
            return f"{secs//60}m ago"
        else:
            return f"{secs//3600}h{(secs%3600)//60}m ago"
    except Exception:
        return ts


def _col(text: str, width: int) -> str:
    text = str(text) if text is not None else "-"
    return text[:width].ljust(width)


def _table(rows: list[list[str]], headers: list[str]):
    widths = [max(len(str(r[i])) for r in ([headers] + rows)) for i in range(len(headers))]
    sep = "  "
    def _row(r):
        return sep.join(str(r[i]).ljust(widths[i]) for i in range(len(r)))
    print(_row(headers))
    print("-" * (sum(widths) + len(sep) * (len(widths) - 1)))
    for r in rows:
        print(_row(r))


# ──────────────────────────────────────────────────────────────
# Subcommands
# ──────────────────────────────────────────────────────────────

def _clients_table(api: _API):
    """Render one clients snapshot. Returns the printed lines as a string."""
    clients = api.get("/clients")
    if not clients:
        return "No clients connected."
    rows = []
    for c in clients:
        s = c["status"]
        parts = []
        if s.get("rep") is not None:
            parts.append(f"rep{s['rep']:02d}")
        if s.get("fold") is not None:
            parts.append(f"f{s['fold']}")
        if s.get("epoch") is not None:
            parts.append(f"ep{s['epoch']}/{s.get('total_epochs','?')}")
            parts.append(f"auc={s.get('last_auc','?')}")
        prog = " ".join(parts) if parts else "-"
        rows.append([
            c["client_id"],
            c["hostname"],
            c["gpu_info"][:30],
            s["state"],
            s.get("run_name") or "-",
            prog,
            _ago(c["last_seen"]),
        ])
    headers = ["ID", "HOST", "GPU", "STATE", "RUN", "PROGRESS", "SEEN"]
    widths = [max(len(str(r[i])) for r in ([headers] + rows)) for i in range(len(headers))]
    sep = "  "
    lines = []
    lines.append(sep.join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
    lines.append("-" * (sum(widths) + len(sep) * (len(widths) - 1)))
    for r in rows:
        lines.append(sep.join(str(r[i]).ljust(widths[i]) for i in range(len(r))))
    return "\n".join(lines)


def cmd_clients(api: _API, args):
    if hasattr(args, "client_id") and args.client_id:
        data = api.get(f"/clients/{args.client_id}")
        s = data["status"]
        print(f"client_id  : {data['client_id']}")
        print(f"hostname   : {data['hostname']}")
        print(f"gpu        : {data['gpu_info']}")
        print(f"last_seen  : {_ago(data['last_seen'])}")
        print(f"state      : {s['state']}")
        if s.get("job_id"):
            print(f"job        : {s['job_id']}  ({s.get('run_name', '')})")
        if s.get("rep") is not None:
            print(f"progress   : rep {s['rep']}  fold {s.get('fold', '?')}  "
                  f"ep {s.get('epoch', '?')}/{s.get('total_epochs', '?')}  "
                  f"({s.get('last_epoch_secs', '?')}s/ep)  "
                  f"auc={s.get('last_auc', '?')}")
        return

    watch = getattr(args, "watch", False)
    interval = getattr(args, "interval", 5)

    if not watch:
        print(_clients_table(api))
        return

    try:
        while True:
            now = datetime.now().strftime("%H:%M:%S")
            print(f"\033[H\033[2J", end="")   # clear screen
            print(f"HyperTower clients  [{now}]  (Ctrl-C to exit)\n")
            print(_clients_table(api))
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped.")


def _rep_label(job: dict) -> str:
    """Extract --rep-index from job args if present."""
    try:
        a = json.loads(job["args"]) if isinstance(job.get("args"), str) else job.get("args", [])
        if "--rep-index" in a:
            idx = a[a.index("--rep-index") + 1]
            return f"rep{int(idx):02d}"
    except Exception:
        pass
    return "-"


def cmd_jobs(api: _API, args):
    params = {}
    if hasattr(args, "state") and args.state:
        params["state"] = args.state
    jobs = api.get("/jobs", **params)
    if not jobs:
        print("No jobs.")
        return
    rows = []
    for j in jobs:
        attempts = j.get("attempts", 0)
        rows.append([
            j["job_id"][:12],
            j["run_name"],
            _rep_label(j),
            j["state"],
            f"{attempts}" if attempts else "-",
            j.get("assigned_to") or "-",
            _ago(j["created_at"]),
            _ago(j.get("started_at")),
            _ago(j.get("completed_at")),
        ])
    _table(rows, ["JOB_ID", "RUN_NAME", "REP", "STATE", "TRIES", "CLIENT", "CREATED", "STARTED", "DONE"])
    pending = sum(1 for j in jobs if j["state"] == "pending")
    running = sum(1 for j in jobs if j["state"] == "running")
    done    = sum(1 for j in jobs if j["state"] == "done")
    failed  = sum(1 for j in jobs if j["state"] == "failed")
    print(f"\n  {len(jobs)} total  |  {pending} pending  {running} running  {done} done  {failed} failed")


def cmd_submit(api: _API, args):
    body = {
        "run_name": args.run_name,
        "module": args.module,
        "args": args.run_args,
        "output_dir": args.output_dir,
        "priority": args.priority,
    }
    resp = api.post("/jobs", body)
    print(f"Queued job {resp['job_id']} ({args.run_name})")


def cmd_submit_batch(api: _API, args):
    with open(args.batch_file) as f:
        jobs = json.load(f)
    for job in jobs:
        resp = api.post("/jobs", job)
        print(f"Queued {resp['job_id']}  ({job['run_name']})")


def cmd_submit_cv(api: _API, args):
    """Submit one job per rep, each writing to repNN under the same run-name."""
    seed_start = args.rep_seed_start
    seed_step  = args.rep_seed_step
    submitted  = []
    for i in range(args.reps):
        seed = seed_start + i * seed_step
        rep_args = [a for a in args.run_args
                    if a not in ("--reps", "--rep-seed-start", "--rep-seed-step")]
        rep_args += [
            "--reps", "1",
            "--rep-seed-start", str(seed),
            "--rep-index", str(i),
        ]
        body = {
            "run_name": args.run_name,
            "module": args.module,
            "args": rep_args,
            "output_dir": args.output_dir,
            "priority": args.priority,
        }
        resp = api.post("/jobs", body)
        submitted.append(resp["job_id"])
        print(f"Queued rep{i:02d}  seed={seed}  job_id={resp['job_id']}")
    print(f"\n{len(submitted)} jobs queued for run '{args.run_name}'")


def cmd_cancel(api: _API, args):
    resp = api.delete(f"/jobs/{args.job_id}")
    print(f"Cancelled {args.job_id}" if resp.get("ok") else resp)


def cmd_clear(api: _API, args):
    body: dict = {}
    if args.all:
        body["all"] = True
    elif args.run_name:
        body["run_name"] = args.run_name
    else:
        body["states"] = args.states or ["done", "failed", "cancelled"]
    resp = api.post("/jobs/clear", body)
    print(f"Cleared {resp['cleared']} jobs.")


# ──────────────────────────────────────────────────────────────
# Parser
# ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default=os.environ.get("HT_SERVER", ""),
                    help="Server URL (or set HT_SERVER)")
    ap.add_argument("--token", default=os.environ.get("HT_TOKEN", ""),
                    help="Shared secret (or set HT_TOKEN)")

    sub = ap.add_subparsers(dest="cmd", required=True)

    # clients
    p_cl = sub.add_parser("clients", help="List clients or inspect one")
    p_cl.add_argument("client_id", nargs="?")
    p_cl.add_argument("--watch", "-w", action="store_true",
                      help="Live monitoring mode — refresh every --interval seconds")
    p_cl.add_argument("--interval", "-n", type=int, default=5,
                      help="Refresh interval in seconds for --watch (default: 5)")

    # jobs
    p_j = sub.add_parser("jobs", help="List jobs")
    p_j.add_argument("--state", choices=["pending", "running", "done", "failed", "cancelled"])

    # submit
    p_s = sub.add_parser("submit", help="Submit a single job")
    p_s.add_argument("--run-name", required=True)
    p_s.add_argument("--module", default="v3.scripts.main.run_cv")
    p_s.add_argument("--output-dir", default="v3/results")
    p_s.add_argument("--priority", type=int, default=0)
    p_s.add_argument("run_args", nargs=argparse.REMAINDER,
                     help="Args after '--' are forwarded to the module")

    # submit-cv
    p_cv = sub.add_parser("submit-cv",
                           help="Submit one job per rep (distributed 10x5 etc.)")
    p_cv.add_argument("--run-name", required=True)
    p_cv.add_argument("--reps", type=int, required=True)
    p_cv.add_argument("--rep-seed-start", type=int, default=100)
    p_cv.add_argument("--rep-seed-step", type=int, default=100)
    p_cv.add_argument("--module", default="v3.scripts.main.run_cv")
    p_cv.add_argument("--output-dir", default="v3/results")
    p_cv.add_argument("--priority", type=int, default=0)
    p_cv.add_argument("run_args", nargs=argparse.REMAINDER,
                      help="Args after '--' forwarded to run_cv (omit --reps/--rep-seed-*)")

    # submit-batch
    p_b = sub.add_parser("submit-batch", help="Submit jobs from a JSON file")
    p_b.add_argument("batch_file")

    # cancel
    p_c = sub.add_parser("cancel", help="Cancel a pending job")
    p_c.add_argument("job_id")

    # clear
    p_cl = sub.add_parser("clear", help="Delete jobs by run-name, state, or everything")
    p_cl.add_argument("--run-name", default=None, help="Delete all jobs with this run-name")
    p_cl.add_argument("--states", nargs="+",
                      default=None,
                      choices=["done", "failed", "cancelled", "pending", "running"],
                      help="Delete jobs in these states (default: done+failed+cancelled)")
    p_cl.add_argument("--all", action="store_true", help="Delete ALL jobs")

    args = ap.parse_args()

    if not args.server:
        ap.error("--server is required (or set HT_SERVER)")
    if not args.token:
        ap.error("--token is required (or set HT_TOKEN)")

    # strip leading "--" from run_args if present
    if hasattr(args, "run_args") and args.run_args and args.run_args[0] == "--":
        args.run_args = args.run_args[1:]

    api = _API(args.server, args.token)

    dispatch = {
        "clients": cmd_clients,
        "jobs": cmd_jobs,
        "submit": cmd_submit,
        "submit-cv": cmd_submit_cv,
        "submit-batch": cmd_submit_batch,
        "cancel": cmd_cancel,
        "clear": cmd_clear,
    }
    dispatch[args.cmd](api, args)


if __name__ == "__main__":
    main()
