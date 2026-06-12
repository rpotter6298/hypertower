"""
HyperTower distributed job server.

Manages a SQLite job queue and a registry of connected clients.
Clients poll for work, push status updates, and report completion.

Usage:
    python -m v4.distributed.server --port 8765 --token <secret>

Environment:
    HT_TOKEN  — fallback if --token is not passed
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
import uvicorn

from .protocol import (
    ClientInfo,
    JobResult,
    JobSpec,
    JobSubmit,
    PollResponse,
    RegisterRequest,
    RegisterResponse,
    StatusPush,
)

# ──────────────────────────────────────────────────────────────
# Global state
# ──────────────────────────────────────────────────────────────

_TOKEN: str = ""
_DB_PATH: Path = Path("v4/distributed/jobs.db")
_REPO_ROOT: Path = Path.cwd()
_CLIENT_TTL: int = 120  # seconds before a client is considered gone
_MAX_ATTEMPTS: int = 4  # max times a job is retried (cumulatively, across clients)
                        # before being marked permanently failed
_CLIENT_FAIL_LIMIT: int = 3  # consecutive job failures from a single client before
                             # that client is quarantined (no more poll dispatches
                             # until it re-registers). A successful job resets it.
_SERVER_START_TS: float = 0.0  # set in main(); used as a reaper grace window

_clients: dict[str, ClientInfo] = {}
_clients_lock = threading.Lock()


def _reap_stale_clients():
    """Background thread: remove silent clients and re-queue their running jobs.

    On startup, the in-memory `_clients` dict is empty until clients re-register
    via the `please_reregister` mechanism.  We skip the running-job re-queue pass
    for the first `_CLIENT_TTL` seconds after startup so still-alive clients have
    time to come back; otherwise the reaper would orphan their jobs.
    """
    while True:
        time.sleep(30)
        cutoff = datetime.now(timezone.utc).timestamp() - _CLIENT_TTL

        with _clients_lock:
            stale = [
                cid
                for cid, c in _clients.items()
                if datetime.fromisoformat(c.last_seen).timestamp() < cutoff
            ]
            for cid in stale:
                print(
                    f"[server] reaped stale client {cid} ({_clients[cid].hostname})",
                    flush=True,
                )
                del _clients[cid]
            known_ids = set(_clients.keys())

        in_grace = (time.time() - _SERVER_START_TS) < _CLIENT_TTL
        if in_grace:
            continue

        with _db() as conn:
            rows = conn.execute(
                "SELECT job_id, assigned_to FROM jobs WHERE state='running'"
            ).fetchall()
            for row in rows:
                if row["assigned_to"] not in known_ids:
                    conn.execute(
                        "UPDATE jobs SET state='pending', assigned_to=NULL, started_at=NULL "
                        "WHERE job_id=?",
                        (row["job_id"],),
                    )
                    print(
                        f"[server] re-queued job {row['job_id']} "
                        f"(client {row['assigned_to']} unknown)",
                        flush=True,
                    )


# ──────────────────────────────────────────────────────────────
# Database helpers
# ──────────────────────────────────────────────────────────────


@contextmanager
def _db():
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _init_db():
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id       TEXT PRIMARY KEY,
                run_name     TEXT NOT NULL,
                module       TEXT NOT NULL,
                args         TEXT NOT NULL,   -- JSON list
                output_dir   TEXT NOT NULL DEFAULT 'v4/results',
                state        TEXT NOT NULL DEFAULT 'pending',
                priority     INTEGER NOT NULL DEFAULT 0,
                assigned_to  TEXT,
                created_at   TEXT NOT NULL,
                started_at   TEXT,
                completed_at TEXT,
                error_msg    TEXT,
                attempts     INTEGER NOT NULL DEFAULT 0
            )
        """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_priority ON jobs(priority)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_run_name ON jobs(run_name)")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_args ON jobs(args)")
        try:
            conn.execute(
                "ALTER TABLE jobs ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0"
            )
        except Exception:
            pass  # column already exists


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_client(client_id: str, hostname: str = "", gpu_info: str = "") -> bool:
    """Re-register a client that survived a server restart.
    Returns True if the client was unknown (placeholder created)."""
    if client_id not in _clients:
        _clients[client_id] = ClientInfo(
            client_id=client_id,
            hostname=hostname or client_id,
            gpu_info=gpu_info or "unknown",
            status=StatusPush(state="idle"),
            last_seen=_now(),
        )
        print(f"[server] re-registered {client_id} (survived restart)", flush=True)
        return True
    return False


# ──────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────

app = FastAPI(title="HyperTower Job Server")


def _check_token(x_token: str = Header(...)):
    if x_token != _TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")


# ── Registration ──────────────────────────────────────────────


@app.post(
    "/register", response_model=RegisterResponse, dependencies=[Depends(_check_token)]
)
def register(req: RegisterRequest, reuse_id: Optional[str] = None):
    with _clients_lock:
        client_id = (
            reuse_id if (reuse_id and reuse_id in _clients) else str(uuid.uuid4())[:8]
        )
        existing_status = (
            _clients[client_id].status
            if client_id in _clients
            else StatusPush(state="idle")
        )
        _clients[client_id] = ClientInfo(
            client_id=client_id,
            hostname=req.hostname,
            gpu_info=req.gpu_info,
            status=existing_status,
            last_seen=_now(),
        )
    action = "re-registered" if reuse_id else "registered"
    print(
        f"[server] {action} {client_id} ({req.hostname} | {req.gpu_info})", flush=True
    )
    return RegisterResponse(client_id=client_id)


# ── Job polling ───────────────────────────────────────────────


@app.post("/poll", response_model=PollResponse, dependencies=[Depends(_check_token)])
def poll(client_id: str):
    with _clients_lock:
        needs_reregister = _ensure_client(client_id)
        _clients[client_id].last_seen = _now()
        # Quarantine: a client that has failed _CLIENT_FAIL_LIMIT jobs in a row
        # is cut off from new dispatches until it re-registers. The client's
        # poll() helper auto-calls _reregister() when please_reregister=True
        # arrives, which clears the streak. Successful completions also reset
        # the streak, so a healthy client never trips this check.
        if _clients[client_id].fail_streak >= _CLIENT_FAIL_LIMIT:
            print(
                f"[server] {client_id} quarantined "
                f"(fail_streak={_clients[client_id].fail_streak}); "
                f"requesting re-register before next dispatch",
                flush=True,
            )
            return PollResponse(job=None, please_reregister=True)

    with _db() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE state='pending' "
            "ORDER BY priority DESC, created_at ASC LIMIT 1"
        ).fetchone()

        if row is None:
            return PollResponse(job=None, please_reregister=needs_reregister)

        job_id = row["job_id"]
        cur = conn.execute(
            "UPDATE jobs SET state='pending', assigned_to=NULL, started_at=NULL "
            "WHERE assigned_to=? AND state='running' AND job_id!=?",
            (client_id, job_id),
        )
        if cur.rowcount:
            print(
                f"[server] reset {cur.rowcount} orphaned running job(s) for {client_id}",
                flush=True,
            )
        conn.execute(
            "UPDATE jobs SET state='running', assigned_to=?, started_at=? WHERE job_id=?",
            (client_id, _now(), job_id),
        )

    job = JobSpec(
        job_id=job_id,
        run_name=row["run_name"],
        module=row["module"],
        args=json.loads(row["args"]),
        output_dir=row["output_dir"],
    )

    with _clients_lock:
        _clients[client_id].status = StatusPush(
            state="syncing", job_id=job_id, run_name=row["run_name"]
        )

    print(f"[server] dispatched {job_id} ({row['run_name']}) → {client_id}", flush=True)
    return PollResponse(job=job, please_reregister=needs_reregister)


# ── Status ────────────────────────────────────────────────────


@app.post("/status/{client_id}", dependencies=[Depends(_check_token)])
def push_status(client_id: str, status: StatusPush):
    with _clients_lock:
        needs_reregister = _ensure_client(client_id)
        _clients[client_id].status = status
        _clients[client_id].last_seen = _now()
    return {"ok": True, "please_reregister": needs_reregister}


@app.get("/clients", dependencies=[Depends(_check_token)])
def list_clients():
    with _clients_lock:
        return list(_clients.values())


@app.get("/clients/{client_id}", dependencies=[Depends(_check_token)])
def get_client(client_id: str):
    with _clients_lock:
        if client_id not in _clients:
            raise HTTPException(status_code=404, detail="Unknown client")
        return _clients[client_id]


# ── Job completion ────────────────────────────────────────────


@app.post("/complete", dependencies=[Depends(_check_token)])
def complete(result: JobResult):
    # Pull the worker that ran this job before touching the row, so we can
    # update its streak regardless of which branch we take below.
    with _db() as conn:
        assigned_row = conn.execute(
            "SELECT assigned_to FROM jobs WHERE job_id=?", (result.job_id,)
        ).fetchone()
    assigned_to = assigned_row["assigned_to"] if assigned_row else None

    with _clients_lock:
        if assigned_to and assigned_to in _clients:
            if result.success:
                _clients[assigned_to].fail_streak = 0
            else:
                _clients[assigned_to].fail_streak += 1
                if _clients[assigned_to].fail_streak >= _CLIENT_FAIL_LIMIT:
                    print(
                        f"[server] {assigned_to} hit fail_streak="
                        f"{_clients[assigned_to].fail_streak}; will quarantine "
                        f"on next poll",
                        flush=True,
                    )

    with _db() as conn:
        if result.success:
            conn.execute(
                "UPDATE jobs SET state='done', completed_at=?, error_msg=NULL WHERE job_id=?",
                (_now(), result.job_id),
            )
            print(f"[server] job {result.job_id} → done", flush=True)

            run_row = conn.execute(
                "SELECT run_name FROM jobs WHERE job_id=?", (result.job_id,)
            ).fetchone()
            if run_row:
                run_name = run_row["run_name"]
                remaining = conn.execute(
                    "SELECT COUNT(*) FROM jobs WHERE run_name=? AND state != 'done'",
                    (run_name,),
                ).fetchone()[0]
                if remaining == 0:
                    total = conn.execute(
                        "SELECT COUNT(*) FROM jobs WHERE run_name=?", (run_name,)
                    ).fetchone()[0]
                    conn.execute("DELETE FROM jobs WHERE run_name=?", (run_name,))
                    print(
                        f"[server] run '{run_name}' complete ({total} jobs) — cleared",
                        flush=True,
                    )
        else:
            row = conn.execute(
                "SELECT attempts FROM jobs WHERE job_id=?", (result.job_id,)
            ).fetchone()
            attempts = (row["attempts"] if row else 0) + 1
            if attempts < _MAX_ATTEMPTS:
                conn.execute(
                    "UPDATE jobs SET state='pending', assigned_to=NULL, started_at=NULL, "
                    "attempts=?, error_msg=? WHERE job_id=?",
                    (attempts, result.error_msg, result.job_id),
                )
                print(
                    f"[server] job {result.job_id} failed (attempt {attempts}/{_MAX_ATTEMPTS}), "
                    f"re-queuing",
                    flush=True,
                )
            else:
                conn.execute(
                    "UPDATE jobs SET state='failed', completed_at=?, attempts=?, error_msg=? "
                    "WHERE job_id=?",
                    (_now(), attempts, result.error_msg, result.job_id),
                )
                print(
                    f"[server] job {result.job_id} failed permanently after "
                    f"{attempts} attempts",
                    flush=True,
                )
    return {"ok": True}


# ── Job queue management ──────────────────────────────────────


@app.post("/jobs", dependencies=[Depends(_check_token)])
def submit_job(job: JobSubmit):
    result_dir = _REPO_ROOT / job.output_dir / job.run_name
    if result_dir.exists() and any(result_dir.rglob("summary.json")):
        print(f"[server] skipped {job.run_name} (results exist on disk)", flush=True)
        return {"job_id": "", "duplicate": False, "skipped": True}

    args_json = json.dumps(job.args)
    job_id    = str(uuid.uuid4())[:12]
    with _db() as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO jobs "
            "(job_id, run_name, module, args, output_dir, priority, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (job_id, job.run_name, job.module, args_json,
             job.output_dir, job.priority, _now()),
        )
        if cur.rowcount == 0:
            existing = conn.execute(
                "SELECT job_id FROM jobs WHERE args=?", (args_json,)
            ).fetchone()
            job_id = existing["job_id"]
            print(f"[server] duplicate ignored ({job.run_name}) → {job_id}", flush=True)
            return {"job_id": job_id, "duplicate": True}
    print(f"[server] queued {job_id} ({job.run_name})", flush=True)
    return {"job_id": job_id, "duplicate": False}


@app.get("/jobs", dependencies=[Depends(_check_token)])
def list_jobs(state: Optional[str] = None):
    with _db() as conn:
        if state:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE state=? ORDER BY created_at DESC", (state,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC"
            ).fetchall()
    return [dict(r) for r in rows]


@app.post("/jobs/clear", dependencies=[Depends(_check_token)])
def clear_jobs(body: dict):
    with _db() as conn:
        if body.get("all"):
            cur = conn.execute("DELETE FROM jobs")
        elif body.get("run_name"):
            cur = conn.execute("DELETE FROM jobs WHERE run_name=?", (body["run_name"],))
        else:
            states = body.get("states", ["done", "failed", "cancelled"])
            placeholders = ",".join("?" * len(states))
            cur = conn.execute(
                f"DELETE FROM jobs WHERE state IN ({placeholders})", states
            )
    print(f"[server] cleared {cur.rowcount} jobs", flush=True)
    return {"cleared": cur.rowcount}


@app.delete("/jobs/{job_id}", dependencies=[Depends(_check_token)])
def cancel_job(job_id: str):
    with _db() as conn:
        conn.execute(
            "UPDATE jobs SET state='cancelled' WHERE job_id=? AND state='pending'",
            (job_id,),
        )
    return {"ok": True}


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument(
        "--token",
        default=os.environ.get("HT_TOKEN", ""),
        help="Shared secret (or set HT_TOKEN env var)",
    )
    ap.add_argument(
        "--db", default="v4/distributed/jobs.db", help="Path to SQLite job database"
    )
    ap.add_argument(
        "--client-ttl",
        type=int,
        default=120,
        help="Seconds of silence before a client is reaped (default: 120)",
    )
    ap.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Max times a failed job is retried before being left as failed (default: 3)",
    )
    ap.add_argument(
        "--root",
        default="",
        help="Repo root for results-existence checks (default: cwd)",
    )
    args = ap.parse_args()

    if not args.token:
        ap.error("--token is required (or set HT_TOKEN)")

    global _TOKEN, _DB_PATH, _REPO_ROOT, _CLIENT_TTL, _MAX_ATTEMPTS, _SERVER_START_TS
    _TOKEN = args.token
    _DB_PATH = Path(args.db)
    _REPO_ROOT = Path(args.root).resolve() if args.root else Path.cwd()
    _CLIENT_TTL = args.client_ttl
    _MAX_ATTEMPTS = args.max_attempts
    _SERVER_START_TS = time.time()
    _init_db()

    reaper = threading.Thread(target=_reap_stale_clients, daemon=True)
    reaper.start()

    print(
        f"[server] listening on {args.host}:{args.port}  client_ttl={_CLIENT_TTL}s",
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
