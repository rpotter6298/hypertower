"""
HyperTower distributed job client daemon.

Registers with the server, polls for jobs, syncs code, runs training,
uploads results, and loops. Parses stdout to stream live status.

Usage:
    python -m v3.distributed.client \\
        --server http://apollo:8765 \\
        --token <secret> \\
        --server-ssh rpotter@apollo \\
        --server-path /home/rpotter/hypertower \\
        [--local-path ~/hypertower] \\
        [--poll-interval 15]

Compatibility test (verify GPU env, 1 epoch dry-run):
    python -m v3.distributed.client ... --test [extra run_cv args]
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import requests

from .protocol import (
    JobResult,
    JobSpec,
    PollResponse,
    RegisterRequest,
    RegisterResponse,
    StatusPush,
)

# ──────────────────────────────────────────────────────────────
# Server HTTP wrapper
# ──────────────────────────────────────────────────────────────

class _Server:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self._h = {"x-token": token}
        self.client_id: str = ""

    def _post(self, path: str, **kw) -> dict:
        r = requests.post(f"{self.base_url}{path}", headers=self._h, timeout=15, **kw)
        r.raise_for_status()
        return r.json()

    def register(self, hostname: str, gpu_info: str) -> str:
        data = self._post("/register",
                          json={"hostname": hostname, "gpu_info": gpu_info})
        self.client_id = data["client_id"]
        self.hostname = hostname
        self.gpu_info = gpu_info
        return self.client_id

    def _reregister(self):
        try:
            self._post("/register",
                       json={"hostname": self.hostname, "gpu_info": self.gpu_info},
                       params={"reuse_id": self.client_id})
        except Exception:
            pass

    def poll(self) -> Optional[JobSpec]:
        data = self._post("/poll", params={"client_id": self.client_id})
        if data.get("please_reregister"):
            self._reregister()
        return JobSpec(**data["job"]) if data.get("job") else None

    def push_status(self, status: StatusPush):
        try:
            r = requests.post(
                f"{self.base_url}/status/{self.client_id}",
                json=status.model_dump(),
                headers=self._h,
                timeout=5,
            )
            if r.ok and r.json().get("please_reregister"):
                self._reregister()
        except Exception:
            pass  # don't crash job on status push failure

    def complete(self, job_id: str, success: bool, error_msg: Optional[str] = None):
        self._post("/complete",
                   json={"job_id": job_id, "success": success, "error_msg": error_msg})


# ──────────────────────────────────────────────────────────────
# GPU info
# ──────────────────────────────────────────────────────────────

def _gpu_info() -> str:
    # NVIDIA
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        if out:
            return " | ".join(out.splitlines())
    except Exception:
        pass
    # AMD
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showproductname", "--csv"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().splitlines()
        names = [l for l in out if l and not l.startswith("device")]
        if names:
            return "AMD: " + " | ".join(names)
    except Exception:
        pass
    # AMD fallback
    try:
        out = subprocess.check_output(
            ["rocminfo"],
            text=True, stderr=subprocess.DEVNULL,
        )
        names = [l.split(":", 1)[1].strip() for l in out.splitlines()
                 if "Marketing Name:" in l]
        if names:
            return "AMD: " + " | ".join(names)
    except Exception:
        pass
    return "no-gpu"


# ──────────────────────────────────────────────────────────────
# rsync helpers
# ──────────────────────────────────────────────────────────────

def _rsync(src: str, dst: str, delete: bool = False):
    cmd = ["rsync", "-az", "--info=progress2"]
    if delete:
        cmd.append("--delete")
    cmd += [src, dst]
    subprocess.run(cmd, check=True)


def _sync_code(server_ssh: str, server_path: str, local_path: str):
    """Pull v3/ source from server → local (overwrites local changes)."""
    src = f"{server_ssh}:{server_path}/v3/"
    dst = f"{local_path}/v3/"
    Path(dst).mkdir(parents=True, exist_ok=True)
    _rsync(src, dst, delete=True)


def _upload_results(server_ssh: str, server_path: str, local_path: str,
                    run_name: str, output_dir: str):
    src = f"{local_path}/{output_dir}/{run_name}/"
    dst = f"{server_ssh}:{server_path}/{output_dir}/{run_name}/"
    # Ensure parent directory exists on server before rsyncing
    remote_parent = f"{server_path}/{output_dir}/{Path(run_name).parent}"
    subprocess.run(["ssh", server_ssh, f"mkdir -p '{remote_parent}'"], check=True)
    _rsync(src, dst)


def _clean_local(local_path: str, run_name: str, output_dir: str):
    target = Path(local_path) / output_dir / run_name
    if target.exists():
        shutil.rmtree(target)
        print(f"[client] cleaned {target}", flush=True)


# ──────────────────────────────────────────────────────────────
# Stdout parsers (match v3_hypertower.py print format)
# ──────────────────────────────────────────────────────────────

# "  ep   3/ 30 (6.5s)  auc=0.75  acc=0.82"
_EP_RE = re.compile(
    r"ep\s+(\d+)/\s*(\d+)\s+\(([0-9.]+)s\)\s+auc=([0-9.nan]+)\s+acc=([0-9.nan]+)"
)
# "[binary:single] fold 2/5"
_FOLD_RE = re.compile(r"fold\s+(\d+)/\d+", re.IGNORECASE)
# "Rep 3  fold_seed=..." or "Rep 3/10  fold_seed=..."
_REP_RE = re.compile(r"Rep\s+(\d+)", re.IGNORECASE)


def _parse_line(line: str) -> dict:
    """Return any structured fields found in a stdout line."""
    out = {}
    m = _FOLD_RE.search(line)
    if m:
        out["fold"] = int(m.group(1))
    m = _REP_RE.search(line)
    if m:
        out["rep"] = int(m.group(1))
    m = _EP_RE.search(line)
    if m:
        out["epoch"] = int(m.group(1))
        out["total_epochs"] = int(m.group(2))
        out["last_epoch_secs"] = float(m.group(3))
        try:
            out["last_auc"] = float(m.group(4))
        except ValueError:
            pass
    return out


# ──────────────────────────────────────────────────────────────
# Core job runner
# ──────────────────────────────────────────────────────────────

def _run_job(job: JobSpec, server: _Server,
             server_ssh: str, server_path: str, local_path: str,
             no_sync: bool = False, num_workers: Optional[int] = None,
             extra_args: list[str] | None = None) -> bool:
    extra_args = extra_args or []
    # 1. Sync code
    if no_sync:
        print(f"[client] skipping sync (--no-sync)", flush=True)
    else:
        print(f"[client] syncing v3/ from server...", flush=True)
        server.push_status(StatusPush(state="syncing", job_id=job.job_id, run_name=job.run_name))
        _sync_code(server_ssh, server_path, local_path)

    # 2. Launch training subprocess
    job_args = job.args
    if num_workers is not None and "--num-workers" not in job_args:
        job_args = job_args + ["--num-workers", str(num_workers)]
    cmd = [sys.executable, "-m", job.module] + job_args + extra_args
    print(f"[client] running: {' '.join(cmd)}", flush=True)
    server.push_status(StatusPush(state="running", job_id=job.job_id, run_name=job.run_name))

    # Write stdout to a temp file so the child's fd table is unmodified —
    # this lets Python 3.14's forkserver communicate over its own pipes without
    # interference. We tail the file from a thread for live status.
    log_dir = Path(local_path) / "v3" / "distributed" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"job_{job.job_id}.log"
    ctx: dict = {}

    def _tail(path: Path):
        with open(path, "r") as f:
            while True:
                raw = f.readline()
                if raw:
                    print(raw, end="", flush=True)
                    info = _parse_line(raw)
                    ctx.update(info)
                    if "epoch" in info:
                        server.push_status(StatusPush(
                            state="running",
                            job_id=job.job_id,
                            run_name=job.run_name,
                            rep=ctx.get("rep"),
                            fold=ctx.get("fold"),
                            epoch=ctx.get("epoch"),
                            total_epochs=ctx.get("total_epochs"),
                            last_epoch_secs=ctx.get("last_epoch_secs"),
                            last_auc=ctx.get("last_auc"),
                        ))
                elif proc.poll() is not None:
                    # Drain any remaining output then exit
                    for raw in f:
                        print(raw, end="", flush=True)
                    break
                else:
                    time.sleep(0.05)

    with open(log_file, "w") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=logf,
            cwd=local_path,
            start_new_session=True,  # own process group so we can kill all workers on failure
        )

    tailer = threading.Thread(target=_tail, args=(log_file,), daemon=True)
    tailer.start()
    proc.wait()
    tailer.join(timeout=5)
    log_file.unlink(missing_ok=True)

    success = proc.returncode == 0

    if not success:
        # Kill any surviving worker processes in the group (e.g. DataLoader workers
        # that outlived the main process after an OOM kill)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass  # already gone

    if not success:
        print(f"[client] job FAILED (rc={proc.returncode})", flush=True)
        return False

    # 3. Upload results (skip when client IS the server — results already in place)
    if no_sync:
        print(f"[client] skipping upload (--no-sync, results already local)", flush=True)
    else:
        print(f"[client] uploading results...", flush=True)
        server.push_status(StatusPush(state="uploading", job_id=job.job_id, run_name=job.run_name))
        _upload_results(server_ssh, server_path, local_path, job.run_name, job.output_dir)

        # 4. Clean local (only remote clients need cleanup)
        _clean_local(local_path, job.run_name, job.output_dir)
    print(f"[client] job {job.job_id} complete.", flush=True)
    return True


# ──────────────────────────────────────────────────────────────
# Compatibility test
# ──────────────────────────────────────────────────────────────

def _compat_test(server: _Server, server_ssh: str, server_path: str,
                 local_path: str, extra_args: list[str], no_sync: bool = False):
    """Run 1 epoch on 1 rep to verify the env works end-to-end."""
    print("[client] === compatibility test ===", flush=True)
    job = JobSpec(
        job_id="compat-test",
        run_name="_compat_test",
        module="v3.scripts.main.run_cv",
        args=[
            "--bridge-mode", "image_only",
            "--backbone", "resnet50",
            "--epochs", "1",
            "--reps", "1",
            "--rep-seed-start", "42",
            "--in-memory-cache",
            "--num-workers", "0",
            "--output-root", "v3/results",
            "--run-name", "_compat_test",
        ] + extra_args,
        output_dir="v3/results",
    )
    ok = _run_job(job, server, server_ssh, server_path, local_path,
                  no_sync=no_sync, extra_args=[a for a in extra_args if a != "--"])
    # always clean up test results
    _clean_local(local_path, "_compat_test", "v3/results")
    if ok:
        print("[client] compatibility test PASSED ✓", flush=True)
    else:
        print("[client] compatibility test FAILED ✗", flush=True)
    return ok


# ──────────────────────────────────────────────────────────────
# Main daemon
# ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--server", required=True,
                    help="Server URL, e.g. http://apollo:8765")
    ap.add_argument("--token", default=os.environ.get("HT_TOKEN", ""),
                    help="Shared secret (or set HT_TOKEN env var)")
    ap.add_argument("--server-ssh", required=True,
                    help="SSH target for rsync, e.g. rpotter@apollo")
    ap.add_argument("--server-path", required=True,
                    help="Absolute path to hypertower root on server")
    ap.add_argument("--local-path",
                    default=str(Path.home() / "hypertower"),
                    help="Absolute path to hypertower root on this machine")
    ap.add_argument("--poll-interval", type=int, default=15,
                    help="Seconds to wait between polls when idle")
    ap.add_argument("--num-workers", type=int, default=None,
                    help="Override DataLoader num_workers for all jobs on this client.")
    ap.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[],
                    help="Extra args appended to every job on this client (e.g. --cache-workers 0). "
                         "Use -- to separate from client args: --extra-args -- --cache-workers 0")
    ap.add_argument("--no-sync", action="store_true",
                    help="Skip rsync of v3/ before each job (use when client IS the server)")
    ap.add_argument("--test", action="store_true",
                    help="Run 1-epoch compatibility test and exit")
    ap.add_argument("extra_args", nargs="*",
                    help="Extra args forwarded to run_cv in --test mode")
    args = ap.parse_args()

    if not args.token:
        ap.error("--token is required (or set HT_TOKEN)")

    hostname = socket.gethostname()
    gpu_info = _gpu_info()
    server = _Server(args.server, args.token)

    client_id = server.register(hostname, gpu_info)
    print(f"[client] registered as {client_id} ({hostname} | {gpu_info})", flush=True)

    if args.test:
        sys.exit(0 if _compat_test(
            server, args.server_ssh, args.server_path,
            args.local_path, args.extra_args,
            no_sync=args.no_sync,
        ) else 1)

    print(f"[client] polling every {args.poll_interval}s...", flush=True)
    while True:
        try:
            job = server.poll()
            if job is None:
                server.push_status(StatusPush(state="idle"))
                time.sleep(args.poll_interval)
                continue

            success = _run_job(
                job, server,
                args.server_ssh, args.server_path, args.local_path,
                no_sync=args.no_sync,
                num_workers=args.num_workers,
                extra_args=[a for a in args.extra_args if a != "--"],
            )
            server.complete(job.job_id, success,
                            error_msg=None if success else "non-zero exit code")
            server.push_status(StatusPush(state="idle"))

        except KeyboardInterrupt:
            print("\n[client] shutting down", flush=True)
            break
        except Exception as exc:
            print(f"[client] error: {exc}", flush=True)
            time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
