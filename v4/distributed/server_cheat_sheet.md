# Distributed Server Cheat Sheet

All commands assume server is running on hades at port 8765.

## Start Server
```bash
python -m v4.distributed.server --token hypertower
```
Run inside tmux so it survives disconnects:
```bash
tmux new -s htserver
python -m v4.distributed.server --token hypertower
# Ctrl-B D to detach
tmux attach -t htserver   # reattach later
```

## Local Workflow (hades as server + client)

```bash
# Terminal 1 — server
python -m v4.distributed.server --token hypertower

# Terminal 2 — client (skip rsync, results already local)
python -m v4.distributed.client \
    --server http://hades:8765 --token hypertower \
    --server-ssh ignored --server-path ignored \
    --local-path /home/rpotter/hypertower \
    --no-sync

# Terminal 3 — dispatch (exits after queuing; client picks up jobs)
python -m v4.distributed.batch_dispatch \
    --server http://hades:8765 --token hypertower \
    --config v4/configs/ensemble_fused.json \
    --batch v4/scripts/experiments/my_batch.json
```

---

## Start Clients

**Hades (server-local, no sync):**
```bash
python -m v4.distributed.client \
    --server http://hades:8765 --token hypertower \
    --server-ssh ignored --server-path ignored \
    --local-path /home/rpotter/hypertower \
    --no-sync
```

**Apollo (remote client):**
```bash
python -m v4.distributed.client \
    --server http://hades:8765 --token hypertower \
    --server-ssh rpotter@hades \
    --server-path /home/rpotter/hypertower \
    --local-path /home/odin/hypertower
```

---

## Monitoring

**Live client monitor (refreshes every 5s):**
```bash
python -m v4.distributed.cli --server http://hades:8765 --token hypertower clients --watch
```

**Faster refresh:**
```bash
python -m v4.distributed.cli --server http://hades:8765 --token hypertower clients --watch --interval 2
```

**Inspect a single client:**
```bash
python -m v4.distributed.cli --server http://hades:8765 --token hypertower clients <client_id>
```

**View job queue:**
```bash
python -m v4.distributed.cli --server http://hades:8765 --token hypertower jobs
```

**Filter by state:**
```bash
python -m v4.distributed.cli --server http://hades:8765 --token hypertower jobs --state pending
python -m v4.distributed.cli --server http://hades:8765 --token hypertower jobs --state running
python -m v4.distributed.cli --server http://hades:8765 --token hypertower jobs --state failed
```

---

## Submitting Jobs

**Batch dispatch (dry run first):**
```bash
python -m v4.distributed.batch_dispatch \
    --server http://hades:8765 --token hypertower \
    --config v4/configs/ensemble_fused.json \
    --batch v4/scripts/experiments/fusion_dim_sweep.json \
    --dry-run
```

**Live submit:**
```bash
python -m v4.distributed.batch_dispatch \
    --server http://hades:8765 --token hypertower \
    --config v4/configs/ensemble_fused.json \
    --batch v4/scripts/experiments/fusion_dim_sweep.json
```

**Fewer reps (e.g. quick test):**
```bash
python -m v4.distributed.batch_dispatch \
    --server http://hades:8765 --token hypertower \
    --config v4/configs/ensemble_fused.json \
    --batch v4/scripts/experiments/fusion_dim_sweep.json \
    --reps 3
```

---

## Queue Management

**Clear failed jobs:**
```bash
python -m v4.distributed.cli --server http://hades:8765 --token hypertower clear --states failed
```

**Clear running jobs (orphan cleanup):**
```bash
python -m v4.distributed.cli --server http://hades:8765 --token hypertower clear --states running
```

**Clear all jobs:**
```bash
python -m v4.distributed.cli --server http://hades:8765 --token hypertower clear --all
```

**Clear by run name:**
```bash
python -m v4.distributed.cli --server http://hades:8765 --token hypertower clear --run-name experiments/fusion_dim_sweep/dim256
```

**Cancel a specific job:**
```bash
python -m v4.distributed.cli --server http://hades:8765 --token hypertower cancel <job_id>
```
