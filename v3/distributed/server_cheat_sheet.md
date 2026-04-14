# Distributed Server Cheat Sheet

All commands assume server is running on hades at port 8765.

## Start Server
```bash
python -m v3.distributed.server --token hypertower
```
Run inside tmux so it survives disconnects:
```bash
tmux new -s htserver
python -m v3.distributed.server --token hypertower
# Ctrl-B D to detach
tmux attach -t htserver   # reattach later
```

## Start Clients

**Hades (server-local, no sync):**
```bash
python -m v3.distributed.client \
    --server http://hades:8765 --token hypertower \
    --server-ssh ignored --server-path ignored \
    --local-path /home/rpotter/hypertower \
    --no-sync
```

**Apollo (remote client):**
```bash
python -m v3.distributed.client \
    --server http://hades:8765 --token hypertower \
    --server-ssh rpotter@hades \
    --server-path /home/rpotter/hypertower \
    --local-path /home/odin/hypertower
```

---

## Monitoring

**Live client monitor (refreshes every 5s):**
```bash
python -m v3.distributed.cli --server http://hades:8765 --token hypertower clients --watch
```

**Faster refresh:**
```bash
python -m v3.distributed.cli --server http://hades:8765 --token hypertower clients --watch --interval 2
```

**Inspect a single client:**
```bash
python -m v3.distributed.cli --server http://hades:8765 --token hypertower clients <client_id>
```

**View job queue:**
```bash
python -m v3.distributed.cli --server http://hades:8765 --token hypertower jobs
```

**Filter by state:**
```bash
python -m v3.distributed.cli --server http://hades:8765 --token hypertower jobs --state pending
python -m v3.distributed.cli --server http://hades:8765 --token hypertower jobs --state running
python -m v3.distributed.cli --server http://hades:8765 --token hypertower jobs --state failed
```

---

## Submitting Jobs

**Phase 3 main grid:**
```bash
python -m v3.scripts.main.phase3.dispatch_phase3 \
    --server http://hades:8765 --token hypertower
```

**Dry run (check what would be submitted):**
```bash
python -m v3.scripts.main.phase3.dispatch_phase3 \
    --server http://hades:8765 --token hypertower --dry-run
```

**Other grids:**
```bash
python -m v3.scripts.main.phase3.dispatch_phase3 \
    --server http://hades:8765 --token hypertower \
    --grid v3/scripts/main/phase3/epoch_grid.json

python -m v3.scripts.main.phase3.dispatch_phase3 \
    --server http://hades:8765 --token hypertower \
    --grid v3/scripts/main/phase3/phase35_grid.json
```

---

## Queue Management

**Clear failed jobs:**
```bash
python -m v3.distributed.cli --server http://hades:8765 --token hypertower clear --states failed
```

**Clear running jobs (orphan cleanup):**
```bash
python -m v3.distributed.cli --server http://hades:8765 --token hypertower clear --states running
```

**Clear all jobs:**
```bash
python -m v3.distributed.cli --server http://hades:8765 --token hypertower clear --all
```

**Clear by run name:**
```bash
python -m v3.distributed.cli --server http://hades:8765 --token hypertower clear --run-name phase3/baseline
```

**Cancel a specific job:**
```bash
python -m v3.distributed.cli --server http://hades:8765 --token hypertower cancel <job_id>
```
