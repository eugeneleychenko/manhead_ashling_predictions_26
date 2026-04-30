# Retrain Pipeline — Testing Notes & Bug Fixes

## What We Were Testing

End-to-end retrain flow via Step 6 in Streamlit:
1. Upload new sales + tour CSVs (e.g., Death Stranding show data)
2. Click "Start Retrain" → files sent to DO droplet
3. Droplet runs: consolidation (Step 2) → training (Step 3) → R² validation → model promotion
4. Streamlit polls for status, shows results when done

---

## Issues Found & Fixes

### 1. Consolidation silently skipped during retrain ⚠️ CRITICAL

**Symptom:** Retrain reported "completed" with good metrics, but uploaded data (Death Stranding) was never added to the master training dataset.

**Root cause:** `retrain_worker.py` called `consolidate_pipeline.main(config_path, skip_spotify=True)` but `main()` accepts **no arguments**. This threw a `TypeError`, which was caught and silently swallowed. Training then ran on the **unchanged** master dataset.

**Fix:** Switched to subprocess invocation instead of direct import:
```python
# Before (broken):
consolidate_pipeline.main(config_path, skip_spotify=True)  # TypeError

# After (working):
subprocess.run([sys.executable, consolidation_script], cwd=repo_root, env={"SKIP_SPOTIFY": "1"})
```

**Commit:** `1c325c8`

---

### 2. Fragile full-row deduplication

**Symptom:** Master dataset shrank from 45,803 → 42,841 rows after retrain. New data should only add rows, never remove them.

**Root cause:** `consolidate_pipeline.py` called `drop_duplicates()` on **all 24 columns**, including floats like `temperature_daily_mean`, `rain`, `snowfall`. Weather values fetched at different times produce slightly different floats, so:
- Same show → different weather floats → treated as unique → duplicates accumulate
- Re-run with consistent weather → duplicates collapse → row count drops

**Fix:** Changed to deduplicate on stable identity columns only:
```python
DEDUPE_COLS = ["artistName", "showDate", "venue name", "venue city",
               "merch category", "productType", "product size"]
df_combined = df_combined.drop_duplicates(subset=DEDUPE_COLS, keep="first")
```

**Commit:** `956419a`

---

### 3. Hardcoded artist name map in consolidation

**Symptom:** Only 5 artists (Air Supply, Deftones, Garbage, Jelly Roll, Lainey Wilson) could be consolidated. Any new artist uploaded via retrain would be ignored unless manually added to a dict in the code.

**Root cause:** `consolidate_pipeline.py` has a hardcoded `artist_map` dict that maps filename slugs to display names:
```python
artist_map = {
    "air-supply": "Air Supply",
    "deftones-mh": "Deftones (MH)",
    # ... only 5 entries
}
```

There IS a `prettify_artist_name()` fallback that converts `death-stranding` → `Death Stranding` via title-casing, so this technically works for simple names. But for names with suffixes like `(MH)`, only the hardcoded entries work.

**Status:** Not yet fixed — the fallback works for the Death Stranding test case but may need a more robust solution for `(MH)` suffixed artists.

---

### 4. Retrain status polling timeout

**Symptom:** User saw `HTTPSConnectionPool Read timed out` error in Step 6 during retrain polling.

**Root cause:** Polling timeout was set to 10 seconds. During active retrain, the droplet's CPU is at 20%+ doing model training, and the status endpoint sometimes takes longer to respond.

**Fix:** Increased polling timeout from 10s → 30s.

**Commit:** `52457ac`

---

### 5. Polling state lost on page refresh

**Symptom:** User kicked off retrain, page refreshed, and the completed status / metrics / balloons never appeared.

**Root cause:** Streamlit session state (`retrain_polling=True`) is lost on page refresh. The completed results are only shown while the polling loop is active — if it stops, there's no way to see them.

**Status:** Partially addressed — added a "Check Retrain Status" button that queries the API on demand. Full fix (auto-show last result on page load) still needed.

---

### 6. OOM kills during model hot-reload

**Symptom:** After retrain completes, gunicorn workers get SIGKILL'd with `oom-kill`.

**Root cause:** Two gunicorn workers each hold the 2.9GB model in memory (~6GB total). When both try to reload the new model simultaneously, peak memory exceeds the 16GB droplet limit.

**Status:** Not yet fixed. The systemd auto-restart recovers the service within seconds, but it's not clean. Possible fixes:
- Reduce to 1 gunicorn worker (saves ~3GB RAM)
- Stagger the reload (reload one worker at a time)
- Use `preload_app=True` with shared memory

---

### 7. Deploy during retrain kills the worker process

**Symptom:** Retrain was running, we deployed new code, gunicorn restarted, retrain subprocess died. State file stuck at `status: running` forever.

**Root cause:** The retrain worker runs as a subprocess of gunicorn. When gunicorn restarts, the subprocess is killed but the `.retrain_state.json` file is never updated to reflect the failure.

**Status:** Manually reset the state file. Should add a PID liveness check — if `.retrain_state.json` says "running" but the PID is dead, auto-reset to "failed".

---

## Current State

| Component | Status |
|-----------|--------|
| Consolidation during retrain | **Fixed** — uses subprocess now |
| Deduplication | **Fixed** — stable identity columns |
| Polling timeout | **Fixed** — 30s |
| Master dataset preview in UI | **Working** — tail + download after retrain |
| OOM on reload | **Known issue** — auto-recovers via systemd |
| Stuck state after killed retrain | **Known issue** — needs PID liveness check |
| Polling state on refresh | **Known issue** — needs persistent status display |

## How to Verify the Consolidation Fix

1. Go to Step 6 on Streamlit
2. Upload a Death Stranding sales + tour CSV
3. Click "Start Retrain"
4. After completion, check master dataset:
```bash
curl -sk -H "X-API-Key: <key>" "https://45.55.126.129/api/master-dataset/tail?n=50" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for row in data['data']:
    if 'Death' in row.get('artistName', ''):
        print(row['artistName'], row['showDate'], row['venue city'])
print(f'Total rows: {data[\"total_rows\"]}')"
```
