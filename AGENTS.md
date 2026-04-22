# Repository Guidelines

## Project Structure

```
Manhead_Handoff/
├── Python_scripts/          # All application code
│   ├── streamlit_app.py     # Streamlit UI (Steps 1–6)
│   ├── predict_all_products_sales_by_size_app.py  # Flask API (predict, retrain, health)
│   ├── run_all_products_sales_by_size_prediction_app.py  # Flask entrypoint
│   ├── consolidate_pipeline.py  # Step 2: merge sales + tour CSVs
│   ├── train_model.py       # Step 3: train ExtraTreesRegressor
│   ├── retrain_worker.py    # Retrain subprocess with validation gate
│   ├── revenue_per_head.py  # Step 5: $/head calculation
│   ├── spotify.py           # Spotify scraping (decoupled)
│   └── audit_logger.py      # JSONL + CSV audit logging
├── Flask/                   # Model artifacts (joblib) + templates
├── CSVs/                    # Input CSVs, merged outputs, features, master dataset
├── scripts/                 # DO droplet provisioning + deploy scripts
├── systemd/                 # manhead-flask.service unit file
├── nginx/                   # Reverse proxy config
├── paths_config.txt         # Local relative paths (development)
├── paths_config_cloud.txt   # Absolute /opt/manhead/ paths (production)
└── .env                     # API key (not committed)
```

## Build & Run Commands

| Command | Purpose |
|---------|---------|
| `python -m venv .venv && .venv/bin/pip install -r requirements.txt` | Create venv + install deps |
| `python Python_scripts/run_all_products_sales_by_size_prediction_app.py` | Start Flask API locally |
| `streamlit run Python_scripts/streamlit_app.py` | Start Streamlit UI locally |
| `bash scripts/provision-droplet.sh` | Create DO droplet + reserved IP |
| `bash scripts/deploy.sh <IP>` | Rsync repo + run server setup |

## Architecture Notes

- **Flask API** serves predictions on port 5001 (production) or 5000 (local), behind nginx with HTTPS.
- **API key auth** is enforced via `X-API-Key` header on all endpoints except `/health`.
- **Retrain** runs as a subprocess (`retrain_worker.py`), writes artifacts to `Flask/.staging/`, validates R² (>10% drop = blocked), then promotes via atomic `os.replace()`. Flask hot-reloads via a `.retrain_complete` signal file.
- **paths_config.txt** is the single source of truth for all file paths. Every script resolves paths by walking upward to find this file.

## Coding Style

- Python 3.13. No type annotations on existing code unless modifying that function.
- 4-space indentation. No trailing whitespace.
- Use `os.path` for paths (not `pathlib`) — matches existing codebase.
- Atomic file writes: use `tempfile` + `os.replace()` for artifacts.
- Logging: use `audit_logger.log_event()` for step-level events, `app.logger` for Flask.

## Commit Guidelines

- Concise imperative subject line (e.g., "Add retrain validation gate").
- Body explains *why*, not *what*.
- Do not commit `.env`, `Flask/model_retrained.joblib`, or `.venv/`.

## Key Constraints

- The model joblib is ~2.9 GB — excluded from git, downloaded from DO Spaces during deploy.
- `requirements.txt` is UTF-16 encoded; pip handles this fine, but text editors may display oddly.
- Consolidation pipeline imports `spotify.py` — set `SKIP_SPOTIFY=1` env var to bypass scraping.
