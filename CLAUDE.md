# Manhead Merch Sales Prediction — Operations Guide

## What this is

A merch sales prediction system for Manhead (concert merchandise company). It predicts how much merch (t-shirts, hoodies, etc.) will sell at upcoming shows based on artist, venue, attendance, weather, and other features. Uses an ExtraTreesRegressor trained on ~45K historical rows.

## Live infrastructure

| Component | Location | URL / IP |
|-----------|----------|----------|
| Streamlit UI | Streamlit Community Cloud (free) | `manheadashlingpredictions26.streamlit.app` |
| Flask API + Model | DigitalOcean Droplet (16GB, 4 vCPU, nyc3) | `45.55.126.129` (reserved IP) |
| Model artifacts (backup) | DO Spaces | `mh-forecast.nyc3.cdn.digitaloceanspaces.com` |
| Source code | GitHub | `github.com/eugeneleychenko/manhead_ashling_predictions_26` |

## Credentials

- **API Key** (for Flask endpoints): `FgdQSbg0iV--q_NZ3NPJtV_QomPC9nc_D0OUVCWTWV8`
- **Streamlit secrets** (Settings → Secrets on Streamlit Cloud):
  ```toml
  PREDICTION_API_BASE_URL = "https://45.55.126.129"
  API_KEY = "FgdQSbg0iV--q_NZ3NPJtV_QomPC9nc_D0OUVCWTWV8"
  ```
- **SSH**: `ssh root@45.55.126.129` (uses your Mac's SSH key "Macbook Air")
- **DO account**: `eugene@triplespeed.co` via `doctl`

## API endpoints (on DO droplet)

All endpoints except `/health` require `X-API-Key` header.

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check (no auth) |
| `/status` | GET | Model artifacts + metrics |
| `/api/predict` | POST | Upload CSV, get predictions |
| `/api/retrain` | POST | Upload sales/tour CSVs, trigger retrain |
| `/api/retrain/status` | GET | Poll retrain progress |
| `/api/artist-metadata` | GET | List current artist metadata |
| `/api/artist-metadata` | POST | Add new artist entries |

## How the system works

### Prediction flow
1. Manhead staff opens Streamlit → uploads formatted CSV in Step 4
2. Streamlit sends CSV to `POST /api/predict` on DO droplet
3. Flask loads model from memory, runs prediction, returns results
4. Staff downloads predictions CSV

### Retrain flow (self-service, no SSH needed)
1. Staff optionally adds new artist metadata in Step 1
2. Staff uploads new sales + tour CSVs in Step 6, clicks "Start Retrain"
3. Streamlit sends files to `POST /api/retrain` on DO droplet
4. Droplet runs consolidation (Step 2) → training (Step 3) → validates R²
5. If R² doesn't drop >10%: new model promoted, Flask hot-reloads
6. If R² drops >10%: old model kept, retrain marked as failed
7. Streamlit polls status and shows old vs new metrics

See `docs/retrain-flow.md` for the full Mermaid sequence diagram.

## Common operations

### Check if the droplet is healthy
```bash
curl -k https://45.55.126.129/health
```

### Check model status + metrics
```bash
curl -sk -H "X-API-Key: FgdQSbg0iV--q_NZ3NPJtV_QomPC9nc_D0OUVCWTWV8" https://45.55.126.129/status
```

### Run a prediction via curl
```bash
curl -sk -H "X-API-Key: FgdQSbg0iV--q_NZ3NPJtV_QomPC9nc_D0OUVCWTWV8" \
  -F "csv_file=@your_formatted_file.csv" \
  https://45.55.126.129/api/predict
```

### SSH into the droplet
```bash
ssh root@45.55.126.129
```

### View Flask logs
```bash
ssh root@45.55.126.129 "journalctl -u manhead-flask --no-pager | tail -50"
```

### Restart Flask service
```bash
ssh root@45.55.126.129 "systemctl restart manhead-flask"
```

### Deploy code changes
After committing + pushing to GitHub:
```bash
cd Manhead_Handoff
bash scripts/deploy.sh 45.55.126.129
```
This rsyncs the repo (excluding the 2.9GB model) and re-runs server setup.

### Rebuild from scratch
```bash
bash scripts/provision-droplet.sh    # Creates new droplet + reserved IP
bash scripts/deploy.sh <NEW_IP>      # Syncs code + downloads model from DO Spaces
```

## File layout on the droplet

```
/opt/manhead/
├── .env                          # API key
├── .venv/                        # Python 3.13 venv
├── .retrain_state.json           # Current retrain job status (shared across gunicorn workers)
├── paths_config.txt              # → cloud version with /opt/manhead/ absolute paths
├── Python_scripts/               # All application code
├── Flask/
│   ├── model_retrained.joblib    # 2.9GB model (downloaded from DO Spaces)
│   ├── robust_scaler_retrained.joblib
│   ├── label_encoder_retrained.joblib
│   ├── last_train_metrics.json
│   ├── .staging/                 # Temp dir for retrain artifacts before promotion
│   └── templates/index.html
├── CSVs/
│   ├── add_sales_reports_files_here/   # Raw sales CSVs
│   ├── add_tour_summary_files_here/    # Raw tour CSVs
│   ├── merged_files/                   # Training dataset + consolidated outputs
│   ├── features/                       # artist_metadata.csv, spotify, venue/holiday
│   └── archive/                        # Archived CSVs from past retrains
└── outputs/
```

## Server stack on the droplet

- **Ubuntu 24.04** with Python 3.13 (deadsnakes PPA)
- **Gunicorn** (2 workers, 5min timeout) bound to `127.0.0.1:5001`
- **Nginx** reverse proxy on ports 80/443, self-signed HTTPS cert
- **systemd** service: `manhead-flask.service`
- Model downloaded from DO Spaces CDN during setup (~2.9GB)

## Two requirements files

- `requirements.txt` — minimal (streamlit, pandas, requests) for Streamlit Community Cloud
- `requirements-server.txt` — full ML stack (scikit-learn, xgboost, lightgbm, etc.) for the DO droplet

This split exists because Streamlit Cloud runs Python 3.14 which can't build `pyarrow==21.0.0` from source.

## Cost

| Item | Cost |
|------|------|
| DO Droplet (s-4vcpu-16gb-amd) | ~$84/mo |
| DO Spaces (artifact backup) | ~$5/mo |
| Streamlit Cloud | $0 |
| Replit (cancelled) | $0 |
| **Total** | **~$89/mo** |
