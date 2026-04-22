Manhead Handoff — Merch Sales Prediction (Streamlit + Flask)

This package runs a 5 step workflow:
1) Add/update artist metadata
2) Consolidate raw sales + tour files into an enriched snapshot and append to master dataset
3) Train/retrain model artifacts
4) Run predictions via Flask API
5) Compute revenue per head ($/head) from the prediction (step 4) output


FOLDER LAYOUT (do not rename folders)

- CSVs/ — inputs + outputs
- CSVs/add_sales_reports_files_here/ — place raw sales report CSVs here
- CSVs/add_tour_summary_files_here/ — place raw tour summary CSVs here
- CSVs/merged_files/ — pipeline outputs + master datasets
- CSVs/features/ — feature CSVs (artist, spotify listeners, venue/holiday)
- debug_info/ — logs used by the workflow, keep this folder as-is (do not delete or rename)
- Flask/ — Flask artifacts + templates (model joblibs, metrics JSON, templates)
- outputs/ — step outputs + Streamlit download copies + Flask downloads
- Python_scripts/ — consolidation + step scripts (includes Streamlit app + Flask runner script)
- paths_config.txt — path configuration (relative to the folder where this file lives)
- requirements.txt — Has all the python dependencies

PREREQUISITES

- Windows PowerShell
- Python 3.13.x (tested with 3.13.9)
- Google Chrome installed (needed for Spotify scraping via Selenium)


SETUP (create venv + install dependencies)

Open PowerShell in this folder (the folder containing paths_config.txt), then run:

python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt


CONFIGURE PATHS

Edit paths_config.txt (in this folder).
All values should remain relative paths (unless explicitly required otherwise).

Required keys (must exist in paths_config.txt):

Consolidation inputs/outputs:
- path_sales_pattern
- path_tour_pattern
- merch_path_out
- tour_path_out
- merged_path
- final_out
- consolidation_stats_json
- master_training_dataset
- master_training_dataset_backup

Feature files:
- spotify_csv
- artist_meta_csv
- venue_meta_csv
- coords_path

Caching:
- geocoding_cache_name
- weather_cache_name

Flask (artifacts + runtime + port):
- flask_artifacts_dir
- model_joblib
- scaler_joblib
- encoder_joblib
- last_train_metrics_json
- flask_templates_dir
- flask_downloads_dir
- flask_logs_dir
- flask_port

Step 5 ($/head):
- revph_script_path
- revph_output_csv
- revph_summary_json

Script paths (called by Streamlit):
- consolidation_script_path
- train_model_script_path


ADD INPUT FILES (used by Step 2)

Place raw input files here:
- Sales reports: CSVs/add_sales_reports_files_here/
- Tour summary reports: CSVs/add_tour_summary_files_here/

Do not change folder names. The consolidation step reads these using the patterns in paths_config.txt.


START FLASK PREDICTION SERVICE (required for Step 4)

Open Terminal 1 (PowerShell) in this folder:

.\.venv\Scripts\activate
python Python_scripts\run_all_products_sales_by_size_prediction_app.py

Keep this terminal running.

If a different port is needed:
1) Update flask_port in paths_config.txt
2) Restart Flask


START STREAMLIT APP (Steps 1–5 UI)

Open Terminal 2 (PowerShell) in this folder:

.\.venv\Scripts\activate
streamlit run Python_scripts\streamlit_app.py


USING THE STREAMLIT STEPS

Step 1: Add New Artist Data
- Adds rows into the artist metadata CSV configured by artist_meta_csv in paths_config.txt

Step 2: Format and Consolidate Data
- Runs the consolidation script configured by consolidation_script_path
- Writes:
  - Consolidated snapshot CSV (final_out)
  - Stats JSON (consolidation_stats_json)
- Appends consolidated snapshot into master dataset (master_training_dataset)

Step 3: Train / Retrain Model
- Runs the training script configured by train_model_script_path
- Reads master_training_dataset
- Writes model artifacts into flask_artifacts_dir

Note: Step 4 and Step 5 can also accept CSVs exported from Manhead’s existing Streamlit formatting flow (your current production tool, not this handoff app). If you want to use a raw file like the 'Tour-Forecast' CSV as input to Step 4 or Step 5, first run it through Manhead’s formatting step to standardize columns/types, then upload the formatted CSV into Step 4/Step 5.

Step 4: Run Predictions
- Uploads a CSV and calls the Flask API:
  http://localhost:<flask_port>/api/predict

Step 5: Calculate Revenue Per Head ($/head)
- Runs revph_script_path
- Writes:
  - revph_output_csv
  - revph_summary_json


OUTPUTS (where to find files)

Important note about downloads:
- Files downloaded using Streamlit "Download" buttons are saved to the browser’s default Downloads folder.
- Streamlit also saves a copy of each downloadable output into the repo outputs/ folder (handled by streamlit_app.py).
- The pipeline outputs listed below are written to disk at the locations configured in paths_config.txt.

Consolidation outputs (Step 2):
- Consolidated snapshot CSV: path in final_out
- Consolidation stats JSON: path in consolidation_stats_json
- Master dataset: path in master_training_dataset
- Master backup dataset: path in master_training_dataset_backup

Model artifacts (Step 3):
- Written to the paths configured in paths_config.txt (flask_artifacts_dir, model_joblib, scaler_joblib, encoder_joblib, last_train_metrics_json)

Flask downloads/logs:
- Downloads written to flask_downloads_dir (e.g., outputs/flask_downloads)
- Logs written to flask_logs_dir (e.g., logs/)

Step 5 outputs:
- Output CSV: path in revph_output_csv
- Summary JSON: path in revph_summary_json


TROUBLESHOOTING

Step 4 fails: Flask API not reachable
- Confirm Flask is running in Terminal 1
- Confirm flask_port in paths_config.txt matches the port Flask is using
- If you retrain the model in Step 3, restart Flask so it reloads updated artifacts

Consolidation fails during Spotify update
- Spotify scraping uses Selenium + Chrome
- Ensure Google Chrome is installed
- Ensure dependencies from requirements.txt are installed
- If scraping fails, consolidation may continue using the existing spotify_csv when possible

PermissionError writing CSVs
- If a CSV is open in Excel (or another program), close it and run again