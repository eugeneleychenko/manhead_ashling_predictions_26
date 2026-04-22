import sys
import os
import json
import uuid
import logging
import threading
import subprocess
from functools import wraps
from joblib import load
from io import BytesIO
import numpy as np
import pandas as pd
import datetime as dt
from flask import Flask, render_template, request, send_from_directory, url_for, jsonify


try:
    from audit_logger import log_event, sha256_bytes
except Exception:
    log_event = None
    sha256_bytes = None

def _safe_log(step: str, status: str, details: dict):
    try:
        if callable(log_event):
            log_event(step=step, status=status, details=details)
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def find_paths_config(start_dir: str, filename: str = "paths_config.txt") -> str:
    cur = os.path.abspath(start_dir)
    while True:
        cand = os.path.join(cur, filename)
        if os.path.exists(cand):
            return cand
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    raise FileNotFoundError(f"Could not find {filename} by searching upward from: {start_dir}")

def load_paths(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"paths_config.txt not found at: {config_path}")
    base_dir = os.path.dirname(os.path.abspath(config_path))
    paths = {}
    with open(config_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key and value:
                if (
                    (not os.path.isabs(value))
                    and (not value.lower().startswith(("http://", "https://")))
                    and (not value.isdigit())
                ):
                    value = os.path.normpath(os.path.join(base_dir, value))
                paths[key] = value
    return paths

config_path = find_paths_config(BASE_DIR)
paths = load_paths(config_path)

template_dir = paths.get("flask_templates_dir")
app = Flask(__name__, template_folder=template_dir if template_dir else None)
app.logger.setLevel(logging.DEBUG)

# ── API Key Auth ──
API_KEY = os.environ.get("MANHEAD_API_KEY", "")
# Also try loading from .env file next to paths_config.txt
if not API_KEY:
    _env_path = os.path.join(os.path.dirname(os.path.abspath(config_path)), ".env")
    if os.path.exists(_env_path):
        with open(_env_path) as _ef:
            for _line in _ef:
                _line = _line.strip()
                if _line.startswith("MANHEAD_API_KEY="):
                    API_KEY = _line.split("=", 1)[1].strip()

# Paths that do NOT require an API key
PUBLIC_PATHS = {"/health"}

@app.before_request
def check_api_key():
    if not API_KEY:
        return  # Auth disabled if no key configured
    if request.path in PUBLIC_PATHS:
        return
    key = request.headers.get("X-API-Key", "")
    if key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

artifact_dir = paths.get("flask_artifacts_dir", BASE_DIR)

try:
    model = load(os.path.join(artifact_dir, "model_retrained.joblib"))
    scaler = load(os.path.join(artifact_dir, "robust_scaler_retrained.joblib"))
    encoder = load(os.path.join(artifact_dir, "label_encoder_retrained.joblib"))
    app.logger.info("Model, scaler and encoder loaded successfully")
except FileNotFoundError as e:
    print("Model/scaler/encoder file not found:", e)
    sys.exit(10)

input_categorical_features = [
    "artistName",
    "Genre",
    "HolidayStatus",
    "venue name",
    "venue city",
    "venue state",
    "venue country",
    "venue postalCode",
    "merch category",
    "productType",
    "product size",
]

input_numerical_features = [
    "Show Day",
    "Show Month",
    "Day of Week Num",
    "attendance",
    "product price",
    "temperature_daily_mean",
    "rain",
    "snowfall",
    "spotifyMonthlyListeners",
    "Instagram",
    "venue capacity",
]

output_features = [
    "artistName",
    "Genre",
    "showDate",
    "HolidayStatus",
    "venue name",
    "venue city",
    "venue state",
    "venue country",
    "venue postalCode",
    "merch category",
    "productType",
    "product size",
    "Show Day",
    "Show Month",
    "Day of Week Num",
    "attendance",
    "product price",
    "temperature_daily_mean",
    "spotifyMonthlyListeners",
    "Instagram",
    "venue capacity",
    "predicted_sales_quantity",
    "%_item_sales_per_category",
]

def build_model_inputs(df, route_tag="html"):
    if "showDate" not in df.columns:
        raise KeyError("Missing required column 'showDate' in input data.")

    df["showDate"] = pd.to_datetime(df["showDate"])
    df["Show Day"] = df["showDate"].dt.day
    df["Show Month"] = df["showDate"].dt.month
    df["Day of Week Num"] = df["showDate"].dt.weekday + 1

    present_num_features = [f for f in input_numerical_features if f in df.columns]
    df_num = df[present_num_features].copy()

    for feature in present_num_features:
        if isinstance(scaler, dict) and feature in scaler:
            df_num[[feature]] = scaler[feature].transform(df_num[[feature]])
        else:
            app.logger.warning(f"[{route_tag}] Scaler for feature '{feature}' not found; leaving unscaled.")

    if not isinstance(encoder, dict):
        raise TypeError("Loaded encoder object is not a dict; cannot map per-feature encoders.")

    encoded_cat_parts = []

    for feature in input_categorical_features:
        if feature not in df.columns:
            app.logger.warning(f"[{route_tag}] Categorical feature '{feature}' not in input data; skipping.")
            continue
        if feature not in encoder:
            app.logger.warning(f"[{route_tag}] Encoder for feature '{feature}' not found; dropping from model input.")
            continue

        series = (
            df[feature]
            .astype(str)
            .str.strip()
            .str.replace("\xa0", " ")
            .str.lower()
        )
        enc = encoder[feature]
        series = series.apply(lambda x: x if x in enc.classes_ else "unknown_category")
        encoded_vals = enc.transform(series)

        encoded_cat_parts.append(pd.DataFrame({feature: encoded_vals}, index=df.index))

    if encoded_cat_parts:
        df_cat_encoded = pd.concat(encoded_cat_parts, axis=1)
    else:
        df_cat_encoded = pd.DataFrame(index=df_num.index)
        app.logger.warning(f"[{route_tag}] No categorical features encoded.")

    df_model_input = pd.concat([df_num, df_cat_encoded], axis=1)
    df_model_input.dropna(inplace=True)

    if hasattr(model, "feature_names_in_"):
        model_features = list(model.feature_names_in_)
        df_model_input = df_model_input.reindex(columns=model_features, fill_value=0)

    df_aligned = df.loc[df_model_input.index].copy()
    return df_model_input, df_aligned

def _maybe_estimate_shows(df_out: pd.DataFrame):
    key = ["artistName", "venue name", "venue city", "venue state", "showDate"]
    if all(c in df_out.columns for c in key):
        try:
            return int(df_out[key].drop_duplicates().shape[0])
        except Exception:
            return None
    return None

@app.route("/", methods=["GET", "POST"])
def index():
    timestamp = dt.datetime.now().strftime("%m_%d_%Y-%I_%M_%S_%p")

    logs_dir = paths.get("flask_logs_dir", BASE_DIR)
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"log_predict_all_products_sales_by_size_app_{timestamp}.log")

    logging.basicConfig(
        filename=log_path,
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s",
    )
    logging.info("Model, scaler and encoder loaded successfully")

    if request.method == "POST" and "csv_file" in request.files:
        file = request.files["csv_file"]
        filename = getattr(file, "filename", "") or ""

        try:
            file_bytes = file.read()
            in_hash = sha256_bytes(file_bytes) if callable(sha256_bytes) else ""
            _safe_log(
                step="flask_predict_html",
                status="start",
                details={"input_file": filename, "input_hash": in_hash, "bytes": int(len(file_bytes))},
            )

            df = pd.read_csv(BytesIO(file_bytes), encoding="utf-8-sig")

            df_model_input, df = build_model_inputs(df, route_tag="html")
            if df_model_input.empty:
                _safe_log(
                    step="flask_predict_html",
                    status="warning",
                    details={"input_file": filename, "input_hash": in_hash, "warning": "No valid rows after cleaning"},
                )
                return render_template("index.html", error_message="No valid rows in the input after cleaning. Please check your data.")

            predictions = model.predict(df_model_input)
            df["predicted_sales_quantity"] = np.round(predictions).astype(int)

            group_cols = ["artistName", "showDate", "productType"]
            missing_group = [c for c in group_cols if c not in df.columns]
            if missing_group:
                _safe_log(
                    step="flask_predict_html",
                    status="error",
                    details={"input_file": filename, "input_hash": in_hash, "error": f"Missing group cols: {missing_group}"},
                )
                return render_template("index.html", error_message=f"Missing columns for percentage calculation: {missing_group}")

            df["%_item_sales_per_category"] = df.groupby(group_cols)["predicted_sales_quantity"].transform(
                lambda x: round((x / x.sum()) * 100, 2)
            )

            if "showDate" in df.columns:
                df["showDate"] = pd.to_datetime(df["showDate"], errors="coerce").dt.strftime("%Y-%m-%d")

            present_output_cols = [c for c in output_features if c in df.columns]
            output_df = df[present_output_cols]

            downloads_dir = paths.get("flask_downloads_dir", os.path.join(BASE_DIR, "downloads"))
            os.makedirs(downloads_dir, exist_ok=True)

            csv_filename = f"predicted_sales_by_size_all_products_{timestamp}.csv"
            file_path = os.path.join(downloads_dir, csv_filename)
            output_df.to_csv(file_path, index=False)

            _safe_log(
                step="flask_predict_html",
                status="success",
                details={
                    "input_file": filename,
                    "input_hash": in_hash,
                    "rows_in": int(len(df)),
                    "cols_in": list(df.columns),
                    "rows_out": int(output_df.shape[0]),
                    "cols_out": list(output_df.columns),
                    "shows_estimate": _maybe_estimate_shows(output_df),
                    "download_csv": file_path,
                },
            )

            download_url = url_for("download_file", filename=csv_filename)
            return render_template("index.html", table=output_df.to_html(), download_url=download_url)

        except Exception as e:
            logging.error(f"Error during prediction process (HTML): {e}", exc_info=True)
            _safe_log(
                step="flask_predict_html",
                status="error",
                details={"input_file": filename, "error": str(e)},
            )
            return render_template("index.html", error_message="An error occurred during prediction. Please check your input data.")

    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    filename = ""
    in_hash = ""
    try:
        if "csv_file" not in request.files:
            return jsonify({"error": "No file part 'csv_file' in the request"}), 400

        file = request.files["csv_file"]
        filename = getattr(file, "filename", "") or ""

        file_bytes = file.read()
        in_hash = sha256_bytes(file_bytes) if callable(sha256_bytes) else ""

        _safe_log(
            step="flask_predict_api",
            status="start",
            details={"input_file": filename, "input_hash": in_hash, "bytes": int(len(file_bytes))},
        )

        df = pd.read_csv(BytesIO(file_bytes), encoding="utf-8-sig")

        df_model_input, df = build_model_inputs(df, route_tag="api")
        if df_model_input.empty:
            _safe_log(
                step="flask_predict_api",
                status="warning",
                details={"input_file": filename, "input_hash": in_hash, "warning": "No rows left after cleaning"},
            )
            return jsonify({"error": "No rows left after cleaning; nothing to predict."}), 400

        predictions = model.predict(df_model_input)
        df["predicted_sales_quantity"] = np.round(predictions).astype(int)

        group_cols = ["artistName", "showDate", "productType"]
        missing_group = [c for c in group_cols if c not in df.columns]
        if missing_group:
            _safe_log(
                step="flask_predict_api",
                status="error",
                details={"input_file": filename, "input_hash": in_hash, "error": f"Missing columns for percentage calculation: {missing_group}"},
            )
            return jsonify({"error": f"Missing columns for percentage calculation: {missing_group}"}), 400

        df["%_item_sales_per_category"] = df.groupby(group_cols)["predicted_sales_quantity"].transform(
            lambda x: round((x / x.sum()) * 100, 2)
        )

        if "showDate" in df.columns:
            df["showDate"] = pd.to_datetime(df["showDate"], errors="coerce").dt.strftime("%Y-%m-%d")

        present_output_cols = [c for c in output_features if c in df.columns]
        output_df = df[present_output_cols]

        _safe_log(
            step="flask_predict_api",
            status="success",
            details={
                "input_file": filename,
                "input_hash": in_hash,
                "rows_in": int(len(df)),
                "cols_in": list(df.columns),
                "rows_out": int(output_df.shape[0]),
                "cols_out": list(output_df.columns),
                "shows_estimate": _maybe_estimate_shows(output_df),
            },
        )

        return jsonify({"data": output_df.to_dict(orient="records")}), 200

    except Exception as e:
        app.logger.error(f"/api/predict error: {e}", exc_info=True)
        _safe_log(
            step="flask_predict_api",
            status="error",
            details={"input_file": filename, "input_hash": in_hash, "error": str(e)},
        )
        return jsonify({"error": str(e)}), 500

@app.route("/downloads/<filename>")
def download_file(filename):
    downloads_dir = paths.get("flask_downloads_dir", os.path.join(BASE_DIR, "downloads"))
    try:
        return send_from_directory(downloads_dir, filename, as_attachment=True)
    except FileNotFoundError:
        logging.error(f"File not found for download: {filename}")
        return "File not found", 404


# ── Health & Status ──

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
    }), 200

@app.route("/status", methods=["GET"])
def status():
    model_path = paths.get("model_joblib", os.path.join(artifact_dir, "model_retrained.joblib"))
    scaler_path = paths.get("scaler_joblib", os.path.join(artifact_dir, "robust_scaler_retrained.joblib"))
    encoder_path = paths.get("encoder_joblib", os.path.join(artifact_dir, "label_encoder_retrained.joblib"))
    metrics_path = paths.get("last_train_metrics_json", os.path.join(artifact_dir, "last_train_metrics.json"))

    artifacts = {
        "model": os.path.exists(model_path),
        "scaler": os.path.exists(scaler_path),
        "encoder": os.path.exists(encoder_path),
    }
    metrics = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
        except Exception:
            pass

    return jsonify({
        "artifacts": artifacts,
        "all_present": all(artifacts.values()),
        "metrics": metrics,
    }), 200


# ── Retrain ──

_retrain_state = {
    "status": "idle",      # idle | started | running | validating | completed | failed
    "job_id": None,
    "started_at": None,
    "finished_at": None,
    "metrics_old": None,
    "metrics_new": None,
    "error": None,
    "pid": None,
}
_retrain_lock_path = os.path.join(os.path.dirname(os.path.abspath(config_path)), ".retrain.lock")
RETRAIN_SIGNAL = os.path.join(artifact_dir, ".retrain_complete")

def _reload_artifacts():
    """Hot-reload model, scaler, encoder from disk."""
    global model, scaler, encoder
    try:
        model = load(paths.get("model_joblib", os.path.join(artifact_dir, "model_retrained.joblib")))
        scaler = load(paths.get("scaler_joblib", os.path.join(artifact_dir, "robust_scaler_retrained.joblib")))
        encoder = load(paths.get("encoder_joblib", os.path.join(artifact_dir, "label_encoder_retrained.joblib")))
        app.logger.info("Artifacts hot-reloaded successfully")
        return True
    except Exception as e:
        app.logger.error(f"Artifact reload failed: {e}")
        return False

def _watch_retrain_signal():
    """Background thread: watch for .retrain_complete signal file."""
    import time
    while True:
        try:
            if os.path.exists(RETRAIN_SIGNAL):
                app.logger.info("Retrain signal detected — reloading artifacts")
                _reload_artifacts()
                os.remove(RETRAIN_SIGNAL)
                _retrain_state["status"] = "completed"
                _retrain_state["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
                # Load new metrics
                metrics_path = paths.get("last_train_metrics_json",
                                         os.path.join(artifact_dir, "last_train_metrics.json"))
                if os.path.exists(metrics_path):
                    with open(metrics_path) as f:
                        _retrain_state["metrics_new"] = json.load(f)
        except Exception as e:
            app.logger.error(f"Retrain watcher error: {e}")
        time.sleep(5)

# Start the watcher thread
_watcher = threading.Thread(target=_watch_retrain_signal, daemon=True)
_watcher.start()


@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    # Check if retrain is already running
    if _retrain_state["status"] in ("started", "running", "validating"):
        return jsonify({
            "error": "Retrain already in progress",
            "job_id": _retrain_state["job_id"],
            "status": _retrain_state["status"],
        }), 409

    repo_root = os.path.dirname(os.path.abspath(config_path))

    # Save uploaded files
    sales_files = request.files.getlist("sales_files[]")
    tour_files = request.files.getlist("tour_files[]")

    if not sales_files and not tour_files:
        return jsonify({"error": "No files uploaded. Provide sales_files[] and/or tour_files[]."}), 400

    sales_dir = os.path.join(repo_root, "CSVs", "add_sales_reports_files_here")
    tour_dir = os.path.join(repo_root, "CSVs", "add_tour_summary_files_here")
    os.makedirs(sales_dir, exist_ok=True)
    os.makedirs(tour_dir, exist_ok=True)

    saved_sales = []
    for f in sales_files:
        if f.filename:
            dest = os.path.join(sales_dir, f.filename)
            f.save(dest)
            saved_sales.append(f.filename)

    saved_tours = []
    for f in tour_files:
        if f.filename:
            dest = os.path.join(tour_dir, f.filename)
            f.save(dest)
            saved_tours.append(f.filename)

    # Capture current metrics for comparison
    metrics_path = paths.get("last_train_metrics_json",
                             os.path.join(artifact_dir, "last_train_metrics.json"))
    old_metrics = None
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path) as f:
                old_metrics = json.load(f)
        except Exception:
            pass

    # Start retrain worker subprocess
    job_id = str(uuid.uuid4())[:8]
    worker_script = os.path.join(repo_root, "Python_scripts", "retrain_worker.py")

    if not os.path.exists(worker_script):
        return jsonify({"error": "retrain_worker.py not found on server"}), 500

    venv_python = os.path.join(repo_root, ".venv", "bin", "python")
    if not os.path.exists(venv_python):
        venv_python = sys.executable  # fallback to current python

    _retrain_state.update({
        "status": "started",
        "job_id": job_id,
        "started_at": dt.datetime.now().isoformat(timespec="seconds"),
        "finished_at": None,
        "metrics_old": old_metrics,
        "metrics_new": None,
        "error": None,
        "pid": None,
    })

    def _run_retrain():
        try:
            _retrain_state["status"] = "running"
            proc = subprocess.Popen(
                [venv_python, worker_script, config_path],
                cwd=repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            _retrain_state["pid"] = proc.pid

            # Stream output to log
            output_lines = []
            for line in proc.stdout:
                output_lines.append(line.rstrip())
                app.logger.info(f"[retrain] {line.rstrip()}")

            proc.wait()

            if proc.returncode != 0:
                _retrain_state["status"] = "failed"
                _retrain_state["error"] = "\n".join(output_lines[-20:])
                _retrain_state["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
            # If return code 0, the watcher thread will pick up .retrain_complete

        except Exception as e:
            _retrain_state["status"] = "failed"
            _retrain_state["error"] = str(e)
            _retrain_state["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")

    thread = threading.Thread(target=_run_retrain, daemon=True)
    thread.start()

    _safe_log("api_retrain", "started", {
        "job_id": job_id,
        "sales_files": saved_sales,
        "tour_files": saved_tours,
    })

    return jsonify({
        "job_id": job_id,
        "status": "started",
        "sales_files_saved": saved_sales,
        "tour_files_saved": saved_tours,
    }), 202


@app.route("/api/retrain/status", methods=["GET"])
def api_retrain_status():
    return jsonify({
        "job_id": _retrain_state["job_id"],
        "status": _retrain_state["status"],
        "started_at": _retrain_state["started_at"],
        "finished_at": _retrain_state["finished_at"],
        "metrics_old": _retrain_state["metrics_old"],
        "metrics_new": _retrain_state["metrics_new"],
        "error": _retrain_state["error"],
    }), 200


if __name__ == "__main__":
    port = int(paths.get("flask_port", "5000"))
    app.run(debug=False, port=port)
