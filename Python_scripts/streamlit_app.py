import os
import sys
import json
import hashlib
import requests
import tempfile
import subprocess
import urllib3
import pandas as pd
import streamlit as st
from io import StringIO
from audit_logger import log_event, sha256_bytes, sha256_file

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = find_paths_config(SCRIPT_DIR)
PATHS = load_paths(CONFIG_PATH)

REPO_ROOT = os.path.dirname(os.path.abspath(CONFIG_PATH))
STREAMLIT_DOWNLOADS_DIR = os.path.join(REPO_ROOT, "outputs")
os.makedirs(STREAMLIT_DOWNLOADS_DIR, exist_ok=True)

def save_streamlit_download_copy(file_name: str, data_bytes: bytes) -> str | None:
    out_path = os.path.join(STREAMLIT_DOWNLOADS_DIR, file_name)
    try:
        with open(out_path, "wb") as f:
            f.write(data_bytes)
        return out_path
    except Exception:
        return None

# API configuration — supports Streamlit secrets for cloud deployment
def _get_secret(key, default=""):
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, default)

_api_base = _get_secret("PREDICTION_API_BASE_URL",
                         f"http://localhost:{int(PATHS.get('flask_port', '5000'))}")
FLASK_API_URL = f"{_api_base.rstrip('/')}/api/predict"
FLASK_RETRAIN_URL = f"{_api_base.rstrip('/')}/api/retrain"
FLASK_RETRAIN_STATUS_URL = f"{_api_base.rstrip('/')}/api/retrain/status"
API_KEY = _get_secret("API_KEY", "")

def _api_headers():
    """Return headers dict with API key if configured."""
    h = {}
    if API_KEY:
        h["X-API-Key"] = API_KEY
    return h

IS_CLOUD = bool(_get_secret("PREDICTION_API_BASE_URL", ""))
FLASK_ARTIST_META_URL = f"{_api_base.rstrip('/')}/api/artist-metadata"

st.set_page_config(page_title="Merch Sales Prediction", layout="wide")

st.title("Merch Sales Prediction by Size")
st.write("Upload and enrich your data, then get predictions from the Flask model API.")

st.markdown("---")
st.header("Step 1: Add New Artist Data")

add_artist_choice = st.radio(
    "Do you want to add new artist information?",
    ("No", "Yes"),
    index=0,
    key="add_artist_choice",
)

ARTIST_METADATA_PATH = PATHS["artist_meta_csv"]
INSTAGRAM_COL = "Instagram_followers"

os.makedirs(os.path.dirname(ARTIST_METADATA_PATH), exist_ok=True)

if "artist_entries" not in st.session_state:
    st.session_state["artist_entries"] = []

if add_artist_choice == "Yes":
    st.write("Add metadata for new artists (up to 5 artists at a time). ")

    num_artist_rows = st.selectbox(
        "How many artist entries do you want to add?",
        options=[1, 2, 3, 4, 5],
        index=0,
        key="num_artist_rows",
    )

    with st.form("artist_form"):
        st.subheader("Artist Information")

        artist_list = []
        for i in range(num_artist_rows):
            st.markdown(f"**Artist Entry {i + 1}**")

            col1, col2, col3 = st.columns(3)

            with col1:
                artist_name = st.text_input(
                    f"Artist Name (Entry {i + 1})",
                    placeholder="Example: Jelly Roll",
                    key=f"artist_name_{i}",
                )

            with col2:
                instagram = st.text_input(
                    f"Instagram Followers (Entry {i + 1})",
                    placeholder="526000",
                    key=f"instagram_{i}",
                )

            with col3:
                genre = st.text_input(
                    f"Genre (Entry {i + 1})",
                    placeholder="Country Rock",
                    key=f"genre_{i}",
                )

            artist_list.append(
                {
                    "artistName": artist_name,
                    INSTAGRAM_COL: instagram,
                    "Genre": genre,
                }
            )

            st.markdown("---")

        submitted = st.form_submit_button("Add new data")

        if submitted:
            cleaned_artists = [row for row in artist_list if row["artistName"].strip()]

            log_event(
                step="step1_artist_meta",
                status="start",
                details={
                    "artist_meta_path": ARTIST_METADATA_PATH,
                    "entries_requested": int(num_artist_rows),
                    "entries_submitted": int(len(artist_list)),
                    "entries_nonempty": int(len(cleaned_artists)),
                },
            )

            if not cleaned_artists:
                log_event(
                    step="step1_artist_meta",
                    status="warning",
                    details={
                        "artist_meta_path": ARTIST_METADATA_PATH,
                        "warning": "No non-empty artistName values submitted",
                    },
                )
                st.error("Please enter at least one artist name.")
            else:
                if IS_CLOUD:
                    # Remote: POST to DO droplet API
                    try:
                        resp = requests.post(
                            FLASK_ARTIST_META_URL,
                            json={"artists": cleaned_artists},
                            headers=_api_headers(),
                            timeout=30,
                            verify=False,
                        )
                        if resp.ok:
                            body = resp.json()
                            st.session_state["artist_entries"].extend(cleaned_artists)
                            st.success(f"Added {body.get('rows_added', len(cleaned_artists))} artist row(s) to server.")
                        else:
                            st.error(f"Failed to add artist data: {resp.text}")
                    except requests.RequestException as e:
                        st.error(f"Could not reach artist metadata API: {e}")
                else:
                    # Local: write to CSV directly
                    if os.path.exists(ARTIST_METADATA_PATH):
                        try:
                            existing_df = pd.read_csv(ARTIST_METADATA_PATH)
                        except PermissionError as e:
                            log_event(
                                step="step1_artist_meta",
                                status="error",
                                details={
                                    "artist_meta_path": ARTIST_METADATA_PATH,
                                    "error": f"PermissionError reading artist metadata: {e}",
                                },
                            )
                            st.error(
                                "Could not read artist_metadata.csv. "
                                "Please close it if it's open in Excel or another program and try again."
                            )
                            existing_df = pd.DataFrame(columns=["artistName", INSTAGRAM_COL, "Genre"])
                        except Exception as e:
                            log_event(
                                step="step1_artist_meta",
                                status="error",
                                details={
                                    "artist_meta_path": ARTIST_METADATA_PATH,
                                    "error": f"Error reading artist metadata: {e}",
                                },
                            )
                            st.error(f"Could not read artist_metadata.csv: {e}")
                            existing_df = pd.DataFrame(columns=["artistName", INSTAGRAM_COL, "Genre"])
                    else:
                        existing_df = pd.DataFrame(columns=["artistName", INSTAGRAM_COL, "Genre"])

                    if "Instagram" in existing_df.columns:
                        if INSTAGRAM_COL not in existing_df.columns:
                            existing_df = existing_df.rename(columns={"Instagram": INSTAGRAM_COL})
                        else:
                            existing_df[INSTAGRAM_COL] = existing_df[INSTAGRAM_COL].fillna(existing_df["Instagram"])
                        existing_df = existing_df.drop(columns=["Instagram"])

                    for col in ["artistName", INSTAGRAM_COL, "Genre"]:
                        if col not in existing_df.columns:
                            existing_df[col] = ""

                    new_df = pd.DataFrame(cleaned_artists)
                    new_df = new_df[["artistName", INSTAGRAM_COL, "Genre"]]

                    updated_df = pd.concat([existing_df, new_df], ignore_index=True)

                    try:
                        updated_df.to_csv(ARTIST_METADATA_PATH, index=False)
                    except PermissionError as e:
                        log_event(
                            step="step1_artist_meta",
                            status="error",
                            details={
                                "artist_meta_path": ARTIST_METADATA_PATH,
                                "error": f"PermissionError writing artist metadata: {e}",
                                "rows_attempted_add": int(len(cleaned_artists)),
                            },
                        )
                        st.error(
                            "Could not update artist_metadata.csv. "
                            "Most likely the file is open in Excel or another application. "
                            "Please close it and try again."
                        )
                    except Exception as e:
                        log_event(
                            step="step1_artist_meta",
                            status="error",
                            details={
                                "artist_meta_path": ARTIST_METADATA_PATH,
                                "error": f"Error writing artist metadata: {e}",
                                "rows_attempted_add": int(len(cleaned_artists)),
                            },
                        )
                        st.error(f"Could not update artist_metadata.csv: {e}")
                    else:
                        st.session_state["artist_entries"].extend(cleaned_artists)

                        file_hash = sha256_file(ARTIST_METADATA_PATH) if os.path.exists(ARTIST_METADATA_PATH) else ""
                        log_event(
                            step="step1_artist_meta",
                            status="success",
                            details={
                                "artist_meta_path": ARTIST_METADATA_PATH,
                                "rows_added": int(len(cleaned_artists)),
                                "rows_total_after": int(updated_df.shape[0]),
                                "artist_meta_hash": file_hash,
                            },
                        )

                        st.success(f"Added {len(cleaned_artists)} artist row(s) to artist_metadata.csv.")

if st.session_state["artist_entries"]:
    st.subheader("Artists added in this session")
    df_artists_session = pd.DataFrame(st.session_state["artist_entries"])
    st.dataframe(df_artists_session, use_container_width=True)

    csv_artists_buf = StringIO()
    df_artists_session.to_csv(csv_artists_buf, index=False)

    artists_csv_str = csv_artists_buf.getvalue()
    save_streamlit_download_copy("artist_info_session.csv", artists_csv_str.encode("utf-8"))

    st.download_button(
        label="Download This Session's Artist Info CSV",
        data=artists_csv_str,
        file_name="artist_info_session.csv",
        mime="text/csv",
    )

st.markdown("---")
st.header("Step 2: Format and Consolidate Data")

st.write(
    "This runs the consolidation script, writes a consolidated snapshot, "
    "and appends it into the continuously-updated master dataset (full-row dedupe)."
)

if st.button("Run consolidation pipeline"):
    script_path = PATHS["consolidation_script_path"]
    final_csv = PATHS["final_out"]
    stats_path = PATHS["consolidation_stats_json"]
    master_path = PATHS.get("master_training_dataset", "")

    log_event(
        step="step2_consolidate",
        status="start",
        details={
            "script_path": script_path,
            "final_out": final_csv,
            "stats_json": stats_path,
            "master_training_dataset": master_path,
        },
    )

    with st.spinner("Running consolidation script..."):
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            log_event(
                step="step2_consolidate",
                status="error",
                details={
                    "script_path": script_path,
                    "final_out": final_csv,
                    "stats_json": stats_path,
                    "master_training_dataset": master_path,
                    "returncode": int(e.returncode) if e.returncode is not None else None,
                    "stdout": (e.stdout or "")[:4000],
                    "stderr": (e.stderr or "")[:4000],
                },
            )

            st.error("Consolidation script failed.")
            if e.stdout:
                st.code(e.stdout)
            if e.stderr:
                st.code(e.stderr)
        else:
            st.success("Consolidation completed successfully.")

            stats = None
            if os.path.exists(stats_path):
                with open(stats_path, "r", encoding="utf-8") as f:
                    stats = json.load(f)

                st.subheader("Master dataset append results")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Master rows before", stats.get("master_rows_before", "—"))
                c2.metric("Snapshot rows", stats.get("new_rows_in_snapshot", "—"))
                c3.metric("Rows added", stats.get("rows_added", "—"))
                c4.metric("Master rows after", stats.get("master_rows_after", "—"))

                st.caption(f"Master path: {stats.get('master_path', '—')}")
            else:
                st.warning(f"Stats file not found: {stats_path}")

            df_final = None
            if os.path.exists(final_csv):
                df_final = pd.read_csv(final_csv)

                st.subheader("Preview of consolidated snapshot (merged_with_weather.csv)")
                st.dataframe(df_final.head(200), use_container_width=True)

                csv_str = df_final.to_csv(index=False)
                save_streamlit_download_copy("merged_with_weather.csv", csv_str.encode("utf-8"))

                st.download_button(
                    label="Download consolidated snapshot CSV",
                    data=csv_str,
                    file_name="merged_with_weather.csv",
                    mime="text/csv",
                )
            else:
                st.error(f"Expected output file not found: {final_csv}")

            details = {
                "script_path": script_path,
                "final_out": final_csv,
                "stats_json": stats_path,
                "master_training_dataset": master_path,
                "stdout": (result.stdout or "")[:2000],
                "stderr": (result.stderr or "")[:2000],
            }

            if stats is not None:
                details.update({
                    "master_rows_before": stats.get("master_rows_before"),
                    "new_rows_in_snapshot": stats.get("new_rows_in_snapshot"),
                    "rows_added": stats.get("rows_added"),
                    "master_rows_after": stats.get("master_rows_after"),
                    "master_path": stats.get("master_path"),
                })

            if os.path.exists(final_csv):
                details["final_out_hash"] = sha256_file(final_csv)
                if df_final is not None:
                    details["final_out_rows"] = int(df_final.shape[0])
                    details["final_out_cols"] = list(df_final.columns)

            if os.path.exists(stats_path):
                details["stats_json_hash"] = sha256_file(stats_path)

            if master_path and os.path.exists(master_path):
                details["master_hash"] = sha256_file(master_path)

            log_event(step="step2_consolidate", status="success", details=details)

st.markdown("---")
st.header("Step 3: Train / Retrain Model")

st.write(
    "This trains the model using the latest MASTER dataset (continuously updated) "
    "and overwrites the model artifacts used by the Flask prediction API."
)

TRAIN_SCRIPT = PATHS["train_model_script_path"]

if st.button("Train model"):
    master_path = PATHS.get("master_training_dataset", "")
    artifact_dir = PATHS.get("flask_artifacts_dir", "")
    model_path = PATHS.get("model_joblib", "")
    scaler_path = PATHS.get("scaler_joblib", "")
    encoder_path = PATHS.get("encoder_joblib", "")
    metrics_path = PATHS.get("last_train_metrics_json", "")

    log_event(
        step="step3_train_streamlit",
        status="start",
        details={
            "train_script": TRAIN_SCRIPT,
            "master_training_dataset": master_path,
            "artifact_dir": artifact_dir,
            "model_joblib": model_path,
            "scaler_joblib": scaler_path,
            "encoder_joblib": encoder_path,
            "metrics_json": metrics_path,
        },
    )

    with st.spinner("Training model from master dataset and updating joblib artifacts..."):
        try:
            result = subprocess.run(
                [sys.executable, TRAIN_SCRIPT],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            log_event(
                step="step3_train_streamlit",
                status="error",
                details={
                    "train_script": TRAIN_SCRIPT,
                    "returncode": int(e.returncode) if e.returncode is not None else None,
                    "stdout": (e.stdout or "")[:4000],
                    "stderr": (e.stderr or "")[:4000],
                },
            )

            st.error("Model training failed.")
            if e.stdout:
                with st.expander("Training logs (stdout)"):
                    st.code(e.stdout)
            if e.stderr:
                with st.expander("Training logs (stderr)"):
                    st.code(e.stderr)
        else:
            st.success("Model trained successfully. Joblib artifacts updated.")

            if result.stdout:
                with st.expander("Training logs (stdout)"):
                    st.code(result.stdout)

            if result.stderr:
                with st.expander("Training logs (stderr / warnings)"):
                    st.code(result.stderr)

            st.info("If your Flask server is already running, restart it to ensure it loads the newly trained artifacts.")

            metrics = None
            if metrics_path and os.path.exists(metrics_path):
                try:
                    with open(metrics_path, "r", encoding="utf-8") as f:
                        metrics = json.load(f)
                except Exception:
                    metrics = None

            details = {
                "train_script": TRAIN_SCRIPT,
                "stdout": (result.stdout or "")[:2000],
                "stderr": (result.stderr or "")[:2000],
                "master_training_dataset": master_path,
                "artifact_dir": artifact_dir,
                "model_joblib": model_path,
                "scaler_joblib": scaler_path,
                "encoder_joblib": encoder_path,
                "metrics_json": metrics_path,
            }

            if master_path and os.path.exists(master_path):
                details["master_hash"] = sha256_file(master_path)
            if model_path and os.path.exists(model_path):
                details["model_hash"] = sha256_file(model_path)
            if scaler_path and os.path.exists(scaler_path):
                details["scaler_hash"] = sha256_file(scaler_path)
            if encoder_path and os.path.exists(encoder_path):
                details["encoder_hash"] = sha256_file(encoder_path)
            if metrics_path and os.path.exists(metrics_path):
                details["metrics_hash"] = sha256_file(metrics_path)

            if isinstance(metrics, dict):
                details.update({
                    "rmse_test": metrics.get("rmse_test"),
                    "mae_test": metrics.get("mae_test"),
                    "r2_test": metrics.get("r2_test"),
                    "adj_r2_test": metrics.get("adj_r2_test"),
                    "rows_total": metrics.get("rows_total"),
                    "rows_train": metrics.get("rows_train"),
                    "rows_test": metrics.get("rows_test"),
                    "trained_at": metrics.get("trained_at"),
                    "input_csv": metrics.get("input_csv"),
                })

            log_event(step="step3_train_streamlit", status="success", details=details)

st.markdown("---")
st.header("Step 4: Run Predictions")

tab_convert, tab_upload = st.tabs(["Convert from AtVenu", "Upload CSV"])

# ── Tab 1: Upload pre-formatted CSV (existing flow) ──
with tab_upload:
    st.write("Upload a pre-formatted 21-column CSV and get predictions from the Flask model API.")

    uploaded_file = st.file_uploader(
        "Upload your CSV file for prediction",
        type=["csv"],
        key="prediction_uploader",
    )

    if uploaded_file is not None:
        st.write("File name:", uploaded_file.name)

        if st.button("Run prediction", key="btn_predict_upload"):
            file_bytes = uploaded_file.getvalue()
            file_hash = sha256_bytes(file_bytes)

            log_event(
                step="step4_predict",
                status="start",
                details={"input_file": uploaded_file.name, "input_hash": file_hash},
            )

            files = {"csv_file": (uploaded_file.name, file_bytes, "text/csv")}

            with st.spinner("Calling prediction API and running model..."):
                try:
                    response = requests.post(FLASK_API_URL, files=files, headers=_api_headers(), timeout=120, verify=False)
                    response.raise_for_status()
                except requests.RequestException as e:
                    log_event(
                        step="step4_predict",
                        status="error",
                        details={"input_file": uploaded_file.name, "input_hash": file_hash, "error": str(e)},
                    )
                    st.error(f"Error calling prediction API: {e}")
                else:
                    try:
                        payload = response.json()
                    except ValueError:
                        log_event(
                            step="step4_predict",
                            status="error",
                            details={"input_file": uploaded_file.name, "input_hash": file_hash, "error": "API did not return valid JSON"},
                        )
                        st.error("API did not return valid JSON.")
                    else:
                        if "error" in payload:
                            log_event(
                                step="step4_predict",
                                status="error",
                                details={"input_file": uploaded_file.name, "input_hash": file_hash, "error": payload.get("error")},
                            )
                            st.error(f"API returned an error: {payload['error']}")
                        else:
                            data = payload.get("data", [])

                            if not data:
                                log_event(
                                    step="step4_predict",
                                    status="warning",
                                    details={"input_file": uploaded_file.name, "input_hash": file_hash, "warning": "API returned no data"},
                                )
                                st.warning("API returned no data.")
                            else:
                                df_pred = pd.DataFrame(data)

                                log_event(
                                    step="step4_predict",
                                    status="success",
                                    details={
                                        "input_file": uploaded_file.name,
                                        "input_hash": file_hash,
                                        "rows_out": int(df_pred.shape[0]),
                                        "cols_out": list(df_pred.columns),
                                    },
                                )

                                st.success("Predictions generated successfully.")
                                st.subheader("Preview of predictions")
                                st.dataframe(df_pred, use_container_width=True)

                                csv_buffer = StringIO()
                                df_pred.to_csv(csv_buffer, index=False)

                                pred_csv_str = csv_buffer.getvalue()
                                save_streamlit_download_copy(
                                    "predicted_sales_streamlit.csv",
                                    pred_csv_str.encode("utf-8"),
                                )
                                st.session_state["predicted_csv_bytes"] = pred_csv_str.encode("utf-8")

                                st.download_button(
                                    label="Download predictions as CSV",
                                    data=pred_csv_str,
                                    file_name="predicted_sales_streamlit.csv",
                                    mime="text/csv",
                                )

# ── Tab 2: Convert from AtVenu inventory xlsx ──
with tab_convert:
    from converter_utils import convert_inventory_to_prediction_input, ConversionInputError, fetch_band_names_from_api

    @st.cache_data(ttl=3600, show_spinner="Loading band list from AtVenu...")
    def _cached_band_names():
        return fetch_band_names_from_api()

    st.write(
        "Upload a raw AtVenu inventory/forecast file (xlsx or csv). "
        "Product prices are fetched live from the AtVenu API."
    )

    band_names = _cached_band_names()

    atvenu_file = st.file_uploader(
        "Upload AtVenu inventory file",
        type=["xlsx", "xls", "csv"],
        key="atvenu_uploader",
    )
    conv_band_name = st.selectbox("Band name", options=[""] + band_names, index=0, key="conv_band")
    conv_weather = st.checkbox("Fetch weather data (slower)", value=False, key="conv_weather")

    if atvenu_file is not None and conv_band_name:
        if st.button("Convert", key="btn_convert"):
            progress_area = st.empty()
            def _progress(msg: str):
                progress_area.info(msg)

            try:
                with st.spinner("Converting..."):
                    conv_df, conv_meta = convert_inventory_to_prediction_input(
                        file_bytes=atvenu_file.getvalue(),
                        file_name=atvenu_file.name,
                        band_name=conv_band_name.strip(),
                        artist_meta_path=PATHS.get("artist_meta_csv"),
                        spotify_path=PATHS.get("spotify_csv"),
                        fetch_weather=conv_weather,
                        progress_callback=_progress,
                    )
            except ConversionInputError as e:
                st.error(f"Conversion error: {e}")
            except Exception as e:
                st.error(f"Unexpected error during conversion: {e}")
            else:
                progress_area.empty()

                # Warnings
                if conv_meta.get("venue_misses"):
                    st.warning(
                        f"{conv_meta['venue_misses']} of {conv_meta['shows_parsed']} shows "
                        f"had no venue match in AtVenu API (attendance/capacity = 0)."
                    )
                if conv_meta.get("price_miss_skus"):
                    st.warning(
                        f"{len(conv_meta['price_miss_skus'])} SKUs had no price in AtVenu API. "
                        f"First few: {', '.join(conv_meta['price_miss_skus'][:10])}"
                    )

                nan_prices = int(conv_df["product price"].isna().sum())
                if nan_prices:
                    st.warning(f"{nan_prices} rows have missing prices (NaN). The model may still predict, but results may be less accurate.")

                st.success(
                    f"Converted {conv_meta['products_parsed']} products x "
                    f"{conv_meta['shows_parsed']} shows = {conv_meta['rows_out']} rows. "
                    f"API prices found: {conv_meta['prices_found']}."
                )

                st.subheader("Preview of converted data")
                st.dataframe(conv_df, use_container_width=True)

                # Download converted CSV
                conv_csv = conv_df.to_csv(index=False)
                save_streamlit_download_copy("converted_atvenu_input.csv", conv_csv.encode("utf-8"))
                st.download_button(
                    label="Download converted CSV",
                    data=conv_csv,
                    file_name="converted_atvenu_input.csv",
                    mime="text/csv",
                    key="dl_converted",
                )

                # Store in session for direct prediction
                st.session_state["converted_csv_bytes"] = conv_csv.encode("utf-8")
                st.session_state["converted_csv_name"] = f"converted_{atvenu_file.name.rsplit('.', 1)[0]}.csv"

    # Run prediction on converted data
    if "converted_csv_bytes" in st.session_state:
        st.markdown("---")
        st.subheader("Run prediction on converted data")
        if st.button("Run prediction on converted file", key="btn_predict_converted"):
            conv_bytes = st.session_state["converted_csv_bytes"]
            conv_name = st.session_state["converted_csv_name"]
            file_hash = sha256_bytes(conv_bytes)

            log_event(
                step="step4_predict_converted",
                status="start",
                details={"input_file": conv_name, "input_hash": file_hash},
            )

            files = {"csv_file": (conv_name, conv_bytes, "text/csv")}

            with st.spinner("Calling prediction API..."):
                try:
                    response = requests.post(FLASK_API_URL, files=files, headers=_api_headers(), timeout=120, verify=False)
                    response.raise_for_status()
                    payload = response.json()
                except Exception as e:
                    st.error(f"Prediction API error: {e}")
                else:
                    if "error" in payload:
                        st.error(f"API error: {payload['error']}")
                    else:
                        data = payload.get("data", [])
                        if not data:
                            st.warning("API returned no data.")
                        else:
                            df_pred = pd.DataFrame(data)
                            log_event(
                                step="step4_predict_converted",
                                status="success",
                                details={"input_file": conv_name, "rows_out": int(df_pred.shape[0])},
                            )
                            st.success("Predictions generated successfully.")
                            st.subheader("Preview of predictions")
                            st.dataframe(df_pred, use_container_width=True)

                            pred_csv = StringIO()
                            df_pred.to_csv(pred_csv, index=False)
                            pred_str = pred_csv.getvalue()
                            save_streamlit_download_copy("predicted_sales_streamlit.csv", pred_str.encode("utf-8"))
                            st.session_state["predicted_csv_bytes"] = pred_str.encode("utf-8")
                            st.download_button(
                                label="Download predictions as CSV",
                                data=pred_str,
                                file_name="predicted_sales_streamlit.csv",
                                mime="text/csv",
                                key="dl_pred_converted",
                            )

st.markdown("---")
st.header("Step 5: Calculate Revenue Per Head")

st.write("Upload the Step 4 prediction output CSV, or use the prediction from the previous step.")

STEP5_SCRIPT_PATH = PATHS["revph_script_path"]
STEP5_OUTPUT_PATH = PATHS["revph_output_csv"]
STEP5_SUMMARY_PATH = PATHS["revph_summary_json"]

if "predicted_csv_bytes" in st.session_state:
    if st.button("Use predicted sales from Step 4", key="btn_use_step4_pred"):
        st.session_state["step5_auto_bytes"] = st.session_state["predicted_csv_bytes"]
        st.session_state["step5_auto_name"] = "predicted_sales_streamlit.csv"
        st.session_state["step5_last_hash"] = None

revph_file = st.file_uploader(
    "Or upload Step 4 output CSV",
    type=["csv"],
    key="revph_uploader_v3",
)

if "step5_last_hash" not in st.session_state:
    st.session_state["step5_last_hash"] = None
if "step5_out_csv_path" not in st.session_state:
    st.session_state["step5_out_csv_path"] = None
if "step5_summary_path" not in st.session_state:
    st.session_state["step5_summary_path"] = None

if "step5_logged_success_hash" not in st.session_state:
    st.session_state["step5_logged_success_hash"] = None

def _run_step5_script(uploaded_bytes: bytes):
    tmp_dir = tempfile.gettempdir()
    file_hash = hashlib.sha256(uploaded_bytes).hexdigest()[:16]

    input_path = os.path.join(tmp_dir, f"step5_input_{file_hash}.csv")
    output_path = STEP5_OUTPUT_PATH
    summary_path = STEP5_SUMMARY_PATH

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    with open(input_path, "wb") as f:
        f.write(uploaded_bytes)

    result = subprocess.run(
        [
            sys.executable,
            STEP5_SCRIPT_PATH,
            "--input", input_path,
            "--output", output_path,
            "--summary", summary_path,
            "--round", "3",
        ],
        capture_output=True,
        text=True,
    )

    return result, output_path, summary_path, file_hash

# Determine input: uploaded file or auto from Step 4
_step5_bytes = None
_step5_name = None
if revph_file is not None:
    _step5_bytes = revph_file.getvalue()
    _step5_name = revph_file.name
elif "step5_auto_bytes" in st.session_state:
    _step5_bytes = st.session_state.pop("step5_auto_bytes")
    _step5_name = st.session_state.pop("step5_auto_name", "predicted_sales_streamlit.csv")

if _step5_bytes is not None:
    st.write("File name:", _step5_name)

    uploaded_bytes = _step5_bytes
    current_hash = hashlib.sha256(uploaded_bytes).hexdigest()[:16]

    if st.session_state["step5_last_hash"] != current_hash:
        log_event(
            step="step5_revph",
            status="start",
            details={"input_file": _step5_name, "input_hash": current_hash},
        )

        with st.spinner("Running Step 5 calculations..."):
            result, out_csv_path, summary_path, file_hash = _run_step5_script(uploaded_bytes)

        if result.returncode != 0:
            log_event(
                step="step5_revph",
                status="error",
                details={
                    "input_file": _step5_name,
                    "input_hash": current_hash,
                    "returncode": int(result.returncode),
                    "stderr": (result.stderr or "")[:4000],
                    "stdout": (result.stdout or "")[:4000],
                },
            )

            st.error("Step 5 script failed.")
            if result.stdout:
                with st.expander("stdout"):
                    st.code(result.stdout)
            if result.stderr:
                with st.expander("stderr"):
                    st.code(result.stderr)
        else:
            st.session_state["step5_last_hash"] = current_hash
            st.session_state["step5_out_csv_path"] = out_csv_path
            st.session_state["step5_summary_path"] = summary_path

            st.session_state["step5_logged_success_hash"] = None

    out_csv_path = st.session_state.get("step5_out_csv_path")
    summary_path = st.session_state.get("step5_summary_path")

    if out_csv_path and os.path.exists(out_csv_path):
        try:
            df_out = pd.read_csv(out_csv_path)
        except Exception as e:
            st.error(f"Could not read Step 5 output CSV: {e}")
            df_out = None

        if df_out is not None:
            st.success("Revenue Per Head Calculated.")

            st.subheader("Preview output (with $/head)")
            st.dataframe(df_out.head(300), use_container_width=True)

            try:
                with open(out_csv_path, "rb") as f:
                    out_bytes = f.read()
                save_streamlit_download_copy("predicted_sales_with_revenue_per_head.csv", out_bytes)
            except Exception:
                pass

            with open(out_csv_path, "rb") as f:
                st.download_button(
                    label="Download Step 5 output CSV",
                    data=f.read(),
                    file_name="predicted_sales_with_revenue_per_head.csv",
                    mime="text/csv",
                )

            summary = None
            if summary_path and os.path.exists(summary_path):
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        summary = json.load(f)
                except Exception as e:
                    st.error(f"Could not read summary JSON: {e}")
                    summary = None

                try:
                    with open(summary_path, "rb") as f:
                        summary_bytes = f.read()
                    save_streamlit_download_copy("revenue_per_head_summary.json", summary_bytes)
                except Exception:
                    pass

            if st.session_state.get("step5_logged_success_hash") != current_hash:
                log_event(
                    step="step5_revph",
                    status="success",
                    details={
                        "input_file": _step5_name,
                        "input_hash": current_hash,
                        "output_csv": out_csv_path,
                        "summary_json": summary_path,
                        "rows_out": int(df_out.shape[0]),
                        "shows": (summary or {}).get("shows"),
                        "warnings": (summary or {}).get("warnings", []),
                        "attendance_inconsistent_shows": (summary or {}).get("attendance_inconsistent_shows"),
                    },
                )
                st.session_state["step5_logged_success_hash"] = current_hash

            if summary:
                summary_df = pd.DataFrame([{
                    "Total Predicted Sales": summary.get("total_predicted_sales"),
                    "Average Sales Per Product": summary.get("avg_sales_per_product"),
                    "Product Types": summary.get("product_types"),
                    "Shows": summary.get("shows"),
                }])

                st.subheader("Summary Statistics")
                st.dataframe(
                    summary_df.style.format({
                        "Total Predicted Sales": "{:,.0f}",
                        "Average Sales Per Product": "{:,.2f}",
                        "Product Types": "{:,.0f}",
                        "Shows": "{:,.0f}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

# ═══════════════════════════════════════════════════════════════
# Step 6: Retrain Model (Remote)
# ═══════════════════════════════════════════════════════════════

st.markdown("---")
st.header("Step 6: Retrain Model")

st.write(
    "Upload new sales and/or tour CSVs, then trigger a retrain on the server. "
    "The model will only be promoted if it passes the validation gate (R² doesn't drop by more than 10%)."
)

import time as _time

col_sales, col_tour = st.columns(2)

with col_sales:
    retrain_sales_files = st.file_uploader(
        "Sales report CSVs",
        type=["csv"],
        accept_multiple_files=True,
        key="retrain_sales",
    )

with col_tour:
    retrain_tour_files = st.file_uploader(
        "Tour summary CSVs",
        type=["csv"],
        accept_multiple_files=True,
        key="retrain_tour",
    )

if "retrain_job_id" not in st.session_state:
    st.session_state["retrain_job_id"] = None
if "retrain_polling" not in st.session_state:
    st.session_state["retrain_polling"] = False

if st.button("Start Retrain", disabled=st.session_state.get("retrain_polling", False)):
    if not retrain_sales_files and not retrain_tour_files:
        st.error("Please upload at least one sales or tour CSV file.")
    else:
        files_payload = []
        for f in (retrain_sales_files or []):
            files_payload.append(("sales_files[]", (f.name, f.getvalue(), "text/csv")))
        for f in (retrain_tour_files or []):
            files_payload.append(("tour_files[]", (f.name, f.getvalue(), "text/csv")))

        try:
            resp = requests.post(FLASK_RETRAIN_URL, files=files_payload,
                                 headers=_api_headers(), timeout=30, verify=False)
            if resp.status_code == 409:
                st.warning("A retrain is already in progress. Please wait for it to finish.")
            elif resp.status_code == 401:
                st.error("Unauthorized — check your API key in Streamlit secrets.")
            elif resp.ok:
                body = resp.json()
                st.session_state["retrain_job_id"] = body.get("job_id")
                st.session_state["retrain_polling"] = True
                st.success(f"Retrain started! Job ID: {body.get('job_id')}")
                st.rerun()
            else:
                st.error(f"Retrain request failed: {resp.text}")
        except requests.RequestException as e:
            st.error(f"Could not reach retrain API: {e}")

# Poll for retrain status
if st.session_state.get("retrain_polling"):
    status_placeholder = st.empty()
    metrics_placeholder = st.empty()

    try:
        resp = requests.get(FLASK_RETRAIN_STATUS_URL, headers=_api_headers(),
                            timeout=10, verify=False)
        if resp.ok:
            status_data = resp.json()
            current_status = status_data.get("status", "unknown")

            if current_status in ("started", "running", "validating"):
                status_placeholder.info(f"Retrain status: **{current_status}** (polling every 10s...)")
                _time.sleep(10)
                st.rerun()

            elif current_status == "completed":
                st.session_state["retrain_polling"] = False
                status_placeholder.success("Retrain completed! New model is live.")

                old_m = status_data.get("metrics_old") or {}
                new_m = status_data.get("metrics_new") or {}

                if old_m or new_m:
                    metrics_placeholder.markdown("### Metrics Comparison")
                    mc1, mc2 = metrics_placeholder.columns(2)
                    with mc1:
                        st.metric("Old R²", f"{old_m.get('r2_test', 'N/A'):.4f}" if isinstance(old_m.get('r2_test'), (int, float)) else "N/A")
                        st.metric("Old RMSE", f"{old_m.get('rmse_test', 'N/A'):.4f}" if isinstance(old_m.get('rmse_test'), (int, float)) else "N/A")
                    with mc2:
                        st.metric("New R²", f"{new_m.get('r2_test', 'N/A'):.4f}" if isinstance(new_m.get('r2_test'), (int, float)) else "N/A")
                        st.metric("New RMSE", f"{new_m.get('rmse_test', 'N/A'):.4f}" if isinstance(new_m.get('rmse_test'), (int, float)) else "N/A")

            elif current_status == "failed":
                st.session_state["retrain_polling"] = False
                error_msg = status_data.get("error", "Unknown error")
                status_placeholder.error(f"Retrain failed: {error_msg}")

            else:
                st.session_state["retrain_polling"] = False
                status_placeholder.info(f"Retrain status: {current_status}")

    except requests.RequestException as e:
        st.session_state["retrain_polling"] = False
        status_placeholder.error(f"Could not check retrain status: {e}")