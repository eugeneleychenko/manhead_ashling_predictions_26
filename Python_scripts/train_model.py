import os
import csv
import json
import tempfile
import numpy as np
import pandas as pd
import datetime as dt
from joblib import dump
from datetime import datetime
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from audit_logger import log_event, sha256_file, get_debug_dir
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


MODEL_FILE = "model_retrained.joblib"
SCALER_FILE = "robust_scaler_retrained.joblib"
ENCODER_FILE = "label_encoder_retrained.joblib"
METRICS_FILE = "last_train_metrics.json"

SEED = 123

def load_paths(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"paths_config.txt not found at: {config_path}")

    base_dir = os.path.dirname(os.path.abspath(config_path))

    paths = {}
    with open(config_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if key and value:
                if not os.path.isabs(value) and not value.lower().startswith(("http://", "https://")):
                    value = os.path.normpath(os.path.join(base_dir, value))
                paths[key] = value
    return paths

def _atomic_joblib_dump(obj, final_path: str):
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    tmp_dir = os.path.dirname(final_path)

    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".joblib", dir=tmp_dir)
    os.close(fd)

    try:
        dump(obj, tmp_path)
        os.replace(tmp_path, final_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

def _clean_text_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("\xa0", " ", regex=False)
        .str.strip()
        .str.lower()
    )

def _to_num(s: pd.Series) -> pd.Series:
    x = s.astype(str)
    x = x.str.replace("\ufeff", "", regex=False)
    x = x.str.replace("\xa0", " ", regex=False)
    x = x.str.strip()
    x = x.str.replace("$", "", regex=False)
    x = x.str.replace(",", "", regex=False)
    x = x.replace({"": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan})

    return pd.to_numeric(x, errors="coerce")

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

def train_and_save(paths_config_path: str, artifact_dir: str):
    paths = load_paths(paths_config_path)

    input_csv = paths["master_training_dataset"]
    artifact_dir = paths.get("flask_artifacts_dir", artifact_dir)

    log_event(
        step="step3_train",
        status="start",
        details={
            "input_csv": input_csv,
            "artifact_dir": artifact_dir,
        },
    )

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Master CSV not found: {input_csv}")

    os.makedirs(artifact_dir, exist_ok=True)

    df = pd.read_csv(input_csv, dtype=str, encoding="utf-8-sig", low_memory=False)

    required_cols = [
        "showDate",
        "artistName",
        "Genre",
        "venue name",
        "venue city",
        "venue state",
        "productType",
        "product size",
        "attendance",
        "product price",
        "quantitySold",
        "temperature_daily_mean",
        "venue capacity",
        "spotifyMonthlyListeners",
        "Instagram",
        "HolidayStatus",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Training CSV missing required columns: {missing}")

    df["showDate"] = pd.to_datetime(df["showDate"], errors="coerce")
    if df["showDate"].isna().all():
        raise ValueError("showDate could not be parsed for any rows.")

    df["attendance"] = _to_num(df["attendance"]).fillna(0)
    df["product price"] = _to_num(df["product price"]).fillna(0)
    df["quantitySold"] = _to_num(df["quantitySold"]).fillna(0)

    df["temperature_daily_mean"] = _to_num(df["temperature_daily_mean"])
    df["venue capacity"] = _to_num(df["venue capacity"])
    df["spotifyMonthlyListeners"] = _to_num(df["spotifyMonthlyListeners"])
    df["Instagram"] = _to_num(df["Instagram"])
    df["HolidayStatus"] = _to_num(df["HolidayStatus"]).fillna(0).astype(int)

    df["spotifyMissing"] = df["spotifyMonthlyListeners"].isna().astype(int)
    df["spotifyMonthlyListeners"] = df["spotifyMonthlyListeners"].fillna(0)

    df["product size"] = df["product size"].fillna("OneSize")
    df["venue capacity"] = df["venue capacity"].fillna(df["attendance"])

    if df["temperature_daily_mean"].notna().any():
        df["temperature_daily_mean"] = df["temperature_daily_mean"].fillna(df["temperature_daily_mean"].median())
    else:
        df["temperature_daily_mean"] = df["temperature_daily_mean"].fillna(0)

    if "Show Day" not in df.columns:
        df["Show Day"] = df["showDate"].dt.day
    if "Show Month" not in df.columns:
        df["Show Month"] = df["showDate"].dt.month
    if "Day of Week Num" not in df.columns:
        df["Day of Week Num"] = df["showDate"].dt.dayofweek + 1

    cat_features = [
        "artistName",
        "Genre",
        "Show Day",
        "Show Month",
        "Day of Week Num",
        "HolidayStatus",
        "venue name",
        "venue city",
        "venue state",
        "productType",
        "product size",
    ]

    num_features = [
        "attendance",
        "product price",
        "spotifyMonthlyListeners",
        "Instagram",
        "temperature_daily_mean",
        "venue capacity",
        "spotifyMissing",
    ]

    for c in cat_features:
        df[c] = _clean_text_series(df[c])

    X = df[cat_features + num_features].copy()
    y = df["quantitySold"].astype(float).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    label_encoders = {}
    X_train_cat = X_train[cat_features].copy()
    X_test_cat = X_test[cat_features].copy()

    for feature in cat_features:
        le = LabelEncoder().fit(X_train_cat[feature].astype(str))

        if "unknown_category" not in le.classes_:
            le.classes_ = np.append(le.classes_, "unknown_category")

        X_train_cat[feature] = le.transform(X_train_cat[feature].astype(str))

        known = set(le.classes_)
        test_vals = X_test_cat[feature].astype(str)
        unknown_mask = ~test_vals.isin(known)
        test_vals = test_vals.where(~unknown_mask, "unknown_category")
        X_test_cat[feature] = le.transform(test_vals)

        label_encoders[feature] = le

    robust_scalers = {}
    X_train_num = X_train[num_features].copy()
    X_test_num = X_test[num_features].copy()

    for feature in num_features:
        scaler = RobustScaler()
        X_train_num[[feature]] = scaler.fit_transform(X_train_num[[feature]])
        X_test_num[[feature]] = scaler.transform(X_test_num[[feature]])
        robust_scalers[feature] = scaler

    X_train_final = pd.concat([X_train_num, X_train_cat], axis=1)
    X_test_final = pd.concat([X_test_num, X_test_cat], axis=1)

    model = ExtraTreesRegressor(
        random_state=SEED,
        n_estimators=2000,
        max_depth=40,
        min_samples_split=6,
        min_samples_leaf=2,
        n_jobs=-1,
    )
    model.fit(X_train_final, y_train)

    y_pred_test = model.predict(X_test_final)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    mae = float(mean_absolute_error(y_test, y_pred_test))
    r2 = float(r2_score(y_test, y_pred_test))

    n = X_test_final.shape[0]
    p = X_test_final.shape[1]
    adj_r2 = float(1 - (1 - r2) * (n - 1) / (max(n - p - 1, 1)))

    metrics = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "input_csv": input_csv,
        "rows_total": int(df.shape[0]),
        "rows_train": int(X_train_final.shape[0]),
        "rows_test": int(X_test_final.shape[0]),
        "rmse_test": rmse,
        "mae_test": mae,
        "r2_test": r2,
        "adj_r2_test": adj_r2,
        "cat_features": cat_features,
        "num_features": num_features,
        "artifact_dir": artifact_dir,
    }

    model_path = paths.get("model_joblib", os.path.join(artifact_dir, MODEL_FILE))
    scaler_path = paths.get("scaler_joblib", os.path.join(artifact_dir, SCALER_FILE))
    encoder_path = paths.get("encoder_joblib", os.path.join(artifact_dir, ENCODER_FILE))
    metrics_path = paths.get("last_train_metrics_json", os.path.join(artifact_dir, METRICS_FILE))

    _atomic_joblib_dump(model, model_path)
    _atomic_joblib_dump(robust_scalers, scaler_path)
    _atomic_joblib_dump(label_encoders, encoder_path)

    os.makedirs(artifact_dir, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    debug_dir = get_debug_dir()
    hist_csv = os.path.join(debug_dir, "train_history.csv")

    row = {
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "adj_r2": adj_r2,
        "input_csv": input_csv,
        "rows_total": int(df.shape[0]),
        "rows_train": int(X_train_final.shape[0]),
        "rows_test": int(X_test_final.shape[0]),
        "model_joblib": model_path,
        "scaler_joblib": scaler_path,
        "encoder_joblib": encoder_path,
        "metrics_json": metrics_path,
        "model_hash": sha256_file(model_path) if model_path and os.path.exists(model_path) else "",
        "scaler_hash": sha256_file(scaler_path) if scaler_path and os.path.exists(scaler_path) else "",
        "encoder_hash": sha256_file(encoder_path) if encoder_path and os.path.exists(encoder_path) else "",
    }

    write_header = not os.path.exists(hist_csv)
    with open(hist_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)

    log_event(step="step3_train", status="success", details=row)

    print("Training completed")
    print(f"Input master -> {input_csv}")
    print(f"Saved model   -> {model_path}")
    print(f"Saved scalers -> {scaler_path}")
    print(f"Saved encoders-> {encoder_path}")
    print(f"Saved metrics -> {metrics_path}")
    print(f"Test RMSE={rmse:.4f} | MAE={mae:.4f} | R2={r2:.4f} | AdjR2={adj_r2:.4f}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = find_paths_config(script_dir)
    paths = load_paths(config_path)
    artifact_dir = paths["flask_artifacts_dir"]

    try:
        train_and_save(config_path, artifact_dir)
    except Exception as e:
        log_event(step="step3_train", status="error", details={"error": str(e)})
        raise
