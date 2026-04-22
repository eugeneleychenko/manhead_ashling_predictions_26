import os
import re
import glob
import json
import requests
import numpy as np
import pandas as pd
import requests_cache
import openmeteo_requests
from retry_requests import retry
from audit_logger import log_event, sha256_file
from spotify import update_spotify_listeners_from_metadata
from pandas.tseries.holiday import USFederalHolidayCalendar

US_STATE_ABBR = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC"
}

CANADA_PROVINCE_ABBR = {
    "AB", "BC", "MB", "NB", "NL", "NS", "NT", "NU", "ON", "PE",
    "QC", "SK", "YT"
}

STATE_NAME_TO_ABBR = {
    "Illinois": "IL",
    "Minnesota": "MN",
    "Manitoba": "MB",
    "Quebec": "QC",
    "Ohio": "OH",
}

CITY_TO_STATE = {
    "Milwaukee": "WI",
    "Seattle": "WA",
    "Denver": "CO",
    "Columbus": "OH",
    "Chicago": "IL",
    "Pittsburgh": "PA",
    "Cleveland": "OH",
    "St Louis": "MO",
    "Baltimore": "MD",
    "Buffalo": "NY",
}


def _safe_log(step: str, status: str, details: dict):
    try:
        log_event(step=step, status=status, details=details)
    except Exception:
        pass


def _safe_makedirs_for(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def prettify_artist_name(name: str):
    if pd.isna(name):
        return name

    s = str(name).strip()
    if not s:
        return s

    if "(" in s and ")" in s:
        return s

    s = s.replace("-", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = " ".join(w.capitalize() for w in s.split())

    return s


def standardize_venue_state(value):
    if pd.isna(value):
        return np.nan

    s = str(value).strip()
    s = re.sub(r"[^\w\s]", "", s)

    if len(s) == 2 and s.isalpha():
        return s.upper()

    title = s.title()
    if title in STATE_NAME_TO_ABBR:
        return STATE_NAME_TO_ABBR[title]

    return s.upper()


def infer_country_from_state(abbr):
    if pd.isna(abbr):
        return np.nan

    code = str(abbr).upper()

    if code in US_STATE_ABBR:
        return "United States"
    if code in CANADA_PROVINCE_ABBR:
        return "Canada"

    return np.nan


def load_paths(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Path config file not found at:\n{config_path}\n"
            f"Please create it based on 'paths_config_example.txt'."
        )

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


def fix_mojibake_city(s):
    if pd.isna(s):
        return s
    try:
        return s.encode("latin1").decode("utf-8")
    except Exception:
        return s


def compute_holiday_status(df: pd.DataFrame) -> pd.Series:
    if "showDate" not in df.columns:
        return pd.Series(0, index=df.index)

    show_dt = pd.to_datetime(df["showDate"], errors="coerce")
    show_d = show_dt.dt.normalize()

    is_weekend = show_dt.dt.dayofweek.isin([5, 6])

    valid_dates = show_d.dropna()
    if valid_dates.empty:
        is_us_holiday = pd.Series(False, index=df.index)
    else:
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=valid_dates.min(), end=valid_dates.max()).normalize()
        holiday_set = set(holidays)
        is_us_holiday = show_d.isin(holiday_set)

    if "venue country" in df.columns:
        country = (
            df["venue country"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        is_us = country.isin(["united states", "usa", "us", ""])
        is_holiday = is_weekend | (is_us_holiday & is_us)
    else:
        is_holiday = is_weekend | is_us_holiday

    return is_holiday.astype(int)


_geo_cache = None
_geo_session = None


def geocode_zip(zip_code: str, country_code: str = "US"):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {
        "postal_code": zip_code,
        "country": country_code,
        "count": 1,
        "language": "en",
        "format": "json",
    }
    try:
        resp = _geo_session.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[GEOCODE] ZIP error for {zip_code}: {e}")
        return None, None

    results = data.get("results") or []
    if not results:
        print(f"[GEOCODE] No ZIP result for {zip_code}")
        return None, None

    r0 = results[0]
    lat = r0.get("latitude")
    lon = r0.get("longitude")

    return lat, lon


def geocode_city_state(city: str, state_code: str | None = None):
    if not city:
        return None, None

    city = str(city).strip()
    country_name = infer_country_from_state(state_code)
    country_code = "CA" if country_name == "Canada" else "US"

    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {
        "name": city,
        "country": country_code,
        "count": 1,
        "language": "en",
        "format": "json",
    }
    try:
        resp = _geo_session.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[GEOCODE] City error for {city}, {state_code}: {e}")
        return None, None

    results = data.get("results") or []
    if not results:
        print(f"[GEOCODE] No city result for {city}, {state_code}")
        return None, None

    r0 = results[0]
    lat = r0.get("latitude")
    lon = r0.get("longitude")

    return lat, lon


def load_city_coords(coords_path: str) -> pd.DataFrame:
    if not os.path.exists(coords_path):
        return pd.DataFrame(columns=["Zip", "latitude", "longitude"])

    coords = pd.read_csv(coords_path, sep=None, engine="python")
    coords.columns = coords.columns.str.strip()

    if "Zip" not in coords.columns:
        raise KeyError(f"coords file at {coords_path} must contain a 'Zip' column.")

    if "Coordinates" in coords.columns:
        coord_series = coords["Coordinates"].fillna("").astype(str)
        split = coord_series.str.split(",", n=1, expand=True)

        first_col = split.iloc[:, 0] if split.shape[1] >= 1 else pd.Series(np.nan, index=coords.index)
        second_col = split.iloc[:, 1] if split.shape[1] >= 2 else pd.Series(np.nan, index=coords.index)

        coords["latitude"] = pd.to_numeric(first_col, errors="coerce")
        coords["longitude"] = pd.to_numeric(second_col.astype(str).str.strip(), errors="coerce")

    if "latitude" not in coords.columns:
        coords["latitude"] = np.nan
    if "longitude" not in coords.columns:
        coords["longitude"] = np.nan

    coords["Zip"] = coords["Zip"].astype(str).str.strip()

    return coords[["Zip", "latitude", "longitude"]]


def save_city_coords(coords: pd.DataFrame, coords_path: str):
    df = coords.copy()
    df["Zip"] = df["Zip"].astype(str).str.strip()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    df["Coordinates"] = df["latitude"].astype(str) + ", " + df["longitude"].astype(str)
    df_out = df[["Zip", "Coordinates"]].drop_duplicates(subset=["Zip"], keep="last")

    _safe_makedirs_for(coords_path)
    df_out.to_csv(coords_path, sep="\t", index=False)
    print(f"[GEOCODE] Saved ZIP coords master to {coords_path}")


def enrich_with_city_coords(df: pd.DataFrame, coords_path: str) -> pd.DataFrame:
    coords = load_city_coords(coords_path)

    if "Zip" not in df.columns:
        print("[GEOCODE] No 'Zip' column in df; skipping ZIP-based geocoding.")
        df["latitude"] = np.nan
        df["longitude"] = np.nan
        return df

    df["Zip"] = df["Zip"].astype(str).str.strip()

    coords["Zip"] = coords["Zip"].astype(str).str.strip()
    df = df.merge(coords, on="Zip", how="left")

    missing_mask = (df["latitude"].isna() | df["longitude"].isna()) & (df["Zip"].str.strip() != "")
    missing_subset = df.loc[missing_mask, ["Zip", "City", "State"]].drop_duplicates()

    print(f"[GEOCODE] Unique ZIPs in data: {df['Zip'].nunique()} | ZIPs needing coords: {len(missing_subset)}")

    new_rows = []
    for _, row in missing_subset.iterrows():
        zip_code = row["Zip"]
        city = row.get("City")
        state = row.get("State")

        if not isinstance(zip_code, str) or not zip_code.strip():
            continue

        lat, lon = geocode_zip(zip_code)

        if (lat is None or lon is None) and pd.notna(city):
            lat, lon = geocode_city_state(city, state)

        if lat is not None and lon is not None:
            new_rows.append({"Zip": zip_code, "latitude": lat, "longitude": lon})
        else:
            print(f"[GEOCODE] Could not geocode ZIP {zip_code} (City={city}, State={state})")

    if new_rows:
        new_coords = pd.DataFrame(new_rows)
        coords = pd.concat([coords, new_coords], ignore_index=True)
        coords = coords.drop_duplicates(subset=["Zip"], keep="last")
        save_city_coords(coords, coords_path)

        df = df.drop(columns=["latitude", "longitude"])
        df = df.merge(coords, on="Zip", how="left")

    missing_after = df["latitude"].isna().sum()
    if missing_after:
        print(f"[GEOCODE] WARNING: {missing_after} rows still missing latitude/longitude after ZIP-based geocoding.")

    return df


def _read_csv_safe(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, encoding="latin1", low_memory=False)


def append_consolidated_to_master(df_new: pd.DataFrame, master_path: str, backup_path: str | None = None) -> dict:
    _safe_makedirs_for(master_path)
    df_new = df_new.copy()

    if not os.path.exists(master_path):
        df_new.to_csv(master_path, index=False)
        return {
            "master_existed": False,
            "master_path": master_path,
            "master_rows_before": 0,
            "new_rows_in_snapshot": int(len(df_new)),
            "rows_added": int(len(df_new)),
            "master_rows_after": int(len(df_new)),
            "dedupe_mode": "full_row",
        }

    df_master = _read_csv_safe(master_path)

    master_rows_before = len(df_master)

    master_cols = list(df_master.columns)
    new_cols = list(df_new.columns)

    if master_cols != new_cols:
        if set(master_cols) == set(new_cols):
            df_new = df_new[master_cols]
        else:
            missing_in_new = [c for c in master_cols if c not in new_cols]
            extra_in_new = [c for c in new_cols if c not in master_cols]
            raise ValueError(
                "MASTER schema mismatch.\n"
                f"Missing in new snapshot: {missing_in_new}\n"
                f"Extra in new snapshot: {extra_in_new}\n"
                f"Master columns: {master_cols}\n"
                f"New columns: {new_cols}"
            )

    df_combined = pd.concat([df_master, df_new], ignore_index=True)
    df_combined = df_combined.drop_duplicates(keep="first")

    master_rows_after = len(df_combined)
    rows_added = master_rows_after - master_rows_before

    if not backup_path:
        backup_path = master_path.replace(".csv", ".__backup__.csv")

    try:
        df_master.to_csv(backup_path, index=False)
    except Exception:
        pass

    tmp_path = master_path.replace(".csv", ".__tmp__.csv")
    df_combined.to_csv(tmp_path, index=False)
    os.replace(tmp_path, master_path)

    return {
        "master_existed": True,
        "master_path": master_path,
        "master_rows_before": int(master_rows_before),
        "new_rows_in_snapshot": int(len(df_new)),
        "rows_added": int(rows_added),
        "master_rows_after": int(master_rows_after),
        "dedupe_mode": "full_row",
        "backup_path": backup_path,
    }


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


def main():
    pipeline_warnings: list[str] = []

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = find_paths_config(script_dir)
    paths = load_paths(config_path)

    master_training_dataset = paths["master_training_dataset"]
    global _geo_cache, _geo_session

    geocoding_cache_name = paths["geocoding_cache_name"]
    _safe_makedirs_for(geocoding_cache_name)
    _geo_cache = requests_cache.CachedSession(geocoding_cache_name, expire_after=86400)
    _geo_session = retry(_geo_cache, retries=3, backoff_factor=0.3)

    path_sales_pattern = paths["path_sales_pattern"]
    merch_path_out = paths["merch_path_out"]
    path_tour_pattern = paths["path_tour_pattern"]
    tour_path_out = paths["tour_path_out"]
    merged_path = paths["merged_path"]
    coords_path = paths["coords_path"]
    spotify_csv = paths["spotify_csv"]
    artist_meta_csv = paths["artist_meta_csv"]
    venue_meta_csv = paths["venue_meta_csv"]
    final_out = paths["final_out"]
    stats_path = paths["consolidation_stats_json"]

    _safe_log(
        step="pipeline_consolidate",
        status="start",
        details={
            "config_path": config_path,
            "path_sales_pattern": path_sales_pattern,
            "path_tour_pattern": path_tour_pattern,
            "merch_path_out": merch_path_out,
            "tour_path_out": tour_path_out,
            "merged_path": merged_path,
            "final_out": final_out,
            "master_training_dataset": master_training_dataset,
            "coords_path": coords_path,
            "spotify_csv": spotify_csv,
            "artist_meta_csv": artist_meta_csv,
            "venue_meta_csv": venue_meta_csv,
            "stats_json": stats_path,
        },
    )

    try:
        all_files_sales = glob.glob(path_sales_pattern)
        dfs_sales = []

        _safe_log(
            step="pipeline_consolidate_sales",
            status="info",
            details={
                "files_found": int(len(all_files_sales)),
                "files_sample": [os.path.basename(p) for p in all_files_sales[:25]],
            },
        )

        for file in all_files_sales:
            df = pd.read_csv(file, header=1, encoding="latin1")
            df = df.dropna(how="all")

            if "City" in df.columns:
                df["City"] = df["City"].apply(fix_mojibake_city)
            if "Venue" in df.columns:
                df["Venue"] = df["Venue"].apply(fix_mojibake_city)

            first_col = df.columns[0]

            df["Sold_num"] = pd.to_numeric(df["Sold"], errors="coerce").fillna(0)
            df = df[~(df[first_col].isna() & (df["Sold_num"] == 0))]

            df[first_col] = df[first_col].fillna("NO_SKU")
            df[first_col] = df[first_col].astype(str).str.strip()

            bad_labels = [
                "SUBTOTAL", "TOTAL APPAREL", "TOTAL OTHER", "TOTAL MUSIC",
                "GRAND TOTAL", "MUSIC", "OTHER", "UPC", "SKU", ""
            ]
            df = df[~df[first_col].str.upper().isin(bad_labels)]

            df = df.drop(columns=["Sold_num"])

            filename = os.path.basename(file)
            df["source_file"] = filename

            match = re.search(r"(\d{2}-\d{2}-\d{4})", filename)
            if match:
                parsed_dt = pd.to_datetime(match.group(1), format="%m-%d-%Y")
                df["date"] = parsed_dt.strftime("%m/%d/%Y")
            else:
                df["date"] = ""

            dfs_sales.append(df)

        combined_sales = pd.concat(dfs_sales, ignore_index=True) if dfs_sales else pd.DataFrame()
        _safe_makedirs_for(merch_path_out)
        combined_sales.to_csv(merch_path_out, index=False)

        merch_hash = sha256_file(merch_path_out) if os.path.exists(merch_path_out) else ""
        _safe_log(
            step="pipeline_consolidate_sales",
            status="success",
            details={
                "merch_path_out": merch_path_out,
                "rows_out": int(combined_sales.shape[0]) if isinstance(combined_sales, pd.DataFrame) else 0,
                "cols_out": list(combined_sales.columns) if isinstance(combined_sales, pd.DataFrame) else [],
                "output_hash": merch_hash,
            },
        )

        expected_tour_cols = [
            "Date", "City", "State", "Zip", "Venue", "Venue Actual %", "Capacity", "Attend",
            "Currency", "Exch. Rate", "Per Head", "Gross", "Tax", "Payment Fees", "Venue Fee",
            "Venue Adjust.", "Vend Fee", "Ext Exp", "Bootleg Exp", "Selling Exp", "Net Receipts",
        ]

        all_files_tour = glob.glob(path_tour_pattern)
        dfs_tour = []

        _safe_log(
            step="pipeline_consolidate_tour",
            status="info",
            details={
                "files_found": int(len(all_files_tour)),
                "files_sample": [os.path.basename(p) for p in all_files_tour[:25]],
            },
        )

        for file in all_files_tour:
            raw = pd.read_csv(file, skiprows=4, header=0, dtype=str, encoding="latin1")

            raw = raw.dropna(how="all")
            raw = raw.dropna(axis=1, how="all")

            if raw.shape[1] >= len(expected_tour_cols):
                df = raw.iloc[:, :len(expected_tour_cols)].copy()
            else:
                df = raw.copy()
                for i in range(df.shape[1], len(expected_tour_cols)):
                    df[f"_extra_{i}"] = ""
                df = df.iloc[:, :len(expected_tour_cols)]

            df.columns = expected_tour_cols

            for c in df.columns:
                df[c] = df[c].astype(str).str.strip()

            df = df[~df["Date"].str.upper().isin(["TOTAL", "AVG/SHOW"])]

            df["source_file"] = os.path.basename(file)
            dfs_tour.append(df)

        if dfs_tour:
            combined_tour = pd.concat(dfs_tour, ignore_index=True)
        else:
            combined_tour = pd.DataFrame(columns=expected_tour_cols + ["source_file"])
            pipeline_warnings.append("No tour files found; combined_tour is empty.")

        _safe_makedirs_for(tour_path_out)
        combined_tour.to_csv(tour_path_out, index=False)

        tour_hash = sha256_file(tour_path_out) if os.path.exists(tour_path_out) else ""
        _safe_log(
            step="pipeline_consolidate_tour",
            status="success",
            details={
                "tour_path_out": tour_path_out,
                "rows_out": int(combined_tour.shape[0]),
                "cols_out": list(combined_tour.columns),
                "output_hash": tour_hash,
                "warnings": pipeline_warnings[-5:],
            },
        )

        merch = pd.read_csv(merch_path_out, dtype=str) if os.path.exists(merch_path_out) else pd.DataFrame()
        tour = pd.read_csv(tour_path_out, dtype=str) if os.path.exists(tour_path_out) else pd.DataFrame()

        def extract_artist(source_file_value: str) -> str:
            base = os.path.basename(str(source_file_value)).lower()
            return base.split("_")[0].strip()

        merch["artist_key"] = merch["source_file"].apply(extract_artist) if "source_file" in merch.columns else ""
        tour["artist_key"] = tour["source_file"].apply(extract_artist) if "source_file" in tour.columns else ""

        merch["date_key"] = pd.to_datetime(merch["date"], format="%m/%d/%Y", errors="coerce") if "date" in merch.columns else pd.NaT
        tour["date_key"] = pd.to_datetime(tour["Date"], format="%m/%d/%Y", errors="coerce") if "Date" in tour.columns else pd.NaT

        merged = pd.merge(
            merch,
            tour,
            on=["artist_key", "date_key"],
            how="left",
            suffixes=("_merch", "_tour")
        )

        merged = merged.rename(columns={"artist_key": "artist_Name"})
        merged = merged.drop(columns=["Date", "date_key"], errors="ignore")

        artist_map = {
            "air-supply": "Air Supply",
            "deftones-mh": "Deftones (MH)",
            "garbage-mh": "Garbage (MH)",
            "jelly-roll-mh": "Jelly Roll (MH)",
            "lainey-wilson-mh": "Lainey Wilson (MH)",
        }
        merged["artist_Name"] = merged["artist_Name"].map(artist_map).fillna(merged["artist_Name"])
        merged["artist_Name"] = merged["artist_Name"].apply(prettify_artist_name)

        _safe_makedirs_for(merged_path)
        merged.to_csv(merged_path, index=False)

        merged_hash = sha256_file(merged_path) if os.path.exists(merged_path) else ""
        _safe_log(
            step="pipeline_consolidate_merge",
            status="success",
            details={
                "merged_path": merged_path,
                "rows_out": int(merged.shape[0]),
                "cols_out": list(merged.columns),
                "output_hash": merged_hash,
            },
        )

        df = pd.read_csv(merged_path)

        if "City" in df.columns:
            df["City"] = df["City"].apply(fix_mojibake_city)
        if "Venue" in df.columns:
            df["Venue"] = df["Venue"].apply(fix_mojibake_city)

        if "showDate" in df.columns:
            df["showDate"] = pd.to_datetime(df["showDate"], errors="coerce").dt.strftime("%Y-%m-%d")
        elif "date" in df.columns:
            df["showDate"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce").dt.strftime("%Y-%m-%d")
        elif "Date" in df.columns:
            df["showDate"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        else:
            df["showDate"] = pd.NaT

        if "City" in df.columns and "State" in df.columns:
            city_split = df["City"].astype(str).str.split(",", n=1, expand=True)
            if city_split.shape[1] > 1:
                state_missing = df["State"].isna() | (df["State"].astype(str).str.strip() == "")
                has_comma = city_split[1].notna()
                mask_comma = state_missing & has_comma
                df.loc[mask_comma, "City"] = city_split.loc[mask_comma, 0].str.strip()
                df.loc[mask_comma, "State"] = city_split.loc[mask_comma, 1].str.strip()

            state_missing = df["State"].isna() | (df["State"].astype(str).str.strip() == "")
            city_clean = df["City"].astype(str).str.strip()
            df.loc[state_missing, "State"] = city_clean.map(CITY_TO_STATE)

        if "State" in df.columns:
            df["State"] = df["State"].apply(standardize_venue_state)

        df = enrich_with_city_coords(df, coords_path)

        missing_coords = 0
        if "latitude" in df.columns and "longitude" in df.columns:
            missing_coords = int((df["latitude"].isna() | df["longitude"].isna()).sum())

        shows = (
            df[["Zip", "showDate", "latitude", "longitude"]]
            .dropna()
            .drop_duplicates()
            .reset_index(drop=True)
        )

        weather_cache_name = paths["weather_cache_name"]
        _safe_makedirs_for(weather_cache_name)
        cache_session = requests_cache.CachedSession(weather_cache_name, expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        url = "https://archive-api.open-meteo.com/v1/archive"

        weather_error_count = 0

        def fetch_weather_for_row(row):
            nonlocal weather_error_count
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            date = row["showDate"]

            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": date,
                "end_date": date,
                "daily": ["temperature_2m_mean", "rain_sum", "snowfall_sum"],
                "timezone": "auto",
                "utm_source": "chatgpt.com",
            }

            try:
                responses = openmeteo.weather_api(url, params=params)
                response = responses[0]

                daily = response.Daily()
                temperature_2m_mean = daily.Variables(0).ValuesAsNumpy()[0]
                rain_sum = daily.Variables(1).ValuesAsNumpy()[0]
                snowfall_sum = daily.Variables(2).ValuesAsNumpy()[0]

                return pd.Series(
                    {
                        "temperature_2m_mean": temperature_2m_mean,
                        "rain_sum": rain_sum,
                        "snowfall_sum": snowfall_sum,
                    }
                )
            except Exception as e:
                weather_error_count += 1
                zip_code = row.get("Zip", "UNKNOWN")
                print(f"[WEATHER] Error for ZIP {zip_code} on {date}: {e}")
                return pd.Series(
                    {
                        "temperature_2m_mean": np.nan,
                        "rain_sum": np.nan,
                        "snowfall_sum": np.nan,
                    }
                )

        if shows.empty:
            pipeline_warnings.append("No shows with coords; skipping weather lookup (all weather fields empty).")
            df["temperature_2m_mean"] = np.nan
            df["rain_sum"] = np.nan
            df["snowfall_sum"] = np.nan
            df_with_weather = df
        else:
            weather_df = shows.apply(fetch_weather_for_row, axis=1)
            for col in ["temperature_2m_mean", "rain_sum", "snowfall_sum"]:
                if col not in weather_df.columns:
                    weather_df[col] = np.nan

            shows_weather = pd.concat([shows, weather_df], axis=1)

            df_with_weather = df.merge(
                shows_weather[["Zip", "showDate", "temperature_2m_mean", "rain_sum", "snowfall_sum"]],
                on=["Zip", "showDate"],
                how="left",
            )

        keep_rename_map = {
            "artist_Name": "artistName",
            "Venue": "venue name",
            "City": "venue city",
            "State": "venue state",
            "Zip": "venue postalCode",
            "Capacity": "venue capacity",
            "Attend": "attendance",
            "Name": "merch category",
            "Type": "productType",
            "Size": "product size",
            "Sold": "quantitySold",
            "Avg. Price": "product price",
            "showDate": "showDate",
            "rain_sum": "rain",
            "snowfall_sum": "snowfall",
            "temperature_2m_mean": "temperature_daily_mean",
        }

        existing_keep_cols = [c for c in keep_rename_map.keys() if c in df_with_weather.columns]
        df_final = df_with_weather[existing_keep_cols].rename(columns=keep_rename_map)

        if "product size" in df_final.columns:
            ps = df_final["product size"].fillna("").astype(str).str.strip()
            ps = ps.replace({"nan": "", "NaN": "", "None": ""})
            ps = ps.mask(ps == "", "One-Size")
            df_final["product size"] = ps

        dt_show = pd.to_datetime(df_final["showDate"], errors="coerce")
        df_final["Show Day"] = dt_show.dt.day
        df_final["Show Month"] = dt_show.dt.month
        df_final["Day of Week Num"] = dt_show.dt.dayofweek + 1

        if "venue country" not in df_final.columns:
            df_final["venue country"] = ""

        if "venue state" in df_final.columns:
            inferred_country = df_final["venue state"].apply(infer_country_from_state)
            df_final["venue country"] = df_final["venue country"].where(
                df_final["venue country"].notna() & (df_final["venue country"] != ""),
                inferred_country
            )

        try:
            update_spotify_listeners_from_metadata(artist_meta_csv, spotify_csv)
        except Exception as e:
            msg = f"Spotify auto-update failed; continuing with existing spotify_csv. Error={e}"
            pipeline_warnings.append(msg)
            print(f"[SPOTIFY] Auto-update failed (continuing with existing spotify_csv): {e}")

        if os.path.exists(spotify_csv):
            spotify_df = pd.read_csv(spotify_csv, dtype=str)
            spotify_df.columns = spotify_df.columns.str.strip()

            if "spotifyMonthlyListeners" not in spotify_df.columns and "monthly_listeners" in spotify_df.columns:
                spotify_df = spotify_df.rename(columns={"monthly_listeners": "spotifyMonthlyListeners"})

            if "artistName" in spotify_df.columns and "spotifyMonthlyListeners" in spotify_df.columns:
                def norm_artist(s):
                    return str(s).lower().replace("_", " ").strip()

                df_final["artistName"] = df_final["artistName"].astype(str).str.strip()
                spotify_df["artistName"] = spotify_df["artistName"].astype(str).str.strip()

                df_final["_artist_k"] = df_final["artistName"].apply(norm_artist)
                spotify_df["_artist_k"] = spotify_df["artistName"].apply(norm_artist)

                spotify_df = spotify_df.drop_duplicates(subset=["_artist_k"], keep="last")

                df_final = df_final.merge(
                    spotify_df[["_artist_k", "spotifyMonthlyListeners"]],
                    on="_artist_k",
                    how="left",
                )
                df_final.drop(columns=["_artist_k"], inplace=True, errors="ignore")
            else:
                pipeline_warnings.append("spotify_monthly_listeners.csv present but missing artistName/spotifyMonthlyListeners.")
                print("[WARN] spotify_monthly_listeners.csv found but missing artistName / spotifyMonthlyListeners.")
                df_final["spotifyMonthlyListeners"] = ""
        else:
            pipeline_warnings.append("spotify_monthly_listeners.csv not found; spotifyMonthlyListeners filled empty.")
            print("[INFO] spotify_monthly_listeners.csv not found; filling spotifyMonthlyListeners with empty values.")
            df_final["spotifyMonthlyListeners"] = ""

        if os.path.exists(artist_meta_csv):
            meta_df = pd.read_csv(artist_meta_csv, dtype=str)
            meta_df.columns = meta_df.columns.str.strip()

            rename_meta = {}
            if "Instagram_followers" in meta_df.columns:
                rename_meta["Instagram_followers"] = "Instagram"
            if "instagram_followers" in meta_df.columns:
                rename_meta["instagram_followers"] = "Instagram"
            if "Instagram followers" in meta_df.columns:
                rename_meta["Instagram followers"] = "Instagram"
            if "instagram followers" in meta_df.columns:
                rename_meta["instagram followers"] = "Instagram"

            meta_df = meta_df.rename(columns=rename_meta)

            if "artistName" in meta_df.columns:
                meta_df["artistName"] = meta_df["artistName"].astype(str).str.strip()
                df_final["artistName"] = df_final["artistName"].astype(str).str.strip()

                def norm_artist(s):
                    return str(s).lower().replace("_", " ").strip()

                meta_df["_artist_k"] = meta_df["artistName"].apply(norm_artist)
                df_final["_artist_k"] = df_final["artistName"].apply(norm_artist)

                meta_df = meta_df.drop_duplicates(subset=["_artist_k"], keep="last")

                cols_to_merge = ["_artist_k"]
                if "Instagram" in meta_df.columns:
                    cols_to_merge.append("Instagram")
                if "Genre" in meta_df.columns:
                    cols_to_merge.append("Genre")

                df_final = df_final.merge(meta_df[cols_to_merge], on="_artist_k", how="left")
                df_final.drop(columns=["_artist_k"], inplace=True, errors="ignore")
            else:
                pipeline_warnings.append("artist_metadata.csv missing artistName; leaving Genre/Instagram empty.")
                print("[WARN] artist_metadata.csv missing artistName; leaving Genre/Instagram empty.")
                df_final["Genre"] = ""
                df_final["Instagram"] = ""
        else:
            pipeline_warnings.append("artist_metadata.csv not found; Genre/Instagram filled empty.")
            print("[INFO] artist_metadata.csv not found; filling Genre and Instagram with empty values.")
            df_final["Genre"] = ""
            df_final["Instagram"] = ""

        if os.path.exists(venue_meta_csv):
            vh = pd.read_csv(venue_meta_csv, dtype=str, sep=None, engine="python")
            vh.columns = vh.columns.str.strip()

            if "showDate" in vh.columns:
                vh["showDate"] = pd.to_datetime(vh["showDate"], errors="coerce").dt.strftime("%Y-%m-%d")

            for col in ["venue name", "venue city", "venue state"]:
                if col in vh.columns:
                    vh[col] = vh[col].fillna("").astype(str).str.strip()

            if "venue city" in vh.columns and "venue state" in vh.columns:
                city_split_vh = vh["venue city"].astype(str).str.split(",", n=1, expand=True)
                if city_split_vh.shape[1] > 1:
                    state_missing_vh = vh["venue state"].isna() | (vh["venue state"].astype(str).str.strip() == "")
                    has_comma_vh = city_split_vh[1].notna()
                    mask_comma_vh = state_missing_vh & has_comma_vh

                    vh.loc[mask_comma_vh, "venue city"] = city_split_vh.loc[mask_comma_vh, 0].str.strip()
                    vh.loc[mask_comma_vh, "venue state"] = city_split_vh.loc[mask_comma_vh, 1].str.strip()

                state_missing_vh = vh["venue state"].isna() | (vh["venue state"].astype(str).str.strip() == "")
                city_clean_vh = vh["venue city"].astype(str).str.strip()
                vh.loc[state_missing_vh, "venue state"] = city_clean_vh.map(CITY_TO_STATE)

            if "venue state" in vh.columns:
                vh["venue state"] = vh["venue state"].apply(standardize_venue_state)

            key_cols = ["showDate", "venue name", "venue city", "venue state"]
            existing_keys = [c for c in key_cols if c in vh.columns]
            vh = vh.drop_duplicates(subset=existing_keys, keep="last")

            for col in ["venue name", "venue city", "venue state"]:
                if col in df_final.columns:
                    df_final[col] = df_final[col].fillna("").astype(str).str.strip()

            merge_cols = ["showDate", "venue name", "venue city", "venue state"]
            merge_cols = [c for c in merge_cols if c in df_final.columns and c in vh.columns]

            vh_cols_to_take = ["HolidayStatus", "venue country", "venue capacity"]
            vh_cols_available = [c for c in vh_cols_to_take if c in vh.columns]

            if merge_cols and vh_cols_available:
                df_final = df_final.merge(
                    vh[merge_cols + vh_cols_available],
                    on=merge_cols,
                    how="left",
                    suffixes=("", "_vh"),
                )

                for c in vh_cols_available:
                    src = c + "_vh"
                    if src in df_final.columns:
                        df_final[c] = df_final[c].where(
                            df_final[c].notna() & (df_final[c] != ""),
                            df_final[src]
                        )
                        df_final.drop(columns=[src], inplace=True)
        else:
            pipeline_warnings.append("venue_holiday_data.csv not found; HolidayStatus/venue country/venue capacity left as-is.")
            print("[INFO] venue_holiday_data.csv not found; HolidayStatus / venue country / venue capacity left as-is.")

        computed_hs = compute_holiday_status(df_final)
        if "HolidayStatus" not in df_final.columns:
            df_final["HolidayStatus"] = computed_hs
        else:
            existing = pd.to_numeric(df_final["HolidayStatus"], errors="coerce")
            df_final["HolidayStatus"] = (
                existing
                .where(existing.isin([0, 1]), computed_hs)
                .fillna(computed_hs)
                .astype(int)
            )

        if "venue capacity" in df_final.columns and "attendance" in df_final.columns:
            vc_raw = df_final["venue capacity"].fillna("").astype(str).str.strip()
            att_raw = df_final["attendance"].fillna("").astype(str).str.strip()

            vc = vc_raw.replace({"nan": "", "NaN": "", "None": ""})
            missing_mask = (vc == "") | (vc == "0")
            df_final.loc[missing_mask, "venue capacity"] = att_raw.loc[missing_mask]

            cap_clean = df_final["venue capacity"].fillna("").astype(str).str.replace(",", "").str.strip()
            att_clean = df_final["attendance"].fillna("").astype(str).str.replace(",", "").str.strip()

            cap_num = pd.to_numeric(cap_clean, errors="coerce")
            att_num = pd.to_numeric(att_clean, errors="coerce")

            too_small = cap_num.notna() & att_num.notna() & (cap_num < att_num)
            df_final.loc[too_small, "venue capacity"] = df_final.loc[too_small, "attendance"]

        desired_order = [
            "artistName", "Genre", "showDate", "Show Day", "Show Month", "Day of Week Num",
            "HolidayStatus", "venue name", "venue city", "venue state", "venue country",
            "venue postalCode", "merch category", "productType", "product size", "attendance",
            "product price", "quantitySold", "rain", "snowfall", "Instagram",
            "spotifyMonthlyListeners", "temperature_daily_mean", "venue capacity",
        ]

        for col in desired_order:
            if col not in df_final.columns:
                df_final[col] = ""

        df_final = df_final[desired_order]
        df_final = df_final.drop_duplicates(keep="first")

        venue_name_clean = df_final["venue name"].fillna("").astype(str).str.strip()
        venue_city_clean = df_final["venue city"].fillna("").astype(str).str.strip()

        replacements = {"nan": "", "NaN": "", "None": ""}
        venue_name_clean = venue_name_clean.replace(replacements)
        venue_city_clean = venue_city_clean.replace(replacements)

        mask_valid_venue = (venue_name_clean != "") & (venue_city_clean != "")
        df_final = df_final[mask_valid_venue].copy()

        _safe_makedirs_for(final_out)
        df_final.to_csv(final_out, index=False)
        print(f"[OK] Wrote consolidated snapshot to: {final_out} | rows={len(df_final)}")

        final_hash = sha256_file(final_out) if os.path.exists(final_out) else ""
        master_before_hash = sha256_file(master_training_dataset) if os.path.exists(master_training_dataset) else ""

        stats = append_consolidated_to_master(df_final, master_training_dataset, paths.get("master_training_dataset_backup"))
        print(
            f"[MASTER] existed={stats['master_existed']} | "
            f"before={stats['master_rows_before']} | "
            f"snapshot_rows={stats['new_rows_in_snapshot']} | "
            f"added={stats['rows_added']} | "
            f"after={stats['master_rows_after']}"
        )
        print(f"[MASTER] path={stats['master_path']}")

        _safe_makedirs_for(stats_path)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"[OK] Wrote stats to: {stats_path}")

        master_after_hash = sha256_file(master_training_dataset) if os.path.exists(master_training_dataset) else ""
        stats_hash = sha256_file(stats_path) if os.path.exists(stats_path) else ""

        details_done = {
            "final_out": final_out,
            "final_rows": int(df_final.shape[0]),
            "final_cols": list(df_final.columns),
            "final_hash": final_hash,
            "master_training_dataset": master_training_dataset,
            "master_hash_before": master_before_hash,
            "master_hash_after": master_after_hash,
            "stats_json": stats_path,
            "stats_hash": stats_hash,
            "master_append": stats,
            "sales_files_found": int(len(all_files_sales)),
            "tour_files_found": int(len(all_files_tour)),
            "missing_coords_rows": int(missing_coords),
            "weather_error_count": int(weather_error_count),
            "warnings": pipeline_warnings[:50],
        }

        _safe_log(step="pipeline_consolidate", status="success", details=details_done)

    except Exception as e:
        _safe_log(
            step="pipeline_consolidate",
            status="error",
            details={
                "error": str(e),
                "warnings": pipeline_warnings[:50],
            },
        )
        raise

if __name__ == "__main__":
    main()
