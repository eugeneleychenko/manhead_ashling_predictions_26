import os
import re
import time
import pandas as pd
from urllib.parse import quote
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from audit_logger import log_event, sha256_file
from selenium.webdriver.support import expected_conditions as EC


def _safe_log(step: str, status: str, details: dict):
    try:
        log_event(step=step, status=status, details=details)
    except Exception:
        pass


NUMBER_RE = re.compile(r"([\d.,]+)\s*([kKmM]?)")
LISTENERS_RE = re.compile(r"([\d.,kKmM]+)\s+monthly listeners", re.I)


def parse_compact_number(txt: str):
    txt = str(txt).replace(",", "")
    m = NUMBER_RE.search(txt)
    if not m:
        return None
    num_str, suffix = m.groups()
    try:
        val = float(num_str)
    except ValueError:
        return None
    if suffix.lower() == "m":
        val *= 1_000_000
    elif suffix.lower() == "k":
        val *= 1_000
    return int(round(val))


def extract_listeners_from_text(text: str):
    m = LISTENERS_RE.search(str(text))
    return parse_compact_number(m.group(1)) if m else None


def get_monthly_listeners_for_artist(driver, artist_name: str, wait_seconds: int = 10):
    base_url = "https://open.spotify.com"
    search_url = f"{base_url}/search/{quote(str(artist_name))}"
    driver.get(search_url)

    wait = WebDriverWait(driver, wait_seconds)

    try:
        cookie_button = wait.until(
            EC.element_to_be_clickable(
                (By.XPATH, "//button[contains(., 'Accept') or contains(., 'Agree')]")
            )
        )
        cookie_button.click()
    except TimeoutException:
        pass
    except Exception:
        pass

    artist_href = None
    try:
        links = driver.find_elements(By.XPATH, "//a[contains(@href, '/artist/')]")
        if links:
            artist_href = links[0].get_attribute("href")
    except Exception:
        pass

    if not artist_href:
        print(f"[WARN] Could not find artist link for: {artist_name}")
        return None

    driver.get(artist_href)

    listeners = None
    try:
        elem = wait.until(
            EC.presence_of_element_located(
                (
                    By.XPATH,
                    "//*[contains(translate(., 'MONTHLY LISTENERS', 'monthly listeners'), 'monthly listeners')]",
                )
            )
        )
        listeners = extract_listeners_from_text(elem.text)
    except TimeoutException:
        try:
            all_elems = driver.find_elements(
                By.XPATH,
                "//*[contains(., 'monthly listeners') or contains(., 'Monthly listeners')]",
            )
            for e in all_elems:
                val = extract_listeners_from_text(e.text)
                if val is not None:
                    listeners = val
                    break
        except Exception:
            pass
    except Exception:
        pass

    if listeners is None:
        print(f"[WARN] Could not parse monthly listeners for: {artist_name}")

    return listeners


def get_monthly_listeners_for_list(artist_names):
    rows = []

    options = webdriver.ChromeOptions()
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    try:
        for name in artist_names:
            name = str(name).strip()
            if not name:
                continue
            print(f"[INFO] Fetching monthly listeners for: {name}")
            try:
                listeners = get_monthly_listeners_for_artist(driver, name)
            except Exception as e:
                print(f"[ERROR] Unexpected error for {name}: {e}")
                listeners = None

            rows.append({"artistName": name, "spotifyMonthlyListeners": listeners})
            time.sleep(2)
    finally:
        driver.quit()

    return pd.DataFrame(rows)


def _normalize_name(name: str) -> str:
    return str(name).strip().lower() if pd.notna(name) else ""


def update_spotify_listeners_from_metadata(
    artist_meta_csv_path: str,
    spotify_out_csv_path: str,
):

    _safe_log(
        step="spotify_update",
        status="start",
        details={
            "artist_meta_csv_path": artist_meta_csv_path,
            "spotify_out_csv_path": spotify_out_csv_path,
            "artist_meta_exists": bool(os.path.exists(artist_meta_csv_path)),
            "spotify_out_exists": bool(os.path.exists(spotify_out_csv_path)),
            "artist_meta_hash": sha256_file(artist_meta_csv_path) if os.path.exists(artist_meta_csv_path) else "",
            "spotify_out_hash_before": sha256_file(spotify_out_csv_path) if os.path.exists(spotify_out_csv_path) else "",
        },
    )

    if not os.path.exists(artist_meta_csv_path):
        print(f"[SPOTIFY] artist_metadata file not found at {artist_meta_csv_path}. Skipping update.")
        _safe_log(
            step="spotify_update",
            status="warning",
            details={
                "warning": "artist_metadata file not found; skipped",
                "artist_meta_csv_path": artist_meta_csv_path,
            },
        )
        return

    meta_df = pd.read_csv(artist_meta_csv_path, dtype=str)
    if "artistName" not in meta_df.columns:
        print("[SPOTIFY] artist_metadata.csv has no 'artistName' column. Skipping update.")
        _safe_log(
            step="spotify_update",
            status="warning",
            details={
                "warning": "artist_metadata.csv missing artistName; skipped",
                "artist_meta_csv_path": artist_meta_csv_path,
                "columns": list(meta_df.columns),
            },
        )
        return

    meta_df["artistName"] = meta_df["artistName"].astype(str).str.strip()
    meta_df = meta_df[meta_df["artistName"] != ""].copy()
    meta_df["key"] = meta_df["artistName"].apply(_normalize_name)

    if meta_df.empty:
        print("[SPOTIFY] No artists in artist_metadata.csv. Skipping update.")
        _safe_log(
            step="spotify_update",
            status="warning",
            details={
                "warning": "No artists in artist_metadata.csv; skipped",
                "artist_meta_csv_path": artist_meta_csv_path,
            },
        )
        return

    if os.path.exists(spotify_out_csv_path):
        spot_df = pd.read_csv(spotify_out_csv_path, dtype=str)
        if "artistName" not in spot_df.columns:
            spot_df["artistName"] = ""
        if "spotifyMonthlyListeners" not in spot_df.columns:
            spot_df["spotifyMonthlyListeners"] = ""
        spot_df["artistName"] = spot_df["artistName"].astype(str).str.strip()
        spot_df["key"] = spot_df["artistName"].apply(_normalize_name)
    else:
        spot_df = pd.DataFrame(columns=["artistName", "spotifyMonthlyListeners", "key"])

    known_keys = set(spot_df["key"].dropna())
    new_meta = meta_df[~meta_df["key"].isin(known_keys)]
    new_artists = new_meta["artistName"].dropna().unique().tolist()

    if not new_artists:
        print("[SPOTIFY] No new artists to fetch. Existing spotify_monthly_listeners is up to date.")
        _safe_log(
            step="spotify_update",
            status="success",
            details={
                "message": "No new artists; spotify CSV already up to date",
                "new_artists_count": 0,
                "spotify_out_csv_path": spotify_out_csv_path,
                "spotify_out_hash_after": sha256_file(spotify_out_csv_path) if os.path.exists(spotify_out_csv_path) else "",
            },
        )
        return

    print(f"[SPOTIFY] New artists detected (will scrape Spotify): {new_artists}")
    _safe_log(
        step="spotify_update",
        status="info",
        details={
            "new_artists_count": int(len(new_artists)),
            "new_artists_sample": new_artists[:50],
        },
    )

    try:
        df_new = get_monthly_listeners_for_list(new_artists)
    except Exception as e:
        _safe_log(
            step="spotify_update",
            status="error",
            details={
                "error": f"Scrape failed: {e}",
                "new_artists_count": int(len(new_artists)),
            },
        )
        raise

    if df_new.empty:
        print("[SPOTIFY] Scraper returned no data. Leaving existing spotify_monthly_listeners as-is.")
        _safe_log(
            step="spotify_update",
            status="warning",
            details={
                "warning": "Scraper returned empty df; leaving spotify CSV unchanged",
                "spotify_out_csv_path": spotify_out_csv_path,
            },
        )
        return

    df_new["artistName"] = df_new["artistName"].astype(str).str.strip()
    df_new["key"] = df_new["artistName"].apply(_normalize_name)

    combined = pd.concat([spot_df, df_new], ignore_index=True)
    combined = combined.sort_values("artistName").drop_duplicates(subset=["key"], keep="last")
    combined = combined[["artistName", "spotifyMonthlyListeners"]]

    out_dir = os.path.dirname(spotify_out_csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    combined.to_csv(spotify_out_csv_path, index=False)
    print(f"[SPOTIFY] Updated spotify_monthly_listeners at {spotify_out_csv_path}")

    _safe_log(
        step="spotify_update",
        status="success",
        details={
            "spotify_out_csv_path": spotify_out_csv_path,
            "rows_out": int(combined.shape[0]),
            "new_rows_added": int(df_new.shape[0]),
            "spotify_out_hash_after": sha256_file(spotify_out_csv_path) if os.path.exists(spotify_out_csv_path) else "",
        },
    )


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
    base_dir = os.path.dirname(os.path.abspath(config_path))
    paths = {}
    with open(config_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k and v:
                if not os.path.isabs(v) and not v.lower().startswith(("http://", "https://")):
                    v = os.path.normpath(os.path.join(base_dir, v))
                paths[k] = v
    return paths


if __name__ == "__main__":
    artists = [
        "Jelly Roll",
        "Michael Franti & Spearhead",
        "Celtic Woman",
    ]

    df_out = get_monthly_listeners_for_list(artists)
    print(df_out)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = find_paths_config(script_dir)
    p = load_paths(cfg)

    output_path = p.get("spotify_csv", "")
    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df_out.to_csv(output_path, index=False)
        print(f"[SPOTIFY] Wrote: {output_path}")

        _safe_log(
            step="spotify_cli_run",
            status="success",
            details={
                "output_path": output_path,
                "rows_out": int(df_out.shape[0]),
                "output_hash": sha256_file(output_path) if os.path.exists(output_path) else "",
            },
        )
