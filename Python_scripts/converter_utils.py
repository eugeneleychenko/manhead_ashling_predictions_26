"""
AtVenu inventory/forecast xlsx → 21-column prediction input converter.

Prices are fetched live from the AtVenu GraphQL API (merchVariants).
Venue data (capacity, attendance) is fetched from the AtVenu GraphQL API.
Genre, Instagram, Spotify come from local CSVs.
Weather is fetched from Open-Meteo (optional).
"""
from __future__ import annotations

import datetime as dt
import os
import re
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANADIAN_PROVINCES = {
    "AB", "BC", "MB", "NB", "NL", "NS", "NT", "NU",
    "ON", "PE", "QC", "SK", "YT",
}

OUTPUT_COLUMNS = [
    "artistName", "Genre", "showDate", "HolidayStatus",
    "venue name", "venue city", "venue state", "venue country",
    "venue postalCode", "merch category", "productType", "product size",
    "attendance", "product price", "temperature_daily_mean", "rain",
    "snowfall", "spotifyMonthlyListeners", "Instagram", "venue capacity",
    "spotifyMissing",
]

US_HOLIDAYS = {
    dt.date(2025, 1, 1), dt.date(2025, 1, 20), dt.date(2025, 2, 17),
    dt.date(2025, 5, 26), dt.date(2025, 6, 19), dt.date(2025, 7, 4),
    dt.date(2025, 9, 1), dt.date(2025, 10, 13), dt.date(2025, 11, 11),
    dt.date(2025, 11, 27), dt.date(2025, 12, 25),
    dt.date(2026, 1, 1), dt.date(2026, 1, 19), dt.date(2026, 2, 16),
    dt.date(2026, 5, 25), dt.date(2026, 6, 19), dt.date(2026, 7, 4),
    dt.date(2026, 9, 7), dt.date(2026, 10, 12), dt.date(2026, 11, 11),
    dt.date(2026, 11, 26), dt.date(2026, 12, 25),
}

ATVENU_API_ENDPOINT = "https://api.atvenu.com"
ATVENU_API_TOKEN = os.environ.get("ATVENU_API_TOKEN", "live_yvYLBo32dRE9z_yCdhwU")
ATTENDANCE_RATIO = 0.80

US_CITY_STATE = {
    "Lancaster": "PA", "Englewood": "NJ", "Flint": "MI", "Prior Lake": "MN",
    "Kansas City": "MO", "Saint Louis": "MO", "St. Louis": "MO",
    "West Palm Beach": "FL", "Miami": "FL", "Myrtle Beach": "SC",
    "Las Vegas": "NV", "North Tonawanda": "NY", "Greensburg": "PA",
    "San Jose": "CA", "San Francisco": "CA", "Los Angeles": "CA",
    "San Diego": "CA", "New York": "NY", "Brooklyn": "NY", "Chicago": "IL",
    "Houston": "TX", "Dallas": "TX", "Austin": "TX", "Denver": "CO",
    "Seattle": "WA", "Portland": "OR", "Phoenix": "AZ", "Atlanta": "GA",
    "Nashville": "TN", "Boston": "MA", "Philadelphia": "PA", "Pittsburgh": "PA",
    "Detroit": "MI", "Minneapolis": "MN", "St. Paul": "MN", "Milwaukee": "WI",
    "Cleveland": "OH", "Columbus": "OH", "Cincinnati": "OH", "Indianapolis": "IN",
    "Charlotte": "NC", "Raleigh": "NC", "Tampa": "FL", "Orlando": "FL",
    "Jacksonville": "FL", "Baltimore": "MD", "Washington": "DC", "Richmond": "VA",
    "Norfolk": "VA", "New Orleans": "LA", "Memphis": "TN", "Louisville": "KY",
    "Oklahoma City": "OK", "Tulsa": "OK", "Salt Lake City": "UT", "Boise": "ID",
    "Omaha": "NE", "Des Moines": "IA", "Buffalo": "NY", "Rochester": "NY",
    "Syracuse": "NY", "Albany": "NY", "Hartford": "CT", "Providence": "RI",
    "Birmingham": "AL", "Knoxville": "TN", "Chattanooga": "TN", "Reno": "NV",
    "Albuquerque": "NM", "El Paso": "TX", "Tucson": "AZ", "Honolulu": "HI",
    "Anchorage": "AK", "Wichita": "KS", "Little Rock": "AR", "Sacramento": "CA",
    "Sarasota": "FL", "Fort Myers": "FL", "Fort Lauderdale": "FL",
    "Pompano Beach": "FL", "Boca Raton": "FL", "Temecula": "CA", "Chandler": "AZ",
    "Red Bank": "NJ", "Oxon Hill": "MD", "Durham": "NC", "Lincoln City": "OR",
    "Hershey": "PA", "Bensalem": "PA", "Greenville": "SC", "Morrison": "CO",
    "Napa": "CA", "Valley Center": "CA", "Rancho Mirage": "CA",
    "Toronto": "ON", "Montreal": "QC", "Vancouver": "BC", "Calgary": "AB",
    "Edmonton": "AB", "Ottawa": "ON", "Winnipeg": "MB", "Halifax": "NS",
}


class ConversionInputError(ValueError):
    """Raised when converter inputs are missing required information."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_num(value: Any) -> float:
    if value is None:
        return 0.0
    value_str = str(value).replace("$", "").replace(",", "").strip()
    if not value_str:
        return 0.0
    try:
        return float(value_str)
    except ValueError:
        return 0.0


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _holiday_status(show_date: dt.date) -> int:
    if show_date.weekday() >= 5:
        return 1
    return 1 if show_date in US_HOLIDAYS else 0


def _venue_country(state_code: str) -> str:
    return "Canada" if state_code.strip().upper() in CANADIAN_PROVINCES else "United States"


def _optional_csv(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()


def _lookup_band_value(
    band_name: str,
    df: pd.DataFrame,
    candidate_cols: list[str],
    value_col: str,
) -> Any:
    if df.empty or value_col not in df.columns:
        return None
    norm_band = _normalize_text(band_name)
    for candidate_col in candidate_cols:
        if candidate_col not in df.columns:
            continue
        norm_series = df[candidate_col].fillna("").astype(str).map(_normalize_text)
        matches = df[norm_series == norm_band]
        if not matches.empty:
            return matches.iloc[0].get(value_col)
    return None


def _genre_for_band(band_name: str, artist_meta_df: pd.DataFrame) -> str:
    val = _lookup_band_value(band_name, artist_meta_df, ["artistName"], "Genre")
    return str(val) if val else "Other"


def _instagram_for_band(band_name: str, artist_meta_df: pd.DataFrame) -> int:
    val = _lookup_band_value(band_name, artist_meta_df, ["artistName"], "Instagram_followers")
    return int(_clean_num(val))


def _spotify_for_band(band_name: str, spotify_df: pd.DataFrame) -> tuple[int, int]:
    val = _lookup_band_value(band_name, spotify_df, ["artistName"], "spotifyMonthlyListeners")
    if val is None or _clean_num(val) <= 0:
        return 0, 1
    return int(_clean_num(val)), 0


# ---------------------------------------------------------------------------
# AtVenu GraphQL: SKU pricing
# ---------------------------------------------------------------------------

_SKU_PRICE_QUERY = """
query($first: Int!, $after: String) {
  organization {
    accounts(first: $first, after: $after) {
      pageInfo { endCursor hasNextPage }
      nodes {
        name
        merchItems(first: 100) {
          nodes {
            name
            productType { name }
            merchVariants { sku size price }
          }
        }
      }
    }
  }
}
"""


_ACCOUNT_NAMES_QUERY = """
query($first: Int!, $after: String) {
  organization {
    accounts(first: $first, after: $after) {
      pageInfo { endCursor hasNextPage }
      nodes { name }
    }
  }
}
"""


def fetch_band_names_from_api() -> list[str]:
    """Fetch all account (band) names from AtVenu. Returns sorted list."""
    headers = {"Content-Type": "application/json", "x-api-key": ATVENU_API_TOKEN}
    names: list[str] = []
    cursor = None

    for _ in range(20):
        payload = {
            "query": _ACCOUNT_NAMES_QUERY,
            "variables": {"first": 50, "after": cursor},
        }
        try:
            resp = requests.post(
                ATVENU_API_ENDPOINT, json=payload, headers=headers, timeout=30
            )
            resp.raise_for_status()
            body = resp.json()
        except Exception:
            break

        if "errors" in body:
            break

        accounts_conn = body.get("data", {}).get("organization", {}).get("accounts", {})
        for account in accounts_conn.get("nodes", []):
            names.append(account["name"])

        pi = accounts_conn.get("pageInfo", {})
        if not pi.get("hasNextPage"):
            break
        cursor = pi.get("endCursor")

    return sorted(names)


def fetch_sku_prices_from_api(band_name: str) -> dict[str, float]:
    """Query AtVenu GraphQL for SKU→price mapping for a band.

    Returns dict like {"DEFTONES-T1071-S": 45.0, ...}.
    Skips items with price <= 0 (retired).
    Stops paginating once the band is found (accounts are alphabetical).
    """
    headers = {"Content-Type": "application/json", "x-api-key": ATVENU_API_TOKEN}
    sku_prices: dict[str, float] = {}
    cursor = None
    norm_band = band_name.lower().strip()
    found_band = False

    for _ in range(30):  # max pages
        payload = {
            "query": _SKU_PRICE_QUERY,
            "variables": {"first": 50, "after": cursor},
        }
        try:
            resp = requests.post(
                ATVENU_API_ENDPOINT, json=payload, headers=headers, timeout=30
            )
            resp.raise_for_status()
            body = resp.json()
        except Exception:
            break

        if "errors" in body:
            break

        accounts_conn = body.get("data", {}).get("organization", {}).get("accounts", {})
        for account in accounts_conn.get("nodes", []):
            if account["name"].lower().strip() == norm_band:
                found_band = True
                for item in account.get("merchItems", {}).get("nodes", []):
                    for v in item.get("merchVariants", []):
                        sku = (v.get("sku") or "").strip()
                        price_raw = v.get("price")
                        if not sku or not price_raw:
                            continue
                        price = float(price_raw)
                        if price > 0:
                            sku_prices[sku] = price

        # Stop paginating once we've found (and passed) the band
        if found_band:
            break

        pi = accounts_conn.get("pageInfo", {})
        if not pi.get("hasNextPage"):
            break
        cursor = pi.get("endCursor")

    return sku_prices


# ---------------------------------------------------------------------------
# AtVenu GraphQL: venue / capacity / attendance
# ---------------------------------------------------------------------------

_SHOWS_QUERY = """
query getFutureShows(
  $firstForAccounts: Int!, $afterForAccounts: String,
  $firstForTours: Int!, $firstForShows: Int!,
  $dateRange: DateRange!
) {
  organization {
    accounts(first: $firstForAccounts, after: $afterForAccounts) {
      pageInfo { endCursor hasNextPage }
      nodes {
        name
        tours(first: $firstForTours) {
          nodes {
            shows(first: $firstForShows, showsOverlap: $dateRange) {
              nodes {
                showDate capacity attendance
                location { name city country }
              }
            }
          }
        }
      }
    }
  }
}
"""


def _fetch_atvenu_api_venues(band_name: str) -> dict[str, dict[str, Any]]:
    """Fetch venue data from AtVenu GraphQL API. Returns city->details dict.
    Stops paginating once the band is found.
    """
    from datetime import datetime, timedelta

    start = datetime.now().strftime("%Y-%m-%d")
    end = (datetime.now() + timedelta(days=365)).strftime("%Y-%m-%d")
    lookup: dict[str, dict[str, Any]] = {}
    cursor = None
    headers = {"Content-Type": "application/json", "x-api-key": ATVENU_API_TOKEN}
    norm_band = band_name.lower().strip()
    found_band = False

    try:
        for _ in range(30):
            payload = {
                "query": _SHOWS_QUERY,
                "variables": {
                    "firstForAccounts": 50,
                    "afterForAccounts": cursor,
                    "firstForTours": 10,
                    "firstForShows": 50,
                    "dateRange": {"start": start, "end": end},
                },
            }
            resp = requests.post(
                ATVENU_API_ENDPOINT, json=payload, headers=headers, timeout=30
            )
            resp.raise_for_status()
            body = resp.json()
            if "errors" in body:
                break
            accounts_conn = body.get("data", {}).get("organization", {}).get("accounts", {})

            for account in accounts_conn.get("nodes", []):
                if account["name"].lower().strip() != norm_band:
                    continue
                found_band = True
                for tour in account.get("tours", {}).get("nodes", []):
                    for show in tour.get("shows", {}).get("nodes", []):
                        loc = show.get("location") or {}
                        city_raw = loc.get("city") or ""
                        city, api_state = "", ""
                        if "," in city_raw:
                            city, api_state = city_raw.split(",", 1)
                            city, api_state = city.strip(), api_state.strip()
                        else:
                            city = city_raw.strip()
                        if not city:
                            continue

                        state = api_state
                        if not state or state == "null":
                            state = US_CITY_STATE.get(city, "")

                        cap = int(show.get("capacity") or 0)
                        attn = show.get("attendance")
                        est = int(attn) if attn else int(cap * ATTENDANCE_RATIO)

                        lookup[city.lower()] = {
                            "venue": loc.get("name", ""),
                            "state": state,
                            "attendance": est,
                            "capacity": cap,
                            "zip": "",
                        }

            if found_band:
                break

            pi = accounts_conn.get("pageInfo", {})
            if not pi.get("hasNextPage"):
                break
            cursor = pi.get("endCursor")
    except Exception:
        pass

    return lookup


def _match_city(city: str, lookup: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    key = city.lower().strip()
    if key in lookup:
        return lookup[key]
    for lk_city, details in lookup.items():
        if key in lk_city or lk_city in key:
            return details
    return None


# ---------------------------------------------------------------------------
# Weather (Open-Meteo)
# ---------------------------------------------------------------------------

def _geocode_city(city: str, state: str) -> tuple[float | None, float | None]:
    # Open-Meteo geocoder works best with city name alone (not "Orlando FL")
    # Try multiple variants: city alone, cleaned city, city + state
    directional = re.compile(r"\s+(North|South|East|West|Northeast|Northwest|Southeast|Southwest)$", re.I)
    variants = [city]
    cleaned = directional.sub("", city).strip()
    if cleaned != city:
        variants.append(cleaned)
    if state:
        variants.append(f"{city}, {state}")

    for query in variants:
        try:
            resp = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": query, "count": 1, "language": "en", "format": "json"},
                timeout=10,
            )
            results = resp.json().get("results", [])
            if results:
                return float(results[0]["latitude"]), float(results[0]["longitude"])
        except Exception:
            pass
    return None, None


def _fetch_weather(lat: float | None, lon: float | None, show_date: str) -> dict[str, float]:
    """Fetch weather for a date. Uses forecast API for future dates, archive API for past."""
    defaults = {"temperature_daily_mean": 15.0, "rain": 0.0, "snowfall": 0.0}
    if lat is None or lon is None:
        return defaults
    try:
        date_obj = dt.datetime.strptime(show_date, "%Y-%m-%d").date()
        today = dt.date.today()
        if date_obj > today:
            api_url = "https://api.open-meteo.com/v1/forecast"
        else:
            api_url = "https://archive-api.open-meteo.com/v1/archive"

        resp = requests.get(
            api_url,
            params={
                "latitude": lat, "longitude": lon,
                "start_date": show_date, "end_date": show_date,
                "daily": "temperature_2m_mean,rain_sum,snowfall_sum",
                "timezone": "auto",
            },
            timeout=15,
        )
        daily = resp.json().get("daily", {})
        return {
            "temperature_daily_mean": float((daily.get("temperature_2m_mean") or [defaults["temperature_daily_mean"]])[0] or defaults["temperature_daily_mean"]),
            "rain": float((daily.get("rain_sum") or [defaults["rain"]])[0] or defaults["rain"]),
            "snowfall": float((daily.get("snowfall_sum") or [defaults["snowfall"]])[0] or defaults["snowfall"]),
        }
    except Exception:
        return defaults


# ---------------------------------------------------------------------------
# Inventory file parser
# ---------------------------------------------------------------------------

def _parse_inventory_file(
    file_bytes: bytes,
    file_name: str,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Parse a raw AtVenu inventory/forecast file (CSV or xlsx).

    Returns (products_df, shows_list) where shows_list has city/date entries.
    """
    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else "csv"
    if ext in ("xlsx", "xls"):
        inventory_df = pd.read_excel(BytesIO(file_bytes))
    else:
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                text = file_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ConversionInputError("Could not decode uploaded file.")
        inventory_df = pd.read_csv(StringIO(text))

    show_pattern = re.compile(
        r"^(.+?)\s*-\s*(\d{1,2}/\d{1,2}/\d{2,4})\s*(?:\(.*\))?$"
    )
    shows: list[dict[str, str]] = []
    for col in inventory_df.columns:
        m = show_pattern.match(col.strip())
        if not m:
            continue
        city = m.group(1).strip()
        date_part = m.group(2).strip()
        try:
            date_obj = dt.datetime.strptime(date_part, "%m/%d/%y")
        except ValueError:
            try:
                date_obj = dt.datetime.strptime(date_part, "%m/%d/%Y")
            except ValueError:
                continue
        shows.append({"city": city, "date": date_obj.strftime("%Y-%m-%d")})

    if not shows:
        raise ConversionInputError(
            "No show columns found. Expected column headers like "
            "'City - MM/DD/YY ($X.XX/head)'."
        )

    skip_items = {"inventory forecast includes: on hand + in transit"}

    products: list[dict[str, Any]] = []
    for _, row in inventory_df.iterrows():
        item_name = row.get("Item Name")
        if pd.isna(item_name) or not str(item_name).strip():
            continue
        if str(item_name).strip().lower() in skip_items:
            continue
        products.append({
            "Item Name": str(item_name).strip(),
            "productType": str(row.get("Product Type", "Unknown")).strip(),
            "product size": str(row.get("Size", "ONE SIZE")).strip() or "ONE SIZE",
            "SKU": str(row.get("SKU", "")).strip(),
        })

    if not products:
        raise ConversionInputError("No product rows found in inventory file.")

    return pd.DataFrame(products), shows


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------

def convert_inventory_to_prediction_input(
    *,
    file_bytes: bytes,
    file_name: str,
    band_name: str,
    artist_meta_path: str | None = None,
    spotify_path: str | None = None,
    fetch_weather: bool = False,
    progress_callback: Any = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert a raw AtVenu inventory/forecast file into the 21-column
    prediction input format.

    Prices are fetched live from the AtVenu GraphQL API.
    Venue data is fetched from the AtVenu GraphQL API.
    """
    band_name = band_name.strip()
    if not band_name:
        raise ConversionInputError("Band name is required.")

    products_df, shows = _parse_inventory_file(file_bytes, file_name)

    if progress_callback:
        progress_callback(f"Parsed {len(products_df)} products and {len(shows)} shows")

    # --- SKU pricing from AtVenu API ---
    if progress_callback:
        progress_callback("Fetching SKU prices from AtVenu API...")
    sku_to_price = fetch_sku_prices_from_api(band_name)
    if progress_callback:
        progress_callback(f"Found {len(sku_to_price)} SKU prices from API")

    # --- Venue data from AtVenu API ---
    if progress_callback:
        progress_callback("Fetching venue data from AtVenu API...")
    venue_lookup = _fetch_atvenu_api_venues(band_name)

    # --- Enrichment CSVs ---
    artist_meta_df = _optional_csv(artist_meta_path)
    spotify_df = _optional_csv(spotify_path)

    genre = _genre_for_band(band_name, artist_meta_df)
    instagram = _instagram_for_band(band_name, artist_meta_df)
    spotify_listeners, spotify_missing = _spotify_for_band(band_name, spotify_df)

    # --- Build output rows ---
    output_rows: list[dict[str, Any]] = []
    venue_match_count = 0
    price_miss_skus: list[str] = []

    for show in shows:
        show_date = show["date"]
        show_date_obj = dt.datetime.strptime(show_date, "%Y-%m-%d").date()
        holiday_status = _holiday_status(show_date_obj)

        venue_info = _match_city(show["city"], venue_lookup)
        if venue_info:
            venue_match_count += 1

        venue_name = venue_info["venue"] if venue_info else ""
        venue_state = venue_info["state"] if venue_info else US_CITY_STATE.get(show["city"], "")
        attendance = venue_info["attendance"] if venue_info else 0
        capacity = venue_info["capacity"] if venue_info else 0
        postal_code = venue_info.get("zip", "") if venue_info else ""
        venue_country = _venue_country(venue_state) if venue_state else "United States"

        weather = {"temperature_daily_mean": 15.0, "rain": 0.0, "snowfall": 0.0}
        if fetch_weather:
            lat, lon = _geocode_city(show["city"], venue_state)
            weather = _fetch_weather(lat, lon, show_date)

        for _, product in products_df.iterrows():
            sku = str(product["SKU"])
            price = sku_to_price.get(sku)
            if price is None and sku and sku != "nan":
                price_miss_skus.append(sku)

            output_rows.append({
                "artistName": band_name,
                "Genre": genre,
                "showDate": show_date,
                "HolidayStatus": holiday_status,
                "venue name": venue_name,
                "venue city": show["city"],
                "venue state": venue_state,
                "venue country": venue_country,
                "venue postalCode": postal_code,
                "merch category": str(product["Item Name"]),
                "productType": str(product["productType"]),
                "product size": str(product["product size"]),
                "attendance": int(attendance),
                "product price": float(price) if price is not None else float("nan"),
                "temperature_daily_mean": float(weather["temperature_daily_mean"]),
                "rain": float(weather["rain"]),
                "snowfall": float(weather["snowfall"]),
                "spotifyMonthlyListeners": int(spotify_listeners),
                "Instagram": int(instagram),
                "venue capacity": int(capacity),
                "spotifyMissing": int(spotify_missing),
            })

    out_df = pd.DataFrame(output_rows)
    out_df = out_df[OUTPUT_COLUMNS]

    unique_price_misses = sorted(set(price_miss_skus))
    metadata = {
        "band_name": band_name,
        "shows_parsed": len(shows),
        "products_parsed": len(products_df),
        "rows_out": int(out_df.shape[0]),
        "venue_matches": venue_match_count,
        "venue_misses": len(shows) - venue_match_count,
        "price_miss_skus": unique_price_misses,
        "prices_found": len(sku_to_price),
        "fetch_weather": bool(fetch_weather),
        "show_dates": [s["date"] for s in shows],
        "show_cities": [s["city"] for s in shows],
    }
    return out_df, metadata
