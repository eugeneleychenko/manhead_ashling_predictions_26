#!/usr/bin/env python3
"""
Automated retrain: pull all AtVenu sales/tour data, then retrain the model.

Run on the DO droplet:
    /opt/manhead/.venv/bin/python /opt/manhead/scripts/auto_retrain.py

What it does:
  1. Iterates all AtVenu accounts (bands)
  2. For each band with settled shows, pulls per-show sales reports + tour summary
  3. Writes CSVs to the standard input folders
  4. Triggers a single retrain (consolidation → training → R² validation)
  5. Logs everything to a JSON file incrementally

The retrain uses the existing retrain_worker.py pipeline.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
import sys
import csv
from collections import OrderedDict

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(REPO_ROOT, "paths_config.txt")
SALES_DIR = os.path.join(REPO_ROOT, "CSVs", "add_sales_reports_files_here")
TOUR_DIR = os.path.join(REPO_ROOT, "CSVs", "add_tour_summary_files_here")
LOG_DIR = os.path.join(REPO_ROOT, "outputs")

ATVENU_API = "https://api.atvenu.com"
ATVENU_KEY = os.environ.get("ATVENU_API_TOKEN", "live_yvYLBo32dRE9z_yCdhwU")
HEADERS = {"Content-Type": "application/json", "x-api-key": ATVENU_KEY}
TIMEOUT = 30

APPAREL_TYPES = {
    "T-Shirt", "Pullover Hoodie", "Hoodie", "Zip-Up Hoodie",
    "Tank Top", "Long Sleeve", "Sweatshirt", "Jacket", "Jersey",
    "Crewneck", "Shorts", "Pants", "Crop Top",
}
MUSIC_TYPES = {"CD", "Vinyl", "LP", "Cassette", "Digital Download"}
SIZE_ORDER = {"S": 0, "M": 1, "L": 2, "XL": 3, "2XL": 4, "3XL": 5, "One-Size": 6, "": 7}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
class IncrementalLog:
    """Writes log to disk after every update."""

    def __init__(self, path: str):
        self.path = path
        self.data = {
            "started_at": dt.datetime.now().isoformat(timespec="seconds"),
            "finished_at": None,
            "bands_processed": 0,
            "bands_skipped": 0,
            "total_sales_files": 0,
            "total_tour_files": 0,
            "retrain": None,
            "bands": [],
        }
        self._flush()

    def add_band(self, entry: dict):
        self.data["bands"].append(entry)
        self.data["bands_processed"] = sum(
            1 for b in self.data["bands"] if b.get("status") == "ok"
        )
        self.data["bands_skipped"] = sum(
            1 for b in self.data["bands"] if b.get("status") == "skipped"
        )
        self.data["total_sales_files"] = sum(
            b.get("sales_files", 0) for b in self.data["bands"]
        )
        self.data["total_tour_files"] = sum(
            b.get("tour_files", 0) for b in self.data["bands"]
        )
        self._flush()

    def set_retrain(self, result: dict):
        self.data["retrain"] = result
        self._flush()

    def finish(self):
        self.data["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
        self._flush()

    def _flush(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)


# ---------------------------------------------------------------------------
# GraphQL helpers
# ---------------------------------------------------------------------------
def _gql(query: str, variables: dict | None = None) -> dict:
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    resp = requests.post(ATVENU_API, json=payload, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    body = resp.json()
    if "errors" in body:
        raise RuntimeError(f"GraphQL errors: {json.dumps(body['errors'])}")
    return body["data"]


def _flt(val) -> float:
    if val is None:
        return 0.0
    return float(val)


def _slugify(text: str, maxlen: int = 40) -> str:
    """Convert band name to filename slug. Handles (MH) suffix cleanly."""
    s = text.strip()
    # "Deftones (MH)" → "deftones-mh"
    s = s.replace("(", "").replace(")", "")
    s = re.sub(r"[^a-z0-9]+", "-", s.lower().strip())
    return s[:maxlen].strip("-")


def _category(ptype: str) -> str:
    if ptype in APPAREL_TYPES:
        return "APPAREL"
    if ptype in MUSIC_TYPES:
        return "MUSIC"
    return "OTHER"


# ---------------------------------------------------------------------------
# AtVenu data fetching
# ---------------------------------------------------------------------------
def fetch_all_accounts() -> list[dict]:
    """Fetch all accounts (bands) from AtVenu."""
    accounts = []
    cursor = None
    for _ in range(40):
        data = _gql(
            """query($after: String) {
              organization {
                accounts(first: 50, after: $after) {
                  pageInfo { hasNextPage endCursor }
                  nodes { uuid name }
                }
              }
            }""",
            {"after": cursor},
        )
        accts = data["organization"]["accounts"]
        accounts.extend(accts["nodes"])
        if not accts["pageInfo"]["hasNextPage"]:
            break
        cursor = accts["pageInfo"]["endCursor"]
    return accounts


def fetch_tours(acct_uuid: str) -> list[dict]:
    data = _gql(
        """query($uuid: UUID!) {
          node(uuid: $uuid) {
            ... on Account {
              tours(first: 50) {
                nodes { uuid name startDate endDate }
              }
            }
          }
        }""",
        {"uuid": acct_uuid},
    )
    return data["node"]["tours"]["nodes"]


def fetch_merch_catalog(acct_uuid: str) -> dict[str, dict]:
    data = _gql(
        """query($uuid: UUID!) {
          node(uuid: $uuid) {
            ... on Account {
              merchItems(first: 200) {
                nodes {
                  uuid name
                  productType { name }
                  merchVariants { uuid sku size price }
                }
              }
            }
          }
        }""",
        {"uuid": acct_uuid},
    )
    lookup = {}
    for item in data["node"]["merchItems"]["nodes"]:
        item_uuid = item["uuid"]
        item_name = item["name"].strip()
        item_type = item["productType"]["name"]
        for v in item["merchVariants"]:
            lookup[v["uuid"]] = {
                "sku": (v.get("sku") or "").strip(),
                "name": item_name,
                "type": item_type,
                "size": v.get("size") or "",
                "price": _flt(v.get("price")),
                "item_uuid": item_uuid,
            }
    return lookup


def fetch_tour_shows(tour_uuid: str) -> list[dict]:
    data = _gql(
        """query($uuid: UUID!) {
          node(uuid: $uuid) {
            ... on Tour {
              name
              shows(first: 100) {
                nodes {
                  uuid showDate state attendance capacity
                  location { name city stateProvince postalCode country }
                  currencyFormat { code }
                  settlements {
                    uuid status exchangeRate
                    expenses { description costAmount taxAmount expenseType }
                    mainCounts(first: 300) {
                      nodes {
                        merchItemUuid merchVariantUuid
                        countIn countOut comps
                        priceOverride calculatedPriceWithTax
                      }
                    }
                    settlementOutput {
                      total {
                        grossSalesAmount adjustedGrossSalesAmount
                        artistCutAmount artistDueAmount
                        venueCutAmount vendFeeAmount
                        taxOnSalesAmount paymentFeesAmount
                        artistReceiptAmount
                      }
                    }
                  }
                }
              }
            }
          }
        }""",
        {"uuid": tour_uuid},
    )
    return data["node"]["shows"]["nodes"]


# ---------------------------------------------------------------------------
# CSV writers (adapted from pull_atvenu_reports.py)
# ---------------------------------------------------------------------------
def write_sales_report(show: dict, variant_lookup: dict, band_slug: str,
                       tour_slug: str, outdir: str) -> str | None:
    if show["state"] != "DONE":
        return None
    settlement = show["settlements"][0] if show.get("settlements") else None
    if not settlement or settlement["status"] != "DONE":
        return None
    if not settlement.get("settlementOutput"):
        return None

    show_date = show["showDate"]
    loc = show.get("location") or {}
    city = loc.get("city") or "unknown"

    variant_sales = {}
    for mc in settlement["mainCounts"]["nodes"]:
        v_uuid = mc["merchVariantUuid"]
        count_in = mc.get("countIn") or 0
        count_out = mc.get("countOut") or 0
        comps = mc.get("comps") or 0
        sold = max(0, count_in - count_out - comps)

        info = variant_lookup.get(v_uuid, {})
        price_override = _flt(mc.get("priceOverride"))
        calc_price = _flt(mc.get("calculatedPriceWithTax"))
        base_price = info.get("price", 0)
        effective_price = (price_override if price_override > 0
                           else calc_price if calc_price > 0
                           else base_price)
        gross = sold * effective_price

        variant_sales[v_uuid] = {
            "sku": info.get("sku", ""),
            "name": info.get("name", "Unknown"),
            "type": info.get("type", "Other"),
            "size": info.get("size", ""),
            "sold": sold, "comps": comps,
            "avg_price": gross / sold if sold > 0 else 0,
            "gross": gross,
            "item_uuid": info.get("item_uuid", mc["merchItemUuid"]),
            "category": _category(info.get("type", "Other")),
        }

    items_by_cat = {"APPAREL": OrderedDict(), "OTHER": OrderedDict(), "MUSIC": OrderedDict()}
    for vs in variant_sales.values():
        cat = vs["category"]
        iuuid = vs["item_uuid"]
        items_by_cat[cat].setdefault(iuuid, []).append(vs)

    city_slug = re.sub(r"[^a-z0-9]+", "-", city.lower().strip())[:20].strip("-")
    # Format date as MM-DD-YYYY for filename (consolidation pipeline expects this)
    date_parts = show_date.split("-")  # YYYY-MM-DD
    date_for_file = f"{date_parts[1]}-{date_parts[2]}-{date_parts[0]}"
    filename = f"{band_slug}_Sales-Report_{tour_slug}-for-{date_for_file}.csv"
    filepath = os.path.join(outdir, filename)

    with open(filepath, "w", newline="") as f:
        for cat in ["APPAREL", "OTHER", "MUSIC"]:
            f.write(f"{cat}\n")
            f.write("SKU,Name,Type,Sex,Size,Sold,Unit % of Total,Comp,Avg. Price,Gross Rev,% of Total\n")

            cat_items = items_by_cat[cat]
            cat_total_sold = sum(v["sold"] for vl in cat_items.values() for v in vl)
            cat_total_gross = sum(v["gross"] for vl in cat_items.values() for v in vl)

            for iuuid, variants in cat_items.items():
                item_sold = sum(v["sold"] for v in variants)
                item_gross = sum(v["gross"] for v in variants)
                variants.sort(key=lambda v: SIZE_ORDER.get(v["size"], 99))

                for v in variants:
                    pct_u = f"{v['sold']/item_sold*100:.0f}%" if item_sold else "0%"
                    pct_r = f"{v['gross']/item_gross*100:.0f}%" if item_gross else "0%"
                    avg_p = f"${v['avg_price']:.2f}" if v["avg_price"] > 0 else "$0.00"
                    gr = f'"${v["gross"]:,.2f}"' if v["gross"] > 0 else "$0.00"
                    sex = "U" if cat == "APPAREL" else ""
                    f.write(f"{v['sku']},{v['name']},{v['type']},{sex},{v['size']},"
                            f"{v['sold']},{pct_u},{v['comps']},{avg_p},{gr},{pct_r}\n")

                sex = "U" if cat == "APPAREL" else ""
                sub_pct = f"{item_sold/cat_total_sold*100:.0f}%" if cat_total_sold else "0%"
                sub_rpct = f"{item_gross/cat_total_gross*100:.0f}%" if cat_total_gross else "0%"
                avg_i = f"${item_gross/item_sold:.2f}" if item_sold else "$0.00"
                f.write(f'SUBTOTAL,{variants[0]["name"]},{variants[0]["type"]},{sex},"",'
                        f'{item_sold},{sub_pct},{sum(v["comps"] for v in variants)},'
                        f'{avg_i},"${item_gross:,.2f}",{sub_rpct}\n\n')

            cat_comps = sum(v["comps"] for vl in cat_items.values() for v in vl)
            f.write(f'TOTAL {cat},"","","","",{cat_total_sold},100%,{cat_comps},'
                    f'"","${cat_total_gross:,.2f}",100%\n\n\n')

        grand_sold = sum(v["sold"] for ci in items_by_cat.values() for vl in ci.values() for v in vl)
        grand_comps = sum(v["comps"] for ci in items_by_cat.values() for vl in ci.values() for v in vl)
        grand_gross = sum(v["gross"] for ci in items_by_cat.values() for vl in ci.values() for v in vl)
        f.write(f'GRAND TOTAL,"","","","",{grand_sold},"",{grand_comps},"","${grand_gross:,.2f}"\n')

    return filepath


def write_tour_summary(shows: list[dict], band_name: str, band_slug: str,
                       tour_name: str, tour_slug: str, outdir: str) -> str | None:
    done_shows = [s for s in shows if s["state"] == "DONE"
                  and s.get("settlements")
                  and s["settlements"][0]["status"] == "DONE"
                  and s["settlements"][0].get("settlementOutput")]
    if not done_shows:
        return None

    done_shows.sort(key=lambda s: s["showDate"])
    first_date = done_shows[0]["showDate"]
    last_date = done_shows[-1]["showDate"]

    filename = (f"{band_slug}_Tour-Summary_{tour_slug}"
                f"-for-{first_date}-to-{last_date}.csv")
    filepath = os.path.join(outdir, filename)

    def _fmt_date(d: str) -> str:
        parts = d.split("-")
        return f"{parts[1]}/{parts[2]}/{parts[0]}"

    rows = []
    for show in done_shows:
        stl = show["settlements"][0]
        totals = stl["settlementOutput"]["total"]
        loc = show.get("location") or {}
        ccy = show.get("currencyFormat", {}).get("code", "USD")
        exch = _flt(stl.get("exchangeRate")) or 1.0
        gross = _flt(totals.get("grossSalesAmount"))
        attn = show.get("attendance") or 0
        cap = show.get("capacity") or 0
        tax = _flt(totals.get("taxOnSalesAmount"))
        pay_fees = _flt(totals.get("paymentFeesAmount"))
        venue_cut = _flt(totals.get("venueCutAmount"))
        vend_fee = _flt(totals.get("vendFeeAmount"))
        artist_cut = _flt(totals.get("artistCutAmount"))
        adj_gross = _flt(totals.get("adjustedGrossSalesAmount"))
        venue_pct = f"{venue_cut / adj_gross * 100:.0f}%" if adj_gross > 0 else "0%"

        ext_exp = bootleg_exp = selling_exp = 0.0
        for exp in stl.get("expenses") or []:
            cost = _flt(exp.get("costAmount"))
            etype = (exp.get("expenseType") or "").upper()
            desc = (exp.get("description") or "").lower()
            if "bootleg" in desc or etype == "BOOTLEG":
                bootleg_exp += cost
            elif etype == "SELLING" or "selling" in desc:
                selling_exp += cost
            else:
                ext_exp += cost

        total_selling = venue_cut + vend_fee + ext_exp + bootleg_exp
        per_head = gross / attn if attn > 0 else 0
        net = artist_cut

        rows.append({
            "date": _fmt_date(show["showDate"]),
            "city": loc.get("city") or "",
            "state": loc.get("stateProvince") or "",
            "zip": loc.get("postalCode") or "",
            "venue": loc.get("name") or "",
            "venue_pct": venue_pct,
            "capacity": cap, "attend": attn,
            "currency": ccy, "exch_rate": exch,
            "per_head": per_head * exch if exch != 1.0 else per_head,
            "gross": gross * exch if exch != 1.0 else gross,
            "tax": tax * exch if exch != 1.0 else tax,
            "pay_fees": pay_fees * exch if exch != 1.0 else pay_fees,
            "venue_fee": venue_cut * exch if exch != 1.0 else venue_cut,
            "venue_adj": 0.0,
            "vend_fee": vend_fee * exch if exch != 1.0 else vend_fee,
            "ext_exp": ext_exp * exch if exch != 1.0 else ext_exp,
            "bootleg_exp": bootleg_exp * exch if exch != 1.0 else bootleg_exp,
            "selling_exp": total_selling * exch if exch != 1.0 else total_selling,
            "net_receipts": net * exch if exch != 1.0 else net,
        })

    with open(filepath, "w", newline="") as f:
        f.write("\n")
        f.write("Tour Summary\n")
        f.write(f"Artist: {band_name}\n")
        f.write(f"{tour_name} for {_fmt_date(first_date)} to {_fmt_date(last_date)}\n")
        cols = ["Date", "City", "State", "Zip", "Venue", "Venue Actual %",
                "Capacity", "Attend", "Currency", "Exch. Rate", "Per Head",
                "Gross", "Tax", "Payment Fees", "Venue Fee", "Venue Adjust.",
                "Vend Fee", "Ext Exp", "Bootleg Exp", "Selling Exp", "Net Receipts"]
        f.write(",".join(cols) + "\n")

        def _money(v):
            return f'"${v:,.2f}"' if abs(v) >= 1000 else f"${v:,.2f}"

        def _num(v):
            return f'"{v:,}"' if v >= 1000 else str(v)

        for r in rows:
            f.write(f'{r["date"]},{r["city"]},"{r["state"]}",{r["zip"]},'
                    f'{r["venue"]},{r["venue_pct"]},'
                    f'{_num(r["capacity"])},{_num(r["attend"])},'
                    f'{r["currency"]},{r["exch_rate"]},'
                    f'${r["per_head"]:.2f},'
                    f'{_money(r["gross"])},{_money(r["tax"])},'
                    f'{_money(r["pay_fees"])},{_money(r["venue_fee"])},'
                    f'{_money(r["venue_adj"])},{_money(r["vend_fee"])},'
                    f'{_money(r["ext_exp"])},{_money(r["bootleg_exp"])},'
                    f'{_money(r["selling_exp"])},{_money(r["net_receipts"])}\n')

    return filepath


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"auto_retrain_{timestamp}.json")
    log = IncrementalLog(log_path)

    print(f"=== Auto Retrain — {timestamp} ===")
    print(f"Log: {log_path}")
    print()

    os.makedirs(SALES_DIR, exist_ok=True)
    os.makedirs(TOUR_DIR, exist_ok=True)

    # ── Phase 1: Pull data from AtVenu ──
    print("Phase 1: Fetching all accounts from AtVenu...")
    accounts = fetch_all_accounts()
    print(f"  Found {len(accounts)} accounts\n")

    for i, acct in enumerate(accounts, 1):
        band_name = acct["name"]
        band_slug = _slugify(band_name)
        prefix = f"[{i}/{len(accounts)}] {band_name}"

        try:
            tours = fetch_tours(acct["uuid"])
        except Exception as e:
            print(f"  {prefix}: ERROR fetching tours — {e}")
            log.add_band({"name": band_name, "status": "error", "error": str(e),
                          "sales_files": 0, "tour_files": 0})
            continue

        if not tours:
            print(f"  {prefix}: no tours, skipping")
            log.add_band({"name": band_name, "status": "skipped", "reason": "no tours",
                          "sales_files": 0, "tour_files": 0})
            continue

        # Fetch merch catalog once per band
        try:
            variant_lookup = fetch_merch_catalog(acct["uuid"])
        except Exception as e:
            print(f"  {prefix}: ERROR fetching merch — {e}")
            log.add_band({"name": band_name, "status": "error", "error": str(e),
                          "sales_files": 0, "tour_files": 0})
            continue

        band_sales = 0
        band_tours = 0
        band_shows = 0

        for tour in tours:
            tour_name = tour["name"]
            tour_slug = _slugify(tour_name)

            try:
                shows = fetch_tour_shows(tour["uuid"])
            except Exception as e:
                print(f"  {prefix}: ERROR fetching shows for '{tour_name}' — {e}")
                continue

            settled = [s for s in shows if s["state"] == "DONE"
                       and s.get("settlements")
                       and s["settlements"][0]["status"] == "DONE"
                       and s["settlements"][0].get("settlementOutput")]
            if not settled:
                continue

            # Write per-show sales reports
            for show in shows:
                fp = write_sales_report(show, variant_lookup, band_slug, tour_slug, SALES_DIR)
                if fp:
                    band_sales += 1

            # Write tour summary
            fp = write_tour_summary(shows, band_name, band_slug, tour_name, tour_slug, TOUR_DIR)
            if fp:
                band_tours += 1

            band_shows += len(settled)

        if band_sales == 0 and band_tours == 0:
            print(f"  {prefix}: no settled shows, skipping")
            log.add_band({"name": band_name, "status": "skipped",
                          "reason": "no settled shows",
                          "sales_files": 0, "tour_files": 0})
        else:
            print(f"  {prefix}: {band_shows} shows, {band_sales} sales files, {band_tours} tour files")
            log.add_band({"name": band_name, "status": "ok",
                          "shows": band_shows,
                          "sales_files": band_sales, "tour_files": band_tours,
                          "variants": len(variant_lookup)})

    print(f"\nPhase 1 complete: {log.data['bands_processed']} bands with data, "
          f"{log.data['bands_skipped']} skipped")
    print(f"  Sales files: {log.data['total_sales_files']}")
    print(f"  Tour files: {log.data['total_tour_files']}")

    if log.data["total_sales_files"] == 0:
        print("\nNo sales data pulled. Nothing to retrain on.")
        log.set_retrain({"status": "skipped", "reason": "no data"})
        log.finish()
        return

    # ── Phase 2: Retrain ──
    print("\n" + "=" * 60)
    print("Phase 2: Triggering retrain...")
    print("=" * 60)

    retrain_script = os.path.join(REPO_ROOT, "Python_scripts", "retrain_worker.py")
    if not os.path.exists(retrain_script):
        print(f"ERROR: retrain_worker.py not found at {retrain_script}")
        log.set_retrain({"status": "error", "reason": "retrain_worker.py not found"})
        log.finish()
        return

    # Read old metrics
    metrics_path = os.path.join(REPO_ROOT, "Flask", "last_train_metrics.json")
    old_metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            old_metrics = json.load(f)
        print(f"  Old model: R²={old_metrics.get('r2_test', 'N/A'):.4f}, "
              f"RMSE={old_metrics.get('rmse_test', 'N/A'):.4f}, "
              f"rows={old_metrics.get('rows_total', 'N/A')}")

    env = os.environ.copy()
    env["SKIP_SPOTIFY"] = "1"

    try:
        result = subprocess.run(
            [sys.executable, retrain_script, CONFIG_PATH],
            cwd=REPO_ROOT, env=env,
            capture_output=True, text=True, timeout=3600,
        )
    except subprocess.TimeoutExpired:
        print("  ERROR: Retrain timed out after 1 hour")
        log.set_retrain({"status": "error", "reason": "timeout"})
        log.finish()
        return

    print(result.stdout[-3000:] if result.stdout else "(no stdout)")
    if result.stderr:
        print(f"  stderr: {result.stderr[-1000:]}")

    # Read new metrics
    new_metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            new_metrics = json.load(f)

    retrain_result = {
        "status": "completed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "old_metrics": old_metrics,
        "new_metrics": new_metrics,
    }

    if result.returncode == 0 and new_metrics:
        new_r2 = new_metrics.get("r2_test", 0)
        old_r2 = old_metrics.get("r2_test", 0) if old_metrics else 0
        delta = ((new_r2 - old_r2) / abs(old_r2) * 100) if old_r2 else 0
        retrain_result["r2_delta_pct"] = round(delta, 2)

        print(f"\n  New model: R²={new_r2:.4f}, "
              f"RMSE={new_metrics.get('rmse_test', 0):.4f}, "
              f"rows={new_metrics.get('rows_total', 'N/A')}")
        print(f"  R² change: {old_r2:.4f} → {new_r2:.4f} ({delta:+.1f}%)")

        if result.returncode == 0:
            print("\n  ✓ Retrain successful — new model promoted")
    else:
        print(f"\n  ✗ Retrain failed (exit code {result.returncode})")
        if result.returncode == 1:
            # Check for validation failure
            staging_fail = os.path.join(REPO_ROOT, "Flask", ".staging", "validation_failure.json")
            if os.path.exists(staging_fail):
                with open(staging_fail) as f:
                    fail_info = json.load(f)
                retrain_result["validation_failure"] = fail_info
                print(f"  Validation gate blocked: {fail_info.get('reason', 'unknown')}")

    log.set_retrain(retrain_result)
    log.finish()

    print(f"\n{'=' * 60}")
    print(f"Done! Log saved to: {log_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
