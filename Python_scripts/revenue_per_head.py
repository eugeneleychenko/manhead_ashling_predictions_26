import os
import sys
import json
import argparse
import pandas as pd
from typing import List
from audit_logger import log_event, sha256_file


def _safe_log(step: str, status: str, details: dict):
    try:
        log_event(step=step, status=status, details=details)
    except Exception:
        pass

def to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(r"[^0-9\.\-]", "", regex=True),
        errors="coerce",
    )

def ensure_col(df: pd.DataFrame, col: str, default, warnings: List[str]) -> pd.DataFrame:
    if col not in df.columns:
        df[col] = default
        warnings.append(f"Missing column '{col}'. Created with default={default!r}.")
    return df

def run(input_csv: str, output_csv: str, summary_json: str, round_decimals: int = 3) -> None:
    warnings: List[str] = []

    _safe_log(
        step="step5_revph_script",
        status="start",
        details={
            "input_csv": input_csv,
            "output_csv": output_csv,
            "summary_json": summary_json,
            "round_decimals": int(round_decimals),
            "input_exists": bool(os.path.exists(input_csv)),
            "input_hash": sha256_file(input_csv) if os.path.exists(input_csv) else "",
            "output_exists_before": bool(os.path.exists(output_csv)),
            "summary_exists_before": bool(os.path.exists(summary_json)),
        },
    )

    try:
        df = pd.read_csv(input_csv)

        required_cols = [
            "artistName",
            "attendance",
            "predicted_sales_quantity",
            "product price",
            "showDate",
            "venue name",
            "venue city",
            "venue state",
        ]

        df = ensure_col(df, "artistName", "unknown_artist", warnings)
        df = ensure_col(df, "venue name", "unknown_venue", warnings)
        df = ensure_col(df, "venue city", "unknown_city", warnings)
        df = ensure_col(df, "venue state", "unknown_state", warnings)
        df = ensure_col(df, "showDate", "", warnings)
        df = ensure_col(df, "attendance", pd.NA, warnings)
        df = ensure_col(df, "predicted_sales_quantity", pd.NA, warnings)
        df = ensure_col(df, "product price", pd.NA, warnings)

        show_key_cols = ["artistName", "venue name", "venue city", "venue state", "showDate"]

        qty_num = to_number(df["predicted_sales_quantity"])
        price_num = to_number(df["product price"])
        attendance_num = to_number(df["attendance"])

        predicted_revenue_row = qty_num * price_num

        inconsistent_shows = 0
        try:
            attendance_nunique = (
                pd.DataFrame({"attendance_num": attendance_num})
                .join(df[show_key_cols])
                .groupby(show_key_cols, dropna=False)["attendance_num"]
                .nunique(dropna=True)
                .reset_index(name="attendance_nunique")
            )
            inconsistent_shows = int((attendance_nunique["attendance_nunique"] > 1).sum())
            if inconsistent_shows > 0:
                warnings.append(
                    f"Attendance inconsistency detected: {inconsistent_shows} show(s) have multiple attendance values across rows. "
                    "Using max(attendance) per show."
                )
        except Exception as e:
            warnings.append(f"Could not compute attendance consistency check: {e}")

        try:
            show_agg = (
                pd.DataFrame(
                    {
                        "predicted_revenue_row": predicted_revenue_row,
                        "attendance_num": attendance_num,
                    }
                )
                .join(df[show_key_cols])
                .groupby(show_key_cols, dropna=False)
                .agg(
                    show_predicted_revenue=("predicted_revenue_row", "sum"),
                    show_attendance=("attendance_num", "max"),
                )
                .reset_index()
            )

            show_agg["$/head"] = show_agg["show_predicted_revenue"] / show_agg["show_attendance"].replace({0: pd.NA})
            show_agg["$/head"] = pd.to_numeric(show_agg["$/head"], errors="coerce").round(round_decimals)

            df_out = df.merge(
                show_agg[show_key_cols + ["$/head"]],
                on=show_key_cols,
                how="left",
            )

            df_out["$/head"] = pd.to_numeric(df_out["$/head"], errors="coerce").round(round_decimals)

        except Exception as e:
            warnings.append(f"Aggregation failed; outputting '$/head' as NA. Error: {e}")
            df_out = df.copy()
            df_out["$/head"] = pd.NA

        df_out.to_csv(output_csv, index=False)

        total_predicted_sales = float(qty_num.sum(skipna=True)) if qty_num.notna().any() else 0.0
        avg_sales_per_product = float(qty_num.mean(skipna=True)) if qty_num.notna().any() else 0.0

        product_types = int(df_out["productType"].nunique(dropna=True)) if "productType" in df_out.columns else 0
        shows = int(df_out[show_key_cols].drop_duplicates().shape[0]) if all(c in df_out.columns for c in show_key_cols) else 0
        rows_with_head = int(pd.notna(df_out.get("$/head")).sum()) if "$/head" in df_out.columns else 0

        summary = {
            "required_cols_expected": required_cols,
            "show_key_cols": show_key_cols,
            "input_rows": int(df.shape[0]),
            "output_rows": int(df_out.shape[0]),
            "shows": shows,
            "rows_with_$/head": rows_with_head,
            "attendance_inconsistent_shows": inconsistent_shows,
            "total_predicted_sales": total_predicted_sales,
            "avg_sales_per_product": avg_sales_per_product,
            "product_types": product_types,
            "round_decimals": int(round_decimals),
            "warnings": warnings,
        }

        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        out_hash = sha256_file(output_csv) if os.path.exists(output_csv) else ""
        summary_hash = sha256_file(summary_json) if os.path.exists(summary_json) else ""

        _safe_log(
            step="step5_revph_script",
            status="success",
            details={
                "input_csv": input_csv,
                "input_hash": sha256_file(input_csv) if os.path.exists(input_csv) else "",
                "output_csv": output_csv,
                "output_hash": out_hash,
                "summary_json": summary_json,
                "summary_hash": summary_hash,
                "input_rows": int(df.shape[0]),
                "output_rows": int(df_out.shape[0]),
                "shows": int(shows),
                "rows_with_$/head": int(rows_with_head),
                "attendance_inconsistent_shows": int(inconsistent_shows),
                "product_types": int(product_types),
                "total_predicted_sales": float(total_predicted_sales),
                "avg_sales_per_product": float(avg_sales_per_product),
                "round_decimals": int(round_decimals),
                "warnings_count": int(len(warnings)),
                "warnings_sample": warnings[:25],
            },
        )

    except Exception as e:
        _safe_log(
            step="step5_revph_script",
            status="error",
            details={
                "input_csv": input_csv,
                "output_csv": output_csv,
                "summary_json": summary_json,
                "round_decimals": int(round_decimals),
                "error": str(e),
                "warnings_count": int(len(warnings)),
                "warnings_sample": warnings[:25],
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description="Step 5: compute show-level $/head from Step 4 output CSV")
    parser.add_argument("--input", required=True, help="Path to input CSV (Step 4 output)")
    parser.add_argument("--output", required=True, help="Path to output CSV (Step 5 output)")
    parser.add_argument("--summary", required=True, help="Path to summary JSON output")
    parser.add_argument("--round", type=int, default=3, help="Decimal rounding for $/head")
    args = parser.parse_args()

    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if os.path.dirname(args.summary):
        os.makedirs(os.path.dirname(args.summary), exist_ok=True)

    run(args.input, args.output, args.summary, round_decimals=args.round)
    print("OK: Step 5 completed successfully.")
    print(f"Output CSV: {args.output}")
    print(f"Summary JSON: {args.summary}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
