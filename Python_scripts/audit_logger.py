import os
import csv
import json
import hashlib
import datetime as dt
from typing import Any, Dict, Optional

DEBUG_DIR_NAME = "debug_info"
AUDIT_JSONL_NAME = "audit_events.jsonl"
AUDIT_CSV_NAME = "audit_events.csv"

def _find_paths_config(start_dir: str, filename: str = "paths_config.txt") -> str:
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

def get_repo_root(start_dir: Optional[str] = None) -> str:
    start_dir = start_dir or os.path.dirname(os.path.abspath(__file__))
    cfg = _find_paths_config(start_dir)
    return os.path.dirname(os.path.abspath(cfg))

def get_debug_dir(start_dir: Optional[str] = None) -> str:

    repo_root = get_repo_root(start_dir=start_dir)
    debug_dir = os.path.join(repo_root, DEBUG_DIR_NAME)
    os.makedirs(debug_dir, exist_ok=True)
    return debug_dir

def sha256_bytes(data: bytes, n: int = 16) -> str:
    return hashlib.sha256(data).hexdigest()[:n]

def sha256_file(path: str, n: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:n]

def _truncate(val: Any, max_len: int = 8000) -> Any:

    if isinstance(val, str) and len(val) > max_len:
        return val[:max_len] + f"...[truncated {len(val) - max_len} chars]"
    return val

def _as_json_str(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return json.dumps({"unserializable_details": str(obj)}, ensure_ascii=False)

def log_event(
    step: str,
    status: str,
    details: Optional[Dict[str, Any]] = None,
    debug_dir: Optional[str] = None,
) -> None:

    try:
        debug_dir = debug_dir or get_debug_dir()
        os.makedirs(debug_dir, exist_ok=True)

        event = {
            "ts": dt.datetime.now().isoformat(timespec="seconds"),
            "step": step,
            "status": status,
            "details": details or {},
        }

        jsonl_path = os.path.join(debug_dir, AUDIT_JSONL_NAME)
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

        csv_path = os.path.join(debug_dir, AUDIT_CSV_NAME)
        write_header = not os.path.exists(csv_path)

        det = details or {}
        input_file = det.get("input_file", "")
        input_hash = det.get("input_hash", "")
        rows_in = det.get("rows_in", det.get("input_rows", ""))
        rows_out = det.get("rows_out", det.get("output_rows", ""))
        shows = det.get("shows", "")
        returncode = det.get("returncode", "")
        error = det.get("error", "")

        warnings_val = det.get("warnings", "")
        messages_val = det.get("messages", "")

        row = {
            "ts": event["ts"],
            "step": step,
            "status": status,
            "input_file": _truncate(input_file),
            "input_hash": _truncate(input_hash),
            "rows_in": rows_in,
            "rows_out": rows_out,
            "shows": shows,
            "returncode": returncode,
            "error": _truncate(error),
            "warnings": _truncate(warnings_val),
            "messages": _truncate(messages_val),
            "details_json": _truncate(_as_json_str(det)),
        }

        fieldnames = list(row.keys())
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            w.writerow(row)

    except Exception:
        return
