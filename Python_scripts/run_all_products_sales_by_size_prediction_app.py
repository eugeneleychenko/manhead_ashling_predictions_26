import os
from audit_logger import log_event
from predict_all_products_sales_by_size_app import app

def _safe_log(step: str, status: str, details: dict):
    try:
        log_event(step=step, status=status, details=details)
    except Exception:
        pass

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

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = find_paths_config(script_dir)
    paths = load_paths(config_path)
    port = int(paths.get("flask_port", "5000"))

    _safe_log(
        step="flask_service",
        status="start",
        details={
            "port": port,
            "config_path": config_path,
            "cwd": os.getcwd(),
        },
    )

    try:
        app.run(debug=False, port=port)
    finally:
        _safe_log(
            step="flask_service",
            status="stop",
            details={"port": port},
        )
