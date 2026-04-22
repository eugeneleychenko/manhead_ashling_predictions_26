#!/usr/bin/env python3
"""
Retrain worker — runs as a subprocess spawned by the Flask API.

Usage:  python retrain_worker.py <paths_config_path>

Steps:
  1. Run consolidation pipeline (Step 2) — skips Spotify scraping
  2. Run model training (Step 3) into a staging directory
  3. Validate new metrics vs old (R2 drop > 10% = blocked)
  4. If valid: promote staged artifacts → Flask/, write .retrain_complete signal
  5. Archive uploaded CSVs for reproducibility
"""

import os
import sys
import json
import shutil
import datetime as dt

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


def main(config_path: str):
    paths = load_paths(config_path)
    repo_root = os.path.dirname(os.path.abspath(config_path))
    artifact_dir = paths.get("flask_artifacts_dir", os.path.join(repo_root, "Flask"))
    staging_dir = os.path.join(artifact_dir, ".staging")
    os.makedirs(staging_dir, exist_ok=True)

    metrics_path = paths.get("last_train_metrics_json",
                             os.path.join(artifact_dir, "last_train_metrics.json"))

    # ── Load old metrics for comparison ──
    old_metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            old_metrics = json.load(f)
        print(f"[retrain] Old metrics: R2={old_metrics.get('r2_test', 'N/A')}, "
              f"RMSE={old_metrics.get('rmse_test', 'N/A')}")

    # ── Step 2: Consolidation (skip Spotify) ──
    print("[retrain] Step 2: Running consolidation pipeline...")
    consolidation_script = paths.get("consolidation_script_path",
                                     os.path.join(repo_root, "Python_scripts", "consolidate_pipeline.py"))

    if os.path.exists(consolidation_script):
        # Import and run consolidation with skip_spotify=True
        sys.path.insert(0, os.path.dirname(consolidation_script))
        try:
            import consolidate_pipeline
            # The consolidation pipeline has a main function that accepts paths_config
            if hasattr(consolidate_pipeline, "run_pipeline"):
                consolidate_pipeline.run_pipeline(config_path, skip_spotify=True)
            elif hasattr(consolidate_pipeline, "main"):
                consolidate_pipeline.main(config_path, skip_spotify=True)
            else:
                # Fallback: run as subprocess with env var to skip spotify
                import subprocess
                env = os.environ.copy()
                env["SKIP_SPOTIFY"] = "1"
                result = subprocess.run(
                    [sys.executable, consolidation_script, config_path],
                    cwd=repo_root, env=env,
                    capture_output=True, text=True, timeout=1800,
                )
                print(result.stdout)
                if result.returncode != 0:
                    print(f"[retrain] Consolidation warnings: {result.stderr}")
        except Exception as e:
            print(f"[retrain] Consolidation error (continuing): {e}")
    else:
        print(f"[retrain] Consolidation script not found at {consolidation_script}, skipping Step 2")

    # ── Step 3: Train model (into staging) ──
    print("[retrain] Step 3: Training model...")

    # Temporarily override artifact paths to write to staging
    staging_paths = dict(paths)
    staging_paths["flask_artifacts_dir"] = staging_dir
    staging_paths["model_joblib"] = os.path.join(staging_dir, "model_retrained.joblib")
    staging_paths["scaler_joblib"] = os.path.join(staging_dir, "robust_scaler_retrained.joblib")
    staging_paths["encoder_joblib"] = os.path.join(staging_dir, "label_encoder_retrained.joblib")
    staging_paths["last_train_metrics_json"] = os.path.join(staging_dir, "last_train_metrics.json")

    # Write a temporary staging config
    staging_config = os.path.join(staging_dir, "paths_config_staging.txt")
    with open(staging_config, "w") as f:
        for k, v in staging_paths.items():
            f.write(f"{k} = {v}\n")

    sys.path.insert(0, os.path.join(repo_root, "Python_scripts"))
    from train_model import train_and_save
    train_and_save(staging_config, staging_dir)

    # ── Validate new metrics ──
    print("[retrain] Validating new model...")
    new_metrics_path = os.path.join(staging_dir, "last_train_metrics.json")
    if not os.path.exists(new_metrics_path):
        print("[retrain] FAIL: No metrics file produced by training")
        sys.exit(1)

    with open(new_metrics_path) as f:
        new_metrics = json.load(f)

    new_r2 = new_metrics.get("r2_test", 0)
    print(f"[retrain] New metrics: R2={new_r2:.4f}, RMSE={new_metrics.get('rmse_test', 'N/A'):.4f}")

    if old_metrics:
        old_r2 = old_metrics.get("r2_test", 0)
        r2_change = (old_r2 - new_r2) / max(abs(old_r2), 1e-9)
        print(f"[retrain] R2 change: {old_r2:.4f} → {new_r2:.4f} (delta={r2_change*100:.2f}%)")

        if r2_change > 0.10:  # R2 dropped by more than 10%
            print(f"[retrain] BLOCKED: R2 degraded by {r2_change*100:.1f}% (threshold: 10%)")
            print("[retrain] Old artifacts kept. New model NOT promoted.")
            # Write failure info
            fail_info = {
                "status": "validation_failed",
                "reason": f"R2 degraded by {r2_change*100:.1f}%",
                "old_r2": old_r2,
                "new_r2": new_r2,
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            }
            with open(os.path.join(staging_dir, "validation_failure.json"), "w") as f:
                json.dump(fail_info, f, indent=2)
            sys.exit(1)

    # ── Promote: move staging artifacts → Flask/ ──
    print("[retrain] Promoting new artifacts...")
    artifact_files = [
        "model_retrained.joblib",
        "robust_scaler_retrained.joblib",
        "label_encoder_retrained.joblib",
        "last_train_metrics.json",
    ]
    for fname in artifact_files:
        src = os.path.join(staging_dir, fname)
        dst = os.path.join(artifact_dir, fname)
        if os.path.exists(src):
            os.replace(src, dst)
            print(f"  {fname} promoted")

    # ── Signal Flask to hot-reload ──
    signal_path = os.path.join(artifact_dir, ".retrain_complete")
    with open(signal_path, "w") as f:
        f.write(dt.datetime.now().isoformat())
    print("[retrain] Signal file written — Flask will hot-reload")

    # ── Archive uploaded CSVs ──
    archive_dir = os.path.join(repo_root, "CSVs", "archive", dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(archive_dir, exist_ok=True)

    for subdir_name in ("add_sales_reports_files_here", "add_tour_summary_files_here"):
        src_dir = os.path.join(repo_root, "CSVs", subdir_name)
        if os.path.isdir(src_dir):
            dest = os.path.join(archive_dir, subdir_name)
            shutil.copytree(src_dir, dest, dirs_exist_ok=True)

    print(f"[retrain] Files archived to {archive_dir}")
    print("[retrain] DONE — retrain successful")

    # Clean staging
    for fname in artifact_files:
        p = os.path.join(staging_dir, fname)
        if os.path.exists(p):
            os.remove(p)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python retrain_worker.py <paths_config_path>")
        sys.exit(1)
    main(sys.argv[1])
