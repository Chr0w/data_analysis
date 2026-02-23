#!/usr/bin/env python3
"""
Generate combined results CSV files from per-run CSV files (1.csv, 2.csv, ...)
in default_amcl, default_02, and alpha_tuning folders.
Uses the first load_percentage of each file (by row count, after skipping the first 100 rows). Set load_percentage at the top of the script (default 0.5).
Each output row is one run: position_RMSE, yaw_RMSE, ESI stats, duration, etc.
"""

import os
import sys
import numpy as np
import pandas as pd
from data_loader import calculate_position_error

# Fraction of each run file to use (by row count, after skipping the first 100 rows).
load_percentage = 1.0


def calculate_yaw_error(gt_yaw, amcl_yaw):
    """Absolute yaw error in degrees (handles wrapping)."""
    diff = gt_yaw - amcl_yaw
    diff = ((diff + 180) % 360) - 180
    return abs(diff)


def process_run_csv(csv_path: str, run_id: int, skip_rows: int = 100, load_pct: float | None = None) -> dict | None:
    """
    Read one run CSV and compute summary metrics. Returns a dict suitable for one row
    of the combined results CSV, or None if the file is missing/invalid.
    """
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path, skiprows=range(1, skip_rows))
    except Exception as e:
        print(f"Warning: Could not read {csv_path}: {e}", file=sys.stderr)
        return None
    df.columns = df.columns.str.strip()

    required = ["timestamp", "ground_truth_x", "ground_truth_y", "amcl_x", "amcl_y"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Warning: Missing columns in {csv_path}: {missing}", file=sys.stderr)
        return None

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if len(df) == 0:
        return None

    # Use only the first load_percentage of rows (by count)
    pct = load_pct if load_pct is not None else load_percentage
    num_rows = max(1, int(len(df) * pct))
    df = df.iloc[:num_rows].copy()

    # Position error per row
    pos_errors = np.array([
        calculate_position_error(
            row["ground_truth_x"], row["ground_truth_y"],
            row["amcl_x"], row["amcl_y"],
        )
        for _, row in df.iterrows()
    ])

    # Yaw error per row (if available)
    has_yaw = "ground_truth_yaw" in df.columns and "amcl_yaw" in df.columns
    if has_yaw:
        yaw_errors = np.array([
            calculate_yaw_error(row["ground_truth_yaw"], row["amcl_yaw"])
            for _, row in df.iterrows()
        ])
    else:
        yaw_errors = np.full(len(df), np.nan)

    # Duration and message count
    t0 = df["timestamp"].iloc[0]
    t1 = df["timestamp"].iloc[-1]
    duration_s = float(t1 - t0) if t1 > t0 else 0.0
    total_messages = len(df)
    msg_rate_hz = total_messages / duration_s if duration_s > 0 else np.nan

    # RMSE and other position/yaw stats
    position_RMSE = float(np.sqrt(np.mean(pos_errors**2)))
    position_mean_error = float(np.mean(pos_errors))
    position_max_error = float(np.max(pos_errors))
    position_std_dev = float(np.std(pos_errors))

    if np.any(np.isfinite(yaw_errors)):
        yaw_finite = yaw_errors[np.isfinite(yaw_errors)]
        yaw_RMSE = float(np.sqrt(np.mean(yaw_finite**2)))
        yaw_mean_error = float(np.mean(yaw_finite))
        yaw_max_error = float(np.max(yaw_finite))
        yaw_std_dev = float(np.std(yaw_finite))
    else:
        yaw_RMSE = yaw_mean_error = yaw_max_error = yaw_std_dev = np.nan

    # ESI stats (if present)
    if "esi" in df.columns:
        esi = pd.to_numeric(df["esi"], errors="coerce").dropna()
        if len(esi) > 0:
            mean_ESI = float(esi.mean())
            ESI_std_dev = float(esi.std())
            ESI_range = float(esi.max() - esi.min())
        else:
            mean_ESI = ESI_std_dev = ESI_range = np.nan
    else:
        mean_ESI = ESI_std_dev = ESI_range = np.nan

    # Covariance trace sum (if present)
    sum_cov_trace = np.nan
    cov_cols = [c for c in df.columns if "cov" in c.lower()]
    if cov_cols:
        try:
            sum_cov_trace = float(pd.to_numeric(df[cov_cols], errors="coerce").sum(axis=1).sum())
        except Exception:
            pass

    return {
        "run_id": run_id,
        "timestamp": t0,
        "total_messages": total_messages,
        "duration_s": duration_s,
        "msg_rate_hz": msg_rate_hz,
        "position_RMSE": position_RMSE,
        "yaw_RMSE": yaw_RMSE,
        "position_mean_error": position_mean_error,
        "yaw_mean_error": yaw_mean_error,
        "position_max_error": position_max_error,
        "yaw_max_error": yaw_max_error,
        "position_std_dev": position_std_dev,
        "yaw_std_dev": yaw_std_dev,
        "mean_ESI": mean_ESI,
        "ESI_std_dev": ESI_std_dev,
        "ESI_range": ESI_range,
        "sum_cov_trace": sum_cov_trace,
    }


def combine_folder(
    data_folder: str,
    output_path: str,
    n_files: int = 30,
    skip_rows: int = 100,
    load_pct: float | None = None,
) -> pd.DataFrame | None:
    """
    Process 1.csv .. n_files.csv in data_folder and write combined results to output_path.
    Returns the combined DataFrame, or None if no valid runs found.
    """
    pct = load_pct if load_pct is not None else load_percentage
    rows = []
    for i in range(1, n_files + 1):
        csv_path = os.path.join(data_folder, f"{i}.csv")
        if not os.path.exists(csv_path):
            # Support zero-padded filenames such as 01.csv, 02.csv, ..., 10.csv.
            csv_path = os.path.join(data_folder, f"{i:02d}.csv")
        row = process_run_csv(csv_path, i, skip_rows=skip_rows, load_pct=pct)
        if row is not None:
            rows.append(row)
    if not rows:
        print(f"Warning: No valid runs in {data_folder}, skipping...", file=sys.stderr)
        return None
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")
    return df


def main():
    user_home = os.path.expanduser("~")
    base = os.path.join(user_home, "pCloudDrive/Offline/PhD/Folders/test_data/article_data")

    default_folder = os.path.join(base, "default_amcl")
    default_out = os.path.join(default_folder, "default_combined_results_new.csv")
    default_02_folder = os.path.join(base, "default_02")
    default_02_out = os.path.join(default_02_folder, "default_02_combined_results_new.csv")
    default_001_folder = os.path.join(base, "default_001")
    default_001_out = os.path.join(default_001_folder, "default_001_combined_results_new.csv")
    tuning_folder = os.path.join(base, "alpha_tuning")
    tuning_out = os.path.join(tuning_folder, "tuning_combined_results_new.csv")

    folders_to_process = [
        (default_folder, default_out, "default_amcl"),
        (default_02_folder, default_02_out, "default_02"),
        (default_001_folder, default_001_out, "default_001"),
        (tuning_folder, tuning_out, "alpha_tuning"),
    ]

    for folder, output, name in folders_to_process:
        if not os.path.isdir(folder):
            print(f"Warning: Folder not found: {folder}, skipping...", file=sys.stderr)
            continue
        print(f"Combining {name} runs (first {load_percentage*100:.0f}% of each file)...")
        result = combine_folder(folder, output, load_pct=load_percentage)
        if result is None:
            print(f"Skipped {name} (no valid runs found)")
    print("Done.")


if __name__ == "__main__":
    main()
